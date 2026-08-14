import json
import os
from logging import Logger
from typing import Iterator

import ijson
import requests
from grammar_utils.parse import LR1Parser  # type: ignore
from search_rdf import Data
from tqdm import tqdm
from universal_ml_utils.io import dump_jsonl, dump_text, load_jsonl
from universal_ml_utils.logging import get_logger

from grasp.functions import parse_iri_or_literal
from grasp.manager.utils import (
    get_common_sparql_prefixes,
    load_index_sparql,
    load_kg_info,
    merge_prefixes,
)
from grasp.sparql.types import Binding
from grasp.sparql.utils import (
    get_basic_auth,
    get_qlever_endpoint,
    has_scheme,
    load_iri_and_literal_parser,
)
from grasp.utils import (
    camel_case_split,
    get_index_dir,
    get_local_name_from_iri,
    ordered_unique,
    split_at_punctuation,
)


def download_data(
    out_dir: str,
    sparql: str,
    prefixes: dict[str, str],
    parser: LR1Parser,
    logger: Logger,
    endpoint: str | None = None,
    params: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
    auth: tuple[str, str] | None = None,
    add_id_as_label: str = "never",
    result_file: str | None = None,
    overwrite: bool = False,
) -> None:
    data_file = os.path.join(out_dir, "data.jsonl")
    if os.path.exists(data_file) and not overwrite:
        logger.info(f"Data already exists at {data_file}, skipping download")
        return

    if result_file is not None:
        logger.info(f"Loading data to {data_file} from file {result_file}")
        bindings = stream_json_file(result_file)
    else:
        assert endpoint is not None, (
            "Endpoint must be provided if no result file is given"
        )
        # only the user name is logged, never the password
        auth_fmt = f"Basic Auth as {auth[0]}" if auth else "no Basic Auth"
        logger.info(
            f"Downloading data to {data_file} from {endpoint} "
            f"with parameters {params or {}}, headers {headers or {}}, "
            f"{auth_fmt}, and SPARQL:\n{sparql}"
        )
        bindings = stream_json(endpoint, sparql, params, headers, auth)

    dump_jsonl(
        prepare_items(bindings, prefixes, parser, add_id_as_label, logger),
        data_file,
    )


def build_data_and_mapping(
    index_dir: str,
    logger: Logger,
    overwrite: bool = False,
) -> None:
    data_file = os.path.join(index_dir, "data.jsonl")
    data_dir = os.path.join(index_dir, "data")
    if not os.path.exists(data_dir) or overwrite:
        # build index data
        logger.info(f"Building data at {data_dir}")
        Data.build_from_jsonl(data_file, data_dir)
    else:
        logger.info(f"Data already exists at {data_dir}, skipping build")


def get_data(
    kg: str,
    index_name: str,
    endpoint: str | None = None,
    index_sparql: str | None = None,
    data_file: str | None = None,
    add_id_as_label: str = "auto",
    params: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
    log_level: str | int | None = None,
    overwrite: bool = False,
) -> None:
    logger = get_logger("GRASP DATA", log_level)

    if add_id_as_label == "auto":
        if index_name == "entities":
            add_id_as_label = "never"
        elif index_name == "properties":
            add_id_as_label = "always"
        elif index_name == "literals":
            add_id_as_label = "empty"
        else:
            raise ValueError(
                "Auto setting for ID-derived labels is not supported "
                f'for index "{index_name}"'
            )

    parser = load_iri_and_literal_parser()

    info = load_kg_info(kg)

    params = params or {}
    if info.params:
        params = {**info.params, **params}

    headers = headers or {}
    if info.headers:
        headers = {**info.headers, **headers}

    endpoint = endpoint or info.endpoint or get_qlever_endpoint(kg)

    auth = get_basic_auth(kg)

    prefixes = get_common_sparql_prefixes()
    prefixes, _, _ = merge_prefixes(prefixes, info.prefixes or {}, logger)
    logger.info(f"Using prefixes:\n{json.dumps(prefixes, indent=2)}")

    kg_dir = get_index_dir(kg)

    index_dir = os.path.join(kg_dir, index_name)
    if index_sparql is None:
        index_sparql = load_index_sparql(index_dir, logger)

    if index_sparql is None:
        raise ValueError(f'No index SPARQL found or set for "{index_name}" index')

    os.makedirs(index_dir, exist_ok=True)
    download_data(
        index_dir,
        index_sparql,
        prefixes,
        parser,
        logger,
        endpoint,
        params,
        headers,
        auth,
        add_id_as_label,
        data_file,
        overwrite,
    )
    dump_text(index_sparql, os.path.join(index_dir, "index.sparql"))
    build_data_and_mapping(index_dir, logger, overwrite)


def stream_json(
    endpoint: str,
    sparql: str,
    params: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
    auth: tuple[str, str] | None = None,
) -> Iterator[dict]:
    try:
        headers = {
            "Accept": "application/sparql-results+json",
            "Content-Type": "application/sparql-query",
            "User-Agent": "grasp-data-bot",
            **(headers or {}),
        }

        response = requests.post(
            endpoint,
            data=sparql,
            params=params,
            headers=headers,
            auth=auth,
            stream=True,
        )
        response.raise_for_status()
    except Exception as e:
        status = None
        # the response body carries the actual explanation, e.g. QLever puts
        # its error message in there, so never drop it from the error
        body = ""
        if isinstance(e, requests.HTTPError) and e.response is not None:
            status = e.response.status_code
            body = e.response.text.strip()

        if status in (401, 403) and auth is None:
            raise ValueError(
                f"Endpoint {endpoint} requires authentication but no "
                "credentials were sent, set GRASP_BASICAUTH_<KG>_USER and "
                "GRASP_BASICAUTH_<KG>_PASSWD for the knowledge graph "
                "in question"
            ) from e

        if status == 400 and auth is not None:
            # QLever rejects any Authorization header that is not a Bearer
            # token, so Basic Auth credentials meant for an authenticating
            # proxy in front of an endpoint turn into a confusing 400 as soon
            # as the endpoint is queried directly
            raise ValueError(
                f"Endpoint {endpoint} rejected the request with 400 while "
                f"Basic Auth credentials were sent: {body}; if this endpoint "
                "does not require authentication, unset "
                "GRASP_BASICAUTH_<KG>_USER and GRASP_BASICAUTH_<KG>_PASSWD "
                "for the knowledge graph in question"
            ) from e

        raise ValueError(
            f"Failed to stream SPARQL results as JSON: {e}"
            + (f" ({body})" if body else "")
        ) from e

    class _StreamReader:
        def __init__(self, response: requests.Response) -> None:
            self.stream = response.iter_content(chunk_size=None)

        def read(self, n: int) -> bytes:
            if n == 0:
                return b""
            return next(self.stream, b"")

    yield from ijson.items(_StreamReader(response), "results.bindings.item")


def stream_json_file(path: str) -> Iterator[dict]:
    with open(path, "r") as f:
        yield from ijson.items(f, "results.bindings.item")


def get_value_from_id_binding(obj_id: Binding, prefixes: dict[str, str]) -> str:
    if obj_id.typ != "uri":
        return obj_id.value

    obj_name = get_local_name_from_iri(obj_id.identifier(), prefixes)
    label = " ".join(camel_case_split(part) for part in split_at_punctuation(obj_name))
    return label.strip()


def parse_binding(
    binding: dict,
    parser: LR1Parser,
) -> tuple[Binding, Binding | None, list[str]]:
    id = Binding.from_dict(binding["id"])

    tag_binding = binding.get("tag", binding.get("tags", None))
    if tag_binding is not None:
        assert tag_binding["type"] == "literal", "Expected tags to be a literal"
        tags = tag_binding["value"].lower().split(",")
    else:
        tags = []

    if "value" not in binding:
        return id, None, tags

    value_binding = Binding.from_dict(binding["value"])
    if value_binding.typ == "literal" and has_scheme(value_binding.value):
        # may be an uri converted to literal, double check
        value_binding = parse_iri_or_literal(value_binding.value, parser)

    return id, value_binding, tags


def id_as_label_field(
    id_binding: Binding,
    prefixes: dict[str, str],
    fields: list[dict],
) -> dict:
    # the id-derived label becomes the main field unless the index query
    # already marked one, so an item whose only field is this one still has a
    # label. Indices can use this on purpose: freebase properties leave their
    # name untagged, making the id-derived schema path ("sports sports team
    # roster from") the main field instead of the ambiguous name ("From").
    main = any("main" in field["tags"] for field in fields)
    return {
        "type": "text",
        "value": get_value_from_id_binding(id_binding, prefixes),
        "tags": [] if main else ["main"],
    }


def prepare_items(
    bindings: Iterator[dict],
    prefixes: dict[str, str],
    parser: LR1Parser,
    add_id_as_label: str = "never",
    logger: Logger | None = None,
) -> Iterator[dict]:
    # collect all labels for an id (which are consecutive in the stream)
    last_id_binding = None
    fields = []
    for num, binding in enumerate(bindings, start=1):
        id_binding, value_binding, tags = parse_binding(binding, parser)

        if logger and num % 1_000_000 == 0:
            logger.info(f"Processed {num:,} bindings so far")

        if logger:
            logger.debug(
                f"Processing binding #{num:,}: id={id_binding.identifier()}, "
                f"value={value_binding}, tags={tags}"
            )

        if (
            last_id_binding is not None
            and id_binding.identifier() != last_id_binding.identifier()
        ):
            # yield previous item
            if add_id_as_label == "always" or (
                add_id_as_label == "empty" and not fields
            ):
                fields.append(id_as_label_field(last_id_binding, prefixes, fields))

            yield {
                "identifier": last_id_binding.identifier(),
                "fields": ordered_unique(fields, key=lambda f: f["value"]),
            }

            fields = []

        last_id_binding = id_binding
        if value_binding is None:
            continue
        elif value_binding.typ == "uri":
            value = get_value_from_id_binding(value_binding, prefixes)
        else:
            value = value_binding.value

        fields.append({"type": "text", "value": value, "tags": tags})

    if last_id_binding is None:
        return

    # dont forget final item
    if add_id_as_label == "always" or (add_id_as_label == "empty" and not fields):
        fields.append(id_as_label_field(last_id_binding, prefixes, fields))

    yield {
        "identifier": last_id_binding.identifier(),
        "fields": ordered_unique(fields, key=lambda f: f["value"]),
    }


def merge_data(
    kgs: list[str],
    index_name: str,
    out_dir: str,
    logger: Logger,
    overwrite: bool = False,
):
    out_dir = os.path.join(out_dir, index_name)
    data_file = os.path.join(out_dir, "data.jsonl")
    kg_info = ", ".join(kgs)
    if os.path.exists(data_file) and not overwrite:
        logger.info(
            f'Merged data for "{index_name}" index of knowledge graphs {kg_info} '
            f"already exists at {data_file}, skipping merge"
        )
        return

    logger.info(
        f'Merging data for "{index_name}" index of knowledge graphs {kg_info} '
        f"into {data_file}"
    )

    os.makedirs(out_dir, exist_ok=True)

    others = []

    for kg in tqdm(kgs[1:], desc="Building mappings for data to merge"):
        kg_data_file = os.path.join(get_index_dir(kg), index_name, "data.jsonl")

        items = load_jsonl(kg_data_file)
        others.append({item["identifier"]: item for item in items})

    # first kg is the main one, to which we add data from the others
    kg = kgs[0]
    kg_data_file = os.path.join(get_index_dir(kg), index_name, "data.jsonl")

    def merge() -> Iterator[str]:
        with open(kg_data_file, "r") as f:
            for line in tqdm(f, desc="Merging data"):
                item = json.loads(line)

                identifier = item["identifier"]

                # collect fields from other kgs and add them
                # to the current item
                seen = set(field["value"] for field in item["fields"])
                for mapping in others:
                    if identifier not in mapping:
                        continue

                    other_item = mapping[identifier]
                    for field in other_item["fields"]:
                        if field["value"] in seen:
                            continue

                        item["fields"].append(field)
                        seen.add(field["value"])

                yield item

    dump_jsonl(merge(), data_file)


def merge_kgs(
    kgs: list[str],
    index_name: str,
    out_kg: str,
    overwrite: bool = False,
    log_level: str | int | None = None,
):
    assert len(kgs) >= 2, "At least two knowledge graphs are required to merge"

    logger = get_logger("GRASP MERGE", log_level)

    out_dir = get_index_dir(out_kg)

    merge_data(kgs, index_name, out_dir, logger, overwrite)

    out_index_dir = os.path.join(out_dir, index_name)
    build_data_and_mapping(out_index_dir, logger, overwrite)
