import logging
import os
import time
from dataclasses import dataclass
from io import BytesIO

import numpy as np
import requests
from PIL import Image
from search_rdf import Data, EmbeddingIndex, FuzzyIndex, KeywordIndex
from search_rdf.model import (
    HuggingFaceImageModel,
    OpenClipModel,
    SentenceTransformerModel,
)
from universal_ml_utils.configuration import load_config
from universal_ml_utils.io import load_json, load_text

from grasp.configs import KgInfo, NamedIndexConfig
from grasp.manager.normalizer import Normalizer
from grasp.search_params import SearchParams, resolve_index_search_params
from grasp.sparql.types import ObjType
from grasp.sparql.utils import find_longest_prefix
from grasp.utils import get_index_dir

SearchIndex = KeywordIndex | EmbeddingIndex | FuzzyIndex
EmbeddingModel = HuggingFaceImageModel | OpenClipModel | SentenceTransformerModel


@dataclass
class KgIndex:
    description: str
    index: SearchIndex
    data: Data
    info_sparql: str | None = None
    normalizer: Normalizer | None = None
    search_params: SearchParams | None = None


def load_data(index_dir: str) -> Data:
    try:
        data = Data.load(os.path.join(index_dir, "data"))
    except Exception as e:
        raise ValueError(f"Failed to load index data from {index_dir}") from e

    return data


def get_index_type_from_data(data: Data) -> str:
    # heuristic to determine index type from data size,
    # used when index type is auto
    if len(data) > 1_000_000:
        return "fuzzy"
    else:
        return "embedding"


def get_auto_index_type(
    index_name: str,
    data: Data | None = None,
) -> str:
    if index_name == "entities":
        return "fuzzy"
    elif index_name == "properties":
        return "embedding"
    elif index_name == "literals":
        if data is not None:
            return get_index_type_from_data(data)
        else:
            # default to embedding for literals if we don't have data to check
            return "embedding"
    else:
        raise ValueError(f'Auto index type not supported for index "{index_name}"')


def try_load_search_index(
    index_dir: str,
    index_type: str,
    logger: logging.Logger | None = None,
) -> SearchIndex | None:
    start = time.monotonic()

    try:
        data = load_data(index_dir)
    except Exception as e:
        if logger is not None:
            logger.warning(f"Failed to load data from {index_dir}: {e}")
        return None

    # resolve auto index type depending on the size of the data
    if index_type == "auto":
        index_type = get_auto_index_type(os.path.basename(index_dir), data)

        if logger is not None:
            logger.debug(
                f"Resolved auto index type to {index_type} based on "
                f"data size ({len(data):,} items)"
            )

    index_dir = os.path.join(index_dir, index_type)
    load_kwargs = {"data": data, "index_dir": index_dir}

    if index_type == "keyword":
        index_cls = KeywordIndex
    elif index_type == "fuzzy":
        index_cls = FuzzyIndex
    elif index_type == "embedding":
        index_cls = EmbeddingIndex
        load_kwargs["embedding_path"] = os.path.join(index_dir, "embedding.safetensors")
    else:
        raise ValueError(f"Unknown index type {index_type}")

    try:
        index = index_cls.load(**load_kwargs)
    except Exception as e:
        if logger is not None:
            logger.warning(f"Failed to load {index_type} index from {index_dir}: {e}")
        return None

    end = time.monotonic()

    if logger is not None:
        logger.debug(
            f"Loading {index_type} index from {index_dir} took {end - start:.2f}s"
        )

    return index


def load_index_sparql(
    index_dir: str,
    logger: logging.Logger | None = None,
) -> str | None:
    index_sparql_path = os.path.join(index_dir, "index.sparql")
    if os.path.exists(index_sparql_path):
        if logger is not None:
            logger.debug(f"Loaded index.sparql from {index_dir}")
        return load_text(index_sparql_path)

    if logger is not None:
        logger.debug(f"No index.sparql found at {index_dir}")
    return None


def load_info_sparql(
    index_dir: str,
    logger: logging.Logger | None = None,
) -> str | None:
    info_sparql_path = os.path.join(index_dir, "info.sparql")
    if os.path.exists(info_sparql_path):
        if logger is not None:
            logger.debug(f"Loaded info.sparql from {index_dir}")
        return load_text(info_sparql_path)

    if logger is not None:
        logger.debug(f"No info.sparql found at {index_dir}")
    return None


def load_index_description(
    index_dir: str,
    logger: logging.Logger | None = None,
) -> str | None:
    info_path = os.path.join(index_dir, "info.json")
    if not os.path.exists(info_path):
        return None

    info = load_json(info_path)
    desc = info.get("description")
    if desc and logger is not None:
        logger.debug(f"Loaded index description from {info_path}")
    return desc


def load_other_indices(
    kg: str,
    indices: list[NamedIndexConfig],
    logger: logging.Logger | None = None,
) -> dict[str, KgIndex]:
    base_index_dir = get_index_dir(kg)
    config_path = os.path.join(base_index_dir, "indices.yaml")
    if not os.path.exists(config_path):
        if logger is not None:
            logger.debug(
                f"No indices.yaml found at {config_path}, skipping other indices"
            )
        return {}

    config = load_config(config_path)

    configured = {index.name: index for index in indices}

    others = {}
    for cfg in config["indices"]:
        name = cfg["name"]
        if name not in configured:
            if logger is not None:
                logger.debug(
                    f"Skipping index {name} as it's not in the specified indices list"
                )
            continue

        desc = cfg.get("description", "No description available")

        sub_index_dir = os.path.join(base_index_dir, name)

        # normalize embedding index type, which is called embedding-with-data
        # in search-rdf, but embedding in the search-rdf python interface
        if cfg["type"].startswith("embedding"):
            cfg["type"] = "embedding"

        index = try_load_search_index(sub_index_dir, cfg["type"], logger)
        if index is None:
            continue

        info_sparql = load_info_sparql(sub_index_dir, logger)
        search_params = resolve_index_search_params(
            index.index_type,
            os.path.join(sub_index_dir, index.index_type),
            configured[name].params,
            name=name,
            logger=logger,
        )
        others[name] = KgIndex(
            desc, index, index.data(), info_sparql, search_params=search_params
        )

    return others


def get_embedding_model_key(index: EmbeddingIndex) -> str:
    assert index.model is not None, "Embedding index must have model metadata"
    provider = index.provider or "sentence-transformer"
    return f"{provider}/{index.model}"


def load_embedding_model(
    index: SearchIndex,
    models: dict[str, EmbeddingModel],
) -> dict[str, EmbeddingModel]:
    if not index.index_type == "embedding":
        return models

    assert isinstance(index, EmbeddingIndex), "Expected an EmbeddingIndex"
    assert index.model is not None, "Embedding index must have model metadata"

    key = get_embedding_model_key(index)
    if key in models:
        return models

    provider = index.provider or "sentence-transformer"
    if provider == "sentence-transformer":
        model = SentenceTransformerModel(index.model)
    elif provider == "open-clip":
        model = OpenClipModel(index.model)
    elif provider == "huggingface-image":
        model = HuggingFaceImageModel(index.model)
    else:
        raise ValueError(f"Unknown embedding model provider {provider}")

    models[key] = model
    return models


def load_kg_info(kg: str, logger: logging.Logger | None = None) -> KgInfo:
    kg_index_dir = get_index_dir(kg)
    info_file = os.path.join(kg_index_dir, "info.json")

    # load info.json if it exists, fall back to empty
    if os.path.exists(info_file):
        info = KgInfo(**load_json(info_file))
    else:
        info = KgInfo()

    if info.prefixes:
        # remove prefixes that conflict with or duplicate common prefixes
        _, _, kg_prefixes = merge_prefixes(
            get_common_sparql_prefixes(),
            info.prefixes,
            logger,
        )
        info.prefixes = kg_prefixes

    return info


def merge_prefixes(
    first: dict[str, str],
    second: dict[str, str],
    logger: logging.Logger | None = None,
    do_raise: bool = False,
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    # merge second into first, returning (merged, first_only, second_only)
    # conflicts from second are dropped with warning or error
    reverse_first = {v: k for k, v in first.items()}

    first_only = {}
    second_only = {}
    merged = dict(first)

    for short, long in second.items():
        if short in first:
            if long != first[short]:
                # name clash with different IRI
                msg = (
                    f'Prefix "{short}" already defined as {first[short]}, '
                    f"cannot redefine as {long}"
                )
                if do_raise:
                    raise RuntimeError(msg)
                if logger is not None:
                    logger.warning(msg)
            # either way, skip (duplicate or conflict)
            continue

        if long in reverse_first:
            # IRI already covered by a different short name
            msg = (
                f"{long} is already covered by prefix "
                f'"{reverse_first[long]}", skipping "{short}"'
            )
            if do_raise:
                raise RuntimeError(msg)
            if logger is not None:
                logger.warning(msg)
            continue

        merged[short] = long
        second_only[short] = long

    for short, long in first.items():
        if short not in second:
            first_only[short] = long

    return merged, first_only, second_only


def get_common_sparql_prefixes() -> dict[str, str]:
    return {
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
        "owl": "http://www.w3.org/2002/07/owl#",
        "xsd": "http://www.w3.org/2001/XMLSchema#",
        "foaf": "http://xmlns.com/foaf/0.1/",
        "skos": "http://www.w3.org/2004/02/skos/core#",
        "dct": "http://purl.org/dc/terms/",
        "dc": "http://purl.org/dc/elements/1.1/",
        "qb": "http://purl.org/linked-data/cube#",
        "prov": "http://www.w3.org/ns/prov#",
        "schema": "http://schema.org/",
        "geo": "http://www.opengis.net/ont/geosparql#",
        "geosparql": "http://www.opengis.net/ont/geosparql#",
        "geof": "http://www.opengis.net/def/function/geosparql/",
        "gn": "http://www.geonames.org/ontology#",
        "bd": "http://www.bigdata.com/rdf#",
        "hint": "http://www.bigdata.com/queryHints#",
        "wikibase": "http://wikiba.se/ontology#",
        "void": "http://rdfs.org/ns/void#",
    }


def find_obj_type_from_prefixes(
    iri: str,
    prefixes: dict[str, str],
    common_prefixes: dict[str, str],
) -> ObjType:
    # we have three cases:
    # 1. the IRI matches a common prefix but not a known prefix -> COMMON (e.g. rdf:type)
    # 2. the IRI matches a known prefix -> UNINDEXED (e.g. wd:Q42)
    # 3. the IRI matches neither -> UNKNOWN (e.g. fb:en.barack_obama in Wikidata)
    is_common = find_longest_prefix(iri, common_prefixes) is not None
    is_known = find_longest_prefix(iri, prefixes) is not None

    if is_common:
        return ObjType.COMMON
    elif is_known:
        return ObjType.UNINDEXED
    else:
        return ObjType.UNKNOWN


def load_image_from_url(url: str) -> np.ndarray:
    try:
        if url.startswith("file://"):
            path = url[len("file://") :]
            image = Image.open(path).convert("RGB")
        else:
            response = requests.get(url, headers={"User-Agent": "grasp-rdf"})
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
        return np.array(image)
    except Exception as e:
        raise IOError(f"Failed to load image from {url}: {e}") from e


def format_index_meta(index: SearchIndex) -> str:
    parts = [f'type="{index.index_type}"']
    if isinstance(index, EmbeddingIndex):
        modalities = index.modality or ["text"]
        parts.append(f'modalities="{"+".join(modalities)}"')
    return ", ".join(parts)


def describe_index_type(index_type: str) -> str:
    if index_type == "keyword":
        return "Retrieves items by overlap between their label words and \
the query keywords. The query keywords can match label words exactly or \
as prefixes. No special query operators like AND/OR are supported."

    elif index_type == "fuzzy":
        return "Retrieves items by overlap between their label words and \
the query keywords. The query keywords must not match label words exactly, but \
some fuzziness is allowed. The longer a query keyword is, the more it can deviate \
from a label word and still be considered a match, though it will also contribute \
less to the overall score. No special query operators like AND/OR are supported."

    elif index_type == "embedding":
        return "Retrieves items by cosine similarity between their \
embeddings and the query embedding. The embedding model used depends on the \
index and may support text, images, or both."

    else:
        raise ValueError(f"Unknown index type {index_type}")
