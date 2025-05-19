import logging
import math
import os
import sys
import tempfile
import time
from typing import Any, Iterable, Type

from search_index import PrefixIndex, SearchIndex
from universal_ml_utils.table import generate_table

from grasp.sparql.constants import (
    READ_TIMEOUT,
    REQUEST_TIMEOUT,
    AskResult,
    Binding,
    ObjType,
    Position,
    SelectResult,
    SelectRow,
    get_endpoint,
)
from grasp.sparql.manager.utils import (
    get_common_sparql_prefixes,
    get_index_desc,
)
from grasp.sparql.mapping import Mapping
from grasp.sparql.selection import Alternative, Selection, group_selections
from grasp.sparql.sparql import (
    SPARQLException,
    ask_to_select,
    autocomplete_prefix,
    autocomplete_sparql,
    clip,
    execute,
    find_longest_prefix,
    fix_prefixes,
    format_iri,
    has_iri,
    load_iri_and_literal_parser,
    load_sparql_parser,
    parse_string,
    prettify,
    query_type,
)


class KgManager:
    entity_mapping_cls: Type[Mapping] = Mapping
    property_mapping_cls: Type[Mapping] = Mapping
    prefixes: dict[str, str]
    kg: str
    endpoint: str

    def __init__(
        self,
        name: str,
        entity_index: SearchIndex,
        property_index: SearchIndex,
        entity_mapping: Mapping,
        property_mapping: Mapping,
        prefixes: dict[str, str] | None = None,
        endpoint: str | None = None,
    ):
        self.kg = name

        self.entity_index = entity_index
        self.property_index = property_index
        self.entity_mapping = entity_mapping
        self.property_mapping = property_mapping

        self.entity_index_desc = get_index_desc(self.entity_index)
        self.property_index_desc = get_index_desc(self.property_index)
        self.default_index_desc = get_index_desc()

        self.sparql_parser = load_sparql_parser()
        self.iri_literal_parser = load_iri_and_literal_parser()

        self.prefixes = prefixes or {}
        values = set(self.prefixes.values())

        # add common prefixes that might not be covered by the
        # specified prefixes
        common_prefixes = get_common_sparql_prefixes()
        for short, long in common_prefixes.items():
            if short in self.prefixes or long in values:
                continue

            self.prefixes[short] = long

        self.endpoint = endpoint or get_endpoint(self.kg)

        self.logger = logging.getLogger(f"{self.kg.upper()} KG MANAGER")

    def prettify(
        self,
        sparql: str,
        indent: int = 2,
        is_prefix: bool = False,
    ) -> str:
        return prettify(sparql, self.sparql_parser, indent, is_prefix)

    def check_sparql(self, sparql: str, is_prefix: bool = False) -> bool:
        try:
            parse_string(
                sparql,
                self.sparql_parser,
                skip_empty=True,
                collapse_single=True,
                is_prefix=is_prefix,
            )
            return True
        except Exception as e:
            self.logger.debug(f"Invalid SPARQL query {sparql}: {e}")
            return False

    def execute_sparql(
        self,
        sparql: str,
        request_timeout: float | tuple[float, float] | None = REQUEST_TIMEOUT,
        max_retries: int = 1,
        force_select_result: bool = False,
        read_timeout: float | None = READ_TIMEOUT,
    ) -> SelectResult | AskResult:
        if force_select_result:
            # ask_to_select returns None if sparql is not an ask query
            sparql = ask_to_select(sparql, self.sparql_parser) or sparql

        sparql = self.fix_prefixes(sparql)

        self.logger.debug(f"Executing SPARQL query against {self.endpoint}:\n{sparql}")
        return execute(
            sparql,
            self.endpoint,
            request_timeout,
            max_retries,
            read_timeout,
        )

    def format_sparql_result(
        self,
        result: SelectResult | AskResult,
        show_top_rows: int = 5,
        show_bottom_rows: int = 5,
        show_left_columns: int = 5,
        show_right_columns: int = 5,
        column_names: list[str] | None = None,
    ) -> str:
        # run sparql against endpoint, format result as string
        if isinstance(result, AskResult):
            return str(result.boolean)

        if result.num_rows == 0:
            return f"Got no rows and {result.num_columns:,} columns"

        assert (
            column_names is None or len(column_names) == result.num_columns
        ), f"Expected {result.num_columns:,} column names"
        assert show_top_rows or show_bottom_rows, "At least one row must be shown"
        assert (
            show_left_columns or show_right_columns
        ), "At least one column must be shown"

        left_end = min(show_left_columns, result.num_columns)
        right_start = result.num_columns - show_right_columns
        if right_start > left_end:
            column_indices = list(range(left_end))
            column_indices.append(-1)
            column_indices.extend(range(right_start, result.num_columns))
        else:
            column_indices = list(range(result.num_columns))

        def format_row(row: SelectRow) -> list[str]:
            formatted_row = []
            for c in column_indices:
                if c < 0:
                    formatted_row.append("...")
                    continue

                val = row[c]
                if val is None:
                    formatted_row.append("")

                elif val.typ == "bnode":
                    formatted_row.append(clip(val.identifier()))

                elif val.typ == "literal":
                    formatted = clip(val.value)
                    if val.lang is not None:
                        formatted += f" ({val.lang})"
                    elif val.datatype is not None:
                        datatype = self.format_iri("<" + val.datatype + ">")
                        formatted += f" ({clip(datatype)})"

                    formatted_row.append(formatted)

                else:
                    assert val.typ == "uri"
                    identifier = val.identifier()
                    formatted = self.format_iri(identifier)

                    # for uri check whether it is in one of the mappings
                    norm = self.entity_mapping.normalize(identifier)
                    map = self.entity_mapping
                    index = self.entity_index
                    if norm is None or norm[0] not in map:
                        norm = self.property_mapping.normalize(identifier)
                        map = self.property_mapping
                        index = self.property_index

                    if norm is not None and norm[0] in map:
                        name = clip(index.get_name(map[norm[0]]))
                        formatted = f"{name} ({formatted})"

                    formatted_row.append(formatted)

            return formatted_row

        # generate a nicely formatted table
        column_names = column_names or result.variables
        header = [column_names[c] if c >= 0 else "..." for c in column_indices]
        top_end = min(show_top_rows, result.num_rows)
        data = [format_row(row) for row in result.rows(end=top_end)]

        bottom_start = result.num_rows - show_bottom_rows
        if bottom_start > top_end:
            data.append(["..."] * len(header))

        bottom_start = max(bottom_start, top_end)
        data.extend(
            format_row(row) for row in result.rows(bottom_start, result.num_rows)
        )

        table = generate_table(
            data,
            [header],
            alignments=["left"] * len(header),
            max_column_width=sys.maxsize,
        )

        formatted = (
            f"Got {result.num_rows:,} row{'s' * (result.num_rows != 1)} and "
            f"{result.num_columns:,} column{'s' * (result.num_columns != 1)}"
        )

        showing = []

        if right_start > left_end:
            # columns restricted
            show_columns = []
            if show_left_columns:
                show_columns.append(f"first {show_left_columns}")
            if show_right_columns:
                show_columns.append(f"last {show_right_columns}")

            showing.append(f"the {' and '.join(show_columns)} columns")

        if bottom_start > top_end:
            # rows restricted
            show_rows = []
            if show_top_rows:
                show_rows.append(f"first {show_top_rows}")
            if show_bottom_rows:
                show_rows.append(f"last {show_bottom_rows}")

            showing.append(f"the {' and '.join(show_rows)} rows")

        if showing:
            formatted += ", showing " + " and ".join(showing) + " below"

        formatted += f":\n{table}"
        return formatted

    def get_formatted_sparql_result(
        self,
        sparql: str,
        request_timeout: float | tuple[float, float] | None = REQUEST_TIMEOUT,
        max_retries: int = 1,
        max_rows: int = 10,
        max_columns: int = 10,
    ) -> str:
        half_rows = math.ceil(max_rows / 2)
        half_columns = math.ceil(max_columns / 2)
        try:
            result = self.execute_sparql(sparql, request_timeout, max_retries)
            return self.format_sparql_result(
                result,
                half_rows,
                half_rows,
                half_columns,
                half_columns,
            )
        except Exception as e:
            return f"SPARQL execution failed:\n{e}"

    def find_longest_prefix(self, iri: str) -> tuple[str, str] | None:
        return find_longest_prefix(iri, self.prefixes)

    def format_iri(self, iri: str, base_uri: str | None = None) -> str:
        return format_iri(iri, self.prefixes, base_uri)

    def fix_prefixes(
        self,
        sparql: str,
        is_prefix: bool = False,
        remove_known: bool = False,
        sort: bool = False,
    ) -> str:
        return fix_prefixes(
            sparql,
            self.sparql_parser,
            self.prefixes,
            is_prefix,
            remove_known,
            sort,
        )

    def build_alternative(
        self,
        data: str,
        variants: set[str] | None = None,
    ) -> Alternative:
        label, _, syns, id, infos = data.rstrip("\r\n").split("\t")

        return Alternative(
            identifier=id,
            short_identifier=self.format_iri(id),
            label=label,
            variants=variants,
            aliases=[s for s in syns.split(";;;") if s],
            infos=sorted(
                (i for i in infos.split(";;;") if i),
                key=lambda i: len(i),
                reverse=True,
            ),
        )

    def parse_bindings(self, result: Iterable[Binding | None]) -> dict[ObjType, Any]:
        entities = {}
        properties = {}
        others = []
        literals = []
        for binding in result:
            if binding is None:
                continue

            elif binding.typ == "bnode":
                # ignore bnodes
                continue

            identifier = binding.identifier()

            if binding.typ == "literal":
                if binding.datatype is not None:
                    info = self.format_iri("<" + binding.datatype + ">")
                elif binding.lang is not None:
                    info = binding.lang
                else:
                    info = None

                literals.append((identifier, binding.value, info))
                continue

            # typ is uri
            unmatched = True
            for id_map, map in [
                (entities, self.entity_mapping),
                (properties, self.property_mapping),
            ]:
                norm = map.normalize(identifier)
                if norm is None:
                    continue

                iri, variant = norm
                if iri not in map:
                    continue

                id = map[iri]
                if id not in id_map:
                    id_map[id] = set()

                if variant is not None:
                    id_map[id].add(variant)

                unmatched = False

            if unmatched:
                others.append((identifier, self.format_iri(identifier), None))

        # sort others by whether they are from one of our known
        # prefixes or not
        others.sort(key=lambda item: self.find_longest_prefix(item[0]) is None)
        return {
            ObjType.ENTITY: entities,
            ObjType.PROPERTY: properties,
            ObjType.OTHER: others,
            ObjType.LITERAL: literals,
        }

    def get_entity_alternatives(
        self,
        id_map: dict[int, set[str]] | None = None,
        query: str | None = None,
        k: int = 10,
        **search_kwargs: Any,
    ) -> list[Alternative]:
        return self.get_index_alternatives(
            self.entity_index,
            id_map,
            self.entity_mapping.default_variants(),
            query,
            k,
            **search_kwargs,
        )

    def get_property_alternatives(
        self,
        id_map: dict[int, set[str]] | None = None,
        query: str | None = None,
        k: int = 10,
        **search_kwargs: Any,
    ) -> list[Alternative]:
        return self.get_index_alternatives(
            self.property_index,
            id_map,
            self.property_mapping.default_variants(),
            query,
            k,
            **search_kwargs,
        )

    def get_index_alternatives(
        self,
        index: SearchIndex,
        id_map: dict[int, set[str]] | None = None,
        default_variants: set[str] | None = None,
        query: str | None = None,
        k: int = 10,
        **search_kwargs: Any,
    ) -> list[Alternative]:
        if id_map is not None:
            index = index.sub_index_by_ids(list(id_map))

        if query is None:
            if id_map is None:
                ids = list(range(min(k, len(index))))
            else:
                ids = sorted(id_map)[:k]
        else:
            kwargs = {}
            if index.get_type() == "similarity":
                # similarity index needs k and min score passed
                # to find_matches
                kwargs["k"] = k
                kwargs["min_score"] = search_kwargs.get("min_score")
            elif index.get_type() == "prefix":
                kwargs["no_refinement"] = search_kwargs.get("no_refinement", False)
                kwargs["min_keyword_length"] = search_kwargs.get("min_keyword_length")

            ids = [id for id, _ in index.find_matches(query, **kwargs)[:k]]

        alternatives = []
        for id in ids:
            if id_map is not None:
                variants = id_map[id]
            else:
                variants = default_variants

            alternative = self.build_alternative(index.get_row(id), variants)
            alternatives.append(alternative)

        return alternatives

    def get_temporary_index_alternatives(
        self,
        data: list[tuple[str, str, str | None]],
        query: str | None = None,
        k: int = 10,
        **search_kwargs: Any,
    ) -> list[Alternative]:
        if query is None:
            return [
                Alternative(
                    identifier=raw,
                    short_identifier=formatted,
                    infos=[info] if info else None,
                )
                for raw, formatted, info in data[:k]
            ]

        with tempfile.TemporaryDirectory() as temp_dir:
            # build index and search in it
            data_file = os.path.join(temp_dir, "data.tsv")
            index_dir = os.path.join(temp_dir, "index")
            os.makedirs(index_dir, exist_ok=True)
            self.logger.debug(
                f"Building temporary index in {temp_dir} "
                f"with data at {data_file} and index in {index_dir}"
            )

            # write raw data to temp file in temp dir
            with open(data_file, "w") as f:
                f.write("label\tscore\tsynonyms\tid\tinfos\n")
                for raw, formatted, info in data:
                    f.write(f"{formatted}\t0\t\t{raw}\t{info or ''}\n")

            # use a prefix index here because it is faster to build
            # and query
            PrefixIndex.build(data_file, index_dir)
            index = PrefixIndex.load(data_file, index_dir)

            alternatives = []
            matches = index.find_matches(
                query,
                min_keyword_length=search_kwargs.get("min_keyword_length"),
                no_refinement=search_kwargs.get("no_refinement", False),
            )
            for id, _ in matches[:k]:
                raw, formatted, info = data[id]
                alternatives.append(
                    Alternative(
                        identifier=raw,
                        short_identifier=formatted,
                        infos=[info] if info else None,
                    )
                )

            return alternatives

    def autocomplete_prefix(
        self,
        prefix: str,
        limit: int | None = None,
    ) -> tuple[str, Position]:
        return autocomplete_prefix(prefix, self.sparql_parser, limit)

    def autocomplete_sparql(
        self,
        sparql: str,
        limit: int | None = None,
    ) -> tuple[str, Position]:
        return autocomplete_sparql(sparql, self.sparql_parser, "search", limit)

    def get_default_search_items(
        self,
        position: Position,
    ) -> dict[ObjType, Any]:
        output = {}
        # entities can be subjects and objects
        add_entities = position in [Position.SUBJECT, Position.OBJECT]
        if add_entities:
            # None (full index) by default
            output[ObjType.ENTITY] = None

        # properties can only be properties
        add_properties = position == Position.PROPERTY
        if add_properties:
            # None (full index) by default
            output[ObjType.PROPERTY] = None

        # literals can only be objects
        add_literals = position == Position.OBJECT
        if add_literals:
            # empty by default
            output[ObjType.LITERAL] = []

        # other iris can always be subjects, properties, and objects
        # empty by default
        output[ObjType.OTHER] = []
        return output

    def get_search_items(
        self,
        sparql: str,
        position: Position,
        max_candidates: int | None = None,
        timeout: float | tuple[float, float] | None = REQUEST_TIMEOUT,
        max_retries: int = 1,
        skip_autocomplete: bool = False,
        fail_on_invalid: bool = False,
    ) -> dict[ObjType, Any]:
        # start with defaults
        search_items = self.get_default_search_items(position)

        typ = query_type(sparql, self.sparql_parser)
        if typ != "select":
            # fall back to full search on non-select queries
            raise SPARQLException("SPARQL query is not a SELECT query")

        elif not has_iri(sparql, self.sparql_parser):
            # contains no iris, with no restriction we do not need
            # to query the endpoint for autocompletion
            raise SPARQLException("SPARQL query contains no IRIs to constrain with")

        self.logger.debug(f"Getting search items with {sparql}")
        try:
            result = self.execute_sparql(
                sparql,
                timeout,
                max_retries,
            )
        except Exception as e:
            self.logger.debug(
                f"Getting autocompletion result for position {position} "
                f"with sparql {sparql} failed with error: {e}"
            )
            raise SPARQLException(f"SPARQL execution failed: {e}")

        # some checks that should not happen, just to be sure
        if not isinstance(result, SelectResult):
            raise SPARQLException("SPARQL query is not a select query")
        elif result.num_columns != 1:
            raise SPARQLException("SPARQL query does not return a single column")
        elif max_candidates is not None and len(result) > max_candidates:
            raise SPARQLException(
                f"Got more than the maximum supported number of {position.value} "
                f"candidates ({max_candidates:,})"
            )

        self.logger.debug(
            f"Got {len(result):,} fitting items for position {position} "
            f"with sparql '{sparql}'"
        )

        # split result into entities, properties, other iris
        # and literals
        start = time.perf_counter()
        parsed_search_items = self.parse_bindings(row[0] for row in result.rows())
        end = time.perf_counter()
        self.logger.debug(
            f"Parsing {len(result):,} search items took {1000 * (end - start):.2f}ms"
        )

        # overwrite defaults where needed
        for obj_type in search_items:
            if obj_type not in parsed_search_items:
                continue

            search_items[obj_type] = parsed_search_items[obj_type]

        return search_items

    def get_selection_alternatives(
        self,
        search_query: str | None,
        search_items: dict[ObjType, Any],
        k: int,
        **search_kwargs: Any,
    ) -> dict[ObjType, list[Alternative]]:
        self.logger.debug(
            f'Getting top {k} selection alternatives with query "{search_query}" for '
            f'object types {", ".join(obj_type.value for obj_type in search_items)}'
        )
        alternatives = {}

        start = time.perf_counter()

        if ObjType.ENTITY in search_items:
            alternatives[ObjType.ENTITY] = self.get_entity_alternatives(
                search_items[ObjType.ENTITY],
                search_query,
                k,
                **search_kwargs,
            )

        if ObjType.PROPERTY in search_items:
            alternatives[ObjType.PROPERTY] = self.get_property_alternatives(
                search_items[ObjType.PROPERTY],
                search_query,
                k,
                **search_kwargs,
            )

        end = time.perf_counter()
        self.logger.debug(
            f"Getting entity and property alternatives "
            f"took {1000 * (end - start):.2f}ms"
        )

        start = time.perf_counter()

        for obj_type in [ObjType.OTHER, ObjType.LITERAL]:
            if obj_type not in search_items:
                continue

            alternatives[obj_type] = self.get_temporary_index_alternatives(
                search_items[obj_type],
                search_query,
                k,
                **search_kwargs,
            )

        end = time.perf_counter()
        self.logger.debug(
            f"Getting other and literal alternatives "
            f"took {1000 * (end - start):.2f}ms"
        )

        return alternatives

    def format_selections(
        self,
        selections: list[Selection],
    ) -> str:
        rename_obj_type = [
            (ObjType.ENTITY, "entities"),
            (ObjType.PROPERTY, "properties"),
        ]
        grouped = group_selections(selections)
        return "\n\n".join(
            f"Using {name}:\n"
            + "\n".join(
                alt.get_selection_string(include_variants=variants)
                for alt, variants in grouped[obj_type]
            )
            for obj_type, name in rename_obj_type
            if obj_type in grouped
        )
