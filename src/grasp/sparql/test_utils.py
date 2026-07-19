import json

import pytest

from grasp.sparql.types import Position
from grasp.sparql.utils import (
    SPARQLException,
    SPARQLExecuteException,
    stream_with_timeout,
    complete_prefix,
    derive_constraint_query_from_sparql,
    find,
    find_connected_top_level_triples,
    fix_prefixes,
    load_iri_and_literal_parser,
    load_sparql_parser,
    parse_string,
    parse_to_string_with_whitespace,
    query_type,
)

SPARQL_PARSER = load_sparql_parser()
IRI_PARSER = load_iri_and_literal_parser()

PREFIXES = {
    "wd": "http://example.org/entity/",
    "wdt": "http://example.org/prop/",
}


class _FakeResponse:
    def __init__(self, data: dict, chunk_size: int = 16) -> None:
        self.data = json.dumps(data).encode("utf-8")
        self.encoding = "utf-8"
        self.chunk_size = chunk_size
        self.closed = False

    def iter_content(self, chunk_size: int | None = None):
        chunk_size = self.chunk_size if chunk_size is None else chunk_size
        for i in range(0, len(self.data), chunk_size):
            yield self.data[i : i + chunk_size]

    def close(self) -> None:
        self.closed = True


def _fix(sparql: str, **kwargs) -> str:
    return fix_prefixes(sparql, SPARQL_PARSER, IRI_PARSER, PREFIXES, **kwargs)


def _parse(sparql: str) -> dict:
    return SPARQL_PARSER.parse(sparql)


def _derive(query: str, **kwargs):
    # the placeholder being resolved is marked <rep>...</rep>
    assert "<rep>" in query, "Expected <rep> marker in test query"
    return derive_constraint_query_from_sparql(query, SPARQL_PARSER, **kwargs)


class TestStreamWithTimeout:
    def test_streams_select_result(self):
        data = {
            "head": {"vars": ["x"]},
            "results": {
                "bindings": [
                    {"x": {"type": "uri", "value": "http://example.org/entity/e1"}},
                    {"x": {"type": "uri", "value": "http://example.org/entity/e2"}},
                ]
            },
        }
        response = _FakeResponse(data)

        result = stream_with_timeout("SELECT ?x WHERE { ?x ?p ?o }", response)  # type: ignore

        assert result == data
        assert response.closed

    def test_errors_when_select_result_exceeds_max_rows(self):
        data = {
            "head": {"vars": ["x"]},
            "results": {
                "bindings": [
                    {"x": {"type": "uri", "value": "http://example.org/entity/e1"}},
                    {"x": {"type": "uri", "value": "http://example.org/entity/e2"}},
                ]
            },
        }
        response = _FakeResponse(data)

        with pytest.raises(SPARQLExecuteException, match="exceeded 1 rows"):
            stream_with_timeout(
                "SELECT ?x WHERE { ?x ?p ?o }",
                response,  # type: ignore
                sparql_result_max_rows=1,
            )
        assert response.closed

    def test_streams_ask_result(self):
        data = {"head": {}, "boolean": True}
        response = _FakeResponse(data)

        result = stream_with_timeout("ASK { ?x ?p ?o }", response)  # type: ignore

        assert result == {"boolean": True}
        assert response.closed


class TestQueryType:
    def test_complete_select(self):
        assert query_type("SELECT ?x WHERE { ?x ?p ?o }", SPARQL_PARSER) == "select"

    def test_complete_ask(self):
        assert query_type("ASK { ?x ?p ?o }", SPARQL_PARSER) == "ask"

    def test_complete_construct(self):
        assert (
            query_type("CONSTRUCT { ?x ?p ?o } WHERE { ?x ?p ?o }", SPARQL_PARSER)
            == "construct"
        )

    def test_complete_describe(self):
        assert query_type("DESCRIBE ?x WHERE { ?x ?p ?o }", SPARQL_PARSER) == "describe"

    def test_prefix_select_missing_triple(self):
        assert (
            query_type(
                "SELECT ?film ?filmLabel WHERE {\n  ?film ",
                SPARQL_PARSER,
                is_prefix=True,
            )
            == "select"
        )

    def test_prefix_select_at_property_position(self):
        assert (
            query_type(
                "SELECT ?x WHERE { ?x ",
                SPARQL_PARSER,
                is_prefix=True,
            )
            == "select"
        )

    def test_prefix_construct(self):
        assert (
            query_type(
                "CONSTRUCT { ?s ?p ?o } WHERE { ?s ",
                SPARQL_PARSER,
                is_prefix=True,
            )
            == "construct"
        )

    def test_prefix_ask(self):
        assert query_type("ASK { ?s ", SPARQL_PARSER, is_prefix=True) == "ask"

    def test_prefix_describe(self):
        assert (
            query_type(
                "DESCRIBE ?x WHERE { ?x ",
                SPARQL_PARSER,
                is_prefix=True,
            )
            == "describe"
        )

    def test_prefix_select_with_subselect(self):
        assert (
            query_type(
                "SELECT ?x WHERE { { SELECT ?film WHERE { ?film ",
                SPARQL_PARSER,
                is_prefix=True,
            )
            == "select"
        )

    def test_unparsable_returns_none(self):
        assert query_type("NOT VALID SPARQL %%%", SPARQL_PARSER) is None


class TestFixPrefixes:
    def test_replaces_iri_with_prefix(self):
        result = _fix("SELECT ?s WHERE { ?s <http://example.org/prop/p1> ?o }")
        assert result == (
            "PREFIX wdt: <http://example.org/prop/>\nSELECT ?s WHERE { ?s wdt:p1 ?o }"
        )

    def test_preserves_spaces(self):
        result = _fix("SELECT  ?s  WHERE  {  ?s  <http://example.org/prop/p1>  ?o  }")
        assert result == (
            "PREFIX wdt: <http://example.org/prop/>\n"
            "SELECT  ?s  WHERE  {  ?s  wdt:p1  ?o  }"
        )

    def test_preserves_newlines_and_indentation(self):
        result = _fix("SELECT ?s WHERE {\n  ?s <http://example.org/prop/p1> ?o\n}")
        assert result == (
            "PREFIX wdt: <http://example.org/prop/>\n"
            "SELECT ?s WHERE {\n  ?s wdt:p1 ?o\n}"
        )

    def test_preserves_tabs(self):
        result = _fix("SELECT\t?s\tWHERE\t{\n\t?s\t<http://example.org/prop/p1>\t?o\n}")
        assert result == (
            "PREFIX wdt: <http://example.org/prop/>\n"
            "SELECT\t?s\tWHERE\t{\n\t?s\twdt:p1\t?o\n}"
        )

    def test_existing_prefix(self):
        result = _fix(
            "PREFIX wd: <http://example.org/entity/>\nSELECT ?s WHERE { ?s wd:e1 ?o }"
        )
        assert result == (
            "PREFIX wd: <http://example.org/entity/>\nSELECT ?s WHERE { ?s wd:e1 ?o }"
        )

    def test_existing_prefix_whitespace_preserved(self):
        result = _fix(
            "PREFIX wd: <http://example.org/entity/>\n"
            "SELECT  ?s  WHERE  {\n  ?s  wd:e1  ?o\n}"
        )
        assert result == (
            "PREFIX wd: <http://example.org/entity/>\n"
            "SELECT  ?s  WHERE  {\n  ?s  wd:e1  ?o\n}"
        )

    def test_no_prefixes_needed(self):
        result = _fix("SELECT  ?s  WHERE  {\n  ?s  ?p  ?o\n}")
        assert result == "SELECT  ?s  WHERE  {\n  ?s  ?p  ?o\n}"

    def test_remove_known(self):
        result = _fix(
            "PREFIX wd: <http://example.org/entity/>\nSELECT ?s WHERE { ?s wd:e1 ?o }",
            remove_known=True,
        )
        assert result == "SELECT ?s WHERE { ?s wd:e1 ?o }"

    def test_sort_prefixes(self):
        result = _fix(
            "SELECT ?s WHERE { "
            "?s <http://example.org/prop/p1> <http://example.org/entity/e1> "
            "}",
            sort=True,
        )
        assert result == (
            "PREFIX wd: <http://example.org/entity/>\n"
            "PREFIX wdt: <http://example.org/prop/>\n"
            "SELECT ?s WHERE { ?s wdt:p1 wd:e1 }"
        )

    def test_unknown_iri_not_replaced(self):
        result = _fix("SELECT ?s WHERE { ?s <http://unknown.org/foo> ?o }")
        assert result == "SELECT ?s WHERE { ?s <http://unknown.org/foo> ?o }"

    def test_preserves_comments(self):
        result = _fix(
            "SELECT ?s WHERE {\n"
            "  # find all properties of entity\n"
            "  ?s <http://example.org/prop/p1> ?o\n"
            "}"
        )
        assert result == (
            "PREFIX wdt: <http://example.org/prop/>\n"
            "SELECT ?s WHERE {\n"
            "  # find all properties of entity\n"
            "  ?s wdt:p1 ?o\n"
            "}"
        )


class TestFindConnectedTopLevelTriples:
    def test_keeps_only_connected_component_of_selected_var(self):
        parse = _parse(
            "SELECT ?b WHERE { "
            "?a <http://example.org/p1> ?b . "
            "?b <http://example.org/p2> ?c . "
            "?x <http://example.org/p3> ?y "
            "}"
        )

        result = find_connected_top_level_triples(parse, "?b")

        assert len(result) == 2
        assert "?a <http://example.org/p1> ?b" in result[0]
        assert "?b <http://example.org/p2> ?c" in result[1]
        assert all("?x <http://example.org/p3> ?y" not in block for block in result)

    def test_keeps_transitively_connected_triples(self):
        parse = _parse(
            "SELECT ?b WHERE { "
            "?a <http://example.org/p1> ?b . "
            "?b <http://example.org/p2> ?c . "
            "?c <http://example.org/p3> ?d . "
            "?x <http://example.org/p4> ?y "
            "}"
        )

        result = find_connected_top_level_triples(parse, "?b")

        assert len(result) == 3
        assert any("?a <http://example.org/p1> ?b" in block for block in result)
        assert any("?b <http://example.org/p2> ?c" in block for block in result)
        assert any("?c <http://example.org/p3> ?d" in block for block in result)
        assert all("?x <http://example.org/p4> ?y" not in block for block in result)

    def test_returns_empty_when_selected_var_not_in_top_level_triples(self):
        parse = _parse(
            "SELECT ?z WHERE { "
            "?a <http://example.org/p1> ?b . "
            "?b <http://example.org/p2> ?c "
            "}"
        )

        result = find_connected_top_level_triples(parse, "?z")

        assert result == []


class TestCompletePrefix:
    def test_determines_subject_position_for_simple_triple_prefix(self):
        _, position, _ = complete_prefix("SELECT ?x WHERE { ", SPARQL_PARSER)
        assert position == "subject"

    def test_determines_property_position_for_simple_triple_prefix(self):
        _, position, _ = complete_prefix("SELECT ?x WHERE { ?s ", SPARQL_PARSER)
        assert position == "property"

    def test_determines_object_position_for_simple_triple_prefix(self):
        _, position, _ = complete_prefix(
            "SELECT ?x WHERE { ?s <http://example.org/p1> ",
            SPARQL_PARSER,
        )
        assert position == "object"

    def test_fails_within_filter_function_arguments(self):
        with pytest.raises(SPARQLException):
            complete_prefix(
                "SELECT ?x WHERE { FILTER(CONTAINS(?label, ",
                SPARQL_PARSER,
            )

    def test_fails_within_property_path(self):
        with pytest.raises(SPARQLException):
            complete_prefix(
                "SELECT ?x WHERE { ?s <http://example.org/p1>/",
                SPARQL_PARSER,
            )


class TestDeriveConstraintQueryFromSparql:
    def test_keeps_only_connected_component(self):
        query = (
            "SELECT ?b WHERE { "
            "?a <http://example.org/p1> ?b . "
            "?b <http://example.org/p2> <rep>x</rep> "
            "}"
        )

        result, _ = _derive(query)

        assert result is not None
        assert "?a <http://example.org/p1> ?b" in result
        assert "?b <http://example.org/p2>" in result
        assert "<http://example.org/p3>" not in result

    def test_drops_disconnected_triples_from_constraint_query(self):
        query = (
            "SELECT ?b WHERE { "
            "?x <http://example.org/p3> ?y . "
            "?a <http://example.org/p1> ?b . "
            "?b <http://example.org/p2> <rep>x</rep> . "
            "?z <http://example.org/p4> ?w "
            "}"
        )

        result, _ = _derive(query)

        assert result is not None
        assert "?a <http://example.org/p1> ?b" in result
        assert "?b <http://example.org/p2>" in result
        assert "?x <http://example.org/p3> ?y" not in result
        assert "?z <http://example.org/p4> ?w" not in result

    def test_keeps_transitively_connected_triples_in_constraint_query(self):
        query = (
            "SELECT ?b WHERE { "
            "?a <http://example.org/p1> ?b . "
            "?b <http://example.org/p2> ?c . "
            "?c <http://example.org/p3> <rep>x</rep> "
            "}"
        )

        result, _ = _derive(query)

        assert result is not None
        assert "?a <http://example.org/p1> ?b" in result
        assert "?b <http://example.org/p2> ?c" in result
        assert "?c <http://example.org/p3>" in result

    def test_raises_when_placeholder_is_not_in_a_triple(self):
        query = "SELECT ?z WHERE { ?a <http://example.org/p1> ?z . FILTER(?z != <rep>x</rep>) }"

        with pytest.raises(SPARQLException):
            _derive(query)

    def test_ignores_triples_inside_optional(self):
        query = (
            "SELECT ?a WHERE { "
            "OPTIONAL { ?a <http://example.org/p2> ?c } . "
            "?a <http://example.org/p1> <rep>x</rep> "
            "}"
        )

        result, _ = _derive(query)

        assert result is not None
        assert "?a <http://example.org/p1>" in result
        assert "http://example.org/p2" not in result

    def test_ignores_triples_inside_union(self):
        query = (
            "SELECT ?a WHERE { "
            "{ ?a <http://example.org/p2> ?b } UNION { ?a <http://example.org/p3> ?c } . "
            "?a <http://example.org/p1> <rep>x</rep> "
            "}"
        )

        result, _ = _derive(query)

        assert result is not None
        assert "?a <http://example.org/p1>" in result
        assert "http://example.org/p2" not in result
        assert "http://example.org/p3" not in result

    def test_ignores_triples_inside_minus(self):
        query = (
            "SELECT ?a WHERE { "
            "?a <http://example.org/p1> ?b . "
            "MINUS { ?b <http://example.org/p2> ?c } . "
            "?b <http://example.org/p3> <rep>x</rep> "
            "}"
        )

        result, _ = _derive(query)

        assert result is not None
        assert "?a <http://example.org/p1> ?b" in result
        assert "?b <http://example.org/p3>" in result
        assert "http://example.org/p2" not in result

    def test_returns_none_for_single_all_variable_triple(self):
        # ?s ?p ?current — all variables, so no constraint
        query = "SELECT ?x WHERE { ?s ?p <rep>x</rep> }"
        result, _ = _derive(query)
        assert result is None

    def test_returns_constraint_for_single_triple_with_iri_predicate(self):
        query = "SELECT ?x WHERE { ?s <http://example.org/p1> <rep>x</rep> }"
        result, _ = _derive(query)
        assert result is not None
        assert "<http://example.org/p1>" in result

    def test_returns_none_for_all_variable_triples(self):
        # nothing resolved anywhere, so there is nothing to constrain on
        query = "SELECT ?x WHERE { ?a ?b ?x . ?x ?c <rep>x</rep> }"
        result, _ = _derive(query)
        assert result is None

    def test_returns_none_constraint_inside_optional(self):
        query = (
            "SELECT ?a WHERE { "
            "?a <http://example.org/p1> ?b . "
            "OPTIONAL { ?b <http://example.org/p2> <rep>x</rep> } "
            "}"
        )

        result, _ = _derive(query)
        assert result is None

    # the following need the full query, not a bare prefix: resolved neighbours
    # to the right of the placeholder, and unresolved placeholders elsewhere.

    def test_uses_resolved_right_neighbour(self):
        # object is a resolved entity to the right of the property placeholder
        query = "SELECT ?x WHERE { ?x <rep>x</rep> <http://example.org/o> }"
        result, position = _derive(query)
        assert result is not None
        assert position == Position.PROPERTY
        assert "<http://example.org/o>" in result

    def test_uses_resolved_neighbours_on_both_sides(self):
        query = (
            "SELECT ?x WHERE { "
            "<http://example.org/s> <http://example.org/p1> ?x . "
            "?x <rep>x</rep> <http://example.org/o> "
            "}"
        )
        result, position = _derive(query)
        assert result is not None
        assert position == Position.PROPERTY
        assert "<http://example.org/s> <http://example.org/p1> ?x" in result
        assert "<http://example.org/o>" in result

    def test_unresolved_placeholder_neighbour_is_ignored(self):
        # unresolved object placeholder becomes a variable, no crash
        query = (
            "SELECT ?x WHERE { <http://example.org/s> <rep>x</rep> <iri>entity</iri> }"
        )
        result, position = _derive(query)
        assert result is not None
        assert position == Position.PROPERTY
        assert "<http://example.org/s>" in result
        assert "<iri>" not in result

    def test_unresolved_placeholder_in_filter_does_not_break_constraint(self):
        # freebase WQSP pattern: topic entity repeated in a FILTER, still
        # unresolved while a property is resolved
        query = (
            "SELECT DISTINCT ?x WHERE { "
            "FILTER(?x != <iri>Jackie Robinson</iri>) "
            "<http://example.org/s> <rep>x</rep> ?y "
            "}"
        )
        result, position = _derive(query)
        assert result is not None
        assert position == Position.PROPERTY
        assert "<http://example.org/s>" in result
        assert "<iri>" not in result

    def test_multiple_unresolved_placeholders_get_distinct_vars(self):
        # two unresolved placeholders must not collapse into one variable
        query = (
            "SELECT ?x WHERE { "
            "<iri>entity a</iri> <http://example.org/p1> ?x . "
            "?x <rep>x</rep> <iri>entity b</iri> "
            "}"
        )
        result, position = _derive(query)
        assert result is not None
        assert position == Position.PROPERTY
        assert "<iri>" not in result

    def test_raises_when_no_current_placeholder(self):
        query = "SELECT ?x WHERE { ?x <http://example.org/p1> ?o }"
        with pytest.raises(SPARQLException):
            derive_constraint_query_from_sparql(query, SPARQL_PARSER)

    # pruning of all-variable padding triples (iri_only)

    def test_drops_all_variable_padding_triple(self):
        # freebase CVT case: resolving the first property; the second hop is
        # still unresolved, so its (all-variable) triple must be dropped
        query = (
            "SELECT ?x WHERE { "
            "<http://example.org/s> <rep>x</rep> ?y . "
            "?y <iri>second hop</iri> ?x "
            "}"
        )
        result, _ = _derive(query)
        assert result is not None
        assert "<http://example.org/s>" in result
        # only the cursor's own triple survives, no second hop
        assert " . " not in result
        assert "?x" not in result

    def test_keeps_resolved_second_hop(self):
        # both hops carry IRIs (subject and a resolved neighbour), so the full
        # two-hop constraint is kept
        query = (
            "SELECT ?x WHERE { "
            "<http://example.org/s> <http://example.org/p1> ?y . "
            "?y <rep>x</rep> <http://example.org/o> "
            "}"
        )
        result, _ = _derive(query)
        assert result is not None
        assert "<http://example.org/s> <http://example.org/p1> ?y" in result
        assert "<http://example.org/o>" in result

    def test_keeps_all_variable_triple_connecting_two_resolved(self):
        # the middle triple is all-variable but links the cursor to a resolved
        # triple, so it (and the resolved triple) must be kept
        query = (
            "SELECT ?x WHERE { "
            "<http://example.org/s> <rep>x</rep> ?y . "
            "?y <iri>mid</iri> ?z . "
            "?z <http://example.org/p3> <http://example.org/o> "
            "}"
        )
        result, _ = _derive(query)
        assert result is not None
        assert "<http://example.org/s>" in result
        # the connecting triple and the far resolved triple are both kept
        assert "?y" in result and "?z" in result
        assert "<http://example.org/o>" in result

    def test_drops_dangling_branch_but_keeps_connector(self):
        # ?y -> ?z -> <o> is a real (connected) constraint and is kept; the
        # ?y -> ?w branch dangles into nothing and is dropped
        query = (
            "SELECT ?x WHERE { "
            "<http://example.org/s> <rep>x</rep> ?y . "
            "?y <iri>mid</iri> ?z . "
            "?z <http://example.org/p3> <http://example.org/o> . "
            "?y <iri>dangling</iri> ?w "
            "}"
        )
        result, _ = _derive(query)
        assert result is not None
        assert "<http://example.org/o>" in result
        assert "?w" not in result


class TestParseToStringWithWhitespace:
    def _roundtrip(self, sparql: str) -> str:
        parse, _ = parse_string(sparql, SPARQL_PARSER)
        return parse_to_string_with_whitespace(parse, sparql.encode())

    def test_root_roundtrip_simple(self):
        sparql = "SELECT ?s WHERE { ?s ?p ?o }"
        assert self._roundtrip(sparql) == sparql

    def test_root_roundtrip_preserves_internal_whitespace(self):
        sparql = "SELECT  ?s\n  ?p\nWHERE {\n  ?s  ?p   ?o .\n} LIMIT 10"
        assert self._roundtrip(sparql) == sparql

    def test_root_roundtrip_preserves_trailing_clause(self):
        sparql = "SELECT ?s WHERE { ?s ?p ?o } LIMIT 10"
        assert self._roundtrip(sparql) == sparql

    def test_subtree_select_clause(self):
        sparql = (
            "SELECT ?id ?value ?tags WHERE {\n  ?s ?p ?o\n} "
            "ORDER BY DESC(?score) ?id DESC(?tags)"
        )
        parse, _ = parse_string(sparql, SPARQL_PARSER)
        clause = find(parse, "SelectClause")
        assert clause is not None
        out = parse_to_string_with_whitespace(clause, sparql.encode())
        assert out == "SELECT ?id ?value ?tags"

    def test_subtree_solution_modifier(self):
        sparql = (
            "SELECT ?id ?value ?type WHERE {\n  ?s ?p ?o\n} ORDER BY ?id ?type ?value"
        )
        parse, _ = parse_string(sparql, SPARQL_PARSER)
        sol_mod = find(parse, "SolutionModifier")
        assert sol_mod is not None
        out = parse_to_string_with_whitespace(sol_mod, sparql.encode())
        assert out == "ORDER BY ?id ?type ?value"

    def test_subtree_does_not_leak_trailing_bytes(self):
        # regression: parse_to_string_with_whitespace used to append
        # encoded[pos:] unconditionally, leaking everything after the
        # subtree's last terminal into the result
        sparql = "SELECT ?x WHERE { ?x ?p ?o } ORDER BY ?x LIMIT 5"
        parse, _ = parse_string(sparql, SPARQL_PARSER)
        clause = find(parse, "SelectClause")
        assert clause is not None
        out = parse_to_string_with_whitespace(clause, sparql.encode())
        assert out == "SELECT ?x"

    def test_subtree_does_not_leak_leading_bytes(self):
        sparql = "PREFIX ex: <http://example.org/>\nSELECT ?x WHERE { ?x ?p ?o }"
        parse, _ = parse_string(sparql, SPARQL_PARSER)
        clause = find(parse, "SelectClause")
        assert clause is not None
        out = parse_to_string_with_whitespace(clause, sparql.encode())
        assert out == "SELECT ?x"
