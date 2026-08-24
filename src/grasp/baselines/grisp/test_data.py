from unittest.mock import Mock

from grasp.baselines.grisp.data import (
    BOI,
    EOI,
    MAX_PLACEHOLDER_QUERIES,
    SEP,
    Info,
    Skeleton,
    display_nl_iri,
    extract_queries_and_variants_from_nl_iri,
    extract_values_from_nl_iri,
    merge_skeletons,
    search_alternatives,
    split_values,
)
from grasp.baselines.grisp.utils import load_sparql_parser
from grasp.sparql.types import Alternative, ObjType

SPARQL_PARSER = load_sparql_parser()

# entities support variants, properties do not, mirroring what the run passes
OBJ_TYPES = {ObjType.ENTITY: False, ObjType.PROPERTY: True}


def nl_iri(value: str) -> dict:
    return {"name": "NL_IRI", "value": f"{BOI}{value}{EOI}"}


def parse(skeleton: str) -> Skeleton:
    return Skeleton.parse(skeleton, SPARQL_PARSER)


class TestPlaceholderValues:
    def test_single_wording(self):
        assert split_values("born in") == ["born in"]

    def test_several_wordings(self):
        assert split_values(f"born in{SEP}place of birth") == [
            "born in",
            "place of birth",
        ]

    def test_blank_wordings_dropped(self):
        assert split_values(f"born in{SEP} {SEP}") == ["born in"]

    def test_empty_value(self):
        assert split_values("") == [""]

    def test_values_from_nl_iri(self):
        assert extract_values_from_nl_iri(nl_iri(f"a{SEP}b")) == ["a", "b"]

    def test_variant_parsed_per_wording(self):
        pairs = extract_queries_and_variants_from_nl_iri(
            nl_iri(f"population (wdt){SEP}number of inhabitants{SEP}population (p)")
        )
        assert pairs == [
            ("population", "wdt"),
            ("number of inhabitants", None),
            ("population", "p"),
        ]

    def test_display_uses_first_wording(self):
        assert (
            display_nl_iri(nl_iri(f"born in{SEP}place of birth"))
            == f"{BOI}born in{EOI}"
        )


class TestBuildQueries:
    def test_wordings_kept_per_object_type(self):
        info = Info(
            prefix="",
            sparql="",
            queries=["population", "inhabitants"],
            variants=["wdt", None],
            values=["population (wdt)", "inhabitants"],
        )
        queries = info.build_queries(OBJ_TYPES)
        # properties support variants, so the variant is stripped off and passed
        # separately; entities do not, so the raw wording is searched
        assert queries[ObjType.PROPERTY] == [
            ("population", "wdt"),
            ("inhabitants", None),
        ]
        assert queries[ObjType.ENTITY] == [
            ("population (wdt)", None),
            ("inhabitants", None),
        ]

    def test_wordings_aligned_across_object_types(self):
        # the loop searches wording i for whatever object type the placeholder
        # turns out to take, so index i must mean the same wording everywhere
        info = Info(
            prefix="",
            sparql="",
            queries=["population", "inhabitants"],
            variants=["wdt", None],
            values=["population (wdt)", "inhabitants"],
        )
        queries = info.build_queries(OBJ_TYPES)
        assert info.num_queries == 2
        assert all(len(pairs) == info.num_queries for pairs in queries.values())


class TestMergeSkeletons:
    def test_wordings_combined(self):
        base = parse("SELECT ?x WHERE { <iri>Harz</iri> <iri>has part</iri> ?x }")
        other = parse(
            "SELECT ?x WHERE { <iri>Harz mountains</iri> <iri>contains</iri> ?x }"
        )
        merged = merge_skeletons(base, other, SPARQL_PARSER)
        assert merged is not None
        assert extract_values_from_nl_iri(merged.nl_iris[0]) == [
            "Harz",
            "Harz mountains",
        ]
        assert extract_values_from_nl_iri(merged.nl_iris[1]) == ["has part", "contains"]

    def test_identical_wordings_not_duplicated(self):
        base = parse("SELECT ?x WHERE { <iri>Harz</iri> <iri>has part</iri> ?x }")
        other = parse("SELECT ?y WHERE { <iri>Harz</iri> <iri>contains</iri> ?y }")
        merged = merge_skeletons(base, other, SPARQL_PARSER)
        assert merged is not None
        assert extract_values_from_nl_iri(merged.nl_iris[0]) == ["Harz"]

    def test_wordings_capped(self):
        merged = parse("SELECT ?x WHERE { <iri>a</iri> <iri>p</iri> ?x }")
        for wording in ("b", "c", "d", "e"):
            other = parse(f"SELECT ?x WHERE {{ <iri>{wording}</iri> <iri>p</iri> ?x }}")
            merged = merge_skeletons(merged, other, SPARQL_PARSER)
            assert merged is not None
        assert (
            len(extract_values_from_nl_iri(merged.nl_iris[0]))
            == MAX_PLACEHOLDER_QUERIES
        )

    def test_best_skeleton_stays_first_everywhere(self):
        # the loop tries wording 0 of every placeholder first, so wording 0 has
        # to come from the best skeleton throughout
        best = parse("SELECT ?x WHERE { <iri>Harz</iri> <iri>has part</iri> ?x }")
        second = parse("SELECT ?x WHERE { <iri>Harz mts</iri> <iri>contains</iri> ?x }")
        third = parse(
            "SELECT ?x WHERE { <iri>Harz range</iri> <iri>includes</iri> ?x }"
        )

        merged = merge_skeletons(best, second, SPARQL_PARSER)
        assert merged is not None
        merged = merge_skeletons(merged, third, SPARQL_PARSER)
        assert merged is not None

        assert extract_values_from_nl_iri(merged.nl_iris[0]) == [
            "Harz",
            "Harz mts",
            "Harz range",
        ]
        assert extract_values_from_nl_iri(merged.nl_iris[1]) == [
            "has part",
            "contains",
            "includes",
        ]

    def test_mismatched_placeholder_count(self):
        base = parse("SELECT ?x WHERE { <iri>a</iri> <iri>p</iri> ?x }")
        other = parse(
            "SELECT ?x WHERE { <iri>a</iri> <iri>p</iri> ?x . ?x <iri>q</iri> ?y }"
        )
        assert merge_skeletons(base, other, SPARQL_PARSER) is None


class TestMergedSkeletonSelection:
    def test_prompt_shows_first_wording_only(self):
        base = parse("SELECT ?x WHERE { <iri>Harz</iri> <iri>has part</iri> ?x }")
        other = parse("SELECT ?x WHERE { <iri>Harz</iri> <iri>contains</iri> ?x }")
        merged = merge_skeletons(base, other, SPARQL_PARSER)
        assert merged is not None

        info = merged.prepare_for_selection()
        assert SEP not in info.sparql
        assert SEP not in merged.materialize_partial()
        # the second placeholder is still unresolved, shown with its first wording
        assert "has part" in merged.materialize_partial()
        assert "contains" not in merged.materialize_partial()

    def test_all_wordings_searched(self):
        base = parse("SELECT ?x WHERE { ?x <iri>has part</iri> <iri>Harz</iri> }")
        other = parse("SELECT ?x WHERE { ?x <iri>contains</iri> <iri>Harz</iri> }")
        merged = merge_skeletons(base, other, SPARQL_PARSER)
        assert merged is not None

        info = merged.prepare_for_selection()
        assert info.queries == ["has part", "contains"]
        assert info.build_queries(OBJ_TYPES)[ObjType.PROPERTY] == [
            ("has part", None),
            ("contains", None),
        ]


class TestSearchAlternatives:
    def manager(self, results: dict[str, list[str]]) -> Mock:
        manager = Mock()
        manager.search_index.side_effect = lambda index_name, query, *args: [
            Alternative(identifier=identifier) for identifier in results.get(query, [])
        ]
        return manager

    def queries(self) -> dict:
        return {ObjType.PROPERTY: [("has part", None), ("contains", None)]}

    def test_searches_the_requested_wording(self):
        manager = self.manager({"has part": ["p1"], "contains": ["p2"]})
        groups = search_alternatives(
            manager,
            [ObjType.PROPERTY],
            {ObjType.PROPERTY: None},
            self.queries(),
            1,
            10,
            Mock(),
        )
        assert [a.identifier for a in groups[ObjType.PROPERTY]] == ["p2"]
        assert manager.search_index.call_count == 1

    def test_empty_result_leaves_group_out(self):
        manager = self.manager({})
        groups = search_alternatives(
            manager,
            [ObjType.PROPERTY],
            {ObjType.PROPERTY: None},
            self.queries(),
            0,
            10,
            Mock(),
        )
        assert groups == {}

    def test_constraint_passed_to_search(self):
        manager = self.manager({"has part": ["p1"]})
        identifier_map = {"p1": ["wdt"]}
        search_alternatives(
            manager,
            [ObjType.PROPERTY],
            {ObjType.PROPERTY: identifier_map},
            self.queries(),
            0,
            10,
            Mock(),
        )
        assert manager.search_index.call_args[0] == (
            "properties",
            "has part",
            10,
            identifier_map,
        )
