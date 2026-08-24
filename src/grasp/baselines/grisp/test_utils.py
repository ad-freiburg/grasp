from grasp.baselines.grisp.utils import (
    load_sparql_parser,
    normalize_skeleton_parse,
    skeleton_wording_key,
)
from grasp.sparql.utils import find_all, parse_to_string

SPARQL_PARSER = load_sparql_parser()


def normalize(skeleton: str, blank_placeholders: bool = True) -> str:
    normalized = normalize_skeleton_parse(
        SPARQL_PARSER.parse(skeleton),
        {},
        blank_placeholders,
    )
    return parse_to_string(normalized)


class TestNormalizeSkeletonParse:
    def test_variables_renamed_by_first_appearance(self):
        assert normalize("SELECT ?x WHERE { ?x <iri>a</iri> ?y }") == normalize(
            "SELECT ?s WHERE { ?s <iri>a</iri> ?o }"
        )

    def test_repeated_variable_keeps_one_name(self):
        normalized = normalize("SELECT ?x WHERE { ?x <iri>a</iri> ?x }")
        assert normalized.count("?var0") == 3
        assert "?var1" not in normalized

    def test_placeholders_blanked(self):
        normalized = normalize("SELECT ?x WHERE { ?x <iri>born in</iri> ?y }")
        assert "<IRI>" in normalized
        assert "born in" not in normalized

    def test_placeholders_kept_when_not_blanked(self):
        normalized = normalize(
            "SELECT ?x WHERE { ?x <iri>born in</iri> ?y }",
            blank_placeholders=False,
        )
        assert "born in" in normalized

    def test_original_parse_not_modified(self):
        parse = SPARQL_PARSER.parse("SELECT ?x WHERE { ?x <iri>a</iri> ?y }")
        normalize_skeleton_parse(parse, {})
        variables = [var["value"] for var in find_all(parse, "VAR1")]
        assert variables == ["?x", "?x", "?y"]


class TestSkeletonWordingKey:
    def wording_key(self, skeleton: str) -> str:
        return skeleton_wording_key(SPARQL_PARSER.parse(skeleton))

    def test_same_key_for_different_wording(self):
        assert self.wording_key(
            "SELECT ?x WHERE { ?x <iri>born in</iri> ?y }"
        ) == self.wording_key("SELECT ?s WHERE { ?s <iri>place of birth</iri> ?o }")

    def test_different_key_for_reordered_triples(self):
        # merging lines placeholders up by document order, so the order of
        # the triples has to match as well
        assert self.wording_key(
            "SELECT ?x WHERE { ?x <iri>a</iri> ?y . ?y <iri>b</iri> <iri>c</iri> }"
        ) != self.wording_key(
            "SELECT ?x WHERE { ?y <iri>b</iri> <iri>c</iri> . ?x <iri>a</iri> ?y }"
        )

    def test_different_key_for_extra_placeholder(self):
        assert self.wording_key(
            "SELECT ?x WHERE { ?x <iri>a</iri> ?y }"
        ) != self.wording_key("SELECT ?x WHERE { ?x <iri>a</iri> <iri>b</iri> }")
