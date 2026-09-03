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
    get_option_token_ids,
    merge_skeletons,
    search_alternatives,
    split_values,
    tokenize_option_answer,
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


# stand-in for a chat tokenizer: one token per role and per content word, with
# ids assigned per token string. word_boundary mimics sentencepiece, which marks
# a word start ("▁A"), so a letter's id differs from its standalone id; prefix
# mimics templates that open the assistant turn with extra tokens, e.g. Qwen3's
# empty <think> block.
class FakeTokenizer:
    unk_token_id = 0

    def __init__(
        self,
        word_boundary: bool = False,
        prefix: tuple[str, ...] = (),
    ) -> None:
        self.word_boundary = word_boundary
        self.prefix = prefix
        self.vocab: dict[str, int] = {"<unk>": self.unk_token_id}
        self.renders = 0
        # a sentencepiece vocab holds both variants, so the bare letter resolves
        # to an id that simply never shows up at a word start
        if word_boundary:
            for letter in "AB":
                self.id_for(letter)

    def id_for(self, token: str) -> int:
        return self.vocab.setdefault(token, len(self.vocab))

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.id_for(token)

    def word(self, token: str) -> str:
        return f"▁{token}" if self.word_boundary else token

    def apply_chat_template(
        self,
        messages: list[dict],
        return_dict: bool = False,
        add_generation_prompt: bool = False,
        enable_thinking: bool = False,
    ) -> dict | list[int]:
        self.renders += 1
        tokens = ["<s>"]
        for message in messages:
            tokens.append(message["role"])
            if message["role"] == "assistant":
                tokens.extend(self.prefix)
            tokens.extend(self.word(w) for w in message["content"].split())
        if add_generation_prompt:
            tokens.append("assistant")

        input_ids = [self.id_for(t) for t in tokens]
        if not return_dict:
            return input_ids
        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}


class TestOptionTokenIds:
    def messages(self, located: str) -> list[dict]:
        return [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": located},
        ]

    def test_byte_level_ids_match_standalone(self):
        tokenizer = FakeTokenizer()
        ids = get_option_token_ids(tokenizer, ("A", "B", "C"))
        assert list(ids) == [tokenizer.convert_tokens_to_ids(o) for o in "ABC"]

    def test_word_boundary_ids_come_from_the_render(self):
        tokenizer = FakeTokenizer(word_boundary=True)
        ids = get_option_token_ids(tokenizer, ("A", "B", "C"))
        assert list(ids) == [tokenizer.convert_tokens_to_ids(f"▁{o}") for o in "ABC"]
        # the bare letters are in the vocab but never rendered at a word start
        assert all(
            tokenizer.convert_tokens_to_ids(o) not in ids for o in "AB"
        )

    def test_tokens_before_the_answer_are_skipped(self):
        tokenizer = FakeTokenizer(prefix=("<think>", "</think>"))
        ids = get_option_token_ids(tokenizer, ("A", "B"))
        assert list(ids) == [tokenizer.convert_tokens_to_ids(o) for o in "AB"]

    def test_single_option(self):
        tokenizer = FakeTokenizer(word_boundary=True)
        assert get_option_token_ids(tokenizer, ("A",)) == (
            tokenizer.convert_tokens_to_ids("▁A"),
        )

    def test_cached_per_tokenizer_and_options(self):
        tokenizer = FakeTokenizer()
        get_option_token_ids(tokenizer, ("A", "B"))
        renders = tokenizer.renders
        get_option_token_ids(tokenizer, ("A", "B"))
        assert tokenizer.renders == renders
        # a different option set is a different key
        get_option_token_ids(tokenizer, ("A", "B", "C"))
        assert tokenizer.renders > renders

    def test_answer_located_with_word_boundary_tokenizer(self):
        tokenizer = FakeTokenizer(word_boundary=True)
        options = ["A", "B", "C"]
        output = tokenize_option_answer(self.messages("C"), options, tokenizer)
        located_id = tokenizer.convert_tokens_to_ids("▁C")
        assert output["input_ids"][output["answer_pos"]] == located_id
        assert output["option_token_ids"][options.index("C")] == located_id

    def test_answer_located_after_template_prefix(self):
        tokenizer = FakeTokenizer(prefix=("<think>", "</think>"))
        output = tokenize_option_answer(self.messages("B"), ["A", "B"], tokenizer)
        assert output["input_ids"][output["answer_pos"]] == (
            tokenizer.convert_tokens_to_ids("B")
        )
