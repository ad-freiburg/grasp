from unittest.mock import Mock

import pytest

import grasp.baselines.grisp.data as data_module
from grasp.baselines.grisp.data import (
    BOI,
    BOR,
    EOI,
    EOR,
    MAX_PLACEHOLDER_QUERIES,
    SEP,
    IRI,
    GRISPSample,
    Info,
    Skeleton,
    display_nl_iri,
    extract_queries_and_variants_from_nl_iri,
    IGNORE_INDEX,
    extract_values_from_nl_iri,
    format_alternatives,
    load_grisp_manager,
    get_option_token_ids,
    is_placeholder,
    is_valid_item,
    merge_skeletons,
    ordered_alternatives_with_interleave,
    search_alternatives,
    searched_at_position,
    selectable_obj_types,
    split_values,
    tokenize_messages,
    tokenize_option_answer,
)
from grasp.baselines.grisp.utils import load_sparql_parser
from grasp.sparql.item import extract_sparql_items
from grasp.sparql.types import Alternative, ObjType, Position, Selection
from grasp.sparql.utils import load_iri_and_literal_parser

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
            sparql="",
            sparqls=[],
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
            sparql="",
            sparqls=[],
            queries=["population", "inhabitants"],
            variants=["wdt", None],
            values=["population (wdt)", "inhabitants"],
        )
        queries = info.build_queries(OBJ_TYPES)
        assert info.num_queries == 2
        assert all(len(pairs) == info.num_queries for pairs in queries.values())


class TestBeamCoherentRendering:
    # two beams that disagree on both placeholders, so wording 1 of one comes
    # from the same beam as wording 1 of the other
    def merged(self, fill_order="left-to-right"):
        base = parse("SELECT ?x WHERE { <iri>Harz</iri> <iri>has part</iri> ?x }")
        base.beams = [[{0}] for _ in base.nl_iris]
        other = parse(
            "SELECT ?x WHERE { <iri>Harz mountains</iri> <iri>contains</iri> ?x }"
        )
        other.beams = [[{1}] for _ in other.nl_iris]
        merged = merge_skeletons(base, other, SPARQL_PARSER, fill_order)
        assert merged is not None
        return merged

    def test_provenance_survives_the_merge(self):
        assert self.merged().beams == [[{0}, {1}], [{0}, {1}]]

    def test_unresolved_placeholders_follow_the_marked_wording_s_beam(self):
        info = self.merged().prepare_for_selection()
        # marking beam 0's wording shows beam 0's wording for the other one too
        assert info.sparql_for_query(0) == (
            f"SELECT ?x WHERE {{ {BOR}Harz{EOR} {BOI}has part{EOI} ?x }}"
        )
        # and marking beam 1's switches the other one over with it, instead of
        # mixing beam 1's "Harz mountains" with beam 0's "has part"
        assert info.sparql_for_query(1) == (
            f"SELECT ?x WHERE {{ {BOR}Harz mountains{EOR} {BOI}contains{EOI} ?x }}"
        )

    def test_resolved_placeholders_show_their_identifier_not_a_wording(self):
        merged = self.merged()
        manager = Mock()
        manager.denormalize.return_value = "wd:Q4066"
        manager.format_iri.return_value = "wd:Q4066"
        merged.add_selection(
            Selection(Alternative("wd:Q4066", label="Harz"), ObjType.ENTITY),
            manager,
        )
        info = merged.prepare_for_selection()
        # the entity is resolved, so only the property is still a wording, and
        # it follows the beam of the wording now being marked
        assert info.sparql_for_query(1) == (
            f"SELECT ?x WHERE {{ wd:Q4066 {BOR}contains{EOR} ?x }}"
        )

    def test_without_provenance_everything_falls_back_to_the_first_wording(self):
        base = parse("SELECT ?x WHERE { <iri>Harz</iri> <iri>has part</iri> ?x }")
        other = parse(
            "SELECT ?x WHERE { <iri>Harz mountains</iri> <iri>contains</iri> ?x }"
        )
        merged = merge_skeletons(base, other, SPARQL_PARSER)
        assert merged is not None
        assert merged.beams is None

        info = merged.prepare_for_selection()
        # the marked placeholder still follows the searched wording, the others
        # cannot and stay on their first
        assert info.sparql_for_query(1) == (
            f"SELECT ?x WHERE {{ {BOR}Harz mountains{EOR} {BOI}has part{EOI} ?x }}"
        )

    def test_a_beam_truncated_out_falls_back_to_the_first_wording(self):
        merged = self.merged()
        # as if merging had dropped beam 1's wording for the second placeholder
        merged.beams = [[{0}, {1}], [{0}]]
        info = merged.prepare_for_selection()
        assert info.sparql_for_query(1) == (
            f"SELECT ?x WHERE {{ {BOR}Harz mountains{EOR} {BOI}has part{EOI} ?x }}"
        )

    def test_beam_choice_prefers_coverage_of_the_unresolved_placeholders(self):
        merged = self.merged()
        # wording 0 of the marked placeholder came from beams 0 and 2, but only
        # beam 2 still has a wording for the other placeholder
        merged.beams = [[{0, 2}, {1}], [{2}, {1}]]
        info = merged.prepare_for_selection()
        assert info.sparql_for_query(0) == (
            f"SELECT ?x WHERE {{ {BOR}Harz{EOR} {BOI}has part{EOI} ?x }}"
        )


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


# entity index holds e1 and the shared c1, class index holds c1 and c2;
# identifiers are not normalized
def class_manager() -> Mock:
    indices = {"entities": {"e1", "c1"}, "classes": {"c1", "c2"}}
    manager = Mock()
    manager.index_names = ["classes", "entities", "properties"]
    manager.normalize.side_effect = lambda identifier, index_name: (identifier, None)
    manager.denormalize.side_effect = lambda identifier, index_name, *args: identifier
    manager.check_identifier.side_effect = (
        lambda identifier, index_name: identifier in indices.get(index_name, set())
    )
    return manager


class TestClasses:
    def test_classes_only_with_class_index(self):
        manager = Mock()
        manager.index_names = ["entities", "properties"]
        assert selectable_obj_types(manager) == [ObjType.ENTITY, ObjType.PROPERTY]
        manager.index_names = ["classes", "entities", "properties"]
        assert selectable_obj_types(manager) == [
            ObjType.ENTITY,
            ObjType.CLASS,
            ObjType.PROPERTY,
        ]

    def test_entities_and_classes_searched_separately(self):
        manager = class_manager()
        results = {
            "entities": ["e1", "c1"],
            "classes": ["c1", "c2"],
        }
        manager.search_index.side_effect = lambda index_name, *args: [
            Alternative(identifier=identifier) for identifier in results[index_name]
        ]
        queries = {
            ObjType.ENTITY: [("city", None)],
            ObjType.CLASS: [("city", None)],
        }
        groups = search_alternatives(
            manager,
            [ObjType.ENTITY, ObjType.CLASS],
            {ObjType.ENTITY: None, ObjType.CLASS: None},
            queries,
            0,
            10,
            Mock(),
        )
        assert [a.identifier for a in groups[ObjType.ENTITY]] == ["e1", "c1"]
        assert [a.identifier for a in groups[ObjType.CLASS]] == ["c1", "c2"]

    def test_entities_and_classes_one_group(self):
        groups = {
            ObjType.PROPERTY: [Alternative(identifier="p1")],
            ObjType.ENTITY: [Alternative(identifier=i) for i in ["e1", "e2"]],
            ObjType.CLASS: [Alternative(identifier="c1")],
        }
        queries = {obj_type: [("x", None)] for obj_type in groups}

        def order(interleave: bool) -> list[str]:
            return [
                alternative.identifier
                for alternative, _, _ in ordered_alternatives_with_interleave(
                    groups,
                    queries,
                    interleave=interleave,
                )
            ]

        assert order(False) == ["e1", "c1", "e2", "p1"]
        assert order(True) == ["e1", "p1", "c1", "e2"]

    def test_entities_and_classes_merged_by_score(self):
        def alternatives(scores: dict[str, float]) -> list[Alternative]:
            return [Alternative(identifier=i, score=s) for i, s in scores.items()]

        groups = {
            ObjType.ENTITY: alternatives({"e1": 0.5, "e2": 0.5}),
            ObjType.CLASS: alternatives({"c1": 1.0, "c2": 0.5}),
        }
        queries = {obj_type: [("x", None)] for obj_type in groups}
        ordered = ordered_alternatives_with_interleave(groups, queries)
        # ties keep the interleaved rank order
        assert [a.identifier for a, _, _ in ordered] == ["c1", "e1", "e2", "c2"]

    def test_index_types_must_match(self, monkeypatch):
        def manager(entity_type: str, class_type: str) -> Mock:
            manager = Mock()
            manager.kg = "kg"
            manager.indices = {
                "entities": Mock(index=Mock(index_type=entity_type)),
                "classes": Mock(index=Mock(index_type=class_type)),
            }
            return manager

        monkeypatch.setattr(
            data_module, "load_kg_manager", lambda config: manager("fuzzy", "fuzzy")
        )
        load_grisp_manager(Mock())
        monkeypatch.setattr(
            data_module, "load_kg_manager", lambda config: manager("fuzzy", "embedding")
        )
        with pytest.raises(ValueError, match="same type"):
            load_grisp_manager(Mock())

    def test_identifier_in_both_groups_is_a_class(self):
        groups = {
            ObjType.ENTITY: [Alternative(identifier=i) for i in ["x", "e1"]],
            ObjType.CLASS: [Alternative(identifier=i) for i in ["c1", "x"]],
        }
        queries = {obj_type: [("x", None)] for obj_type in groups}
        ordered = ordered_alternatives_with_interleave(groups, queries)
        assert [(a.identifier, obj_type) for a, obj_type, _ in ordered] == [
            ("x", ObjType.CLASS),
            ("c1", ObjType.CLASS),
            ("e1", ObjType.ENTITY),
        ]

    def test_classes_shown_with_entities(self):
        alternatives = [
            (Alternative(identifier="e1"), ObjType.ENTITY, None),
            (Alternative(identifier="c1"), ObjType.CLASS, None),
            (Alternative(identifier="p1"), ObjType.PROPERTY, None),
        ]
        formatted = format_alternatives(alternatives)
        assert "Entity alternatives" in formatted
        assert "Property alternatives" in formatted
        assert "Class alternatives" not in formatted


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
        generation_keyword: bool = False,
    ) -> None:
        self.word_boundary = word_boundary
        self.prefix = prefix
        self.chat_template = "{% generation %}" if generation_keyword else "{{ x }}"
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
        return_assistant_tokens_mask: bool = False,
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
        enc = {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}
        if return_assistant_tokens_mask:
            # the assistant turn is everything after the last role token
            start = len(tokens) - len(tokens[::-1][: tokens[::-1].index("assistant")])
            enc["assistant_masks"] = [
                1 if i >= start else 0 for i in range(len(tokens))
            ]
        return enc


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
        assert all(tokenizer.convert_tokens_to_ids(o) not in ids for o in "AB")

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


class TestMessageMasking:
    def messages(self) -> list[dict]:
        return [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "the user question"},
            {"role": "assistant", "content": "the answer"},
        ]

    def labelled(self, tokenizer: FakeTokenizer, mask_inputs: bool) -> tuple[int, int]:
        out = tokenize_messages(self.messages(), tokenizer, mask_inputs)
        labels = out["labels"]
        return sum(1 for x in labels if x != IGNORE_INDEX), len(labels)

    def test_only_the_assistant_turn_is_labelled_without_generation_keyword(self):
        # templates lacking {% generation %} take the manual-label fallback
        tokenizer = FakeTokenizer()
        assert "{% generation %}" not in tokenizer.chat_template
        n_labelled, n_tokens = self.labelled(tokenizer, mask_inputs=True)
        # "the answer" is two content words, everything before it is prompt
        assert n_labelled == 2
        assert n_labelled < n_tokens

    def test_only_the_assistant_turn_is_labelled_with_generation_keyword(self):
        tokenizer = FakeTokenizer(generation_keyword=True)
        assert "{% generation %}" in tokenizer.chat_template
        n_labelled, _ = self.labelled(tokenizer, mask_inputs=True)
        assert n_labelled == 2

    def test_both_masking_paths_agree(self):
        without = self.labelled(FakeTokenizer(), mask_inputs=True)
        with_keyword = self.labelled(
            FakeTokenizer(generation_keyword=True), mask_inputs=True
        )
        assert without == with_keyword

    def test_mask_inputs_false_labels_everything(self):
        tokenizer = FakeTokenizer()
        n_labelled, n_tokens = self.labelled(tokenizer, mask_inputs=False)
        assert n_labelled == n_tokens


class PropertyOnlyManager:
    sparql_parser = SPARQL_PARSER
    iri_literal_parser = load_iri_and_literal_parser()
    prefixes = {}

    def normalize(self, iri: str, index_name: str):
        return iri, None

    def check_identifier(self, identifier: str, index_name: str) -> bool:
        return index_name == "properties" and identifier == "http://ex.org/p"

    def get_info_for_identifiers_from_index(self, identifiers, index_name):
        return {}

    def build_alternative_with_info(self, identifier, info=None, variants=None):
        return Alternative(identifier, variants=variants)


def test_properties_only_searched_at_predicate_position():
    sparql = (
        "SELECT ?s WHERE { ?s <http://ex.org/p> ?o . ?x ?y <http://ex.org/p> . "
        "?x ?y <http://other.org/x> }"
    )
    _, items = extract_sparql_items(sparql, PropertyOnlyManager())  # type: ignore
    assert [(item.obj_type, item.position) for item in items] == [
        (ObjType.PROPERTY, Position.PROPERTY),
        (ObjType.PROPERTY, Position.OBJECT),
        (ObjType.UNKNOWN, Position.OBJECT),
    ]
    # unknown items are never searched, they are only valid if allowed
    assert [searched_at_position(item) for item in items] == [True, False, False]
    assert [is_valid_item(item) for item in items] == [True, False, False]
    assert [is_valid_item(item, allow_unknown=True) for item in items] == [
        True,
        False,
        True,
    ]


class UnindexedManager(PropertyOnlyManager):
    prefixes = {"ex": "http://ex.org/"}

    def denormalize(self, identifier: str, index_name: str, variant=None):
        return f"denormalized:{identifier}"

    def format_iri(self, iri: str) -> str:
        return iri


def test_unindexed_iris_are_placeholders_but_not_selectable():
    manager = UnindexedManager()
    sparql = "SELECT ?s WHERE { ?s <http://ex.org/p> <http://ex.org/q> }"
    _, items = extract_sparql_items(sparql, manager)  # type: ignore
    assert [item.obj_type for item in items] == [ObjType.PROPERTY, ObjType.UNINDEXED]
    assert all(is_placeholder(item) for item in items)

    iris = [IRI.from_item(item, manager) for item in items]  # type: ignore
    assert [(iri.identifier, iri.selectable) for iri in iris] == [
        ("denormalized:http://ex.org/p", True),
        ("http://ex.org/q", False),
    ]

    def sample(*parts) -> GRISPSample:
        return GRISPSample(kg="kg", questions=["q"], sparql=list(parts))

    assert sample("?s ", iris[0], " ", iris[1]).has_selection_targets
    assert not sample("?s ?p ", iris[1]).has_selection_targets
