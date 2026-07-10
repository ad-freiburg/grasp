########################################################################################
#  This code builds on cea.py from github.com/ad-freiburg/grasp/src/grasp/tasks/ but   #
#  is modified for general text entity linking instead of just table cell annotation.  #
########################################################################################

import re
import unicodedata
from typing import Any

from pydantic import BaseModel

from grasp.configs import GraspConfig
from grasp.examples import Sample
from grasp.functions import find_manager, parse_iri_or_literal
from grasp.manager import KgManager, format_kgs
from grasp.model import Message
from grasp.sparql.types import Alternative, ObjType
from grasp.tasks.base import FeedbackTask, GraspTask
from grasp.utils import (
    FunctionCallException,
    format_enumerate,
    format_list,
    format_notes,
    format_section,
)


class Annotation(BaseModel):
    identifier: str
    entity: str
    label: str | None = None
    aliases: list[str] | None = None
    infos: list[str] | None = None


class TextAnnotation(Annotation):
    start_index: int
    end_index: int


class Text(BaseModel):
    data: str
    annotate_from: int | None = None
    annotate_up_to: int | None = None
    special_instructions: str | None = None

    @property
    def length(self) -> int:
        return len(self.data)

    @property
    def start(self) -> int:
        return self.annotate_from if self.annotate_from else 0

    @property
    def end(self) -> int:
        return self.annotate_up_to if self.annotate_up_to else len(self.data)

    def trim(self, context: int | None = None) -> tuple["Text", int]:
        """
        Trims the Text to the start/end values if context is 0, trims the Text to
        start/end plus context otherwise. If context is None, does not trim the Text.
        """
        if context and context < 0:
            raise ValueError(f"context '{context}' must be non negative.")
        if self.start and (self.start >= self.length or self.start < 0):
            raise ValueError(
                f"annotate_from '{self.start}' must be less than length '{self.length}'"
                f" and greater than zero."
            )
        if self.end and (self.end <= 0 or (self.start and self.end <= self.start)):
            raise ValueError(
                f"annotate_up_to '{self.end}' must be greater than annotate_from "
                f"'{self.start}' and less than length '{self.length}'."
            )

        # without context the text is not trimmed
        if context is None:
            return self, 0

        # 4 variables: start/end of new context and start/end of new annotation window
        new_start, new_end = self.start, self.end
        window_start, window_end = 0, self.length

        if self.start > 0:
            new_start = max(0, self.start - context)
            window_start = self.start - new_start

        if self.end < self.length:
            new_end = min(self.length, self.end + context)
            window_end = self.end - new_start

        trimmed = Text(
            data=self.data[new_start:new_end],
            annotate_from=window_start,
            annotate_up_to=window_end,
        )
        return trimmed, new_start


class EntityLinkingSample(Sample):
    text: Text
    annotations: list[TextAnnotation]

    def input(self) -> Any:
        return self.text

    def queries(self) -> list[str]:
        annots = AnnotationState(self.text)
        return [annots.format()]


class AnnotationState:
    def __init__(
        self,
        text: Text,
        context: int | None = None,
    ) -> None:
        self.text, self.offset = text.trim(context)
        self.annotation_window: slice = slice(self.text.start, self.text.end)
        self.annotations: dict[tuple[int, int], Annotation] = {}

    def annotate(
        self,
        start_index: int,
        end_index: int,
        annotation: Annotation | None,
    ) -> Annotation | None:
        aws = self.annotation_window.stop - self.annotation_window.start
        if start_index < 0 or start_index >= aws:
            raise ValueError(f"Start_index {start_index} out of bounds")
        if end_index <= start_index or end_index > aws:
            raise ValueError(f"End_index {end_index} out of bounds")
        start_index += self.annotation_window.start
        end_index += self.annotation_window.start
        current = self.annotations.pop((start_index, end_index), None)
        if annotation is not None:
            self.annotations[(start_index, end_index)] = annotation
        return current

    def get(self, start_index: int, end_index: int) -> Annotation | None:
        return self.annotations.get((start_index, end_index), None)

    def to_dict(self) -> dict:
        return {
            "formatted": self.format(),
            "predictions": [
                {
                    "entity_reference": a.entity,
                    "start_char": s + self.offset,
                    "end_char": e + self.offset,
                }
                for (s, e), a in self.annotations.items()
            ],
        }

    def format(
        self, only_current_window: bool = False, list_entities: bool = True
    ) -> str:
        """
        Returns a string with the current annotation state of the text.
        Annotations are visualized in the following format: '[annotated words](q123)',
        '[[Nested [annotations](q123)](q456) are supported](q789)'.
        If only_current_window is true, only the text of the current annotation window
        is shown. If list_entities is true, the used entities are listed at the end.'
        """
        result = self.text.data
        # item[0] is (start, end), we sort by end first, then by negative start
        sorted_annotations = sorted(
            self.annotations.items(), key=lambda item: (item[0][1], -item[0][0])
        )
        # go through annotations from highest end index first
        nested_list = []
        while sorted_annotations:
            ann = sorted_annotations.pop()
            start_idx = ann[0][0]
            end_idx = ann[0][1]
            start_offset = 0
            end_offset = 0
            for i in range(len(nested_list) - 1, -1, -1):
                # start of other annotation after end of current one -> unimportant
                if nested_list[i] >= end_idx:
                    nested_list.pop(i)
                # start of other annotation before current end but not current start
                elif nested_list[i] < end_idx and nested_list[i] >= start_idx:
                    end_offset += 1
                # we don't need to see the rest of the list
                elif nested_list[i] < start_idx:
                    start_offset += 1
                    end_offset += 1
            # prepend current start to the nested list
            nested_list = [start_idx] + nested_list

            if only_current_window and (
                start_idx < self.annotation_window.start
                or end_idx > self.annotation_window.stop
            ):
                continue

            prefix = result[: start_idx + start_offset]
            words = result[start_idx + start_offset : end_idx + end_offset]
            suffix = result[end_idx + end_offset :]
            result = prefix + "[" + words + "](" + ann[1].entity + ")" + suffix

        # trim to only show the current window
        if only_current_window:
            added_length = len(result) - self.text.length
            result = result[
                self.annotation_window.start : self.annotation_window.stop
                + added_length
            ]

        if list_entities:
            entities: dict[str, Alternative] = {}
            for annot in self.annotations.values():
                if annot.identifier in entities:
                    continue

                alternative = Alternative(
                    annot.identifier,
                    short_identifier=annot.entity,
                    label=annot.label,
                    aliases=annot.aliases,
                    info=annot.infos,
                )
                entities[annot.identifier] = alternative

            if entities:
                annotations = format_list(
                    alt.get_selection_string() for _, alt in sorted(entities.items())
                )
                result += f"\n\nAnnotated entities:\n{annotations}"

        return result


def rules() -> list[str]:
    return [
        (
            "If you cannot find any suitable entity mention in the text excerpt, "
            "leave the excerpt unannotated and just finalize."
        ),
        (
            "If an entity is cut off at the end of the text or beginning of the text, "
            "don't annotate it. Use the context to see if it is cut off or not."
        ),
        (
            "If there are multiple suitable entities for a number of words, choose "
            "the one that fits best in the context of the text and is more general."
        ),
        (
            "Annotate every occurence of an entity you find in the excerpt even if it "
            "is mentioned multiple times. If it occurs again, annotate it again."
        ),
        (
            "If you recognize an entity but cannot find it in the knowledge graph, "
            "annotate it as null."
        ),
        (
            "Do not link coreferences that do not contain at least part of the name "
            "but **do** link entity mentions that contain at least a part of a name."
        ),
        "If the user specifies additional instructions follow those instructions.",
    ]


def system_information() -> str:
    return """\
You are an entity linking system that does entity recognition and entity disambiguation according to the provided rules and instructions.
Your task is to annotate words in a given text excerpt with entities from the available knowledge graphs.

You need to **exactly follow these step-by-step instructions** to annotate the text:
1. Find entity mentions in the text excerpt by going through it from the begining word by word and determine if it constitutes an entity according to the rules.
2. Determine what the text might be about and think about how the entity mentions might be represented with entities in the knowledge graph(s). 
3. Use the provided functions to search and explore the knowledge graph(s) to find the entities.
4. Use the annotate function to annotate every entity mention.
5. When you are certain that the annotation of the current excerpt is correct and complete use the finalize function."""


def functions(managers: list[KgManager]) -> list[dict]:
    kgs = [manager.kg for manager in managers]
    fns = [
        {
            "name": "annotate",
            "description": """\
Annotate a word or a sequence of words with an entity from the specified knowledge \
graph by writing the exact words to be annotated as 'words_to_be_annotated'.
Specify the words further by the occurrence_index, if the words only occur once, set it to 0,
if you only want to annotate the second occurence, just set it to 1.
Careful, sometimes a word you want to annotate can be a substring of another word earlier in the text excerpt, \
so always keep that in mind and adjust the occurrence_index accordingly.
This function overwrites any previous annotation of the words.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "kg": {
                        "type": "string",
                        "enum": kgs,
                        "description": "The knowledge graph to use for the annotation.",
                    },
                    "words_to_be_annotated": {
                        "type": "string",
                        "description": "The exact words to be annotated written exactly like in the original text.",
                    },
                    "occurrence_index": {
                        "type": "integer",
                        "description": "Index of the occurrence of the words in the text.",
                    },
                    "entity": {
                        "type": ["string", "null"],
                        "description": "The IRI of the entity to annotate the words with, or null if unknown.",
                    },
                },
                "required": [
                    "kg",
                    "words_to_be_annotated",
                    "occurrence_index",
                    "entity",
                ],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "name": "delete_annotation",
            "description": """\
Delete the annotation of a word or a sequence of words \
by writing the exact words whose annotation should be deleted as 'words_to_be_annotated'.
Specify the words further by the occurrence_index, if the words only occur once, set it to 0,
if you only want to delete the annotation of the second occurence, just set it to 1.
Careful, sometimes an annotation you want to delete can be a substring of another word earlier in the text excerpt, \
so always keep that in mind and adjust the occurrence_index accordingly.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "words_to_be_annotated": {
                        "type": "string",
                        "description": "The exact words whose annotation should be deleted, written exactly like in the original text.",
                    },
                    "occurrence_index": {
                        "type": "integer",
                        "description": "Index of the occurrence of the words in the text.",
                    },
                },
                "required": ["words_to_be_annotated", "occurrence_index"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "name": "show_current_annotations",
            "description": "Show the current annotation state of the excerpt of the text to annotate.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "name": "finalize",
            "description": "Finalize your annotations in the given excerpt and stop the annotation process.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            "strict": True,
        },
    ]
    return fns


def prepare_annotation(manager: KgManager, entity: str) -> Annotation:
    binding = parse_iri_or_literal(entity, manager.iri_literal_parser, manager.prefixes)
    if binding is None or binding.typ != "uri":
        raise ValueError(f"Entity {entity} is not a valid IRI")

    identifier = binding.identifier()

    norm = manager.normalize(identifier, ObjType.ENTITY.index_name)
    if norm is not None:
        identifier, _ = norm

    infos = manager.get_info_for_identifiers_from_index(
        [identifier], ObjType.ENTITY.index_name
    )

    # format normalized identifier again, so always
    # prefixed form is shown if available
    entity = manager.format_iri(identifier)
    # extract fields from info dict
    info = infos.get(identifier, {})
    label = info.get("label")
    aliases = info.get("alias", [])
    infos = info.get("info", [])

    return Annotation(
        identifier=identifier,
        entity=entity,
        label=label,
        aliases=aliases,
        infos=infos,
    )


def annotate(
    managers: list[KgManager],
    kg: str,
    words_to_be_annotated: str,
    occurrence_index: int,
    entity: str | None,
    state: AnnotationState,
    known: set[str],
    know_before_annotate: bool = False,
    show_state_after_annotation: bool = True,
) -> str:
    """
    A function for the llm to call to annotate the words_to_be_annotated in the text
    with the entity and knowledge graph. The occurrence_index helps to distinguish
    between different occurrences of the words in the text excerpt.
    """
    manager, _ = find_manager(managers, kg)
    sequence = state.text.data[state.annotation_window]

    # normalizing, because some llms are heavily biased towards specific characters like
    # the ascii apostrophe although they are technically able to output the correct one.
    def normalize(string: str) -> str:
        return unicodedata.normalize("NFC", string).replace("‘", "'").replace("’", "'")

    words_to_be_annotated = normalize(words_to_be_annotated)
    sequence = normalize(sequence)

    word_matches = [
        m.span() for m in re.finditer(re.escape(words_to_be_annotated), sequence)
    ]

    if not word_matches:
        raise ValueError(
            f"No match found for the given words to be annotated "
            f"'{words_to_be_annotated}' in the current annotation window."
            "(Did you use the correct characters when specifying the words?)"
        )

    if occurrence_index < 0:
        raise ValueError(f"occurrence_index '{occurrence_index}' must be non negative.")

    if occurrence_index >= len(word_matches):
        raise ValueError(
            f"occurrence_index '{occurrence_index}' must be less than "
            f"number of matches: {len(word_matches)}."
        )

    start_idx, end_idx = word_matches[occurrence_index]

    try:
        if entity is None:
            annotation = Annotation(identifier="<NIL>", entity="<NIL>")
        else:
            annotation = prepare_annotation(manager, entity)
            if know_before_annotate and annotation.identifier not in known:
                raise FunctionCallException(
                    f"The entity {entity} cannot be used for annotation "
                    "without being known from previous function call results. "
                    "This does not mean it is invalid, but you should verify "
                    "that it indeed exists (e.g., by listing example triples) "
                    "in the knowledge graphs first."
                )

        current = state.annotate(start_idx, end_idx, annotation)

    except ValueError as e:
        raise FunctionCallException(str(e)) from e

    if current is None:
        result = (
            f"Annotated text sequence [{start_idx}: {end_idx}] "
            f"'{sequence[start_idx:end_idx]}' with entity '{entity}'."
        )
    else:
        result = (
            f"Updated annotation of text sequence [{start_idx}, {end_idx}] "
            f"'{sequence[start_idx:end_idx]}' from '{current.entity}' to '{entity}'."
        )
    if show_state_after_annotation:
        result += (
            "\n\nThe current annotation state of the text excerpt is the following:\n"
            f"{state.format(only_current_window=True, list_entities=False)}"
        )
    return result


def delete_annotation(
    words_to_be_annotated: str,
    occurrence_index: int,
    state: AnnotationState,
    show_state_after_annotation=True,
) -> str:
    """
    A function for the llm to call to delete the annotation of the
    words_to_be_annotated in the text. The occurrence_index helps to
    distinguish between different occurrences of the words in the text.
    """
    sequence = state.text.data[state.annotation_window]

    # normalizing, because some llms are heavily biased towards specific characters like
    # the ascii apostrophe although they are technically able to output the correct one.
    def normalize(string: str) -> str:
        return unicodedata.normalize("NFC", string).replace("‘", "'").replace("’", "'")

    words_to_be_annotated = normalize(words_to_be_annotated)
    sequence = normalize(sequence)

    word_matches = [
        m.span() for m in re.finditer(re.escape(words_to_be_annotated), sequence)
    ]

    if not word_matches:
        raise ValueError(
            f"No match found for the given words_to_be_annotated "
            f"'{words_to_be_annotated}' in the current annotation window."
            "(Did you use the correct characters when specifying the words?)"
        )

    if occurrence_index < 0:
        raise ValueError(f"occurrence_index '{occurrence_index}' must be non negative.")

    if occurrence_index >= len(word_matches):
        raise ValueError(
            f"occurrence_index '{occurrence_index}' must be less than "
            f"number of matches: {len(word_matches)}."
        )

    start_idx, end_idx = word_matches[occurrence_index]

    # deleting an annotation
    try:
        current = state.annotate(start_idx, end_idx, None)
    except ValueError as e:
        raise FunctionCallException(str(e)) from e

    if current is None:
        raise FunctionCallException(
            f"Text sequence [{start_idx}, {end_idx}] '{sequence[start_idx:end_idx]}' "
            "is not annotated so there is no annotation to delete."
        )

    result = (
        f"Deleted annotation '{current.entity}' from text sequence "
        f"[{start_idx}, {end_idx}] '{sequence[start_idx:end_idx]}'."
    )
    if show_state_after_annotation:
        result += (
            "\n\nThe current annotation state of the text excerpt is the following:\n"
            f"{state.format(only_current_window=True, list_entities=False)}"
        )
    return result


def input_instructions(
    state: AnnotationState, special_instructions: str | None = None
) -> str:
    user_input = (
        "This is the full text only for context to better understand entities "
        "that are not clear from the excerpt alone.\n\n"
        "=== START FULL TEXT FOR CONTEXT ===\n"
        f"{state.format(only_current_window=False)}\n"
        "=== END FULL TEXT FOR CONTEXT ===\n\n"
        "The following is the excerpt of the text that you need to annotate:\n\n"
        "=== START TEXT EXCERPT TO ANNOTATE ===\n"
        f"{state.format(only_current_window=True)}\n"
        "=== END TEXT EXCERPT TO ANNOTATE ===\n"
    )
    if special_instructions:
        user_input = (
            "These are additional instructions that you need to follow.\n\n"
            "=== START ADDITIONAL INSTRUCTIONS ===\n"
            f"{special_instructions}\n"
            "=== END ADDITIONAL INSTRUCTIONS ===\n\n"
        ) + user_input
    return user_input


def input_and_state(
    input: Any,
    config: GraspConfig,
) -> tuple[str, AnnotationState]:
    try:
        text = Text(**input)
    except Exception as e:
        raise ValueError(
            "Entity Linking task input must be a dict with a 'data' and optional "
            "'annotate_from', 'annotate_up_to', and 'special_instructions' fields."
        ) from e

    el_kwargs = config.task_kwargs.get("entity-linking", {})
    annots = AnnotationState(text, context=el_kwargs.get("context"))
    instructions = input_instructions(annots, text.special_instructions)

    return instructions, annots


def call_function(
    config: GraspConfig,
    managers: list[KgManager],
    fn_name: str,
    fn_args: dict,
    known: set[str],
    state: AnnotationState | None = None,
    example_indices: dict | None = None,
) -> str | None:
    assert isinstance(state, AnnotationState), (
        "Annotations must be provided as state for entity linking task"
    )
    assert not example_indices, (
        "Example indices are not supported for entity linking task"
    )

    el_kwargs = config.task_kwargs.get("entity-linking", {})
    know_before_annotate = el_kwargs.get("know_before_annotate", True)
    show_state_after_annotate = el_kwargs.get("show_state_after_annotate", True)

    if fn_name == "annotate":
        return annotate(
            managers,
            fn_args["kg"],
            fn_args["words_to_be_annotated"],
            fn_args["occurrence_index"],
            fn_args["entity"],
            state,
            known,
            know_before_annotate,
            show_state_after_annotate,
        )

    elif fn_name == "delete_annotation":
        return delete_annotation(
            fn_args["words_to_be_annotated"],
            fn_args["occurrence_index"],
            state,
            show_state_after_annotate,
        )

    elif fn_name == "show_current_annotations":
        return state.format(only_current_window=True, list_entities=True)

    elif fn_name == "finalize":
        return "Finalized the annotation process."

    else:
        raise ValueError(f"Unknown function '{fn_name}'")


def feedback_system_message(
    managers: list[KgManager],
    kg_notes: dict[str, list[str]],
    notes: list[str],
) -> str:
    return "\n\n".join(
        [
            "You are a text annotation assistant providing feedback on the "
            "output of a text annotation system for a given input text.",
            format_section(
                "Available knowledge graphs",
                format_kgs(managers, kg_notes),
            ),
            format_section(
                "General notes across knowledge graphs",
                format_notes(notes, enumerated=True),
            ),
            format_section(
                "Rules to follow",
                format_enumerate(rules()) if rules() else "None",
            ),
            "Provide your feedback with the give_feedback function.",
        ]
    )


def feedback_instructions(inputs: list[str], output: dict) -> str:
    assert inputs, "At least one input is required for feedback"

    sections = []
    if len(inputs) > 1:
        sections.append(
            format_section(
                "Previous inputs",
                "\n\n".join(i.strip() for i in inputs[:-1]),
            )
        )

    sections.append(format_section("Input", inputs[-1].strip()))
    sections.append(format_section("Annotations", output["formatted"]))
    return "\n\n".join(sections)


class EntityLinkingTask(GraspTask, FeedbackTask):
    name = "entity-linking"

    def system_information(self) -> str:
        return system_information()

    def rules(self) -> list[str]:
        return rules()

    def function_definitions(self) -> list[dict]:
        return functions(self.managers)

    def call_function(
        self,
        fn_name: str,
        fn_args: dict,
        known: set[str],
        example_indices: dict | None = None,
    ) -> str:
        return call_function(
            self.config,
            self.managers,
            fn_name,
            fn_args,
            known,
            self.state,
            example_indices,
        )

    def done(self, fn_name: str) -> bool:
        return fn_name == "finalize"

    def setup(self, input: Any) -> str:
        instructions, self.state = input_and_state(input, self.config)
        return instructions

    def output(self, messages: list[Message]) -> dict:
        return self.state.to_dict()

    @property
    def default_input_field(self) -> str | None:
        return "text"

    @classmethod
    def sample_cls(cls) -> type[EntityLinkingSample] | None:
        return EntityLinkingSample

    def feedback_system_message(
        self, kg_notes: dict[str, list[str]], notes: list[str]
    ) -> str:
        return feedback_system_message(self.managers, kg_notes, notes)

    def feedback_instructions(self, inputs: list[str], output: dict) -> str:
        return feedback_instructions(inputs, output)

