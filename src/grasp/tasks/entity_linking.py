##########################################################################
#  The code is in large parts taken from the cea.py of the ad-freiburg   #
#  git repo of grasp but modified for general text entity linking.       #
##########################################################################

import re
from typing import Any, Iterator, Optional

from pydantic import BaseModel

from grasp.configs import GraspConfig
from grasp.functions import find_manager
from grasp.manager import KgManager, format_kgs
from grasp.sparql.types import Alternative, ObjType
from grasp.sparql.utils import parse_into_binding
from grasp.tasks.examples import Sample
from grasp.utils import FunctionCallException, format_list, format_notes


class Annotation(BaseModel):
    identifier: str
    entity: str
    label: str | None = None
    synonyms: list[str] | None = None
    infos: list[str] | None = None


class TextAnnotation(Annotation):
    start_index: int
    end_index: int


class EntityLinkingSample(Sample):
    text: str
    annotations: list[TextAnnotation]

    def input(self) -> Any:
        return self.text.model_dump()

    def queries(self) -> list[str]:
        annots = AnnotationState(self.text)
        return [annots.format()]


class AnnotationState:
    def __init__(
        self,
        text: str,
        method: str = "prefix",
    ) -> None:

        self.text = text

        self.max_annotation_window_size = 300

        self.annotation_window: slice = slice(0, 0)
        self.annotations: dict[tuple[int, int], Annotation] = {}
        self.word_indices: dict[int, tuple[int, int]] = {}

        self.method = method
        self.unviewed = [(0, len(self.text))]

    def show_unviewed_ranges(self):
        return str(self.unviewed)

    def change_annotation_window(self, start, end):

        result = ""
        if start < 0:
            return (f"Your start index {start} needs to be non negative.")
        if start >= len(self.text) - 1:
            return (f"Your start index {start} is beyond the end of the text.")
        if end <= start:
            return (f"Your end index {end} is before your start index {start}.")
        if end > start + self.max_annotation_window_size:
            result += f"Your annotation window size exceeds the maximum allowed annotation window size of {self.max_annotation_window_size}, it was adjusted automatically."
            end = start + self.max_annotation_window_size
        new_start_idx = start
        new_end_idx = min(end, len(self.text))

        unviewed_length = len(self.unviewed)
        i = 0
        while i < unviewed_length:
            # range start in current window
            if self.unviewed[i][0] >= new_start_idx and self.unviewed[i][0] < new_end_idx:

                # range end in current window -> remove it
                if self.unviewed[i][1] <= new_end_idx:
                    self.unviewed.pop(i)
                    i -= 1
                    unviewed_length -= 1
                else:
                    self.unviewed[i] = (new_end_idx, self.unviewed[i][1])

            # range start before current window
            elif self.unviewed[i][0] < new_start_idx:

                # range end in current window -> shorten it
                if self.unviewed[i][1] >= new_start_idx and self.unviewed[i][1] <= new_end_idx:
                    self.unviewed[i] = (self.unviewed[i][0], new_start_idx)

                # range end after current window -> split it
                elif self.unviewed[i][1] > new_end_idx:
                    insert_idx = self.unviewed.find(self.unviewed[i]) + 1
                    self.unviewed.insert(insert_idx, (new_end_idx, self.unviewed[i][1]))
                    self.unviewed[i] = (self.unviewed[i][0], new_start_idx)
                    i += 1
                    unviewed_length += 1
            i += 1

            

        self.annotation_window = slice(new_start_idx, new_end_idx)
        current_excerpt = self.text[self.annotation_window]
        result += (f"The next text sequence of the full Text of lenght {len(self.text)} "
                   f"to annotate is the excerpt [{new_start_idx}: {new_end_idx}]:\n")

        if self.method == "indices":
            currently_at_word = False
            self.word_indices = {}
            idx = 0
            for i, c in enumerate(current_excerpt):
                if c != " " and c != "." and not currently_at_word:
                    currently_at_word = True
                    s = i
                    result += c
                elif (c == " " or c == ".") and not currently_at_word:
                    result += c
                elif c != " " and c != "." and currently_at_word:
                    result += c
                elif (c == " " or c == ".") and currently_at_word:
                    currently_at_word = False
                    e = i
                    result += f"({idx}){c}"
                    self.word_indices[idx] = (s, e)
                    idx += 1
        else:
            result += current_excerpt

        return result


    def annotate(
        self,
        start_index: int,
        end_index: int,
        annotation: Annotation | None,
        annotate_whole_words: bool = False,
    ) -> Annotation | None:
        if annotate_whole_words:
            if start_index not in self.word_indices:
                raise ValueError(f"Start_index {start_index} not a valid word index")
            if end_index not in self.word_indices:
                raise ValueError(f"End_index {end_index} not a valid word index")
            start_index = self.word_indices[start_index][0]
            end_index = self.word_indices[end_index][1]

        if start_index < 0 or start_index >= len(self.text):
            raise ValueError(f"Start_index {start_index} out of bounds")

        if end_index <= start_index or end_index >= len(self.text):
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
            "predictions": [{"entity_reference": a.entity, "start_char": s, "end_char": e} for (s, e), a in self.annotations.items()]
        }


    def format(self, only_current_window=False) -> str:
        result = self.text
        sorted_annotations = sorted(
            self.annotations.items(),
            key=lambda item: item[0][1],  # item[0] is (start, end)
        )
        for ann in reversed(sorted_annotations):
            start_idx = ann[0][0]
            end_idx = ann[0][1]
            if (
                only_current_window
                and start_idx < self.annotation_window.start
                or end_idx > self.annotation_window.stop
            ):
                continue

            result = (
                result[:start_idx]
                + "["
                + result[start_idx:end_idx]
                + "]("
                + ann[1].entity
                + ")"
                + result[end_idx:]
            )

        if only_current_window:
            result = result[self.annotation_window.start : self.annotation_window.stop]

        entities: dict[str, Alternative] = {}
        for annot in self.annotations.values():
            if annot.identifier in entities:
                continue

            alternative = Alternative(
                annot.identifier,
                short_identifier=annot.entity,
                label=annot.label,
                aliases=annot.synonyms,
                infos=annot.infos,
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
        "If you cannot find a suitable entity for a sentence, leave it unannotated.",
        "If there are multiple suitable entities for a word or number of words, choose the one that "
        "fits best in the context of the text, or the one that is more popular/general.",
        "If the same entity occurs multiple times in the text, annotate all occurrences.",
        "Before stopping, always check your current annotations.",
    ]


def system_information() -> str:
    return """\
You are an entity annotation assistant. \
Your job is to annotate words from a given text with entities \
from the available knowledge graphs.

You should follow a step-by-step approach to annotate the text:
1. Determine what the text might be about and think about how the words might be \
represented with entities in the knowledge graphs. 
2. Annotate the words in the given excerpt of the text. \
Use the provided functions to search and explore the knowledge graphs. \
You may need to adapt your annotations based on new insights along the way.
3. When you are certain, there are no annotations to be made in the current \
sequence use the change_annotation_window function to view the next excerpt.
4. Use the stop function to finalize your annotations and stop the \
annotation process."""


def functions(managers: list[KgManager], config: GraspConfig) -> list[dict]:
    kgs = [manager.kg for manager in managers]
    method = config.task_kwargs.get("entity-linking", {}).get("method", "prefix")
    if method == "prefix":
        fns = [
            {
                "name": "annotate",
                "description": """\
    Annotate a word or a sequence of words with an entity from the specified knowledge \
    graph by writing the exact words to be annotated as 'words_to_be_annotated'.
    If the annotation fails because the words occur more than once in the given excerpt\
    you can input a couple of words leading up to the words to be annotated as prefix
    (only use if necessary) or you can input a couple words after as suffix \
    (keep it short, only use if necessary).
    You can set 'entity' to 'None' to delete an existing annotation.
    This function overwrites any previous annotation of the words.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "kg": {
                            "type": "string",
                            "enum": kgs,
                            "description": "The knowledge graph to use for the annotation",
                        },
                        "prefix": {
                            "type": "string",
                            "description": "Empty string (or a couple of words before the words to be annotated if necessary)",
                        },
                        "words_to_be_annotated": {
                            "type": "string",
                            "description": "The exact words to be annotated",
                        },
                        "suffix": {
                            "type": "string",
                            "description": "Empty string (or a couple of words after the words to be annotated if necessary)",
                        },
                        "entity": {
                            "type": "string",
                            "description": "The IRI of the entity to annotate the words with",
                        },
                    },
                    "required": ["kg", "prefix", "words_to_be_annotated", "suffix", "entity"],
                    "additionalProperties": False,
                },
                "strict": True,
            },]
    elif method == "markdown":
        fns = [
            {
                "name": "annotate",
                "description": """\
    Annotate a word or a sequence of words with an entity from the specified knowledge \
    graph by annotating the words in the following format: [words to be annotated](entity id). \
    You can set 'entity' to 'None' to delete an existing annotation.
    This function overwrites any previous annotation of the words.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "kg": {
                            "type": "string",
                            "enum": kgs,
                            "description": "The knowledge graph to use for the annotation",
                        },
                        "text_to_be_annotated": {
                            "type": "string",
                            "description": "The whole text to be annotated (your current excerpt) with the words to be annotated \
                            in '[' and ']' brackets followed by the entity id in '(' and ')' brackets \
                            e.g.: [words to be annotated](entity id)"
                        },
                    },
                    "required": ["kg", "text_to_be_annotated"],
                    "additionalProperties": False,
                },
                "strict": True,
            },]
    elif method == "indices":
        fns = [
            {
                "name": "annotate",
                "description": """\
    Annotate a word or a sequence of words with an entity from the specified knowledge \
    graph by inputing the index that you see in brackets behind the words to be \
    annotated as 'start_index' and 'end_index' (they are the same if its only one word).
    You can set 'entity' to 'None' to delete an existing annotation.
    This function overwrites any previous annotation of the words.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "kg": {
                            "type": "string",
                            "enum": kgs,
                            "description": "The knowledge graph to use for the annotation",
                        },
                        "start_index": {
                            "type": "integer",
                            "description": "The start index of the words to be annotated",
                        },
                        "end_index": {
                            "type": "integer",
                            "description": "The end index of the words to be annotated (inclusive)",
                        },
                        "entity": {
                            "type": "string",
                            "description": "The IRI of the entity to annotate the words with",
                        },
                    },
                    "required": ["kg", "start_index", "end_index", "entity"],
                    "additionalProperties": False,
                },
                "strict": True,
            },]
   
    fns.extend([
        {
            "name": "show_annotations",
            "description": "Show the current annotations for the full text or only the current excerpt if 'only_current_window' is set to True.",
            "parameters": {
                "type": "object",
                "properties": {
                    "only_current_window": {
                        "type": "boolean",
                        "description": "set to False to show all annotations, set to True to only show current window",
                    }
                },
                "required": ["only_current_window"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "name": "change_annotation_window",
            "description": "Change the current annotation window and show a small sequence of the full text to be annotated.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_index": {
                        "type": "integer",
                        "description": "The start index of the new window",
                    },
                    "end_index": {
                        "type": "integer",
                        "description": "The end index of the new window",
                    },
                },
                "required": ["start_index", "end_index"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "name": "stop",
            "description": "Finalize your annotations and stop the annotation process.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            "strict": True,
        },
    ])
    return fns


def prepare_annotation(manager: KgManager, entity: str) -> Annotation:
    binding = parse_into_binding(entity, manager.iri_literal_parser, manager.prefixes)
    if binding is None or binding.typ != "uri":
        raise ValueError(f"Entity {entity} is not a valid IRI")

    identifier = binding.identifier()
    
    norm = manager.normalize(identifier, ObjType.ENTITY)
    if norm is not None:
        identifier, _ = norm
    
    infos = manager.get_infos_for_identifiers_of_type([identifier], ObjType.ENTITY)
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
          

#def prepare_annotation(
#    manager: KgManager,
#    entity: str,
#    with_infos: bool = True,
#) -> Annotation:
#    binding = parse_into_binding(entity, manager.iri_literal_parser, manager.prefixes)
#    if binding is None or binding.typ != "uri":
#        raise ValueError(f"Entity {entity} is not a valid IRI")
#
#    identifier = binding.identifier()
#
#    label = None
#    synonyms = None
#    infos = None
#
#    map = manager.entity_mapping
#    norm = map.normalize(identifier)
#    if norm is not None and norm[0] in map:
#        id = map[norm[0]]
#        _, label, *synonyms = manager.entity_index.get_row(id)
#
#    if with_infos:
#        all_infos = manager.get_infos_for_items(
#            [identifier],
#            manager.entity_info_sparql,
#        )
#        infos = all_infos.get(identifier, [])
#
#    return Annotation(
#        identifier=identifier,
#        entity=entity,
#        label=label,
#        synonyms=synonyms,
#        infos=infos,
#    )


def annotate_indices(
    managers: list[KgManager],
    kg: str,
    start_idx: int,
    end_idx: int,
    entity: str,
    state: AnnotationState,
    known: set[str],
    know_before_use: bool = False,
) -> str:
    manager, _ = find_manager(managers, kg)


    # deleting an annotation
    if entity == "None":
        try:
            current = state.annotate(start_idx, end_idx, None, True)
        except ValueError as e:
            raise FunctionCallException(str(e)) from e

        if current is None:
            raise FunctionCallException(
                f"Words [{start_idx}, {end_idx}] is not annotated"
            )

        return f"Deleted annotation {current.entity} from Words [{start_idx}, {end_idx}]"

    # annotating
    else:
        try:
            annotation = prepare_annotation(manager, entity)
            if know_before_use and annotation.identifier not in known:
                raise FunctionCallException(
                    f"The entity {entity} cannot be used for annotation "
                    "without being known from previous function call results. "
                    "This does not mean it is invalid, but you should verify "
                    "that it indeed exists in the knowledge graphs first."
                )

            current = state.annotate(start_idx, end_idx, annotation, True)

        except ValueError as e:
            raise FunctionCallException(str(e)) from e

        if current is None:
            return (
                f"Annotated Words [{start_idx}: {end_idx}] with entity {entity}"
            )
        else:
            return f"Updated annotation of Words [{start_idx}, {end_idx}] from {current.entity} to {entity}"

def annotate_markdown(
    managers: list[KgManager],
    kg: str,
    text_to_be_annotated: str,
    state: AnnotationState,
    known: set[str],
    know_before_use: bool = False,
) -> str:
    manager, _ = find_manager(managers, kg)

    sequence = state.text[state.annotation_window]
    sequence_length = len(sequence)
    annotation_length = len(text_to_be_annotated)

    potential_start_list = []
    found_end = False
    start_idx = 0
    end_idx = 0
    correction_amount = 0
    result = ""
    for i in range(annotation_length):
        # if we find a [ bracket, we remember its index in case there are nested parentheses
        if text_to_be_annotated[i] == "[" and not found_end:
            potential_start_list.append(i)

        # if we have already found a starting bracket and now the combination ](, we know it could be an annotation
        elif len(potential_start_list) > 0 and text_to_be_annotated[i-1: i+1] == "](":
            found_end = True
            start_idx = potential_start_list.pop(-1)
            end_idx = i-2

        # if we find a closing ] bracket without the round opening bracket afterwards, we remove the corresponding opening [ bracket
        elif text_to_be_annotated[i-1] == "]" and len(potential_start_list) > 0 and not found_end:
            potential_start_list.pop(-1)

        # if we find a round closing bracket after already finding the other prerequisites
        elif text_to_be_annotated[i] == ")" and found_end:
            entity = text_to_be_annotated[end_idx+3: i]
            potential_start_list = []
            found_end = False
            start_idx -= correction_amount
            end_idx -= correction_amount

            # deleting an annotation
            if entity == "None":
                try:
                    current = state.annotate(start_idx, end_idx, None)
                except ValueError as e:
                    raise FunctionCallException(str(e)) from e
        
                if current is None:
                    raise FunctionCallException(
                        f"Text sequence [{start_idx}, {end_idx}] is not annotated"
                    )
        
                result += f"Deleted annotation {current.entity} from Text sequence [{start_idx}, {end_idx}]. "
        
            # annotating
            else:
                try:
                    annotation = prepare_annotation(manager, entity)
                    if know_before_use and annotation.identifier not in known:
                        raise FunctionCallException(
                            f"The entity {entity} cannot be used for annotation "
                            "without being known from previous function call results. "
                            "This does not mean it is invalid, but you should verify "
                            "that it indeed exists in the knowledge graphs first."
                        )
        
                    current = state.annotate(start_idx, end_idx, annotation)
        
                except ValueError as e:
                    raise FunctionCallException(str(e)) from e
        
                if current is None:
                    result += f"Annotated text sequence [{start_idx}: {end_idx}] with entity {entity}. "
                else:
                    result += f"Updated annotation of text sequence [{start_idx}, {end_idx}] from {current.entity} to {entity}. "
            correction_amount += (i - end_idx + 1)
    return result



def annotate_prefix(
    managers: list[KgManager],
    kg: str,
    prefix: str,
    words_to_be_annotated: str,
    suffix: str,
    entity: str,
    state: AnnotationState,
    known: set[str],
    know_before_use: bool = False,
) -> str:
    manager, _ = find_manager(managers, kg)

    sequence = state.text[state.annotation_window]
    sequence_length = len(sequence)
    prefix_length = len(prefix)
    suffix_length = len(suffix)
    annotation_length = len(words_to_be_annotated)

    potential_matches = [m.span() for m in re.finditer(words_to_be_annotated, sequence)]

    if len(potential_matches) < 1:
        return f"no match found for the given words to be annotated '{words_to_be_annotated}'"

    pm_cntr = 0
    for pm in potential_matches:
        start_idx, end_idx = pm
        # check if the prefix is even possible length wise
        if start_idx >= prefix_length and prefix == sequence[start_idx - prefix_length: start_idx]:
            if sequence_length >= end_idx + suffix_length and suffix == sequence[end_idx: end_idx + suffix_length]:
                pm_cntr += 1
            elif sequence_length > end_idx + suffix_length and (" " + suffix) == sequence[end_idx: end_idx + suffix_length + 1]:
                pm_cntr += 1
        # check if the prefix is even possible length wise if we assume a blank space between prefix and words to be annotated
        elif start_idx > prefix_length and (prefix + " ") == sequence[start_idx - prefix_length - 1: start_idx]:
            if sequence_length >= end_idx + suffix_length and suffix == sequence[end_idx: end_idx + suffix_length]:
                pm_cntr += 1
            elif sequence_length > end_idx + suffix_length and (" " + suffix) == sequence[end_idx: end_idx + suffix_length + 1]:
                pm_cntr += 1

        
    if pm_cntr < 1:
        return f"match found for words to be annotated '{words_to_be_annotated}' but no match found for the given prefix '{prefix}' or suffix '{suffix}'."
    if pm_cntr > 1:
        return f"more than one possible match found for the given prefix '{prefix}' and words '{words_to_be_annotated}' and suffix '{suffix}'."

    start_idx, end_idx = potential_matches[0]

    # deleting an annotation
    if entity == "None":
        try:
            current = state.annotate(start_idx, end_idx, None)
        except ValueError as e:
            raise FunctionCallException(str(e)) from e

        if current is None:
            raise FunctionCallException(
                f"Text sequence [{start_idx}, {end_idx}] is not annotated"
            )

        return f"Deleted annotation {current.entity} from Text sequence [{start_idx}, {end_idx}]"

    # annotating
    else:
        try:
            annotation = prepare_annotation(manager, entity)
            if know_before_use and annotation.identifier not in known:
                raise FunctionCallException(
                    f"The entity {entity} cannot be used for annotation "
                    "without being known from previous function call results. "
                    "This does not mean it is invalid, but you should verify "
                    "that it indeed exists in the knowledge graphs first."
                )

            current = state.annotate(start_idx, end_idx, annotation)

        except ValueError as e:
            raise FunctionCallException(str(e)) from e

        if current is None:
            return (
                f"Annotated text sequence [{start_idx}: {end_idx}] '{sequence[start_idx: end_idx]}' with entity {entity}"
            )
        else:
            return f"Updated annotation of text sequence [{start_idx}, {end_idx}] '{sequence[start_idx: end_idx]}' from {current.entity} to {entity}"



def input_instructions(state: AnnotationState) -> str:
    instructions = """\
Annotate the following text with entities from the available knowledge graphs. \
If there already are annotations for some words, they are shown in parentheses \
after the word value.

You will be given the full text in the beginning and then you will be given an \
excerpt of the text to annotate. When you're done with a sequence and call the \
function 'change_annotation_window' you will be given the next excerpt of the text to annotate. \
To get the first sequence you need to call 'change_annotation_window'. \n
"""

    instructions += state.format()
    return instructions


def input_and_state(input: Any, config: GraspConfig) -> tuple[str, AnnotationState]:

    annots = AnnotationState(input, config.task_kwargs.get("entity-linking", {}).get("method", "prefix"))
    instructions = input_instructions(annots)
    return instructions, annots


def call_function(
    config: GraspConfig,
    managers: list[KgManager],
    fn_name: str,
    fn_args: dict,
    known: set[str],
    state: AnnotationState | None = None,
    example_indices: dict | None = None,
) -> str:
    print("function call:", fn_name, fn_args)
    assert isinstance(state, AnnotationState), (
        "Annotations must be provided as state for entity linking task"
    )
    assert not example_indices, (
        "Example indices are not supported for entity linking task"
    )

    method = config.task_kwargs.get("entity-linking", {}).get("method", "prefix")

    if fn_name == "annotate":
        if method == "markdown":
            return annotate_markdown(
                managers,
                fn_args["kg"],
                fn_args["text_to_be_annotated"],
                state,
                known,
                config.know_before_use,
            )

        elif method == "indices":
            return annotate_indices(
                managers,
                fn_args["kg"],
                fn_args["start_index"],
                fn_args["end_index"],
                fn_args["entity"],
                state,
                known,
                config.know_before_use,
            )

        elif method == "prefix":
            return annotate_prefix(
                managers,
                fn_args["kg"],
                fn_args["prefix"],
                fn_args["words_to_be_annotated"],
                fn_args["suffix"],
                fn_args["entity"],
                state,
                known,
                config.know_before_use,
            )

    elif fn_name == "show_annotations":
        return state.format(fn_args["only_current_window"])

    elif fn_name == "change_annotation_window":
        return state.change_annotation_window(fn_args["start_index"], fn_args["end_index"])

    elif fn_name == "stop":
        if state.unviewed:
            return "The following text ranges have not been viewed:" + state.show_unviewed_ranges() + "You need to explore the full text to be able to stop."
        else:
            return "Stopping"

    else:
        raise ValueError(f"Unknown function {fn_name}")


def output(state: AnnotationState) -> dict:
    return state.to_dict()


def feedback_system_message(
    managers: list[KgManager],
    kg_notes: dict[str, list[str]],
    notes: list[str],
) -> str:
    return f"""\
You are a text annotation assistant providing feedback on the \
output of a text annotation system for a given input text.

The system has access to the following knowledge graphs:
{format_kgs(managers, kg_notes)}

The system was provided the following notes across all knowledge graphs:
{format_notes(notes)}

The system was provided the following rules to follow:
{format_list(rules())}

Provide your feedback with the give_feedback function."""


def feedback_instructions(inputs: list[str], output: dict) -> str:
    assert inputs, "At least one input is required for feedback"

    if len(inputs) > 1:
        prompt = (
            "Previous inputs:\n" + "\n\n".join(i.strip() for i in inputs[:-1]) + "\n\n"
        )

    else:
        prompt = ""

    prompt += f"Input:\n{inputs[-1].strip()}"
    prompt += f"\n\nAnnotations:\n{output['formatted']}"
    return prompt


# ── Task class ──────────────────────────────────────────────────────────────


from grasp.model import Message  # noqa: E402
from grasp.tasks.base import FeedbackTask, GraspTask  # noqa: E402


class EntityLinkingTask(GraspTask, FeedbackTask):
    name = "entity-linking"

    def system_information(self) -> str:
        return system_information()

    def rules(self) -> list[str]:
        return rules()

    def function_definitions(self) -> list[dict]:
        return functions(self.managers, self.config)

    def call_function(
        self,
        fn_name: str,
        fn_args: dict,
        known: set[str],
        state: Any,
        example_indices: dict | None,
    ) -> str:
        return call_function(
            self.config, self.managers, fn_name, fn_args, known, state, example_indices
        )

    def done(self, fn_name: str) -> bool:
        return fn_name == "stop"

    def setup(self, input: Any) -> tuple[str, Any]:
        return input_and_state(input, self.config)

    def output(self, messages: list[Message], state: Any) -> dict:
        return output(state)

    @property
    def default_input_field(self) -> str | None:
        return "table"

    @classmethod
    def sample_cls(cls) -> type[EntityLinkingSample]:
        return EntityLinkingSample

    def feedback_system_message(
        self, kg_notes: dict[str, list[str]], notes: list[str]
    ) -> str:
        return feedback_system_message(self.managers, kg_notes, notes)

    def feedback_instructions(self, inputs: list[str], output: dict) -> str:
        return feedback_instructions(inputs, output)
