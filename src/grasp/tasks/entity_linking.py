########################################################################################
#  This code builds on cea.py from github.com/ad-freiburg/grasp/src/grasp/tasks/ but   #
#  is modified for general text entity linking instead of just table cell annotation.  #
########################################################################################

import re
import unicodedata
from typing import Any

from pydantic import BaseModel

from grasp.configs import GraspConfig
from grasp.functions import find_manager
from grasp.manager import KgManager, format_kgs
from grasp.sparql.types import Alternative, ObjType
from grasp.model import Message
from grasp.sparql.utils import parse_into_binding
from grasp.tasks.base import FeedbackTask, GraspTask
from grasp.tasks.examples import Sample
from grasp.utils import FunctionCallException, format_list, format_notes


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
    start: int | None = None
    end: int | None = None

    @property
    def length(self) -> int:
        return len(self.data)

    def trim(self, context: int | None = None) -> tuple["Text", int]:
        """
        Trims the Text to the start/end values if context is 0 OR trims the Text to 
        start/end plus context otherwise. If context = 'None' does not trim the Text.
        """
        if context and context < 0:
            raise ValueError(f"context {context} must be non negative")
        if self.start and self.start >= self.length:
            raise ValueError(f"start {self.start} must be less than length of the text {self.length}")
        if self.end and self.end <= 0:
            raise ValueError(f"start {self.end} must be greater than zero")
        if self.start and self.end and self.end <= self.start:
            raise ValueError(f"end {self.end} must be greater than start {self.start}")
        
        # we need 4 variables: start/end of new text and then start/end of the part to be annotated
        
        # if context is None, we view the whole text, there's no trimming
        if context is None:
            new_start = 0
            new_end = self.length
            context = 0
            if self.start is None or self.start < 0:
                window_start = 0
            else:
                window_start = self.start
            if self.end is None or self.end > self.length:
                window_end = self.length
            else:
                window_end = self.end

        # with context, we trim the text
        else:
            if self.start is None or self.start < 0:
                new_start = 0
                window_start = 0
            else:
                new_start = max(0, self.start - context)
                window_start = self.start - new_start

            if self.end is None or self.end > self.length:
                new_end = self.length
                window_end = self.length
            else:
                new_end = min(self.length, self.end + context)
                window_end = self.end - new_start


        trimmed = Text(
            data=self.data[new_start:new_end],
            start=window_start,
            end=window_end,
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
        method: str = "prefix",
        use_annotation_window: bool = False,
    ) -> None:

        self.text, self.offset = text.trim(context)

        self.max_annotation_window_size = 300
        self.use_annotation_window = use_annotation_window
        
        # if we don't use an annotation window, the window is just set to the text start/end
        self.annotation_window: slice = (
            slice(0, 0) if self.use_annotation_window
            else slice(self.text.start, self.text.end)
        )
        
        # if we don't use an annotation window the full text is viewed per definition
        self.unviewed: list[tuple[int, int]] = (
            [(self.text.start, self.text.end)] if self.use_annotation_window else []
        )

        self.annotations: dict[tuple[int, int], Annotation] = {}
        self.word_indices: dict[int, tuple[int, int]] = {}

        self.method = method

    def show_unviewed_ranges(self):
        """
        Gives a string representation of the self.unviewed list of unviewed ranges.
        """
        if not self.use_annotation_window:
            raise RuntimeError("Flag 'use_annotation_window' is False, function 'show_unviewed_ranges' should not be called.")
        return str(self.unviewed)

    def remove_from_unviewed(self, start_idx: int, end_idx: int) -> None:
        """
        Remove or truncate all intervalls between start_idx and end_idx from the 
        self.unviewed list of intervals that have not been viewed. 
        """
        unviewed_length = len(self.unviewed)
        i = 0
        # go through all unviewed ranges:
        while i < unviewed_length:
            # range start in current window
            if self.unviewed[i][0] >= start_idx and self.unviewed[i][0] < end_idx:
                # range end also in current window -> remove it
                if self.unviewed[i][1] <= end_idx:
                    self.unviewed.pop(i)
                    i -= 1
                    unviewed_length -= 1
                # range end beyond current window -> shorten it
                else:
                    self.unviewed[i] = (end_idx, self.unviewed[i][1])

            # range start before current window
            elif self.unviewed[i][0] < start_idx:
                # range end in current window -> shorten it
                if self.unviewed[i][1] >= start_idx and self.unviewed[i][1] <= end_idx:
                    self.unviewed[i] = (self.unviewed[i][0], start_idx)
                # range end after current window -> split it
                elif self.unviewed[i][1] > end_idx:
                    insert_idx = self.unviewed.index(self.unviewed[i]) + 1
                    self.unviewed.insert(insert_idx, (end_idx, self.unviewed[i][1]))
                    self.unviewed[i] = (self.unviewed[i][0], start_idx)
                    i += 1
                    unviewed_length += 1
            i += 1
    
    def create_word_indices(self) -> None:
        """
        A heuristic to create indices for words in order to enable the index 
        annotation function, changes self.word_indices.
        """
        if self.method != "indices":
            raise FunctionCallException(
                "method is not 'indices', function create_indices should not be called"
            )

        currently_at_word = False
        currently_at_number = False
        self.word_indices = {}
        idx = 0
        current_excerpt = self.text.data[self.annotation_window]

        number_characters = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}
        number_punctuation = {".", ","}
        punctuation_characters = {
            " ", ".", ",", "-", ":", ";", "[", "]", "(", ")", "{", "}", "\n", "'", "\"", "’", "‘", "?", "="
        }

        for i, c in enumerate(current_excerpt):
            # start word
            if (c not in punctuation_characters 
                and c not in number_characters 
                and not currently_at_word 
                and not currently_at_number
            ):
                currently_at_word = True
                s = i
            # start number
            elif c in number_characters and not currently_at_word and not currently_at_number:
                currently_at_number = True
                s = i
            # '.' or ',' in the middle of number
            elif (
                c in number_punctuation and currently_at_number 
                and len(current_excerpt) > i and current_excerpt[i + 1] in number_characters
            ):
                continue
            # end of word
            elif c in punctuation_characters and currently_at_word:
                currently_at_word = False
                e = i
                self.word_indices[idx] = (s, e)
                idx += 1
            # end of number
            elif c not in number_characters and currently_at_number:
                currently_at_number = False
                e = i
                self.word_indices[idx] = (s, e)
                # if the current character is not punctuation we don't increment yet
                if c in punctuation_characters:
                    idx += 1
        
    def change_annotation_window(self, start, end) -> str:
        """
        Changes the window in which the model can make annotations to the given
        start and end values. 
        """
        if not self.use_annotation_window:
            raise RuntimeError(
                "Flag 'use_annotation_window' is False, function "
                "'change_annotation_window' should not be called."
            )

        result = ""
        if start < self.text.start:
            raise ValueError(f"Your start index {start} cannot be less than {self.text.start}.")
        if start + 1  >= self.text.end:
            raise ValueError(f"Your start index {start} cannot be greater than {self.text.end}.")
        if end <= start:
            raise ValueError(f"Your end index {end} is before your start index {start}.")
        if end > start + self.max_annotation_window_size:
            result += (
                f"Your annotation window size exceeds the maximum allowed annotation" 
                f"window size of {self.max_annotation_window_size}, it was adjusted. "
            )
            end = start + self.max_annotation_window_size
        new_start_idx = start
        new_end_idx = min(end, self.text.end if self.text.end else self.text.length)

        # remove this new section from the unviewed ranges
        self.remove_from_unviewed(new_start_idx, new_end_idx)

        self.annotation_window = slice(new_start_idx, new_end_idx)
        current_excerpt = self.text.data[self.annotation_window]
        result += (f"Annotation Window changed to excerpt [{new_start_idx}: "
                   f"{new_end_idx}] of the full Text of lenght {self.text.length}.\n")

        return result

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
    
    def delete_annotations_in_current_window(self):
        delete_list = []
        for ann in self.annotations:
            if (ann[0] >= self.annotation_window.start 
                or ann[1] < self.annotation_window.stop):
                delete_list += [ann]
        for ann in delete_list:
            self.annotations.pop(ann)

    def get(self, start_index: int, end_index: int) -> Annotation | None:
        return self.annotations.get((start_index, end_index), None)

    def to_dict(self) -> dict:
        return {
            "formatted": self.format(),
            "predictions": [
                {"entity_reference": a.entity,
                 "start_char": s + self.offset,
                 "end_char": e + self.offset} for (s, e), a in self.annotations.items()
            ]
        }

    def format(self, only_current_window=False, add_word_indices=False) -> str:
        """
        Returns a string with the current annotation state of the text.
        Annotations are visualized in the following format: '[annotated words](q123)',
        '[[Nested [annotations](q123)](q456) are supported](q789)'.
        If only_current_window is true, only the text of the current annotation window
        is shown. If add_word_indices is true, the function 'create_word_indices' is
        used to generate indices for words which are then shown in the folling format:
        'word(1) and(2) composed(3)-word(4) and [annotated(5) words(6)](q123).'
        """
        result = self.text.data
        if add_word_indices:
            self.create_word_indices()
        else:
            self.word_indices = {}

        # item[0] is (start, end), we sort by end first, then by negative start
        sorted_annotations = sorted(
            self.annotations.items(),
            key=lambda item: (item[0][1], -item[0][0])
        )

        sorted_indices = sorted(
            self.word_indices.items(),
            key=lambda item: item[1][1]
        )


        # go through annotations from highest end index first
        nested_list = []
        while sorted_annotations or sorted_indices:
            currently_annotating_index = False
            if (not sorted_annotations or
                sorted_indices and sorted_annotations
                and sorted_indices[-1][1][1] + self.annotation_window.start
                > sorted_annotations[-1][0][1]
            ):
                currently_annotating_index = True
                ind = sorted_indices.pop() 
                start_idx = end_idx = ind[1][1] + self.annotation_window.start
            else:
                ann = sorted_annotations.pop() 
                start_idx = ann[0][0]
                end_idx = ann[0][1]

            start_offset = 0
            end_offset = 0
            for i in range(len(nested_list) -1, -1, -1):
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

            if (
                only_current_window
                and (start_idx < self.annotation_window.start
                or end_idx > self.annotation_window.stop)
            ):
                continue

            prefix = result[:start_idx + start_offset]
            words = result[start_idx + start_offset:end_idx + end_offset]
            suffix = result[end_idx + end_offset:]
            if currently_annotating_index:
                result = (prefix + "(" + str(ind[0]) + ")" + suffix)
            else:
                result = (prefix + "[" + words + "](" + ann[1].entity + ")" + suffix)


        # trim to only show the current window
        if only_current_window:
            result = result[
                self.annotation_window.start
                :self.annotation_window.stop - self.text.length - 1
            ]

        entities: dict[str, Alternative] = {}
        for annot in self.annotations.values():
            if annot.identifier in entities:
                continue

            alternative = Alternative(
                annot.identifier,
                short_identifier=annot.entity,
                label=annot.label,
                aliases=annot.aliases,
                infos=annot.infos,
            )
            entities[annot.identifier] = alternative

        if entities:
            annotations = format_list(
                alt.get_selection_string() for _, alt in sorted(entities.items())
            )
            result += f"\n\nAnnotated entities:\n{annotations}"

        return result


def rules(only_named: bool = False) -> list[str]:
    return [
        "If you cannot find a suitable entity in a sentence, leave it unannotated.",
        "If there are multiple suitable entities for a number of words, choose the one that "
        "fits best in the context of the text, or the one that is more popular/general.",
        "If the same entity occurs multiple times in the text, annotate all occurrences.",
        "Before stopping, always check your current annotations.",
        "Annotate named entities " + "only." if only_named else "and unnamed entities.",
        "If you think you are done with annotating, stop, don't try to do more.",
        "If you think you are stuck in a reasoning loop, you need to move on.",
    ]


def system_information() -> str:
    return """\
You are an entity annotation assistant. \
Your job is to annotate words from a given text with entities \
from the available knowledge graphs.

You should follow a step-by-step approach to annotate the text:
1. Determine what the text might be about and think about how the words might be \
represented with entities in the knowledge graphs. 
2. Annotate the words in the given text. \
Use the provided functions to search and explore the knowledge graphs. \
You may need to adapt your annotations based on new insights along the way.
3. When you are certain, there are no annotations to be made use the stop function \
to finalize your annotations and stop the annotation process."""


def functions(managers: list[KgManager], config: GraspConfig) -> list[dict]:
    kgs = [manager.kg for manager in managers]
    method = config.task_kwargs.get("entity-linking", {}).get("method", "prefix")
    if method not in {"prefix", "indices", "markdown"}:
        raise ValueError(f"annotation method {method} needs to be one of: indices, prefix, markdown")
    use_annotation_window = config.task_kwargs.get(
        "entity-linking", {}).get("use_annotation_window", False
    )
    if method == "prefix":
        fns = [
            {
                "name": "annotate",
                "description": """\
    Annotate a word or a sequence of words with an entity from the specified knowledge \
    graph by writing the exact words to be annotated as 'words_to_be_annotated'.
    If the annotation fails you can input EITHER a couple of words before the words to be \
    annotated as prefix OR you can input a couple words after as suffix.\
    Do not use Newlines as suffix or prefix, use the next word in those cases.\
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
                        "optional_short_prefix": {
                            "type": "string",
                            "description": "OPTIONAL: a word or two before the words to be annotated",
                        },
                        "exact_words_to_be_annotated": {
                            "type": "string",
                            "description": "The exact words to be annotated written exactly like in the original text",
                        },
                        "optional_short_suffix": {
                            "type": "string",
                            "description": "OPTIONAL: a word or two after the words to be annotated",
                        },
                        "entity": {
                            "type": "string",
                            "description": "The IRI of the entity to annotate the words with",
                        },
                    },
                    "required": ["kg", "optional_short_prefix", "exact_words_to_be_annotated", "optional_short_suffix", "entity"],
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
    You need to annotate the full window in one go. Every call overwrites the old annotations. \
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
   
    if use_annotation_window:
        fns.extend([
            {
                "name": "change_annotation_window",
                "description": "Change the current window out of the full text in which you can annotate entities.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start_index": {
                            "type": "integer",
                            "description": "Start index of the new annotation window",
                        },
                        "end_index": {
                            "type": "integer",
                            "description": "End index of the new annotation window",
                        },
                    },
                    "required": ["start_index", "end_index"],
                    "additionalProperties": False,
                },
                "strict": True,
            }, 
            {
                "name": "show_current_text_and_annotations",
                "description": "Show the text to annotate with current annotations.",
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
            }
        ])
    else:
        fns.extend([
            {
                "name": "show_current_text_and_annotations",
                "description": "Show the text to annotate with current state of the annotations.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
                "strict": True,
            }])

    fns.extend([
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
          

def annotate_indices(
    managers: list[KgManager],
    kg: str,
    start: int,
    end: int,
    entity: str,
    state: AnnotationState,
    known: set[str],
    know_before_use: bool = False,
) -> str:
    """
    A function for the llm to call to annotate the text with start and end word 
    indices that refer to the state.word_indices list to translate to characters
    in the original text.
    """
    manager, _ = find_manager(managers, kg)

    if start not in state.word_indices:
        raise ValueError(f"Start_index {start} not a valid word index")
    if end not in state.word_indices:
        raise ValueError(f"End_index {end} not a valid word index")
    start_idx = state.word_indices[start][0]
    end_idx = state.word_indices[end][1]

    # deleting an annotation
    if entity == "None":
        try:
            current = state.annotate(start_idx, end_idx, None)
        except ValueError as e:
            raise FunctionCallException(str(e)) from e

        if current is None:
            raise FunctionCallException(
                f"Words [{start}, {end}] is not annotated"
            )

        return f"Deleted annotation {current.entity} from Words [{start}, {end}]"

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

        sequence = state.text.data[state.annotation_window]
        if current is None:
            return (
                f"Annotated text sequence [{start}: {end}] "
                f"'{sequence[start_idx: end_idx]}' with entity {entity}"
            )
        else:
            return (
                f"Updated annotation of text sequence [{start}, {end}] "
                f"'{sequence[start_idx: end_idx]}' from {current.entity} to {entity}"
            )


def annotate_markdown(
    managers: list[KgManager],
    kg: str,
    text_to_be_annotated: str,
    state: AnnotationState,
    known: set[str],
    know_before_use: bool = False,
) -> str:
    """
    A function for the llm to call to annotate the text by writing markdown annotations
    in the format: 'original text [words to be annotated](q123) rest of original text' 
    to translate to characters and annotations in the original text.
    """
    manager, _ = find_manager(managers, kg)

    sequence = state.text.data[state.annotation_window]
    annotation_length = len(text_to_be_annotated)

    potential_start_list = []
    found_end = False
    start_idx = 0
    end_idx = 0
    correction_amount = 0
    result = ""
    state.delete_annotations_in_current_window()
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

            # annotating
            try:
                annotation = prepare_annotation(manager, entity)
                if know_before_use and annotation.identifier not in known:
                    result += (
                        f"The entity {entity} cannot be used for annotation "
                        "without being known from previous function call results. "
                        "This does not mean it is invalid, but you should verify "
                        "that it indeed exists in the knowledge graphs first."
                    )
                else:
                    state.annotate(start_idx, end_idx, annotation)
                    result += (
                        f"Annotated text sequence [{start_idx}: {end_idx}] "
                        f"'{sequence[start_idx: end_idx]}' with entity {entity} "
                    )
        
            except ValueError as e:
                raise FunctionCallException(str(e)) from e

            correction_amount = (i - end_idx + 1)
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
    """
    A function for the llm to call to annotate the text with the exact string to be 
    annotated and optionally prefix and/or suffix to distinguish between different
    occurrences of the words in the original text.
    """
    manager, _ = find_manager(managers, kg)
    sequence = state.text.data[state.annotation_window]
    # normalizing, because some llms are heavily biased towards specific characters like
    # the ascii apostrophe although they are technically able to output the correct one.
    def normalize(string: str) -> str:
        return unicodedata.normalize("NFC", string).replace("‘", "'").replace("’", "'")

    words_to_be_annotated = normalize(words_to_be_annotated)
    sequence = normalize(sequence)
    prefix = normalize(prefix)
    suffix = normalize(suffix)
    sequence_length = len(sequence)
    prefix_length = len(prefix)
    suffix_length = len(suffix)

    word_matches = [m.span() for m in re.finditer(re.escape(words_to_be_annotated), sequence)]

    if not word_matches:
        raise ValueError(
            f"No match found for the given words to be annotated "
            f"'{words_to_be_annotated}' in the current annotation window."
        )

    # find prefix/suffix matches for the word matches candidates
    pm_cntr = 0
    pm_list = []
    for pm in word_matches:
        start_idx, end_idx = pm
        m = 0
        n = 0
        ignored_characters = {" ", "\n", "\r", ".", ",", "\"", "'", ":", "“", "”", "„", "´", "’"}
        brk = False
        while start_idx >= prefix_length + m:
            if prefix == sequence[start_idx - prefix_length - m: start_idx - m]:
                while sequence_length >= end_idx + suffix_length + n:
                    if suffix == sequence[end_idx + n: end_idx + suffix_length + n]:
                        pm_cntr += 1
                        start, end = pm
                        pm_list.append(pm)
                        brk = True
                        break
                    if sequence[end_idx + n] not in ignored_characters:
                        brk = True
                        break
                    n += 1
            m += 1
            if sequence[start_idx - m] not in ignored_characters or brk:
                break


    if pm_cntr < 1:
        raise ValueError(
            f"Match found for words to be annotated '{words_to_be_annotated}' but "
            f"no match found for the given prefix '{prefix}' or suffix '{suffix}'."
            "Maybe try leaving out either the prefix or suffix."
        )

    if pm_cntr > 1:
        raise ValueError(
            f"{pm_cntr} possible matches found for the given prefix '{prefix}' "
            f"and words '{words_to_be_annotated}' and suffix '{suffix}'."
            "Try adding a word for context either in the prefix or suffix."
        )

    start_idx, end_idx = start, end

    # deleting an annotation
    if entity == "None":
        try:
            current = state.annotate(start_idx, end_idx, None)
        except ValueError as e:
            raise FunctionCallException(str(e)) from e

        if current is None:
            raise FunctionCallException(
                f"Text sequence [{start_idx}, {end_idx}] "
                f"'{sequence[start_idx: end_idx]}' is not annotated"
            )

        return (
            f"Deleted annotation {current.entity} from Text sequence "
            f"[{start_idx}, {end_idx}] '{sequence[start_idx: end_idx]}'"
        )

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
                f"Annotated text sequence [{start_idx}: {end_idx}] "
                f"'{sequence[start_idx: end_idx]}' with entity {entity}"
            )
        else:
            return (
                f"Updated annotation of text sequence [{start_idx}, {end_idx}] "
                f"'{sequence[start_idx: end_idx]}' from {current.entity} to {entity}"
            )


def input_instructions(state: AnnotationState) -> str:
    instructions = (
        "Annotate the following text with entities from the available knowledge graphs. "
        "Annotate every occurance of an entity, if it occurs again annotate it again! \n" 
        "You will be given the full text in the beginning only for context. \n"
        "[Start of full Text]\n\n"
    ) 

    instructions += state.format()

    instructions += (
        "\n\n[End of full Text]\n\n"
        "To start annotating and when you're done with a sequence first call the function "
        "'change_annotation_window' to set the next window of the text to annotate "
        " and then 'show_current_text_and_annotations' with 'only_current_window' True"
        " to show the text in the current window.\n"
    ) if state.use_annotation_window else (
        "\n\n[End of full Text]\n\n"
        "To see which specific excerpt of the text you should annotate call "
        "'show_current_text_and_annotations' before starting, the result is the only "
        "excerpt of the text you need to annotate, the full text can be longer.\n"
    )

    return instructions


def input_and_state(input: Any, config: GraspConfig) -> tuple[str, AnnotationState]:
    try:
        text = Text(**input)
    except Exception as e:
        raise ValueError(
            "Entity Linking task input must be a dict with a 'data' field "
            "and optional 'start' and/or 'end' fields"
        ) from e

    annots = AnnotationState(
        text,
        context=config.task_kwargs.get("entity-linking", {}).get("context", None),
        method=config.task_kwargs.get("entity-linking", {}).get("method", "prefix"),
        use_annotation_window=config.task_kwargs.get("entity-linking", {}).get("use_annotation_window", False)
    )
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
) -> str | None:
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
                fn_args["optional_short_prefix"],
                fn_args["exact_words_to_be_annotated"],
                fn_args["optional_short_suffix"],
                fn_args["entity"],
                state,
                known,
                config.know_before_use,
            )

    elif fn_name == "show_current_text_and_annotations":
        return state.format(
            fn_args["only_current_window"] if state.use_annotation_window else True,
            method=="indices"
        )

    elif fn_name == "change_annotation_window":
        return state.change_annotation_window(fn_args["start_index"], fn_args["end_index"])

    elif fn_name == "stop":
        if state.unviewed:
            return (
                "The following text ranges have not been viewed:" 
                + state.show_unviewed_ranges() 
                + ". You need to explore the full text to be able to stop."
            )
        else:
            return "Stopping"

    else:
        raise ValueError(f"Unknown function {fn_name}")


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


class EntityLinkingTask(GraspTask, FeedbackTask):
    name = "entity-linking"

    def system_information(self) -> str:
        return system_information()

    def rules(self) -> list[str]:
        return rules(self.config.task_kwargs.get("entity-linking", {}).get("only_named_entities", False))

    def function_definitions(self) -> list[dict]:
        return functions(self.managers, self.config)

    def call_function(
        self,
        fn_name: str,
        fn_args: dict,
        known: set[str],
        example_indices: dict | None = None,
    ) -> str | None:
        return call_function(
            self.config, self.managers, fn_name, fn_args, known, self.state, example_indices
        )

    def done(self, fn_name: str) -> bool:
        return fn_name == "stop" and not self.state.unviewed

    def setup(self, input: Any) -> str:
        instructions, self.state = input_and_state(input, self.config)
        return instructions

    def output(self, messages: list[Message]) -> dict:
        return self.state.to_dict()

    @property
    def default_input_field(self) -> str | None:
        return "text"

    @classmethod
    def sample_cls(cls) -> type[EntityLinkingSample]:
        return EntityLinkingSample

    def feedback_system_message(
        self, kg_notes: dict[str, list[str]], notes: list[str]
    ) -> str:
        return feedback_system_message(self.managers, kg_notes, notes)

    def feedback_instructions(self, inputs: list[str], output: dict) -> str:
        return feedback_instructions(inputs, output)