from typing import Any, Iterator, Optional

from pydantic import BaseModel

from grasp.configs import GraspConfig
from grasp.functions import TaskFunctions, find_manager
from grasp.manager import KgManager, format_kgs
from grasp.sparql.types import Alternative
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


class EntityMention(BaseModel):
    span: tuple[int, int]
    recognized_by: Optional[str]
    id: int
    linked_by: str
    candidates: list[str] = []


class Article(BaseModel):
    id: int
    title: str
    text: str
    entity_mentions: Optional[list[EntityMention]] = None


class EntityLinkingSample(Sample):
    article: Article
    annotations: list[TextAnnotation]

    def input(self) -> Any:
        return self.article.text.model_dump()

    def queries(self) -> list[str]:
        annots = AnnotationState(self.article.text)
        return [annots.format()]


class AnnotationState:
    def __init__(
            self,
            article: Article,
            context_rows,
            min_annotation_window_size: int = 100,
            max_annotation_window_size: int = 200,
            annotation_window_overlap: int = 20) -> None:

        self.article: Article = article

        self.min_annotation_window_size = min_annotation_window_size
        self.max_annotation_window_size = max_annotation_window_size
        self.annotation_window_overlap = annotation_window_overlap

        self.annotation_window: slice  = slice(0, 0)
        self.annotations: dict[tuple[int, int], Annotation] = {}

    def next(self):
        new_start_idx = max(self.annotation_window.stop - self.annotation_window_overlap, 0)

        new_end_idx = min(new_start_idx + self.min_annotation_window_size, len(self.article.text) - 1)

        if new_start_idx >= len(self.article.text) - 1:
            return "You reached the end of the text, there is no new sequence to annotate."

        self.annotation_window = slice(new_start_idx, new_end_idx)

        return "The next text sequence to annotate is:\n\n" + self.article.text[self.annotation_window]

    def annotate(
        self,
        start_index: int,
        end_index: int,
        annotation: Annotation | None,
    ) -> Annotation | None:
        if start_index < 0 or start_index >= len(self.article.text):
            raise ValueError(f"Start_index {start_index} out of bounds")

        if end_index <= start_index or end_index >= len(self.article.text):
            raise ValueError(f"End_index {end_index} out of bounds")

        start_index += self.annotation_window.start
        end_index += self.annotation_window.start

        current = self.annotations.pop((start_index, end_index), None)
        if annotation is not None:
            self.annotations[(start_index, end_index)] = annotation
        return current

    def get(self, start_index: int, end_index: int) -> Annotation | None:
        return self.annotations.get((start_index, end_index), None)

    def to_dict_new(self) -> Article:
        output: Article = self.article 
        entity_mentions: list[EntityMention] = []
        for (start_idx, end_idx), annotation in self.annotations.items():
            m = EntityMention(span=(start_idx, end_idx), recognized_by="grasp", id=annotation, linked_by="grasp")
            entity_mentions.append(m)
        
        return output

    def to_dict(self) -> dict:
        return {
            "formatted": self.format(),
            "annotations": [annot.model_dump() for (row, column), annot in self.annotations.items()],
        }

    def iter(self) -> Iterator[list[Annotation | None]]:
        for r in range(self.table.height):
            yield [self.get(r, c) for c in range(self.table.width)]

    def format(self, only_current_window=False) -> str:
        result = self.article.text
        sorted_annotations = sorted(
            self.annotations.items(),
            key=lambda item: item[0][1]  # item[0] is (start, end)
        )
        for ann in reversed(sorted_annotations):
            start_idx = ann[0][0]
            end_idx = ann[0][1]
            if only_current_window and start_idx < self.annotation_window.start or end_idx > self.annotation_window.stop:
                continue

            result = result[:start_idx] + "[" + result[start_idx:end_idx+1] + "](" + ann[1].entity + ")" + result[end_idx+1:]

        if only_current_window: result = result[self.annotation_window.start: self.annotation_window.stop]


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
1. Determine what the text might be about and hink about how the words might be \
represented with entities in the knowledge graphs. 
2. Annotate the words in the given excerpt of the text. \
Use the provided functions to search and explore the knowledge graphs. \
You may need to adapt your annotations based on new insights along the way.
3. When you are certain, there are no annotations to be made in the current \
sequence use the next function to view the next excerpt.
4. Use the stop function to finalize your annotations and stop the \
annotation process."""


def functions(managers: list[KgManager]) -> TaskFunctions:
    kgs = [manager.kg for manager in managers]
    fns = [
        {
            "name": "annotate",
            "description": """\
Annotate a word or a sequence of words with an entity from the specified knowledge \
graph by inputing the currently given sequence up to before the words to be \
annotated as prefix and then inputting the words to be annotated.
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
                        "description": "The current excerpt of the text sequence until up to before the words to be annotated",
                    },
                    "words_to_be_annotated": {
                        "type": "string",
                        "description": "The exact words to be annotated",
                    },
                    "entity": {
                        "type": "string",
                        "description": "The IRI of the entity to annotate the words with",
                    },
                },
                "required": ["kg", "prefix", "words_to_be_annotated", "entity"],
                "additionalProperties": False,
            },
            "strict": True,
        },
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
            "name": "next",
            "description": "Show next small sequence of the full text to be annotated.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
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
    ]
    return fns, call_function


def prepare_annotation(
    manager: KgManager,
    entity: str,
    with_infos: bool = True,
) -> Annotation:
    binding = parse_into_binding(entity, manager.iri_literal_parser, manager.prefixes)
    if binding is None or binding.typ != "uri":
        raise ValueError(f"Entity {entity} is not a valid IRI")

    identifier = binding.identifier()

    label = None
    synonyms = None
    infos = None

    map = manager.entity_mapping
    norm = map.normalize(identifier)
    if norm is not None and norm[0] in map:
        id = map[norm[0]]
        _, label, *synonyms = manager.entity_index.get_row(id)

    if with_infos:
        all_infos = manager.get_infos_for_items(
            [identifier],
            manager.entity_info_sparql,
        )
        infos = all_infos.get(identifier, [])

    return Annotation(
        identifier=identifier,
        entity=entity,
        label=label,
        synonyms=synonyms,
        infos=infos,
    )


def annotate(
    managers: list[KgManager],
    kg: str,
    prefix: str,
    words_to_be_annotated: str,
    entity: str,
    state: AnnotationState,
    known: set[str],
    know_before_use: bool = False,
) -> str:
    manager, _ = find_manager(managers, kg)

    sequence = state.article.text[state.annotation_window]
    sequence_length = len(sequence)
    prefix_length = len(prefix)
    annotation_length = len(words_to_be_annotated)

    for i in range(prefix_length):
        if prefix[i] == sequence[i]: continue
        return "prefix doesn't match the original text sequence. prefix:\n" + prefix \
            + "\noriginal text sequence:\n" + sequence[:min(prefix_length, sequence_length)]
    
    for i in range(annotation_length):
        if words_to_be_annotated[i] == sequence[i + prefix_length]: continue
        return "words to be annotated don't match the original text sequence"
    
    start_idx = prefix_length
    end_idx = prefix_length + annotation_length - 1

    # deleting an annotation
    if entity == "None":
        try:
            current = state.annotate(start_idx, end_idx, None)
        except ValueError as e:
            raise FunctionCallException(str(e)) from e

        if current is None:
            raise FunctionCallException(f"Text sequence [{start_idx}, {end_idx}] is not annotated")

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
            return f"Annotated text sequence [{start_idx}: {end_idx}] with entity {entity}"
        else:
            return f"Updated annotation of text sequence [{start_idx}, {end_idx}] from {current.entity} to {entity}"



def input_instructions(state: AnnotationState) -> str:
    instructions = """\
Annotate the following text with entities from the available knowledge graphs. \
If there already are annotations for some words, they are shown in parentheses \
after the word value.

You will be given the full text in the beginning and then you will be given an \
excerpt of the text to annotate. When you're done with a sequence and call the \
function 'next' you will be given the next excerpt of the text to annotate. \
To get the first sequence you need to call 'next'. \n
"""


    instructions += state.format()
    return instructions


def input_and_state(input: Any, config: GraspConfig) -> tuple[str, AnnotationState]:
    try:
        article = Article(**input)
    except Exception as e:
        raise ValueError(
            "Entity linking task input must be a dict with 'data' field"
        ) from e

    task_kwargs = config.task_kwargs.get("cea", {})
    context_rows = task_kwargs.get("context_rows", None)

    annots = AnnotationState(article, context_rows)
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
    assert not example_indices, "Example indices are not supported for entity linking task"

    if fn_name == "annotate":
        return annotate(
            managers,
            fn_args["kg"],
            fn_args["prefix"],
            fn_args["words_to_be_annotated"],
            fn_args["entity"],
            state,
            known,
            config.know_before_use,
        )

    elif fn_name == "show_annotations":
        return state.format(fn_args["only_current_window"])

    elif fn_name == "next":
        return state.next()

    elif fn_name == "stop":
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
