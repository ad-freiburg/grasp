import sys
from typing import Any, Iterator

from pydantic import BaseModel
from universal_ml_utils.table import generate_table

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


class CellAnnotation(Annotation):
    row: int
    column: int


class Table(BaseModel):
    header: list[str]
    data: list[list[str]]
    annotate_rows: list[int] | None = None
    annotate_columns: list[int] | None = None
    description: str | None = None

    @property
    def width(self) -> int:
        return len(self.header)

    @property
    def height(self) -> int:
        return len(self.data)

    def trim(self, context: int) -> tuple["Table", int]:
        if context >= self.height:
            return self, 0
        elif self.annotate_rows is None:
            # all rows are to be annotated
            # context_rows does not apply
            return self, 0

        start = max(0, min(self.annotate_rows) - context)
        end = min(self.height, max(self.annotate_rows) + context + 1)

        trimmed = Table(
            header=self.header,
            data=self.data[start:end],
            annotate_rows=[r - start for r in self.annotate_rows],
            annotate_columns=self.annotate_columns,
        )
        return trimmed, start

    def clean(self) -> "Table":
        def clean(s: str) -> str:
            return " ".join(s.strip().split())

        cleaned = Table(
            header=[clean(h) for h in self.header],
            data=[[clean(cell) for cell in row] for row in self.data],
            annotate_rows=self.annotate_rows,
            annotate_columns=self.annotate_columns,
        )
        return cleaned


class CeaSample(Sample):
    table: Table
    annotations: list[CellAnnotation]

    def input(self) -> Any:
        return self.table.model_dump()

    def queries(self) -> list[str]:
        annots = AnnotationState(self.table)
        return [annots.format()]


class AnnotationState:
    def __init__(self, table: Table, context_rows: int | None = None) -> None:
        assert len(table.header) > 0, "Header must not be empty"
        assert all(len(row) == len(table.header) for row in table.data), (
            "All rows must have the same length as the header"
        )

        if context_rows is None:
            self.table = table
            self.offset = 0
        else:
            trimmed, offset = table.trim(context_rows)
            self.table = trimmed
            self.offset = offset

        self.table = self.table.clean()

        # convert to sets for faster lookup
        self.rows = (
            set(self.table.annotate_rows)
            if self.table.annotate_rows is not None
            else None
        )
        self.cols = (
            set(self.table.annotate_columns)
            if self.table.annotate_columns is not None
            else None
        )

        # map from cell (row, column) to annoation
        self.annotations: dict[tuple[int, int], Annotation] = {}

    def annotate(
        self,
        row: int,
        column: int,
        annotation: Annotation | None,
    ) -> Annotation | None:
        if row < 0 or row >= self.table.height:
            raise ValueError(f"Row {row} out of bounds")

        if self.rows is not None and row not in self.rows:
            raise ValueError(f"Row {row} must not be annotated")

        if column < 0 or column >= self.table.width:
            raise ValueError(f"Column {column} out of bounds")

        if self.cols is not None and column not in self.cols:
            raise ValueError(f"Column {column} must not be annotated")

        current = self.annotations.pop((row, column), None)
        if annotation is not None:
            self.annotations[(row, column)] = annotation
        return current

    def get(self, row: int, column: int) -> Annotation | None:
        return self.annotations.get((row, column), None)

    def to_dict(self) -> dict:
        return {
            "formatted": self.format(),
            "annotations": [
                CellAnnotation(
                    row=row + self.offset,
                    column=column,
                    **annot.model_dump(),
                ).model_dump()
                for (row, column), annot in self.annotations.items()
            ],
        }

    def iter(self) -> Iterator[list[Annotation | None]]:
        for r in range(self.table.height):
            yield [self.get(r, c) for c in range(self.table.width)]

    def format(self) -> str:
        data = [
            [str(i)]
            + [
                col + (f" ({annot.entity})" if annot is not None else "")
                for col, annot in zip(row, annots)
            ]
            for i, (row, annots) in enumerate(zip(self.table.data, self.iter()))
        ]
        header = ["Row"] + [
            f"Column {i}: {name}" for i, name in enumerate(self.table.header)
        ]
        table = generate_table(
            data=data,
            headers=[header],
            max_column_width=sys.maxsize,
        )

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
            table += f"\n\nAnnotated entities:\n{annotations}"

        return table


def rules() -> list[str]:
    return [
        "If you cannot find a suitable entity for a cell, leave it unannotated.",
        "If there are multiple suitable entities for a cell, choose the one that "
        "fits best in the context of the table, or the one that is more popular/general.",
        "If you find common patterns within or across rows and columns, executing a corresponding SPARQL query "
        "to retrieve multiple entities at once might be easier than searching for each cell individually.",
        "If the same entity occurs multiple times in the table, annotate all occurrences.",
        "Before stopping, always check your current annotations.",
    ]


def system_information() -> str:
    return """\
You are an entity annotation assistant. \
Your job is to annotate cells from a given table with entities \
from the available knowledge graphs.

You should follow a step-by-step approach to annotate the cells:
1. Determine what the table might be about and what the different columns \
might represent. Think about how the cells might be represented with entities \
in the knowledge graphs.
2. Annotate the cells, starting with the ones that are easiest to annotate. \
Use the provided functions to search and explore the knowledge graphs. \
You may need to adapt your annotations based on new insights along the way.
3. Use the stop function to finalize your annotations and stop the \
annotation process."""


def functions(managers: list[KgManager]) -> list[dict]:
    kgs = [manager.kg for manager in managers]
    fns = [
        {
            "name": "annotate",
            "description": """\
Annotate a cell in the table with an entity from the specified knowledge graph.
This function overwrites any previous annotation of the cell.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "kg": {
                        "type": "string",
                        "enum": kgs,
                        "description": "The knowledge graph to use for the annotation",
                    },
                    "row": {
                        "type": "integer",
                        "description": "The row index of the cell to annotate (0-based, ignoring header)",
                    },
                    "column": {
                        "type": "integer",
                        "description": "The column index of the cell to annotate (0-based, ignoring header)",
                    },
                    "entity": {
                        "type": "string",
                        "description": "The IRI of the entity to annotate the cell with",
                    },
                },
                "required": ["kg", "row", "column", "entity"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "name": "delete_annotation",
            "description": "Delete the annotation of a cell in the table.",
            "parameters": {
                "type": "object",
                "properties": {
                    "row": {
                        "type": "integer",
                        "description": "The row index of the cell to clear (0-based, ignoring header)",
                    },
                    "column": {
                        "type": "integer",
                        "description": "The column index of the cell to clear (0-based, ignoring header)",
                    },
                },
                "required": ["row", "column"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "name": "show_annotations",
            "description": "Show the current annotations for the table.",
        },
        {
            "name": "stop",
            "description": "Finalize your annotations and stop the annotation process.",
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
    info = infos.get(identifier, {})

    # format normalized identifier again, so always
    # prefixed form is shown if available
    entity = manager.format_iri(identifier)
    # extract fields from info dict
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
    row: int,
    column: int,
    entity: str,
    state: AnnotationState,
    known: set[str],
    know_before_annotate: bool = False,
) -> str:
    manager, _ = find_manager(managers, kg)

    try:
        annotation = prepare_annotation(manager, entity)
        if know_before_annotate and annotation.identifier not in known:
            raise FunctionCallException(
                f"The entity {entity} cannot be used for annotation "
                "without being known from previous function call results. "
                "This does not mean it is invalid, but you should verify "
                "that it indeed exists (e.g., by listing example triples) "
                "in the knowledge graphs first."
            )

        current = state.annotate(row, column, annotation)
    except ValueError as e:
        raise FunctionCallException(str(e)) from e

    if current is None:
        return f"Annotated cell ({row}, {column}) with entity {entity}"
    else:
        return f"Updated annotation of cell ({row}, {column}) from {current.entity} to {entity}"


def delete(row: int, column: int, state: AnnotationState) -> str:
    try:
        current = state.annotate(row, column, None)
    except ValueError as e:
        raise FunctionCallException(str(e)) from e

    if current is None:
        raise FunctionCallException(f"Cell ({row}, {column}) is not annotated")

    return f"Deleted annotation {current.entity} from cell ({row}, {column})"


def input_instructions(state: AnnotationState) -> str:
    instructions = """\
Annotate the following table with entities from the available knowledge graphs. \
If there already are annotations for some cells, they are shown in parentheses \
after the cell value.

"""

    if state.table.description is not None:
        instructions += f"Table description:\n{state.table.description}\n\n"

    if state.rows is not None and len(state.rows) != state.table.height:
        rows_to_annotate = ", ".join(str(r) for r in sorted(state.rows))
        sfx = "s" if len(state.rows) != 1 else ""
        instructions += f"Only annotate row{sfx} {rows_to_annotate}.\n\n"
    else:
        instructions += "Annotate all rows.\n\n"

    if state.cols is not None and len(state.cols) != state.table.width:
        cols_to_annotate = ", ".join(str(c) for c in sorted(state.cols))
        sfx = "s" if len(state.cols) != 1 else ""
        instructions += f"Only annotate column{sfx} {cols_to_annotate}.\n\n"
    else:
        instructions += "Annotate all columns.\n\n"

    instructions += state.format()
    return instructions


def input_and_state(input: Any, config: GraspConfig) -> tuple[str, AnnotationState]:
    try:
        table = Table(**input)
    except Exception as e:
        raise ValueError(
            "CEA task input must be a dict with 'header' and 'data' fields"
        ) from e

    task_kwargs = config.task_kwargs.get("cea", {})
    context_rows = task_kwargs.get("context_rows", None)

    annots = AnnotationState(table, context_rows)
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
    assert isinstance(state, AnnotationState), (
        "Annotations must be provided as state for CEA task"
    )
    assert not example_indices, "Example indices are not supported for CEA task"

    cea_kwargs = config.task_kwargs.get("cea", {})
    know_before_annotate = cea_kwargs.get("know_before_annotate", True)

    if fn_name == "annotate":
        return annotate(
            managers,
            fn_args["kg"],
            fn_args["row"],
            fn_args["column"],
            fn_args["entity"],
            state,
            known,
            know_before_annotate,
        )

    elif fn_name == "delete_annotation":
        return delete(fn_args["row"], fn_args["column"], state)

    elif fn_name == "show_annotations":
        return state.format()

    elif fn_name == "stop":
        return "Stopping"

    else:
        raise ValueError(f"Unknown function {fn_name}")


def feedback_system_message(
    managers: list[KgManager],
    kg_notes: dict[str, list[str]],
    notes: list[str],
) -> str:
    return "\n\n".join(
        [
            "You are a table annotation assistant providing feedback on the "
            "output of a table annotation system for a given input table.",
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


class CeaTask(GraspTask, FeedbackTask):
    name = "cea"

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
        example_indices: dict | None,
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
        return fn_name == "stop"

    def setup(self, input: Any) -> str:
        instructions, self.state = input_and_state(input, self.config)
        return instructions

    def output(self, messages: list[Message]) -> dict:
        return self.state.to_dict()

    @property
    def default_input_field(self) -> str | None:
        return "table"

    @classmethod
    def sample_cls(cls) -> type[CeaSample] | None:
        return CeaSample

    def feedback_system_message(
        self, kg_notes: dict[str, list[str]], notes: list[str]
    ) -> str:
        return feedback_system_message(self.managers, kg_notes, notes)

    def feedback_instructions(self, inputs: list[str], output: dict) -> str:
        return feedback_instructions(inputs, output)
