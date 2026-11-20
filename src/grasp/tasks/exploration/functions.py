from pydantic import BaseModel, Field

from grasp.manager import KgManager
from grasp.model import FunctionDefinition
from grasp.utils import FunctionCallException, clip, format_notes


class AddNoteArgs(BaseModel):
    note: str = Field(description="The note to add")


class DeleteNoteArgs(BaseModel):
    num: float = Field(description="The number of the note to delete")


class UpdateNoteArgs(BaseModel):
    num: float = Field(description="The number of the note to update")
    note: str = Field(description="The new note replacing the old one")


class ShowNotesArgs(BaseModel):
    pass


# for knowledge graph specific notes, kg comes first; null for general notes
class KgNoteArgs(BaseModel):
    kg: str | None = Field(default=None, description="The knowledge graph of the note")


class KgAddNoteArgs(AddNoteArgs, KgNoteArgs):
    pass


class KgDeleteNoteArgs(DeleteNoteArgs, KgNoteArgs):
    pass


class KgUpdateNoteArgs(UpdateNoteArgs, KgNoteArgs):
    pass


class KgShowNotesArgs(ShowNotesArgs, KgNoteArgs):
    pass


def note_function_definitions(
    managers: list[KgManager],
    general: bool = True,
    kg_specific: bool = True,
) -> list[FunctionDefinition]:
    assert general or kg_specific, "at least one of general or kg_specific must be True"

    kgs: list[str | None] = [manager.kg for manager in managers]
    if general:
        kgs.append(None)

    if general and kg_specific:
        general_or_kg = "general or knowledge graph specific"
        hint = " (kg null for a general note)"
    elif kg_specific:
        general_or_kg = "knowledge graph specific"
        hint = ""
    else:
        general_or_kg = "general"
        hint = ""

    add, delete, update, show = (
        (KgAddNoteArgs, KgDeleteNoteArgs, KgUpdateNoteArgs, KgShowNotesArgs)
        if kg_specific
        else (AddNoteArgs, DeleteNoteArgs, UpdateNoteArgs, ShowNotesArgs)
    )
    return [
        FunctionDefinition(
            "add_note",
            f"Add a {general_or_kg} note{hint}.",
            add,
            {"kg": kgs},
        ),
        FunctionDefinition(
            "delete_note",
            f"Delete a {general_or_kg} note{hint}.",
            delete,
            {"kg": kgs},
        ),
        FunctionDefinition(
            "update_note",
            f"Update a {general_or_kg} note{hint}.",
            update,
            {"kg": kgs},
        ),
        FunctionDefinition(
            "show_notes",
            f"Show current {general_or_kg} notes{hint}.",
            show,
            {"kg": kgs},
        ),
        FunctionDefinition("stop", "Stop the note taking process."),
    ]


def check_note(note: str, max_note_length: int) -> None:
    if len(note) > max_note_length:
        raise FunctionCallException(
            f"Note with {len(note):,} characters exceeds maximum "
            f"length of {max_note_length:,} characters"
        )


def show_notes(notes: list[str]) -> str:
    return format_notes(notes, enumerated=True)


def add_note(
    name: str,
    notes: list[str],
    note: str,
    max_notes: int,
    max_note_length: int,
) -> str:
    if len(notes) >= max_notes:
        raise FunctionCallException(f"Cannot add more than {max_notes} {name} notes")

    check_note(note, max_note_length)

    notes.append(note)
    return f"Added {name} note {len(notes)}: {clip(note, 64)}"


def delete_note(name: str, notes: list[str], num: int | float) -> str:
    num = int(num)
    if num < 1 or num > len(notes):
        raise FunctionCallException(f"Could not find {name} note {num}")

    num -= 1
    note = notes.pop(num)
    return (
        f"Deleted {name} note {num + 1}: {clip(note, 64)}\n\n"
        f"Notes after {name} note {num + 1} have been shifted up by one position."
    )


def update_note(
    name: str,
    notes: list[str],
    num: int | float,
    note: str,
    max_note_length: int,
) -> str:
    num = int(num)
    if num < 1 or num > len(notes):
        raise FunctionCallException(f"Could not find {name} note {num}")

    check_note(note, max_note_length)

    num -= 1
    notes[num] = note
    return f"Updated {name} note {num + 1}: {clip(note, 64)}"


def call_function(
    kg_notes: dict[str, list[str]],
    notes: list[str],
    fn_name: str,
    fn_args: dict,
    max_notes: int,
    max_note_length: int,
) -> str:
    if fn_name == "stop":
        return "Stopped process"

    # kg is null or missing for general notes
    kg = KgNoteArgs.model_validate(fn_args).kg
    if kg is None:
        notes_to_use = notes
        name = "general"
    else:
        if kg not in kg_notes:
            kg_notes[kg] = []
        notes_to_use = kg_notes[kg]
        name = f'"{kg}"'

    if fn_name == "add_note":
        add_args = AddNoteArgs.model_validate(fn_args)
        return add_note(
            name,
            notes_to_use,
            add_args.note,
            max_notes,
            max_note_length,
        )

    elif fn_name == "delete_note":
        delete_args = DeleteNoteArgs.model_validate(fn_args)
        return delete_note(name, notes_to_use, delete_args.num)

    elif fn_name == "update_note":
        update_args = UpdateNoteArgs.model_validate(fn_args)
        return update_note(
            name,
            notes_to_use,
            update_args.num,
            update_args.note,
            max_note_length,
        )

    elif fn_name == "show_notes":
        return show_notes(notes_to_use)

    else:
        raise ValueError(f"Unknown function {fn_name}")
