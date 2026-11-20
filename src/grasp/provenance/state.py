from dataclasses import dataclass
from typing import Any, Sequence

from pydantic import BaseModel, field_serializer

from grasp.function_args import ArgKind, arg_kinds
from grasp.functions import identifiers, iri_spans
from grasp.manager import KgManager, UnknownKgError, find_manager
from grasp.model import FunctionDefinition, Message, ToolCall
from grasp.provenance.trace import (
    STOPPING,
    Usage,
    arg_usage,
    call_ref,
    iter_tool_calls,
    note_iris,
)


class ProvenanceState(BaseModel):
    # what is stored in outputs and continued in follow-ups: the known items,
    # and the ids of the tool calls that first showed and first used an item
    known: set[str] = set()
    shown: dict[str, str] = {}
    used: dict[str, str] = {}

    @field_serializer("known")
    def sorted_known(self, known: set[str]) -> list[str]:
        return sorted(known)


@dataclass
class TraceCall:
    # a tool call of the messages as dict, with its step
    step: int
    call: dict


# tool calls of the messages by id
Calls = dict[str, TraceCall]


class Provenance:
    def __init__(
        self,
        kg_notes: dict[str, list[str]],
        notes: list[str],
        state: ProvenanceState | None = None,
    ) -> None:
        self.state = state or ProvenanceState()
        # known items with a call that showed them, or known without one,
        # e.g. from the task setup or older follow-ups
        self.attributed = set(self.state.known)
        self.kg_notes = kg_notes
        self.notes = notes
        self.noted_by_kg: dict[str, dict[str, str]] = {}
        # marked arguments by function, the items and queries calls use
        self.kinds: dict[str, dict[str, ArgKind]] = {}

    @property
    def known(self) -> set[str]:
        # passed to the functions, which add the items their results show
        return self.state.known

    def add_functions(self, fns: list[FunctionDefinition]) -> None:
        for fn in fns:
            if fn.args is not None:
                self.kinds[fn.name] = arg_kinds(fn.args)

    def attribute_known(self) -> None:
        # items known so far without a tool call, e.g. from the task setup
        self.attributed |= self.state.known

    def record(
        self,
        managers: list[KgManager],
        tool_call: ToolCall,
        messages: Sequence[Message | dict],
    ) -> dict[str, Any] | None:
        # the provenance of the items a finished call uses, then adds the items it
        # showed and used to the state; None without items or for stopping calls
        new = self.state.known - self.attributed
        try:
            manager = find_manager(managers, tool_call.args.get("kg", ""))
        except UnknownKgError:
            # calls without or with an unknown knowledge graph
            manager = None
        items = None
        if manager is not None and tool_call.name not in STOPPING:
            kinds = self.kinds.get(tool_call.name, {})
            usage = arg_usage(manager, tool_call.args, kinds)
            try:
                items = self.arg_items(manager, usage, kinds, messages)
            except Exception:
                pass
            used, _ = usage
            for iris in used.values():
                for iri in iris:
                    self.state.used.setdefault(iri, tool_call.id)

        for iri in new:
            self.state.shown[iri] = tool_call.id
        self.attributed |= new
        return items

    def query(
        self,
        manager: KgManager,
        sparql: str,
        messages: Sequence[Message | dict],
    ) -> dict[str, Any]:
        # the provenance of the items of a final query, never raises
        try:
            spans = iri_spans(manager, sparql)
        except Exception as e:
            return {"error": f"Failed to find the items of the query: {e}"}

        calls = tool_calls(messages)
        items: dict[str, dict[str, Any]] = {}
        for iri, written, start, end in spans:
            if iri not in items:
                items[iri] = self.item(manager, iri, written, calls)
            items[iri]["spans"].append([start, end])
        return {"items": list(items.values())}

    def arg_items(
        self,
        manager: KgManager,
        usage: Usage,
        kinds: dict[str, ArgKind],
        messages: Sequence[Message | dict],
    ) -> dict[str, Any] | None:
        calls = tool_calls(messages)
        used, spans = usage
        items: dict[str, dict[str, Any]] = {}
        for iri, written, start, end in spans:
            if iri not in items:
                items[iri] = self.item(manager, iri, written, calls)
            items[iri]["spans"].append([start, end])

        # other arguments that are IRIs, e.g. the subject of a list call
        for key, iris in sorted(used.items()):
            if kinds[key] == ArgKind.SPARQL:
                continue
            for iri in sorted(iris):
                if iri not in items:
                    written = manager.format_iri(iri)
                    items[iri] = self.item(manager, iri, written, calls)
                items[iri].setdefault("args", []).append(key)

        return {"items": list(items.values())} if items else None

    def noted(self, manager: KgManager) -> dict[str, str]:
        if manager.kg not in self.noted_by_kg:
            notes = self.kg_notes.get(manager.kg, []) + self.notes
            self.noted_by_kg[manager.kg] = note_iris(manager, notes)
        return self.noted_by_kg[manager.kg]

    def item(
        self,
        manager: KgManager,
        iri: str,
        written: str,
        calls: Calls,
    ) -> dict[str, Any]:
        # an item with its status and the calls that showed and first used it
        ids = identifiers(manager, iri)
        shown = first_call(ids, self.state.shown, calls)
        used = first_call(ids, self.state.used, calls)
        item: dict[str, Any] = {
            "identifier": iri,
            "written": written,
            "label": manager.get_iri_label(iri),
            "spans": [],
        }
        if shown is not None:
            item["shown_by"] = self.ref(manager, iri, shown, calls)
        if used is not None:
            item["first_used_by"] = self.ref(manager, iri, used, calls)

        noted = self.noted(manager)
        note = next((noted[i] for i in ids if i in noted), None)
        if shown is None and ids & self.attributed:
            # known without a tool call that showed it
            item["status"] = "retrieved"
        elif shown is None and note is not None:
            item["status"] = "noted"
            item["note"] = note
        elif shown is None:
            item["status"] = "unverified"
        elif used is None or calls[shown].step < calls[used].step:
            item["status"] = "retrieved"
        else:
            item["status"] = "verified"
        return item

    def ref(
        self,
        manager: KgManager,
        iri: str,
        id: str,
        calls: Calls,
    ) -> dict[str, Any]:
        traced = calls[id]
        kinds = self.kinds.get(traced.call["name"], {})
        usage = arg_usage(manager, traced.call.get("args") or {}, kinds)
        return call_ref(manager, iri, traced.call, traced.step, usage)


def tool_calls(messages: Sequence[Message | dict]) -> Calls:
    return {
        call["id"]: TraceCall(step, call)
        for step, call in enumerate(iter_tool_calls(messages), start=1)
    }


def first_call(ids: set[str], sources: dict[str, str], calls: Calls) -> str | None:
    # the earliest call of the messages among the sources of the identifiers
    found = [sources[i] for i in ids if sources.get(i) in calls]
    return min(found, key=lambda id: calls[id].step, default=None)
