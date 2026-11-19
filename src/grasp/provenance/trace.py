# Trace provenance: for every knowledge graph item a tool call or a final
# SPARQL query uses, the first earlier tool call that showed it, i.e. added it
# to the known items, and the first earlier tool call that used it in its
# arguments. Items that no tool call showed may come from the notes. Relies on
# the known items recorded per tool call (ToolCall.new_known), so older traces
# have none.

import re
from typing import Any, Iterator

from grasp.functions import find_manager, identifiers, iri_spans
from grasp.manager import KgManager
from grasp.manager.utils import get_common_sparql_prefixes
from grasp.model import Message, ToolCall
from grasp.sparql.utils import load_iri_text_parser, parse_into_binding

COMMON_PREFIXES = get_common_sparql_prefixes()

# tool calls that end the trace, their query is the output
STOPPING = {"answer", "cancel"}

TEXT_PARSER = load_iri_text_parser()
SCHEME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")

# IRIs per argument, and the IRI spans of a SPARQL argument
Usage = tuple[dict[str, set[str]], list[tuple[str, str, int, int]]]


def message_dict(message: Message | dict) -> dict:
    if isinstance(message, Message):
        return message.model_dump()
    return message


def iter_tool_calls(messages: list[Message | dict]) -> Iterator[dict]:
    for message in map(message_dict, messages):
        content = message.get("content")
        if not isinstance(content, dict):
            continue

        for tool_call in content.get("tool_calls") or []:
            if tool_call["name"] in STOPPING:
                continue
            yield tool_call


def note_iris(manager: KgManager, notes: list[str]) -> dict[str, str]:
    # IRIs in the notes, mapped to the first note containing them; read like
    # a query: full IRIs, and prefixed names resolved with the knowledge graph
    # prefixes and the PREFIX declarations in the notes; common vocabulary is
    # left out
    prefixes = {**manager.prefixes}
    terms = []
    for note in notes:
        data = note.encode()
        tokens = [
            (name, start, end)
            for name, (start, end) in TEXT_PARSER.lex(note)
            if name is not None
        ]
        for i, (name, start, end) in enumerate(tokens):
            if name != "PREFIX":
                terms.append((note, data[start:end].decode()))
                continue

            # PREFIX, a prefix and an IRI, separated by whitespace
            decl = tokens[i + 1 : i + 3]
            if [kind for kind, *_ in decl] != ["PNAME_NS", "IRIREF"]:
                continue
            (_, ns_start, ns_end), (_, iri_start, iri_end) = decl
            if data[end:ns_start].strip() or data[ns_end:iri_start].strip():
                continue
            short = data[ns_start : ns_end - 1].decode()
            long = data[iri_start + 1 : iri_end - 1].decode()
            prefixes.setdefault(short, long)

    iris: dict[str, str] = {}
    for note, value in terms:
        if not value.startswith("<") and value.split(":")[0] not in prefixes:
            continue

        binding = parse_into_binding(value, manager.iri_literal_parser, prefixes)
        if (
            binding is None
            or binding.typ != "uri"
            or not SCHEME_PATTERN.match(binding.value)
        ):
            continue

        longest = manager.find_longest_prefix(binding.value)
        if longest is not None and longest[0] in COMMON_PREFIXES:
            continue

        iris.setdefault(binding.value, note.strip())
    return iris


def flatten_args(args: dict, prefix: str = "") -> Iterator[tuple[str, Any]]:
    # nested arguments with dotted keys, as the website shows them
    for key, value in args.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            yield from flatten_args(value, name)
        elif value is not None:
            yield name, value


def arg_usage(manager: KgManager, args: dict) -> Usage:
    # arguments that are no IRIs or SPARQL queries, e.g. search queries, use
    # no IRIs
    used: dict[str, set[str]] = {}
    sparql_spans = []
    for key, value in flatten_args(args):
        if key == "sparql" and isinstance(value, str):
            try:
                sparql_spans = iri_spans(manager, value)
            except Exception:
                pass
            used[key] = {iri for iri, *_ in sparql_spans}
            continue

        used[key] = set()
        for v in value if isinstance(value, list) else [value]:
            if not isinstance(v, str):
                continue
            binding = parse_into_binding(
                v.strip(),
                manager.iri_literal_parser,
                manager.prefixes,
            )
            if binding is not None and binding.typ == "uri":
                used[key].add(binding.value)
    return used, sparql_spans


def result_paths(value: Any, ids: set[str], path: str = "") -> Iterator[str]:
    # dotted paths to the parts of a structured result showing one of the
    # identifiers, e.g. items.0 for a search result or rows.2.city for a
    # table value; parts show an item with an identifier or iri field, or
    # are an IRI value of a table row
    if isinstance(value, dict):
        if (
            value.get("identifier") in ids
            or value.get("iri") in ids
            or (value.get("type") == "uri" and value.get("value") in ids)
        ):
            yield path
        items = value.items()
    elif isinstance(value, list):
        items = enumerate(value)
    else:
        return

    for key, child in items:
        yield from result_paths(child, ids, f"{path}.{key}" if path else str(key))


def shown_part(result: dict, path: str) -> dict[str, Any] | None:
    # the part of a structured result at a path, for showing it: a search
    # item, a table row with the variable of the value, or a shown value
    keys = path.split(".")
    parents = []
    node: Any = result
    try:
        for key in keys:
            parents.append(node)
            node = node[int(key)] if isinstance(node, list) else node[key]
    except (KeyError, IndexError, ValueError, TypeError):
        return None

    if not isinstance(node, dict):
        return None
    elif "identifier" in node:
        return {"type": "item", "item": node}
    elif len(keys) >= 3 and keys[-3] == "rows":
        # the value of a table row, parents are the table, its rows and the row
        table = parents[-3]
        return {
            "type": "row",
            "columns": table.get("columns", []),
            "row": parents[-1],
            "variable": keys[-1],
        }
    elif "value" in node:
        return {"type": "text", "text": node["value"]}
    return None


def call_ref(
    manager: KgManager,
    iri: str,
    call: dict,
    step: int,
    usage: Usage,
) -> dict[str, Any]:
    # the website looks up the arguments of the call by its id
    ids = identifiers(manager, iri)
    used, sparql_spans = usage
    ref: dict[str, Any] = {
        "id": call.get("id"),
        "step": step,
        "tool": call["name"],
        # where the item is in the arguments: argument names, and spans in the
        # SPARQL argument
        "used_args": sorted(key for key, iris in used.items() if ids & iris),
        "sparql_spans": [[start, end] for i, _, start, end in sparql_spans if i in ids],
    }

    # where the item is in the structured result, for showing and
    # highlighting it
    paths = list(result_paths(call.get("result") or {}, ids))
    if paths:
        ref["result_paths"] = paths
        shown = shown_part(call["result"], paths[0])
        if shown is not None:
            ref["shown"] = shown
    elif (example := shown_example(manager, call.get("result") or {}, ids)) is not None:
        ref["result_paths"], ref["shown"] = example
    return ref


def shown_example(
    manager: KgManager,
    result: dict,
    ids: set[str],
) -> tuple[list[str], dict[str, Any]] | None:
    # the first example whose query shows the item, with the query line
    # containing it, for items no selection covers, e.g. in SERVICE clauses
    for i, example in enumerate(result.get("examples") or []):
        sparql = example.get("sparql") or ""
        try:
            spans = iri_spans(manager, sparql, skip_common=False)
        except Exception:
            continue

        start = next((start for iri, _, start, _ in spans if iri in ids), None)
        if start is None:
            continue

        begin = sparql.rfind("\n", 0, start) + 1
        end = sparql.find("\n", start)
        line = sparql[begin : end if end >= 0 else len(sparql)].strip()
        shown = {"type": "example", "question": example.get("question"), "line": line}
        return [f"examples.{i}"], shown
    return None


def item_provenance(
    manager: KgManager,
    iri: str,
    calls: list[dict],
    usages: list[Usage],
    noted: dict[str, str],
) -> dict[str, Any]:
    # status of an item and the calls that showed and first used it, among
    # the given earlier calls
    ids = identifiers(manager, iri)
    shown_at = next(
        (i for i, call in enumerate(calls) if iri in (call.get("new_known") or [])),
        None,
    )
    used_at = next(
        (
            i
            for i, (used, _) in enumerate(usages)
            if any(ids & u for u in used.values())
        ),
        None,
    )

    item: dict[str, Any] = {}
    if shown_at is not None:
        item["shown_by"] = call_ref(
            manager,
            iri,
            calls[shown_at],
            shown_at + 1,
            usages[shown_at],
        )
    if used_at is not None:
        item["first_used_by"] = call_ref(
            manager,
            iri,
            calls[used_at],
            used_at + 1,
            usages[used_at],
        )

    note = next((noted[i] for i in ids if i in noted), None)
    if shown_at is None and note is not None:
        item["status"] = "noted"
        item["note"] = note
    elif shown_at is None:
        item["status"] = "unverified"
    elif used_at is None or shown_at < used_at:
        item["status"] = "retrieved"
    else:
        item["status"] = "verified"
    return item


def new_item(
    manager: KgManager,
    iri: str,
    written: str,
    calls: list[dict],
    usages: list[Usage],
    noted: dict[str, str],
) -> dict[str, Any]:
    return {
        "identifier": iri,
        "written": written,
        "label": manager.get_iri_label(iri),
        "spans": [],
        **item_provenance(manager, iri, calls, usages, noted),
    }


def recorded(calls: list[dict]) -> bool:
    # older traces have no known items per tool call
    return not calls or any(call.get("new_known") is not None for call in calls)


def trace_provenance(
    manager: KgManager,
    sparql: str,
    messages: list[Message | dict],
    notes: list[str],
) -> dict[str, Any] | None:
    # the items of a final query, compared with all tool calls; None for
    # older traces, never raises
    calls = list(iter_tool_calls(messages))
    if not recorded(calls):
        return None

    try:
        spans = iri_spans(manager, sparql)
    except Exception as e:
        return {"error": f"Failed to find the items of the query: {e}"}

    usages = [arg_usage(manager, call.get("args") or {}) for call in calls]
    noted = note_iris(manager, notes)

    items: dict[str, dict[str, Any]] = {}
    for iri, written, start, end in spans:
        if iri not in items:
            items[iri] = new_item(manager, iri, written, calls, usages, noted)
        items[iri]["spans"].append([start, end])
    return {"items": list(items.values())}


def call_provenance(
    managers: list[KgManager],
    messages: list[Message | dict],
    tool_call: ToolCall,
    kg_notes: dict[str, list[str]],
    notes: list[str],
) -> dict[str, Any] | None:
    # the items in the arguments of a tool call, compared with the tool calls
    # before it; None for calls without a knowledge graph or items, stopping
    # calls whose query is the output, and older traces; never raises
    kg = tool_call.args.get("kg")
    if tool_call.name in STOPPING or not isinstance(kg, str):
        return None

    try:
        manager, _ = find_manager(managers, kg)
        calls = list(iter_tool_calls(messages))
        index = next(i for i, call in enumerate(calls) if call["id"] == tool_call.id)
        if not recorded(calls[:index]):
            return None

        usages = [
            arg_usage(manager, call.get("args") or {}) for call in calls[: index + 1]
        ]
        noted = note_iris(manager, kg_notes.get(kg, []) + notes)
        calls, (used, spans), usages = calls[:index], usages[index], usages[:index]

        items: dict[str, dict[str, Any]] = {}
        for iri, written, start, end in spans:
            if iri not in items:
                items[iri] = new_item(manager, iri, written, calls, usages, noted)
            items[iri]["spans"].append([start, end])

        # other arguments that are IRIs, e.g. the subject of a list call
        for key, iris in sorted(used.items()):
            if key == "sparql":
                continue
            for iri in sorted(iris):
                if iri not in items:
                    written = manager.format_iri(iri)
                    items[iri] = new_item(manager, iri, written, calls, usages, noted)
                items[iri].setdefault("args", []).append(key)
    except Exception:
        return None

    return {"items": list(items.values())} if items else None
