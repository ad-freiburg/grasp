# Trace provenance: for every knowledge graph item in a final SPARQL query, the
# first tool call that showed it, i.e. added it to the known items, and the
# first tool call that used it in its arguments. Relies on the known items
# recorded per tool call (ToolCall.new_known), so older traces have none.
# Items that no tool call showed may still come from the instruction, e.g.
# from its notes.

import re
from typing import Any, Iterator

from grasp.functions import identifiers, iri_spans
from grasp.manager import KgManager
from grasp.manager.utils import get_common_sparql_prefixes
from grasp.model import Message
from grasp.sparql.utils import parse_into_binding

COMMON_PREFIXES = get_common_sparql_prefixes()

# tool calls that end the trace and do not provide information
STOPPING = {"answer", "cancel"}

# IRIs and prefixed names in text, following the terminals IRIREF, PNAME_NS
# and PNAME_LN of the SPARQL grammar
_BASE = (
    "A-Za-z\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u02ff\u0370-\u037d"
    "\u037f-\u1fff\u200c-\u200d\u2070-\u218f\u2c00-\u2fef"
    "\u3001-\ud7ff\uf900-\ufdcf\ufdf0-\ufffd\U00010000-\U000effff"
)
_CHARS = _BASE + "_\\-0-9\u00b7\u0300-\u036f\u203f-\u2040"
_PLX = r"%[0-9A-Fa-f]{2}|\\[_~.\-!$&'()*+,;=/?#@%]"
_PN_PREFIX = f"[{_BASE}](?:[{_CHARS}.]*[{_CHARS}])?"
_PN_LOCAL = (
    f"(?:[{_BASE}_:0-9]|{_PLX})(?:(?:[{_CHARS}.:]|{_PLX})*(?:[{_CHARS}:]|{_PLX}))?"
)
_IRIREF = r'<[^<>"{}|^`\\\x00-\x20]*>'
TERM_PATTERN = re.compile(
    f"{_IRIREF}|(?<![{_CHARS}.:])(?:{_PN_PREFIX})?:(?:{_PN_LOCAL})?"
)
PREFIX_PATTERN = re.compile(
    f"PREFIX\\s+((?:{_PN_PREFIX})?):\\s*<([^<>\\s]*)>",
    re.IGNORECASE,
)
SCHEME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")


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


def instruction_iris(
    manager: KgManager, messages: list[Message | dict]
) -> dict[str, str]:
    # IRIs in the system messages, mapped to the first line containing them;
    # read like a query: full IRIs, and prefixed names resolved with the
    # common and knowledge graph prefixes and the PREFIX lines of the text;
    # common vocabulary is left out
    iris: dict[str, str] = {}
    for message in map(message_dict, messages):
        content = message.get("content")
        if message.get("role") != "system" or not isinstance(content, str):
            continue

        prefixes = {**manager.prefixes}
        for short, long in PREFIX_PATTERN.findall(content):
            prefixes.setdefault(short, long)

        for line in content.splitlines():
            for match in TERM_PATTERN.finditer(line):
                value = match.group()
                if not value.startswith("<") and value.split(":")[0] not in prefixes:
                    continue

                binding = parse_into_binding(
                    value, manager.iri_literal_parser, prefixes
                )
                if (
                    binding is None
                    or binding.typ != "uri"
                    or not SCHEME_PATTERN.match(binding.value)
                ):
                    continue

                longest = manager.find_longest_prefix(binding.value)
                if longest is not None and longest[0] in COMMON_PREFIXES:
                    continue

                iris.setdefault(binding.value, line.strip())
    return iris


def flatten_args(args: dict, prefix: str = "") -> Iterator[tuple[str, Any]]:
    # nested arguments with dotted keys, as the website shows them
    for key, value in args.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            yield from flatten_args(value, name)
        elif value is not None:
            yield name, value


def arg_usage(
    manager: KgManager,
    args: dict,
) -> tuple[dict[str, set[str]], list[tuple[str, str, int, int]]]:
    # IRIs per argument, and the IRI spans of a SPARQL argument; arguments
    # that are no IRIs or SPARQL queries, e.g. search queries, use none
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
                v.strip(), manager.iri_literal_parser, manager.prefixes
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
    usage: tuple[dict[str, set[str]], list[tuple[str, str, int, int]]],
) -> dict[str, Any]:
    ids = identifiers(manager, iri)
    used, sparql_spans = usage
    ref: dict[str, Any] = {
        "id": call.get("id"),
        "step": step,
        "tool": call["name"],
        "args": call.get("args"),
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


def trace_provenance(
    manager: KgManager,
    sparql: str,
    messages: list[Message | dict],
) -> dict[str, Any] | None:
    """
    Per knowledge graph item of the query, its spans in the query, the first
    tool call that added it to the known items and the first tool call that
    used it in its arguments. None if the trace has no known items recorded.
    Never raises.
    """
    calls = list(iter_tool_calls(messages))
    if not any(call.get("new_known") is not None for call in calls):
        return None

    try:
        spans = iri_spans(manager, sparql)
    except Exception as e:
        return {"error": f"Failed to find the items of the query: {e}"}

    usages = [arg_usage(manager, call.get("args") or {}) for call in calls]
    instructed = instruction_iris(manager, messages)

    items: dict[str, dict[str, Any]] = {}
    for iri, written, start, end in spans:
        if iri in items:
            items[iri]["spans"].append([start, end])
            continue

        item: dict[str, Any] = {
            "identifier": iri,
            "written": written,
            "label": manager.get_iri_label(iri),
            "spans": [[start, end]],
        }

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

        if shown_at is not None:
            item["shown_by"] = call_ref(
                manager, iri, calls[shown_at], shown_at + 1, usages[shown_at]
            )
        if used_at is not None:
            item["first_used_by"] = call_ref(
                manager, iri, calls[used_at], used_at + 1, usages[used_at]
            )

        instruction = next((instructed[i] for i in ids if i in instructed), None)
        if shown_at is None and instruction is not None:
            # no tool result showed it, but the instruction contains it
            item["status"] = "instructed"
            item["instruction"] = {"text": instruction}
        elif shown_at is None:
            item["status"] = "unverified"
        elif used_at is None or shown_at < used_at:
            item["status"] = "retrieved"
        else:
            item["status"] = "verified"

        items[iri] = item

    return {"items": list(items.values())}
