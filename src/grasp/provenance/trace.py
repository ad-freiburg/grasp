import re
from typing import Any, Iterator, Sequence

from grasp.function_args import ArgKind
from grasp.functions import identifiers, iri_spans
from grasp.manager import KgManager
from grasp.manager.utils import get_common_sparql_prefixes
from grasp.model import Message
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


def iter_tool_calls(messages: Sequence[Message | dict]) -> Iterator[dict]:
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


def arg_usage(
    manager: KgManager,
    args: dict,
    kinds: dict[str, ArgKind],
) -> Usage:
    # the IRIs of the arguments marked as items or queries
    used: dict[str, set[str]] = {}
    sparql_spans = []
    for key, value in flatten_args(args):
        kind = kinds.get(key)
        if kind == ArgKind.SPARQL and isinstance(value, str):
            try:
                sparql_spans = iri_spans(manager, value)
            except Exception:
                pass
            used[key] = {iri for iri, *_ in sparql_spans}
            continue
        elif kind is None:
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
