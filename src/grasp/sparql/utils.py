import time
import uuid
from copy import deepcopy
from typing import Any, Iterator
from urllib.parse import urlparse, urlunparse

import ijson
import requests
from grammar_utils.parse import LR1Parser  # type: ignore
from ijson.common import ObjectBuilder
from requests.exceptions import JSONDecodeError

from grasp.sparql.types import AskResult, Binding, Position, SelectResult
from grasp.utils import read_resource

# default request timeout
# 6 seconds for establishing a connection, 30 seconds for processing query
# and beginning to receive the response
REQUEST_TIMEOUT = (6, 30)

# default read timeout
# if you cannot read the full response in 10 seconds, it is likely too large
READ_TIMEOUT = 10

QLEVER_API = "https://qlever.dev/api"


def get_qlever_endpoint(kg: str) -> str:
    return f"{QLEVER_API}/{kg}"


class SPARQLException(Exception):
    def __init__(self, message: str, query: str | None = None) -> None:
        super().__init__(message)
        self.query = query


def load_sparql_grammar() -> tuple[str, str]:
    sparql_grammar = read_resource("grasp.sparql.grammar", "sparql.y")
    sparql_lexer = read_resource("grasp.sparql.grammar", "sparql.l")
    return sparql_grammar, sparql_lexer


def load_sparql_parser() -> LR1Parser:
    sparql_grammar, sparql_lexer = load_sparql_grammar()
    return LR1Parser(sparql_grammar, sparql_lexer)


def load_iri_and_literal_grammar() -> tuple[str, str]:
    il_grammar = read_resource("grasp.sparql.grammar", "iri_literal.y")
    il_lexer = read_resource("grasp.sparql.grammar", "iri_literal.l")
    return il_grammar, il_lexer


def load_iri_and_literal_parser() -> LR1Parser:
    iri_and_literal_grammar, iri_and_literal_lexer = load_iri_and_literal_grammar()
    return LR1Parser(iri_and_literal_grammar, iri_and_literal_lexer)


def find_longest_prefix(iri: str, prefixes: dict[str, str]) -> tuple[str, str] | None:
    longest = None
    for short, long in prefixes.items():
        if not iri.startswith(long):
            continue
        if longest is None or len(long) > len(longest[1]):
            longest = short, long
    return longest


def strip_literal(s: str) -> str:
    if s.startswith('"') and s.endswith('"'):
        s = s.strip('"')
    elif s.startswith("'") and s.endswith("'"):
        s = s.strip("'")
    return s


def parse_into_binding(
    input: str,
    parser: LR1Parser,
    prefixes: dict[str, str] | None = None,
) -> Binding | None:
    try:
        parse, _ = parse_string(
            input,
            parser,
            skip_empty=True,
            collapse_single=True,
        )
    except Exception:
        return None

    match parse["name"]:
        case "BLANK_NODE_LABEL":
            return Binding(
                typ="bnode",
                # strip leading "_:" from blank node label
                value=input[2:],
            )

        case "IRIREF":
            return Binding(
                typ="uri",
                value=input[1:-1],
            )

        case "PNAME_LN" | "PNAME_NS":
            pfx, name = input.split(":", 1)
            if prefixes is None or pfx not in prefixes:
                return None

            uri = prefixes[pfx] + name

            # prefixed IRI
            return Binding(
                typ="uri",
                value=uri,
            )

        case lit if lit.startswith("STRING_LITERAL"):
            # string literal -> strip quotes
            return Binding(
                typ="literal",
                value=strip_literal(parse["value"]),
            )

        case lit if lit.startswith("INTEGER"):
            # integer literal
            return Binding(
                typ="literal",
                value=parse["value"],
                datatype="http://www.w3.org/2001/XMLSchema#int",
            )

        case lit if lit.startswith("DECIMAL"):
            # decimal literal
            return Binding(
                typ="literal",
                value=parse["value"],
                datatype="http://www.w3.org/2001/XMLSchema#decimal",
            )

        case lit if lit.startswith("DOUBLE"):
            # double literal
            return Binding(
                typ="literal",
                value=parse["value"],
                datatype="http://www.w3.org/2001/XMLSchema#double",
            )

        case lit if lit in ("true", "false"):
            # boolean literal
            return Binding(
                typ="literal",
                value=parse["value"],
                datatype="http://www.w3.org/2001/XMLSchema#boolean",
            )

        case "RDFLiteral":
            if len(parse["children"]) == 2:
                # langtag
                lit, langtag = parse["children"]

                return Binding(
                    typ="literal",
                    value=strip_literal(lit["value"]),
                    lang=langtag["value"][1:],
                )

            elif len(parse["children"]) == 3:
                # datatype
                lit, _, datatype = parse["children"]
                if datatype["name"] == "IRIREF":
                    datatype = datatype["value"][1:-1]
                else:
                    pfx, name = datatype["value"].split(":", 1)
                    if prefixes is None or pfx not in prefixes:
                        return None

                    datatype = prefixes[pfx] + name

                return Binding(
                    typ="literal",
                    value=strip_literal(lit["value"]),
                    datatype=datatype,
                )

        case other:
            raise ValueError(
                f"Unexpected type {other} for IRI or literal: {input}",
            )


def parse_to_string(parse: dict) -> str:
    def _flatten(parse: dict) -> str:
        if "value" in parse:
            return parse["value"]
        elif "children" in parse:
            children = []
            for p in parse["children"]:
                child = _flatten(p)
                if child != "":
                    children.append(child)
            return " ".join(children)
        else:
            return ""

    return _flatten(parse)


def parse_to_string_with_whitespace(parse: dict, encoded: bytes) -> str:
    # rebuild string from (sub-)parse tree, preserving original whitespace
    # between terminals using byte_span from the encoded input. only emits
    # bytes that lie between the first and last surviving terminal, so the
    # function is safe to call on a subtree as well as on the root parse.
    parts: list[str] = []
    pos: int | None = None
    for terminal in find_terminals(parse):
        byte_span = terminal.get("byte_span")
        if byte_span is not None:
            start, end = byte_span
            if pos is not None:
                # copy original bytes (whitespace) between previous and this terminal
                parts.append(encoded[pos:start].decode(errors="replace"))
            pos = end
        elif parts:
            # newly created terminal without byte_span, add a space separator
            parts.append(" ")
        parts.append(terminal["value"])
    return "".join(parts)


def parse_string(
    input: str,
    parser: LR1Parser,
    collapse_single: bool = False,
    skip_empty: bool = False,
    is_prefix: bool = False,
) -> tuple[dict, str]:
    if is_prefix:
        parse, rest = parser.prefix_parse(
            input.encode(),
            skip_empty=skip_empty,
            collapse_single=collapse_single,
        )
        rest_str = bytes(rest).decode(errors="replace")
    else:
        parse = parser.parse(
            input,
            skip_empty=skip_empty,
            collapse_single=collapse_single,
        )
        rest_str = ""

    return parse, rest_str


def find(
    parse: dict,
    name: str | set[str],
    skip: set[str] | None = None,
    last: bool = False,
) -> dict | None:
    all = find_all(parse, name, skip)
    if not last:
        return next(all, None)
    else:
        last_item = None
        for item in all:
            last_item = item
        return last_item


def find_all(
    parse: dict,
    name: str | set[str],
    skip: set[str] | None = None,
) -> Iterator[dict]:
    if skip is not None and parse["name"] in skip:
        return
    elif isinstance(name, str) and parse["name"] == name:
        yield parse
    elif isinstance(name, set) and parse["name"] in name:
        yield parse
    else:
        for child in parse.get("children", []):
            yield from find_all(child, name, skip)


def find_terminals(parse: dict) -> Iterator[dict]:
    if "value" in parse:
        yield parse
    else:
        for child in parse.get("children", []):
            yield from find_terminals(child)


def span(parse: dict) -> tuple[int, int] | None:
    min = max = None
    for terminal in find_terminals(parse):
        span = terminal.get("byte_span")
        if span is None:
            continue

        if min is None or span[0] < min:
            min = span[0]
        if max is None or span[1] > max:
            max = span[1]

    if min is not None and max is not None:
        return min, max
    else:
        return None


def remove_node(node: dict) -> None:
    # remove a node from the parse tree while preserving its byte span,
    # so that parse_to_string_with_whitespace skips the original bytes
    s = span(node)
    node.pop("children", None)
    if s is not None:
        node["value"] = ""
        node["byte_span"] = s


def normalize(sparql: str, parser: LR1Parser, is_prefix: bool = False) -> str:
    # normalize SPARQL by changing variable names to ?v1, ?v2, ...
    parse, rest = parse_string(
        sparql + " " * is_prefix,
        parser,
        skip_empty=True,
        collapse_single=True,
        is_prefix=is_prefix,
    )

    var_rename = {}
    for var in find_all(parse, {"VAR1", "VAR2"}):
        var_name = var["value"][1:]  # remove ? or $
        if var_name not in var_rename:
            var_rename[var_name] = len(var_rename) + 1

        var["value"] = f"?v{var_rename[var_name]}"

    if is_prefix and rest and rest[-1] == " ":
        rest = rest[:-1]

    return parse_to_string(parse) + rest


def complete_prefix(
    prefix: str,
    parser: LR1Parser,
) -> tuple[dict, Position, str]:
    # autocomplete by adding 1 to 3 variables to the query,
    # completing and then parsing it to find the current position
    # in the query triple block
    parse, rest = parse_string(
        prefix + " ",
        parser,
        is_prefix=True,
    )

    # build bracket stack to fix brackets later
    bracket_stack = []
    for item in find_all(
        parse,
        {"{", "}", "(", ")"},
    ):
        if item["name"] in ["{", "("]:
            bracket_stack.append(item["name"])
            continue

        assert bracket_stack, "bracket stack is empty"
        last = bracket_stack[-1]
        expected = "(" if item["name"] == ")" else "{"
        assert last == expected, (
            f"expected {expected} bracket in the stack but got {last}"
        )
        bracket_stack.pop()

    if rest in ["{", "("]:
        bracket_stack.append(rest)

    def close_brackets(s: str) -> str:
        for b in reversed(bracket_stack):
            if b == "{":
                s += " }"
            else:
                s += " )"
        return s

    for i, position in enumerate(Position):
        vars = [uuid.uuid4().hex for _ in range(3 - i)]
        current_var = f"?{vars[0]}"

        full_query = prefix.strip() + " " + " ".join(f"?{v}" for v in vars)
        full_query = close_brackets(full_query)

        # check if query is valid now
        try:
            parse, _ = parse_string(full_query, parser)
        except Exception:
            continue

        if var_in_triple(parse, current_var):
            return parse, position, current_var

    raise SPARQLException("Failed to complete SPARQL prefix", prefix)


def var_in_triple(parse: dict, var: str) -> bool:
    return any(
        any(
            child.get("value") == var
            for v in find_all(triple, "Var")
            for child in v.get("children", [])
        )
        for triple in find_all(parse, "TriplesSameSubjectPath")
    )


def var_only_triple(triple: dict) -> bool:
    return len(list(find_all(triple, "Var"))) == 3


def infer_position_from_prefix(prefix: str, parser: LR1Parser) -> Position:
    _, position, _ = complete_prefix(prefix, parser)
    return position


def find_connected_top_level_triple_nodes(parse: dict, select_var: str) -> list[dict]:
    triple_blocks = list(
        find_all(
            parse,
            "TriplesSameSubjectPath",
            skip={"GraphPatternNotTriples"},
        )
    )

    def find_vars(sub_parse: dict) -> set[str]:
        variables = set()
        for var in find_all(sub_parse, "Var"):
            children = var.get("children", [])
            assert len(children) == 1, "Expected Var node to have exactly one child"
            token = children[0]
            value = token.get("value")
            assert isinstance(value, str) and value.startswith("?"), (
                f"Expected variable token value, got {value!r}"
            )
            variables.add(value)
        return variables

    triple_var_sets = [find_vars(block) for block in triple_blocks]
    if select_var not in set().union(*triple_var_sets):
        return []

    keep = set()
    reachable_vars = {select_var}
    changed = True
    while changed:
        changed = False
        for i, var_set in enumerate(triple_var_sets):
            if i in keep or not reachable_vars.intersection(var_set):
                continue

            keep.add(i)
            reachable_vars.update(var_set)
            changed = True

    return [triple_blocks[i] for i in sorted(keep)]


def find_connected_top_level_triples(parse: dict, select_var: str) -> list[str]:
    return [
        parse_to_string(node)
        for node in find_connected_top_level_triple_nodes(parse, select_var)
    ]


def derive_constraint_query_from_prefix(
    prefix: str,
    parser: LR1Parser,
    limit: int | None = None,
) -> tuple[str | None, Position]:
    parse, position, select_var = complete_prefix(prefix, parser)

    triple_nodes = find_connected_top_level_triple_nodes(parse, select_var)

    if not triple_nodes:
        return None, position

    if len(triple_nodes) == 1 and var_only_triple(triple_nodes[0]):
        return None, position

    triple_blocks = [parse_to_string(node) for node in triple_nodes]
    final_query = (
        f"SELECT DISTINCT {select_var} WHERE {{ " + " . ".join(triple_blocks) + " }"
    )
    if limit is not None:
        final_query += f" LIMIT {limit}"

    return final_query, position


def query_type(sparql: str, parser: LR1Parser, is_prefix: bool = False) -> str | None:
    try:
        parse, _ = parse_string(sparql + " " * is_prefix, parser, is_prefix=is_prefix)
    except Exception:
        return None

    query_type = find(parse, "QueryType")
    if query_type is not None:
        name = query_type["children"][0]["name"]
        return name[:-5].lower()  # remove "Query" suffix

    # Prefix parse trees don't build the QueryType wrapper node; infer from the
    # outermost keyword token (CONSTRUCT/DESCRIBE/ASK can't appear in subselects)
    if find(parse, "CONSTRUCT") is not None:
        return "construct"
    if find(parse, "DESCRIBE") is not None:
        return "describe"
    if find(parse, "ASK") is not None:
        return "ask"
    return "select"


def ask_to_select(
    sparql: str,
    parser: LR1Parser,
    limit: int | None = None,
) -> str | None:
    parse = parser.parse(sparql)

    sub_parse = find(parse, "QueryType")
    assert sub_parse is not None

    ask_query = sub_parse["children"][0]
    if ask_query["name"] != "AskQuery":
        return None

    # find all triples
    triples = list(find_all(ask_query, "TriplesSameSubjectPath"))
    for triple in triples:
        # find first var in triple
        var = find(triple, "Var")
        if var is not None:
            continue

        iri = find(triple, "iri")
        assert iri is not None

        # triple block does not have a var
        # introduce one in VALUES clause and replace iri with var
        var = uuid.uuid4().hex
        triple["children"].append(
            {
                "name": "ValuesClause",
                "children": [
                    {"name": "VALUES", "value": "VALUES"},
                    {"name": "Var", "value": f"?{var}"},
                    {"name": "{", "value": "{"},
                    deepcopy(iri),
                    {"name": "}", "value": "}"},
                ],
            }
        )
        iri.pop("children")
        iri["name"] = "Var"
        iri["value"] = f"?{var}"

    # ask query has a var, convert to select
    ask_query["name"] = "SelectQuery"
    # replace ASK terminal with SelectClause
    ask_query["children"][0] = {
        "name": "SelectClause",
        "children": [
            {"name": "SELECT", "value": "SELECT"},
            {"name": "*", "value": "*"},
        ],
    }
    # return if no limit is to be added
    if not limit:
        return parse_to_string(parse)

    limit_clause = find(ask_query, "LimitClause", skip={"SubSelect"})
    if limit_clause is None:
        return parse_to_string(parse) + " LIMIT 1"
    else:
        limit_clause["children"] = [
            {
                "name": "LIMIT",
                "value": "LIMIT",
            },
            {
                "name": "INTEGER",
                "value": "1",
            },
        ]
        return parse_to_string(parse)


def fix_prefixes(
    sparql: str,
    sparql_parser: LR1Parser,
    iri_parser: LR1Parser,
    prefixes: dict[str, str],
    remove_known: bool = False,
    sort: bool = False,
) -> str:
    parse, rest = parse_string(sparql, sparql_parser)

    reverse_prefixes = {long: short for short, long in prefixes.items()}

    exist = {}
    for prefix_decl in find_all(parse, "PrefixDecl"):
        assert len(prefix_decl["children"]) == 3
        first = prefix_decl["children"][1]["value"]
        second = prefix_decl["children"][2]["value"]

        short = first.split(":", 1)[0]
        long = second[1:-1]
        exist[short] = long

    base_decl = find(parse, "BaseDecl", last=True)
    if base_decl:
        base_uri = base_decl["children"][1]["value"][1:-1]
    else:
        base_uri = None

    skip = {"Prologue", "PrefixDecl", "BaseDecl"}

    seen = set()
    for iri in find_all(parse, "IRIREF", skip=skip):
        formatted = format_iri(
            iri["value"],
            iri_parser,
            prefixes,
            base_uri=base_uri,
            wrap=True,
        )
        if is_iri(formatted):
            continue

        pfx, _ = formatted.split(":", 1)
        iri["value"] = formatted
        iri["name"] = "PNAME_LN"
        seen.add(pfx)

    for pfx in find_all(parse, {"PNAME_NS", "PNAME_LN"}, skip=skip):
        short, val = pfx["value"].split(":", 1)

        long = exist.get(short, "")
        if reverse_prefixes.get(long, short) != short:
            short = reverse_prefixes[long]

        pfx["value"] = f"{short}:{val}"
        seen.add(short)

    updated_prologue = []
    for pfx in seen:
        if pfx in prefixes:
            if remove_known:
                continue
            long = prefixes[pfx]
        elif pfx in exist:
            long = exist[pfx]
        else:
            continue

        updated_prologue.append(
            {
                "name": "PrefixDecl",
                "children": [
                    {"name": "PREFIX", "value": "PREFIX"},
                    {"name": "PNAME_NS", "value": f"{pfx}:"},
                    {"name": "IRIREF", "value": wrap_iri(long)},
                ],
            }
        )

    if sort:
        updated_prologue.sort(key=lambda pfx: pfx["children"][1]["value"])

    # build prologue string with newline-separated prefix declarations
    prologue_str = "\n".join(parse_to_string(decl) for decl in updated_prologue)

    # remove original prologue from tree so it's not included in body reconstruction
    encoded = sparql.encode()
    prologue = find(parse, "Prologue")
    if prologue:
        remove_node(prologue)

    # rebuild body preserving original whitespace
    body = parse_to_string_with_whitespace(parse, encoded)

    result = prologue_str.strip() + "\n" + body.strip()
    return (result + rest).strip()


def prettify(
    sparql: str,
    parser: LR1Parser,
    indent: int = 2,
    is_prefix: bool = False,
) -> str:
    parse, rest = parse_string(
        sparql + " " * is_prefix,
        parser,
        skip_empty=True,
        is_prefix=is_prefix,
    )

    # some simple rules for prettifing:
    # 1. new lines after prologue (PrologueDecl) and triple blocks
    # (TriplesBlock)
    # 2. new lines after { and before }
    # 3. increase indent after { and decrease before }

    assert indent > 0, "indent step must be positive"
    current_indent = 0
    s = ""
    last = None

    def _pretty(parse: dict) -> bool:
        nonlocal current_indent
        nonlocal s
        nonlocal last
        newline = False

        if parse["name"] == "(" or (last and last["name"] == "("):
            s = s.rstrip()
        elif parse["name"] == ")":
            s = s.rstrip()

        if "value" in parse:
            if parse["name"] in ["UNION", "MINUS"]:
                s = s.rstrip() + " "

            elif parse["name"] == "}":
                current_indent -= indent
                s = s.rstrip()
                s += "\n" + " " * current_indent

            elif parse["name"] == "{":
                current_indent += indent

            s += parse["value"]

        elif len(parse["children"]) == 1:
            newline = _pretty(parse["children"][0])

        else:
            for i, child in enumerate(parse["children"]):
                if i > 0 and not newline:  # and child["name"] != "(":
                    s += " "

                newline = _pretty(child)

        if not newline and parse["name"] in [
            "{",
            "}",
            ".",
            "PrefixDecl",
            "BaseDecl",
            "TriplesBlock",
            "GraphPatternNotTriples",
            "GroupClause",
            "HavingClause",
            "OrderClause",
            "LimitClause",
            "OffsetClause",
        ]:
            s += "\n" + " " * current_indent
            newline = True

        last = parse

        return newline

    newline = _pretty(parse)
    if newline:
        s = s.rstrip()

    return (s.strip() + " " + rest).strip()


class SPARQLExecuteException(SPARQLException):
    def __init__(
        self,
        message: str,
        query: str,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message, query)
        self.status_code = status_code

    @property
    def is_other_error(self) -> bool:
        return self.status_code is None

    @property
    def is_client_error(self) -> bool:
        return self.status_code is not None and int(self.status_code / 100) == 4

    @property
    def is_server_error(self) -> bool:
        return self.status_code is not None and int(self.status_code / 100) == 5


def _stream_with_timeout(
    sparql: str,
    response: requests.Response,
    seconds: float | None = None,
    sparql_result_max_rows: int | None = None,
) -> Any:
    start = time.monotonic()

    class Stream:
        def __init__(self, response: requests.Response) -> None:
            self.stream = response.iter_content(chunk_size=None)

        def read(self, n: int) -> bytes:
            if n == 0:
                return b""

            chunk = next(self.stream, b"")
            if chunk and seconds and time.monotonic() - start > seconds:
                raise SPARQLExecuteException(
                    f"Took longer than {seconds} seconds to read SPARQL result",
                    sparql,
                )
            return chunk

    variables: list[str] = []
    bindings: list[dict] = []
    row_count = 0
    row_builder: ObjectBuilder | None = None
    row_depth = 0

    try:
        for prefix, event, value in ijson.parse(Stream(response)):
            if prefix == "head.vars.item" and event == "string":
                variables.append(value)
                continue

            if prefix == "boolean" and event == "boolean":
                return {"boolean": value}

            if prefix == "results.bindings.item" and event == "start_map":
                row_count += 1
                if (
                    sparql_result_max_rows is not None
                    and row_count > sparql_result_max_rows
                ):
                    raise SPARQLExecuteException(
                        f"SPARQL result exceeded {sparql_result_max_rows:,} rows",
                        sparql,
                    )

                row_builder = ObjectBuilder()
                row_builder.event(event, value)
                row_depth = 1
                continue

            if row_builder is None:
                continue

            row_builder.event(event, value)
            if event in ("start_map", "start_array"):
                row_depth += 1
            elif event in ("end_map", "end_array"):
                row_depth -= 1

            if row_depth == 0:
                bindings.append(row_builder.value)  # type: ignore
                row_builder = None

        return {"head": {"vars": variables}, "results": {"bindings": bindings}}

    finally:
        response.close()


def execute(
    sparql: str,
    endpoint: str,
    request_timeout: float | tuple[float, float] | None = REQUEST_TIMEOUT,
    read_timeout: float | None = READ_TIMEOUT,
    max_retries: int = 0,
    headers: dict[str, str] | None = None,
    params: dict[str, str] | None = None,
    sparql_result_max_rows: int | None = None,
    **kwargs: Any,
) -> SelectResult | AskResult:
    if headers is None:
        headers = {}
    if params is None:
        params = {}

    max_retries = max(0, max_retries)
    for i in range(max_retries + 1):
        try:
            response = requests.post(
                endpoint,
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/sparql-results+json",
                    "User-Agent": "grasp-rdf",
                    **headers,
                },
                data={**params, "query": sparql},
                timeout=request_timeout,
                stream=True,
                **kwargs,
            )

            response.raise_for_status()

            res = _stream_with_timeout(
                sparql,
                response,
                read_timeout,
                sparql_result_max_rows,
            )
            if "boolean" in res:
                return AskResult(res["boolean"])
            else:
                return SelectResult.from_json(res)

        except SPARQLExecuteException as e:
            # retry if not last retry
            if i < max_retries:
                continue

            raise e

        except requests.Timeout as e:
            # retry if not last retry
            if i < max_retries:
                continue

            # format timeout
            if request_timeout is None:
                timeout_fmt = ""
            elif isinstance(request_timeout, tuple):
                conn_tm, query_tm = request_timeout
                timeout_fmt = (
                    f" (connection_timeout={conn_tm}s, query_timeout={query_tm}s)"
                )
            else:
                timeout_fmt = f" (timeout={request_timeout}s)"

            raise SPARQLExecuteException(
                f"SPARQL query timed out{timeout_fmt}", sparql
            ) from e

        except requests.RequestException as e:
            # try to get qlever exception
            status = None
            body = None
            qlever_ex = None
            if e.response is not None:
                status = e.response.status_code  # type: ignore
                try:
                    body = e.response.json()  # type: ignore
                    qlever_ex = body["exception"] if "exception" in body else None
                except JSONDecodeError:
                    body = e.response.text

            client_error = status and int(status / 100) == 4

            # immediately return on client error
            if client_error and qlever_ex:
                raise SPARQLExecuteException(
                    qlever_ex,
                    sparql,
                    status_code=status,
                ) from e
            elif client_error:
                raise SPARQLExecuteException(
                    body if body else "Client error",
                    sparql,
                    status_code=status,
                ) from e
            # retry on server error if not last retry
            elif i < max_retries - 1:
                continue
            elif qlever_ex:
                raise SPARQLExecuteException(
                    qlever_ex,
                    sparql,
                    status_code=status,
                ) from e
            else:
                raise SPARQLExecuteException(
                    body if body else "Server error",
                    sparql,
                    status_code=status,
                ) from e

    raise SPARQLExecuteException(
        f"Maximum retries reached ({max_retries})",
        sparql,
    )


def is_iri(iri: str) -> bool:
    return iri.startswith("<") and iri.endswith(">")


def wrap_iri(iri: str) -> str:
    return f"<{iri}>"


def has_scheme(iri: str) -> bool:
    return "://" in iri


def format_iri(
    iri: str,
    parser: LR1Parser,
    prefixes: dict[str, str],
    base_uri: str | None = None,
    wrap: bool = False,
) -> str:
    try:
        parse, _ = parse_string(
            # need to wrap for parser
            wrap_iri(iri) if not is_iri(iri) else iri,
            parser,
            skip_empty=True,
            collapse_single=True,
        )
    except Exception:
        return iri

    if parse["name"] != "IRIREF":
        # no iri, return as is
        return iri

    iri = parse["value"][1:-1]  # strip angle brackets
    if not has_scheme(iri):
        if base_uri is None:
            # return as-is if no base URI is given
            return wrap_iri(iri) if wrap else iri

        base_uri = base_uri if not is_iri(base_uri) else base_uri[1:-1]
        # resolve relative IRI against base URI
        iri = base_uri + iri

    longest = find_longest_prefix(iri, prefixes)
    if longest is None:
        return wrap_iri(iri) if wrap else iri

    short, long = longest
    val = iri[len(long) :]

    short_iri = short + ":" + val

    try:
        # try to parse prefixed IRI to check if it is valid, if not return full IRI
        # e.g., special characters are not as well supported in prefixed IRIs as in full IRIs
        parse_string(short_iri, parser)
        return short_iri
    except Exception:
        return wrap_iri(iri) if wrap else iri


def prepare_identifier_for_sparql(identifier: str, parser: LR1Parser) -> str:
    binding = parse_into_binding(identifier, parser)

    if binding is None and has_scheme(identifier):
        binding = parse_into_binding(wrap_iri(identifier), parser)

    if binding is None:
        return identifier

    return binding.sparql()


def format_literal(
    literal: str,
    parser: LR1Parser,
    prefixes: dict[str, str],
    base_uri: str | None = None,
) -> str:
    binding = parse_into_binding(literal, parser, prefixes)

    if binding is None:
        return literal

    if binding.typ == "literal" and binding.datatype is not None:
        formatted_dt = format_iri(
            binding.datatype,
            parser,
            prefixes,
            base_uri,
            wrap=True,
        )
        return f'"{binding.value}"^^{formatted_dt}'

    return binding.identifier()


def format_identifier(
    identifier: str,
    parser: LR1Parser,
    prefixes: dict[str, str],
    base_uri: str | None = None,
    wrap: bool = False,
) -> str:
    binding = parse_into_binding(identifier, parser, prefixes)

    if binding is None and has_scheme(identifier):
        binding = parse_into_binding(wrap_iri(identifier), parser, prefixes)

    if binding is None:
        return identifier

    if binding.typ == "uri":
        return format_iri(identifier, parser, prefixes, base_uri, wrap)

    elif binding.typ == "literal":
        return format_literal(identifier, parser, prefixes, base_uri)

    return binding.identifier()


def load_qlever_prefixes(endpoint: str) -> dict[str, str]:
    parse = urlparse(endpoint)
    parse.encode()
    split = parse.path.split("/")
    assert len(split) >= 1, "Endpoint path must contain at least one segment"
    split.insert(len(split) - 1, "prefixes")
    path = "/".join(split)
    parse = parse._replace(path=path)
    prefix_url = urlunparse(parse)

    response = requests.get(prefix_url)
    response.raise_for_status()
    prefixes = {}
    for line in response.text.splitlines():
        line = line.strip()
        if not line:
            continue
        assert line.startswith("PREFIX "), "Each line must start with 'PREFIX '"
        _, rest = line.split(" ", 1)
        prefix, uri = rest.split(":", 1)
        uri = uri.strip()
        assert is_iri(uri), "Prefix must be in IRI format"
        prefixes[prefix.strip()] = uri[1:-1]

    return prefixes


def load_entity_index_sparql() -> str:
    return read_resource("grasp.sparql.queries", "entity.index.sparql")


def load_property_index_sparql() -> str:
    return read_resource("grasp.sparql.queries", "property.index.sparql")


def load_literal_index_sparql() -> str:
    return read_resource("grasp.sparql.queries", "literal.index.sparql")


def load_entity_info_sparql() -> str:
    return read_resource("grasp.sparql.queries", "entity.info.sparql")


def load_property_info_sparql() -> str:
    return read_resource("grasp.sparql.queries", "property.info.sparql")


def load_literal_info_sparql() -> str:
    return read_resource("grasp.sparql.queries", "literal.info.sparql")
