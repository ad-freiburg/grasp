import json
import os
from importlib import resources
from typing import Any, Callable, Iterable, Iterator, TypeVar
from urllib.parse import unquote_plus

from pydantic import BaseModel
from termcolor import colored

from grasp.model import Message, Response, ToolCall


# IRI label derivation (also used by grasp.build.data and grasp.build.shapes)

_IRI_PUNCTUATION = {"_", "-", "."}


def split_iri(iri: str) -> tuple[str, str]:
    if "://" not in iri:
        return "", iri
    last = max(iri.rfind("#"), iri.rfind("/"))
    return ("", iri) if last == -1 else (iri[:last], iri[last + 1 :])


def split_at_punctuation(s: str) -> Iterator[str]:
    start = 0
    for i, c in enumerate(s):
        if c not in _IRI_PUNCTUATION:
            continue
        yield s[start:i]
        start = i + 1
    if start < len(s):
        yield s[start:]


def camel_case_split(s: str) -> str:
    words, last = [], 0
    for i, c in enumerate(s):
        if c.isupper() and i > 0 and s[i - 1].islower():
            words.append(s[last:i])
            last = i
    if last < len(s):
        words.append(s[last:])
    return " ".join(words)


def get_local_name_from_iri(iri: str, prefixes: dict[str, str]) -> str:
    from grasp.sparql.utils import find_longest_prefix

    pfx = find_longest_prefix(iri, prefixes)
    if pfx is None:
        _, obj_name = split_iri(iri)
    else:
        _, long = pfx
        obj_name = iri[len(long) :]

    return unquote_plus(obj_name)


def derive_label_from_iri(iri: str, prefixes: dict[str, str]) -> str:
    obj_name = get_local_name_from_iri(iri, prefixes)
    return " ".join(
        camel_case_split(part) for part in split_at_punctuation(obj_name)
    ).strip()


def get_index_dir(kg: str | None = None) -> str:
    index_dir = os.getenv("GRASP_INDEX_DIR", None)
    if index_dir is None:
        home_dir = os.path.expanduser("~")
        index_dir = os.path.join(home_dir, ".grasp", "index")

    if kg is not None:
        index_dir = os.path.join(index_dir, kg)

    return index_dir


def link(src: str, dst: str) -> None:
    src = os.path.abspath(src)
    dst = os.path.abspath(dst)
    if os.path.lexists(dst):
        os.remove(dst)

    rel = os.path.relpath(src, os.path.dirname(dst))
    os.symlink(rel, dst)


def get_available_knowledge_graphs() -> list[str]:
    index_dir = get_index_dir()
    if not os.path.exists(index_dir):
        return []

    return [
        name
        for name in os.listdir(index_dir)
        if os.path.isdir(os.path.join(index_dir, name))
    ]


class FunctionCallException(Exception):
    pass


def format_prefixes(prefixes: dict[str, str] | None = None, indent: int = 0) -> str:
    if not prefixes:
        return "None"

    return format_list(
        (f"{short}: {long}" for short, long in sorted(prefixes.items())),
        indent=indent,
    )


def format_notes(
    notes: list[str] | None = None,
    indent: int = 0,
    enumerated: bool = False,
) -> str:
    if not notes:
        return "None"
    elif enumerated:
        return format_enumerate(notes, indent)
    else:
        return format_list(notes, indent)


def format_list(items: Iterable[str], indent: int = 0) -> str:
    indent_str = " " * indent
    return "\n".join(f"{indent_str}- {item}" for item in items)


def format_enumerate(
    items: Iterable[str],
    indent: int = 0,
    start: int = 0,
) -> str:
    indent_str = " " * indent
    return "\n".join(
        f"{indent_str}{i + 1}. {item}" for i, item in enumerate(items, start=start)
    )


def format_kg_notes(
    kg_notes: dict[str, list[str]] | None = None,
    enumerated: bool = False,
) -> str:
    if not kg_notes:
        return "None"
    return format_list(
        f'"{kg}":\n{format_notes(notes, indent=2, enumerated=enumerated)}'
        for kg, notes in kg_notes.items()
    )


def format_model(model: BaseModel | None = None) -> str:
    if model is None:
        return "None"
    return model.model_dump_json(indent=2)


def format_error(reason: str, content: str) -> str:
    header = colored(f"ERROR (reason={reason})", "red", attrs=["bold", "underline"])
    return f"{header}\n{content}"


def format_message(message: Message) -> str:
    if isinstance(message.content, str):
        name = message.name or message.role
        header = colored(f"{name.upper()}", "magenta")
        return f"{header}\n{message.content}"
    else:
        return format_response(message.content)


def format_response(response: Response) -> str:
    header = colored(f"MODEL (usage={response.usage})", "blue")

    content = ""
    reasoning = response.reasoning
    if reasoning is not None:
        if reasoning.content is not None:
            content += f"Reasoning:\n{reasoning.content}\n\n"
        if reasoning.summary is not None:
            content += f"Reasoning summary:\n{reasoning.summary}\n\n"
        if reasoning.encrypted_content is not None:
            enc = reasoning.encrypted_content
            if len(enc) > 64:
                enc = enc[:64] + "..."
            content += f"Encrypted reasoning content:\n{enc}\n\n"

    if response.message is not None:
        if response.has_reasoning_content:
            content += "Content:\n"

        if isinstance(response.message, str):
            content += f"{response.message}\n\n"
        else:
            content += f"{response.message.content}\n\n"

    for tool_call in response.tool_calls:
        content += format_tool_call(tool_call) + "\n\n"

    return f"{header}\n{content.strip()}"


def format_tool_call(tool_call: ToolCall) -> str:
    name = colored(tool_call.name, "green")
    fn_args_str = colored(json.dumps(tool_call.args, indent=2), "yellow")
    content = f"{name}({fn_args_str})"
    if tool_call.result is not None:
        content += f":\n{tool_call.result}"
    return content


SKIP_ROLES = {"system", "config", "functions"}


def format_trace(output: dict, skip_system: bool = False) -> str:
    parts = []

    # header
    task = output.get("task", "unknown")
    elapsed = output.get("elapsed")
    error = output.get("error")
    header = colored(f"TRACE (task={task}", "cyan", attrs=["bold"])
    if elapsed is not None:
        header += colored(f", elapsed={elapsed:.2f}s", "cyan", attrs=["bold"])
    header += colored(")", "cyan", attrs=["bold"])
    parts.append(header)

    # input
    ipt = output.get("input")
    if ipt is not None:
        ipt_header = colored("INPUT", "magenta")
        parts.append(f"{ipt_header}\n{ipt}")

    # messages
    messages = output.get("messages", [])
    step = 0
    for msg_dict in messages:
        msg = Message(**msg_dict)

        if skip_system and msg.role in SKIP_ROLES:
            continue

        if isinstance(msg.content, Response):
            step += 1
            step_header = colored(f"Step {step}", "cyan", attrs=["bold"])
            parts.append(f"{step_header}\n{format_response(msg.content)}")
        else:
            parts.append(format_message(msg))

    # final output
    final_output = output.get("output")
    if isinstance(final_output, dict):
        final_output = final_output.get("formatted")
    if final_output is not None:
        out_header = colored("OUTPUT", "green", attrs=["bold"])
        parts.append(f"{out_header}\n{final_output}")

    # error
    if error is not None:
        parts.append(format_error("trace", error))

    return "\n\n".join(parts)


def is_server_error(message: str | None, hard_only: bool = False) -> bool:
    if message is None:
        return False

    strict_phrases = [
        "500 Server Error",
        "502 Server Error",
        "503 Server Error",
        "504 Server Error",
        "Bad Gateway",
        "Internal Server Error",
        "Service Unavailable",
        "Gateway Timeout",
    ]
    if hard_only:
        return any(phrase.lower() in message.lower() for phrase in strict_phrases)

    phrases = [
        # Current SPARQL execution errors.
        "SPARQL query timed out",
        "Took longer than",
        "SPARQL result exceeded",
        "upstream request timeout",
        "Bad Gateway",
        "Server Error",
        "Tried to allocate",
        "out of memory",
        # Older/raw requests error formats.
        "503 Server Error",  # qlever not available
        "502 Server Error",  # proxy error
        "(read timeout=6)",  # qlever not reachable
        "(connect timeout=6)",  # qlever not reachable
        "403 Client Error",  # wrong URL / API key
    ]
    return any(phrase.lower() in message.lower() for phrase in phrases)


def is_invalid_evaluation(evaluation: dict, empty_target_valid: bool = False) -> bool:
    if evaluation["target"]["err"] is not None:
        return True

    elif not empty_target_valid and evaluation["target"]["size"] == 0:
        return True

    return False


def is_retryable_evaluation(evaluation: dict) -> bool:
    target = evaluation.get("target", {})
    if target.get("err") is not None or target.get("size") == 0:
        return True

    prediction = evaluation.get("prediction", {})
    return is_server_error(prediction.get("err"))


def is_tool_fail(message: dict) -> bool:
    if message["role"] != "tool":
        return False

    content = message["content"]
    return is_server_error(content, hard_only=True)


def is_error(message: dict) -> bool:
    # old error format
    return message["role"] == "error"


def is_invalid_output(
    output: dict | None,
    none_output_invalid: bool = False,
) -> bool:
    if output is None:
        return True

    has_error = output.get("error") is not None
    if has_error:
        return True

    if none_output_invalid and output.get("output") is None:
        return True

    for message in output.get("messages", []):
        try:
            # new format
            msg = Message(**message)
            if not isinstance(msg.content, Response):
                continue

            if any(
                is_server_error(tool_call.result, hard_only=True)
                for tool_call in msg.content.tool_calls
            ):
                return True

            continue
        except Exception:
            pass

        # old format
        if is_tool_fail(message) or is_error(message):
            return True

    return False


def parse_key_value_pairs(headers: list[str]) -> dict[str, str]:
    # each parameter is formatted as key:value
    header_dict = {}
    for header in headers:
        key, value = header.split(":", 1)
        header_dict[key.strip()] = value.strip()
    return header_dict


def clip(s: str, max_len: int = 128, respect_word_boundaries: bool = True) -> str:
    if len(s) <= max_len:
        return s

    elif not respect_word_boundaries:
        if max_len <= 3:
            return s[:max_len]

        half = (max_len - 3) // 2
        return s[:half] + "..." + s[-half:]

    if max_len <= 5:
        return s[:max_len]

    half = (max_len - 5) // 2  # account for spaces around "..."
    first = half
    while first > 0 and not s[first].isspace():
        first -= 1

    last = len(s) - half
    while last < len(s) and last > 0 and not s[last - 1].isspace():
        last += 1

    if first <= 0 or last >= len(s):
        # at least 1 word on either side, fall back
        # to character clipping otherwise
        return clip(s, max_len, respect_word_boundaries=False)

    return s[:first] + " ... " + s[last:]


T = TypeVar("T")


def ordered_unique(
    lst: list[T],
    key: Callable[[T], Any] | None = None,
    filter: Callable[[T], bool] | None = None,
) -> list[T]:
    seen = set()
    unique = []
    for item in lst:
        if filter is not None and not filter(item):
            continue

        k = key(item) if key is not None else item
        if k in seen:
            continue

        seen.add(k)
        unique.append(item)

    return unique


def read_resource(package: str, resource: str) -> str:
    with resources.files(package).joinpath(resource).open() as f:
        return f.read()
