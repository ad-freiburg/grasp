import glob
import os
import random
from typing import Callable, Iterator

from grammar_utils.parse import LR1Parser  # type: ignore
from torch.utils.data import Sampler
from transformers import PreTrainedTokenizerBase
from universal_ml_utils.io import load_json

from grasp.sparql.utils import parse_to_string
from grasp.utils import read_resource

# placeholders are replaced by this when a skeleton is normalized
BLANK_PLACEHOLDER = "<IRI>"


def load_sparql_grammar() -> tuple[str, str]:
    sparql_grammar = read_resource("grasp.baselines.grisp.grammar", "sparql.y")
    sparql_lexer = read_resource("grasp.baselines.grisp.grammar", "sparql.l")
    return sparql_grammar, sparql_lexer


def load_sparql_parser() -> LR1Parser:
    sparql_grammar, sparql_lexer = load_sparql_grammar()
    return LR1Parser(sparql_grammar, sparql_lexer)


# copy with variables renamed to ?var0, ?var1, ... by first appearance, so the
# join structure survives, and optionally with placeholders blanked
def normalize_skeleton_parse(
    parse: dict,
    var_map: dict[str, str],
    blank_placeholders: bool = True,
) -> dict:
    name = parse["name"]
    value = parse.get("value")
    children = parse.get("children")

    if name in ("VAR1", "VAR2") and value is not None:
        if value not in var_map:
            var_map[value] = f"?var{len(var_map)}"
        return {"name": name, "value": var_map[value]}

    elif name == "NL_IRI" and blank_placeholders:
        return {"name": name, "value": BLANK_PLACEHOLDER}

    normalized: dict = {"name": name}
    if value is not None:
        normalized["value"] = value
    if children is not None:
        normalized["children"] = [
            normalize_skeleton_parse(child, var_map, blank_placeholders)
            for child in children
        ]
    return normalized


# blanked skeleton in document order: skeletons sharing it differ only in
# wording and variable names, so their placeholders line up one to one
def skeleton_wording_key(parse: dict) -> str:
    return parse_to_string(normalize_skeleton_parse(parse, {}, True))


def set_chat_template(tokenizer: PreTrainedTokenizerBase) -> PreTrainedTokenizerBase:
    # set custom chat template for single turn generation
    chat_template = """\
{{- bos_token }}
{%- for message in messages %}
    {%- if message['role'] != 'assistant' %}
        {{- message['role'].capitalize() + ' input:\n' }}
        {{- message['content'] + '\n\n' }}
    {%- else %}
        {{- 'Answer:\n' }}
        {% generation %}
          {{- message['content'].strip() + eos_token }}
        {% endgeneration %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- 'Answer:\n' }}
{%- endif %}"""
    tokenizer.chat_template = chat_template  # type: ignore
    return tokenizer


def find_latest_checkpoint(run_directory: str) -> str | None:
    def latest_ckpt_key(checkpoint_dir: str) -> int:
        path = os.path.join(checkpoint_dir, "trainer_state.json")
        state = load_json(path)
        return -state["global_step"]

    return find_checkpoint(run_directory, latest_ckpt_key)


def find_best_checkpoint(run_directory: str) -> str | None:
    def best_ckpt_key(checkpoint_dir: str) -> int | float:
        path = os.path.join(checkpoint_dir, "trainer_state.json")
        state = load_json(path)
        global_step = state["global_step"]

        log_entry = next(
            (
                entry
                for entry in state["log_history"]
                if entry["step"] == global_step and entry.get("eval_loss") is not None
            ),
        )
        # sort by eval loss
        return log_entry["eval_loss"]

    return find_checkpoint(run_directory, best_ckpt_key)


def find_checkpoint(
    run_directory: str,
    key: Callable[[str], int | float],
) -> str | None:
    # all subdir starting with checkpoint-*
    checkpoints = glob.glob(os.path.join(run_directory, "checkpoint-*"))
    if not checkpoints:
        return None

    checkpoints.sort(key=key)
    return checkpoints[0]


class SeededRandomSampler(Sampler):
    def __init__(self, n: int, seed: int, epoch: int = 0) -> None:
        self.n = n
        self.seed = seed
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        rand = random.Random(self.seed + self.epoch)
        permutation = list(range(self.n))
        rand.shuffle(permutation)
        self.epoch += 1
        return iter(permutation)

    def __len__(self) -> int:
        return self.n
