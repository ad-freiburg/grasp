import argparse
import json
import os
import random
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from logging import Logger
from typing import Generator

import torch
from grammar_utils.parse import LR1Parser  # type: ignore
from peft import AutoPeftModelForCausalLM, PeftModel
from pydantic import BaseModel
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from universal_ml_utils.configuration import load_config
from universal_ml_utils.io import dump_json, dump_jsonl, load_jsonl
from universal_ml_utils.logging import get_logger, setup_logging
from universal_ml_utils.ops import consume_generator, extract_field, map_generator

from grasp.baselines.grisp.data import (
    ALT_LABELS,
    RESULT_UNAVAILABLE,
    RESULT_UNRESOLVED,
    VALIDATION_OPTIONS,
    FillOrder,
    OracleSkeletonUnavailable,
    OrderedAlternatives,
    Skeleton,
    count_alternatives,
    find_alternative_groups,
    get_improvement_prompt,
    get_selection_prompt_and_options,
    get_skeleton_prompt,
    get_validation_prompt,
    gold_sparql_to_nl_skeleton,
    ordered_alternatives_with_interleave,
)
from grasp.baselines.grisp.train import GRISPTrainConfig
from grasp.baselines.grisp.utils import (
    find_best_checkpoint,
    load_sparql_parser,
    set_chat_template,
)
from grasp.configs import KgConfig
from grasp.manager import KgManager, load_kg_manager
from grasp.sparql.metrics import f1_score
from grasp.sparql.types import AskResult, ObjType, Selection, SelectResult
from grasp.sparql.utils import (
    SPARQLExecuteException,
)
from grasp.tasks.utils import format_sparql_result, prepare_sparql_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GRISP model")
    parser.add_argument(
        "config",
        type=str,
        help="Path to GRISP run configuration",
    )
    parser.add_argument(
        "run_directory",
        type=str,
        help="Path to the training run directory",
    )
    parser.add_argument(
        "--selection-run",
        type=str,
        default=None,
        help="Path to the training run directory for the selection model, "
        "if different from the main model",
    )
    parser.add_argument(
        "-l",
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level",
    )
    parser.add_argument(
        "--all-loggers",
        action="store_true",
        help="Set log level for all loggers instead of just the main one",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cpu, cuda, etc.)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="Data type to use for model weights (auto, float16, bfloat16, float32, etc.)",
    )

    # add two subparsers: run and file for running grisp
    # on a single question and on a benchmark file
    input_parsers = parser.add_subparsers(dest="command", required=True)
    run_parser = input_parsers.add_parser(
        "run",
        help="Run GRISP on a single question",
    )
    run_parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Question to run GRISTP on, if not given, read from stdin",
    )
    run_parser.add_argument(
        "-if",
        "--input-format",
        type=str,
        choices=["text", "json"],
        default="text",
        help="Format of the input (raw text or JSON)",
    )
    run_parser.add_argument(
        "--input-field",
        type=str,
        default="question",
        help="Field to extract input from",
    )

    file_parser = input_parsers.add_parser(
        "file",
        help="Run GRISP on a benchmark file",
    )
    file_parser.add_argument(
        "--progress",
        action="store_true",
        help="Show a progress bar",
    )
    file_parser.add_argument(
        "-i",
        "--input-file",
        type=str,
        help="Path to file in JSONL format to run GRISP on, if not given, read JSONL from stdin",
    )
    file_parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the inputs",
    )
    file_parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Skip the first N inputs",
    )
    file_parser.add_argument(
        "--take",
        type=int,
        default=None,
        help="Limit number of inputs (after skipping) to N",
    )
    file_parser.add_argument(
        "--input-field",
        type=str,
        default="question",
        help="Field to extract input from",
    )
    file_parser.add_argument(
        "--oracle-skeleton",
        action="store_true",
        help="Skip skeleton generation; derive the skeleton from each sample's "
        "gold 'sparql' field. Used to isolate IRI-selection errors.",
    )
    file_parser.add_argument(
        "--oracle-correctness",
        type=float,
        default=None,
        metavar="THRESHOLD",
        help="Enable the oracle correctness probe with the given F1 threshold "
        "(e.g. 0.8). Skeletons are generated normally, but the first resolved "
        "candidate whose result reaches THRESHOLD F1 against the gold 'sparql' "
        "field is returned (upper bound of a perfect query validator on top of "
        "the empty check). Falls back to the first resolved candidate if none "
        "clear the threshold. Omit to disable.",
    )
    file_parser.add_argument(
        "--trace",
        action="store_true",
        help="Persist every intermediate event (selections, backtracks, fails, "
        "validations) as a 'trace' list on each sample's output.",
    )
    file_parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        help="File to write the output to",
    )
    file_parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry failed inputs (only used with --output-file)",
    )
    file_parser.add_argument(
        "--none-output-invalid",
        action="store_true",
        help="Consider outputs with None as invalid (only used with --retry-failed)",
    )
    file_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output file",
    )

    return parser.parse_args()


MAX_IRIS = 131_072


class GRISPRunConfig(BaseModel):
    knowledge_graph: KgConfig

    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"

    temperature: float | None = 0.4
    min_p: float | None = None
    top_k: int | None = None
    top_p: float | None = 0.9
    repeat_penalty: float | None = None
    do_sample: bool = True

    skeleton_n: int = 8
    skeleton_top_k: int = 3

    selection_max_time: float = 60.0
    selection_top_k: int = 10
    constrain: bool = True
    backtrack: bool = True
    rerank: bool = True
    check_empty: bool = True

    # order in which the natural-language placeholders of a skeleton are
    # resolved. The model is trained with random fill orders (see
    # grisp.data.prepare_selection), so it is order-agnostic and any of these
    # can be selected at inference:
    # - left-to-right / right-to-left: by document position
    # - entities-then-properties / properties-then-entities: one object-type
    #   group first, then the other, each in document order
    # - triple-wise-entities-then-properties: triple by triple (document order),
    #   entities before properties within each triple
    # - random: a fixed random permutation per skeleton
    # non-left-to-right orders let the constraint filter see resolved neighbours
    # on both sides of the current placeholder.
    fill_order: FillOrder = "triple-wise-entities-then-properties"

    # learned skeleton improvement (task 4) with query validation (task 3).
    # When improve_skeletons is enabled, each generated skeleton is resolved
    # and, if fully resolved, scored by the validator; the skeleton (partial or
    # fully resolved) is then improved in place up to improve_rounds times per
    # skeleton, and the best-scoring fully-resolved candidate overall is
    # returned. improve_threshold controls early-exit: if set, the first
    # candidate with P(valid) >= improve_threshold is accepted immediately;
    # set it to null/None to disable early-exit and always run every skeleton
    # through all rounds, returning the best-scoring candidate.
    improve_skeletons: bool = False
    improve_threshold: float | None = 0.9
    improve_rounds: int = 1
    improve_n: int = 4

    skeleton_disable_adapter: bool = False
    selection_disable_adapter: bool = False
    # TODO: for later, we would need to support specifying dedicated
    # validation and improvement models as well
    validate_disable_adapter: bool = False
    improve_disable_adapter: bool = False


@dataclass
class SelectionOutcome:
    # sparql is set only when the skeleton fully resolved; otherwise the
    # skeleton is partial. selections holds the *deepest* set of selections
    # reached (not the eroded state left after backtracking), so the
    # improvement model gets meaningful evidence even for partial skeletons.
    # resolved is the query rendered at that deepest point (with any unresolved
    # placeholders left as natural language); for a fully resolved skeleton it
    # equals sparql.
    # (The failure reason is already emitted as a "fail" event for tracing, so
    # it is not duplicated here.)
    sparql: str | None
    selections: list[Selection] = field(default_factory=list)
    resolved: str = ""


class GRISPModel:
    def __init__(self, model: PreTrainedModel | PeftModel, disable: bool):
        self.model = model
        self.disable = disable and isinstance(model, PeftModel)

    @contextmanager
    def get(self):
        if self.disable:
            with self.model.disable_adapter():  # type: ignore
                yield self.model
        else:
            yield self.model


def generate_skeletons_from_prompt(
    model: GRISPModel,
    tokenizer: PreTrainedTokenizerBase,
    cfg: GRISPRunConfig,
    input: list[dict],
    parser: LR1Parser,
    logger: Logger,
    n: int,
    top_k: int,
) -> list[Skeleton]:
    with model.get() as m:
        device = next(m.parameters()).device
        enc = tokenizer.apply_chat_template(
            input,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            enable_thinking=False,
        ).to(device)  # type: ignore
        prompt_length = enc["input_ids"].shape[1]  # type: ignore

        fmt = tokenizer.decode(enc["input_ids"][0])  # type: ignore
        logger.debug(f"Generating skeletons:\n{fmt}")

        outputs = m.generate(  # type: ignore
            **enc,
            generation_config=GenerationConfig(
                num_beams=n,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
                min_p=cfg.min_p,
                repetition_penalty=cfg.repeat_penalty,
                do_sample=cfg.do_sample,
                max_new_tokens=512,
                renormalize_logits=True,
                num_return_sequences=n,
                return_dict_in_generate=True,
                output_scores=True,
            ),
        )

    skeletons = []
    seen = set()
    for i in range(len(outputs["sequences"])):
        token_ids = outputs["sequences"][i]
        decoded_token_ids = token_ids[prompt_length:]
        decoded = tokenizer.decode(decoded_token_ids, skip_special_tokens=True)

        if n > 1:
            score = outputs["sequences_scores"][i].item()
            logger.debug(f"Generated skeleton with score={score:.5f}:\n{decoded}")
        else:
            logger.debug(f"Generated skeleton:\n{decoded}")

        if decoded in seen:
            logger.debug("Already seen skeleton, skipping")
            continue

        try:
            skeleton = Skeleton.parse(decoded, parser, cfg.fill_order)  # type: ignore
        except Exception as e:
            logger.warning(f"Failed to parse skeleton, skipping: {e}")
            continue

        seen.add(decoded)
        skeletons.append(skeleton)

    # only take top k skeletons, others are just for logging
    logger.debug(
        f"Generated {len(skeletons)} valid unique skeletons, taking top {top_k}"
    )
    return skeletons[:top_k]


def generate_skeletons(
    model: GRISPModel,
    tokenizer: PreTrainedTokenizerBase,
    cfg: GRISPRunConfig,
    question: str,
    manager: KgManager,
    parser: LR1Parser,
    logger: Logger,
) -> list[Skeleton]:
    input = get_skeleton_prompt(manager.kg, question)
    return generate_skeletons_from_prompt(
        model,
        tokenizer,
        cfg,
        input,
        parser,
        logger,
        n=cfg.skeleton_n,
        top_k=cfg.skeleton_top_k,
    )


def generate_improved_skeletons(
    model: GRISPModel,
    tokenizer: PreTrainedTokenizerBase,
    cfg: GRISPRunConfig,
    question: str,
    bad_skeleton: str,
    manager: KgManager,
    parser: LR1Parser,
    logger: Logger,
    sparql: str | None = None,
    selections: str | None = None,
    result: str | None = None,
) -> list[Skeleton]:
    input = get_improvement_prompt(
        manager.kg,
        question,
        bad_skeleton,
        sparql,
        selections,
        result,
    )
    # the per-skeleton loop refines one skeleton at a time, so keep only the
    # single best rewrite
    return generate_skeletons_from_prompt(
        model,
        tokenizer,
        cfg,
        input,
        parser,
        logger,
        n=cfg.improve_n,
        top_k=1,
    )


def rerank_alternatives(
    model: GRISPModel,
    tokenizer: PreTrainedTokenizerBase,
    manager: KgManager,
    question: str,
    sparql: str,
    selections: list[Selection],
    alternatives: OrderedAlternatives,
    logger: Logger,
) -> list[tuple[int | None, float]]:
    prompt, options = get_selection_prompt_and_options(
        manager,
        question,
        sparql,
        selections,
        alternatives,
    )
    num_alternatives = count_alternatives(alternatives)

    enc = tokenizer.apply_chat_template(
        prompt,
        add_generation_prompt=True,
        return_dict=True,
        enable_thinking=False,
    )
    input_ids = enc["input_ids"]  # type: ignore

    fmt = tokenizer.decode(input_ids)  # type: ignore
    logger.debug(f"Reranking alternatives:\n{fmt}")
    logger.debug(f"Last 10 input ids for reranking: {input_ids[-10:]}")  # type: ignore

    with model.get() as m:
        device = next(m.parameters()).device
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
        option_ids = []
        for option in options:
            option_token_ids = tokenizer.encode(
                option,
                add_special_tokens=False,
            )  # type: ignore
            assert len(option_token_ids) == 1, "Option must be a single token"
            logger.debug(f"Option '{option}' token id: {option_token_ids[0]}")
            option_ids.append(option_token_ids[0])

        option_ids = torch.tensor(option_ids, dtype=torch.long, device=device)

        # shape [1, S, V]
        with torch.inference_mode():
            logits = m(input_ids.unsqueeze(0)).logits
            logger.debug(f"Score logits shape: {logits.shape}")

    # get last logits [V]
    logits = logits[0, -1]
    # get option logits, [|O|]
    logits = logits[option_ids]
    # normalize options
    scores = torch.softmax(logits, dim=-1)
    # sort
    sorted_scores, sorted_indices = torch.sort(scores, descending=True)

    sorted_scores = sorted_scores.tolist()
    sorted_indices = sorted_indices.tolist()
    logger.debug(
        "Reranked alternatives:\n"
        + "\n".join(
            f"Rank {i + 1}: "
            f"{ALT_LABELS[index] if index < num_alternatives else 'None'} "
            f"(score={score:.2%})"
            for i, (score, index) in enumerate(zip(sorted_scores, sorted_indices))
        )
    )

    return [
        (index if index < num_alternatives else None, score)
        for index, score in zip(sorted_indices, sorted_scores)
    ]


def validate_query(
    model: GRISPModel,
    tokenizer: PreTrainedTokenizerBase,
    manager: KgManager,
    question: str,
    sparql: str,
    selections: str | None,
    result: str | None,
    logger: Logger,
) -> float:
    # learned query validation (task 3): return P(valid), the softmax mass on
    # the VALID option over the two single-token options.
    prompt = get_validation_prompt(
        manager.kg,
        question,
        sparql,
        selections,
        result,
    )
    enc = tokenizer.apply_chat_template(
        prompt,
        add_generation_prompt=True,
        return_dict=True,
        enable_thinking=False,
    )
    input_ids = enc["input_ids"]  # type: ignore
    logger.debug(f"Validating query:\n{tokenizer.decode(input_ids)}")  # type: ignore

    with model.get() as m:
        device = next(m.parameters()).device
        ids = torch.tensor(input_ids, dtype=torch.long, device=device)
        option_ids = []
        for option in VALIDATION_OPTIONS:
            option_token_ids = tokenizer.encode(option, add_special_tokens=False)
            assert len(option_token_ids) == 1, "Validation option must be single token"
            option_ids.append(option_token_ids[0])
        option_ids = torch.tensor(option_ids, dtype=torch.long, device=device)

        with torch.inference_mode():
            logits = m(ids.unsqueeze(0)).logits

    logits = logits[0, -1][option_ids]
    scores = torch.softmax(logits, dim=-1)
    p_valid = scores[0].item()
    logger.debug(f"Validation P(valid)={p_valid:.2%}")
    return p_valid


def is_api_failure(exception: Exception) -> bool:
    exc_msg = str(exception).lower()
    is_timeout = isinstance(exception, TimeoutError) or "timeout" in exc_msg
    is_server_fail = any(str(code) in exc_msg for code in range(500, 600))
    return is_timeout or is_server_fail


def select_iris(
    skeleton: Skeleton,
    model: GRISPModel,
    tokenizer: PreTrainedTokenizerBase,
    cfg: GRISPRunConfig,
    question: str,
    manager: KgManager,
    logger: Logger,
) -> Generator[dict, None, SelectionOutcome]:
    start = time.monotonic()
    # init empty memo
    memo: dict[str, OrderedAlternatives] = {}

    # deepest set of selections reached; preserved across backtracking so a
    # partial (unresolved) skeleton still carries the items it managed to fill.
    # deepest_query is that deepest state rendered as a (partial) query: the
    # skeleton with the selected IRIs substituted in (no prefix declarations,
    # since skeletons carry none).
    deepest: list[Selection] = []
    deepest_query: str = skeleton.materialize_partial()

    supports_variants = {
        obj_type: manager.get_normalizer(obj_type.index_name).supports_variants
        for obj_type in [ObjType.ENTITY, ObjType.PROPERTY]
    }

    while True:
        if time.monotonic() - start > cfg.selection_max_time:
            yield {"type": "fail", "reason": "timeout"}
            logger.debug("Selection process timed out, abandoning skeleton")
            return SelectionOutcome(None, deepest, deepest_query)

        if skeleton.done:
            if not cfg.check_empty:
                break

            try:
                # reject empty queries
                sparql = skeleton.materialize()
                logger.debug(f"Checking result of final SPARQL query:\n{sparql}")
                result = manager.execute_sparql(
                    skeleton.materialize(),
                    # 6 seconds to execute query, 3 to read result
                    request_timeout=(3.5, 6.0),
                    read_timeout=3.0,
                )
                logger.debug(f"Result:\n{manager.format_sparql_result(result)}")
                reject = result.is_empty
            except SPARQLExecuteException as e:
                logger.warning(f"Error executing final SPARQL to check emptiness:\n{e}")
                reject = e.is_client_error
            except Exception as e:
                logger.warning(f"Unexpected error executing final SPARQL:\n{e}")
                reject = True

            if not reject:
                yield {"type": "validation", "result": "passed"}
                break

            yield {"type": "validation", "result": "failed"}

            if skeleton.replaced == 0 or not cfg.backtrack:
                yield {"type": "fail", "reason": "validation_failed"}
                logger.debug("Final SPARQL query is empty, abandoning skeleton")
                return SelectionOutcome(None, deepest, deepest_query)

            logger.debug(
                "Final SPARQL query is empty, backtracking to previous placeholder"
            )
            skeleton.pop_selection()
            yield {"type": "backtrack", "reason": "validation_failed"}
            continue

        info = skeleton.prepare_for_selection()
        queries = info.build_queries(supports_variants)

        # set alternatives for current placeholder
        if info.prefix not in memo:
            alternative_groups = find_alternative_groups(
                manager,
                info.prefix,
                queries,
                cfg.selection_top_k,
                logger,
                skip_constraint=not cfg.constrain,
                max_candidates=MAX_IRIS,
            )
            memo[info.prefix] = ordered_alternatives_with_interleave(
                alternative_groups,
                queries,
                interleave=not cfg.rerank,
            )

        alternatives = memo[info.prefix]
        ranking = None

        if cfg.rerank:
            # use model to rerank alternatives before selecting
            # will return an empty list if 'None' is top ranked
            # such that we can continue with backtracking
            ranking = rerank_alternatives(
                model,
                tokenizer,
                manager,
                question,
                info.sparql,
                skeleton.selections,
                alternatives,
                logger,
            )

        yield {
            "type": "alternatives",
            "index": skeleton.replaced,
            "prefix": info.prefix,
            "sparql": info.sparql,
            "query": info.query,
            "variant": info.variant,
            "alternatives": [
                {
                    "identifier": alternative.get_identifier(),
                    "label": alternative.get_label(),
                    "variants": alternative.variants,
                    "obj_type": obj_type.value,
                    "variant": variant,
                }
                for alternative, obj_type, variant in alternatives
            ],
            "ranking": ranking,
        }

        if ranking is not None:
            # apply ranking
            ranked_alts = []
            for index, _ in ranking:
                if index is None:
                    break

                ranked_alts.append(alternatives[index])

            alternatives = ranked_alts
            memo[info.prefix] = alternatives

        if len(alternatives) == 0:
            if skeleton.replaced == 0:
                logger.debug(
                    "No valid alternatives left for the first placeholder, abandoning skeleton"
                )
                yield {"type": "fail", "reason": "no_alternatives"}
                return SelectionOutcome(None, deepest, deepest_query)

            elif not cfg.backtrack:
                logger.debug(
                    "No valid alternatives left for the current placeholder, "
                    "abandoning skeleton due to backtracking disabled"
                )
                yield {"type": "fail", "reason": "no_alternatives"}
                return SelectionOutcome(None, deepest, deepest_query)

            logger.debug(
                "No valid alternatives left for the current placeholder, backtracking"
            )
            skeleton.pop_selection()
            yield {"type": "backtrack", "reason": "no_alternatives"}
            continue

        # just try out next alternative in order
        alternative, obj_type, variant = alternatives[0]
        alternatives = alternatives[1:]
        memo[info.prefix] = alternatives
        if not alternative.variants:
            # just to be sure to have no parsing errors
            variant = None

        if variant is not None:
            if not alternative.variants or variant not in alternative.variants:
                logger.debug(
                    f"Variant '{variant}' not found in alternative variants, "
                    f"trying next alternative"
                )
                yield {
                    "type": "continue",
                    "reason": "invalid_variant",
                    "identifier": alternative.get_identifier(),
                    "label": alternative.get_label(),
                    "variant": variant,
                    "variants": alternative.variants,
                }
                continue

            logger.debug(
                f"Variant '{variant}' found in alternative "
                f"variants ({alternative.variants})"
            )

        show_variants = [variant] if variant is not None else None
        selection = Selection(
            alternative=alternative,
            variant=variant,
            obj_type=obj_type,
        )
        skeleton.add_selection(selection, manager)
        if skeleton.replaced > len(deepest):
            deepest = list(skeleton.selections)
            deepest_query = skeleton.materialize_partial()
        logger.debug(
            f"Adding the following alternative at placholder {skeleton.replaced}/{skeleton.total}:\n"
            f"{alternative.get_selection_string(include_variants=show_variants)} "
        )

        yield {
            "type": "select",
            "identifier": alternative.get_identifier(),
            "label": alternative.get_label(),
            "variant": variant,
        }

    # the executable query needs prefix declarations added back via
    # fix_prefixes; the display copy (resolved) is the raw materialized skeleton,
    # which already carries no prefix declarations
    materialized = skeleton.materialize()
    sparql = manager.fix_prefixes(materialized)
    return SelectionOutcome(sparql, list(skeleton.selections), materialized)


def generate(
    model: GRISPModel,
    tokenizer: PreTrainedTokenizerBase,
    cfg: GRISPRunConfig,
    question: str,
    manager: KgManager,
    parser: LR1Parser,
    logger: Logger,
    select_model: GRISPModel | None = None,
    select_tokenizer: PreTrainedTokenizerBase | None = None,
    yield_output: bool = False,
    gold_sparql: str | None = None,
    oracle_skeleton: bool = False,
    oracle_correctness_threshold: float | None = None,
) -> Generator[dict, None, dict]:
    sparql = None
    error = None
    start = time.monotonic()

    # both oracle modes consume the same gold query (the sample's 'sparql'
    # field) and are mutually exclusive:
    # - oracle_skeleton: derive the skeleton from the gold query (isolate
    #   selection errors)
    # - oracle_correctness_threshold (not None): generate skeletons normally,
    #   but return the first resolved candidate whose result reaches this F1
    #   threshold against the gold query (upper bound of a perfect validator on
    #   top of the empty check)
    use_oracle_skeleton = oracle_skeleton
    use_oracle_correctness = oracle_correctness_threshold is not None
    assert not (use_oracle_skeleton and use_oracle_correctness), (
        "oracle skeleton and oracle correctness cannot be combined"
    )
    assert not (
        (use_oracle_skeleton or use_oracle_correctness) and gold_sparql is None
    ), "oracle modes require a gold_sparql"

    # learned skeleton improvement (task 4) and its query validation (task 3)
    # only apply when generating from scratch, not under either oracle mode.
    use_improve = (
        cfg.improve_skeletons and not use_oracle_skeleton and not use_oracle_correctness
    )
    validate_model = select_model or model
    validate_tokenizer = select_tokenizer or tokenizer

    def execute_and_format(query: str) -> dict:
        result, selections = prepare_sparql_result(
            query,
            manager.kg,
            [manager],
            max_rows=10,
            max_columns=10,
            # same as for autocompletion and check
            request_timeout=(3.5, 6.0),
            read_timeout=3.0,
        )
        # result.result is None when execution failed (timeout / backend down /
        # parse error). Feed the validator and improvement model a stable
        # "unavailable" marker in that case so a transient failure is not read
        # as a wrong query; keep the raw formatted text in "formatted" for the
        # human-facing output and debugging.
        # prepare_sparql_result returns parse-derived selections even when
        # execution fails, so the validator/improvement model still sees the
        # resolved items when there is no result.
        preview = result.formatted if result.result is not None else RESULT_UNAVAILABLE
        return {
            "sparql": result.sparql,
            "selections": manager.format_selections(selections),
            "result": preview,
            "formatted": format_sparql_result(manager, result, selections),
        }

    # best candidate seen so far (by validator score); used to pick the final
    # answer when no candidate clears the acceptance threshold.
    best: dict | None = None

    # gold result for the oracle correctness probe, executed once up front with
    # the same timeouts used elsewhere in the loop. None if unavailable (the
    # probe then cannot score and falls back to baseline selection).
    gold_result: SelectResult | AskResult | None = None
    if use_oracle_correctness:
        try:
            assert gold_sparql is not None
            gold_result = manager.execute_sparql(
                gold_sparql,
                request_timeout=(3.5, 6.0),
                read_timeout=3.0,
            )
        except Exception as e:
            logger.warning(f"Oracle correctness: failed to execute gold query:\n{e}")
            gold_result = None

    def oracle_f1(query: str) -> float:
        # F1 of a resolved candidate query's result against the gold result,
        # mirroring the offline evaluation metric
        if gold_result is None:
            return 0.0
        try:
            pred_result = manager.execute_sparql(
                query,
                request_timeout=(3.5, 6.0),
                read_timeout=3.0,
            )
        except Exception as e:
            logger.debug(f"Oracle correctness: candidate failed to execute:\n{e}")
            return 0.0
        return f1_score(pred_result, gold_result)

    try:
        if use_oracle_skeleton:
            logger.debug("Using oracle skeleton from gold SPARQL")
            assert gold_sparql is not None
            nl = gold_sparql_to_nl_skeleton(gold_sparql, manager)
            skeletons = [Skeleton.parse(nl, parser, cfg.fill_order)]
        else:
            skeletons = generate_skeletons(
                model,
                tokenizer,
                cfg,
                question,
                manager,
                parser,
                logger,
            )

        yield {
            "type": "skeletons",
            "skeletons": [skeleton.nl_sparql for skeleton in skeletons],
        }

        accept = cfg.improve_threshold
        rounds = cfg.improve_rounds if use_improve else 0
        seen_skeletons = {skeleton.nl_sparql for skeleton in skeletons}
        idx = 0

        def resolve(
            skeleton: Skeleton, idx: int
        ) -> Generator[dict, None, "SelectionOutcome"]:
            outcome = yield from map_generator(
                lambda selection: {
                    "type": "selection",
                    "skeleton": idx,
                    "selection": selection,
                },
                select_iris(
                    skeleton,
                    select_model or model,
                    select_tokenizer or tokenizer,
                    cfg,
                    question,
                    manager,
                    logger,
                ),
            )
            return outcome

        if use_oracle_correctness:
            # resolve every skeleton and return the first whose result reaches
            # the F1 threshold against the gold query. If none clear it, fall
            # back to the first fully-resolved skeleton (baseline behavior), so
            # the eval delta isolates the gain from picking a correct skeleton
            # when one exists among the candidates.
            assert oracle_correctness_threshold is not None
            first_resolved: str | None = None
            for skeleton in skeletons:
                cur_idx = idx
                idx += 1
                outcome = yield from resolve(skeleton, cur_idx)
                if outcome.sparql is None:
                    continue
                if first_resolved is None:
                    first_resolved = outcome.sparql
                f1 = oracle_f1(outcome.sparql)
                yield {
                    "type": "oracle_correctness",
                    "skeleton": cur_idx,
                    "f1": f1,
                    "result": (
                        "passed" if f1 >= oracle_correctness_threshold else "failed"
                    ),
                }
                if f1 >= oracle_correctness_threshold:
                    logger.debug(f"Oracle correctness: accepting candidate F1={f1:.2%}")
                    sparql = outcome.sparql
                    break
            if sparql is None:
                sparql = first_resolved
        elif not use_improve:
            # original behavior: take the first fully resolved skeleton
            for skeleton in skeletons:
                cur_idx = idx
                idx += 1
                outcome = yield from resolve(skeleton, cur_idx)
                if outcome.sparql is not None:
                    sparql = outcome.sparql
                    break
        else:
            # per-skeleton: resolve, validate (if fully resolved), then improve
            # the skeleton in place up to `rounds` times. Partial skeletons skip
            # validation and are improved directly using their deepest partial
            # selections as evidence.
            #
            # The validator always scores resolved candidates so the final answer
            # can be picked as the best-scoring one. `accept` (improve_threshold)
            # only controls the early-exit: when set, the first candidate clearing
            # it is returned immediately; when None, every skeleton is improved
            # through all rounds and the best-scoring candidate overall is returned.
            for skeleton in skeletons:
                current = skeleton
                for r in range(rounds + 1):
                    cur_idx = idx
                    idx += 1
                    outcome = yield from resolve(current, cur_idx)

                    if outcome.sparql is not None:
                        cand_out = execute_and_format(outcome.sparql)
                        score = validate_query(
                            validate_model,
                            validate_tokenizer,
                            manager,
                            question,
                            cand_out["sparql"],
                            cand_out["selections"],
                            cand_out["result"],
                            logger,
                        )
                        yield {
                            "type": "validation",
                            "skeleton": cur_idx,
                            "score": score,
                            "result": (
                                "passed"
                                if accept is not None and score >= accept
                                else "failed"
                            ),
                        }
                        if best is None or score > best["score"]:
                            best = {
                                "sparql": outcome.sparql,
                                "score": score,
                                "out": cand_out,
                            }
                        if accept is not None and score >= accept:
                            logger.debug(
                                f"Accepting candidate with P(valid)={score:.2%}"
                            )
                            sparql = outcome.sparql
                            break
                        evidence_sparql = cand_out["sparql"]
                        evidence_selections = cand_out["selections"]
                        evidence_result = cand_out["result"]
                    else:
                        # partial skeleton: no executable query to validate, show
                        # the partially resolved query and its deepest selections
                        evidence_sparql = outcome.resolved
                        evidence_selections = manager.format_selections(
                            outcome.selections
                        )
                        evidence_result = RESULT_UNRESOLVED

                    # improve this skeleton in place (unless out of rounds)
                    if r >= rounds:
                        break
                    improved = generate_improved_skeletons(
                        model,
                        tokenizer,
                        cfg,
                        question,
                        current.nl_sparql,
                        manager,
                        parser,
                        logger,
                        sparql=evidence_sparql,
                        selections=evidence_selections,
                        result=evidence_result,
                    )
                    improved = [
                        s for s in improved if s.nl_sparql not in seen_skeletons
                    ]
                    if not improved:
                        break
                    seen_skeletons.update(s.nl_sparql for s in improved)
                    current = improved[0]
                    yield {
                        "type": "improvement",
                        "skeleton": cur_idx,
                        "skeletons": [s.nl_sparql for s in improved],
                    }

                if sparql is not None:
                    break

            # no candidate accepted via early-exit (or early-exit disabled in
            # improve-only mode); fall back to the best-scoring fully-resolved
            # candidate seen across all skeletons
            if sparql is None and best is not None:
                sparql = best["sparql"]
                logger.debug(
                    f"Using best-scoring candidate P(valid)={best['score']:.2%}"
                )

    except OracleSkeletonUnavailable as e:
        logger.warning(f"Skipping sample, oracle skeleton unavailable: {e}")
        error = {
            "reason": "oracle_skeleton_unavailable",
            "content": str(e),
        }
    except Exception as e:
        logger.error(f"Error generating SPARQL query: {e}")
        error = {
            "reason": "failure",
            "content": str(e),
        }

    out = {
        "sparql": None,
        "kg": manager.kg,
        "selections": None,
        "result": None,
        "endpoint": manager.endpoint,
        "formatted": "No SPARQL query generated or found",
    }
    if sparql is not None:
        # reuse the already-executed result for the chosen candidate if available
        # (only validated candidates carry a precomputed "out")
        if (
            best is not None
            and best.get("out") is not None
            and best["sparql"] == sparql
        ):
            out.update(best["out"])
        else:
            out.update(execute_and_format(sparql))

    end = time.monotonic()
    output = {
        "type": "output",
        "error": error,
        "output": out,
        "elapsed": end - start,
    }

    if yield_output:
        yield output

    return output


def is_invalid_output(output: dict | None, none_output_invalid: bool = False) -> bool:
    return (
        output is None
        or output.get("error") is not None
        or (output["output"] is None and none_output_invalid)
    )


def load_model_and_tokenizer(
    directory: str,
    device: str,
    dtype: str,
    logger: Logger,
) -> tuple[PreTrainedModel | PeftModel, PreTrainedTokenizerBase]:
    checkpoint = find_best_checkpoint(directory)
    assert checkpoint is not None, f"No best checkpoint found in {directory}"
    logger.info(f"Best checkpoint found at {checkpoint}")

    train_cfg_path = os.path.join(directory, "config.yaml")
    train_cfg = GRISPTrainConfig(**load_config(train_cfg_path))

    if train_cfg.lora is not None:
        model = AutoPeftModelForCausalLM.from_pretrained(
            checkpoint,
            dtype=dtype,
            device_map=device,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            dtype=dtype,
            device_map=device,
        )

    model.config.use_cache = True
    model.eval()
    logger.info(f"Loaded model {model.config.name_or_path}:\n{model}")

    tokenizer = AutoTokenizer.from_pretrained(model.config.name_or_path)  # type: ignore

    if train_cfg.overwrite_chat_template:
        tokenizer = set_chat_template(tokenizer)

    return model, tokenizer


def main(args: argparse.Namespace) -> None:
    logger = get_logger("GRISP", args.log_level)
    if args.all_loggers:
        setup_logging(args.log_level)

    train_cfg_path = os.path.join(args.run_directory, "config.yaml")
    train_cfg = GRISPTrainConfig(**load_config(train_cfg_path))
    assert train_cfg.type in ["skeleton", "both", "all"], (
        "Main run should be of type 'skeleton', 'both', or 'all'"
    )

    run_cfg = GRISPRunConfig(**load_config(args.config))
    logger.debug(f"Using run configuration:\n{run_cfg.model_dump_json(indent=2)}")

    logger.info(f"Loading model from {args.run_directory}")
    model, tokenizer = load_model_and_tokenizer(
        args.run_directory,
        args.device,
        args.dtype,
        logger,
    )

    skeleton_tokenizer = tokenizer
    skeleton_model = GRISPModel(model, run_cfg.skeleton_disable_adapter)

    if train_cfg.type == "skeleton" and args.selection_run is None:
        logger.warning(
            "Main model is skeleton only, selection quality may be suboptimal"
        )
        selection_model = GRISPModel(model, run_cfg.selection_disable_adapter)
        selection_tokenizer = tokenizer
    elif train_cfg.type == "skeleton":
        logger.info(f"Loading selection model from {args.selection_run}")
        sel_model, selection_tokenizer = load_model_and_tokenizer(
            args.selection_run,
            args.device,
            args.dtype,
            logger,
        )
        selection_model = GRISPModel(sel_model, run_cfg.selection_disable_adapter)
    else:
        # train_cfg.type == "both": main model handles both stages,
        # but wrap independently so the selection adapter flag is respected
        selection_model = GRISPModel(model, run_cfg.selection_disable_adapter)
        selection_tokenizer = tokenizer

    logger.info(
        f"Using model {skeleton_model.model.config.name_or_path} for skeleton generation"  # type: ignore
        f" (adapter disabled={skeleton_model.disable})"
    )
    logger.info(
        f"Using model {selection_model.model.config.name_or_path} for selection"  # type: ignore
        f" (adapter disabled={selection_model.disable})"
    )

    manager = load_kg_manager(run_cfg.knowledge_graph)
    manager.load_models()

    parser = load_sparql_parser()

    # adapted from GRASP cli
    run_on_file = args.command == "file"
    outputs = []
    if run_on_file:
        if args.input_file is None:
            inputs = [json.loads(line) for line in sys.stdin]
        else:
            inputs = load_jsonl(args.input_file)

        # id fallback in case of missing ids
        # before shuffling/skipping/taking
        for i, ipt in enumerate(inputs):
            id = extract_field(ipt, "id")
            if id is None:
                ipt["id"] = str(i)

        if args.shuffle:
            assert train_cfg.seed is not None, (
                "Seed must be set for deterministic shuffling"
            )
            random.seed(train_cfg.seed)
            random.shuffle(inputs)

        skip = max(0, args.skip)
        take = args.take or len(inputs)
        inputs = inputs[skip : skip + take]

        if args.output_file:
            if os.path.exists(args.output_file) and not args.overwrite:
                outputs = load_jsonl(args.output_file)

            # save info in config file next to output file
            output_stem, _ = os.path.splitext(args.output_file)
            config_file = output_stem + ".config.json"

            dump_json(train_cfg.model_dump(), config_file, indent=2)

        if args.progress:
            # wrap with tqdm
            inputs = tqdm(inputs, desc="GRISP")

    else:
        if args.input is None:
            ipt = sys.stdin.read()
        else:
            ipt = args.input

        if args.input_format == "json":
            inputs = [json.loads(ipt)]
        else:
            inputs = [{"question": ipt}]
            args.input_field = "question"  # overwrite

    oracle_skeleton = run_on_file and args.oracle_skeleton
    oracle_correctness_threshold = args.oracle_correctness if run_on_file else None
    oracle_correctness = oracle_correctness_threshold is not None
    assert not (oracle_skeleton and oracle_correctness), (
        "--oracle-skeleton and --oracle-correctness cannot be combined"
    )
    trace = run_on_file and args.trace

    for i, ipt in enumerate(inputs):
        id = extract_field(ipt, "id") or "unknown"

        gold_sparql = None
        if oracle_skeleton or oracle_correctness:
            gold_sparql = extract_field(ipt, "sparql")
            assert gold_sparql is not None, (
                f"--oracle-{'skeleton' if oracle_skeleton else 'correctness'} "
                f"requires a 'sparql' field on every sample, missing on input "
                f"{i:,} (id={id})"
            )

        ipt = extract_field(ipt, args.input_field)
        assert ipt is not None, f"Question not found for input {i:,}"

        if i < len(outputs):
            # overwrite id
            output = outputs[i]
            output["id"] = id
            if not args.retry_failed or not is_invalid_output(
                output,
                args.none_output_invalid,
            ):
                continue

        gen = generate(
            skeleton_model,
            skeleton_tokenizer,
            run_cfg,
            ipt,
            manager,
            parser,
            logger,
            selection_model,
            selection_tokenizer,
            gold_sparql=gold_sparql,
            oracle_skeleton=oracle_skeleton,
            oracle_correctness_threshold=oracle_correctness_threshold,
        )
        if trace:
            events: list[dict] = []
            try:
                while True:
                    events.append(next(gen))
            except StopIteration as e:
                output = e.value
            output["output"]["trace"] = events
        else:
            output = consume_generator(gen)

        output["config"] = run_cfg.model_dump()
        output["id"] = id

        if not run_on_file:
            print(json.dumps(output))
            break

        elif args.output_file is None:
            print(json.dumps(output))
            continue

        if i < len(outputs):
            outputs[i] = output
        else:
            outputs.append(output)

        dump_jsonl(outputs, args.output_file)

    if run_on_file and args.output_file is not None:
        # final dump, necessary if no new outputs were added
        # but some outputs were updated with ids
        dump_jsonl(outputs, args.output_file)


if __name__ == "__main__":
    main(parse_args())
