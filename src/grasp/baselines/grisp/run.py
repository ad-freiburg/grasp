import argparse
import json
import os
import random
import sys
import time
from contextlib import contextmanager
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
    OracleSkeletonUnavailable,
    OrderedAlternatives,
    Skeleton,
    count_alternatives,
    find_alternative_groups,
    get_selection_prompt_and_options,
    get_skeleton_prompt,
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
from grasp.sparql.types import ObjType, Selection
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

    skeleton_disable_adapter: bool = False
    selection_disable_adapter: bool = False


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
                num_beams=cfg.skeleton_n,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
                min_p=cfg.min_p,
                repetition_penalty=cfg.repeat_penalty,
                do_sample=cfg.do_sample,
                max_new_tokens=512,
                renormalize_logits=True,
                num_return_sequences=cfg.skeleton_n,
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

        if cfg.skeleton_n > 1:
            score = outputs["sequences_scores"][i].item()
            logger.debug(f"Generated skeleton with score={score:.5f}:\n{decoded}")
        else:
            logger.debug(f"Generated skeleton:\n{decoded}")

        if decoded in seen:
            logger.debug("Already seen skeleton, skipping")
            continue

        try:
            skeleton = Skeleton.parse(decoded, parser)  # type: ignore
        except Exception as e:
            logger.warning(f"Failed to parse skeleton, skipping: {e}")
            continue

        seen.add(decoded)
        skeletons.append(skeleton)

    # only take top k skeletons, others are just for logging
    logger.debug(
        f"Generated {len(skeletons)} valid unique skeletons, "
        f"taking top {cfg.skeleton_top_k}"
    )
    return skeletons[: cfg.skeleton_top_k]


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


def is_api_failure(exception: Exception) -> bool:
    exc_msg = str(exception).lower()
    is_timeout = isinstance(exception, TimeoutError) or "timeout" in exc_msg
    is_server_fail = any(str(code) in exc_msg for code in range(500, 600))
    return is_timeout or is_server_fail


def select_iris_left_to_right(
    skeleton: Skeleton,
    model: GRISPModel,
    tokenizer: PreTrainedTokenizerBase,
    cfg: GRISPRunConfig,
    question: str,
    manager: KgManager,
    logger: Logger,
) -> Generator[dict, None, str | None]:
    start = time.monotonic()
    # init empty memo
    memo: dict[str, OrderedAlternatives] = {}

    supports_variants = {
        obj_type: manager.get_normalizer(obj_type.index_name).supports_variants
        for obj_type in [ObjType.ENTITY, ObjType.PROPERTY]
    }

    while True:
        if time.monotonic() - start > cfg.selection_max_time:
            yield {"type": "fail", "reason": "timeout"}
            logger.debug("Selection process timed out, abandoning skeleton")
            return None

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
                return None

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
                return None

            elif not cfg.backtrack:
                logger.debug(
                    "No valid alternatives left for the current placeholder, "
                    "abandoning skeleton due to backtracking disabled"
                )
                yield {"type": "fail", "reason": "no_alternatives"}
                return None

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

    # convert back to sparql and fix prefixes
    sparql = skeleton.materialize()
    return manager.fix_prefixes(sparql)


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
) -> Generator[dict, None, dict]:
    sparql = None
    error = None
    start = time.monotonic()

    try:
        if gold_sparql is not None:
            logger.debug("Using oracle skeleton from gold SPARQL")
            nl = gold_sparql_to_nl_skeleton(gold_sparql, manager)
            skeletons = [Skeleton.parse(nl, parser)]
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

        for i, skeleton in enumerate(skeletons):
            selections = select_iris_left_to_right(
                skeleton,
                select_model or model,
                select_tokenizer or tokenizer,
                cfg,
                question,
                manager,
                logger,
            )

            sparql = yield from map_generator(
                lambda selection: {
                    "type": "selection",
                    "skeleton": i,
                    "selection": selection,
                },
                selections,
            )

            # take first fully assigned skeleton as final answer
            if sparql is not None:
                break

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
        result, selections = prepare_sparql_result(
            sparql,
            manager.kg,
            [manager],
            max_rows=10,
            max_columns=10,
            # same as for autocompletion and check
            request_timeout=(3.5, 6.0),
            read_timeout=3.0,
        )
        out["sparql"] = result.sparql
        out["selections"] = manager.format_selections(selections)
        out["result"] = result.formatted
        out["formatted"] = format_sparql_result(manager, result, selections)

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
    assert train_cfg.type in ["skeleton", "both"], (
        "Main run should either be of type 'skeleton' or 'both'"
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
    trace = run_on_file and args.trace

    for i, ipt in enumerate(inputs):
        id = extract_field(ipt, "id") or "unknown"

        gold_sparql = None
        if oracle_skeleton:
            gold_sparql = extract_field(ipt, "sparql")
            assert gold_sparql is not None, (
                f"--oracle-skeleton requires a 'sparql' field on every sample, "
                f"missing on input {i:,} (id={id})"
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
