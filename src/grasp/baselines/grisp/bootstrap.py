import argparse
import os
import random
import re
from logging import Logger
from typing import Iterator

from tqdm import tqdm
from universal_ml_utils.configuration import load_config
from universal_ml_utils.io import dump_jsonl, load_jsonl
from universal_ml_utils.logging import get_logger, setup_logging
from universal_ml_utils.ops import consume_generator

from grasp.baselines.grisp.data import (
    RESULT_UNAVAILABLE,
    RESULT_UNRESOLVED,
    GRISPMaterializedSample,
    GRISPSample,
    ValidationSample,
    get_improvement_prompt,
    get_skeleton_prompt,
    get_validation_prompt,
    load_samples,
    materialize_skeleton,
    materialize_sparql,
)
from grasp.baselines.grisp.run import (
    GRISPModel,
    GRISPRunConfig,
    SelectionOutcome,
    generate_skeletons_from_prompt,
    load_model_and_tokenizer,
    select_iris,
)
from grasp.baselines.grisp.train import GRISPTrainConfig
from grasp.baselines.grisp.utils import load_sparql_parser
from grasp.evaluate import get_result_or_error
from grasp.manager import KgManager, load_kg_manager
from grasp.sparql.metrics import f1_score
from grasp.sparql.types import AskResult, SelectResult
from grasp.tasks.utils import prepare_sparql_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap GRISP validation and improvement training data "
        "from a trained GRISP model"
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to GRISP run configuration",
    )
    parser.add_argument(
        "run_directory",
        type=str,
        help="Path to the trained GRISP run directory",
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the GRISP-converted JSONL file (output of "
        "grasp.baselines.grisp.data) -- the same file materialize.py consumes",
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to output JSONL file of materialized (training) samples",
    )
    parser.add_argument(
        "num_bootstraps",
        type=int,
        help="Number of times to run the whole bootstrap procedure per question "
        "(analogous to materialize.py's num_materializations): each run samples "
        "one skeleton, resolves it, and builds validation/improvement examples. "
        "Duplicate skeletons are deduplicated within a question.",
    )
    parser.add_argument(
        "--val-output-file",
        type=str,
        default=None,
        help="If set, split the input into train/val exactly like materialize.py "
        "(shuffle with --seed, take --val-split as val) and write the "
        "deterministic validation set here",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Fraction of the input used for validation (only with "
        "--val-output-file); must match the value passed to materialize.py",
    )
    parser.add_argument(
        "--selection-run",
        type=str,
        default=None,
        help="Path to a separate selection model run directory",
    )
    parser.add_argument(
        "--valid-threshold",
        type=float,
        default=0.5,
        help="F1 at or above which a query is labeled valid (locates the "
        "assistant token; the soft target stays [f1, 1 - f1])",
    )
    parser.add_argument(
        "--keep-threshold",
        type=float,
        default=1.0,
        help="F1 at or above which an improvement sample is a 'keep' (skeleton "
        "-> itself); below it the target is the gold skeleton ('fix')",
    )
    parser.add_argument(
        "--keep-as-gold-prob",
        type=float,
        default=0.2,
        help="For keep-eligible candidates (F1 >= keep-threshold), probability of "
        "using the gold skeleton as the target instead of the skeleton itself, "
        "so the high-F1 region also gets some canonical-form (gold) targets.",
    )
    parser.add_argument(
        "--target-alias-prob",
        type=float,
        default=0.2,
        help="Per-placeholder probability of using a random alias instead of the "
        "canonical label in a gold improvement target, matching the skeleton "
        "task's target augmentation (skeleton_p). 0 keeps targets canonical.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=22,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--f1-timeout",
        type=float,
        default=60.0,
        help="Timeout in seconds for executing queries to compute F1. Should be "
        "generous (longer than the inference budget) so that slow-but-correct "
        "queries are labeled valid rather than invalid.",
    )
    parser.add_argument(
        "--drop-result-prob",
        type=float,
        default=0.1,
        help="Probability of masking the result preview as unavailable in a "
        "validation sample (same target), to make the validator robust to "
        "query timeouts / backend outages at inference time",
    )
    parser.add_argument(
        "--exact-after",
        type=int,
        default=1024,
        help="Use exact F1 for results larger than this many rows",
    )
    parser.add_argument(
        "--max-result-rows",
        type=int,
        default=10_000_000,
        help="Maximum number of result rows to fetch when computing F1",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cpu, cuda, etc.)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="Data type for model weights",
    )
    parser.add_argument(
        "--is-val",
        action="store_true",
        help="Produce a deterministic validation set: greedy decoding, one "
        "sample per question, canonical (un-sampled) gold targets, no result "
        "masking, and no stratified subsampling (all examples kept). Run on a "
        "benchmark split disjoint from training to avoid leakage.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file(s) if they exist. Without this, an "
        "existing output is resumed: already-written records are skipped and "
        "new ones appended.",
    )
    parser.add_argument(
        "--all-loggers",
        action="store_true",
        help="Set log level for all loggers instead of just the main one",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level",
    )
    return parser.parse_args()


def structural_pattern(skeleton: str) -> str:
    return re.sub(r"<iri>[^<]*</iri>", "<IRI>", skeleton)


def sample_skeleton(
    skeleton_model: GRISPModel,
    tokenizer,
    run_cfg: GRISPRunConfig,
    question: str,
    manager: KgManager,
    parser,
    logger: Logger,
):
    # sample a single skeleton (no beam search); diversity comes from repeated
    # sampling across the num_bootstraps loop
    skeletons = generate_skeletons_from_prompt(
        skeleton_model,
        tokenizer,
        run_cfg,
        get_skeleton_prompt(manager.kg, question),
        parser,
        logger,
        n=1,
        top_k=1,
    )
    return skeletons[0] if skeletons else None


def resolve_skeleton(
    skeleton,
    selection_model: GRISPModel,
    selection_tokenizer,
    run_cfg: GRISPRunConfig,
    question: str,
    manager: KgManager,
    logger: Logger,
) -> SelectionOutcome:
    # drive selection to completion, discarding the intermediate events
    return consume_generator(
        select_iris(
            skeleton,
            selection_model,
            selection_tokenizer,
            run_cfg,
            question,
            manager,
            logger,
        )
    )


def main(args: argparse.Namespace) -> None:
    logger = get_logger("GRISP BOOTSTRAP", args.log_level)
    if args.all_loggers:
        setup_logging(args.log_level)

    if args.is_val and args.val_output_file is not None:
        raise ValueError("Cannot specify --val-output-file when --is-val is set.")

    train_cfg = GRISPTrainConfig(
        **load_config(os.path.join(args.run_directory, "config.yaml"))
    )

    run_cfg = GRISPRunConfig(**load_config(args.config))
    # mine from vanilla GRISP behavior: never run the learned validation /
    # improvement self-correction loop while mining (disabling improvement also
    # disables the validation scoring it gates). Everything else (constrain,
    # backtrack, check_empty) is taken from the run config -- with constrain and
    # check_empty enabled, hard questions naturally yield partial skeletons
    # (failed selection / empty-then-backtracked queries) instead of always
    # resolving to something.
    run_cfg.improve_skeletons = False

    logger.info(f"Loading model from {args.run_directory}")
    model, tokenizer = load_model_and_tokenizer(
        args.run_directory, args.device, args.dtype, logger
    )
    skeleton_model = GRISPModel(model, run_cfg.skeleton_disable_adapter)

    if train_cfg.type == "skeleton" and args.selection_run is not None:
        logger.info(f"Loading selection model from {args.selection_run}")
        sel_model, selection_tokenizer = load_model_and_tokenizer(
            args.selection_run, args.device, args.dtype, logger
        )
        selection_model = GRISPModel(sel_model, run_cfg.selection_disable_adapter)
    else:
        selection_model = GRISPModel(model, run_cfg.selection_disable_adapter)
        selection_tokenizer = tokenizer

    manager = load_kg_manager(run_cfg.knowledge_graph)
    manager.load_models()
    parser = load_sparql_parser()

    samples = load_samples([args.input_file])
    logger.info(f"Loaded {len(samples):,} samples from {args.input_file}")

    # resume support: one output record is written per input sample (empty
    # records for skipped samples), so the number of already-written records is
    # exactly the number of input samples consumed. A stopped run can be
    # continued by re-running the same command; existing records are skipped and
    # new ones appended (mirrors materialize.py).
    skip = 0
    if os.path.exists(args.output_file) and not args.overwrite:
        skip = len(load_jsonl(args.output_file))
        logger.info(
            f"Output file {args.output_file} already exists, "
            f"skipping {skip:,} existing records"
        )

    val_skip = 0
    if (
        args.val_output_file is not None
        and os.path.exists(args.val_output_file)
        and not args.overwrite
    ):
        val_skip = len(load_jsonl(args.val_output_file))
        logger.info(
            f"Validation output file {args.val_output_file} already exists, "
            f"skipping {val_skip:,} existing records"
        )

    # split exactly like materialize.py: shuffle with the same seed, take the
    # first val_split fraction as validation. With the same input file, --seed
    # and --val-split, this reproduces materialize.py's train/val partition.
    # Resume offsets (skip / val_skip) are applied on top of the split.
    if args.val_output_file is not None:
        val_size = int(len(samples) * args.val_split)
        random.seed(args.seed)
        random.shuffle(samples)
        val_samples = samples[val_skip:val_size]
        train_samples = samples[val_size + skip :]
        logger.info(
            f"Split into {val_size:,} val / {len(samples) - val_size:,} train "
            f"(seed={args.seed}, val_split={args.val_split}); "
            f"{len(train_samples):,} train / {len(val_samples):,} val remaining"
        )
    else:
        random.seed(args.seed)
        val_samples = None
        train_samples = samples[skip:]

    # cache query -> result to avoid re-executing identical queries
    result_cache: dict[str, SelectResult | AskResult | None] = {}

    def execute(sparql: str) -> SelectResult | AskResult | None:
        if sparql not in result_cache:
            result, err = get_result_or_error(
                sparql,
                manager.endpoint,
                args.f1_timeout,
                args.max_result_rows,
            )
            if err is not None:
                logger.debug(f"Error executing query for F1:\n{err}\n\n{sparql}")
            result_cache[sparql] = result
        return result_cache[sparql]

    # cache the inference-style (truncated) preview + selections per query, so
    # the validator/improvement evidence matches exactly what inference shows
    preview_cache: dict[str, tuple[str, str]] = {}

    def format_preview(sparql: str) -> tuple[str, str]:
        if sparql not in preview_cache:
            result, selections = prepare_sparql_result(
                sparql,
                manager.kg,
                [manager],
                max_rows=10,
                max_columns=10,
                request_timeout=(3.5, 6.0),
                read_timeout=3.0,
            )
            preview = (
                result.formatted if result.result is not None else RESULT_UNAVAILABLE
            )
            preview_cache[sparql] = (manager.format_selections(selections), preview)
        return preview_cache[sparql]

    def process(samples: list, is_val: bool, stats: dict) -> Iterator[dict]:
        # yields exactly one materialized record per input sample (an empty
        # record when nothing could be mined), so records are written as they
        # are produced and a stopped run can be resumed by record count.
        # validation/eval set is fully deterministic: greedy decoding, one
        # sample per question, canonical (un-sampled) targets, no masking
        cfg = run_cfg.model_copy(update={"do_sample": False}) if is_val else run_cfg
        n_bootstraps = 1 if is_val else args.num_bootstraps
        alias_p = 0.0 if is_val else args.target_alias_prob
        drop_p = 0.0 if is_val else args.drop_result_prob
        keep_gold_p = 0.0 if is_val else args.keep_as_gold_prob
        desc = "val" if is_val else "train"

        for sample in tqdm(samples, desc=f"Bootstrapping ({desc})"):
            assert isinstance(sample, GRISPSample)

            # reconstruct gold query + canonical skeleton from the GRISP parts
            gold_sparql = manager.fix_prefixes(materialize_sparql(sample.sparql))
            gold_result = execute(gold_sparql)
            if gold_result is None or gold_result.is_empty:
                stats["skipped"] += 1
                # bootstrap only produces validation/improvement data, so the
                # skeleton/selection arrays are always empty -- omit them
                yield GRISPMaterializedSample().model_dump(
                    exclude={"skeletons", "selections"}
                )
                continue
            gold_nl = materialize_skeleton(sample.sparql, is_val=True)

            # paraphrase augmentation for train (matching materialize_sample)
            question = (
                sample.questions[0] if is_val else random.choice(sample.questions)
            )

            # stack all of this question's mined variants into one record, so the
            # materialized datasets cycle through them across epochs like the
            # epoch variants produced by materialize.py
            validations: list[ValidationSample] = []
            improvements: list = []
            seen_skel: set[str] = set()
            seen_val: set[str] = set()
            gold_pattern = structural_pattern(gold_nl)

            for _ in range(n_bootstraps):
                skeleton = sample_skeleton(
                    skeleton_model, tokenizer, cfg, question, manager, parser, logger
                )
                if skeleton is None:
                    continue
                skeleton_nl = skeleton.nl_sparql  # capture before resolution mutates
                if skeleton_nl in seen_skel:
                    continue
                seen_skel.add(skeleton_nl)

                if structural_pattern(skeleton_nl) == gold_pattern:
                    stats["skel_struct_same"] += 1
                else:
                    stats["skel_struct_diff"] += 1

                outcome = resolve_skeleton(
                    skeleton,
                    selection_model,
                    selection_tokenizer,
                    cfg,
                    question,
                    manager,
                    logger,
                )

                if outcome.sparql is not None:
                    pred_sparql = outcome.sparql
                    pred_result = execute(pred_sparql)
                    f1 = (
                        f1_score(pred_result, gold_result, args.exact_after)
                        if pred_result is not None
                        else 0.0
                    )
                    stats["val_f1_bins"][min(int(f1 * 10), 10)] += 1
                    # outcome.resolved is the raw materialized skeleton (selected
                    # IRIs substituted, no prefix declarations), shown as the
                    # improvement prompt's "Resolved query"
                    resolved_sparql = outcome.resolved
                    selections_str, preview = format_preview(pred_sparql)

                    if pred_sparql not in seen_val:
                        seen_val.add(pred_sparql)
                        val_result = preview
                        if random.random() < drop_p:
                            val_result = RESULT_UNAVAILABLE
                        validations.append(
                            ValidationSample(
                                messages=get_validation_prompt(
                                    manager.kg,
                                    question,
                                    pred_sparql,
                                    selections_str,
                                    val_result,
                                    valid=f1 >= args.valid_threshold,
                                ),
                                target_dist=[f1, 1.0 - f1],
                            )
                        )
                else:
                    # partial skeleton: no executable query (no validation);
                    # evidence is the partially resolved query, its deepest
                    # selections, and an unresolved-result marker
                    f1 = None
                    resolved_sparql = outcome.resolved
                    selections_str = manager.format_selections(outcome.selections)
                    preview = RESULT_UNRESOLVED

                # improvement target: keep (itself) for good skeletons, else fix
                # toward the gold skeleton (occasionally for keep-eligible too)
                if f1 is not None and f1 >= args.keep_threshold:
                    to_gold = (
                        skeleton_nl.strip() != gold_nl.strip()
                        and random.random() < keep_gold_p
                    )
                elif skeleton_nl.strip() != gold_nl.strip():
                    to_gold = True
                else:
                    continue  # nothing to learn (skeleton already canonical)

                if to_gold:
                    # alias-sample the gold target like task 1's skeleton targets
                    target = materialize_skeleton(
                        sample.sparql, is_val=False, p=alias_p
                    )
                else:
                    target = skeleton_nl  # keep, used verbatim
                    stats["keep"] += 1

                improvements.append(
                    get_improvement_prompt(
                        manager.kg,
                        question,
                        skeleton_nl,
                        resolved_sparql,
                        selections_str,
                        preview,
                        improved=target,
                    )
                )

            if not validations and not improvements:
                stats["skipped"] += 1
            else:
                stats["records"] += 1
                stats["validations"] += len(validations)
                stats["improvements"] += len(improvements)

            yield GRISPMaterializedSample(
                validations=validations,
                improvements=improvements,
            ).model_dump(exclude={"skeletons", "selections"})

    def run(samples: list, is_val: bool, output_file: str, append: bool) -> None:
        desc = "val" if is_val else "train"
        stats = {
            "records": 0,
            "validations": 0,
            "improvements": 0,
            "keep": 0,
            "skipped": 0,
            "skel_struct_same": 0,
            "skel_struct_diff": 0,
            # 11 bins: index i covers [i/10, (i+1)/10), except index 10 = 1.0
            "val_f1_bins": [0] * 11,
        }
        dump_jsonl(
            process(samples, is_val, stats),
            output_file,
            "a" if append else "w",
        )
        total_skel = stats["skel_struct_same"] + stats["skel_struct_diff"]
        skel_div = stats["skel_struct_diff"] / total_skel if total_skel > 0 else 0.0
        bins = stats["val_f1_bins"]
        total_val_f1 = sum(bins)
        partial = sum(bins[1:10])
        val_partial_frac = partial / total_val_f1 if total_val_f1 > 0 else 0.0
        f1_hist = "  ".join(f"{i / 10:.1f}:{bins[i]:,}" for i in range(11))
        logger.info(
            f"[{desc}] {stats['records']:,} non-empty records: "
            f"{stats['validations']:,} validation, "
            f"{stats['improvements']:,} improvement "
            f"({stats['improvements'] - stats['keep']:,} fix, "
            f"{stats['keep']:,} keep), {stats['skipped']:,} skipped"
        )
        logger.info(
            f"[{desc}] skeleton diversity: {skel_div:.1%} structurally novel vs gold "
            f"({stats['skel_struct_diff']:,}/{total_skel:,} unique skeletons sampled); "
            f"if near 0% the bootstrap adds no new structural signal"
        )
        logger.info(
            f"[{desc}] validation F1 histogram (bin:count): {f1_hist}; "
            f"partial={partial:,} ({val_partial_frac:.1%}); "
            f"if partial is near 0% the validator trains as binary classifier only"
        )

    run(train_samples, args.is_val, args.output_file, append=skip > 0)
    if args.val_output_file is not None and val_samples is not None:
        run(val_samples, True, args.val_output_file, append=val_skip > 0)


if __name__ == "__main__":
    main(parse_args())
