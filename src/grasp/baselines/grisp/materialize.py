import argparse
import os
import random
from typing import Iterator

from tqdm import tqdm
from universal_ml_utils.io import dump_jsonl, load_jsonl
from universal_ml_utils.logging import get_logger, setup_logging

from grasp.baselines.grisp.data import (
    GRISPMaterializedSample,
    GRISPSample,
    SelectionSample,
    load_samples,
    prepare_selection,
    prepare_skeleton,
)
from grasp.configs import KgConfig, KgInfo
from grasp.manager import KgManager, load_kg_manager
from grasp.utils import get_available_knowledge_graphs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize GRISP data for a given number of materializations per sample"
    )
    parser.add_argument(
        "knowledge_graph",
        type=str,
        choices=get_available_knowledge_graphs(),
        help="Knowledge graph to use",
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input file containing data samples",
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to the output file to save materialized data",
    )
    parser.add_argument(
        "num_materializations",
        type=int,
        help="Number of materializations per sample",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default=None,
        help="Custom SPARQL endpoint for the knowledge graph",
    )
    parser.add_argument(
        "--skeleton-p",
        type=float,
        default=0.2,
        help="Augmentation probability for skeletons",
    )
    parser.add_argument(
        "--drop-infos-p",
        type=float,
        default=0.05,
        help="Probability of dropping all alternative infos (emulates endpoint failure)",
    )
    parser.add_argument(
        "--drop-target-p",
        type=float,
        default=0.1,
        help="Probability of removing the correct alternative (trains None-of-above selection)",
    )
    parser.add_argument(
        "--shuffle-alts-p",
        type=float,
        default=0.1,
        help="Probability of shuffling alternatives within each type bucket",
    )
    parser.add_argument(
        "--constrain-p",
        type=float,
        default=0.1,
        help="Probability of constraining the alternatives with a SPARQL query "
        "against the endpoint (like at inference time); expensive, so keep low",
    )
    parser.add_argument(
        "--val-output-file",
        type=str,
        default=None,
        help="Path to the output file to save validation materialized data",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Proportion of data to use for validation split (only if --val-output-file is provided)",
    )
    parser.add_argument(
        "--is-val",
        action="store_true",
        help="Whether the input file is a validation set",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=22,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="Qwen/Qwen3-Embedding-0.6B",
        help="Text embedding model to use for embedding index",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it exists",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level",
    )
    return parser.parse_args()


def materialize_sample(
    sample: GRISPSample,
    manager: KgManager,
    n: int,
    is_val: bool = False,
    skeleton_p: float = 0.2,
    drop_infos_p: float = 0.05,
    drop_target_p: float = 0.1,
    shuffle_alts_p: float = 0.1,
    constrain_p: float = 0.1,
) -> GRISPMaterializedSample:
    if is_val:
        n = 1

    skeletons = [prepare_skeleton(sample, is_val, skeleton_p) for _ in range(n)]

    if sample.has_placeholders:
        selections = []
        for _ in range(n):
            messages, options, target = prepare_selection(
                sample,
                manager,
                is_val,
                skeleton_p,
                drop_infos_p,
                drop_target_p,
                shuffle_alts_p,
                constrain_p,
            )
            selections.append(
                SelectionSample(
                    messages=messages, options=options, target=target
                )
            )
    else:
        selections = []

    return GRISPMaterializedSample(
        skeletons=skeletons,
        selections=selections,
    )


def main(args: argparse.Namespace) -> None:
    # to show info from kg manager
    setup_logging(args.log_level)
    logger = get_logger("MATERIALIZATION", args.log_level)

    if args.is_val and args.val_output_file is not None:
        raise ValueError("Cannot specify --val-output-file when --is-val is set.")

    samples = load_samples([args.input_file])
    skip = 0
    if os.path.exists(args.output_file) and not args.overwrite:
        skip = len(load_jsonl(args.output_file))
        logger.info(
            f"Output file {args.output_file} already exists, "
            f"skipping {skip:,} existing materialized samples"
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
            f"skipping {val_skip:,} existing materialized samples"
        )

    desc = "validation" if args.is_val else "training"

    if args.val_output_file is not None:
        val_size = int(len(samples) * args.val_split)
        random.seed(args.seed)
        random.shuffle(samples)
        val_samples = samples[val_skip:val_size]
        samples = samples[val_size + skip :]
    else:
        val_samples = None
        samples = samples[skip:]

    # only override the kg's own info if an endpoint was given, an explicit
    # KgInfo(endpoint=None) counts as set and would discard it
    config = KgConfig(
        kg=args.knowledge_graph,
        info=KgInfo(endpoint=args.endpoint) if args.endpoint else None,
    )
    manager = load_kg_manager(config)
    manager.load_models()

    def materialize_train() -> Iterator[dict]:
        for sample in tqdm(samples, desc=f"Materializing {desc} samples"):
            assert isinstance(sample, GRISPSample), (
                "Expected non-materialized GRISP sample"
            )
            materialized_sample = materialize_sample(
                sample,
                manager,
                args.num_materializations,
                args.is_val,
                args.skeleton_p,
                args.drop_infos_p,
                args.drop_target_p,
                args.shuffle_alts_p,
                args.constrain_p,
            )

            yield materialized_sample.model_dump()

    dump_jsonl(
        materialize_train(),
        args.output_file,
        "w" if skip == 0 else "a",
    )

    if args.val_output_file is None or val_samples is None:
        return

    def materialize_val() -> Iterator[dict]:
        for sample in tqdm(val_samples, desc="Materializing validation samples"):
            assert isinstance(sample, GRISPSample), (
                "Expected non-materialized GRISP sample"
            )
            materialized_sample = materialize_sample(sample, manager, 1, is_val=True)
            yield materialized_sample.model_dump()

    dump_jsonl(
        materialize_val(),
        args.val_output_file,
        "w" if val_skip == 0 else "a",
    )


if __name__ == "__main__":
    main(parse_args())
