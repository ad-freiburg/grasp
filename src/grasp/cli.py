import argparse
import json
import os
import random
import sys
from datetime import datetime
from importlib import metadata

from search_rdf.model import SentenceTransformerModel
from termcolor import colored
from tqdm import tqdm
from universal_ml_utils.configuration import load_config
from universal_ml_utils.io import (
    dump_json,
    dump_jsonl,
    dump_text,
    load_json,
    load_jsonl,
    load_text,
)
from universal_ml_utils.logging import get_logger, setup_logging
from universal_ml_utils.ops import consume_generator, extract_field

from grasp.build import get_data
from grasp.build.data import merge_kgs
from grasp.build.index import build_index
from grasp.build.shapes import build_shapes
from grasp.configs import (
    GraspConfig,
    JudgeConfig,
    NotesFromExplorationConfig,
    NotesFromOutputsConfig,
    NotesFromSamplesConfig,
    NotesGenerateQuestionsConfig,
    ServerConfig,
    ShapeConfig,
)
from grasp.core import generate, load_notes, setup
from grasp.evaluate import evaluate_f1, evaluate_with_expert, evaluate_with_judge
from grasp.examples import load_example_indices, task_to_index
from grasp.functions import find_manager
from grasp.notes import (
    generate_questions,
    take_notes_from_exploration,
    take_notes_from_outputs,
    take_notes_from_samples,
)
from grasp.server import serve
from grasp.shapes import ShapeIndex, ShapeSample
from grasp.tasks import Task, get_task
from grasp.utils import (
    format_trace,
    get_available_knowledge_graphs,
    get_index_dir,
    is_invalid_output,
    link,
    parse_key_value_pairs,
)


def add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "config",
        type=str,
        help="Path to the GRASP configuration file",
    )


def add_task_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        choices=[task.value for task in Task],
        default=Task.SPARQL_QA.value,
        help="Task to run/consider",
    )


def add_overwrite_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output",
    )


def parse_args() -> argparse.Namespace:
    available_kgs = get_available_knowledge_graphs()

    parser = argparse.ArgumentParser(
        prog="grasp",
        description="GRASP: Generic Reasoning and SPARQL generation across Knowledge Graphs",
    )

    subparsers = parser.add_subparsers(
        title="commands",
        description="Available commands",
        dest="command",
        required=True,
    )

    # run GRASP server
    server_parser = subparsers.add_parser("serve", help="Start a GRASP server")
    add_config_arg(server_parser)

    # run GRASP on a single input
    run_parser = subparsers.add_parser(
        "run",
        help="Run GRASP on a single input",
    )
    add_config_arg(run_parser)
    run_parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Input for task (e.g., a question for 'sparql-qa'), "
        "if not given, read from stdin",
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
        default=None,
        help="Field to extract input from (if None, a task-specific default is used, "
        "but only if input format is 'json')",
    )
    add_task_arg(run_parser)

    # run GRASP on file with inputs
    file_parser = subparsers.add_parser(
        "file",
        help="Run GRASP on a file with inputs in JSONL format",
    )
    add_config_arg(file_parser)
    file_parser.add_argument(
        "--progress",
        action="store_true",
        help="Show a progress bar",
    )
    file_parser.add_argument(
        "-i",
        "--input-file",
        type=str,
        help="Path to file in JSONL format to run GRASP on, if not given, read JSONL from stdin",
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
        default=None,
        help="Field to extract input from (if None, a task-specific default is used)",
    )
    file_parser.add_argument(
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
        help="Consider None outputs as invalid when retrying failed inputs",
    )
    add_task_arg(file_parser)
    add_overwrite_arg(file_parser)

    # run GRASP note taking
    note_parser = subparsers.add_parser(
        "notes",
        help="Take notes on interactions of GRASP with one or more knowledge graphs",
    )

    # add second level for note parser
    note_subparsers = note_parser.add_subparsers(
        title="note commands",
        description="Available note commands",
        dest="note_command",
        required=True,
    )

    note_samples_parser = note_subparsers.add_parser(
        "samples",
        help="Take notes for a task and one or more knowledge graphs "
        "by running GRASP on exemplary task samples",
    )
    add_config_arg(note_samples_parser)
    note_samples_parser.add_argument(
        "output_dir",
        type=str,
        help="Save note taking results in this directory",
    )
    add_task_arg(note_samples_parser)
    add_overwrite_arg(note_samples_parser)

    note_interactions_parser = note_subparsers.add_parser(
        "outputs",
        help="Take notes from existing outputs / runs of GRASP",
    )
    add_config_arg(note_interactions_parser)
    note_interactions_parser.add_argument(
        "output_dir",
        type=str,
        help="Save note taking results in this directory",
    )
    add_task_arg(note_interactions_parser)
    add_overwrite_arg(note_interactions_parser)

    note_explore_parser = note_subparsers.add_parser(
        "explore",
        help="Take notes for a task and one or more knowledge graphs "
        "by exploring the knowledge graphs (without any task samples or outputs)",
    )
    add_config_arg(note_explore_parser)
    note_explore_parser.add_argument(
        "output_dir",
        type=str,
        help="Save note taking results in this directory",
    )
    add_overwrite_arg(note_explore_parser)

    note_generate_parser = note_subparsers.add_parser(
        "generate-questions",
        help="Generate plausible user questions over the configured "
        "knowledge graphs as bootstrap samples for note taking",
    )
    add_config_arg(note_generate_parser)
    note_generate_parser.add_argument(
        "output_dir",
        type=str,
        help="Save generated samples in this directory",
    )
    add_overwrite_arg(note_generate_parser)

    # evaluate GRASP output
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate GRASP output against a reference file (only for 'sparql-qa' task)",
    )

    eval_subparsers = eval_parser.add_subparsers(
        title="evaluation commands",
        description="Available evaluation commands",
        dest="evaluate_command",
        required=True,
    )
    eval_f1_parser = eval_subparsers.add_parser(
        "f1",
        help="Evaluate GRASP output using F1 score based on query results",
    )
    eval_f1_parser.add_argument(
        "knowledge_graph",
        type=str,
        choices=available_kgs,
        help="Knowledge graph the input questions refer to",
    )
    eval_f1_parser.add_argument(
        "input_file",
        type=str,
        help="Path to file with question-sparql pairs in JSONL format",
    )
    eval_f1_parser.add_argument(
        "prediction_file",
        type=str,
        help="Path to file with GRASP predictions as produced by the 'file' command",
    )
    eval_f1_parser.add_argument(
        "--endpoint",
        type=str,
        help="SPARQL endpoint to use for evaluation",
    )
    eval_f1_parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Maximum duration for a single query in seconds",
    )
    eval_f1_parser.add_argument(
        "--sparql-result-max-rows",
        type=int,
        default=10_000_000,
        help="Maximum number of SPARQL result rows to parse before failing evaluation",
    )
    eval_f1_parser.add_argument(
        "--exact-after",
        type=int,
        default=1024,
        help="Result size after which exact F1 score instead of assignment F1 score "
        "is used (due to performance reasons)",
    )
    eval_f1_parser.add_argument(
        "--fix-prefixes",
        action="store_true",
        help="Try to fix missing prefix issues in target and prediction SPARQL queries "
        "before evaluating them",
    )

    eval_judge_parser = eval_subparsers.add_parser(
        "judge",
        help="Evaluate GRASP outputs by picking the best using a judge model",
    )
    eval_judge_parser.add_argument(
        "config",
        type=str,
        help="Path to the GRASP configuration file (used for the judge)",
    )
    eval_judge_parser.add_argument(
        "input_file",
        type=str,
        help="Path to file with inputs in JSONL format",
    )
    eval_judge_parser.add_argument(
        "prediction_files",
        type=str,
        nargs="+",
        help="Paths to files with GRASP predictions as produced by the 'file' command",
    )
    eval_judge_parser.add_argument(
        "evaluation_file",
        type=str,
        help="Path to file to write the evaluation results to",
    )
    eval_judge_parser.add_argument(
        "--reformat-sparql",
        action="store_true",
        help="Whether to re-run candidate formatting based on the SPARQL "
        "query before judging.",
    )
    eval_judge_parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Maximum duration for a single query in seconds if reformatting is enabled",
    )
    eval_judge_parser.add_argument(
        "--sparql-result-max-rows",
        type=int,
        default=10_000_000,
        help="Maximum number of SPARQL result rows to parse before failing reformatting",
    )
    eval_judge_parser.add_argument(
        "--with-ground-truth-reference",
        action="store_true",
        help="Show the formatted ground-truth SPARQL result to the judge.",
    )

    eval_expert_parser = eval_subparsers.add_parser(
        "expert",
        help="Launch a blind expert evaluation Streamlit app over GRASP outputs",
    )
    eval_expert_parser.add_argument(
        "input_file",
        type=str,
        help="Path to file with inputs in JSONL format",
    )
    eval_expert_parser.add_argument(
        "prediction_files",
        type=str,
        nargs="+",
        help="Paths to files with GRASP predictions as produced by the 'file' command",
    )
    eval_expert_parser.add_argument(
        "evaluation_file",
        type=str,
        help="Path to file to read/write the expert evaluation JSON",
    )
    eval_expert_parser.add_argument(
        "--kg-config",
        type=str,
        default=None,
        help="Optional path to a KgConfig YAML; if provided, ground-truth SPARQL "
        "from the input file is executed against the corresponding KG",
    )
    eval_expert_parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Optional port for the Streamlit server",
    )
    eval_expert_parser.add_argument(
        "--base-url-path",
        type=str,
        default="/expert",
        help="Base URL path for the Streamlit app; defaults to /expert",
    )

    eval_parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Rerun failed evaluations",
    )
    add_overwrite_arg(eval_parser)

    # get data for GRASP indices
    data_parser = subparsers.add_parser(
        "data",
        help="Get data for knowledge graph index",
    )
    data_parser.add_argument(
        "knowledge_graph",
        type=str,
        help="Knowledge graph to get data for",
    )
    data_parser.add_argument(
        "index",
        choices=["entities", "properties", "literals"],
        help="Index to get data for",
    )
    data_parser.add_argument(
        "--endpoint",
        type=str,
        help="SPARQL endpoint of the knowledge graph "
        "(if not given, the endpoint at qlever.cs.uni-freiubrg.de/api/<kg> is used)",
    )
    data_parser.add_argument(
        "--index-sparql",
        type=str,
        help="Path to file with custom index SPARQL query",
    )
    data_parser.add_argument(
        "--query-parameters",
        type=str,
        nargs="*",
        help="Extra query parameters sent to the knowledge graph endpoint",
    )
    data_parser.add_argument(
        "--query-headers",
        type=str,
        nargs="*",
        help="Extra HTTP headers sent to the knowledge graph endpoint when querying",
    )
    data_parser.add_argument(
        "--replace",
        type=str,
        nargs="*",
        help="Variables with format {key} in SPARQL queries to replace with values in format key:value",
    )
    data_parser.add_argument(
        "--add-id-as-label",
        type=str,
        default="auto",
        choices=["auto", "never", "always", "empty"],
        help="When to add a label fallback based on ID",
    )
    data_parser.add_argument(
        "--data-file",
        type=str,
        default=None,
        help="Path to file with SPARQL results in JSON format (skip live query)",
    )
    add_overwrite_arg(data_parser)

    # merge multiple knowledge graphs
    merge_parser = subparsers.add_parser(
        "merge",
        help="Merge data from multiple knowledge graphs. The first knowledge graph is the primary one, "
        "to which data from the other knowledge graphs is added. Therefore, the merged knowledge graph will "
        "have the same number of entities and properties as the first knowledge graph.",
    )
    merge_parser.add_argument(
        "knowledge_graphs",
        type=str,
        nargs="+",
        choices=available_kgs,
        help="Knowledge graphs to merge",
    )
    merge_parser.add_argument(
        "index",
        choices=["entities", "properties", "literals"],
        help="Index to merge data for",
    )
    merge_parser.add_argument(
        "knowledge_graph",
        type=str,
        help="Name of the merged knowledge graph",
    )
    add_overwrite_arg(merge_parser)

    # build GRASP indices
    index_parser = subparsers.add_parser(
        "index",
        help="Build an index for a knowledge graph",
    )
    index_parser.add_argument(
        "knowledge_graph",
        type=str,
        choices=available_kgs,
        help="Knowledge graph to build indices for",
    )
    index_parser.add_argument(
        "index",
        choices=["entities", "properties", "literals", "shapes"],
        help="Index to build",
    )
    index_parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Path to GRASP configuration file (required when building the shapes index)",
    )
    index_parser.add_argument(
        "--index-type",
        type=str,
        choices=["auto", "keyword", "fuzzy", "embedding"],
        default="auto",
        help="Type of index to build (not applicable for shapes)",
    )
    index_parser.add_argument(
        "--emb-model",
        type=str,
        default="Qwen/Qwen3-Embedding-0.6B",
        help="Embedding model to use when building embedding index",
    )
    index_parser.add_argument(
        "--emb-device",
        type=str,
        default=None,
        help="Device to use for embedding model when building embedding index",
    )
    index_parser.add_argument(
        "--emb-dim",
        type=int,
        help="Embedding dimensionality when building embedding index",
    )
    index_parser.add_argument(
        "--emb-batch-size",
        type=int,
        default=256,
        help="Batch size when building embedding index",
    )
    index_parser.add_argument(
        "--max-classes",
        type=int,
        default=500,
        help="Maximum number of classes to profile (shapes only)",
    )
    add_overwrite_arg(index_parser)

    # build example index
    example_parser = subparsers.add_parser(
        "examples",
        help="Build an example index used for few-shot learning (only for 'sparql-qa' task)",
    )
    example_parser.add_argument(
        "examples_file",
        type=str,
        help="Path to file with examples in JSONL format",
    )
    example_parser.add_argument(
        "output_dir",
        type=str,
        help="Directory to save the example index",
    )
    example_parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for building the example index",
    )
    example_parser.add_argument(
        "--emb-model",
        type=str,
        default="Qwen/Qwen3-Embedding-0.6B",
        help="Embedding model to use when building examples index",
    )
    example_parser.add_argument(
        "description",
        type=str,
        help="Short description of this example index (shown in system prompt)",
    )
    add_task_arg(example_parser)
    add_overwrite_arg(example_parser)

    # AI-assisted setup for a knowledge graph
    setup_parser = subparsers.add_parser(
        "setup",
        help="Configure a knowledge graph using an AI agent",
    )
    setup_parser.add_argument(
        "knowledge_graph",
        type=str,
        help="Knowledge graph to configure",
    )
    setup_parser.add_argument(
        "phase",
        choices=[
            "standard",
            "info",
            "entities",
            "properties",
            "literals",
            "indices",
            "shapes",
        ],
        help="Setup phase to run: 'standard' runs info+indices (entities+properties+literals); 'indices' runs the three index phases only",
    )
    setup_parser.add_argument(
        "config",
        type=str,
        help="Path to GRASP configuration file",
    )
    setup_parser.add_argument(
        "--notes",
        type=str,
        default=None,
        help="Notes passed to the setup agent (single-phase runs only: info, entities, properties, literals, shapes)",
    )

    # visualize trace from GRASP output
    show_parser = subparsers.add_parser(
        "show",
        help="Visualize the interaction trace from GRASP output",
    )
    show_parser.add_argument(
        "file",
        nargs="?",
        type=str,
        help="File with GRASP output to visualize, if not given, read from stdin",
    )
    show_parser.add_argument(
        "--skip-system",
        action="store_true",
        help="Skip system, config, and functions messages",
    )
    show_parser.add_argument(
        "--format",
        choices=["json", "jsonl"],
        default="json",
        help="Format of the output (JSON or JSONL)",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {metadata.version('grasp-rdf')}",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="GRASP log level",
    )
    parser.add_argument(
        "--all-loggers",
        action="store_true",
        help="Enable logging for all loggers, not only the GRASP-specific ones",
    )
    return parser.parse_args()


def run_grasp(args: argparse.Namespace) -> None:
    logger = get_logger("GRASP", args.log_level)
    config = GraspConfig(**load_config(args.config))

    managers, models = setup(config)

    examples_model = models.get(f"sentence-transformer/{config.embedding_model}")
    if examples_model is not None:
        assert isinstance(examples_model, SentenceTransformerModel), (
            f"Expected examples embedding model to be a SentenceTransformerModel, "
            f"got {type(examples_model)}"
        )

    example_indices = load_example_indices(
        args.task,
        config,
        examples_model or config.embedding_model,
    )

    notes, kg_notes = load_notes(config)

    if args.input_field is None:
        input_field = get_task(args.task, managers, config).default_input_field
    else:
        input_field = args.input_field

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
            assert config.seed is not None, (
                "Seed must be set for deterministic shuffling"
            )
            random.seed(config.seed)
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

            dump_json(config.model_dump(), config_file, indent=2)

        if args.progress:
            # wrap with tqdm
            inputs = tqdm(inputs, desc=f"GRASP for {args.task}")

    else:
        if args.input is None:
            ipt = sys.stdin.read()
        else:
            ipt = args.input

        if args.input_format == "json":
            inputs = [json.loads(ipt)]
        else:
            inputs = [{"input": ipt}]
            input_field = "input"  # overwrite

    for i, ipt in enumerate(inputs):
        id = extract_field(ipt, "id") or "unknown"

        if input_field is not None:
            ipt = extract_field(ipt, input_field)

        assert ipt is not None, f"Input not found for input {i:,}"

        if i < len(outputs):
            # overwrite id
            output = outputs[i]
            output["id"] = id
            if not args.retry_failed or not is_invalid_output(
                output,
                args.none_output_invalid,
            ):
                continue

        output = consume_generator(
            generate(
                args.task,
                ipt,
                config,
                managers,
                kg_notes,
                notes,
                example_indices=example_indices,
                logger=logger,
            )
        )

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


def serve_grasp(args: argparse.Namespace) -> None:
    config = ServerConfig(**load_config(args.config))

    serve(config, args.log_level)


def get_grasp_data(args: argparse.Namespace) -> None:
    replace = parse_key_value_pairs(args.replace or [])
    params = parse_key_value_pairs(args.query_parameters or [])
    headers = parse_key_value_pairs(args.query_headers or [])

    if args.index_sparql is not None:
        args.index_sparql = load_text(args.index_sparql).strip()
        for key, value in replace.items():
            args.index_sparql = args.index_sparql.replace(f"{{{key}}}", value)

    get_data(
        args.knowledge_graph,
        args.index,
        args.endpoint,
        args.index_sparql,
        args.data_file,
        args.add_id_as_label,
        params,
        headers,
        args.log_level,
        args.overwrite,
    )


def take_grasp_notes(args: argparse.Namespace) -> None:
    note_cmd = args.note_command

    config = load_config(args.config)

    if note_cmd == "samples":
        take_notes_from_samples(
            args.task,
            NotesFromSamplesConfig(**config),
            args.output_dir,
            args.overwrite,
            args.log_level,
        )
    elif note_cmd == "outputs":
        take_notes_from_outputs(
            args.task,
            NotesFromOutputsConfig(**config),
            args.output_dir,
            args.overwrite,
            args.log_level,
        )
    elif note_cmd == "explore":
        take_notes_from_exploration(
            NotesFromExplorationConfig(**config),
            args.output_dir,
            args.overwrite,
            args.log_level,
        )
    elif note_cmd == "generate-questions":
        generate_questions(
            NotesGenerateQuestionsConfig(**config),
            args.output_dir,
            args.overwrite,
            args.log_level,
        )


def evaluate_grasp(args: argparse.Namespace) -> None:
    eval_cmd = args.evaluate_command

    if eval_cmd == "f1":
        evaluate_f1(
            args.knowledge_graph,
            args.input_file,
            args.prediction_file,
            args.endpoint,
            args.overwrite,
            args.timeout,
            args.retry_failed,
            args.exact_after,
            args.fix_prefixes,
            args.sparql_result_max_rows,
            args.log_level,
        )

    elif eval_cmd == "judge":
        judge_config = JudgeConfig(**load_config(args.config))

        evaluate_with_judge(
            args.input_file,
            args.prediction_files,
            args.evaluation_file,
            judge_config,
            args.overwrite,
            args.retry_failed,
            args.reformat_sparql,
            args.timeout,
            args.sparql_result_max_rows,
            args.with_ground_truth_reference,
            args.log_level,
        )

    elif eval_cmd == "expert":
        evaluate_with_expert(
            args.input_file,
            args.prediction_files,
            args.evaluation_file,
            args.kg_config,
            args.port,
            args.base_url_path,
            args.log_level,
        )


def show_grasp(args: argparse.Namespace) -> None:
    if args.file is not None:
        if args.format == "jsonl":
            inputs = load_jsonl(args.file)
        else:
            inputs = [load_json(args.file)]
    elif args.format == "jsonl":
        inputs = (json.loads(line) for line in sys.stdin if line.strip())
    else:
        inputs = [json.loads(sys.stdin.read())]

    separator = colored("=" * 80, "cyan")
    first = True
    for trace in inputs:
        if not first:
            print(f"\n{separator}\n")
        first = False
        print(format_trace(trace, skip_system=args.skip_system))


def setup_grasp(args: argparse.Namespace) -> None:
    phase = args.phase

    if phase == "shapes":
        _shapes_setup_grasp(args)
        return

    logger = get_logger("GRASP SETUP", args.log_level)
    config = GraspConfig(**load_config(args.config))

    managers, _ = setup(config)
    if not managers:
        logger.error("No KG managers available for setup")
        return
    elif len(managers) == 1:
        manager = managers[0]
    else:
        manager, _ = find_manager(managers, args.knowledge_graph)

    notes, kg_notes = load_notes(config)
    kg_dir = get_index_dir(manager.kg)

    multi_phase = phase in ("standard", "indices")

    if multi_phase and args.notes is not None:
        logger.error(
            f"--notes is not supported for phase '{phase}'; "
            "it is only supported for single-phase runs "
            "(info, entities, properties, literals, shapes)"
        )
        return

    phase_note = args.notes if not multi_phase else None

    selected = {
        "standard": {"info", "entities", "properties", "literals"},
        "indices": {"entities", "properties", "literals"},
        "info": {"info"},
        "entities": {"entities"},
        "properties": {"properties"},
        "literals": {"literals"},
    }[phase]

    all_phases = [
        ("info", {"phase": "info", "notes": phase_note}),
        ("entities", {"phase": "index", "name": "entities", "notes": phase_note}),
        ("properties", {"phase": "index", "name": "properties", "notes": phase_note}),
        ("literals", {"phase": "index", "name": "literals", "notes": phase_note}),
    ]
    phases = [payload for tag, payload in all_phases if tag in selected]

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def dump_latest(path: str, payload, text: bool = False) -> str:
        stem, ext = os.path.splitext(path)
        stamped = f"{stem}.{timestamp}{ext}"
        if text:
            dump_text(payload, stamped)
        else:
            dump_json(payload, stamped, indent=2)
        link(stamped, path)
        return stamped

    for phase_input in phases:
        p = phase_input["phase"]
        if p == "index":
            trace_dir = os.path.join(kg_dir, phase_input["name"])
            p += f" ({phase_input['name']})"
        else:
            trace_dir = kg_dir

        logger.info(f"Starting setup {p} phase for knowledge graph {manager.kg}")

        result = consume_generator(
            generate(
                "auto-setup",
                phase_input,
                config,
                [manager],
                kg_notes,
                notes,
                logger=logger,
            )
        )

        trace_path = os.path.join(trace_dir, "auto-setup.trace.json")
        stamped = dump_latest(trace_path, result)
        logger.info(f"Saved setup {p} trace to {stamped} (latest: {trace_path})")

        output = result.get("output")
        if output is None:
            logger.error(f"Setup {p} phase did not produce output")
            continue

        if phase_input["phase"] == "info":
            path = os.path.join(kg_dir, "info.json")
            stamped = dump_latest(path, output["info"])
            logger.info(f"Saved prefixes and description to {stamped} (latest: {path})")
            continue

        name = phase_input["name"]
        for typ in ["index", "info"]:
            sparql = output["sparql"].get(typ)
            path = os.path.join(kg_dir, name, f"{typ}.sparql")

            if sparql is None:
                if os.path.lexists(path):
                    os.unlink(path)
                    logger.info(
                        f"Unlinked {name} {typ} SPARQL (query set to None; previous timestamped queries are retained)"
                    )
                continue

            stamped = dump_latest(path, sparql, text=True)
            logger.info(f"Saved {name} {typ} SPARQL to {stamped} (latest: {path})")

        path = os.path.join(kg_dir, name, "info.json")
        stamped = dump_latest(path, output["info"])
        logger.info(f"Saved {name} description to {stamped} (latest: {path})")


def _shapes_setup_grasp(args: argparse.Namespace) -> None:
    logger = get_logger("GRASP SHAPES SETUP", args.log_level)
    config = GraspConfig(**load_config(args.config))

    managers, _ = setup(config)
    if not managers:
        logger.error("No KG managers available")
        return
    elif len(managers) == 1:
        manager = managers[0]
    else:
        manager, _ = find_manager(managers, args.knowledge_graph)

    notes, kg_notes = load_notes(config)
    kg_dir = get_index_dir(manager.kg)

    logger.info(f"Starting shape index setup for knowledge graph '{manager.kg}'")

    result = consume_generator(
        generate(
            "shapes-setup",
            {"kg": manager.kg, "notes": args.notes},
            config,
            [manager],
            kg_notes,
            notes,
            logger=logger,
        )
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def dump_latest(path: str, payload, text: bool = False) -> str:
        stem, ext = os.path.splitext(path)
        stamped = f"{stem}.{timestamp}{ext}"
        if text:
            dump_text(payload, stamped)
        else:
            dump_json(payload, stamped, indent=2)
        link(stamped, path)
        return stamped

    shapes_dir = os.path.join(kg_dir, "shapes")
    os.makedirs(shapes_dir, exist_ok=True)

    trace_path = os.path.join(shapes_dir, "shapes-setup.trace.json")
    stamped = dump_latest(trace_path, result)
    logger.info(f"Saved setup trace to {stamped} (latest: {trace_path})")

    output = result.get("output")
    if output is None or output.get("pattern") is None:
        logger.error("Setup did not produce a pattern")
        return

    setup_path = os.path.join(shapes_dir, "setup.json")
    stamped = dump_latest(
        setup_path,
        {"pattern": output["pattern"], "description": output.get("description")},
    )
    logger.info(f"Saved pattern to {stamped} (latest: {setup_path})")


def shapes_build_grasp(args: argparse.Namespace) -> None:
    assert args.config is not None, (
        "A config file is required: grasp index <kg> shapes <config>"
    )

    logger = get_logger("GRASP SHAPES BUILD", args.log_level)
    config = GraspConfig(**load_config(args.config))

    managers, _ = setup(config)
    if not managers:
        logger.error("No KG managers available")
        return

    manager, _ = find_manager(managers, args.knowledge_graph)

    kg_dir = get_index_dir(manager.kg)
    setup_path = os.path.join(kg_dir, "shapes", "setup.json")
    if not os.path.exists(setup_path):
        logger.error(
            f"No setup found at {setup_path}. "
            f"Run 'grasp setup {manager.kg} shapes <config>' first."
        )
        return

    setup_data = load_json(setup_path)
    pattern = setup_data.get("pattern")
    if not pattern:
        logger.error("setup.json does not contain a pattern")
        return

    shapes_dir = os.path.join(kg_dir, "shapes")
    index_dir = os.path.join(shapes_dir, "index")
    samples_file = os.path.join(index_dir, "samples.jsonl")

    total_classes: int | None = None
    if os.path.exists(samples_file) and not args.overwrite:
        logger.info(f"Shapes already exist at {samples_file}, skipping profiling")
        samples = [ShapeSample.model_validate(s) for s in load_jsonl(samples_file)]
        info_path = os.path.join(index_dir, "info.json")
        if os.path.exists(info_path):
            total_classes = load_json(info_path).get("total_classes")
    else:
        samples, total_classes = build_shapes(
            pattern,
            shapes_dir,
            manager,
            shape_config=manager.shape_config or ShapeConfig(),
            max_classes=args.max_classes,
            log_level=args.log_level,
            request_timeout=config.sparql_request_timeout,
            read_timeout=config.sparql_read_timeout,
        )

    model = SentenceTransformerModel(args.emb_model)
    ShapeIndex.build(
        samples,
        index_dir,
        model,
        args.emb_batch_size,
        args.overwrite,
        args.log_level,
        total_classes=total_classes,
    )


def main():
    args = parse_args()
    if args.all_loggers:
        setup_logging(args.log_level)

    if args.command == "data":
        get_grasp_data(args)

    elif args.command == "merge":
        merge_kgs(
            args.knowledge_graphs,
            args.index,
            args.knowledge_graph,
            args.overwrite,
            args.log_level,
        )

    elif args.command == "index":
        if args.index == "shapes":
            shapes_build_grasp(args)
        else:
            build_index(
                args.knowledge_graph,
                args.index,
                args.index_type,
                args.emb_model,
                args.emb_device,
                args.emb_batch_size,
                args.emb_dim,
                args.log_level,
                args.overwrite,
            )

    elif args.command == "notes":
        take_grasp_notes(args)

    elif args.command == "run" or args.command == "file":
        run_grasp(args)

    elif args.command == "serve":
        serve_grasp(args)

    elif args.command == "evaluate":
        evaluate_grasp(args)

    elif args.command == "setup":
        setup_grasp(args)

    elif args.command == "show":
        show_grasp(args)

    elif args.command == "examples":
        model = SentenceTransformerModel(args.emb_model)
        index = task_to_index(args.task)
        index.build(
            args.examples_file,
            args.output_dir,
            model,
            args.batch_size,
            args.overwrite,
            args.log_level,
            args.description,
        )


if __name__ == "__main__":
    main()
