import json
import os
import random
import shutil
import string
import subprocess
import sys
from collections import Counter
from logging import Logger
from pathlib import Path

from tqdm import tqdm
from universal_ml_utils.io import dump_json, load_json, load_jsonl
from universal_ml_utils.logging import get_logger

from grasp.configs import JudgeConfig
from grasp.manager import load_kg_manager
from grasp.manager.utils import get_common_sparql_prefixes, load_kg_info, merge_prefixes
from grasp.model import Message, get_model
from grasp.model.base import Model
from grasp.sparql.metrics import f1_score
from grasp.sparql.types import AskResult, SelectResult
from grasp.sparql.utils import (
    execute,
    get_qlever_endpoint,
    load_iri_and_literal_parser,
    load_sparql_parser,
)
from grasp.sparql.utils import fix_prefixes as fix_sparql_prefixes
from grasp.tasks.sparql_qa import prepare_formatted_output
from grasp.tasks.sparql_qa.examples import SparqlQaSample
from grasp.utils import (
    format_message,
    is_invalid_output,
    is_retryable_evaluation,
    is_server_error,
)


def get_evaluation_file(prediction_file: str) -> str:
    base = os.path.splitext(prediction_file)[0]
    return f"{base}.evaluation.json"


def get_result_or_error(
    sparql: str,
    endpoint: str,
    timeout: float = 300.0,
    sparql_result_max_rows: int | None = None,
) -> tuple[SelectResult | AskResult | None, str | None]:
    try:
        result = execute(
            sparql,
            endpoint,
            request_timeout=timeout,
            read_timeout=timeout,
            sparql_result_max_rows=sparql_result_max_rows,
        )
        return result, None
    except Exception as e:
        return None, str(e)


def get_result_size(result: SelectResult | AskResult | None) -> int:
    if result is None:
        return 0
    elif isinstance(result, AskResult):
        return 1
    else:
        return len(result)


def load_predictions_and_evaluations(
    prediction_file: str,
    overwrite: bool = False,
) -> tuple[list, dict]:
    evaluation_file = get_evaluation_file(prediction_file)

    predictions = load_jsonl(prediction_file)

    evaluations = {}
    if os.path.exists(evaluation_file) and not overwrite:
        evaluations = load_json(evaluation_file)

    return predictions, evaluations


def load_inputs(input_file: str) -> dict[str, SparqlQaSample]:
    inputs: dict[str, SparqlQaSample] = {}
    for sample in load_jsonl(input_file):
        sample = SparqlQaSample(**sample)
        assert sample.id not in inputs, f"Duplicate id {sample.id}"
        assert sample.id is not None, "Sample id must not be None"
        inputs[sample.id] = sample
    return inputs


def evaluate_f1(
    kg: str,
    input_file: str,
    prediction_file: str,
    endpoint: str | None = None,
    overwrite: bool = False,
    timeout: float = 300.0,
    retry_failed: bool = False,
    exact_after: int = 1024,
    fix_prefixes: bool = False,
    sparql_result_max_rows: int | None = 10_000_000,
    log_level: str | int | None = None,
) -> None:
    logger = get_logger("GRASP EVALUATION", log_level)

    sparql_parser = load_sparql_parser()
    iri_literal_parser = load_iri_and_literal_parser()

    prefixes = get_common_sparql_prefixes()
    info = load_kg_info(kg)
    prefixes, _, _ = merge_prefixes(prefixes, info.prefixes or {}, logger)

    endpoint = endpoint or info.endpoint or get_qlever_endpoint(kg)

    def fix(sparql: str) -> str:
        if not fix_prefixes:
            return sparql

        try:
            return fix_sparql_prefixes(
                sparql,
                sparql_parser,
                iri_literal_parser,
                prefixes,
            )
        except Exception as e:
            logger.warning(f"Error fixing prefixes:\n{e}\n\nSPARQL:\n{sparql}")
            return sparql

    evaluation_file = get_evaluation_file(prediction_file)
    predictions, evaluations = load_predictions_and_evaluations(
        prediction_file,
        overwrite,
    )

    inputs = load_inputs(input_file)

    logger.info(
        f"Evaluating {len(predictions):,} predictions from {prediction_file} "
        f"for {len(inputs):,} inputs from {input_file} "
        f"against SPARQL endpoint {endpoint}"
    )

    num_invalid_outputs = 0
    num_invalid_evaluations = 0
    for pred in tqdm(
        predictions,
        desc="Evaluating",
        leave=False,
    ):
        assert pred.get("task", "sparql-qa") == "sparql-qa", (
            "Only SPARQL QA task is supported for evaluation"
        )
        if is_invalid_output(pred):
            num_invalid_outputs += 1
            continue

        id = pred["id"]
        if id in evaluations:
            evaluation = evaluations[id]
            if not retry_failed or not is_retryable_evaluation(evaluation):
                continue

        sparql = fix(inputs[id].sparql)
        target_result, target_err = get_result_or_error(
            sparql,
            endpoint,
            timeout,
            sparql_result_max_rows,
        )
        evaluations[id] = {
            "target": {
                "sparql": sparql,
                "err": target_err,
                "size": get_result_size(target_result),
            },
        }

        if target_result is None:
            num_invalid_evaluations += 1
            dump_json(evaluations, evaluation_file)
            continue

        output = pred["output"]
        sparql = None
        score = 0.0
        pred_err = "No prediction"
        pred_result = None

        if output is not None and output["sparql"] is not None:
            sparql = fix(output["sparql"])
            pred_result, pred_err = get_result_or_error(
                sparql,
                endpoint,
                timeout,
                sparql_result_max_rows,
            )

        if pred_result is not None:
            score = f1_score(pred_result, target_result, exact_after)

        evaluations[id]["prediction"] = {
            "sparql": sparql,
            "err": pred_err,
            "size": get_result_size(pred_result),
            "score": score,
            "elapsed": pred["elapsed"],
        }
        dump_json(evaluations, evaluation_file)

        if pred_result is None:
            num_invalid_evaluations += 1

    dump_json(evaluations, evaluation_file)
    logger.info(f"{len(evaluations):,} evaluation results saved to {evaluation_file}")
    f1_scores = [
        eval["prediction"]["score"]
        for eval in evaluations.values()
        if "prediction" in eval and eval["target"]["size"] > 0
    ]
    f1_avg = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    logger.info(f"Average F1 score (ignoring empty targets): {f1_avg:.2%}")
    logger.info(f"Invalid outputs: {num_invalid_outputs:,}")
    logger.info(f"Invalid evaluations: {num_invalid_evaluations:,}")


def judge_candidates(
    model: Model,
    question: str,
    candidates: list[str],
    logger: Logger,
    ground_truth: str | None = None,
) -> tuple[str, int | None]:
    if len(candidates) > len(string.ascii_uppercase):
        raise ValueError(
            f"Too many candidates ({len(candidates)}), max is {len(string.ascii_uppercase)}"
        )

    candidate_chars = string.ascii_uppercase[: len(candidates)]
    candidate_str = ", ".join(candidate_chars)

    functions = [
        {
            "name": "judge",
            "description": "Provide the final judgement for the SPARQL query candidates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "explanation": {
                        "type": "string",
                        "description": "An short explanation summarizing the reasoning behind the verdict.",
                    },
                    "verdict": {
                        "type": ["string", "null"],
                        "description": (
                            f"The verdict of the judgement. One of {candidate_str}, or null if all "
                            "candidates are equally good."
                        ),
                        "enum": list(candidate_chars) + [None],
                    },
                },
                "additionalProperties": False,
                "required": ["explanation", "verdict"],
            },
            "strict": True,
        }
    ]

    judge_input = (
        "Judge the following SPARQL query candidates for the given question.\n\n"
        f"Question:\n{question}\n\n"
    )

    if ground_truth is not None:
        judge_input += f"Reference SPARQL:\n{ground_truth}\n\n"

    judge_input += "\n\n".join(
        f"Candidate {char}:\n{cand}" for char, cand in zip(candidate_chars, candidates)
    )

    messages = [
        Message.system(
            """\
You are an expert judge for evaluating SPARQL queries.

You are given a question and two or more SPARQL query candidates \
that attempt to answer the question. Your task is to determine \
which of the candidate queries is the best answer to the question, \
or whether they are all equally good. The query logic and correctness \
should be your primary criteria for judgement. \
Note that some candidates might indicate that no SPARQL query \
has been generated/found, which should not automatically be considered worse \
than a generated SPARQL query that is incorrect or irrelevant to the question.

Importantly, you should not prefer one answer over the other purely because of \
secondary factors such as formatting, verbosity, additional information, \
or human readability (unless it is requested in the question). If the \
core content and logic are equally good, you should also judge them as being \
equally good. Similarly, questions can be ambiguous, and different candidates \
might have different interpretations of them that are equally valid. If no \
single interpretation is clearly better than the others, you should also judge \
them as being equally good.

If you are given a reference SPARQL query, do not treat it as the single correct \
ground truth answer that must be matched.

Think before you finalize your answer with the provided judge function.""",
        ),
        Message.user(judge_input),
    ]

    logger.debug(format_message(messages[-1]))

    try:
        response = model(messages, functions)
    except Exception as e:
        raise ValueError(f"Error during judging: {e}")

    if len(response.tool_calls) != 1:
        raise ValueError("Expected exactly one tool call for judgement")

    tool_call = response.tool_calls[0]
    explanation = tool_call.args["explanation"]
    verdict = tool_call.args["verdict"]
    logger.debug(f"Verdict: {verdict}\n{explanation}")
    if verdict is None:
        return explanation, None

    return explanation, candidate_chars.index(verdict)


def evaluate_with_judge(
    input_file: str,
    prediction_files: list[str],
    evaluation_file: str,
    judge_config: JudgeConfig,
    overwrite: bool = False,
    retry_failed: bool = False,
    reformat_sparql: bool = False,
    timeout: float = 300.0,
    sparql_result_max_rows: int | None = 10_000_000,
    with_ground_truth_reference: bool = False,
    log_level: str | int | None = None,
):
    logger = get_logger("GRASP EVALUATION", log_level)

    manager = None
    if reformat_sparql or with_ground_truth_reference:
        assert judge_config.knowledge_graph is not None, (
            "Reformatting or showing ground truth requires "
            "judge_config.knowledge_graph to be set"
        )
        manager = load_kg_manager(judge_config.knowledge_graph)
        logger.info(
            f'Loaded KG manager for "{judge_config.knowledge_graph.kg}" '
            f"to format SPARQL queries using the endpoint at {manager.endpoint}"
        )

    def get_candidates(outputs: list[dict]) -> list[str]:
        candidates = []

        for output in outputs:
            formatted = "No SPARQL generated or found"
            output = output.get("output")

            if output is None:
                candidates.append(formatted)
                continue

            if not reformat_sparql:
                formatted = output.get("formatted") or formatted
                if is_server_error(formatted):
                    raise ValueError(
                        "Candidate formatting contains a retryable server error:\n"
                        f"{formatted}"
                    )
                candidates.append(formatted)
                continue

            assert judge_config.knowledge_graph is not None
            assert manager is not None

            if output.get("sparql") is None:
                candidates.append("No SPARQL generated or found")
                continue

            output = prepare_formatted_output(
                output["sparql"],
                judge_config.knowledge_graph.kg,
                [manager],
                request_timeout=timeout,
                read_timeout=timeout,
                sparql_result_max_rows=sparql_result_max_rows,
            )
            if is_server_error(output["formatted"]):
                raise ValueError(
                    "Candidate formatting contains a retryable server error:\n"
                    f"{output['formatted']}"
                )
            candidates.append(output["formatted"])

        return candidates

    def get_ground_truth(sparql: str) -> str:
        assert judge_config.knowledge_graph is not None
        assert manager is not None
        output = prepare_formatted_output(
            sparql,
            judge_config.knowledge_graph.kg,
            [manager],
            request_timeout=timeout,
            read_timeout=timeout,
            sparql_result_max_rows=sparql_result_max_rows,
        )
        if is_server_error(output["formatted"]):
            raise ValueError(
                "Ground-truth formatting contains a retryable server error:\n"
                f"{output['formatted']}"
            )
        return output["formatted"]

    def group_predictions(predictions: list) -> dict:
        grouped: dict = {}
        for pred in predictions:
            assert pred.get("task", "sparql-qa") == "sparql-qa", (
                "Only SPARQL QA task is supported for judge evaluation"
            )

            id = pred["id"]
            assert id not in grouped, f"Duplicate prediction for id {id}"
            grouped[id] = pred

        return grouped

    # group predictions by id
    predictions = [
        group_predictions(load_jsonl(prediction_file))
        for prediction_file in prediction_files
    ]

    for preds, pred_file in zip(predictions, prediction_files):
        logger.info(f"Loaded {len(preds):,} valid predictions from {pred_file}")

    evaluations = {
        "prediction_files": prediction_files,
        "judge_config": judge_config.model_dump(),
        "evaluations": {},
    }
    if os.path.exists(evaluation_file) and not overwrite:
        evaluations = load_json(evaluation_file)

    def dump_evaluations():
        # create summary (histogram of verdicts)
        verdict_dist = Counter(
            evaluation["verdict"]
            for evaluation in evaluations["evaluations"].values()
            if evaluation["err"] is None
        )

        summary = {}

        for idx, count in verdict_dist.most_common():
            pred_file = prediction_files[idx] if idx is not None else "tie"

            rel_count = count / max(1, verdict_dist.total())

            summary[pred_file] = {
                "count": count,
                "ratio": rel_count,
            }

        evaluations["summary"] = summary
        dump_json(evaluations, evaluation_file)

    inputs = load_inputs(input_file)

    logger.info(
        f"Evaluating {len(prediction_files)} prediction files "
        f"on {len(inputs):,} inputs from {input_file} with "
        f"{len(evaluations['evaluations']):,} existing evaluations"
    )

    model = get_model(judge_config)
    random.seed(judge_config.seed)

    for id, sample in tqdm(
        inputs.items(),
        total=len(inputs),
        desc="Evaluating",
        leave=False,
    ):
        if id in evaluations["evaluations"]:
            evaluation = evaluations["evaluations"][id]
            if not retry_failed or not is_server_error(evaluation["err"]):
                continue

        candidates = []
        for preds, pred_file in zip(predictions, prediction_files):
            if id not in preds:
                logger.debug(f"Skipping missing prediction in {pred_file} for id {id}")
                break

            pred = preds[id]
            if is_invalid_output(pred):
                logger.debug(
                    f"Skipping invalid prediction in {pred_file} for id {id}:"
                    f"\n{json.dumps(pred, indent=2)}"
                )
                break

            candidates.append(pred)

        if len(candidates) != len(prediction_files):
            # not every prediction file has a prediction for this id
            continue

        # shuffle candidates to avoid position bias in judging
        indices = list(range(len(candidates)))
        random.shuffle(indices)
        candidates = [candidates[i] for i in indices]

        evaluation: dict = {
            "explanation": None,
            "verdict": None,
            "err": None,
        }

        try:
            candidates_formatted = get_candidates(candidates)
            ground_truth = (
                get_ground_truth(sample.sparql) if with_ground_truth_reference else None
            )

            explanation, verdict = judge_candidates(
                model,
                sample.question,
                candidates_formatted,
                logger,
                ground_truth,
            )
            evaluation["explanation"] = explanation
            evaluation["verdict"] = indices[verdict] if verdict is not None else None

        except Exception as e:
            logger.warning(f"Error during judgment of sample {id}: {e}")
            evaluation["err"] = str(e)

        evaluations["evaluations"][id] = evaluation
        dump_evaluations()

    dump_evaluations()
    for pred_file, summary in evaluations["summary"].items():
        ratio = summary["ratio"]
        count = summary["count"]
        logger.info(f"{pred_file}: {ratio:.2%} ({count})")


def evaluate_with_expert(
    input_file: str,
    prediction_files: list[str],
    evaluation_file: str,
    kg_config: str | None = None,
    port: int | None = None,
    base_url_path: str = "/expert",
    log_level: str | int | None = None,
) -> int:
    logger = get_logger("GRASP EVALUATION", log_level)

    expert_app = Path(__file__).resolve().parent / "apps" / "expert.py"
    if not expert_app.exists():
        logger.error(f"Expert app not found at {expert_app}")
        return 1

    streamlit = shutil.which("streamlit")
    if streamlit is None:
        logger.error(
            "The 'streamlit' executable was not found. Install it with "
            "`pip install grasp-rdf[expert]` (or `pip install streamlit`) "
            "to use the expert evaluation app."
        )
        return 1

    cmd: list[str] = [streamlit, "run", str(expert_app)]
    if port is not None:
        cmd += ["--server.port", str(port)]
    normalized_base_url_path = base_url_path.strip("/")
    cmd.append("--browser.gatherUsageStats=false")
    cmd.append(f"--server.baseUrlPath={normalized_base_url_path}")
    cmd.append("--server.enableCORS=false")
    cmd.append("--server.enableXsrfProtection=false")
    cmd.append("--")
    cmd.append(input_file)
    cmd.extend(prediction_files)
    cmd += ["--evaluation", evaluation_file]
    if kg_config:
        cmd += ["--kg-config", kg_config]

    logger.info(f"Launching expert app: {' '.join(cmd)}")
    try:
        return subprocess.run(cmd, check=False).returncode
    except KeyboardInterrupt:
        return 130
    finally:
        sys.stdout.flush()
