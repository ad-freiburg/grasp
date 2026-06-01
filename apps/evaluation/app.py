import re
import sys
from collections import defaultdict
from functools import reduce
from pathlib import Path

import natsort
import pandas as pd
import streamlit as st
from universal_ml_utils.io import load_jsonl
from universal_ml_utils.logging import get_logger

from grasp.apps.shared import (
    display_name_from_file,
    render_messages,
    render_output_panel,
    try_load_json,
    try_load_model_outputs,
    try_load_rank_json,
)
from grasp.utils import (
    is_invalid_evaluation,
    is_invalid_output,
    is_retryable_evaluation,
    is_server_error,
)

logger = get_logger("EVALUATION APP")

STYLE_BEST = "background-color: #005500; color: white; font-weight: bold"
STYLE_BEST_TIE = "background-color: #666600; color: white; font-weight: bold"
STYLE_SECOND = "background-color: #003366; color: white"
STYLE_INVALID = "background-color: #990000; color: white"
STYLE_MISSING = "background-color: #444444; color: #ffffff"

# Set page configuration
st.set_page_config(page_title="SPARQL QA Evaluation", page_icon="📊", layout="wide")


def widget_changed(tracker_key: str, current_value) -> bool:
    """Return True if `current_value` differs from the last value stored under `tracker_key`."""
    changed = st.session_state.get(tracker_key) != current_value
    st.session_state[tracker_key] = current_value
    return changed


def select_kg_and_benchmark(available_data: dict) -> tuple[str, str]:
    """Render the group + benchmark sidebar selectboxes with conventional defaults."""
    kg_options = list(available_data.keys())
    default_index = kg_options.index("wikidata") if "wikidata" in kg_options else 0
    selected_kg = st.sidebar.selectbox("Select Group", kg_options, index=default_index)

    benchmark_options = list(available_data[selected_kg].keys())
    default_benchmark = (
        "qald10"
        if selected_kg == "wikidata"
        else "wqsp"
        if selected_kg == "freebase"
        else benchmark_options[0]
    )
    default_index = (
        benchmark_options.index(default_benchmark)
        if default_benchmark in benchmark_options
        else 0
    )
    selected_benchmark = st.sidebar.selectbox(
        "Select Benchmark", benchmark_options, index=default_index
    )
    return selected_kg, selected_benchmark


@st.cache_data(ttl=30)
def load_available_data() -> dict:
    """Find all available benchmarks and models."""
    data_root = Path(sys.argv[1])
    benchmarks = {}

    # Find all directories that might contain our data structure
    for kg_dir in data_root.glob("*"):
        if not kg_dir.is_dir():
            continue

        kg = kg_dir.name

        for benchmark_dir in kg_dir.glob("*"):
            if not benchmark_dir.is_dir():
                continue

            benchmark = benchmark_dir.name

            test_file = benchmark_dir / "test.jsonl"
            outputs_dir = benchmark_dir / "outputs"

            if test_file.exists() and outputs_dir.exists() and outputs_dir.is_dir():
                # This directory follows our expected structure
                # Find model output files (.jsonl) and their evaluation files (.evaluation.json)
                model_files = []
                for model_file in outputs_dir.glob("*.jsonl"):
                    # Skip evaluation jsonl files
                    if ".evaluation." in model_file.name:
                        continue

                    # Check if there's an evaluation file
                    eval_file = model_file.parent / f"{model_file.stem}.evaluation.json"

                    model_files.append((model_file, eval_file))

                if model_files:
                    if kg not in benchmarks:
                        benchmarks[kg] = {}

                    models_info = {
                        display_name_from_file(model_file): {
                            "output_file": str(model_file),
                            "eval_file": str(eval_file),
                        }
                        for model_file, eval_file in model_files
                    }

                    benchmarks[kg][benchmark] = {
                        "test_file": str(test_file),
                        "models": models_info,
                    }

    return benchmarks


@st.cache_data(ttl=30)
def load_ranking_data() -> dict:
    """Find all available judge evaluation files, organized by filename."""
    data_root = Path(sys.argv[1])
    rankings = {}

    # Find all directories that might contain judge data
    for kg_dir in data_root.glob("*"):
        if not kg_dir.is_dir():
            continue

        kg = kg_dir.name

        for benchmark_dir in kg_dir.glob("*"):
            if not benchmark_dir.is_dir():
                continue

            benchmark = benchmark_dir.name
            rank_dir = benchmark_dir / "rank"

            if rank_dir.exists() and rank_dir.is_dir():
                # Find all judge evaluation JSON files
                for rank_file in rank_dir.glob("*.json"):
                    # Group by filename (without extension)
                    filename = rank_file.stem

                    if filename not in rankings:
                        rankings[filename] = []

                    rankings[filename].append(
                        {"kg": kg, "benchmark": benchmark, "filepath": str(rank_file)}
                    )

    return rankings


def calculate_average_steps_and_time(
    outputs_dict: dict,
) -> tuple[float | None, float | None]:
    """
    Calculate average steps and time from model outputs.

    Returns:
        tuple: (avg_steps, avg_time) or (None, None) if no data
    """
    if not outputs_dict:
        return None, None

    total_steps = 0
    total_time = 0
    count = 0

    for output in outputs_dict.values():
        if "messages" in output:
            total_steps += sum(
                1
                for msg in output["messages"]
                if msg.get("role") not in ["user", "system"]
            )

        if "elapsed" in output:
            total_time += output["elapsed"]

        count += 1

    if count == 0:
        return None, None

    return (total_steps / count, total_time / count)


def calculate_metrics(
    ground_truth: list,
    model_outputs: dict,
    model_evaluations: dict,
    empty_target_valid: bool = False,
) -> dict:
    total = len(ground_truth)

    num_outputs = len(model_outputs)
    num_invalid_outputs = sum(
        is_invalid_output(output) for output in model_outputs.values()
    )

    num_evaluations = len(model_evaluations)

    num_invalid_evaluations = 0
    num_retryable_evaluations = 0
    total_f1 = 0.0
    total_accuracy = 0.0

    for id, evaluation in model_evaluations.items():
        invalid_evaluation = is_invalid_evaluation(
            evaluation,
            empty_target_valid,
        )
        num_invalid_evaluations += invalid_evaluation
        num_retryable_evaluations += is_retryable_evaluation(evaluation)
        if invalid_evaluation:
            continue

        # get f1 score
        if "prediction" in evaluation:
            f1_score = evaluation["prediction"]["score"]
        else:
            f1_score = 0.0

        total_f1 += f1_score
        total_accuracy += float(f1_score == 1.0)

    num_valid_evaluations = num_evaluations - num_invalid_evaluations

    f1_score = total_f1 / max(num_valid_evaluations, 1)
    accuracy = total_accuracy / max(num_valid_evaluations, 1)

    # Calculate average steps and time over the (possibly filtered) outputs
    avg_steps, avg_time = calculate_average_steps_and_time(model_outputs)

    return {
        "num_total": total,
        "num_outputs": num_outputs,
        "num_invalid_outputs": num_invalid_outputs,
        "num_evaluations": num_evaluations,
        "num_invalid_evaluations": num_invalid_evaluations,
        "num_retryable_evaluations": num_retryable_evaluations,
        "accuracy": accuracy,
        "f1": f1_score,
        "time": avg_time if avg_time is not None else 0,
        "steps": avg_steps if avg_steps is not None else 0,
    }


def load_and_process_data(
    test_file: str,
    model_info: dict,
    restrict_to_common_valid: bool = False,
    empty_target_valid: bool = False,
) -> tuple[list, dict, dict, dict]:
    """Load and process data for the selected benchmark and models."""
    # Load test data (ground truth)
    ground_truth = load_jsonl(test_file)

    # Load model outputs and evaluation data
    model_outputs = {}
    model_eval_data = {}

    for model_name, model_files in model_info.items():
        output_file = model_files["output_file"]
        eval_file = model_files["eval_file"]

        model_outputs[model_name] = try_load_model_outputs(output_file)

        eval_data = try_load_json(eval_file, default={})
        # Restrict to ids for which we have model outputs
        model_eval_data[model_name] = {
            id: ev for id, ev in eval_data.items() if id in model_outputs[model_name]
        }

    # Find common ids that are valid (output and evaluation) across
    # all SELECTED models (only those in model_info)
    if restrict_to_common_valid:
        all_ids = []
        for model_name, evaluations in model_eval_data.items():
            # Only consider models that were explicitly selected
            if model_name not in model_info:
                continue

            outputs = model_outputs[model_name]
            valid_ids = set(
                id
                for id, evaluation in evaluations.items()
                if not is_invalid_evaluation(evaluation, empty_target_valid)
                and not is_invalid_output(outputs[id])
            )
            all_ids.append(valid_ids)

        common_ids = set()
        if all_ids:
            common_ids = reduce(lambda x, y: x.intersection(y), all_ids)

        # Filter outputs and evaluations to common IDs
        model_outputs = {
            model_name: {
                id: output for id, output in outputs.items() if id in common_ids
            }
            for model_name, outputs in model_outputs.items()
        }

        model_eval_data = {
            model_name: {id: eval for id, eval in evals.items() if id in common_ids}
            for model_name, evals in model_eval_data.items()
        }

        # Filter ground truth to common IDs
        ground_truth = [gt for gt in ground_truth if gt["id"] in common_ids]

    # Calculate metrics
    metrics = {}
    for model_name, outputs in model_outputs.items():
        metrics[model_name] = calculate_metrics(
            ground_truth,
            outputs,
            model_eval_data[model_name],
            empty_target_valid=empty_target_valid,
        )

    return ground_truth, model_outputs, model_eval_data, metrics


def setup_model_selection(
    available_models, selected_models_dict: dict | list | None = None
) -> dict | list:
    """
    Setup model selection UI with regex filtering and checkboxes in expanders.

    Parameters:
    - available_models: Dictionary or list of available models
    - selected_models_dict: Optional dictionary to populate with selected models

    Returns:
    - Dictionary with selected models (key: model name, value: True if selected)
    """
    # Regex filter — widget key is the source of truth
    model_regex = st.sidebar.text_input(
        "Filter models by regex pattern",
        key="model_regex",
        help="Enter a regex pattern to automatically select matching models and deselect non-matching ones. Example: 'llama|phi' selects all LLaMA and Phi models.",
    )

    # Detect regex changes across reruns
    regex_changed = widget_changed("previous_model_regex", model_regex)

    selected_models = {} if selected_models_dict is None else selected_models_dict

    # Group models by name (before the first dot)
    model_groups = defaultdict(list)
    model_list = list(
        available_models.keys()
        if isinstance(available_models, dict)
        else available_models
    )
    for model_display_name in model_list:
        model_name = (
            model_display_name.split(" (")[0]
            if " (" in model_display_name
            else model_display_name
        )
        model_groups[model_name].append(model_display_name)

    # Compile regex once (if any)
    compiled_regex = None
    if model_regex:
        try:
            compiled_regex = re.compile(model_regex)
        except re.error as e:
            st.sidebar.error(f"Invalid regex pattern: {e}")

    # On regex change, update widget state directly before widgets render.
    # Streamlit prioritises session_state[key] over `value=` on re-renders,
    # so writing the widget key is the only way to programmatically toggle it.
    if regex_changed:
        if not model_regex:
            for variant in model_list:
                st.session_state[f"model_{variant}"] = True
        elif compiled_regex is not None:
            for variant in model_list:
                st.session_state[f"model_{variant}"] = bool(
                    compiled_regex.search(variant)
                )

    # Warn if a non-empty regex matches nothing
    if compiled_regex is not None and not any(
        compiled_regex.search(v) for v in model_list
    ):
        st.sidebar.warning(f"No models match the pattern '{model_regex}'")

    # Render checkboxes. Widget keys are the source of truth.
    for model_name, variants in sorted(model_groups.items()):
        with st.sidebar.expander(f"**{model_name}**", expanded=False):
            for variant in sorted(variants):
                display_name = variant.replace(model_name + " (", "").replace(")", "")
                checkbox_label = "default" if display_name == variant else display_name

                key = f"model_{variant}"
                if key not in st.session_state:
                    st.session_state[key] = (
                        bool(compiled_regex.search(variant))
                        if compiled_regex is not None
                        else True
                    )
                selected = st.checkbox(checkbox_label, key=key)

                if isinstance(selected_models, dict):
                    selected_models[variant] = selected
                elif isinstance(selected_models, list) and selected:
                    selected_models.append(variant)

    return selected_models


# Additional view functions
def show_predictions_view(available_data: dict) -> None:
    """Show a view focused on examining model outputs in detail."""
    st.title("Outputs Analysis")

    # Sidebar for benchmark and model selection
    st.sidebar.title("Benchmark Settings")

    selected_kg, selected_benchmark = select_kg_and_benchmark(available_data)

    # Get available models for this benchmark
    benchmark_info = available_data[selected_kg][selected_benchmark]
    available_models = benchmark_info["models"]

    # Add empty ground truth handling option
    empty_target_valid = st.sidebar.checkbox(
        "Count empty ground truth as valid",
        value=False,
        help="When checked, ground truth with size 0 (empty result sets) will be counted as valid",
    )

    # Allow user to select a model
    st.sidebar.markdown("---")
    st.sidebar.subheader("Select a Model")
    model_options = list(available_models.keys())

    # Preferred model to select by default when first loading
    preferred_model = "gpt-41 (search_extended_with_feedback)"

    # Store the selected model in session state to persist between benchmark changes
    if "predictions_view_model" not in st.session_state:
        # Initialize with preferred model
        st.session_state.predictions_view_model = preferred_model

    # Add regex filter for model selection
    model_regex = st.sidebar.text_input(
        "Filter models by regex pattern", key="predictions_model_regex"
    )

    # Determine which model options to display
    display_options = model_options
    if model_regex:
        try:
            regex = re.compile(model_regex)
            filtered_model_options = [
                model for model in model_options if regex.search(model)
            ]
            if filtered_model_options:
                display_options = filtered_model_options
            else:
                st.sidebar.warning(f"No models match the pattern '{model_regex}'")
        except re.error as e:
            st.sidebar.error(f"Invalid regex pattern: {e}")

    # Find index for the model selection based on stored value or preferred model
    if st.session_state.predictions_view_model in display_options:
        # Use previously selected model if available in current options
        default_index = display_options.index(st.session_state.predictions_view_model)
    else:
        # Otherwise use preferred model if available, or first model if not
        default_index = next(
            (i for i, m in enumerate(display_options) if preferred_model == m), 0
        )

    # Show select box with appropriate default index
    selected_model = st.sidebar.selectbox("Model", display_options, index=default_index)

    # Store selected model in session state for next time
    st.session_state.predictions_view_model = selected_model

    # Filter type
    prediction_options = [
        "All Outputs",
        "Invalid Outputs",
        "Invalid Evaluations",
        "Retryable Evaluations",
    ]
    prediction_type = st.sidebar.radio("Output Type", prediction_options)

    # Load data
    test_file = benchmark_info["test_file"]
    model_info = {selected_model: available_models[selected_model]}

    ground_truth, model_outputs, model_eval_data, metrics = load_and_process_data(
        test_file,
        model_info,
        restrict_to_common_valid=False,
        empty_target_valid=empty_target_valid,
    )

    # Load configuration from external config file instead of model output
    config_file = available_models[selected_model]["output_file"].replace(
        ".jsonl", ".config.json"
    )
    config_data = try_load_json(config_file, default={})

    # Get outputs and evaluations for the selected model
    model_name = selected_model
    outputs = model_outputs[model_name]
    evaluations = model_eval_data[model_name] if model_name in model_eval_data else {}

    # Filter outputs based on the selected type
    filtered_outputs = {}
    if prediction_type == "Invalid Outputs":
        filtered_outputs = {
            id: output for id, output in outputs.items() if is_invalid_output(output)
        }
    elif prediction_type == "Invalid Evaluations":
        filtered_outputs = {
            id: output
            for id, output in outputs.items()
            if id in evaluations
            and is_invalid_evaluation(
                evaluations[id], empty_target_valid=empty_target_valid
            )
        }
    elif prediction_type == "Retryable Evaluations":
        filtered_outputs = {
            id: output
            for id, output in outputs.items()
            if id in evaluations and is_retryable_evaluation(evaluations[id])
        }
    else:  # All Outputs
        filtered_outputs = outputs

    # Show info about the number of outputs
    st.info(
        f"Found {len(filtered_outputs)} {prediction_type.lower()} for model {model_name}"
    )

    if not filtered_outputs:
        st.warning(f"No {prediction_type.lower()} found for this model.")
        return

    # Always sort by ID using natural sorting
    sorted_ids = natsort.natsorted(filtered_outputs.keys())

    # Create a selection dropdown with ID and question
    # Create a dictionary mapping IDs to questions from ground truth
    id_to_question = {
        ex.get("id"): ex.get("question", "No question") for ex in ground_truth
    }

    # Create selection options with ID and question
    selection_options = [
        f"{id} - {id_to_question.get(id, 'No question')}" for id in sorted_ids
    ]

    # Show the dropdown
    selected_option = st.selectbox("Select an example:", selection_options)

    # Extract the ID from the selected option
    selected_id = selected_option.split(" - ")[0] if selected_option else None

    if not selected_id:
        return

    # Display the selected example
    output = filtered_outputs[selected_id]

    # Find corresponding ground truth
    gt_example = next((ex for ex in ground_truth if ex.get("id") == selected_id), None)

    # Main container for the example
    with st.container():
        # Display question from ground truth and ID
        st.subheader("Question")
        question = id_to_question.get(selected_id, "No question found")
        st.write(question)
        st.write(f"ID: {selected_id}")

        # Display ground truth if available
        if gt_example:
            st.subheader("Ground Truth SPARQL")
            st.code(gt_example.get("sparql", ""), language="sparql")

        # Display model output
        st.subheader("Model Output")

        new_format = "output" in output
        if new_format:
            sparql_query = (output["output"] or {}).get(
                "sparql", "No SPARQL query generated or found"
            )
        else:
            sparql_query = output.get("sparql", "No SPARQL query generated or found")

        st.code(sparql_query, language="sparql")

        # Display evaluation if available
        if selected_id in evaluations:
            eval_data = evaluations[selected_id]
            st.subheader("Evaluation")

            # Create columns for evaluation metrics (use more columns for better spacing)
            eval_cols = st.columns([1, 1, 1.5])
            with eval_cols[0]:
                if "prediction" in eval_data and "score" in eval_data["prediction"]:
                    f1_score = eval_data["prediction"]["score"]
                    st.metric("F1", f"{f1_score:.2f}")
                else:
                    st.metric("F1", "N/A")

            with eval_cols[1]:
                if "prediction" in eval_data and "elapsed" in eval_data["prediction"]:
                    elapsed = eval_data["prediction"]["elapsed"]
                    st.metric("Time (s)", f"{elapsed:.3f}")
                else:
                    st.metric("Time (s)", "N/A")

            with eval_cols[2]:
                if is_retryable_evaluation(eval_data):
                    st.metric("Status", "🔁 Retryable")
                elif is_invalid_evaluation(eval_data, empty_target_valid):
                    # Check if invalid due to empty ground truth
                    if (
                        not empty_target_valid
                        and "target" in eval_data
                        and eval_data["target"].get("size", None) == 0
                        and eval_data["target"].get("err", None) is None
                    ):
                        st.metric("Status", "❌ Empty Ground Truth")
                    else:
                        st.metric("Status", "❌ Invalid")
                elif (
                    "prediction" in eval_data
                    and eval_data["prediction"].get("score", 0) == 1.0
                ):
                    st.metric("Status", "✅ Exact Match")
                else:
                    st.metric("Status", "⚠️ Partial Match")

            # Show any error message
            if "error" in eval_data:
                st.error(f"Error: {eval_data['error']}")

            # Show ground truth errors if available
            if (
                "target" in eval_data
                and eval_data["target"] is not None
                and "err" in eval_data["target"]
                and eval_data["target"]["err"] is not None
            ):
                st.error(f"Ground Truth Error: {eval_data['target']['err']}")

            # Show prediction errors if available
            if (
                "prediction" in eval_data
                and eval_data["prediction"] is not None
                and "err" in eval_data["prediction"]
                and eval_data["prediction"]["err"] is not None
            ):
                st.error(f"Prediction Error: {eval_data['prediction']['err']}")

        # Display model configuration if available
        if config_data:
            with st.expander("Model Configuration"):
                st.json(config_data)

        st.subheader("Generation Process")
        render_messages(output, new_format=new_format)


def validate_ranking_consistency(
    benchmark_entries: list, rank_data_by_path: dict, check_judge_model: bool = True
) -> None:
    """
    Validate consistency across judge files.

    Checks:
    1. Judge model consistency across all KGs and benchmarks
    2. Prediction file paths don't reference other directories
    3. Same set of models compared across all benchmarks

    Displays warnings in Streamlit UI for any inconsistencies found.
    """
    judge_configs = {}
    prediction_file_sets = {}

    for entry in benchmark_entries:
        kg = entry["kg"]
        benchmark = entry["benchmark"]
        rank_data = rank_data_by_path.get(entry["filepath"])
        if not rank_data:
            continue

        if "judge_config" in rank_data:
            judge_configs[f"{kg}/{benchmark}"] = rank_data["judge_config"]

        if "summary" in rank_data:
            prediction_files = set()
            for key in rank_data["summary"].keys():
                if key == "tie":
                    continue
                prediction_files.add(Path(key).stem)
            prediction_file_sets[f"{kg}/{benchmark}"] = prediction_files

    # Check 1: Judge model consistency
    if check_judge_model and judge_configs:
        judge_models = {}
        for benchmark_key, judge_config in judge_configs.items():
            # Only compare the model name
            model_name = judge_config.get("model", "Unknown")
            if model_name not in judge_models:
                judge_models[model_name] = []
            judge_models[model_name].append(benchmark_key)

        if len(judge_models) > 1:
            warning_msg = "⚠️ **Inconsistent judge models detected!** Different judge files use different judge models:\n"
            for model_name, benchmarks in judge_models.items():
                warning_msg += f"\n  {model_name} (used by {', '.join(benchmarks)})"
            st.warning(warning_msg)

    # Check 2: Same set of models compared
    if prediction_file_sets:
        # Get all unique sets of prediction files
        unique_sets = {}
        for benchmark_key, file_set in prediction_file_sets.items():
            set_key = frozenset(file_set)
            if set_key not in unique_sets:
                unique_sets[set_key] = []
            unique_sets[set_key].append(benchmark_key)

        if len(unique_sets) > 1:
            warning_msg = "⚠️ **Inconsistent prediction files detected!** Different benchmarks are comparing different sets of models:\n"
            for i, (file_set, benchmarks) in enumerate(unique_sets.items(), 1):
                warning_msg += (
                    f"\n  Set {i} (used by {', '.join(benchmarks)}): {sorted(file_set)}"
                )
            st.warning(warning_msg)


def judge_label_from_rank(filename: str, rank_data: dict) -> str:
    model = rank_data.get("judge_config", {}).get("model")
    if model:
        return model

    if "expert_config" in rank_data:
        return rank_data.get("expert_config", {}).get("evaluator") or "expert"

    match = re.search(r"-judge-(.+)$", filename)
    if match:
        return match.group(1)

    return "expert"


def prediction_file_signature(rank_data: dict) -> tuple[str, ...]:
    return tuple(Path(path).name for path in rank_data.get("prediction_files", []))


def show_ranking_view(ranking_data: dict) -> None:
    """Show a view for judge evaluations across multiple benchmarks."""
    st.title("Judge View - Cross-Benchmark Comparison")

    if not ranking_data:
        st.warning(
            "No judge evaluation files found. Please make sure judge files are in the 'rank' subdirectories."
        )
        return

    # Sidebar for judge comparison selection
    st.sidebar.title("Judge Settings")

    # Get all available judge filenames
    ranking_options = sorted(ranking_data.keys())

    if not ranking_options:
        st.warning("No judge files found.")
        return

    # Regex filter for judge comparisons
    ranking_regex = st.sidebar.text_input(
        "Filter judge comparisons by regex pattern", key="ranking_regex"
    )
    filtered_ranking_options = ranking_options
    if ranking_regex:
        try:
            regex = re.compile(ranking_regex)
            filtered = [r for r in ranking_options if regex.search(r)]
            if filtered:
                filtered_ranking_options = filtered
            else:
                st.sidebar.warning(
                    f"No judge comparisons match the pattern '{ranking_regex}'"
                )
        except re.error as e:
            st.sidebar.error(f"Invalid regex pattern: {e}")

    # Select judge comparison(s) to view. Multiple compatible judge files are
    # rendered as separate rows for the same benchmark.
    selected_rankings = st.sidebar.multiselect(
        "Select Judge Comparison(s)",
        filtered_ranking_options,
        default=filtered_ranking_options[:1],
    )

    if not selected_rankings:
        st.warning("Select at least one judge comparison.")
        return

    display_names = [display_name_from_file(ranking) for ranking in selected_rankings]
    st.subheader(f"Comparison: {', '.join(display_names)}")

    # Get all benchmarks for this judge comparison
    benchmark_entries = []
    for ranking in selected_rankings:
        for entry in ranking_data[ranking]:
            benchmark_entries.append({**entry, "ranking": ranking})

    if not benchmark_entries:
        st.warning("No benchmark data found for the selected judge comparison(s).")
        return

    # Load each rank file at most once per rerun (cache hits across reruns)
    rank_data_by_path = {
        entry["filepath"]: try_load_rank_json(entry["filepath"])
        for entry in benchmark_entries
    }

    # Validate consistency across all judge files
    validate_ranking_consistency(
        benchmark_entries,
        rank_data_by_path,
        check_judge_model=len(selected_rankings) == 1,
    )

    entries_by_benchmark = defaultdict(list)
    for entry in benchmark_entries:
        entries_by_benchmark[(entry["kg"], entry["benchmark"])].append(entry)

    incompatible = []
    for (kg, benchmark), entries in entries_by_benchmark.items():
        signatures = {
            prediction_file_signature(rank_data_by_path.get(entry["filepath"], {}))
            for entry in entries
        }
        if len(signatures) > 1:
            incompatible.append(f"{kg}/{benchmark}")

    if incompatible:
        st.error(
            "Selected judge files compare different prediction files and cannot be merged. "
            f"Affected benchmark(s): {', '.join(sorted(incompatible))}"
        )
        return

    # Organize benchmarks by group for selection
    entries_by_kg = defaultdict(list)
    for entry in benchmark_entries:
        entries_by_kg[entry["kg"]].append(entry)

    kg_options = sorted(entries_by_kg.keys())
    if not kg_options:
        st.warning("No groups found for the selected judge comparison.")
        return

    default_kg_index = kg_options.index("wikidata") if "wikidata" in kg_options else 0
    selected_kg = st.sidebar.selectbox(
        "Select Group",
        kg_options,
        index=default_kg_index,
    )

    benchmark_options = sorted(
        {entry["benchmark"] for entry in entries_by_kg[selected_kg]}
    )

    if not benchmark_options:
        st.warning(f"No benchmarks available for group {selected_kg}.")
        return

    default_benchmark_index = 0
    selected_benchmark = st.sidebar.selectbox(
        "Select Benchmark",
        benchmark_options,
        index=default_benchmark_index,
    )

    selected_entries = [
        entry
        for entry in entries_by_kg[selected_kg]
        if entry["benchmark"] == selected_benchmark
    ]

    selected_entry = selected_entries[0] if selected_entries else None

    selected_kg_entries = entries_by_kg[selected_kg]

    judge_labels = sorted(
        {
            judge_label_from_rank(
                entry["ranking"], rank_data_by_path.get(entry["filepath"], {})
            )
            for entry in selected_kg_entries
        }
    )
    if judge_labels:
        st.caption(f"**Judge Models:** {', '.join(judge_labels)}")

    # First pass: collect all unique models for the selected group to establish ordering
    sorted_models = None
    for entry in selected_kg_entries:
        rank_data = rank_data_by_path.get(entry["filepath"])
        if not rank_data or "prediction_files" not in rank_data:
            continue

        prediction_models = [
            display_name_from_file(f) for f in rank_data["prediction_files"]
        ]

        if sorted_models is None:
            sorted_models = prediction_models
        elif sorted(sorted_models) != sorted(prediction_models):
            sorted_models = None
            break

    if sorted_models is None:
        st.warning(
            "Could not establish a consistent set of models across benchmarks for this group and judge comparison."
        )
        return

    # Sort models and assign letters
    model_to_letter = {
        model: chr(65 + i) for i, model in enumerate(sorted_models)
    }  # A=65 in ASCII

    # Display model legend
    if sorted_models:
        st.markdown("**Model Legend:**")
        legend_items = "\n".join(
            [f"- **{model_to_letter[model]}**: {model}" for model in sorted_models]
        )
        st.markdown(legend_items)

    # Process all benchmarks to build comprehensive table (one row per benchmark)
    table_rows = []
    any_scores_anywhere = False
    ranking_to_variant = {
        ranking: f"J{i + 1}" for i, ranking in enumerate(selected_rankings)
    }
    if ranking_to_variant:
        st.markdown("**Judge Variants:**")
        st.markdown(
            "\n".join(
                f"- **{variant}**: `{ranking}`"
                for ranking, variant in ranking_to_variant.items()
            )
        )

    for entry in selected_kg_entries:
        kg = entry["kg"]
        benchmark = entry["benchmark"]
        rank_file = entry["filepath"]

        try:
            rank_data = rank_data_by_path.get(rank_file, {})
            if "summary" not in rank_data:
                continue

            summary = rank_data["summary"]

            # Load model outputs to calculate additional metrics
            rank_dir = Path(rank_file).parent
            model_outputs_cache = {
                key: try_load_model_outputs(
                    rank_dir.parent / "outputs" / Path(key).name
                )
                for key in summary.keys()
                if key != "tie"
            }

            total_evals = len(rank_data.get("evaluations", {}))
            valid_evals = sum(
                1
                for eval_data in rank_data.get("evaluations", {}).values()
                if eval_data.get("err") is None
            )
            retryable_evals = sum(
                1
                for eval_data in rank_data.get("evaluations", {}).values()
                if is_server_error(eval_data.get("err"))
            )

            row_data = {
                "Benchmark": benchmark,
                "Judge": judge_label_from_rank(
                    entry["ranking"], rank_data_by_path.get(rank_file, {})
                ),
                "Variant": ranking_to_variant.get(entry["ranking"], "?"),
                "Valid Evals": f"{valid_evals}/{total_evals} (retryable {retryable_evals})",
            }

            model_wins = {}
            model_steps = {}
            model_time = {}
            model_avg_score = {}
            model_score_n = {}
            has_any_score = False
            tie_count = 0

            for key, value in summary.items():
                if key == "tie":
                    tie_count = value["count"]
                    continue

                model_display_name = display_name_from_file(key)
                avg_steps, avg_time = calculate_average_steps_and_time(
                    model_outputs_cache.get(key, {})
                )

                model_wins[model_display_name] = value["count"]
                model_steps[model_display_name] = avg_steps
                model_time[model_display_name] = avg_time

                avg_score = value.get("avg_score") if isinstance(value, dict) else None
                model_avg_score[model_display_name] = avg_score
                model_score_n[model_display_name] = (
                    value.get("n_scores", 0) if isinstance(value, dict) else 0
                )
                if avg_score is not None:
                    has_any_score = True
                    any_scores_anywhere = True

            # Calculate total for percentages
            total_comparisons = sum(model_wins.values()) + tie_count

            # Add columns for each model in alphabetical letter order
            for model in sorted_models:
                letter = model_to_letter[model]
                wins = model_wins.get(model, 0)
                percentage = (
                    (wins / total_comparisons * 100) if total_comparisons > 0 else 0
                )
                row_data[f"{letter} Wins"] = f"{percentage:.1f}% ({wins})"
                # Store raw value for determining winner
                row_data[f"_{letter}_wins_raw"] = wins

                avg_score = model_avg_score.get(model)
                n_scores = model_score_n.get(model, 0)
                row_data[f"{letter} Avg Score"] = (
                    f"{avg_score:.2f}/5 (n={n_scores})"
                    if avg_score is not None
                    else "—"
                )
                row_data[f"_{letter}_avg_score_raw"] = avg_score

            # Add ties column
            tie_percentage = (
                (tie_count / total_comparisons * 100) if total_comparisons > 0 else 0
            )
            row_data["Ties"] = f"{tie_percentage:.1f}% ({tie_count})"
            row_data["_ties_raw"] = tie_count

            # Format aggregate steps and time columns
            steps_parts = []
            time_parts = []
            for model in sorted_models:
                steps = model_steps.get(model)
                time = model_time.get(model)
                steps_parts.append(f"{steps:.1f}" if steps is not None else "—")
                time_parts.append(f"{time:.2f}" if time is not None else "—")

            row_data["Avg Steps"] = " / ".join(steps_parts)
            row_data["Avg Time"] = " / ".join(time_parts)

            # Determine which letters or "Ties" have the max wins
            max_wins = max(
                [
                    row_data.get(f"_{model_to_letter[model]}_wins_raw", 0)
                    for model in sorted_models
                ]
                + [row_data["_ties_raw"]]
            )
            row_data["_max_wins"] = max_wins

            winners = []
            for model in sorted_models:
                letter = model_to_letter[model]
                if row_data.get(f"_{letter}_wins_raw", 0) == max_wins and max_wins > 0:
                    winners.append(letter)
            if row_data["_ties_raw"] == max_wins and max_wins > 0:
                winners.append("Ties")
            row_data["_winners"] = winners

            table_rows.append(row_data)

        except Exception as e:
            st.error(f"Error loading {kg}/{benchmark}: {str(e)}")
            continue

    if not table_rows:
        st.warning("No valid judge data found for the selected comparison(s).")
        return

    # Create DataFrame
    df = pd.DataFrame(table_rows).sort_values(
        ["Benchmark", "Judge", "Variant"]
    )

    # Define column order
    display_columns = ["Benchmark", "Judge", "Variant"]
    for model in sorted_models:
        letter = model_to_letter[model]
        display_columns.append(f"{letter} Wins")
    if any_scores_anywhere:
        for model in sorted_models:
            letter = model_to_letter[model]
            display_columns.append(f"{letter} Avg Score")
    display_columns.extend(["Ties", "Avg Steps", "Avg Time", "Valid Evals"])

    df_display = df[display_columns]

    # Create styling function to highlight winning columns
    def highlight_winner(row):
        styles = [""] * len(row)
        row_data = df.loc[row.name]
        winners = row_data.get("_winners", [])

        winner_style = STYLE_BEST_TIE if len(winners) > 1 else STYLE_BEST
        for winner in winners:
            winner_col = f"{winner} Wins" if winner != "Ties" else "Ties"
            if winner_col in display_columns:
                col_idx = display_columns.index(winner_col)
                styles[col_idx] = winner_style

        return styles

    styled_df = df_display.style.apply(highlight_winner, axis=1)
    st.dataframe(styled_df, width="stretch", hide_index=True)

    # Show summary statistics
    st.caption(
        f"Showing {len(selected_kg_entries)} judge file(s) for {len(table_rows)} benchmark(s) in group '{selected_kg}'"
    )

    # Detailed sample view for the selected group and benchmark
    if not selected_entry:
        return

    if len(selected_entries) > 1:
        detail_options = {
            judge_label_from_rank(
                entry["ranking"], rank_data_by_path.get(entry["filepath"], {})
            ): entry
            for entry in selected_entries
        }
        selected_judge_label = st.sidebar.selectbox(
            "Select Judge for Details",
            sorted(detail_options.keys()),
            index=0,
        )
        selected_entry = detail_options[selected_judge_label]

    selected_rank_data = rank_data_by_path.get(selected_entry["filepath"])
    if not selected_rank_data:
        st.warning("Failed to load judge data for detailed view.")
        return

    evaluations = selected_rank_data.get("evaluations", {})
    if not evaluations:
        st.info("No detailed evaluations available for the selected benchmark.")
        return

    benchmark_dir = Path(selected_entry["filepath"]).parent.parent
    test_file = benchmark_dir / "test.jsonl"

    try:
        ground_truth_examples = load_jsonl(test_file)
    except Exception as exc:
        logger.warning(f"Failed to load test file {test_file}: {exc}")
        ground_truth_examples = []

    id_to_question = {}
    id_to_ground_truth = {}
    for example in ground_truth_examples:
        if not isinstance(example, dict):
            continue
        example_id = example.get("id")
        if example_id is None:
            continue
        id_to_question[example_id] = example.get("question", "No question provided")
        id_to_ground_truth[example_id] = example

    sorted_ids = natsort.natsorted(evaluations.keys())
    if not sorted_ids:
        st.info("No sample identifiers found in the evaluations.")
        return

    st.markdown("---")
    st.subheader(f"Sample Explorer: {selected_kg} / {selected_benchmark}")

    selected_id = st.selectbox(
        "Select an example:",
        sorted_ids,
        format_func=lambda x: f"{x} - {id_to_question.get(x, 'No question provided')}",
        key=f"judge_sample_{selected_kg}_{selected_benchmark}",
    )

    if not selected_id:
        return

    summary = selected_rank_data.get("summary", {})
    outputs_dir = benchmark_dir / "outputs"
    model_outputs = {}
    summary_model_entries = []
    model_display_order = []
    seen_models = set()

    for path in selected_rank_data["prediction_files"]:
        model_display_name = display_name_from_file(path)
        output_path = outputs_dir / Path(path).name

        summary_model_entries.append(
            {
                "display_name": model_display_name,
                "output_path": output_path,
                "summary_key": path,
            }
        )

        if model_display_name in seen_models:
            continue

        model_outputs[model_display_name] = try_load_model_outputs(output_path)
        model_display_order.append(model_display_name)
        seen_models.add(model_display_name)

    evaluation_entry = evaluations.get(selected_id, {})

    question_text = id_to_question.get(selected_id)
    if question_text:
        st.markdown(f"**Question:** {question_text}")

    ground_truth_entry = id_to_ground_truth.get(selected_id)
    if ground_truth_entry:
        ground_truth_sparql = ground_truth_entry.get("sparql")
        if ground_truth_sparql:
            with st.expander("Ground Truth SPARQL"):
                st.code(ground_truth_sparql, language="sparql")

    st.markdown("**Judge Verdict**")
    if not evaluation_entry:
        st.info("Judge verdict not available for the selected example.")
    else:
        verdict_value = evaluation_entry.get("verdict")

        winning_model_name = None
        if isinstance(verdict_value, int) and 0 <= verdict_value < len(
            summary_model_entries
        ):
            winning_model_name = summary_model_entries[verdict_value]["display_name"]

        verdict_display = (
            f"- Verdict Index: `{verdict_value}`"
            if verdict_value is not None
            else "- Verdict Index: `None`"
        )
        if winning_model_name:
            verdict_display += f" → **{winning_model_name}**"
        st.markdown(verdict_display)

        explanation_text = evaluation_entry.get("explanation")
        err_text = evaluation_entry.get("err")

        if explanation_text:
            st.markdown(explanation_text)
        if err_text:
            st.error(f"Judge error: {err_text}")

        scores_entry = evaluation_entry.get("scores")
        if scores_entry:
            score_rows = []
            for k, v in scores_entry.items():
                try:
                    idx = int(k)
                except (TypeError, ValueError):
                    continue
                if not (0 <= idx < len(summary_model_entries)):
                    continue
                score_rows.append(
                    {
                        "Model": summary_model_entries[idx]["display_name"],
                        "Score": f"{v}/5",
                    }
                )
            if score_rows:
                st.markdown("**Per-candidate Scores**")
                st.table(score_rows)

    st.markdown("**Model Outputs**")
    if not model_display_order:
        st.info("No model outputs available for this benchmark.")
    else:
        columns_per_row = 3
        for i in range(0, len(model_display_order), columns_per_row):
            current_models = model_display_order[i : i + columns_per_row]
            cols = st.columns(len(current_models))
            for col, model_name in zip(cols, current_models):
                outputs = model_outputs.get(model_name, {})
                output_entry = outputs.get(selected_id)

                with col.container(border=True):
                    st.markdown(f"**{model_name}**")
                    render_output_panel(output_entry)


def show_comprehensive_view(available_data: dict) -> None:
    """Show a comprehensive view with a large table of metrics across KGs and benchmarks."""
    st.title("Comprehensive Model Comparison")

    # Add settings to sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Comprehensive View Settings")

    # Add metric selector - only allow accuracy and f1
    metric_options = {
        "F1 Score (%)": "avg_f1",
        "Accuracy (%)": "accuracy",
    }
    selected_metric = st.sidebar.selectbox(
        "Select metric to display:", options=list(metric_options.keys()), index=0
    )
    metric_key = metric_options[selected_metric]
    show_eval_counts = st.sidebar.checkbox(
        "Show evaluated counts",
        value=False,
        help="When checked, each table cell also shows how many outputs are currently used for evaluation.",
    )

    # Option to restrict evaluation to common examples
    restrict_to_common = st.sidebar.checkbox(
        "Only evaluate on examples where all models have valid outputs and evaluations",
        value=False,
        help="When checked, only examples where all selected models have valid outputs and evaluations will be included in the comparison",
    )

    # Add empty ground truth handling option
    empty_target_valid = st.sidebar.checkbox(
        "Count empty ground truth as valid",
        value=False,
        help="When checked, ground truth with size 0 (empty result sets) will be counted as valid",
    )

    # Create a list of all available models across all benchmarks
    all_available_models = set()
    for kg_data in available_data.values():
        for benchmark_info in kg_data.values():
            all_available_models.update(benchmark_info["models"].keys())

    # Add model selection
    st.sidebar.markdown("---")
    st.sidebar.subheader("Select Models to Include")

    # Use shared model selection function
    selected_models = setup_model_selection(available_models=all_available_models)

    # Create a dictionary to hold all metrics across all KGs and benchmarks
    all_metrics = {}

    # Keep track of groups and their benchmarks for hierarchical columns
    kg_benchmarks = defaultdict(list)

    # Process each group and benchmark
    with st.spinner(
        "Loading comprehensive metrics across all groups and benchmarks..."
    ):
        for kg_name, kg_data in available_data.items():
            for benchmark_name, benchmark_info in kg_data.items():
                # Don't filter here, we'll do it after gathering all data

                # Load test data
                test_file = benchmark_info["test_file"]

                # Get models for this benchmark
                model_info = benchmark_info["models"]

                # Skip if no models
                if not model_info:
                    continue

                # Track this benchmark under its group
                kg_benchmarks[kg_name].append(benchmark_name)

                # Filter models based on user selection
                filtered_model_info = {
                    model_name: model_details
                    for model_name, model_details in model_info.items()
                    if model_name in selected_models and selected_models[model_name]
                }

                # Skip if no selected models for this benchmark
                if not filtered_model_info:
                    continue

                # Load and process data with restriction if selected
                _, _, _, metrics = load_and_process_data(
                    test_file,
                    filtered_model_info,
                    restrict_to_common_valid=restrict_to_common,
                    empty_target_valid=empty_target_valid,
                )

                # Store metrics for each model
                for model_name, model_metrics in metrics.items():
                    if model_name not in all_metrics:
                        all_metrics[model_name] = {}

                    # Store with separate group and benchmark keys for hierarchical display
                    if kg_name not in all_metrics[model_name]:
                        all_metrics[model_name][kg_name] = {}

                    all_metrics[model_name][kg_name][benchmark_name] = {
                        "avg_f1": model_metrics["f1"],
                        "accuracy": model_metrics["accuracy"],
                        "outputs": model_metrics["num_outputs"],
                        "evaluated": model_metrics["num_evaluations"],
                        "invalid_preds": model_metrics["num_invalid_outputs"],
                    }

    # If there are no metrics, show a warning and return
    if not all_metrics:
        st.warning("No metrics available for comprehensive view.")
        return

    # Group benchmarks by group
    benchmark_by_kg = {}
    for kg, benchmarks in kg_benchmarks.items():
        benchmark_by_kg[kg] = sorted(benchmarks)

    # Setup benchmark selection in its own section in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Select Benchmarks to Include")

    # Optional group regex filter before benchmark selection.
    group_regex = st.sidebar.text_input(
        "Filter groups by regex pattern",
        key="group_regex",
        help="Enter a regex pattern to automatically select matching groups and deselect non-matching ones. Example: 'wikidata|freebase' selects Wikidata and Freebase groups.",
    )

    compiled_group_regex = None
    if group_regex:
        try:
            compiled_group_regex = re.compile(group_regex)
        except re.error as e:
            st.sidebar.error(f"Invalid group regex pattern: {e}")

    selected_group_names = set(benchmark_by_kg)
    if compiled_group_regex is not None:
        selected_group_names = {
            kg for kg in benchmark_by_kg if compiled_group_regex.search(kg)
        }
        if not selected_group_names:
            st.sidebar.warning("No groups match the pattern. Showing all groups.")
            selected_group_names = set(benchmark_by_kg)

    benchmark_by_kg = {
        kg: benchmarks
        for kg, benchmarks in benchmark_by_kg.items()
        if kg in selected_group_names
    }

    # Regex filter — widget key is the source of truth
    benchmark_regex = st.sidebar.text_input(
        "Filter benchmarks by regex pattern",
        key="benchmark_regex",
        help="Enter a regex pattern to automatically select matching benchmarks and deselect non-matching ones. Example: 'wwq|lcquad' selects WWQ and LC-QuAD benchmarks.",
    )

    # Detect regex changes across reruns
    regex_changed = widget_changed("previous_benchmark_regex", benchmark_regex)

    # Compile regex once (if any)
    compiled_regex = None
    if benchmark_regex:
        try:
            compiled_regex = re.compile(benchmark_regex)
        except re.error as e:
            st.sidebar.error(f"Invalid regex pattern: {e}")

    # On regex change, update widget state directly before widgets render.
    if regex_changed:
        for kg, benchmarks in benchmark_by_kg.items():
            for benchmark in benchmarks:
                widget_key = f"benchmark_{kg}_{benchmark}"
                if not benchmark_regex:
                    st.session_state[widget_key] = True
                elif compiled_regex is not None:
                    st.session_state[widget_key] = bool(
                        compiled_regex.search(benchmark)
                    )

    # Render checkboxes. Widget keys are the source of truth.
    selected_benchmarks = {}
    for kg in sorted(benchmark_by_kg.keys()):
        with st.sidebar.expander(f"**{kg}**", expanded=False):
            for benchmark in sorted(benchmark_by_kg[kg]):
                key = f"benchmark_{kg}_{benchmark}"
                if key not in st.session_state:
                    st.session_state[key] = (
                        bool(compiled_regex.search(benchmark))
                        if compiled_regex is not None
                        else True
                    )
                selected = st.checkbox(benchmark, key=key)
                selected_benchmarks[(kg, benchmark)] = selected

    # If no benchmarks selected, treat as all-selected for filtering
    if not any(selected_benchmarks.values()):
        st.sidebar.warning("No benchmarks selected. Showing all benchmarks.")
        selected_benchmarks = {key: True for key in selected_benchmarks}

    # Now filter benchmarks based on selected group and benchmark options
    filtered_kg_benchmarks = defaultdict(list)
    for kg, benchmarks in kg_benchmarks.items():
        if kg not in selected_group_names:
            continue
        for benchmark in benchmarks:
            # Check if this (kg, benchmark) tuple is selected
            if selected_benchmarks.get((kg, benchmark), True):
                filtered_kg_benchmarks[kg].append(benchmark)

    # Replace kg_benchmarks with filtered version
    kg_benchmarks = filtered_kg_benchmarks

    # Prepare data for a pandas MultiIndex DataFrame
    # Sort groups and benchmarks for consistent display
    sorted_kgs = sorted(kg_benchmarks.keys())

    selected_benchmark_keys = {
        (kg, benchmark)
        for kg in sorted_kgs
        for benchmark in kg_benchmarks[kg]
    }
    displayed_model_names = [
        model_name
        for model_name, per_kg in all_metrics.items()
        if any(
            per_kg.get(kg, {}).get(benchmark, {}).get("outputs", 0) > 0
            for kg, benchmark in selected_benchmark_keys
        )
    ]

    if not displayed_model_names:
        st.warning("No models have outputs for the selected benchmarks.")
        return

    # Create a list of tuples for the MultiIndex columns
    column_tuples = [("Model", "")]  # First column is just the model name

    # Add tuples for each group and benchmark combination
    for kg in sorted_kgs:
        for benchmark in sorted(kg_benchmarks[kg]):
            column_tuples.append((kg, benchmark))

    # Create MultiIndex
    columns = pd.MultiIndex.from_tuples(column_tuples)

    # Prepare data rows
    data_rows = []
    for model_name in sorted(displayed_model_names):
        row = [model_name]  # Start with model name

        # Add data for each group and benchmark
        for kg in sorted_kgs:
            for benchmark in sorted(kg_benchmarks[kg]):
                # Check if we have metrics for this combination
                if (
                    kg in all_metrics[model_name]
                    and benchmark in all_metrics[model_name][kg]
                ):
                    metrics_data = all_metrics[model_name][kg][benchmark]
                    # Percentage with 1 decimal; 0.0 if nothing was evaluated
                    if metrics_data["evaluated"] == 0:
                        value = "0.0"
                    else:
                        value = f"{metrics_data[metric_key] * 100:.1f}"
                    if show_eval_counts:
                        value += f" ({metrics_data['evaluated']})"
                    row.append(value)
                else:
                    row.append("—")  # Em dash for missing data

        data_rows.append(row)

    # Create the DataFrame
    df = pd.DataFrame(data_rows, columns=columns)

    # Display the table with the metric name in the title
    st.subheader(f"{selected_metric} Across All Groups and Benchmarks")

    # Find best and second-best models for each benchmark
    rankings = {}
    for kg in sorted_kgs:
        for benchmark in sorted(kg_benchmarks[kg]):
            model_values = []
            for model_name in displayed_model_names:
                per_kg = all_metrics[model_name]
                metrics_data = per_kg.get(kg, {}).get(benchmark)
                if metrics_data and metrics_data["evaluated"] > 0:
                    model_values.append((model_name, metrics_data[metric_key]))

            model_values.sort(key=lambda x: x[1], reverse=True)
            distinct_values = []
            for _, value in model_values:
                if value not in distinct_values:
                    distinct_values.append(value)

            rankings[(kg, benchmark)] = {}
            for model, value in model_values:
                rankings[(kg, benchmark)][model] = distinct_values.index(value)

    # Build a per-cell CSS style DataFrame mirroring df's shape
    style_df = pd.DataFrame("", index=df.index, columns=df.columns)
    for i in range(len(df.index)):
        model_name = df.iloc[i, 0]
        for j, col in enumerate(df.columns[1:], 1):
            if df.iloc[i, j] == "—":
                style_df.iloc[i, j] = STYLE_MISSING
                continue

            kg, benchmark = col
            cell_metrics = all_metrics.get(model_name, {}).get(kg, {}).get(benchmark)
            if cell_metrics and cell_metrics.get("invalid_preds", 0) > 0:
                style_df.iloc[i, j] = STYLE_INVALID
                continue

            rank = rankings.get((kg, benchmark), {}).get(model_name)
            if rank == 0:
                best_tie = sum(
                    1
                    for value in rankings.get((kg, benchmark), {}).values()
                    if value == 0
                ) > 1
                style_df.iloc[i, j] = STYLE_BEST_TIE if best_tie else STYLE_BEST
            elif rank == 1:
                style_df.iloc[i, j] = STYLE_SECOND

    styled_df = df.style.apply(lambda _: style_df, axis=None)
    st.dataframe(styled_df, width="stretch")

    # Add a note about the data and color coding
    st.caption(
        "Note: Dark gray cells indicate no data available for that combination. Best model per benchmark is highlighted in dark green, second best in dark blue. Red cells indicate models with invalid outputs."
    )

    # Show summary statistics
    st.subheader("Summary Statistics")

    # Count total models and benchmarks
    total_models = len(displayed_model_names)
    total_kgs = len(sorted_kgs)
    total_benchmarks = sum(len(benchmarks) for benchmarks in kg_benchmarks.values())

    # Create 3 columns for the statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Models", total_models)
    with col2:
        st.metric("Groups", total_kgs)
    with col3:
        st.metric("Benchmarks", total_benchmarks)


# Main app
def main() -> None:
    st.title("SPARQL Question-Answering Evaluation")

    # Load available benchmarks and models
    available_data = load_available_data()

    if not available_data:
        st.error(
            "No benchmarks found. Please make sure the data structure follows the expected pattern."
        )
        return

    # Create a view selector
    view_options = [
        "Benchmark View",
        "Comprehensive View",
        "Outputs View",
        "Judge View",
    ]
    # Benchmark View is the default (index=0)
    selected_view = st.sidebar.radio("Select View", view_options, index=0)

    # Show the appropriate view based on selection
    if selected_view == "Comprehensive View":
        # The select_only variable is defined inside the function, so we don't need to pass it
        show_comprehensive_view(available_data)
    elif selected_view == "Outputs View":
        show_predictions_view(available_data)
    elif selected_view == "Judge View":
        # Load judge data and show judge view
        ranking_data = load_ranking_data()
        show_ranking_view(ranking_data)
    else:  # Benchmark View
        # Sidebar for benchmark and model selection
        st.sidebar.title("Benchmark Settings")

        selected_kg, selected_benchmark = select_kg_and_benchmark(available_data)

        # Option to restrict evaluation to common examples - moved to top, without heading
        restrict_to_common = st.sidebar.checkbox(
            "Only evaluate on examples where all models have valid outputs and evaluations",
            value=False,
            help="When checked, only examples where all selected models have valid outputs and evaluations will be included in the comparison",
        )

        # Add empty ground truth handling option
        empty_target_valid = st.sidebar.checkbox(
            "Count empty ground truth as valid",
            value=False,
            help="When checked, ground truth with size 0 (empty result sets) will be counted as valid",
        )

        # Get available models for this benchmark
        benchmark_info = available_data[selected_kg][selected_benchmark]
        available_models = benchmark_info["models"]

        # Allow selecting multiple models for comparison using checkboxes
        st.sidebar.markdown("---")
        st.sidebar.subheader("Select Models to Compare")

        # Initialize list for currently selected models
        selected_models = []

        # Use shared model selection function
        setup_model_selection(
            available_models=available_models, selected_models_dict=selected_models
        )

        # Filter to only selected models
        model_files = {
            model: available_models[model]
            for model in selected_models
            if model in available_models
        }

        # Main content
        if not model_files:
            st.warning("Please select at least one model for comparison.")
            return

        # Load and process data
        ground_truth, model_outputs, model_eval_data, metrics = load_and_process_data(
            benchmark_info["test_file"],
            model_files,
            restrict_to_common_valid=restrict_to_common,
            empty_target_valid=empty_target_valid,
        )

        # Display metrics with benchmark size
        example_count = len(ground_truth)
        st.subheader(
            f"Performance Metrics for {selected_kg} - {selected_benchmark} ({example_count} examples)"
        )

        # No information banner needed

        # Removed metrics visualization

        # Display metrics table
        # Format the combined predictions column
        combined_predictions = []
        for m in metrics:
            num_outputs = metrics[m]["num_outputs"]
            num_evaluations = metrics[m]["num_evaluations"]
            # Calculate values that might not be directly available in the metrics
            num_without_evaluation = num_outputs - num_evaluations
            num_invalid_outputs = metrics[m]["num_invalid_outputs"]
            num_invalid_evaluations = metrics[m]["num_invalid_evaluations"]
            num_retryable_evaluations = metrics[m]["num_retryable_evaluations"]

            # Format as:
            # total outputs (missing/invalid/retryable/invalid outputs)
            combined_predictions.append(
                f"{num_outputs} ({num_without_evaluation}/{num_invalid_evaluations}/{num_retryable_evaluations}/{num_invalid_outputs})*"
            )

        metrics_df = pd.DataFrame(
            {
                "Model": list(metrics.keys()),
                "Info*": combined_predictions,
                "Accuracy (%)": [
                    round(metrics[m]["accuracy"] * 100, 1) for m in metrics
                ],  # 1 decimal for percentages
                "Average F1 Score (%)": [
                    round(metrics[m]["f1"] * 100, 1) for m in metrics
                ],  # 1 decimal for percentages
                "Avg. Steps": [
                    round(metrics[m].get("steps", 0), 1) for m in metrics
                ],  # 1 decimal for average steps
                "Avg. Time (sec)": [
                    round(metrics[m].get("time", 0), 3) for m in metrics
                ],  # 3 decimals for time in seconds
            }
        )

        st.dataframe(metrics_df, width="stretch")

        # Add explanation for the info column
        empty_ground_truth_text = (
            "" if empty_target_valid else " or those with empty ground truth results"
        )
        st.caption(
            f"* Info format: Outputs (Missing Evaluations/Invalid Evaluations/Retryable Evaluations/Invalid Outputs) - 'Outputs' is the total number of model outputs, 'Missing Evaluations' counts outputs without an evaluation, 'Invalid Evaluations' counts evaluations with errors{empty_ground_truth_text}, 'Retryable Evaluations' counts evaluations with backend/runtime errors that --retry-failed will retry, 'Invalid Outputs' counts model outputs with errors. Note: Accuracy and F1 scores are calculated over all non-invalid evaluations."
        )

    st.sidebar.markdown("---")
    if st.sidebar.button(
        "Clear cache and reload",
        help="Clear cached benchmark, output, and evaluation data, then reload the app.",
    ):
        st.cache_data.clear()
        st.rerun()


if __name__ == "__main__":
    main()
