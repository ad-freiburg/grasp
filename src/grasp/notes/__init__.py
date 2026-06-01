import os
import random
import time
import uuid
from logging import Logger

import yaml
from search_rdf.model import SentenceTransformerModel
from tqdm import tqdm, trange
from universal_ml_utils.io import dump_json, dump_jsonl, load_jsonl
from universal_ml_utils.logging import get_logger
from universal_ml_utils.ops import consume_generator

from grasp.configs import (
    GraspConfig,
    NotesConfig,
    NotesFromExplorationConfig,
    NotesFromOutputsConfig,
    NotesFromSamplesConfig,
    NotesGenerateQuestionsConfig,
    NoteTakingConfig,
)
from grasp.core import generate, load_notes, setup
from grasp.examples import Sample, load_example_indices
from grasp.functions import find_manager
from grasp.manager import KgManager
from grasp.model import Message, get_model
from grasp.notes.utils import format_output
from grasp.tasks import get_task
from grasp.tasks.cea import AnnotationState, CeaSample, prepare_annotation
from grasp.tasks.exploration import (
    FunctionalExplorationState,
    StructuralExplorationState,
)
from grasp.tasks.exploration.functions import call_function, note_function_definitions
from grasp.tasks.question_generation import QuestionGenerationState
from grasp.tasks.sparql_qa.examples import SparqlQaSample
from grasp.tasks.utils import format_sparql_result, prepare_sparql_result
from grasp.utils import (
    format_kg_notes,
    format_list,
    format_message,
    format_notes,
    format_response,
    link,
)


def take_notes_from_samples(
    task: str,
    config: NotesFromSamplesConfig,
    out_dir: str,
    overwrite: bool = False,
    log_level: str | int | None = None,
) -> None:
    if os.path.exists(out_dir) and not overwrite:
        raise FileExistsError(f"Output directory {out_dir} already exists")

    logger = get_logger("GRASP NOTE TAKING", log_level)
    agent_logger = get_logger("GRASP AGENT", log_level)

    managers, models = setup(config)
    examples_model = models.get(f"sentence-transformer/{config.embedding_model}")
    if examples_model is not None:
        assert isinstance(examples_model, SentenceTransformerModel), (
            f"Expected examples embedding model to be a SentenceTransformerModel, "
            f"got {type(examples_model)}"
        )

    example_indices = load_example_indices(
        task,
        config,
        examples_model or config.embedding_model,
    )
    notes, kg_notes = load_notes(config)

    sample_cls = get_task(task, managers, config).sample_cls()
    assert sample_cls is not None, f"Task {task} does not support samples"

    assert config.seed is not None, "Seed must be set for adaptation"

    all_samples: list[tuple[str, Sample]] = []
    for sample_cfg in config.samples:
        samples = [
            (sample_cfg.kg, sample_cls(**sample))
            for sample in load_jsonl(sample_cfg.file)
        ]
        if config.samples_per_file is not None:
            random.seed(config.seed)
            random.shuffle(samples)
            samples = samples[: config.samples_per_file]

        all_samples.extend(samples)  # type: ignore

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.dump(config.model_dump(), f)

    # Pre-generate sample sequence in epochs so every sample is seen once
    # per epoch before any is revisited across rounds.
    n_per_round = min(config.samples_per_round, len(all_samples))
    sequence: list[tuple[str, Sample]] = []
    epoch = 0
    while len(sequence) < config.num_rounds * n_per_round:
        random.seed(config.seed + epoch)
        sequence.extend(random.sample(all_samples, len(all_samples)))
        epoch += 1

    for r in trange(config.num_rounds, desc="Taking notes from samples"):
        samples = sequence[r * n_per_round : (r + 1) * n_per_round]

        outputs = []
        for kg, sample in tqdm(samples, desc="Running GRASP on samples", leave=False):
            manager, _ = find_manager(managers, kg)

            output = consume_generator(
                generate(
                    task,
                    sample.input(),
                    config,
                    [manager],
                    kg_notes,
                    notes,
                    example_indices=example_indices,
                    logger=agent_logger,
                )
            )
            outputs.append(output)

        if config.ignore_ground_truth:
            ground_truths = None
        else:
            ground_truths = prepare_ground_truths(samples, managers, config)

        output = take_notes(
            outputs,
            managers,
            kg_notes,
            notes,
            config,
            logger,
            ground_truths,
        )

        output["task"] = task
        output["samples"] = [
            {"kg": kg, "sample": sample.model_dump()} for kg, sample in samples
        ]
        output["traces"] = outputs
        dump_json(output, os.path.join(out_dir, f"output.{task}.round_{r}.json"))

        for kg, kg_specific_notes in kg_notes.items():
            out_file = os.path.join(out_dir, f"notes.{task}.{kg}.round_{r}.json")
            dump_json(kg_specific_notes, out_file, indent=2)
            link(out_file, os.path.join(out_dir, f"notes.{task}.{kg}.json"))

        out_file = os.path.join(out_dir, f"notes.{task}.round_{r}.json")
        dump_json(notes, out_file, indent=2)
        link(out_file, os.path.join(out_dir, f"notes.{task}.json"))


def take_notes_from_outputs(
    task: str,
    config: NotesFromOutputsConfig,
    out_dir: str,
    overwrite: bool = False,
    log_level: str | int | None = None,
) -> None:
    if os.path.exists(out_dir) and not overwrite:
        raise FileExistsError(f"Output directory {out_dir} already exists")

    logger = get_logger("GRASP NOTE TAKING", log_level)

    managers, _ = setup(config)
    notes, kg_notes = load_notes(config)

    assert config.seed is not None, "Seed must be set for adaptation"

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.dump(config.model_dump(), f)

    all_outputs = []
    for output_file in config.outputs:
        outputs = load_jsonl(output_file)
        assert all(output["task"] == task for output in outputs), (
            f"All outputs in {output_file} must be for task {task}"
        )
        if config.outputs_per_file is not None:
            random.seed(config.seed)
            random.shuffle(outputs)
            outputs = outputs[: config.outputs_per_file]

        all_outputs.extend(outputs)

    for r in trange(config.num_rounds, desc="Taking notes from outputs"):
        random.seed(config.seed + r)

        outputs = random.sample(
            all_outputs,
            min(config.outputs_per_round, len(all_outputs)),
        )

        output = take_notes(
            outputs,
            managers,
            kg_notes,
            notes,
            config,
            logger,
        )

        output["task"] = task
        output["traces"] = outputs
        dump_json(output, os.path.join(out_dir, f"output.{task}.round_{r}.json"))

        for kg, kg_specific_notes in kg_notes.items():
            out_file = os.path.join(out_dir, f"notes.{task}.{kg}.round_{r}.json")
            dump_json(kg_specific_notes, out_file, indent=2)
            link(out_file, os.path.join(out_dir, f"notes.{task}.{kg}.json"))

        out_file = os.path.join(out_dir, f"notes.{task}.round_{r}.json")
        dump_json(notes, out_file, indent=2)
        link(out_file, os.path.join(out_dir, f"notes.{task}.json"))


def take_notes_from_exploration(
    config: NotesConfig,
    out_dir: str,
    overwrite: bool = False,
    log_level: str | int | None = None,
) -> None:
    if os.path.exists(out_dir) and not overwrite:
        raise FileExistsError(f"Output directory {out_dir} already exists")

    agent_logger = get_logger("GRASP AGENT", log_level)

    managers, models = setup(config)
    examples_model = models.get(f"sentence-transformer/{config.embedding_model}")
    if examples_model is not None:
        assert isinstance(examples_model, SentenceTransformerModel), (
            f"Expected examples embedding model to be a SentenceTransformerModel, "
            f"got {type(examples_model)}"
        )

    example_indices = load_example_indices(
        "sparql-qa",
        config,
        examples_model or config.embedding_model,
    )
    notes, kg_notes = load_notes(config)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.dump(config.model_dump(), f)

    assert isinstance(config, NotesFromExplorationConfig)
    if config.mode == "functional":
        task_name = "exploration_functional"
        state = FunctionalExplorationState(notes=notes)
        # rebind
        notes = state.notes
    elif config.mode == "structural":
        task_name = "exploration_structural"
        state = StructuralExplorationState(kg_notes=kg_notes)
        # rebind
        kg_notes = state.kg_notes
    else:
        raise ValueError(f"Unknown exploration mode: {config.mode}")

    for r in trange(config.num_rounds, desc="Taking notes from exploration"):
        output = consume_generator(
            generate(
                task_name,
                state,
                config,
                managers,
                kg_notes,
                notes,
                example_indices=example_indices,
                logger=agent_logger,
            )
        )

        dump_json(output, os.path.join(out_dir, f"output.exploration.round_{r}.json"))

        if config.mode == "structural":
            for kg, kg_specific_notes in kg_notes.items():
                out_file = os.path.join(
                    out_dir, f"notes.exploration.{kg}.round_{r}.json"
                )
                dump_json(kg_specific_notes, out_file, indent=2)
                link(out_file, os.path.join(out_dir, f"notes.exploration.{kg}.json"))
        else:
            out_file = os.path.join(out_dir, f"notes.exploration.round_{r}.json")
            dump_json(notes, out_file, indent=2)
            link(out_file, os.path.join(out_dir, "notes.exploration.json"))


def generate_questions(
    config: NotesGenerateQuestionsConfig,
    out_dir: str,
    overwrite: bool = False,
    log_level: str | int | None = None,
) -> None:
    if os.path.exists(out_dir) and not overwrite:
        raise FileExistsError(f"Output directory {out_dir} already exists")

    agent_logger = get_logger("GRASP AGENT", log_level)

    managers, _ = setup(config)
    notes, kg_notes = load_notes(config)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.dump(config.model_dump(), f)

    state = QuestionGenerationState()

    for r in trange(config.num_rounds, desc="Generating questions"):
        output = consume_generator(
            generate(
                "question-generation",
                state,
                config,
                managers,
                kg_notes,
                notes,
                logger=agent_logger,
            )
        )

        dump_json(
            output,
            os.path.join(out_dir, f"question-generation.trace.round_{r}.json"),
        )

        for kg, kg_questions in state.questions.items():
            samples = [
                SparqlQaSample(
                    id=uuid.uuid4().hex,
                    question=q,
                    sparql="",
                ).model_dump()
                for q in kg_questions
            ]
            out_file = os.path.join(out_dir, f"samples.{kg}.round_{r}.jsonl")
            dump_jsonl(samples, out_file)
            link(out_file, os.path.join(out_dir, f"samples.{kg}.jsonl"))


def rules() -> list[str]:
    return [
        "Do not take notes on things that are already handled well by the agent.",
        "As you hit the limits on the number of notes and their length, \
gradually merge and generalize your notes, discard unnecessary details, and move \
notes that can be useful across knowledge graphs to the general section.",
    ]


def system_instructions(max_notes: int, max_note_length: int) -> str:
    return f"""\
You are a note-taking assistant. Your task is to \
inspect the traces of a knowledge graph agent performing a certain task, and to \
take notes about the agent's outputs as well as the used knowledge \
graphs and functions.

Your notes should help the agent to better understand and navigate the task \
and knowledge graphs in the future. You are limited to a maximum of {max_notes} notes \
per knowledge graph, plus {max_notes} general notes for insights that apply \
across knowledge graphs or to the task in general. Each note is limited to a maximum of \
{max_note_length} characters to ensure it is concise and to the point.

The notes should generalize to new unseen task inputs, rather than being specific to \
the task traces and outputs at hand.

Before calling a note-taking function, provide reasoning for what you are doing and why. \
Before stopping, make sure to check all notes (not only the ones touched in this iteration) \
for the above mentioned criteria and clean them if needed.

Examples of potentially useful types of notes to include:
- overall structure, domain coverage, and schema of the knowledge graphs
- peculiarities of the knowledge graphs
- strategies when encountering certain types of questions or errors
- tips for when and how to use certain functions

Additional rules to follow:
{format_list(rules())}"""


def prepare_ground_truth(
    sample: Sample,
    kg: str,
    managers: list[KgManager],
    config: GraspConfig,
) -> str:
    if isinstance(sample, SparqlQaSample):
        result, selections = prepare_sparql_result(
            sample.sparql,
            kg,
            managers,
            config.result_max_rows,
            config.result_max_columns,
            request_timeout=(
                config.sparql_connection_timeout,
                config.sparql_query_timeout,
            ),
            read_timeout=config.sparql_read_timeout,
        )
        manager, _ = find_manager(managers, kg)
        return format_sparql_result(manager, result, selections)

    elif isinstance(sample, CeaSample):
        manager, _ = find_manager(managers, kg)

        annots = AnnotationState(sample.table)
        for annot in sample.annotations:
            full_annot = prepare_annotation(manager, annot.entity)
            annots.annotate(annot.row, annot.column, full_annot)

        return annots.format()

    else:
        raise ValueError(f"Unsupported or unknown sample type {type(sample)}")


def prepare_ground_truths(
    samples: list[tuple[str, Sample]],
    managers: list[KgManager],
    config: GraspConfig,
) -> list[str] | None:
    ground_truths = []
    for kg, sample in samples:
        gt = prepare_ground_truth(sample, kg, managers, config)
        ground_truths.append(gt)
    return ground_truths


def note_taking_instructions(
    outputs: list[dict],
    ground_truths: list[str] | None = None,
) -> str:
    formatted = []
    for i, output in enumerate(outputs):
        messages = [Message(**msg) for msg in output["messages"]]

        if i == 0:
            assert messages[0].role == "system"
            formatted.append(f"Task instructions for the agent:\n{messages[0].content}")

        assert messages[1].role == "user"
        input = messages[1].content

        content = f"Input {i + 1}:\n{input}"

        if ground_truths is not None:
            gt = ground_truths[i]
            content += f"\n\nReference Output:\n{gt}"

        content += f"\n\nAgent trace:\n{format_output(output['output'], messages)}"

        formatted.append(content)

    fmt = "\n\n".join(formatted)

    return f"""\
Look at the current notes (which might be the same notes provided \
to the agent). Then add to, delete from, or update them based on the \
given agent traces below.

{fmt}"""


def take_notes(
    outputs: list[dict],
    managers: list[KgManager],
    kg_notes: dict[str, list[str]],
    notes: list[str],
    config: NoteTakingConfig,
    logger: Logger,
    ground_truths: list[str] | None = None,
) -> dict:
    messages = [
        Message(
            role="system",
            content=system_instructions(config.max_notes, config.max_note_length),
        ),
        Message(
            role="user",
            content=note_taking_instructions(outputs, ground_truths),
        ),
    ]

    for msg in messages:
        logger.debug(format_message(msg))

    functions = note_function_definitions(managers)

    num_messages = len(messages)

    # if a note taking model is configured, use it as a full
    # replacement for the model config; otherwise fall back to
    # the parent grasp config
    nt_config = config.note_taking_model or config
    model = get_model(nt_config)
    start = time.monotonic()
    error = None

    while len(messages) - num_messages < config.max_steps:
        try:
            response = model(messages, functions)
        except Exception as e:
            msg = f"Error while calling LLM API during note taking: {e}"
            logger.error(msg)
            error = {"content": msg, "reason": "api_error"}
            break

        if response.is_empty:
            msg = "LLM API returned empty response during note taking"
            logger.error(msg)
            error = {"content": msg, "reason": "empty_response"}
            break

        messages.append(Message(role="assistant", content=response))

        for tool_call in response.tool_calls:
            try:
                result = call_function(
                    kg_notes,
                    notes,
                    tool_call.name,
                    tool_call.args,
                    config.max_notes,
                    config.max_note_length,
                )
            except Exception as e:
                result = f"Call to function {tool_call.name} returned an error:\n{e}"

            tool_call.result = result

        # only log now once tool call results are set
        logger.debug(format_response(response))

        if not response.tool_calls or any(
            tool_call.name == "stop" for tool_call in response.tool_calls
        ):
            break

    end = time.monotonic()
    # build and return output
    return {
        "type": "output",
        "output": {
            "formatted": f"""\
Note taking completed.

Notes for knowledge graphs:
{format_kg_notes(kg_notes)}

General notes:
{format_notes(notes)}""",
        },
        "messages": [msg.model_dump() for msg in messages],
        "elapsed": end - start,
        "error": error,
    }
