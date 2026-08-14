import os
import random
import uuid

import yaml
from pydantic import BaseModel
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
)
from grasp.core import generate, load_notes, setup
from grasp.examples import Sample, load_example_indices
from grasp.functions import find_manager
from grasp.manager import KgManager
from grasp.model import get_model
from grasp.tasks import get_task
from grasp.tasks.cea import AnnotationState, CeaSample
from grasp.tasks.entities import prepare_entity
from grasp.tasks.exploration import (
    FunctionalExplorationState,
    StructuralExplorationState,
)
from grasp.tasks.notes_from_samples import NotesFromSamplesState
from grasp.tasks.notes_from_traces import NotesFromTracesState
from grasp.tasks.question_generation import QuestionGenerationState
from grasp.tasks.sparql_qa.examples import SparqlQaSample
from grasp.tasks.utils import format_sparql_result, prepare_sparql_result
from grasp.utils import link


def dump_config(config: BaseModel, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.dump(config.model_dump(), f)


def take_notes_from_samples(
    task: str,
    config: NotesFromSamplesConfig,
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
        task,
        config,
        examples_model or config.embedding_model,
    )
    notes, kg_notes = load_notes(config)

    # optional dedicated note-taking model fully replaces the parent config's
    # model for the note-taking step; the agent runs on the parent config above
    note_taking_model = (
        get_model(config.note_taking_model) if config.note_taking_model else None
    )

    sample_cls = get_task(task, managers, config).sample_cls()
    assert sample_cls is not None, f"Task {task} does not support samples"

    assert config.seed is not None, "Seed must be set for adaptation"
    dump_config(config, out_dir)

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

    # Pre-generate sample sequence in epochs so every sample is seen once
    # per epoch before any is revisited across rounds.
    n_per_round = min(config.samples_per_round, len(all_samples))
    sequence: list[tuple[str, Sample]] = []
    epoch = 0
    while len(sequence) < config.num_rounds * n_per_round:
        random.seed(config.seed + epoch)
        sequence.extend(random.sample(all_samples, len(all_samples)))
        epoch += 1

    # when not running the agent, the note-taker gets the underlying task's
    # instructions so it knows what "solving" a sample means
    task_instructions = (
        None
        if config.run_agent
        else get_task(task, managers, config).system_information()
    )

    for r in trange(config.num_rounds, desc="Taking notes from samples"):
        samples = sequence[r * n_per_round : (r + 1) * n_per_round]

        if config.ignore_ground_truth:
            ground_truths = None
        else:
            ground_truths = prepare_ground_truths(samples, managers, config)

        if config.run_agent:
            outputs = []
            for kg, sample in tqdm(
                samples, desc="Running GRASP on samples", leave=False
            ):
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

            state: NotesFromTracesState = NotesFromTracesState(
                notes=notes,
                kg_notes=kg_notes,
                outputs=outputs,
                ground_truths=ground_truths,
            )
            note_task = "notes-from-traces"
        else:
            outputs = []
            state = NotesFromSamplesState(
                notes=notes,
                kg_notes=kg_notes,
                samples=[
                    {
                        "kg": kg,
                        "input": sample.input(),
                        "reference": None
                        if ground_truths is None
                        else ground_truths[i],
                    }
                    for i, (kg, sample) in enumerate(samples)
                ],
                task_instructions=task_instructions,
            )
            note_task = "notes-from-samples"

        # rebind so the note-taking functions mutate the same objects we dump below
        notes, kg_notes = state.notes, state.kg_notes

        output = consume_generator(
            generate(
                note_task,
                state,
                config,
                managers,
                kg_notes,
                notes,
                example_indices=example_indices,
                logger=agent_logger,
                custom_model=note_taking_model,
            )
        )

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

    agent_logger = get_logger("GRASP AGENT", log_level)

    managers, _ = setup(config)
    notes, kg_notes = load_notes(config)

    note_taking_model = (
        get_model(config.note_taking_model) if config.note_taking_model else None
    )

    assert config.seed is not None, "Seed must be set for adaptation"

    dump_config(config, out_dir)

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

    # Pre-generate sample sequence in epochs so every sample is seen once
    # per epoch before any is revisited across rounds.
    n_per_round = min(config.outputs_per_round, len(all_outputs))
    sequence: list[dict] = []
    epoch = 0
    while len(sequence) < config.num_rounds * n_per_round:
        random.seed(config.seed + epoch)
        sequence.extend(random.sample(all_outputs, len(all_outputs)))
        epoch += 1

    for r in trange(config.num_rounds, desc="Taking notes from outputs"):
        outputs = sequence[r * n_per_round : (r + 1) * n_per_round]

        state = NotesFromTracesState(notes=notes, kg_notes=kg_notes, outputs=outputs)
        # rebind so the note-taking functions mutate the same objects we dump below
        notes, kg_notes = state.notes, state.kg_notes

        output = consume_generator(
            generate(
                "notes-from-traces",
                state,
                config,
                managers,
                kg_notes,
                notes,
                logger=agent_logger,
                custom_model=note_taking_model,
            )
        )

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

    dump_config(config, out_dir)

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

    dump_config(config, out_dir)

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
            full_annot = prepare_entity(manager, annot.entity)
            annots.annotate(annot.row, annot.column, full_annot)

        return annots.format()

    else:
        raise ValueError(f"Unsupported or unknown sample type {type(sample)}")


def prepare_ground_truths(
    samples: list[tuple[str, Sample]],
    managers: list[KgManager],
    config: GraspConfig,
) -> list[str]:
    ground_truths = []
    for kg, sample in samples:
        gt = prepare_ground_truth(sample, kg, managers, config)
        ground_truths.append(gt)
    return ground_truths
