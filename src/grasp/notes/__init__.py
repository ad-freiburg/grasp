import os
import random
import uuid

import yaml
from pydantic import BaseModel
from search_rdf.model import SentenceTransformerModel
from tqdm import tqdm, trange
from universal_ml_utils.io import dump_json, dump_jsonl, load_json, load_jsonl
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
from grasp.tasks.entity_linking import AnnotationState as TextAnnotationState
from grasp.tasks.entity_linking import EntityLinkingSample
from grasp.tasks.exploration import (
    FunctionalExplorationState,
    StructuralExplorationState,
)
from grasp.tasks.notes_from_samples import NotesFromSamplesState
from grasp.tasks.notes_from_traces import NotesFromTracesState
from grasp.tasks.question_generation import QuestionGenerationState
from grasp.tasks.sparql_qa.examples import SparqlQaSample
from grasp.tasks.utils import format_sparql_result, prepare_sparql_result
from grasp.utils import is_model_error, link


def check_model_error(output: dict, what: str) -> None:
    # a round lost to a model error is not a round without findings
    if is_model_error(output):
        raise RuntimeError(
            f"{what} failed with a model error: {output['error']['content'][:500]}"
        )


def dump_config(config: BaseModel, out_dir: str, start_round: int = 0) -> None:
    os.makedirs(out_dir, exist_ok=True)
    # a resumed run keeps the config of the original run next to its own
    name = (
        "config.yaml" if start_round == 0 else f"config.start_round_{start_round}.yaml"
    )
    with open(os.path.join(out_dir, name), "w") as f:
        yaml.dump(config.model_dump(), f)


def prepare_out_dir(
    out_dir: str,
    overwrite: bool,
    start_round: int,
    num_rounds: int,
) -> None:
    if not 0 <= start_round < num_rounds:
        raise ValueError(
            f"Start round must be between 0 and {num_rounds - 1}, got {start_round}"
        )
    elif start_round > 0 and not os.path.isdir(out_dir):
        raise FileNotFoundError(
            f"Cannot resume at round {start_round}, "
            f"output directory {out_dir} does not exist"
        )
    elif start_round == 0 and os.path.exists(out_dir) and not overwrite:
        raise FileExistsError(f"Output directory {out_dir} already exists")


def previous_round_file(out_dir: str, name: str, start_round: int) -> str:
    path = os.path.join(out_dir, name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cannot resume at round {start_round}, {path} of the previous "
            "round is missing"
        )
    return path


def load_previous_kg_notes(
    config: GraspConfig,
    out_dir: str,
    prefix: str,
    start_round: int,
) -> dict[str, list[str]]:
    # a round only writes notes of KGs that have some
    kg_notes = {}
    for kg in config.knowledge_graphs:
        name = f"notes.{prefix}.{kg.kg}.round_{start_round - 1}.json"
        path = os.path.join(out_dir, name)
        if os.path.exists(path):
            kg_notes[kg.kg] = load_json(path)
    return kg_notes


def load_previous_notes(
    config: GraspConfig,
    out_dir: str,
    prefix: str,
    start_round: int,
) -> tuple[list[str], dict[str, list[str]]]:
    name = f"notes.{prefix}.round_{start_round - 1}.json"
    notes = load_json(previous_round_file(out_dir, name, start_round))
    return notes, load_previous_kg_notes(config, out_dir, prefix, start_round)


def take_notes_from_samples(
    task: str,
    config: NotesFromSamplesConfig,
    out_dir: str,
    overwrite: bool = False,
    log_level: str | int | None = None,
    start_round: int = 0,
) -> None:
    prepare_out_dir(out_dir, overwrite, start_round, config.num_rounds)

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
    if start_round > 0:
        notes, kg_notes = load_previous_notes(config, out_dir, task, start_round)

    # optional dedicated note-taking model fully replaces the parent config's
    # model for the note-taking step; the agent runs on the parent config above
    note_taking_model = (
        get_model(config.note_taking_model) if config.note_taking_model else None
    )

    sample_cls = get_task(task, managers, config).sample_cls()
    assert sample_cls is not None, f"Task {task} does not support samples"

    assert config.seed is not None, "Seed must be set for adaptation"
    dump_config(config, out_dir, start_round)

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

    for r in trange(start_round, config.num_rounds, desc="Taking notes from samples"):
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
                check_model_error(output, f"Agent run on a sample in round {r}")
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
                        # format the input as the agent would receive it
                        "input": get_task(task, managers, config).setup(sample.input()),
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
        check_model_error(output, f"Note-taking round {r}")

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
    start_round: int = 0,
) -> None:
    prepare_out_dir(out_dir, overwrite, start_round, config.num_rounds)

    agent_logger = get_logger("GRASP AGENT", log_level)

    managers, _ = setup(config)
    notes, kg_notes = load_notes(config)
    if start_round > 0:
        notes, kg_notes = load_previous_notes(config, out_dir, task, start_round)

    note_taking_model = (
        get_model(config.note_taking_model) if config.note_taking_model else None
    )

    assert config.seed is not None, "Seed must be set for adaptation"

    dump_config(config, out_dir, start_round)

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

    for r in trange(start_round, config.num_rounds, desc="Taking notes from outputs"):
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
        check_model_error(output, f"Note-taking round {r}")

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
    start_round: int = 0,
) -> None:
    assert isinstance(config, NotesFromExplorationConfig)
    prepare_out_dir(out_dir, overwrite, start_round, config.num_rounds)

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

    dump_config(config, out_dir, start_round)

    if config.mode == "functional":
        task_name = "exploration_functional"
        if start_round > 0:
            # the general notes are the only state, the KG notes are context
            name = f"notes.exploration.round_{start_round - 1}.json"
            notes = load_json(previous_round_file(out_dir, name, start_round))
        state = FunctionalExplorationState(notes=notes)
        # rebind
        notes = state.notes
    elif config.mode == "structural":
        task_name = "exploration_structural"
        explored = {}
        if start_round > 0:
            name = f"output.exploration.round_{start_round - 1}.json"
            previous = load_json(previous_round_file(out_dir, name, start_round))
            kg_notes = load_previous_kg_notes(
                config, out_dir, "exploration", start_round
            )
            if not kg_notes:
                raise FileNotFoundError(
                    f"Cannot resume at round {start_round}, no KG notes of the "
                    f"previous round in {out_dir}"
                )
            explored = (previous.get("output") or {}).get("explored") or {}
        state = StructuralExplorationState(kg_notes=kg_notes, explored=explored)
        # rebind
        kg_notes = state.kg_notes
    else:
        raise ValueError(f"Unknown exploration mode: {config.mode}")

    for r in trange(
        start_round, config.num_rounds, desc="Taking notes from exploration"
    ):
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

        check_model_error(output, f"Exploration round {r}")

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
    start_round: int = 0,
) -> None:
    prepare_out_dir(out_dir, overwrite, start_round, config.num_rounds)

    agent_logger = get_logger("GRASP AGENT", log_level)

    managers, _ = setup(config)
    notes, kg_notes = load_notes(config)

    dump_config(config, out_dir, start_round)

    state = QuestionGenerationState()
    if start_round > 0:
        # the previous round must have finished; it may have added no questions
        name = f"question-generation.trace.round_{start_round - 1}.json"
        previous = load_json(previous_round_file(out_dir, name, start_round))
        if is_model_error(previous):
            raise RuntimeError(
                f"Cannot resume at round {start_round}, the previous round "
                "failed with a model error"
            )
        for kg in config.knowledge_graphs:
            name = f"samples.{kg.kg}.round_{start_round - 1}.jsonl"
            path = os.path.join(out_dir, name)
            if os.path.exists(path):
                samples = load_jsonl(path)
                state.questions[kg.kg] = [s["question"] for s in samples]

    for r in trange(start_round, config.num_rounds, desc="Generating questions"):
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
        check_model_error(output, f"Question generation round {r}")

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

    elif isinstance(sample, EntityLinkingSample):
        manager, _ = find_manager(managers, kg)

        annots = TextAnnotationState(sample.text)
        for annot in sample.annotations:
            full_annot = prepare_entity(manager, annot.entity)
            # indices are relative to the annotation window, as in annotate
            annots.annotate(annot.start, annot.end, full_annot)

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
