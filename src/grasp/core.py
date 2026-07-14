import json
import time
import traceback
import uuid
from copy import deepcopy
from logging import Logger
from typing import Any, Generator

from openai import APIConnectionError
from universal_ml_utils.io import load_json
from universal_ml_utils.logging import get_logger

from grasp.configs import GraspConfig
from grasp.examples import ExampleIndex
from grasp.functions import call_function, kg_functions
from grasp.manager import KgManager, format_kgs, load_kg_manager
from grasp.manager.utils import (
    EmbeddingModel,
    describe_index_type,
    get_common_sparql_prefixes,
)
from grasp.model import Message, Model, Response, ToolCall, get_model
from grasp.tasks import get_task
from grasp.tasks import rules as general_rules
from grasp.tasks.base import GraspTask
from grasp.tasks.feedback import format_feedback, generate_feedback
from grasp.tasks.sparql_qa.examples import find_examples
from grasp.utils import (
    format_enumerate,
    format_error,
    format_message,
    format_notes,
    format_prefixes,
    format_response,
    format_section,
)


def system_instructions(
    task: GraspTask,
    managers: list[KgManager],
    kg_notes: dict[str, list[str]],
    notes: list[str],
    example_indices: dict | None = None,
) -> str:
    index_types = set()
    for manager in managers:
        for idx in manager.indices.values():
            if idx.index is not None:
                index_types.add(idx.index.index_type)

    index_infos = []
    for index_type in sorted(index_types):
        desc = describe_index_type(index_type)
        index_infos.append(f'"{index_type}": {desc}')

    blocks = [task.system_information()]

    if index_infos:
        blocks.append(
            format_section(
                "Knowledge graph index types",
                "\n".join(index_infos),
            )
        )

    if managers:
        blocks.append(
            format_section(
                "Available knowledge graphs",
                format_kgs(managers, kg_notes, example_indices),
            )
        )

    if notes:
        blocks.append(
            format_section(
                "General notes across knowledge graphs",
                format_notes(notes, enumerated=True),
            )
        )

    common_prefixes = get_common_sparql_prefixes()
    if common_prefixes:
        blocks.append(
            format_section(
                "Common SPARQL prefixes",
                format_prefixes(common_prefixes, bullet=""),
            )
        )

    if common_prefixes or any(manager.kg_prefixes for manager in managers):
        blocks.append(
            "All SPARQL prefixes above can be used implicitly in SPARQL queries "
            "and function calls."
        )

    rules = general_rules() + task.rules()
    if rules:
        blocks.append(format_section("Rules to follow", format_enumerate(rules)))

    return "\n\n".join(blocks)


def setup(config: GraspConfig) -> tuple[list[KgManager], dict[str, EmbeddingModel]]:
    models: dict[str, EmbeddingModel] = {}
    managers: list[KgManager] = []
    for kg in config.knowledge_graphs:
        manager = load_kg_manager(kg)
        models = manager.load_models(models, embedding_model=config.embedding_model)
        managers.append(manager)

    return managers, models


def load_notes(config: GraspConfig) -> tuple[list[str], dict[str, list[str]]]:
    if config.notes_file is None:
        general_notes = []
    else:
        general_notes = load_json(config.notes_file)

    kg_notes = {}
    for kg in config.knowledge_graphs:
        if kg.notes_file is None:
            continue

        kg_notes[kg.kg] = load_json(kg.notes_file)

    return general_notes, kg_notes  # type: ignore


def generate(
    task_name: str,
    input: Any,
    config: GraspConfig,
    managers: list[KgManager],
    kg_notes: dict[str, list[str]] | None = None,
    notes: list[str] | None = None,
    example_indices: dict[str, ExampleIndex] | None = None,
    past_messages: list[Message] | None = None,
    past_known: set[str] | None = None,
    logger: Logger = get_logger("GRASP"),
    yield_output: bool = False,
    custom_model: Model | None = None,
) -> Generator[dict, None, dict]:
    if task_name != "sparql-qa" and task_name != "general-qa":
        # disable examples for tasks other than sparql-qa and general-qa
        # to avoid errors due to missing implementations
        config = deepcopy(config)
        config.force_examples = None
        logger.debug(f"Disabling examples for {task_name} task")
    if task_name == "general-qa":
        config = deepcopy(config)
        config.tool_choice = "auto"
        logger.debug("Setting tool choice to auto for general-qa task")
    if task_name == "cea":
        config = deepcopy(config)
        config.know_before_use = True
        logger.debug("Enabling know-before-use for cea task")

    task = get_task(task_name, managers, config, past_known)

    input = task.setup(input)

    # setup functions (after setup so tasks can configure based on input)
    fns = kg_functions(
        managers,
        config.fn_set,
        config.list_k,
        config.search_k,
        config.search_max_pages,
    )
    fns.extend(task.function_definitions())
    yield {"type": "input", "input": input}

    model = custom_model or get_model(config)

    feedback_notes = notes
    feedback_kg_notes = kg_notes
    if config.notes_only_for_feedback:
        notes = None
        kg_notes = None

    # setup messages
    system_instruction = system_instructions(
        task,
        managers,
        kg_notes or {},
        notes or [],
        example_indices,
    )
    yield {
        "type": "system",
        "config": config.model_dump(),
        "functions": fns,
        "system_message": system_instruction,
    }

    # log stuff
    config_msg = Message(role="config", content=config.model_dump_json(indent=2))
    logger.debug(format_message(config_msg))

    fn_msg = Message(
        role="functions",
        content=json.dumps([fn["name"] for fn in fns], indent=2),
    )
    logger.debug(format_message(fn_msg))

    # handle past
    if past_messages:
        first, *others = past_messages
        assert isinstance(first.content, str)
        assert first.role == "system", "First past message should be system"
        messages = [Message.system(content=first.content), *others]
    else:
        messages = [Message.system(content=system_instruction)]

    start = time.monotonic()

    # add user input
    messages.append(Message.user(content=input))

    if (
        config.force_examples
        and example_indices
        and config.force_examples in example_indices
        # dont do it on follow ups
        and not past_messages
    ):
        result = find_examples(
            managers,
            example_indices,  # type: ignore
            config.force_examples,
            input,
            config.random_examples,
            config.num_examples,
            task.known,
            config.result_max_rows,
            config.result_max_columns,
        )

        name = "get_random_examples"
        args = {"kg": config.force_examples, "page": 1}
        if not config.random_examples:
            name = "search_example"
            args["query"] = input

        tool_call = ToolCall(
            id=uuid.uuid4().hex,
            name=name,
            args=args,
            result=result,
        )
        message = Message.assistant(
            Response(
                id=f"msg_{uuid.uuid4().hex}",
                tool_calls=[tool_call],
            )
        )
        messages.append(message)

        # yield to user as tool call
        yield {
            "type": "tool",
            "name": name,
            "args": args,
            "result": result,
        }

    # log all messages so far
    for msg in messages:
        logger.debug(format_message(msg))

    num_messages = len(messages)
    error: dict | None = None

    # keep track of last serialized message to detect loops
    # if model emits the same message twice, we are stuck
    last_resp_hash: str | None = None
    # number of loops detected so far; we allow up to config.max_loops
    # loops before giving up
    loops = 0
    retries = 0
    while len(messages) - num_messages < config.max_steps:
        try:
            response = model(messages, fns)
        except APIConnectionError:
            error = {
                "content": "LLM API timed out or failed to connect",
                "reason": "timeout",
            }
            logger.error("LLM API timed out or failed to connect")
            break
        except Exception as e:
            error = {
                "content": f"Failed to generate response:\n{e}\n\n"
                f"{traceback.format_exc()}",
                "reason": "api",
            }
            logger.error(format_error(**error))
            break

        if response.is_empty:
            error = {
                "content": "Empty response from LLM API",
                "reason": "empty",
            }
            logger.error(format_error(**error))
            break

        messages.append(Message.assistant(response))

        resp_hash = response.hash()
        if last_resp_hash == resp_hash:
            loops += 1
            if loops > config.max_loops:
                error = {
                    "content": "LLM appears to be stuck in a loop",
                    "reason": "loop",
                }
                logger.error(format_error(**error))
                break

            # the duplicate response's tool calls are not executed; mark
            # their results so the message history stays valid
            for tool_call in response.tool_calls:
                tool_call.result = "Not executed due to loop"

            # inform the model that a loop was detected and let it recover
            loop_msg = (
                "A loop was detected: your last response was identical to the "
                "previous one. Please change your approach to make progress."
            )
            messages.append(Message.user(loop_msg, name="loop"))
            yield {"type": "loop", "content": loop_msg}
            logger.warning(f"Loop detected ({loops}/{config.max_loops})")
            # reset so the injected message gets a fresh chance
            last_resp_hash = None
            continue

        last_resp_hash = resp_hash

        # yield message if there is content
        if response.has_content:
            yield {
                "type": "model",
                **response.get_content(),
                "elapsed": response.elapsed,
            }

        # no tool calls mean we should stop
        should_stop = not response.tool_calls

        # execute tool calls
        for tool_call in response.tool_calls:
            fn_start = time.monotonic()
            try:
                result = call_function(
                    config,
                    managers,
                    tool_call.name,
                    tool_call.args,
                    task.known,
                    task,
                    example_indices,
                )
            except Exception as e:
                tool_call.error = str(e)
                result = f"Call to function {tool_call.name} returned an error:\n{e}"

                # log full tracback for debugging
                # import traceback
                #
                # traceback_str = "".join(traceback.format_exc())
                # logger.error(f"Full traceback:\n{traceback_str}")

            tool_call.result = result
            tool_call.elapsed = time.monotonic() - fn_start

            yield {
                "type": "tool",
                "name": tool_call.name,
                "args": tool_call.args,
                "result": tool_call.result,
                "elapsed": tool_call.elapsed,
            }

            if task.done(tool_call.name):
                # we are done
                should_stop = True

        # only log now to show also tool calls results
        logger.debug(format_response(response))

        can_give_feedback = config.feedback and retries < config.max_feedbacks

        if should_stop and not can_give_feedback:
            # done
            break

        elif not should_stop:  # and (choice.message.tool_calls or alternating):
            # not done yet
            continue

        elif not can_give_feedback:
            # no feedback possible, despite answer or cancel
            break

        # get latest output
        output = task.output(messages)
        if output is None:
            break

        # provide feedback
        try:
            inputs = [
                message.content
                for message in messages
                if isinstance(message.content, str) and message.role == "user"
            ]
            feedback = generate_feedback(
                model,
                task,
                feedback_kg_notes or kg_notes or {},
                feedback_notes or notes or [],
                inputs,  # type: ignore
                output,
                logger,
            )
        except Exception as e:
            error = {
                "content": f"Failed to generate feedback:\n{e}",
                "reason": "feedback",
            }
            logger.error(format_error(**error))
            break

        if feedback is None:
            # no feedback
            break

        messages.append(Message.user(format_feedback(feedback), name="feedback"))
        yield {
            "type": "feedback",
            "status": feedback["status"],
            "feedback": feedback["feedback"],
        }

        if feedback["status"] == "done":
            break

        # if not done, continue
        retries += 1
        # reset loop detection
        last_resp_hash = None

    output = task.output(messages)

    out_msg = Message(
        role="output",
        content="No output"
        if output is None
        else output.get("formatted", json.dumps(output, indent=2)),
    )
    logger.info(format_message(out_msg))

    end = time.monotonic()
    output = {
        "type": "output",
        "task": task_name,
        "output": output,
        "elapsed": end - start,
        "error": error,
        "messages": [message.model_dump() for message in messages],
        "known": list(task.known),
    }

    if yield_output:
        yield output

    return output
