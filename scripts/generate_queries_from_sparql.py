import argparse
import json
import os
from enum import Enum
from time import sleep

import torch
from pydantic import BaseModel
from tqdm import tqdm
from universal_ml_utils.io import dump_json, load_json, load_jsonl
from universal_ml_utils.logging import get_logger, setup_logging
from universal_ml_utils.ops import batch
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

from grasp.sparql.data import (
    get_sparql_items,
    natural_sparql_from_items,
    selections_from_items,
)
from grasp.sparql.manager import KgManager, load_kg_manager
from grasp.sparql.sparql import (
    find,
    find_all,
    parse_string,
    parse_to_string,
)

MAX_RESULTS = 8192


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input file")
    parser.add_argument("output", type=str, help="Output file")
    parser.add_argument(
        "-kg",
        "--knowledge-graph",
        type=str,
        default="wikidata",
    )
    parser.add_argument(
        "--kg-endpoint",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=22,
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=1,
        help="Batch size",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry failed examples",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file",
    )
    return parser.parse_args()


def system_prompt(kg: str) -> str:
    return f"""You are a helpful assistant that generates questions \
for SPARQL queries over {kg}.

You are given a SPARQL query, the entities and properties it uses, \
its natural language representation, and its execution result. \
You should then do the following:

1. Thinking
Think about the question that the SPARQL query may answer. \
If you come to the conclusion that the SPARQL query is invalid or in any other way \
not suitable for generating a question, you can skip the generation step and output \
a null value instead.

2. Generation
a) Clean the SPARQL query from unnecessary parts, e.g. unused variables, and \
replace all variable names and string literals with more meaningful equivalents. \
The cleaned query should be executable and therefore use the correct identifiers \
from the original SPARQL query and not their natural language labels.
b) Generate the question based on the cleaned SPARQL query. The question should \
be concise and not sound like a description of the SPARQL query itself. \
Avoid verbatim mentions of entities and properties in the question as much as possible, \
but rather ask about the information that the query retrieves. The question can be \
formulated in an asking or requesting manner, e.g. "What is the population of Germany?" \
or "number of people living in Germany".
c) Generate up to 3 paraphrases for the question, keeping the meaning the same.
d) Judge the complexity of the SPARQL query into simple, medium, or complex depending \
on the number of entities, properties, or advanced SPARQL features used.

Your output should be a JSON object of the following format:
{{
    "thought": "Your thoughts on the SPARQL query and the question it answers",
    "generation": {{
        "cleaned_sparql": "The cleaned SPARQL query",
        "question": "The question that the SPARQL query answers",
        "paraphrases": ["Paraphrase 1", ..., "Paraphrase 3"]
        "complexity": "simple/medium/complex",
    }} | null
}}
"""


def prompt(
    sparql: str,
    natural_sparql: str,
    selections: str,
    result: str,
) -> str:
    prompt = f"""\
SPARQL query:
{sparql}
"""

    if selections:
        prompt += f"""
{selections}
"""

    prompt += f"""
Natural language SPARQL query:
{natural_sparql}

Execution result:
{result}"""
    return prompt


class Complexity(str, Enum):
    simple = "simple"
    medium = "medium"
    complex = "complex"


class Generation(BaseModel):
    cleaned_sparql: str
    question: str
    complexity: Complexity
    paraphrases: list[str]


class Output(BaseModel):
    thought: str
    generation: Generation | None


def remove_service(manager: KgManager, sparql: str) -> str:
    parse, _ = parse_string(sparql, manager.sparql_parser)

    for service in find_all(parse, "ServiceGraphPattern"):
        var_or_iri = service["children"][2]

        iri = find(var_or_iri, "IRIREF")
        if iri is not None and iri["value"] == "<http://wikiba.se/ontology#label>":
            service.pop("children")
            continue

        pname = find(var_or_iri, "PNAME_LN")
        if pname is not None and pname["value"] == "wikibase:label":
            service.pop("children")
            continue

    return parse_to_string(parse)


def remove_unused_variables(manager: KgManager, sparql: str) -> str:
    parse, _ = parse_string(sparql, manager.sparql_parser)

    clause = find(parse, "SelectClause")
    if clause is None:
        return sparql

    used = set()
    for var in find_all(parse, "Var", skip={"SelectClause"}):
        used.add(var["children"][0]["value"])

    for var in find_all(clause, "SelectVar"):
        children = var["children"]
        if len(children) != 1:
            continue

        val = children[0]["children"][0]["value"]
        # keep Label variables from service clauses
        if val not in used and not val.endswith("Label"):
            var.pop("children")

    return parse_to_string(parse)


MAX_RETRIES = 5


def preprocess(
    sparql: str,
    manager: KgManager,
) -> tuple[dict, list]:
    preprocessed: dict = {"raw_sparql": sparql, "errors": []}

    sparql = remove_unused_variables(manager, sparql)
    sparql = remove_service(manager, sparql)
    preprocessed["sparql_no_service"] = sparql

    sparql, items = get_sparql_items(sparql, manager)
    preprocessed["sparql"] = sparql

    natural_sparql = natural_sparql_from_items(items)
    preprocessed["natural_sparql"] = natural_sparql

    if any(item.invalid for item in items):
        unknowns = [item.alternative.get_identifier() for item in items if item.invalid]
        raise RuntimeError(f"unknown_sparql_items: {unknowns}")

    result = None
    i = 0
    while True:
        try:
            result = manager.execute_sparql(sparql)
            result = manager.format_sparql_result(result)
            break
        except Exception as e:
            if i < MAX_RETRIES:
                sleep(i**2)
                i += 1
                continue

            raise RuntimeError(f"execute_sparql: {e}") from e

    if result is None:
        raise RuntimeError(f"execute_sparql: no result within {MAX_RETRIES} retries")

    preprocessed["result"] = result
    selections = manager.format_selections(selections_from_items(items))

    messages = [
        {"role": "system", "content": system_prompt(manager.kg)},
        {
            "role": "user",
            "content": prompt(
                sparql,
                natural_sparql,
                selections,
                result,
            ),
        },
    ]

    return preprocessed, messages


def postprocess(output: Output, manager: KgManager) -> dict:
    postprocessed: dict = {"empty": None, "result": None}
    if output.generation is None:
        return postprocessed

    sparql = output.generation.cleaned_sparql
    try:
        result = manager.execute_sparql(sparql)
        postprocessed["empty"] = result.is_empty
        result = manager.format_sparql_result(result)
        postprocessed["result"] = result
    except Exception as e:
        raise RuntimeError(f"execute_cleaned_sparql: {e}") from e

    return postprocessed


def complete(llm: LLM, sampling_params: SamplingParams, inputs: list) -> list[Output]:
    try:
        completions = llm.chat(
            messages=inputs,
            sampling_params=sampling_params,
            use_tqdm=False,
        )
    except Exception as e:
        raise RuntimeError(f"completion: {e}") from e

    parsed = []
    for completion in completions:
        try:
            text = completion.outputs[0].text
            parsed.append(Output(**json.loads(text)))
        except Exception as e:
            raise RuntimeError(f"output_parsing: {e}") from e

    return parsed


def run(args: argparse.Namespace) -> None:
    setup_logging()
    logger = get_logger("SPARQL TO QUESTION", args.log_level)

    sparqls = list(enumerate(load_jsonl(args.input)))

    outputs = {}
    if os.path.exists(args.output) and not args.overwrite:
        outputs = load_json(args.output)

    manager = load_kg_manager(args.knowledge_graph, endpoint=args.kg_endpoint)

    llm = LLM(args.model, tensor_parallel_size=torch.cuda.device_count())
    params = SamplingParams(
        seed=args.seed,
        top_p=0.9,
        temperature=0.2,
        max_tokens=8192,
        guided_decoding=GuidedDecodingParams(json=Output.model_json_schema()),
    )

    pbar = tqdm(desc="Generating questions", total=len(sparqls))

    for batched in batch(sparqls, args.batch_size):
        preprocessed, inputs = [], []
        for id, sparql in batched:
            pbar.update(1)

            # skip existing
            if id in outputs and (not args.retry_failed or "error" not in outputs[id]):
                continue

            try:
                res = preprocess(sparql, manager)
            except Exception as e:
                logger.error(f"Error in preprocessing: {e}")
                outputs[id] = {"error": str(e)}
                continue

            pre, inp = res
            preprocessed.append((id, pre))
            inputs.append(inp)

            logger.debug(f"Preprocessed:\n{json.dumps(pre, indent=2)}")
            logger.debug(f"Input:\n{json.dumps(inp, indent=2)}")

        try:
            generations = complete(
                llm,
                params,
                inputs,
            )
            assert len(generations) == len(preprocessed)
        except Exception as e:
            logger.error(f"Error in generation: {e}")
            # skip all without recording errors
            continue

        for (id, pre), output in zip(preprocessed, generations):
            try:
                post = postprocess(output, manager)
            except Exception as e:
                logger.error(f"Error in postprocessing: {e}")
                outputs[id] = {"error": str(e)}
                continue

            logger.debug(f"Output:\n{output.model_dump_json(indent=2)}")
            logger.debug(f"Postprocessed:\n{json.dumps(post, indent=2)}")
            outputs[id] = {"pre": pre, "output": output.model_dump(), "post": post}

        if (len(outputs) + 1) % 100 == 0:
            dump_json(outputs, args.output, indent=2)

    dump_json(outputs, args.output, indent=2)

    pbar.close()


if __name__ == "__main__":
    run(parse_args())
