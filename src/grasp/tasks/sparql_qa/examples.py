import math
import random
from typing import Any

from grasp.configs import GraspConfig
from grasp.functions import (
    find_manager,
    format_page_suffix,
    paginate_results,
    validate_page,
    validate_query,
)
from grasp.manager import KgManager
from grasp.examples import ExampleIndex, Sample
from grasp.tasks.utils import format_sparql_result, prepare_sparql_result
from grasp.utils import format_list


class SparqlQaSample(Sample):
    question: str
    sparql: str
    paraphrases: list[str] = []
    info: dict[str, Any] = {}

    def input(self) -> str:
        return self.question

    def queries(self) -> list[str]:
        return [self.question] + self.paraphrases


class SparqlQaExampleIndex(ExampleIndex):
    sample_cls = SparqlQaSample


def functions(config: GraspConfig) -> list[dict]:
    example_kgs = [kg.kg for kg in config.knowledge_graphs if kg.examples is not None]

    if not example_kgs:
        return []

    example_info = format_list(f'"{kg}"' for kg in example_kgs)

    if config.random_examples:
        fn = {
            "name": "get_random_examples",
            "description": f"""\
Find random examples of SPARQL-question-pairs over the specified knowledge graph. \
At most {config.num_examples} examples are returned per page (use pagination \
up to page {config.search_max_pages} to see more examples).

For example, to find examples of SPARQL-question-pairs over Wikidata, do the following:
get_random_examples(kg="wikidata", page=1)

Currently, examples are available for the following knowledge graphs:
{example_info}""",
            "parameters": {
                "type": "object",
                "properties": {
                    "kg": {
                        "type": "string",
                        "enum": example_kgs,
                        "description": "The knowledge graph to find examples for",
                    },
                    "page": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": config.search_max_pages,
                        "description": "The page number to return. Use pagination "
                        "to see more examples.",
                    },
                },
                "required": ["kg", "page"],
                "additionalProperties": False,
            },
            "strict": True,
        }

    else:
        fn = {
            "name": "search_example",
            "description": f"""\
Find SPARQL-question-pairs over the specified knowledge graph that \
try to answer a similar question to the one provided (e.g., can be the \
user question itself, a sub-question or reformulation of it, or \
any other question of interest). At most {config.num_examples} examples \
are returned per page (use pagination up to page {config.search_max_pages} \
to see more examples).

For example, to find similar SPARQL-question-pairs to the question \
"What is the capital of France?" over Wikidata, do the following:
search_example(kg="wikidata", query="What is the capital of France?", page=1)

Currently, examples are available for the following knowledge graphs:
{example_info}""",
            "parameters": {
                "type": "object",
                "properties": {
                    "kg": {
                        "type": "string",
                        "enum": example_kgs,
                        "description": "The knowledge graph to find examples for",
                    },
                    "query": {
                        "type": "string",
                        "description": "The question to find examples for",
                    },
                    "page": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": config.search_max_pages,
                        "description": "The page number to return. Use pagination "
                        "to see more examples.",
                    },
                },
                "required": ["kg", "query", "page"],
                "additionalProperties": False,
            },
            "strict": True,
        }

    return [fn]


def call_function(
    config: GraspConfig,
    managers: list[KgManager],
    fn_name: str,
    fn_args: dict,
    known: set[str],
    example_indices: dict[str, SparqlQaExampleIndex] | None = None,
) -> str:
    ei = example_indices or {}

    if fn_name == "get_random_examples":
        return get_random_examples(
            managers,
            ei,
            fn_args["kg"],
            config.num_examples,
            known,
            config.result_max_rows,
            config.result_max_columns,
            fn_args.get("page") or 1,
            config.search_max_pages,
            config.seed,
        )

    elif fn_name == "search_example":
        return search_example(
            managers,
            ei,
            fn_args["kg"],
            fn_args["query"],
            config.num_examples,
            known,
            config.result_max_rows,
            config.result_max_columns,
            fn_args.get("page") or 1,
            config.search_max_pages,
        )

    else:
        raise ValueError(f"Unknown function {fn_name}")


def format_examples(
    kg: str,
    managers: list[KgManager],
    examples: list[SparqlQaSample],
    known: set[str],
    max_rows: int,
    max_cols: int,
    page: int,
    total_pages: int,
    more: bool,
    start_index: int = 0,
) -> str:
    manager, _ = find_manager(managers, kg)
    exs = []

    for i, example in enumerate(examples, start_index):
        result, selections = prepare_sparql_result(
            example.sparql,
            kg,
            managers,
            max_rows,
            max_cols,
            known,
        )

        exs.append(
            f"Example {i + 1}:\n{example.question}\n\n"
            + format_sparql_result(manager, result, selections)
        )

    suffix = format_page_suffix(page, total_pages, more)
    if not exs:
        return f"No examples found ({suffix})"

    return f"Examples ({suffix}):\n" + "\n\n".join(exs)


def get_random_examples(
    managers: list[KgManager],
    example_indices: dict[str, SparqlQaExampleIndex],
    kg: str,
    num_examples: int,
    known: set[str],
    max_rows: int,
    max_cols: int,
    page: int = 1,
    max_pages: int = 10,
    seed: int | None = None,
) -> str:
    validate_page(page, max_pages)

    if kg not in example_indices:
        return f"No example index for knowledge graph {kg}"

    example_index = example_indices[kg]
    visible_count = min(len(example_index), num_examples * max_pages)
    more = len(example_index) > visible_count
    total_pages = max(1, math.ceil(visible_count / num_examples))
    start = (page - 1) * num_examples
    end = page * num_examples

    rand = random.Random(seed if seed is not None else len(example_index))
    perm = list(range(len(example_index)))
    rand.shuffle(perm)
    examples = [example_index[i] for i in perm[start:end]]

    return format_examples(
        kg,
        managers,
        examples,  # type: ignore
        known,
        max_rows,
        max_cols,
        page,
        total_pages,
        more,
        start,
    )


def search_example(
    managers: list[KgManager],
    example_indices: dict[str, SparqlQaExampleIndex],
    kg: str,
    query: str,
    num_examples: int,
    known: set[str],
    max_rows: int,
    max_cols: int,
    page: int = 1,
    max_pages: int = 10,
) -> str:
    validate_query(query)
    validate_page(page, max_pages)

    if kg not in example_indices:
        return f"No example index for knowledge graph {kg}"

    example_index = example_indices[kg]
    all_examples = list(example_index.search(query, num_examples * max_pages + 1))
    examples, total_pages, more = paginate_results(
        all_examples, num_examples, page, max_pages
    )
    start = (page - 1) * num_examples

    return format_examples(
        kg,
        managers,
        examples,
        known,
        max_rows,
        max_cols,
        page,
        total_pages,
        more,
        start,
    )


def find_examples(
    managers: list[KgManager],
    example_indices: dict[str, SparqlQaExampleIndex],
    kg: str,
    question: str,
    random_examples: bool,
    num_examples: int,
    known: set[str],
    max_rows: int,
    max_cols: int,
    page: int = 1,
    max_pages: int = 10,
) -> str:
    if random_examples:
        return get_random_examples(
            managers,
            example_indices,
            kg,
            num_examples,
            known,
            max_rows,
            max_cols,
            page,
            max_pages,
        )

    else:
        return search_example(
            managers,
            example_indices,
            kg,
            question,
            num_examples,
            known,
            max_rows,
            max_cols,
            page,
            max_pages,
        )
