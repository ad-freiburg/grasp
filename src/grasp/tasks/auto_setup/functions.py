from typing import Literal

from grammar_utils.parse import LR1Parser  # type: ignore
from pydantic import BaseModel, Field

from grasp.function_args import Sparql, function_definition
from grasp.sparql.utils import (
    find,
    parse_string,
    parse_to_string_with_whitespace,
)

STOP_FUNCTION = function_definition("stop", "Stop the setup process.")


class SetDescriptionArgs(BaseModel):
    description: str = Field(description="The description")


class SetQueryArgs(BaseModel):
    type: Literal["index", "info"] = Field(
        description="Whether this is an index query or an info query",
    )
    sparql: Sparql | None = Field(
        description="The SPARQL query, or null to clear / unset the current query.",
    )


class ShowIndexSetupArgs(BaseModel):
    index: str = Field(description="The index to show the setup for")


class AddPrefixArgs(BaseModel):
    short: str = Field(description="The short prefix name (e.g. 'wd')")
    namespace: str = Field(
        description="The full IRI namespace (e.g. 'http://www.wikidata.org/entity/')",
    )


class DeletePrefixArgs(BaseModel):
    short: str = Field(description="The short prefix name to delete")


class UpdatePrefixArgs(BaseModel):
    short: str = Field(description="The short prefix name to update")
    namespace: str = Field(description="The new full IRI namespace")


def set_description_function(description: str) -> dict:
    return function_definition("set_description", description, SetDescriptionArgs)


def index_functions(others: list[str]) -> list[dict]:
    fns: list[dict] = [
        function_definition(
            "show_setup",
            "Show the current index and info SPARQL queries for the knowledge graph.",
        ),
        function_definition(
            "set_query",
            "Set or clear the index or info SPARQL query for the knowledge graph.",
            SetQueryArgs,
        ),
        set_description_function(
            "Set a concise description of what this index contains and how it is built. "
            "Typically a single sentence is sufficient."
        ),
    ]

    # lets the agent check the other indices that are already set up,
    # so that overlaps with them are deliberate
    if others:
        fns.append(
            function_definition(
                "show_index_setup",
                "Show the setup (index SPARQL, info SPARQL, and description) "
                "of another index of this knowledge graph that is already set up.",
                ShowIndexSetupArgs,
                {"index": others},
            )
        )

    fns.append(STOP_FUNCTION)
    return fns


def info_functions() -> list[dict]:
    return [
        function_definition(
            "show_setup",
            "Show the current prefixes and description of the knowledge graph.",
        ),
        function_definition(
            "add_prefix",
            "Add a new prefix for the knowledge graph, mapping a short "
            "prefix name to its full IRI namespace (e.g. 'wd' for "
            "'http://www.wikidata.org/entity/'). Only knowledge graph specific "
            "prefixes need to be added - common prefixes like rdf, rdfs, and "
            "xsd are available by default.",
            AddPrefixArgs,
        ),
        function_definition(
            "delete_prefix",
            "Delete an existing knowledge graph prefix.",
            DeletePrefixArgs,
        ),
        function_definition(
            "update_prefix",
            "Update the namespace of an existing knowledge graph prefix.",
            UpdatePrefixArgs,
        ),
        set_description_function(
            "Set a concise description of what the knowledge graph contains. "
            "Typically a single sentence about the domain and scope of the "
            "knowledge graph is sufficient."
        ),
        STOP_FUNCTION,
    ]


def clean(s: str) -> str:
    return "".join(s.split())


def find_select_clause(parse: dict, enc: bytes) -> str:
    clause = find(parse, "SelectClause", skip={"SubSelect"})
    if clause is None:
        raise ValueError("No SELECT clause found in query")
    return parse_to_string_with_whitespace(clause, enc)


def find_solution_modifier(parse: dict, enc: bytes) -> str:
    sol_mod = find(parse, "SolutionModifier", skip={"SubSelect"})
    if sol_mod is None:
        raise ValueError("No solution modifier found in query")
    return parse_to_string_with_whitespace(sol_mod, enc)


def validate_select_clause(parse: dict, enc: bytes, target: str):
    select = find_select_clause(parse, enc)
    if clean(select) != clean(target):
        raise ValueError(f"SELECT clause must be '{target}', but got '{select}'")


def validate_solution_modifier(parse: dict, enc: bytes, target: str):
    sol_mod = find_solution_modifier(parse, enc)
    if clean(sol_mod) != clean(target):
        raise ValueError(f"Solution modifier must be '{target}', but got '{sol_mod}'")


INDEX_SPARQL_SELECT = "SELECT DISTINCT ?id ?value ?tags"
INDEX_SPARQL_SOL_MOD = "ORDER BY DESC(?score) ?id DESC(?tags)"


def validate_index_sparql(parser: LR1Parser, sparql: str):
    parse, _ = parse_string(sparql, parser)
    enc = sparql.encode()

    validate_select_clause(parse, enc, INDEX_SPARQL_SELECT)
    validate_solution_modifier(parse, enc, INDEX_SPARQL_SOL_MOD)


INFO_SPARQL_SELECT = "SELECT DISTINCT ?id ?value ?type"
INFO_SPARQL_SOL_MOD = "ORDER BY ?id ?type ?value"


def validate_info_sparql(parser: LR1Parser, sparql: str):
    parse, _ = parse_string(sparql, parser)
    enc = sparql.encode()

    validate_select_clause(parse, enc, INFO_SPARQL_SELECT)
    validate_solution_modifier(parse, enc, INFO_SPARQL_SOL_MOD)
