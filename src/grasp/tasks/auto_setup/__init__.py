import os

from pydantic import BaseModel

from grasp.configs import KgInfo
from grasp.functions import check_known, update_known_from_alts
from grasp.manager import DEFAULT_DESCRIPTIONS, KgManager
from grasp.manager.utils import (
    get_common_sparql_prefixes,
    load_index_description,
    load_index_sparql,
    load_info_sparql,
    load_kg_info,
    merge_prefixes,
)
from grasp.model import Message
from grasp.sparql.item import extract_sparql_items
from grasp.sparql.types import ObjType
from grasp.sparql.utils import (
    get_qlever_endpoint,
    load_entity_index_sparql,
    load_entity_info_sparql,
    load_literal_index_sparql,
    load_literal_info_sparql,
    load_property_index_sparql,
    load_property_info_sparql,
)
from grasp.tasks.auto_setup.functions import (
    index_functions,
    info_functions,
    validate_index_sparql,
    validate_info_sparql,
)
from grasp.tasks.base import GraspTask
from grasp.tasks.functions import find_frequent, find_frequent_function_definition
from grasp.utils import FunctionCallException, format_prefixes, get_index_dir

# maps index name to index sparql and info sparql reference defaults
REFERENCE_SETUP = {
    "entities": {
        "index": load_entity_index_sparql(),
        "info": load_entity_info_sparql(),
    },
    "properties": {
        "index": load_property_index_sparql(),
        "info": load_property_info_sparql(),
    },
    "literals": {
        "index": load_literal_index_sparql(),
        "info": load_literal_info_sparql(),
    },
}


class IndexState(BaseModel):
    index_sparql: str | None = None
    info_sparql: str | None = None
    description: str | None = None


class InfoState(BaseModel):
    prefixes: dict[str, str] | None = None
    description: str | None = None


def update_known_from_sparql(sparql: str, known: set[str], manager: KgManager) -> None:
    disable_info_retrieval = manager.disable_info_retrieval
    try:
        # temporarily disable info retrieval
        manager.set_info_retrieval(enable=False)
        _, items = extract_sparql_items(sparql, manager)
        for obj_type in [ObjType.ENTITY, ObjType.PROPERTY]:
            normalizer = manager.get_normalizer(obj_type.index_name)
            update_known_from_alts(
                known,
                (item.alternative for item in items if not item.is_literal),
                normalizer,
            )
    except Exception:
        pass
    finally:
        manager.set_info_retrieval(disable_info_retrieval)


def load_index_state(manager: KgManager, name: str, known: set[str]) -> IndexState:
    index_dir = get_index_dir(manager.kg)
    sub_index_dir = os.path.join(index_dir, name)
    index_sparql = load_index_sparql(sub_index_dir, manager.logger)
    if index_sparql is not None:
        update_known_from_sparql(index_sparql, known, manager)

    info_sparql = load_info_sparql(sub_index_dir, manager.logger)
    if info_sparql is not None:
        update_known_from_sparql(info_sparql, known, manager)

    description = load_index_description(sub_index_dir, manager.logger)
    return IndexState(
        index_sparql=index_sparql,
        info_sparql=info_sparql,
        description=description,
    )


def load_info_state(manager: KgManager) -> InfoState:
    kg_info = load_kg_info(manager.kg)
    return InfoState(
        prefixes=kg_info.prefixes,
        description=kg_info.description,
    )


def format_query(manager: KgManager, query: str | None) -> str:
    if query is None:
        return "None"
    return manager.fix_prefixes(query, remove_known=True)


def format_index_state(state: IndexState, manager: KgManager) -> str:
    return f"""\
Description:
{state.description or "None"}

Index SPARQL:
{format_query(manager, state.index_sparql)}

Info SPARQL:
{format_query(manager, state.info_sparql)}"""


def format_info_state(state: InfoState) -> str:
    return f"""\
Description:
{state.description or "None"}

Prefixes:
{format_prefixes(state.prefixes)}"""


class AutoSetupTask(GraspTask):
    name = "auto-setup"

    def system_information(self) -> str:
        manager = self.managers[0]

        if self.input["phase"] == "index":
            return self._index_system_information(manager)
        else:
            return self._info_system_information()

    def _index_system_information(self, manager: KgManager) -> str:
        name = self.input["name"]
        reference = REFERENCE_SETUP[name]

        return f"""\
You are a knowledge graph setup assistant. Your task is to explore \
the {manager.kg} knowledge graph and come up with or improve the setup \
- an index SPARQL query, an info SPARQL query, and a description - \
of the {name} index.

The index SPARQL query is used to build a search index over {name}. \
It must be a SELECT query returning:
- ?id: the IRI
- ?value: the string literal to index for search (e.g., a label or alias)
- ?tags: "main" for the preferred indexed literal for an IRI, unbound \
for other indexed literals of the same IRI
Results should be ordered by relevance score descending, then by ?id and ?tags descending. \
Default to the total number of occurrences of an IRI as its relevance score. \
If a knowledge graph provides another measure of relevance you can use that instead \
(e.g., wikibase:sitelinks for Wikidata). When choosing literals to include in the \
index SPARQL, make sure that they uniquely or near-uniquely identify the corresponding \
IRI, as each indexed literal for an IRI is treated individually during search. Any \
descriptive information shared by many IRIs (e.g., their type or common properties) \
should rather be included in the info SPARQL for disambiguation. A good rule of thumb: \
if searching for the literal is likely to return the target IRI among the top results, \
include it. For example, for Angela Merkel include "Angela Merkel" or "Merkel", but not \
"politician" or "human".

The info SPARQL query retrieves additional context for {name} retrieved \
via search. It must be a SELECT query returning:
- ?id: the IRI
- ?value: a descriptive string literal (e.g., a label, alias, description, or type)
- ?type: one of "label" or "alias" (similar to ?tags in the index SPARQL), \
or "other" for any additional descriptive information
It must contain the placeholder {{IDS}} which will be replaced with a \
space-separated list of IRIs at query time. The information retrieved per IRI should \
be limited to the most important ones (10 or fewer) to keep the query efficient \
and the search results concise. The goal is to help to characterize and distinguish \
different IRIs, and not to list all their associated information.

The description should be a concise summary of what the {name} index is \
about and what data it contains.

You should follow a step-by-step approach:
1. Look at and understand the current setup. It might also help to look at the \
default reference setup below.
2. Thorougly explore the knowledge graph using the provided functions to understand \
its structure. If the current setup is non-empty, validate it and \
try to find potential issues or improvements.
3. Update the setup based on your findings. Make sure to verify \
and execute SPARQL queries against the knowledge graph before setting them.
4. Perform a final review of the setup. If you see any issues, \
go back to step 2 and repeat, otherwise stop.

Below is a reference setup you can use as a starting point if \
no {name} index setup is available yet. It is generic and thus \
may not be optimal for the knowledge graph at hand.

Reference {name} index SPARQL:
{format_query(manager, reference["index"])}

Reference {name} info SPARQL:
{format_query(manager, reference["info"])}

Reference {name} index description:
{DEFAULT_DESCRIPTIONS[name]}"""

    def _info_system_information(self) -> str:
        manager = self.managers[0]

        return f"""\
You are a knowledge graph setup assistant. Your task is to explore \
the {manager.kg} knowledge graph and come up with or improve its general \
setup, which consists of a set of prefixes and a description.

The prefixes map short names to IRI namespaces (e.g., "wd" is short for \
"http://www.wikidata.org/entity/"). Only knowledge graph specific \
prefixes are needed as common ones like rdf, rdfs, owl, xsd are \
already available by default.

The description should be a concise summary of what the knowledge graph \
is about.

You should follow a step-by-step approach:
1. Look at and understand the current setup.
2. Thorougly explore the knowledge graph using the provided functions to \
discover its scope, structure, and frequently used IRI namespaces. If \
the current setup is non-empty, validate it and try to find potential issues \
or improvements.
3. Update the current setup based on your findings.
4. Perform a final review of the setup. If you see any issues, go back to step 2 \
and repeat, otherwise stop."""

    def rules(self) -> list[str]:
        if self.input["phase"] == "index":
            return self._index_rules()
        else:
            return self._info_rules()

    def _index_rules(self) -> list[str]:
        name = self.input["name"]
        rules = [
            "If the user provides additional notes about the desired setup, make sure to follow them.",
            "Make the SPARQL queries as efficient as possible. For example, put VALUES { {IDS} } clauses "
            "in the info SPARQL inside each UNION, and avoid heavy string operations and string filters for the index "
            "SPARQL. You also don't need to handle duplicates yourself in the SPARQL (e.g., via DISTINCT or GROUP BY), "
            "as the indexing and info retrieval processes will take care of that. "
            "The info SPARQL should run in a fraction of a second, whereas "
            "the index SPARQL is allowed to take many minutes or even hours on the full knowledge graph. "
            "The latter should be tested on a subset of the data (e.g., using VALUES) to stay "
            "within the lower time limits during development.",
            f"Do not use different scores for the same id in the {name} index SPARQL, as the ids are "
            "required to be returned in contiguous blocks for the indexing process.",
        ]

        if name in ("entities", "properties"):
            rules.append(
                f"To include {name} in the index and make them searchable even if they do not have "
                "have any associated literals, use their IRIs as values by binding ?id to ?value directly "
                "in the index SPARQL. During indexing the local part of the IRI (after a known prefix, "
                "or the last slash or hash) will be extracted and indexed as the value, so make sure it is "
                "meaningful for search. This also means you do not need to extract the local part in the SPARQL "
                "yourself."
            )

        if name == "entities":
            rules.append(
                f"Not all {name} in the knowledge graph need to be searchable and should be covered by the index. "
                f"Typical examples are identifier-like, internal, or intermediate {name} "
                "without any descriptive associated literals."
            )

        if name == "literals":
            rules.append(
                "For a literals index, ?id and ?value are typically the same: bind ?id to the literal itself "
                "and bind ?value to STR(?id). Since literals are their own identifiers, there is no separate "
                "IRI to use as ?id."
            )
            rules.append(
                "Only index standalone literals - string values that appear as objects in the knowledge graph "
                "(e.g., enumerations like amenity types, status codes, license names). Do not index IRIs (use "
                "the entity / property indices for those) and do not index literals that the entity index "
                "already covers as entity-associated literals (literals indexed under their owning entity, "
                "searchable via that entity). Inspect the entity-index setup to see which entity-associated literals "
                "the entity index already covers in this knowledge graph and avoid duplicating them in the "
                "literals index."
            )
            rules.append(
                "If you cannot find a meaningful set of standalone literals to index, skip both the index "
                "SPARQL and the info SPARQL - leaving them unset is the right outcome and is preferred over "
                "a low-quality index."
            )

        return rules

    def _info_rules(self) -> list[str]:
        return [
            "If the user provides additional notes about the setup, make sure to follow them.",
            "Avoid mentions of specific details in the knowledge graph description, but "
            "focus on the bigger picture.",
        ]

    def function_definitions(self) -> list[dict]:
        kgs = [m.kg for m in self.managers]
        functions = [find_frequent_function_definition(kgs, self.config.list_k)]
        if self.input["phase"] == "index":
            functions.extend(index_functions(self.input["name"]))
        else:
            functions.extend(info_functions())
        return functions

    def call_function(
        self,
        fn_name: str,
        fn_args: dict,
        known: set[str],
        example_indices: dict | None,
    ) -> str:
        assert self.state is not None
        manager = self.managers[0]

        if fn_name == "find_frequent":
            return find_frequent(
                self.managers,
                fn_args["kg"],
                fn_args["position"],
                fn_args.get("subject"),
                fn_args.get("property"),
                fn_args.get("object"),
                fn_args.get("page", 1),
                self.config.list_k,
                known,
                self.config.sparql_request_timeout,
                self.config.sparql_read_timeout,
            )

        elif fn_name == "show_setup":
            return self.show_setup()

        elif fn_name == "show_entity_setup":
            # use a fresh empty `known` so the running self.known set
            # is not polluted by IRIs from the entity index sparql
            entity_state = load_index_state(manager, "entities", set())
            return format_index_state(entity_state, manager)

        elif fn_name == "set_query":
            return self.set_query(manager, **fn_args, known=known)

        elif fn_name == "add_prefix":
            return self.add_prefix(manager, fn_args["short"], fn_args["namespace"])

        elif fn_name == "delete_prefix":
            return self.delete_prefix(manager, fn_args["short"])

        elif fn_name == "update_prefix":
            return self.update_prefix(manager, fn_args["short"], fn_args["namespace"])

        elif fn_name == "set_description":
            return self.set_description(fn_args["description"])

        elif fn_name == "stop":
            return "Stopping"

        raise FunctionCallException(f"Unknown function {fn_name}")

    def set_description(self, description: str) -> str:
        assert isinstance(self.state, (IndexState, InfoState))
        self.state.description = description
        return "Description updated."

    def set_query(
        self,
        manager: KgManager,
        type: str,
        sparql: str | None,
        known: set[str],
    ) -> str:
        assert isinstance(self.state, IndexState)
        name = self.input["name"]

        # null clears / unsets the query
        if sparql is None:
            index = manager.try_get(name)
            if type == "info":
                self.state.info_sparql = None
                if index is not None:
                    index.info_sparql = None
            else:
                self.state.index_sparql = None
            return f"Cleared the {name} {type} SPARQL"

        # always check whether the query contains only known IRIs
        check_known(manager, sparql, known)

        validation_fn = (
            validate_index_sparql if type == "index" else validate_info_sparql
        )

        try:
            validation_fn(manager.sparql_parser, sparql)
            sparql = manager.fix_prefixes(sparql)
        except ValueError as e:
            raise FunctionCallException(
                f"Invalid {name} {type} SPARQL:\n{str(e)}"
            ) from e

        index = manager.try_get(name)
        if type == "info":
            self.state.info_sparql = sparql
            # write directly to manager if available
            if index is not None:
                index.info_sparql = sparql
        else:
            self.state.index_sparql = sparql

        msg = f"{name.capitalize()} {type} SPARQL updated"
        if type == "info" and index is not None:
            msg += " and used in subsequent function calls"
        return msg

    def apply_prefixes(self, manager: KgManager, prefixes: dict[str, str]) -> None:
        merged, _, kg_prefixes = merge_prefixes(
            get_common_sparql_prefixes(),
            prefixes,
            manager.logger,
            do_raise=True,
        )
        self.state.prefixes = kg_prefixes
        manager.kg_prefixes = kg_prefixes
        manager.prefixes = merged

    def show_setup(self) -> str:
        if self.input["phase"] == "index":
            return format_index_state(self.state, self.managers[0])
        else:
            return format_info_state(self.state)

    def add_prefix(self, manager: KgManager, short: str, namespace: str) -> str:
        assert isinstance(self.state, InfoState)
        if not self.state.prefixes:
            self.state.prefixes = {}
        elif short in self.state.prefixes:
            raise FunctionCallException(f"Prefix '{short}' already exists.")

        try:
            self.apply_prefixes(manager, {**self.state.prefixes, short: namespace})
        except RuntimeError as e:
            raise FunctionCallException(str(e)) from e

        return f"Prefix '{short}' added and available for subsequent function calls."

    def delete_prefix(self, manager: KgManager, short: str) -> str:
        assert isinstance(self.state, InfoState)
        if not self.state.prefixes:
            raise FunctionCallException("No prefixes to delete.")
        elif short not in self.state.prefixes:
            raise FunctionCallException(f"Prefix '{short}' does not exist.")

        try:
            self.apply_prefixes(
                manager,
                {k: v for k, v in self.state.prefixes.items() if k != short},
            )
        except RuntimeError as e:
            raise FunctionCallException(str(e)) from e

        return f"Prefix '{short}' deleted and no longer available for subsequent function calls."

    def update_prefix(self, manager: KgManager, short: str, namespace: str) -> str:
        assert isinstance(self.state, InfoState)
        if not self.state.prefixes:
            raise FunctionCallException("No prefixes to update.")
        elif short not in self.state.prefixes:
            raise FunctionCallException(f"Prefix '{short}' does not exist.")

        try:
            self.apply_prefixes(manager, {**self.state.prefixes, short: namespace})
        except RuntimeError as e:
            raise FunctionCallException(str(e)) from e

        return f"Prefix '{short}' updated and available for subsequent function calls."

    def done(self, fn_name: str) -> bool:
        return fn_name == "stop"

    def setup(self, input: dict) -> str:
        assert isinstance(input, dict), "Input for auto-setup must be a dict"
        self.input = input

        manager = self.managers[0]
        if self.input["phase"] == "index":
            self.state = load_index_state(manager, self.input["name"], self.known)
            if self.input.get("notes"):
                return input["notes"]
            else:
                return f'Set up the index and info SPARQLs for {self.input["name"]} \
of the "{manager.kg}" knowledge graph.'

        # info phase
        self.state = load_info_state(manager)
        if self.input.get("notes"):
            return input["notes"]
        else:
            return f'Set up the prefixes and description for the "{manager.kg}" knowledge graph.'

    def output(self, messages: list[Message]) -> dict:
        if self.input["phase"] == "index":
            assert isinstance(self.state, IndexState)
            return {
                "type": "output",
                "phase": "index",
                "name": self.input["name"],
                "sparql": {
                    "index": self.state.index_sparql,
                    "info": self.state.info_sparql,
                },
                "info": {
                    "description": self.state.description,
                },
            }
        else:
            assert isinstance(self.state, InfoState)
            manager = self.managers[0]

            # build kg info
            info = KgInfo()
            # unchanged fields via manager
            if manager.endpoint != get_qlever_endpoint(manager.kg):
                info.endpoint = manager.endpoint
            if manager.params:
                info.params = manager.params
            if manager.headers:
                info.headers = manager.headers

            # updated fields via state
            if self.state.description:
                info.description = self.state.description
            if self.state.prefixes:
                info.prefixes = self.state.prefixes

            return {
                "type": "output",
                "phase": "info",
                "info": info.model_dump(exclude_unset=True),
            }
