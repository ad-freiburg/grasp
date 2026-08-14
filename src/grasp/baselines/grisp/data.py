import argparse
import os
import random
import re
import string
from dataclasses import dataclass
from logging import Logger
from typing import Literal

import torch
from grammar_utils.parse import LR1Parser  # type: ignore
from pydantic import BaseModel
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerBase
from universal_ml_utils.io import dump_jsonl, load_jsonl
from universal_ml_utils.logging import get_logger, setup_logging
from universal_ml_utils.ops import partition_by

from grasp.baselines.grisp.utils import load_sparql_parser
from grasp.configs import KgConfig, KgInfo
from grasp.manager import KgManager, load_kg_manager
from grasp.sparql.item import Item, extract_sparql_items
from grasp.sparql.types import Alternative, ObjType, Position, Selection
from grasp.sparql.utils import (
    find_all,
    infer_position_from_prefix,
    query_type,
    replace_unresolved_placeholders,
)
from grasp.tasks.sparql_qa.examples import SparqlQaSample
from grasp.utils import format_list, get_available_knowledge_graphs

BOI = "<iri>"
EOI = "</iri>"

BOR = "<rep>"
EOR = "</rep>"

# binary labels for the query validation task (task 3). The valid label is
# placed first so target_dist = [f1, 1 - f1] reads as [P(valid), P(invalid)].
VALID_LABEL = "A"
INVALID_LABEL = "B"
VALIDATION_OPTIONS = [VALID_LABEL, INVALID_LABEL]

# Stable marker shown in place of a result preview when the query could not be
# executed within the time budget or the backend was unavailable. This is
# deliberately distinct from an *empty* result: an empty result is evidence the
# query is wrong, whereas an unavailable result is absence of evidence and must
# not be read as wrong. Used consistently at inference and in bootstrapped
# training data so the validator learns to judge from the query in this case.
RESULT_UNAVAILABLE = "Unavailable (timeout reached or backend down)"

# shown as the result for a partially resolved skeleton: it still has unfilled
# placeholders, so it could not be executed at all -- distinct from
# RESULT_UNAVAILABLE (a fully resolved query whose execution failed), so the
# improvement model is not told a wrong reason for the missing result.
RESULT_UNRESOLVED = "Unavailable (skeleton not fully resolved)"

ALT_LABELS = string.ascii_uppercase + string.digits

IGNORE_INDEX = -100

Messages = list[dict[str, str]]
AlternativeGroups = dict[ObjType, list[Alternative]]
OrderedAlternatives = list[tuple[Alternative, ObjType, str | None]]

# order in which a skeleton's natural-language placeholders are resolved. All
# orders reduce to a fixed permutation of placeholder indices computed once per
# skeleton (see Skeleton._compute_order); the resolution loop then fills them in
# that order and backtracks by popping the last-added selection, so the memo and
# backtracking logic in select_iris stays identical across orders.
FillOrder = Literal[
    "left-to-right",
    "right-to-left",
    "entities-then-properties",
    "properties-then-entities",
    "triple-wise-entities-then-properties",
    "random",
]


class IRI(BaseModel):
    identifier: str
    label: str
    aliases: list[str]

    @staticmethod
    def from_item(item: Item, manager: KgManager) -> "IRI":
        label = item.alternative.label or item.alternative.get_identifier()
        aliases = item.alternative.aliases or []

        if item.variant is not None:
            # add variant to label and aliases in brackets
            #  e.g., "population (wdt)"
            label = f"{label} ({item.variant})"
            aliases = [f"{alias} ({item.variant})" for alias in aliases]

        identifier = manager.denormalize(
            item.alternative.identifier,
            item.obj_type.index_name,
            item.variant,
        )
        assert identifier is not None, (
            f"Failed to denormalize identifier {item.alternative.identifier}"
        )
        identifier = manager.format_iri(identifier)

        return IRI(identifier=identifier, label=label, aliases=aliases)


class GRISPSample(BaseModel):
    kg: str
    questions: list[str]
    sparql: list[str | IRI]

    @property
    def has_placeholders(self) -> bool:
        return any(isinstance(part, IRI) for part in self.sparql)


class SelectionSample(BaseModel):
    messages: Messages
    options: list[str]
    target: str


class ValidationSample(BaseModel):
    # Query validation (task 3). Single-token classification over
    # VALIDATION_OPTIONS, trained against a *soft* target distribution
    # target_dist = [f1, 1 - f1], where f1 is the F1 score between the
    # predicted query result and the gold query result.
    messages: Messages
    options: list[str] = list(VALIDATION_OPTIONS)
    target_dist: list[float]


class GRISPMaterializedSample(BaseModel):
    skeletons: list[Messages] = []
    selections: list[SelectionSample] = []
    # bootstrapped tasks (filled by grisp.bootstrap, empty for gold data)
    validations: list[ValidationSample] = []
    improvements: list[Messages] = []

    @property
    def has_skeletons(self) -> bool:
        return len(self.skeletons) > 0

    @property
    def has_selections(self) -> bool:
        return len(self.selections) > 0

    @property
    def has_validations(self) -> bool:
        return len(self.validations) > 0

    @property
    def has_improvements(self) -> bool:
        return len(self.improvements) > 0


def extract_value_from_nl_iri(nl_iri: dict) -> str:
    return nl_iri["value"][len(BOI) : -len(EOI)].strip()


def extract_query_and_variant_from_nl_iri(nl_iri: dict) -> tuple[str, str | None]:
    query = extract_value_from_nl_iri(nl_iri)
    variant: str | None = None

    m = re.search(r" \(([^)]*)\)$", query)
    if m is not None:
        # remove variant in parentheses at the end
        query = query[: m.start()].strip()
        variant = m.group(1).strip()

    return query, variant


@dataclass
class Info:
    prefix: str
    sparql: str
    query: str
    variant: str | None
    value: str

    def build_queries(
        self,
        obj_types: dict[ObjType, bool],
    ) -> dict[ObjType, tuple[str, str | None]]:
        queries = {}
        for obj_type, supports_variants in obj_types.items():
            if supports_variants and self.variant is not None:
                queries[obj_type] = (self.query, self.variant)
            else:
                queries[obj_type] = (self.value, None)
        return queries


class Skeleton:
    @staticmethod
    def parse(
        sparql: str,
        parser: LR1Parser,
        fill_order: FillOrder = "left-to-right",
        order: list[int] | None = None,
    ) -> "Skeleton":
        sparql_parse = parser.parse(sparql)
        return Skeleton(sparql, sparql_parse, parser, fill_order, order)

    def __init__(
        self,
        sparql: str,
        sparql_parse: dict,
        parser: LR1Parser | None = None,
        fill_order: FillOrder = "left-to-right",
        order: list[int] | None = None,
    ) -> None:
        self.sparql_parse = sparql_parse
        self.sparql_encoded = sparql.encode()
        # placeholders in document (byte) order
        self.nl_iris = list(find_all(self.sparql_parse, "NL_IRI"))
        # selections/identifiers are stored in *fill* order (a stack), so
        # pop_selection() undoes the most recent selection for backtracking.
        # self.order maps fill step k -> placeholder (document) index, i.e. the
        # k-th selection fills self.nl_iris[self.order[k]].
        self.selections: list[Selection] = []
        self.identifiers: list[str] = []
        self.order = self.compute_order(parser, fill_order, order)

    def compute_order(
        self,
        parser: LR1Parser | None,
        fill_order: FillOrder,
        order: list[int] | None,
    ) -> list[int]:
        n = len(self.nl_iris)
        if order is not None:
            assert sorted(order) == list(range(n)), (
                "explicit order must be a permutation of the placeholder indices"
            )
            return list(order)

        if fill_order == "left-to-right":
            return list(range(n))
        elif fill_order == "right-to-left":
            return list(reversed(range(n)))
        elif fill_order == "random":
            # seed from the skeleton text so the permutation is stable per
            # skeleton (reproducible across runs and consistent across the
            # backtracking within a single resolution)
            rng = random.Random(self.sparql_encoded)
            perm = list(range(n))
            rng.shuffle(perm)
            return perm
        elif fill_order == "entities-then-properties":
            assert parser is not None, (
                "parser is required for the entities-then-properties fill order"
            )
            positions = [self.infer_position(i, parser) for i in range(n)]
            # entities first, then properties, each group in ltr document order;
            # anything not clearly a property is treated as an entity
            entities, properties = partition_by(
                range(len(positions)),
                lambda i: positions[i] != Position.PROPERTY,
            )
            return entities + properties
        elif fill_order == "properties-then-entities":
            assert parser is not None, (
                "parser is required for the properties-then-entities fill order"
            )
            positions = [self.infer_position(i, parser) for i in range(n)]
            # properties first, then entities, each group in ltr document order;
            # anything not clearly a property is treated as an entity
            properties, entities = partition_by(
                range(len(positions)),
                lambda i: positions[i] == Position.PROPERTY,
            )
            return properties + entities
        elif fill_order == "triple-wise-entities-then-properties":
            assert parser is not None, (
                "parser is required for the "
                "triple-wise-entities-then-properties fill order"
            )
            positions = [self.infer_position(i, parser) for i in range(n)]
            # process triples in document order; within each triple resolve its
            # entities first (document order) then its properties, so every
            # property is filled only after both of its endpoints in that triple
            triple_order: list[int] = []
            for group in self.triple_groups():
                entities, properties = partition_by(
                    group,
                    lambda i: positions[i] != Position.PROPERTY,
                )
                triple_order.extend(entities + properties)
            return triple_order

        # not reachable but still kept for completeness
        raise ValueError(f"Unknown fill order: {fill_order}")

    def triple_groups(self) -> list[list[int]]:
        # group placeholder indices by the innermost triple
        # (TriplesSameSubjectPath) they belong to, with the groups ordered by
        # document position. Placeholders that are not part of any triple (e.g.
        # inside VALUES/FILTER) form a final group so the result stays a full
        # permutation. Grouping is done via NL_IRI descendants of each triple
        # block, since the block nodes themselves carry no byte span.
        n = len(self.nl_iris)
        span_to_idx = {tuple(self.nl_iris[i]["byte_span"]): i for i in range(n)}

        blocks = list(find_all(self.sparql_parse, "TriplesSameSubjectPath"))
        block_members: list[set[int]] = []
        for block in blocks:
            members = set()
            for nl_iri in find_all(block, "NL_IRI"):
                idx = span_to_idx.get(tuple(nl_iri["byte_span"]))
                if idx is not None:
                    members.add(idx)
            block_members.append(members)

        # assign each placeholder to the innermost (smallest) block containing it
        assigned: dict[int, int] = {}
        for idx in range(n):
            containing = [
                bi for bi, members in enumerate(block_members) if idx in members
            ]
            if containing:
                assigned[idx] = min(containing, key=lambda bi: len(block_members[bi]))

        groups: dict[int, list[int]] = {}
        orphans: list[int] = []
        for idx in range(n):  # ascending -> document order within each group
            if idx in assigned:
                groups.setdefault(assigned[idx], []).append(idx)
            else:
                orphans.append(idx)

        # order the triple groups by the document position of their first member
        result = [groups[bi] for bi in sorted(groups, key=lambda bi: groups[bi][0])]
        if orphans:
            result.append(orphans)
        return result

    def infer_position(self, idx: int, parser: LR1Parser) -> Position:
        # structural position (subject/property/object) of placeholder idx,
        # inferred from the raw skeleton truncated right before it. Independent
        # of whether earlier placeholders are resolved, so it is stable to use
        # for ordering before any selection has been made.
        byte_start, _ = self.nl_iris[idx]["byte_span"]
        prefix = self.sparql_encoded[:byte_start].decode()
        try:
            return infer_position_from_prefix(prefix, parser)
        except Exception:
            # fall back to treating it as an entity
            return Position.SUBJECT

    @property
    def nl_sparql(self) -> str:
        return self.sparql_encoded.decode()

    @property
    def replaced(self) -> int:
        return len(self.selections)

    @property
    def total(self) -> int:
        return len(self.nl_iris)

    @property
    def done(self) -> bool:
        return len(self.selections) >= len(self.nl_iris)

    def get_filled_placeholders(self) -> dict[int, str]:
        # map placeholder (document) index -> selected identifier for the
        # selections made so far
        return {
            self.order[k]: self.identifiers[k] for k in range(len(self.identifiers))
        }

    def render(self, require_done: bool) -> str:
        if require_done:
            assert self.done, "Not all NL IRIs have been replaced"

        filled = self.get_filled_placeholders()
        sparql = ""
        start = 0
        for j, nl_iri in enumerate(self.nl_iris):
            byte_start, byte_end = nl_iri["byte_span"]
            sparql += self.sparql_encoded[start:byte_start].decode()
            # resolved placeholders are substituted; not-yet-resolved ones are
            # left as their natural-language token (only possible in the partial
            # case, since require_done guarantees all are filled otherwise)
            sparql += filled.get(j, str(nl_iri["value"]))
            start = byte_end

        sparql += self.sparql_encoded[start:].decode()
        return sparql

    def materialize(self) -> str:
        return self.render(require_done=True)

    def materialize_partial(self) -> str:
        # render the skeleton with the placeholders resolved so far replaced by
        # their identifiers; any not-yet-resolved placeholders are left as
        # natural language. Equivalent to materialize() once the skeleton is done.
        return self.render(require_done=False)

    def prepare_for_selection(self) -> Info:
        assert not self.done, "All NL IRIs have already been replaced"
        # next placeholder to fill, per the fill order
        idx = self.order[len(self.selections)]
        filled = self.get_filled_placeholders()

        cur = self.nl_iris[idx]
        cur_start, cur_end = cur["byte_span"]

        # prefix: everything before the current placeholder, with all already
        # resolved placeholders substituted (under non-left-to-right orders a
        # resolved placeholder may sit either side of the current one) and any
        # not-yet-resolved placeholder left as natural language
        prefix = ""
        start = 0
        for i, nl_iri in enumerate(self.nl_iris):
            byte_start, byte_end = nl_iri["byte_span"]
            if byte_start >= cur_start:
                break
            prefix += self.sparql_encoded[start:byte_start].decode()
            prefix += filled.get(i, str(nl_iri["value"]))
            start = byte_end
        prefix += self.sparql_encoded[start:cur_start].decode()

        query, variant = extract_query_and_variant_from_nl_iri(cur)
        value = extract_value_from_nl_iri(cur)

        # tail: everything after the current placeholder, again with resolved
        # placeholders substituted and unresolved ones left as natural language
        tail = ""
        start = cur_end
        for i, nl_iri in enumerate(self.nl_iris):
            byte_start, byte_end = nl_iri["byte_span"]
            if byte_start <= cur_start:
                continue
            tail += self.sparql_encoded[start:byte_start].decode()
            tail += filled.get(i, str(nl_iri["value"]))
            start = byte_end
        tail += self.sparql_encoded[start:].decode()

        sparql = prefix + f"{BOR}{value}{EOR}" + tail
        return Info(
            prefix=prefix,
            sparql=sparql,
            query=query,
            variant=variant,
            value=value,
        )

    def add_selection(self, selection: Selection, manager: KgManager) -> None:
        assert not self.done, "All NL IRIs have already been replaced"
        identifier = manager.denormalize(
            selection.alternative.identifier,
            selection.obj_type.index_name,
            selection.variant,
        )
        assert identifier is not None, "Failed to denormalize identifier"
        identifier = manager.format_iri(identifier)
        label = selection.alternative.get_label()
        if label is not None:
            label += f" ({identifier})"
        else:
            label = identifier

        self.selections.append(selection)
        self.identifiers.append(identifier)

    def pop_selection(self) -> Selection:
        assert len(self.selections) > 0, "No selections to pop"
        selection = self.selections.pop()
        self.identifiers.pop()
        return selection


def get_skeleton_prompt(
    kg: str,
    question: str,
    sparql: str | None = None,
) -> list[dict]:
    system = f"""\
You are an expert SPARQL query generator. \
Your task is to generate SPARQL query skeletons over \
the {kg} knowledge graph for answering user questions.
Instead of actual IRIs, you should generate natural language \
placeholders surrounded by {BOI} and {EOI} tags. \
The placeholders may contain optional additional information \
helpful for disambiguation in brackets, e.g., "population (wdt)" \
for wikidata properties."""

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]

    if sparql is not None:
        messages.append({"role": "assistant", "content": sparql})

    return messages


def ordered_alternatives(
    alternative_groups: AlternativeGroups,
    queries: dict[ObjType, tuple[str, str | None]],
) -> OrderedAlternatives:
    return ordered_alternatives_with_interleave(
        alternative_groups,
        queries,
        interleave=False,
    )


def ordered_alternatives_with_interleave(
    alternative_groups: AlternativeGroups,
    queries: dict[ObjType, tuple[str, str | None]],
    interleave: bool = False,
) -> OrderedAlternatives:
    variants = {obj_type: variant for obj_type, (_, variant) in queries.items()}

    entities = [
        (alternative, ObjType.ENTITY, variants.get(ObjType.ENTITY))
        for alternative in alternative_groups.get(ObjType.ENTITY, [])
    ]
    properties = [
        (alternative, ObjType.PROPERTY, variants.get(ObjType.PROPERTY))
        for alternative in alternative_groups.get(ObjType.PROPERTY, [])
    ]

    if not interleave or not entities or not properties:
        return entities + properties

    ordered = []
    max_len = max(len(entities), len(properties))
    for i in range(max_len):
        if i < len(entities):
            ordered.append(entities[i])
        if i < len(properties):
            ordered.append(properties[i])
    return ordered


def count_alternatives(alternatives: OrderedAlternatives) -> int:
    return len(alternatives)


def format_alternatives(alternatives: OrderedAlternatives) -> str:
    if len(alternatives) == 0:
        return "No alternatives found"

    assert len(alternatives) < len(ALT_LABELS) - 1, (
        f"Number of alternatives must be less than {len(ALT_LABELS) - 1}"
    )

    grouped = {}

    for label, (alternative, obj_type, variant) in zip(ALT_LABELS, alternatives):
        alt = alternative.get_selection_string(
            show_matched_label=False,
            include_variants=[],
        )
        if obj_type not in grouped:
            grouped[obj_type] = []
        grouped[obj_type].append(f"{label}. {alt}")

    multiple_types = len(grouped) > 1
    alt_groups = []
    for obj_type, alts in grouped.items():
        if multiple_types:
            alt_group = f"{obj_type.value.capitalize()} alternatives:\n"
        else:
            alt_group = "Alternatives:\n"
        alt_group += "\n".join(alts)
        alt_groups.append(alt_group)

    # always add none alternative
    none_lab = ALT_LABELS[len(alternatives)]
    alt_groups.append(f"{none_lab}. None of the above")
    top_k_string = "\n\n".join(alt_groups)

    return top_k_string


def find_alternative_groups(
    manager: KgManager,
    sparql: str,
    queries: dict[ObjType, tuple[str, str | None]],
    top_k: int,
    logger: Logger,
    skip_constraint: bool = False,
    max_candidates: int | None = None,
) -> AlternativeGroups:
    # prefix left of the current placeholder, with any unresolved placeholders
    # turned into variables so it parses
    prefix = replace_unresolved_placeholders(sparql.split(BOR, 1)[0])
    sparql_query_type = query_type(prefix, manager.sparql_parser, is_prefix=True)
    constraint_sparql = None

    try:
        if not skip_constraint and sparql_query_type == "select":
            logger.debug(f"Deriving constraint query from SPARQL:\n{sparql}")
            constraint_sparql, position = manager.derive_constraint_query_from_sparql(
                sparql,
                limit=None if max_candidates is None else max_candidates + 1,
            )
            logger.debug(f"Derived constraint SPARQL:\n{constraint_sparql}")
        else:
            position = infer_position_from_prefix(prefix, manager.sparql_parser)

        logger.debug(
            f"Determined query type and position: "
            f"'{sparql_query_type}', '{position.value}'"
        )
    except Exception as e:
        logger.debug(f"Error analyzing SPARQL:\n{e}")
        # full search across both indices as fallback
        position = None

    if position is None:
        obj_types = [ObjType.ENTITY, ObjType.PROPERTY]
    elif position == Position.PROPERTY:
        obj_types = [ObjType.PROPERTY]
    else:
        obj_types = [ObjType.ENTITY]

    identifier_maps: dict[ObjType, dict[str, list[str]] | None] = {
        obj_type: None for obj_type in obj_types
    }
    if not skip_constraint and position is not None and constraint_sparql is not None:
        # can only be one
        obj_type = obj_types[0]
        try:
            logger.debug(
                f"Searching for candidate IRIs at position {position.value} "
                f"with constraint SPARQL:\n{constraint_sparql}"
            )
            identifier_map = manager.get_candidate_ids(
                obj_type.index_name,
                constraint_sparql,
                max_candidates,
                # 6 seconds to execute query, 3 to read result
                request_timeout=(3.5, 6.0),
                read_timeout=3.0,
            )
            logger.debug(
                f"Got {len(identifier_map)} candidate IRIs for position {position.value}"
            )
            identifier_maps[obj_type] = identifier_map
        except Exception as e:
            logger.warning(f"Error getting candidate IRIs: {e}")
    else:
        logger.debug(
            "Skipping constraint-based IRI filtering, full search will be performed"
        )

    alternative_groups = {}
    for obj_type in obj_types:
        assert obj_type in queries, f"Missing query for object type {obj_type}"
        query, _ = queries[obj_type]
        logger.debug(f"Searching with query '{query}' in '{obj_type.index_name}' index")
        alternatives = manager.search_index(
            obj_type.index_name,
            query,
            top_k,
            identifier_maps[obj_type],
        )
        if alternatives:
            alternative_groups[obj_type] = alternatives

    logger.debug(
        "Found "
        f"{count_alternatives(ordered_alternatives(alternative_groups, queries))} alternatives:\n"
        + format_alternatives(ordered_alternatives(alternative_groups, queries))
    )
    return alternative_groups


def get_selection_prompt_and_options(
    manager: KgManager,
    question: str,
    sparql: str,
    selections: list[Selection],
    alternatives: OrderedAlternatives,
) -> tuple[list[dict], list[str]]:
    system = f"""\
You are a SPARQL expert. Your task is to select the best fitting \
{manager.kg} item for replacing a natural-language placeholder \
in a SPARQL skeleton. The placeholder to be replaced is marked \
{BOR}...{EOR} in the skeleton. There may also be other unresolved \
placeholders, marked {BOI}...{EOI}.

You are given the user question, the SPARQL skeleton, \
info about already resolved placeholders, \
and a list of alternatives for the current placeholder. \
You should output the letter corresponding to the best \
fitting alternative."""

    messages = [
        {"role": "system", "content": system},
    ]

    user = f"Question:\n{question}\n\nSPARQL skeleton:\n{sparql}"
    if selections:
        user += f"\n\n{manager.format_selections(selections)}"

    user += f"\n\n{format_alternatives(alternatives)}"
    messages.append({"role": "user", "content": user})

    options = ALT_LABELS[: count_alternatives(alternatives) + 1]
    return messages, list(options)


def get_validation_prompt(
    kg: str,
    question: str,
    sparql: str,
    selections: str | None = None,
    result: str | None = None,
    valid: bool | None = None,
) -> Messages:
    system = f"""\
You are a SPARQL expert. Your task is to decide whether the given SPARQL \
query over the {kg} knowledge graph correctly and completely answers the \
user question.
You are given the user question, the SPARQL query, info about the items \
used in it, and a preview of its result. Output {VALID_LABEL} if the query \
correctly answers the question, or {INVALID_LABEL} if it does not."""

    user = f"Question:\n{question}\n\nSPARQL query:\n{sparql}"
    if selections:
        user += f"\n\n{selections}"
    if result is not None:
        user += f"\n\nResult:\n{result}"

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    if valid is not None:
        messages.append(
            {"role": "assistant", "content": VALID_LABEL if valid else INVALID_LABEL}
        )

    return messages


def get_improvement_prompt(
    kg: str,
    question: str,
    skeleton: str,
    sparql: str | None = None,
    selections: str | None = None,
    result: str | None = None,
    improved: str | None = None,
) -> Messages:
    # Neutral framing on purpose: the candidate is *not* asserted to be wrong.
    # The model is asked to improve the candidate only if there is something to
    # improve, so that a false-negative validation verdict does not pressure the
    # model into needlessly rewriting a good skeleton. The same evidence the
    # validator saw (the resolved query, its chosen items, and a result preview)
    # is provided as a hint for deciding whether a rewrite is warranted. The
    # wording covers both a fully resolved query and a partially resolved one
    # (where some placeholders could not be filled and the result is missing).
    system = f"""\
You are an expert SPARQL query generator. Given a user question and a \
candidate SPARQL query skeleton over the {kg} knowledge graph, try to improve \
the skeleton so that it better answers the question. If there is nothing to \
improve, keep it as is.
As hints, you are also given the query resolved from the skeleton, info about \
the items chosen to fill its placeholders, and a preview of that query's \
result. The resolved query may be incomplete: placeholders that could not be \
resolved are left as natural language, and the result may be missing.
Like the candidate, use natural language placeholders surrounded by {BOI} \
and {EOI} tags instead of actual IRIs. The placeholders may contain optional \
additional information helpful for disambiguation in brackets, e.g., \
"population (wdt)" for wikidata properties."""

    user = f"Question:\n{question}\n\nCandidate skeleton:\n{skeleton}"
    if sparql is not None:
        user += f"\n\nResolved query:\n{sparql}"
    if selections:
        user += f"\n\n{selections}"
    if result is not None:
        user += f"\n\nResult:\n{result}"

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    if improved is not None:
        messages.append({"role": "assistant", "content": improved})

    return messages


class OracleSkeletonUnavailable(Exception):
    pass


def gold_sparql_to_nl_skeleton(
    sparql: str,
    manager: KgManager,
    is_val: bool = True,
    p: float = 0.2,
) -> str:
    # match the training data pipeline (see preparation in main below):
    # fix/strip known prefixes and prettify before extracting items, so the
    # resulting NL skeleton lies on the training distribution.
    # is_val=True (default) -> deterministic canonical labels (used for oracle
    # inference). is_val=False with p -> alias sampling like the skeleton task,
    # for use as a training target.
    sparql = manager.fix_prefixes(sparql, remove_known=True)
    sparql = manager.prettify(sparql)
    sparql, items = extract_sparql_items(sparql, manager)

    # same validity rules as training data prep (main below):
    # unindexed items must have a label, and unknown items are not allowed
    invalid = [
        item
        for item in items
        if (item.is_unindexed and not item.has_label) or item.is_unknown
    ]
    if invalid:
        raise OracleSkeletonUnavailable(
            "invalid items: "
            + ", ".join(
                f"{it.alternative.get_identifier()} ({it.obj_type.name})"
                for it in invalid
            )
        )

    parts: list[str | IRI] = []
    cursor = 0
    for item in items:
        # literals/common/unknown are emitted as-is; everything else becomes a placeholder
        if item.is_literal or item.is_common or item.is_unknown:
            continue

        start, end = item.item_span
        parts.append(sparql[cursor:start])
        parts.append(IRI.from_item(item, manager))
        cursor = end

    parts.append(sparql[cursor:])

    return materialize_skeleton(parts, is_val=is_val, p=p)


def materialize_skeleton(
    parts: list[str | IRI],
    is_val: bool = False,
    p: float = 0.2,
) -> str:
    formatted_parts = []
    for part in parts:
        if isinstance(part, str):
            formatted_parts.append(part)
            continue

        # choose main label 80% of the time,
        # otherwise choose a random alias if available
        if not is_val and part.aliases and random.random() < p:
            iri = random.choice(part.aliases)
        else:
            iri = part.label

        formatted_parts.append(f"{BOI}{iri}{EOI}")

    return "".join(formatted_parts)


def materialize_sparql(parts: list[str | IRI]) -> str:
    formatted_parts = []
    for part in parts:
        if isinstance(part, str):
            formatted_parts.append(part)
            continue

        formatted_parts.append(part.identifier)

    return "".join(formatted_parts)


def materialize_sample(
    sample: GRISPSample,
    is_val: bool = False,
    p: float = 0.2,
) -> tuple[str, str]:
    if is_val:
        question = sample.questions[0]
    else:
        question = random.choice(sample.questions)

    return question, materialize_skeleton(sample.sparql, is_val, p)


def tokenize_messages(
    messages: list[dict[str, str]],
    tokenizer: PreTrainedTokenizerBase,
    mask_inputs: bool,
) -> dict:
    if not mask_inputs:
        enc: dict = tokenizer.apply_chat_template(
            messages,
            return_dict=True,
            enable_thinking=False,
        )  # type: ignore
        enc["labels"] = enc["input_ids"]  # type: ignore
        return enc  # type: ignore

    enc = tokenizer.apply_chat_template(
        messages,
        return_assistant_tokens_mask=True,
        return_dict=True,
        enable_thinking=False,
    )  # type: ignore

    mask = enc["assistant_masks"]
    chat_temp = tokenizer.chat_template
    assert chat_temp is not None, "Expected chat template to be set in tokenizer"
    if "{% generation %}" not in chat_temp or all(m == 0 for m in mask):
        # invalid assitant tokens mask, fallback to computing labels
        # manually
        prompt_ids = tokenizer.apply_chat_template(
            messages[:-1],
            add_generation_prompt=True,
            return_dict=True,
            enable_thinking=False,
        )
        prompt_len = len(prompt_ids)
        non_prompt_ids = enc["input_ids"][prompt_len:]
        labels = [IGNORE_INDEX] * prompt_len + non_prompt_ids
    else:
        labels = [
            id if mask == 1 else IGNORE_INDEX
            for id, mask in zip(enc["input_ids"], mask, strict=True)
        ]

    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": labels,
    }


def load_samples(
    file_paths: list[str],
    materialized: bool = False,
) -> list[GRISPSample] | list[GRISPMaterializedSample]:
    samples = []
    for path in file_paths:
        loaded_samples = load_jsonl(path)
        samples.extend(
            (
                GRISPSample(**sample)
                if not materialized
                else GRISPMaterializedSample(**sample)
                for sample in loaded_samples
            )
        )
    return samples


def tokenize_and_log(
    messages: Messages,
    tokenizer: PreTrainedTokenizerBase,
    mask_inputs: bool,
    logger: Logger,
) -> dict[str, torch.Tensor]:
    output = tokenize_messages(messages, tokenizer, mask_inputs)
    logger.debug(f"Sample:\n{tokenizer.decode(output['input_ids'])}")
    logger.debug(f"Length: {len(output['input_ids']):,}")
    label_ids = [label for label in output["labels"] if label != IGNORE_INDEX]
    logger.debug(
        f"Last 10 Input IDS: {output['input_ids'][-10 - len(label_ids) : -len(label_ids)]}"
    )
    logger.debug(f"Target IDs: {label_ids}")
    target = tokenizer.decode(label_ids)
    logger.debug(f"Target:\n{target}")
    return output


def tokenize_option_answer(
    messages: Messages,
    options: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict:
    # Shared tokenization for single-token classification tasks (selection and
    # validation). The assistant turn (messages[-1]) holds the located option
    # letter; the restricted-CE / soft-CE loss is applied at its position.
    enc: dict = tokenizer.apply_chat_template(
        messages,
        return_dict=True,
        enable_thinking=False,
    )  # type: ignore
    prompt_enc: dict = tokenizer.apply_chat_template(
        messages[:-1],
        add_generation_prompt=True,
        return_dict=True,
        enable_thinking=False,
    )  # type: ignore
    option_token_ids = [tokenizer.convert_tokens_to_ids(o) for o in options]
    assert all(
        t is not None and t != tokenizer.unk_token_id for t in option_token_ids
    ), f"Option letters not single tokens: {options}"

    located = messages[-1]["content"]
    assert located in options, (
        f"Assistant content '{located}' is not one of the options {options}"
    )
    located_id = option_token_ids[options.index(located)]

    # Some chat templates emit extra tokens between the generation prompt and
    # the assistant content (e.g. Qwen3 inserts <think>\n\n</think>\n\n when
    # enable_thinking=False). Locate the answer letter by searching forward
    # from the prompt boundary for the first occurrence of the located token id.
    input_ids = enc["input_ids"]
    search_start = len(prompt_enc["input_ids"])
    answer_pos = next(
        (i for i in range(search_start, len(input_ids)) if input_ids[i] == located_id),
        None,
    )
    assert answer_pos is not None, (
        f"Could not locate token id {located_id} for option "
        f"'{located}' in assistant turn (search from index {search_start})"
    )

    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        # Restricted CE replaces NTP on the answer position; EOS loss dropped.
        "labels": [IGNORE_INDEX] * len(enc["input_ids"]),
        "answer_pos": answer_pos,
        "option_token_ids": option_token_ids,
    }


def tokenize_selection(
    sample: SelectionSample,
    tokenizer: PreTrainedTokenizerBase,
) -> dict:
    output = tokenize_option_answer(sample.messages, sample.options, tokenizer)
    target_idx = sample.options.index(sample.target)
    # hard one-hot target distribution (special case of the soft validation loss)
    target_dist = [0.0] * len(sample.options)
    target_dist[target_idx] = 1.0
    output["target_idx"] = target_idx
    output["target_dist"] = target_dist
    return output


def tokenize_validation(
    sample: ValidationSample,
    tokenizer: PreTrainedTokenizerBase,
) -> dict:
    assert len(sample.target_dist) == len(sample.options), (
        f"target_dist length {len(sample.target_dist)} != "
        f"number of options {len(sample.options)}"
    )
    output = tokenize_option_answer(sample.messages, sample.options, tokenizer)
    # argmax option is the one written into the assistant turn (located above)
    target_idx = max(
        range(len(sample.target_dist)),
        key=lambda i: sample.target_dist[i],
    )
    output["target_idx"] = target_idx
    output["target_dist"] = list(sample.target_dist)
    return output


def tokenize_selection_and_log(
    sample: SelectionSample,
    tokenizer: PreTrainedTokenizerBase,
    logger: Logger,
) -> dict:
    output = tokenize_selection(sample, tokenizer)
    logger.debug(f"Selection sample:\n{tokenizer.decode(output['input_ids'])}")
    logger.debug(f"Length: {len(output['input_ids']):,}")
    logger.debug(
        f"Answer pos: {output['answer_pos']}, target idx: {output['target_idx']}, "
        f"target: '{sample.target}'"
    )
    return output


def tokenize_validation_and_log(
    sample: ValidationSample,
    tokenizer: PreTrainedTokenizerBase,
    logger: Logger,
) -> dict:
    output = tokenize_validation(sample, tokenizer)
    logger.debug(f"Validation sample:\n{tokenizer.decode(output['input_ids'])}")
    logger.debug(f"Length: {len(output['input_ids']):,}")
    logger.debug(
        f"Answer pos: {output['answer_pos']}, target dist: {sample.target_dist}"
    )
    return output


class GRISPMaterializedSkeletonDataset(Dataset):
    def __init__(
        self,
        samples: list[GRISPMaterializedSample],
        tokenizer: PreTrainedTokenizerBase,
        mask_inputs: bool = True,
        log_level: str | None = None,
    ) -> None:
        self.samples = [sample for sample in samples if sample.has_skeletons]
        self.tokenizer = tokenizer
        self.mask_inputs = mask_inputs

        self.logger = get_logger(
            "GRISP MATERIALIZED SKELETON DATASET",
            log_level,
        )

        self.counter = [0] * len(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]

        count = self.counter[idx]
        messages = sample.skeletons[count % len(sample.skeletons)]
        self.counter[idx] += 1
        self.logger.debug(
            f"({type(self).__name__}) Accessing sample {idx} count {count}"
        )

        return tokenize_and_log(
            messages,
            self.tokenizer,
            self.mask_inputs,
            self.logger,
        )


def prepare_skeleton(
    sample: GRISPSample,
    is_val: bool = False,
    p: float = 0.2,
) -> Messages:
    question, skeleton = materialize_sample(sample, is_val, p)
    return get_skeleton_prompt(sample.kg, question, skeleton)


class GRISPSkeletonDataset(Dataset):
    def __init__(
        self,
        samples: list[GRISPSample],
        tokenizer: PreTrainedTokenizerBase,
        mask_inputs: bool = True,
        is_val: bool = False,
        p: float = 0.2,
        log_level: str | None = None,
    ) -> None:
        self.samples = samples
        self.tokenizer = tokenizer
        self.mask_inputs = mask_inputs
        self.is_val = is_val
        self.p = p

        self.logger = get_logger(f"GRISP SKELETON DATASET ({is_val=})", log_level)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]

        messages = prepare_skeleton(sample, self.is_val, self.p)

        return tokenize_and_log(
            messages,
            self.tokenizer,
            self.mask_inputs,
            self.logger,
        )


class GRISPMaterializedSelectionDataset(Dataset):
    def __init__(
        self,
        samples: list[GRISPMaterializedSample],
        tokenizer: PreTrainedTokenizerBase,
        mask_inputs: bool = True,
        log_level: str | None = None,
    ) -> None:
        self.samples = [sample for sample in samples if sample.has_selections]
        self.tokenizer = tokenizer
        self.mask_inputs = mask_inputs

        self.logger = get_logger("GRISP MATERIALIZED SELECTION DATASET", log_level)

        self.counter = [0] * len(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        count = self.counter[idx]
        selection = sample.selections[count % len(sample.selections)]
        self.counter[idx] += 1
        self.logger.debug(
            f"({type(self).__name__}) Accessing sample {idx} count {count}"
        )

        return tokenize_selection_and_log(
            selection,
            self.tokenizer,
            self.logger,
        )


class GRISPMaterializedValidationDataset(Dataset):
    def __init__(
        self,
        samples: list[GRISPMaterializedSample],
        tokenizer: PreTrainedTokenizerBase,
        mask_inputs: bool = True,
        log_level: str | None = None,
    ) -> None:
        self.samples = [sample for sample in samples if sample.has_validations]
        self.tokenizer = tokenizer
        self.mask_inputs = mask_inputs

        self.logger = get_logger("GRISP MATERIALIZED VALIDATION DATASET", log_level)

        self.counter = [0] * len(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        count = self.counter[idx]
        validation = sample.validations[count % len(sample.validations)]
        self.counter[idx] += 1
        self.logger.debug(
            f"({type(self).__name__}) Accessing sample {idx} count {count}"
        )

        return tokenize_validation_and_log(
            validation,
            self.tokenizer,
            self.logger,
        )


class GRISPMaterializedImprovementDataset(Dataset):
    def __init__(
        self,
        samples: list[GRISPMaterializedSample],
        tokenizer: PreTrainedTokenizerBase,
        mask_inputs: bool = True,
        log_level: str | None = None,
    ) -> None:
        self.samples = [sample for sample in samples if sample.has_improvements]
        self.tokenizer = tokenizer
        self.mask_inputs = mask_inputs

        self.logger = get_logger("GRISP MATERIALIZED IMPROVEMENT DATASET", log_level)

        self.counter = [0] * len(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]

        count = self.counter[idx]
        messages = sample.improvements[count % len(sample.improvements)]
        self.counter[idx] += 1
        self.logger.debug(
            f"({type(self).__name__}) Accessing sample {idx} count {count}"
        )

        return tokenize_and_log(
            messages,
            self.tokenizer,
            self.mask_inputs,
            self.logger,
        )


def prepare_selection(
    sample: GRISPSample,
    manager: KgManager,
    is_val: bool = False,
    skeleton_p: float = 0.2,
    drop_infos_p: float = 0.05,
    drop_target_p: float = 0.1,
    shuffle_alts_p: float = 0.1,
    constrain_p: float = 0.1,
) -> tuple[Messages, list[str], str]:
    question, skeleton = materialize_sample(sample, is_val, skeleton_p)
    sparql = materialize_sparql(sample.sparql)

    supports_variants = {
        obj_type: manager.get_normalizer(obj_type.index_name).supports_variants
        for obj_type in [ObjType.ENTITY, ObjType.PROPERTY]
    }

    _, items = extract_sparql_items(sparql, manager)
    items = [item for item in items if item.is_entity_or_property]
    assert len(items) > 0, "No valid item to replace found in sample"

    parser = load_sparql_parser()
    # train the model to be fill-order agnostic: resolve a uniformly random
    # subset of the placeholders, in random order, and have it select one of the
    # remaining ones. A fixed random permutation makes every inference-time fill
    # order (left-to-right, right-to-left, entities-then-properties, random) an
    # in-distribution special case, so a single model supports all of them.
    order = list(range(len(items)))
    random.shuffle(order)
    skeleton = Skeleton.parse(skeleton, parser, order=order)

    upper = random.randint(0, len(items) - 1)
    for j in range(upper):
        skeleton.add_selection(items[order[j]].selection, manager)

    item = items[order[upper]]
    target_alt = item.selection.alternative

    info = skeleton.prepare_for_selection()
    queries = info.build_queries(supports_variants)

    # TODO: fix hardcoded k values
    k = 10 if is_val else random.randint(2, 10)

    # deriving and executing a constraint query per selection is expensive, so
    # only constrain a fraction of the training samples, just enough to keep
    # endpoint-filtered candidate sets in distribution. Validation always
    # constrains because alternatives are constrained at inference (see run.py).
    constrain = is_val or random.random() < constrain_p

    if item.is_entity_or_property:
        selection_logger = get_logger("GRISP SELECTION PREP")
        alternative_groups = find_alternative_groups(
            manager,
            info.sparql,
            queries,
            k,
            logger=selection_logger,
            skip_constraint=not constrain,
        )
    else:
        alternative_groups = {}

    drop_infos = not is_val and random.random() < drop_infos_p
    drop_target = not is_val and random.random() < drop_target_p
    shuffle_alts = not is_val and random.random() < shuffle_alts_p

    if shuffle_alts:
        # shuffle within each obj-type bucket while preserving grouped layout
        for current in alternative_groups.values():
            random.shuffle(current)

    alternatives = ordered_alternatives(alternative_groups, queries)

    if drop_infos:
        for alt, _, _ in alternatives:
            if alt.info:
                alt.info.clear()

    target_option: int | None = None
    for i, (alt, obj_type, _) in enumerate(alternatives):
        if alt != target_alt or obj_type != item.obj_type:
            continue

        # drop target alternative 20% of the time during training
        if drop_target:
            alternatives.pop(i)
        else:
            target_option = i

        break

    prompt, options = get_selection_prompt_and_options(
        manager,
        question,
        info.sparql,
        skeleton.selections,
        alternatives,
    )

    # if target option is None, we need to select the last
    # option, which is the "None of the above" option
    option = options[-1] if target_option is None else options[target_option]
    prompt.append({"role": "assistant", "content": option})
    return prompt, options, option


class GRISPSelectionDataset(Dataset):
    def __init__(
        self,
        samples: list[GRISPSample],
        manager: KgManager,
        tokenizer: PreTrainedTokenizerBase,
        mask_inputs: bool = True,
        is_val: bool = False,
        skeleton_p: float = 0.2,
        drop_infos_p: float = 0.05,
        drop_target_p: float = 0.1,
        shuffle_alts_p: float = 0.1,
        constrain_p: float = 0.1,
        log_level: str | None = None,
    ) -> None:
        self.parser = load_sparql_parser()
        self.manager = manager
        self.tokenizer = tokenizer
        self.mask_inputs = mask_inputs
        self.is_val = is_val

        self.skeleton_p = skeleton_p
        self.drop_infos_p = drop_infos_p
        self.drop_target_p = drop_target_p
        self.shuffle_alts_p = shuffle_alts_p
        self.constrain_p = constrain_p

        self.logger = get_logger(f"GRISP SELECTION DATASET ({is_val=})", log_level)

        self.samples = [sample for sample in samples if sample.has_placeholders]
        self.logger.info(
            f"Filtered {len(samples):,} samples to "
            f"{len(self.samples):,} samples with placeholders"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        messages, options, target = prepare_selection(
            sample,
            self.manager,
            self.is_val,
            self.skeleton_p,
            self.drop_infos_p,
            self.drop_target_p,
            self.shuffle_alts_p,
            self.constrain_p,
        )

        return tokenize_selection_and_log(
            SelectionSample(messages=messages, options=options, target=target),
            self.tokenizer,
            self.logger,
        )


def pad(values: list[list[int]], pad_value: int, max_length: int) -> torch.Tensor:
    padded = []

    max_length = min(max(len(seq) for seq in values), max_length)
    # pad max length to multiple of 8 for better efficiency
    if max_length % 8 != 0:
        max_length += 8 - (max_length % 8)

    for seq in values:
        if len(seq) <= max_length:
            seq = seq + [pad_value] * (max_length - len(seq))
        else:
            seq = seq[:max_length]
        padded.append(seq)

    return torch.tensor(padded, dtype=torch.long)


class GRISPCollator:
    def __init__(
        self,
        pad_token_id: int,
        max_length: int,
        log_level: str | int | None = None,
    ) -> None:
        self.max_length = max_length
        self.pad_values = {
            "input_ids": pad_token_id,
            "attention_mask": 0,
            "labels": IGNORE_INDEX,
        }

        self.logger = get_logger("GRISP COLLATOR", log_level)
        self.logger.info(
            f"Collating batch items to max length {self.max_length} with the "
            f"following pad values: {self.pad_values}"
        )

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        assert len(batch) > 0, "Batch must not be empty"
        keys = ["input_ids", "attention_mask", "labels"]
        output = {
            key: pad(
                [sample[key] for sample in batch],
                pad_value=self.pad_values[key],
                max_length=self.max_length,
            )
            for key in keys
        }

        # restricted-cross-entropy metadata; next-token rows (skeleton/
        # improvement) get sentinel values. is_rce marks the rows trained with
        # restricted/soft cross-entropy over option tokens (selection +
        # validation); the other rows are trained with plain next-token
        # prediction.
        B = len(batch)
        max_opts = max((len(s.get("option_token_ids", [])) for s in batch), default=0)
        max_opts = max(max_opts, 1)
        opt_ids = torch.zeros((B, max_opts), dtype=torch.long)
        opt_mask = torch.zeros((B, max_opts), dtype=torch.bool)
        # soft target distribution over options; one-hot rows recover hard CE
        target_dist = torch.zeros((B, max_opts), dtype=torch.float)
        answer_pos = torch.full((B,), -1, dtype=torch.long)
        is_rce = torch.zeros(B, dtype=torch.bool)

        for i, s in enumerate(batch):
            if "option_token_ids" not in s:
                continue
            n = len(s["option_token_ids"])
            opt_ids[i, :n] = torch.tensor(s["option_token_ids"], dtype=torch.long)
            opt_mask[i, :n] = True
            target_dist[i, :n] = torch.tensor(s["target_dist"], dtype=torch.float)
            answer_pos[i] = s["answer_pos"]
            is_rce[i] = True

        output["option_token_ids"] = opt_ids
        output["option_mask"] = opt_mask
        output["target_dist"] = target_dist
        output["answer_pos"] = answer_pos
        output["is_rce"] = is_rce

        if (
            torch.all(output["labels"] == IGNORE_INDEX).item()
            and not is_rce.any().item()
        ):
            seq_lens = output["attention_mask"].sum(dim=1).tolist()
            input_lens = [len(s["input_ids"]) for s in batch]
            label_lens = [len(s["labels"]) for s in batch]
            n_nonign_per_row = [
                sum(1 for x in s["labels"] if x != IGNORE_INDEX) for s in batch
            ]
            self.logger.warning(
                f"Batch has no next-token labels and no option-CE rows; "
                f"loss will be zero (no gradient signal).\n"
                f"  max_length={self.max_length}, padded_shape={tuple(output['input_ids'].shape)}\n"
                f"  per-row attention sums: {seq_lens}\n"
                f"  per-row pre-pad input_ids lens: {input_lens}\n"
                f"  per-row pre-pad labels lens:    {label_lens}\n"
                f"  per-row non-IGNORE label counts (pre-pad): {n_nonign_per_row}"
            )

        return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare data for GRISP model training"
    )
    parser.add_argument(
        "knowledge_graph",
        type=str,
        choices=get_available_knowledge_graphs(),
        help="Knowledge graph to prepare data for",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default=None,
        help="SPARQL endpoint for the knowledge graph",
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to input JSONL file containing query-SPARQL pairs",
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to output JSONL file to save the processed data",
    )
    parser.add_argument(
        "--allow-unknown",
        action="store_true",
        help="Allow (directly predict) unkown items instead of skipping the sample",
    )
    parser.add_argument(
        "--info-retrieval",
        action="store_true",
        help="Retrieve infos for items from the endpoint instead of taking "
        "label and aliases from the index. Off by default: samples only keep "
        "label and aliases, both of which the index has, so the live query "
        "costs one request per item and its infos are discarded",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=22,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it exists",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    setup_logging(args.log_level)
    logger = get_logger("GRISP DATA", args.log_level)

    # only override the kg's own info if an endpoint was given, an explicit
    # KgInfo(endpoint=None) counts as set and would discard it
    cfg = KgConfig(
        kg=args.knowledge_graph,
        info=KgInfo(endpoint=args.endpoint) if args.endpoint else None,
    )
    manager = load_kg_manager(cfg)
    manager.set_info_retrieval(enable=args.info_retrieval)

    if os.path.exists(args.output_file) and not args.overwrite:
        raise FileExistsError(
            f"Output file {args.output_file} already exists. "
            f"Use --overwrite to overwrite it."
        )

    samples = load_jsonl(args.input_file)
    logger.info(f"Loaded {len(samples):,} samples from {args.input_file}")
    random.seed(args.seed)

    invalid = 0
    error = 0
    outputs = []
    for sample in tqdm(samples, desc="Preparing samples"):
        sample = SparqlQaSample(**sample)

        try:
            sparql = manager.fix_prefixes(sample.sparql, remove_known=True)
            sparql = manager.prettify(sparql)
            sparql, items = extract_sparql_items(sparql, manager)

            invalid_items = [
                item
                for item in items
                if (item.is_unindexed and not item.has_label)
                or (item.is_unknown and not args.allow_unknown)
            ]
            if invalid_items:
                invalid += 1
                invalid_str = format_list(
                    item.alternative.get_selection_string(
                        include_variants=[item.variant] if item.variant else None
                    )
                    for item in invalid_items
                )

                logger.debug(
                    f"Invalid sample {sample.id}:\n\n{sample.question}\n\n"
                    f"{sparql}\n\n{invalid_str}"
                )
                continue

            parts = []
            start = 0
            for item in items:
                # literals, common, and unknown (if allowed) should be predicted directly
                if item.is_literal or item.is_common or item.is_unknown:
                    continue

                item_start, item_end = item.item_span

                parts.append(sparql[start:item_start])
                parts.append(IRI.from_item(item, manager))

                start = item_end

            if start < len(sparql):
                parts.append(sparql[start:])

            grisp_sample = GRISPSample(
                kg=args.knowledge_graph,
                questions=[sample.question] + sample.paraphrases,
                sparql=parts,
            )

            outputs.append(grisp_sample.model_dump())

        except Exception as e:
            error += 1
            logger.debug(
                f"Error processing sample {sample.id}:\n"
                f"{sample.model_dump_json(indent=2)}\n\n{e}"
            )
            continue

    dump_jsonl(outputs, args.output_file)

    logger.info(f"Total samples processed: {len(samples):,}")
    inv_frac = invalid / len(samples)
    logger.info(f"Total invalid samples skipped: {invalid:,} ({inv_frac:.2%})")
    err_frac = error / len(samples)
    logger.info(f"Total errors encountered: {error:,} ({err_frac:.2%})")


if __name__ == "__main__":
    main(parse_args())
