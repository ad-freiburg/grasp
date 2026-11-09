import math
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Any

from tqdm import tqdm
from universal_ml_utils.logging import get_logger

from grasp.configs import ShapeConfig
from grasp.manager import KgManager
from grasp.manager.normalizer import Normalizer
from grasp.shapes import (
    ClassProfile,
    PropertyProfile,
    Target,
    TargetBnode,
    TargetClass,
    TargetIri,
    TargetLiteral,
)
from grasp.sparql.types import AskResult, SelectResult
from grasp.utils import derive_label_from_iri, get_local_name_from_iri

# where to look up class labels, the class index first
CLASS_INDICES = ["classes", "entities"]


# instantiate a pattern by choosing the class term and the variable the
# membership is anchored on. anchoring on ?o (or ?s) is what turns an outgoing
# pattern into an incoming one; the pattern itself is directionless.
def bind_pattern(
    pattern: str,
    class_term: str = "?class",
    member_var: str = "?instance",
) -> str:
    return pattern.replace("?instance", member_var).replace("{CLASS}", class_term)


def iri_term(class_iri: str) -> str:
    return f"<{class_iri}>"


# dedup here plus COUNT(*) outside scales where COUNT(DISTINCT ?instance) does
# not, since the latter materializes the whole membership relation.
def distinct_members(pattern: str, class_term: str = "?class") -> str:
    cp = bind_pattern(pattern, class_term)
    inner = "\n".join(f"      {line}" for line in cp.strip().splitlines())
    project = "?class ?instance" if class_term == "?class" else "?instance"
    # a class is always an IRI; filtering here also prunes before the DISTINCT
    if class_term == "?class":
        inner += "\n      FILTER(ISIRI(?class))"
    return f"  {{\n    SELECT DISTINCT {project} WHERE {{\n{inner}\n    }}\n  }}"


def build_total_entities_query(pattern: str) -> str:
    # ordered so that max_classes keeps the biggest classes, not an arbitrary cut
    return (
        f"SELECT ?class (COUNT(*) AS ?totalEntities)\n"
        f"WHERE {{\n"
        f"{distinct_members(pattern)}\n"
        f"}}\n"
        f"GROUP BY ?class\n"
        f"ORDER BY DESC(?totalEntities)"
    )


def wrap_pattern(pattern: str) -> str:
    inner = "\n".join(f"    {line}" for line in pattern.strip().splitlines())
    return f"  {{\n{inner}\n  }}"


@dataclass(frozen=True)
class InstanceSample:
    total: int
    limit: int
    windows: int
    ordered: bool = True

    @property
    def per_window(self) -> int:
        return max(1, self.limit // self.windows)

    # instances actually profiled, 0 if the class is small enough to be exact
    @property
    def size(self) -> int:
        if self.total <= self.limit:
            return 0
        return self.per_window * self.windows


# full extension of a class, or a sample of it when profiling all of it would
# exhaust the endpoint; spread windows because one slice skews the property
# distribution, ORDER BY (optional) because LIMIT/OFFSET is undefined without it
def members_block(
    pattern: str,
    class_iri: str,
    member_var: str = "?instance",
    sample: InstanceSample | None = None,
) -> str:
    bound = bind_pattern(pattern, iri_term(class_iri), member_var)
    if sample is None or not sample.size:
        return wrap_pattern(bound)

    # stride >= per_window whenever total > limit, so windows never overlap
    stride = sample.total // sample.windows
    inner = "\n".join(f"        {line}" for line in bound.strip().splitlines())
    order = f"ORDER BY {member_var} " if sample.ordered else ""
    windows = [
        f"    {{\n"
        f"      SELECT DISTINCT {member_var} WHERE {{\n{inner}\n      }}\n"
        f"      {order}LIMIT {sample.per_window} OFFSET {i * stride}\n"
        f"    }}"
        for i in range(sample.windows)
    ]
    return "  {\n" + "\n    UNION\n".join(windows) + "\n  }"


def build_per_class_property_frequency_query(
    pattern: str, class_iri: str, sample: InstanceSample | None = None
) -> str:
    ip = members_block(pattern, class_iri, sample=sample)
    # ?bnodeCount rides along as an extra aggregate so the node-kind split costs
    # no additional query.
    return (
        f"SELECT ?p (COUNT(*) AS ?tripleCount) (COUNT(DISTINCT ?instance) AS ?entityCount)"
        f" (SUM(IF(isBlank(?o), 1, 0)) AS ?bnodeCount)\n"
        f"WHERE {{\n"
        f"{ip}\n"
        f"  ?instance ?p ?o .\n"
        f"}}\n"
        f"GROUP BY ?p\n"
        f"ORDER BY DESC(?tripleCount)"
    )


def build_per_class_literal_profile_query(
    pattern: str, class_iri: str, sample: InstanceSample | None = None
) -> str:
    ip = members_block(pattern, class_iri, sample=sample)
    return (
        f"SELECT ?p ?datatype (COUNT(*) AS ?count)\n"
        f"WHERE {{\n"
        f"{ip}\n"
        f"  ?instance ?p ?o .\n"
        f"  FILTER(isLiteral(?o))\n"
        f"  BIND(DATATYPE(?o) AS ?datatype)\n"
        f"}}\n"
        f"GROUP BY ?p ?datatype\n"
        f"ORDER BY ?p DESC(?count)"
    )


def build_per_class_range_profile_query(
    pattern: str, class_iri: str, sample: InstanceSample | None = None
) -> str:
    ip = members_block(pattern, class_iri, sample=sample)
    op = wrap_pattern(bind_pattern(pattern, "?targetClass", "?o"))
    # blank-node objects are excluded here so that they are counted exactly once,
    # by ?bnodeCount above, instead of also landing in a class bucket.
    return (
        f"SELECT ?p ?targetClass (COUNT(*) AS ?count)\n"
        f"WHERE {{\n"
        f"{ip}\n"
        f"  ?instance ?p ?o .\n"
        f"  FILTER(!isBlank(?o))\n"
        f"{op}\n"
        f"}}\n"
        f"GROUP BY ?p ?targetClass\n"
        f"ORDER BY ?p DESC(?count)"
    )


def build_per_class_incoming_frequency_query(
    pattern: str, class_iri: str, sample: InstanceSample | None = None
) -> str:
    op = members_block(pattern, class_iri, "?o", sample)
    return (
        f"SELECT ?p (COUNT(*) AS ?tripleCount) (COUNT(DISTINCT ?o) AS ?entityCount)"
        f" (SUM(IF(isBlank(?s), 1, 0)) AS ?bnodeCount)\n"
        f"WHERE {{\n"
        f"{op}\n"
        f"  ?s ?p ?o .\n"
        f"}}\n"
        f"GROUP BY ?p\n"
        f"ORDER BY DESC(?tripleCount)"
    )


def build_per_class_incoming_source_query(
    pattern: str, class_iri: str, sample: InstanceSample | None = None
) -> str:
    op = members_block(pattern, class_iri, "?o", sample)
    sp = wrap_pattern(bind_pattern(pattern, "?sourceClass", "?s"))
    return (
        f"SELECT ?p ?sourceClass (COUNT(*) AS ?count)\n"
        f"WHERE {{\n"
        f"{op}\n"
        f"  ?s ?p ?o .\n"
        f"  FILTER(!isBlank(?s))\n"
        f"{sp}\n"
        f"}}\n"
        f"GROUP BY ?p ?sourceClass\n"
        f"ORDER BY ?p DESC(?count)"
    )


def build_per_class_total_query(pattern: str, class_iri: str) -> str:
    ip = distinct_members(pattern, iri_term(class_iri))
    return f"SELECT (COUNT(*) AS ?totalEntities)\nWHERE {{\n{ip}\n}}"


XSD_NAMESPACE = "http://www.w3.org/2001/XMLSchema#"
RDF_DATATYPE_IRIS = {
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#langString",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#HTML",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#XMLLiteral",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#JSON",
}


def is_datatype_iri(iri: str) -> bool:
    return iri.startswith(XSD_NAMESPACE) or iri in RDF_DATATYPE_IRIS


def nested_counter() -> dict[str, dict[str, int]]:
    return defaultdict(lambda: defaultdict(int))


@dataclass
class PropertyFreq:
    triple_count: int = 0
    entity_count: int = 0

    def add(self, other: "PropertyFreq") -> None:
        self.triple_count += other.triple_count
        self.entity_count += other.entity_count


# property profile in one direction, outgoing or incoming
@dataclass
class DirectedMaps:
    freq: dict[str, PropertyFreq] = field(default_factory=dict)
    literals: dict[str, dict[str, int]] = field(default_factory=nested_counter)
    # target classes when outgoing, source classes when incoming
    classes: dict[str, dict[str, int]] = field(default_factory=nested_counter)
    bnodes: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def merge(self, other: "DirectedMaps") -> None:
        for p, freq in other.freq.items():
            self.freq.setdefault(p, PropertyFreq()).add(freq)
        for p, dtype_counts in other.literals.items():
            for dt, c in dtype_counts.items():
                self.literals[p][dt] += c
        for p, class_counts in other.classes.items():
            for t, c in class_counts.items():
                self.classes[p][t] += c
        for p, c in other.bnodes.items():
            self.bnodes[p] += c


@dataclass
class ClassMaps:
    out: DirectedMaps = field(default_factory=DirectedMaps)
    inc: DirectedMaps = field(default_factory=DirectedMaps)

    def merge(self, other: "ClassMaps") -> None:
        self.out.merge(other.out)
        self.inc.merge(other.inc)

    @classmethod
    def from_instance_rows(
        cls,
        freq_rows: list[dict],
        lit_rows: list[dict],
        range_rows: list[dict],
        manager: KgManager,
    ) -> "ClassMaps":
        maps = cls()
        for row in freq_rows:
            maps.out.freq[row["p"]] = PropertyFreq(
                triple_count=int(row["tripleCount"]),
                entity_count=int(row["entityCount"]),
            )
            maps.out.bnodes[row["p"]] += int(row.get("bnodeCount") or 0)
        for row in lit_rows:
            dt = manager.format_iri(row["datatype"], wrap=True)
            maps.out.literals[row["p"]][dt] += int(row["count"])
        for row in range_rows:
            maps.out.classes[row["p"]][row["targetClass"]] += int(row["count"])
        return maps

    @classmethod
    def from_incoming_rows(
        cls,
        freq_rows: list[dict],
        source_rows: list[dict],
    ) -> "ClassMaps":
        maps = cls()
        for row in freq_rows:
            maps.inc.freq[row["p"]] = PropertyFreq(
                triple_count=int(row["tripleCount"]),
                entity_count=int(row["entityCount"]),
            )
            maps.inc.bnodes[row["p"]] += int(row.get("bnodeCount") or 0)
        for row in source_rows:
            maps.inc.classes[row["p"]][row["sourceClass"]] += int(row["count"])
        return maps

    @classmethod
    def from_schema_rows(cls, rows: list, manager: KgManager) -> "ClassMaps":
        maps = cls()
        for row in rows:
            prop = row.get("property")
            if prop is None:
                continue

            direction = row.get("dir")
            if direction is None:
                raise ValueError(
                    "Schema pattern returned a row without a bound '?dir'. "
                    'Every branch of the pattern must bind ?dir to "out" or '
                    '"in" (e.g. via BIND("out" AS ?dir)).'
                )
            if direction.value == "out":
                side = maps.out
            elif direction.value == "in":
                side = maps.inc
            else:
                raise ValueError(
                    f"Schema pattern bound ?dir to '{direction.value}', "
                    'expected "out" or "in".'
                )

            p = prop.value
            side.freq.setdefault(p, PropertyFreq())
            peer = row.get("other")
            if peer is None:
                continue
            if peer.typ == "literal":
                dt_iri = peer.datatype or f"{XSD_NAMESPACE}string"
                side.literals[p][manager.format_iri(dt_iri, wrap=True)] += 1
            elif peer.typ == "uri" and is_datatype_iri(peer.value):
                side.literals[p][manager.format_iri(peer.value, wrap=True)] += 1
            elif peer.typ == "uri":
                side.classes[p][peer.value] += 1
            else:
                # anonymous class expression (e.g. an owl:unionOf domain/range)
                side.bnodes[p] += 1
            side.freq[p].triple_count += 1
        return maps


@dataclass
class NormalizedMaps:
    freq: dict[str, PropertyFreq]
    literals: dict[str, dict[str, int]]
    ranges: dict[str, dict[str, int]]
    bnodes: dict[str, int]
    prop_variants: dict[str, list[str]]
    target_variants: dict[str, dict[str, list[str]]]


def build_schema_class_query(pattern: str) -> str:
    cp = wrap_pattern(bind_pattern(pattern))
    # same IRI-only rule as the instance path; no count here to order by
    return f"SELECT DISTINCT ?class\nWHERE {{\n{cp}\n  FILTER(ISIRI(?class))\n}}"


def build_schema_profile_query(pattern: str, class_iri: str) -> str:
    sp = wrap_pattern(bind_pattern(pattern, iri_term(class_iri)))
    return f"SELECT DISTINCT ?property ?other ?dir\nWHERE {{\n{sp}\n}}"


def build_iri_exists_query(iri: str) -> str:
    # Class-hood cannot be probed with the shape pattern itself: a real class
    # that simply has no matching properties fails it just like a non-existent
    # one. So we only check whether the IRI occurs in the graph at all.
    return f"ASK {{\n  {{ <{iri}> ?p ?o }}\n  UNION\n  {{ ?s ?p <{iri}> }}\n}}"


def property_rank(iri: str, manager: KgManager) -> int:
    data = manager.try_get_data("properties")
    if data is None:
        return 0  # no index: all equal, triple_count tiebreaker takes over
    norm = manager.normalize(iri, "properties")
    if norm is not None:
        iri = norm[0]
    id = data.id_from_identifier(iri)
    return id if id is not None else len(data)  # unknown props sort last


def cardinality_tag(coverage: float, avg_values: float) -> str:
    if coverage >= 0.90 and avg_values <= 1.10:
        return ""
    elif coverage < 0.90 and avg_values <= 1.10:
        return "?"
    elif coverage >= 0.90 and avg_values > 1.10:
        return "+"
    else:
        return "*"


def resolve_label(
    iri: str,
    index_names: list[str],
    short_iri: str,
    manager: KgManager,
    variants: list[str] | None = None,
) -> str:
    label = next(
        (
            label
            for name in index_names
            if (label := manager.get_label(iri, name)) is not None
        ),
        None,
    )
    if label:
        local_name = get_local_name_from_iri(iri, manager.prefixes)
        derived = derive_label_from_iri(iri, manager.prefixes)
        if local_name.lower() == label.lower() or derived.lower() == label.lower():
            label = None

    parts: list[str] = []
    if label:
        parts.append(label)  # type: ignore[arg-type]
    if variants:
        parts.append(f"as {'/'.join(variants)}")

    if parts:
        return f"{short_iri} ({', '.join(parts)})"

    return short_iri


def bracketed(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    return f"[ {' '.join(items)} ]"


def render_target(target: Target, manager: KgManager) -> str:
    if isinstance(target, TargetLiteral):
        return target.datatype
    if isinstance(target, TargetBnode):
        return "BNODE"
    if isinstance(target, TargetClass):
        return resolve_label(
            target.iri,
            CLASS_INDICES,
            target.short_iri,
            manager,
            variants=target.variants or None,
        )
    return "IRI"


@dataclass
class ShapeView:
    # "out" for the properties of the instances, "in" for those pointing to them
    direction: str
    properties: list[PropertyProfile]
    page: int
    total_pages: int
    # over all pages
    total: int
    # rare instance properties left out
    rare: int = 0
    # properties shown for the other direction without a query
    other: int = 0


def is_rare(prop: PropertyProfile, share_base: int, min_share: float) -> bool:
    # only rare *instance* properties; a property with no instance usage
    # (entity_count == 0) is a pure schema declaration, not noise.
    return (
        share_base > 0
        and prop.entity_count > 0
        and prop.entity_count / share_base < min_share
    )


def rank_properties(
    props: list[PropertyProfile],
    ranking: list[str],
) -> list[PropertyProfile]:
    # ranked ones first, the rest keeps its order
    pos = {iri: i for i, iri in enumerate(ranking)}
    return sorted(props, key=lambda prop: pos.get(prop.iri, len(pos)))


def select_shape_view(
    profile: ClassProfile,
    shape_config: ShapeConfig,
    k: int,
    direction: str = "out",
    page: int = 1,
    ranking: list[str] | None = None,
) -> ShapeView:
    def common(props: list[PropertyProfile]) -> list[PropertyProfile]:
        return [
            p
            for p in props
            if not is_rare(p, profile.share_base, shape_config.min_property_share)
        ]

    if direction == "out":
        props, other = profile.properties, profile.incoming
    else:
        props, other = profile.incoming, profile.properties

    rare = 0
    if ranking is None:
        shown = common(props)
        rare = len(props) - len(shown)
    else:
        # searching shows all, best matches first
        shown = rank_properties(props, ranking)

    return ShapeView(
        direction=direction,
        properties=shown[(page - 1) * k : page * k],
        page=page,
        total_pages=max(1, math.ceil(len(shown) / k)),
        total=len(shown),
        rare=rare,
        other=len(common(other)),
    )


def cardinality_tag_for_property(prop: PropertyProfile, total_entities: int) -> str:
    if total_entities <= 0:
        return ""
    coverage = prop.entity_count / total_entities
    avg_vals = prop.triple_count / prop.entity_count if prop.entity_count > 0 else 1.0
    return cardinality_tag(coverage, avg_vals)


DIRECTION_NAMES = {"out": "outgoing", "in": "incoming"}


def render_shape(
    profile: ClassProfile,
    view: ShapeView,
    manager: KgManager,
    shape_config: ShapeConfig,
) -> str:
    header = resolve_label(profile.iri, CLASS_INDICES, profile.short_iri, manager)
    lines = [f"{header} {{"]

    def render(prop: PropertyProfile, inverse: bool) -> str:
        prop_str = resolve_label(
            prop.iri,
            ["properties"],
            prop.short_iri,
            manager,
            variants=prop.variants or None,
        )
        if prop.targets:
            value_str = bracketed([render_target(t, manager) for t in prop.targets])
        else:
            value_str = "IRI"

        if inverse:
            # no cardinality tag: coverage and values-per-entity are defined over
            # the focus class' own instances and do not carry over to inverse edges.
            return f"  ^{prop_str} {value_str} ;"

        tag = cardinality_tag_for_property(prop, profile.share_base)
        tag_suffix = f" {tag}" if tag else ""
        return f"  {prop_str} {value_str}{tag_suffix} ;"

    inverse = view.direction == "in"
    lines.extend(render(prop, inverse) for prop in view.properties)

    name = DIRECTION_NAMES[view.direction]
    info = []
    if not view.total and not view.rare:
        # make the empty case explicit; an empty body on its own is ambiguous
        info.append(f"no {name} properties found for this class")
    elif view.total_pages > 1 or view.rare or view.other:
        info.append(
            f"page {view.page} of {view.total_pages}, {view.total:,} {name} "
            + ("property" if view.total == 1 else "properties")
        )

    if view.rare:
        share = f"{shape_config.min_property_share:.0%}"
        info.append(
            f"{view.rare:,} more used by less than {share} of the instances "
            "(pass a query to rank all properties including these)"
        )

    if view.other:
        other = "in" if view.direction == "out" else "out"
        info.append(f'{view.other:,} {DIRECTION_NAMES[other]} (direction="{other}")')

    if info:
        lines.append(f"  # {', '.join(info)}")

    lines.append("}")

    return "\n".join(lines)


def emit_pseudo_shex(
    profile: ClassProfile,
    manager: KgManager,
    shape_config: ShapeConfig,
    k: int = 10,
    direction: str = "out",
    page: int = 1,
    ranking: list[str] | None = None,
) -> str:
    view = select_shape_view(profile, shape_config, k, direction, page, ranking)
    return render_shape(profile, view, manager, shape_config)


def collect_iris(view: ShapeView, manager: KgManager) -> list[str]:
    iris: list[str] = []
    prop_norm = manager.get_normalizer("properties")
    ent_norm = manager.get_normalizer("entities")
    for prop in view.properties:
        iris.append(prop.iri)
        for v in prop.variants:
            denorm = prop_norm.denormalize(prop.iri, v)
            if denorm:
                iris.append(denorm)
        for t in prop.targets:
            if not isinstance(t, TargetClass):
                continue
            iris.append(t.iri)
            for v in t.variants:
                denorm = ent_norm.denormalize(t.iri, v)
                if denorm:
                    iris.append(denorm)
    return iris


def group_by_normalized(maps: DirectedMaps, manager: KgManager) -> NormalizedMaps:
    prop_normalizer = manager.get_normalizer("properties")
    ent_normalizer = manager.get_normalizer("entities")

    def norm_key(iri: str, normalizer: Normalizer) -> tuple[str, str | None]:
        norm = normalizer.normalize(iri)
        if norm is None:
            return iri, None
        return norm

    prop_key = partial(norm_key, normalizer=prop_normalizer)
    ent_key = partial(norm_key, normalizer=ent_normalizer)

    new_freq: dict[str, PropertyFreq] = {}
    new_lit: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    new_range: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    new_bnodes: dict[str, int] = defaultdict(int)
    prop_variants: dict[str, list[str]] = defaultdict(list)
    target_variants: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for p_iri, freq in maps.freq.items():
        k, v = prop_key(p_iri)
        new_freq.setdefault(k, PropertyFreq()).add(freq)
        if v is not None and v not in prop_variants[k]:
            prop_variants[k].append(v)

    for p_iri, dtype_counts in maps.literals.items():
        k, _ = prop_key(p_iri)
        for dt, c in dtype_counts.items():
            new_lit[k][dt] += c

    for p_iri, tc_counts in maps.classes.items():
        k, _ = prop_key(p_iri)
        for t_iri, c in tc_counts.items():
            tk, tv = ent_key(t_iri)
            new_range[k][tk] += c
            if tv is not None and tv not in target_variants[k][tk]:
                target_variants[k][tk].append(tv)

    for p_iri, c in maps.bnodes.items():
        k, _ = prop_key(p_iri)
        new_bnodes[k] += c

    # sort variants
    for k in list(prop_variants.keys()):
        prop_variants[k] = sorted(prop_variants[k])

    for k in list(target_variants.keys()):
        for tk in list(target_variants[k].keys()):
            target_variants[k][tk] = sorted(target_variants[k][tk])

    return NormalizedMaps(
        freq=new_freq,
        literals=new_lit,
        ranges=new_range,
        bnodes=new_bnodes,
        prop_variants=prop_variants,
        target_variants=target_variants,
    )


def assemble_direction(
    maps: DirectedMaps,
    shape_config: ShapeConfig,
    manager: KgManager,
) -> list[PropertyProfile]:
    norm = group_by_normalized(maps, manager)

    common_props = sorted(
        norm.freq.items(),
        key=lambda item: (
            property_rank(item[0], manager),
            -item[1].triple_count,
        ),
    )

    properties: list[PropertyProfile] = []
    for p_iri, freq in common_props:
        targets: list[Target] = []

        for dt, c in norm.literals.get(p_iri, {}).items():
            targets.append(TargetLiteral(datatype=dt, triple_count=c))

        for t_iri, c in norm.ranges.get(p_iri, {}).items():
            targets.append(
                TargetClass(
                    iri=t_iri,
                    short_iri=manager.format_iri(t_iri, wrap=True),
                    variants=norm.target_variants.get(p_iri, {}).get(t_iri, []),
                    triple_count=c,
                )
            )

        # blank nodes are counted separately from the class buckets (the class
        # queries filter them out), so they never double count.
        bnode_count = norm.bnodes.get(p_iri, 0)
        if bnode_count > 0:
            targets.append(TargetBnode(triple_count=bnode_count))

        # gap between total triples and the classified buckets are untyped IRIs.
        # require the gap to be both positive and a meaningful share of total.
        typed_sum = sum(t.triple_count for t in targets)
        gap = freq.triple_count - typed_sum
        if (
            gap > 0
            and freq.triple_count > 0
            and gap / freq.triple_count >= shape_config.min_target_share
        ):
            targets.append(TargetIri(triple_count=gap))

        targets.sort(key=lambda t: t.triple_count, reverse=True)
        targets = targets[: shape_config.max_targets_per_property]

        properties.append(
            PropertyProfile(
                iri=p_iri,
                short_iri=manager.format_iri(p_iri, wrap=True),
                triple_count=freq.triple_count,
                entity_count=freq.entity_count,
                variants=norm.prop_variants.get(p_iri, []),
                targets=targets,
            )
        )

    return properties


def assemble_profile(
    class_iri: str,
    maps: ClassMaps,
    total_entities: int,
    shape_config: ShapeConfig,
    manager: KgManager,
    profiled_entities: int = 0,
) -> ClassProfile:
    return ClassProfile(
        iri=class_iri,
        short_iri=manager.format_iri(class_iri, wrap=True),
        total_entities=total_entities,
        profiled_entities=profiled_entities,
        properties=assemble_direction(maps.out, shape_config, manager),
        incoming=assemble_direction(maps.inc, shape_config, manager),
    )


def select_values(result: SelectResult) -> list[dict[str, Any]]:
    return [{var: row[var].value for var in result.variables} for row in result.rows()]


def collect_class_maps(
    class_iri: str,
    manager: KgManager,
    execute: Callable[[str], SelectResult],
    instance_pattern: str | None = None,
    schema_pattern: str | None = None,
    sample: InstanceSample | None = None,
) -> ClassMaps:
    maps = ClassMaps()
    if instance_pattern is not None:
        freq_rows = select_values(
            execute(
                build_per_class_property_frequency_query(
                    instance_pattern, class_iri, sample
                )
            )
        )
        lit_rows = select_values(
            execute(
                build_per_class_literal_profile_query(
                    instance_pattern, class_iri, sample
                )
            )
        )
        range_rows = select_values(
            execute(
                build_per_class_range_profile_query(instance_pattern, class_iri, sample)
            )
        )
        maps.merge(
            ClassMaps.from_instance_rows(freq_rows, lit_rows, range_rows, manager)
        )
        in_freq_rows = select_values(
            execute(
                build_per_class_incoming_frequency_query(
                    instance_pattern, class_iri, sample
                )
            )
        )
        in_source_rows = select_values(
            execute(
                build_per_class_incoming_source_query(
                    instance_pattern, class_iri, sample
                )
            )
        )
        maps.merge(ClassMaps.from_incoming_rows(in_freq_rows, in_source_rows))
    if schema_pattern is not None:
        rows = list(
            execute(build_schema_profile_query(schema_pattern, class_iri)).rows()
        )
        maps.merge(ClassMaps.from_schema_rows(rows, manager))
    return maps


def compute_shape(
    class_iri: str,
    manager: KgManager,
    instance_pattern: str | None = None,
    schema_pattern: str | None = None,
    shape_config: ShapeConfig | None = None,
) -> ClassProfile:
    if shape_config is None:
        shape_config = ShapeConfig()
    if instance_pattern is None and schema_pattern is None:
        raise ValueError("compute_shape requires an instance and/or schema pattern")

    def execute(query: str) -> SelectResult:
        result = manager.execute_sparql(
            query,
            shape_config.request_timeout,
            shape_config.read_timeout,
            sparql_result_max_rows=shape_config.sparql_result_max_rows,
        )
        assert isinstance(result, SelectResult), "Expected SELECT query result"
        return result

    total = 0
    sample = None
    try:
        if instance_pattern is not None:
            # large classes are sampled, as in the precomputed shapes
            total_rows = select_values(
                execute(build_per_class_total_query(instance_pattern, class_iri))
            )
            total = int(total_rows[0]["totalEntities"]) if total_rows else 0
            sample = InstanceSample(
                total=total,
                limit=shape_config.max_profile_instances,
                windows=shape_config.profile_windows,
                ordered=shape_config.ordered_samples,
            )

        maps = collect_class_maps(
            class_iri,
            manager,
            execute,
            instance_pattern,
            schema_pattern,
            sample=sample,
        )
    except Exception as e:
        raise RuntimeError(
            f"One of the queries for computing the shape of '{class_iri}' failed:\n{e}"
        )

    profile = assemble_profile(
        class_iri,
        maps,
        total,
        shape_config,
        manager,
        sample.size if sample is not None else 0,
    )

    # an empty profile is ambiguous: the class may exist but have no properties
    # matching the pattern, or the IRI may not be in the graph at all. Only in
    # that case is the extra probe worth its query.
    if not profile.properties and not profile.incoming and profile.total_entities == 0:
        result = manager.execute_sparql(
            build_iri_exists_query(class_iri),
            shape_config.request_timeout,
            shape_config.read_timeout,
        )
        assert isinstance(result, AskResult), "Expected ASK query result"
        if not result.boolean:
            raise RuntimeError(f"'{class_iri}' does not exist in the knowledge graph")

    return profile


def build_shapes(
    manager: KgManager,
    instance_pattern: str | None = None,
    schema_pattern: str | None = None,
    shape_config: ShapeConfig = ShapeConfig(),
    max_classes: int = 500,
    log_level: str | int | None = None,
    request_timeout: float | tuple[float, float] | None = None,
    read_timeout: float | None = None,
) -> tuple[list[ClassProfile], int]:
    logger = get_logger("GRASP SHAPES BUILD", log_level)
    assert instance_pattern is not None or schema_pattern is not None, (
        "build_shapes requires an instance and/or schema pattern"
    )

    def execute(query: str) -> SelectResult:
        logger.debug(f"Running query:\n{query}")
        result = manager.execute_sparql(
            query,
            request_timeout,
            read_timeout,
            sparql_result_max_rows=shape_config.sparql_result_max_rows,
        )
        assert isinstance(result, SelectResult), "Expected SELECT query result"
        return result

    logger.info("Discovering classes")
    instance_total: dict[str, int] = {}
    if instance_pattern is not None:
        try:
            total_rows = select_values(
                execute(build_total_entities_query(instance_pattern))
            )
        except Exception as e:
            raise ValueError(
                f"Instance pattern validation failed: could not query classes ({e}). "
                "Check that the pattern correctly connects instances to class nodes."
            ) from e
        instance_total = {row["class"]: int(row["totalEntities"]) for row in total_rows}

    schema_classes: list[str] = []
    if schema_pattern is not None:
        try:
            class_rows = select_values(
                execute(build_schema_class_query(schema_pattern))
            )
        except Exception as e:
            raise ValueError(
                f"Schema pattern validation failed: could not query classes ({e}). "
                "Check that the pattern binds ?property (and optionally ?target) for {CLASS}."
            ) from e
        schema_classes = [row["class"] for row in class_rows]

    schema_set = set(schema_classes)
    # union of discovered classes: instance classes first (in discovery order),
    # then schema-only classes.
    ordered = list(instance_total.keys()) + [
        c for c in schema_classes if c not in instance_total
    ]
    if not ordered:
        raise ValueError(
            "Pattern validation failed: no classes found. "
            "Check that the pattern(s) correctly identify classes."
        )
    total_classes = len(ordered)
    all_class_iris = ordered[:max_classes]
    logger.info(f"Building shapes for {len(all_class_iris)} classes")

    profiles = []
    skipped = 0
    sampled = 0
    for c_iri in tqdm(all_class_iris, desc="Profiling classes"):
        # total comes from the discovery query above, not a per-class query
        total = instance_total.get(c_iri, 0)
        sample = InstanceSample(
            total=total,
            limit=shape_config.max_profile_instances,
            windows=shape_config.profile_windows,
            ordered=shape_config.ordered_samples,
        )
        try:
            maps = collect_class_maps(
                c_iri,
                manager,
                execute,
                instance_pattern if c_iri in instance_total else None,
                schema_pattern if c_iri in schema_set else None,
                sample=sample,
            )
        except Exception as e:
            logger.warning(
                f"Skipping {c_iri}: profiling query failed ({e}). "
                "To retry, increase timeouts in the shape config and re-run with --overwrite."
            )
            skipped += 1
            continue

        if sample.size:
            sampled += 1

        profiles.append(
            assemble_profile(
                c_iri,
                maps,
                total,
                shape_config,
                manager,
                sample.size,
            )
        )

    if skipped:
        logger.warning(f"Skipped {skipped:,} class(es) due to query failures")

    if sampled:
        logger.info(f"Sampled {sampled:,} class(es) too large to profile exhaustively")
    logger.info(f"Built {len(profiles)} shapes")
    return profiles, total_classes
