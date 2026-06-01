import os
from collections import defaultdict
from typing import Any

from tqdm import tqdm
from universal_ml_utils.io import dump_text
from universal_ml_utils.logging import get_logger

from grasp.configs import ShapeConfig
from grasp.manager import KgManager
from grasp.shapes import (
    ClassProfile,
    PropertyProfile,
    ShapeSample,
    Target,
    TargetClass,
    TargetIri,
    TargetLiteral,
)
from grasp.sparql.types import SelectResult
from grasp.utils import derive_label_from_iri, get_local_name_from_iri


def class_pattern(pattern: str) -> str:
    return pattern.replace("{CLASS}", "?class")


def object_pattern(pattern: str) -> str:
    p = pattern.replace("?instance", "?o")
    return p.replace("{CLASS}", "?targetClass")


def instance_pattern(pattern: str, class_iri: str) -> str:
    return pattern.replace("{CLASS}", f"<{class_iri}>")


def subquery(pattern: str) -> str:
    cp = class_pattern(pattern)
    return f"  {{\n    SELECT DISTINCT ?class WHERE {{\n      {cp}\n    }}\n  }}"


def membership(pattern: str) -> str:
    cp = class_pattern(pattern)
    inner = "\n".join(f"    {line}" for line in cp.strip().splitlines())
    return f"  {{\n{inner}\n  }}"


def objectmembership(pattern: str) -> str:
    op = object_pattern(pattern)
    inner = "\n".join(f"    {line}" for line in op.strip().splitlines())
    return f"  {{\n{inner}\n  }}"


def build_property_frequency_query(pattern: str) -> str:
    return (
        f"SELECT ?class ?p (COUNT(*) AS ?tripleCount) "
        f"(COUNT(DISTINCT ?instance) AS ?entityCount)\n"
        f"WHERE {{\n"
        f"{subquery(pattern)}\n"
        f"{membership(pattern)}\n"
        f"  ?instance ?p ?o .\n"
        f"}}\n"
        f"GROUP BY ?class ?p\n"
        f"ORDER BY ?class DESC(?tripleCount)"
    )


def build_literal_profile_query(pattern: str) -> str:
    return (
        f"SELECT ?class ?p ?datatype (COUNT(*) AS ?count)\n"
        f"WHERE {{\n"
        f"{subquery(pattern)}\n"
        f"{membership(pattern)}\n"
        f"  ?instance ?p ?o .\n"
        f"  FILTER(isLiteral(?o))\n"
        f"  BIND(DATATYPE(?o) AS ?datatype)\n"
        f"}}\n"
        f"GROUP BY ?class ?p ?datatype\n"
        f"ORDER BY ?class ?p DESC(?count)"
    )


def build_range_profile_query(pattern: str) -> str:
    return (
        f"SELECT ?class ?p ?targetClass (COUNT(*) AS ?count)\n"
        f"WHERE {{\n"
        f"{subquery(pattern)}\n"
        f"{membership(pattern)}\n"
        f"  ?instance ?p ?o .\n"
        f"{objectmembership(pattern)}\n"
        f"}}\n"
        f"GROUP BY ?class ?p ?targetClass\n"
        f"ORDER BY ?class ?p DESC(?count)"
    )


def build_total_entities_query(pattern: str) -> str:
    return (
        f"SELECT ?class (COUNT(DISTINCT ?instance) AS ?totalEntities)\n"
        f"WHERE {{\n"
        f"{subquery(pattern)}\n"
        f"{membership(pattern)}\n"
        f"}}\n"
        f"GROUP BY ?class"
    )


def wrap_pattern(pattern: str) -> str:
    inner = "\n".join(f"    {line}" for line in pattern.strip().splitlines())
    return f"  {{\n{inner}\n  }}"


def build_per_class_property_frequency_query(pattern: str, class_iri: str) -> str:
    ip = wrap_pattern(instance_pattern(pattern, class_iri))
    return (
        f"SELECT ?p (COUNT(*) AS ?tripleCount) (COUNT(DISTINCT ?instance) AS ?entityCount)\n"
        f"WHERE {{\n"
        f"{ip}\n"
        f"  ?instance ?p ?o .\n"
        f"}}\n"
        f"GROUP BY ?p\n"
        f"ORDER BY DESC(?tripleCount)"
    )


def build_per_class_literal_profile_query(pattern: str, class_iri: str) -> str:
    ip = wrap_pattern(instance_pattern(pattern, class_iri))
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


def build_per_class_range_profile_query(pattern: str, class_iri: str) -> str:
    ip = wrap_pattern(instance_pattern(pattern, class_iri))
    op = wrap_pattern(object_pattern(pattern))
    return (
        f"SELECT ?p ?targetClass (COUNT(*) AS ?count)\n"
        f"WHERE {{\n"
        f"{ip}\n"
        f"  ?instance ?p ?o .\n"
        f"{op}\n"
        f"}}\n"
        f"GROUP BY ?p ?targetClass\n"
        f"ORDER BY ?p DESC(?count)"
    )


def build_per_class_total_query(pattern: str, class_iri: str) -> str:
    ip = wrap_pattern(instance_pattern(pattern, class_iri))
    return f"SELECT (COUNT(DISTINCT ?instance) AS ?totalEntities)\nWHERE {{\n{ip}\n}}"


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
    index_name: str,
    short_iri: str,
    manager: KgManager,
    variants: list[str] | None = None,
) -> str:
    label = manager.get_label(iri, index_name)
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


def get_label_and_aliases(
    iri: str,
    index_name: str,
    manager: KgManager,
) -> tuple[str | None, list[str]]:
    data = manager.try_get_data(index_name)
    if data is None:
        return None, []
    norm = manager.normalize(iri, index_name)
    key = norm[0] if norm is not None else iri
    id = data.id_from_identifier(key)
    if id is None:
        return None, []
    return data.main_field(id) or data.field(id, 0), data.fields(id) or []


def bracketed(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    return f"[ {' '.join(items)} ]"


def render_target(target: Target, manager: KgManager) -> str:
    if isinstance(target, TargetLiteral):
        return target.datatype
    if isinstance(target, TargetClass):
        return resolve_label(
            target.iri,
            "entities",
            target.short_iri,
            manager,
            variants=target.variants or None,
        )
    return "IRI"


def select_properties(
    profile: ClassProfile,
    shape_config: ShapeConfig,
    dense: bool,
) -> tuple[list[PropertyProfile], int, int]:
    cap = (
        shape_config.dense_max_properties_per_class
        if dense
        else shape_config.max_properties_per_class
    )
    candidates = profile.properties[:cap]
    omitted = max(0, len(profile.properties) - cap)

    kept: list[PropertyProfile] = []
    filtered = 0
    for prop in candidates:
        if profile.total_entities > 0:
            coverage = prop.entity_count / profile.total_entities
            if coverage < shape_config.min_property_share:
                filtered += 1
                continue
        kept.append(prop)

    return kept, omitted, filtered


def cardinality_tag_for_property(prop: PropertyProfile, total_entities: int) -> str:
    if total_entities <= 0:
        return ""
    coverage = prop.entity_count / total_entities
    avg_vals = prop.triple_count / prop.entity_count if prop.entity_count > 0 else 1.0
    return cardinality_tag(coverage, avg_vals)


def emit_pseudo_shex(
    profile: ClassProfile,
    manager: KgManager,
    shape_config: ShapeConfig,
    dense: bool = False,
) -> str:
    header = resolve_label(profile.iri, "entities", profile.short_iri, manager)
    lines = [f"{header} {{"]

    kept, omitted, filtered = select_properties(profile, shape_config, dense)

    for prop in kept:
        prop_str = resolve_label(
            prop.iri,
            "properties",
            prop.short_iri,
            manager,
            variants=prop.variants or None,
        )
        tag = cardinality_tag_for_property(prop, profile.total_entities)
        tag_suffix = f" {tag}" if tag else ""

        if prop.targets:
            value_str = bracketed([render_target(t, manager) for t in prop.targets])
        else:
            value_str = "IRI"
        lines.append(f"  {prop_str} {value_str}{tag_suffix} ;")

    if omitted or filtered:
        parts = []
        if omitted:
            parts.append(f"{omitted:,} omitted (cap)")
        if filtered:
            parts.append(f"{filtered:,} filtered (low coverage)")
        lines.append(f"  # ... {', '.join(parts)}")

    lines.append("}")

    return "\n".join(lines)


def collect_iris(
    profile: ClassProfile,
    manager: KgManager,
    shape_config: ShapeConfig,
    dense: bool = False,
) -> list[str]:
    iris: list[str] = []
    prop_norm = manager.get_normalizer("properties")
    ent_norm = manager.get_normalizer("entities")
    kept, _, _ = select_properties(profile, shape_config, dense)
    for prop in kept:
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


def group_by_normalized(
    freq_map: dict[str, dict],
    lit_counts: dict[str, dict[str, int]],
    range_counts: dict[str, dict[str, int]],
    manager: KgManager,
) -> tuple[
    dict[str, dict],
    dict[str, dict[str, int]],
    dict[str, dict[str, int]],
    dict[str, list[str]],
    dict[str, dict[str, list[str]]],
]:
    """Group property IRIs (and target-class IRIs) by their normalized form.

    Returns regrouped freq_map / literal-count map / range-count map keyed on
    the normalized property IRI, plus a prop_variants map
    (normalized prop iri -> ordered variant short names) and a target_variants
    map (normalized prop iri -> normalized target iri -> ordered variants).
    Counts are summed when merging variants.
    """
    prop_normalizer = manager.get_normalizer("properties")
    ent_normalizer = manager.get_normalizer("entities")

    def prop_key(iri: str) -> tuple[str, str | None]:
        norm = prop_normalizer.normalize(iri)
        if isinstance(norm, tuple) and len(norm) == 2:
            return norm[0], norm[1]
        return iri, None

    def ent_key(iri: str) -> tuple[str, str | None]:
        norm = ent_normalizer.normalize(iri)
        if isinstance(norm, tuple) and len(norm) == 2:
            return norm[0], norm[1]
        return iri, None

    prop_default = prop_normalizer.default_variants() or []
    ent_default = ent_normalizer.default_variants() or []

    def sort_variants(vs: list[str], order: list[str]) -> list[str]:
        if not order:
            return vs
        order_idx = {v: i for i, v in enumerate(order)}
        return sorted(vs, key=lambda v: order_idx.get(v, len(order)))

    new_freq: dict[str, dict] = {}
    new_lit: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    new_range: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    prop_variants: dict[str, list[str]] = defaultdict(list)
    target_variants: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for p_iri, freq in freq_map.items():
        k, v = prop_key(p_iri)
        if k not in new_freq:
            new_freq[k] = {"triple_count": 0, "entity_count": 0}
        new_freq[k]["triple_count"] += freq.get("triple_count", 0)
        new_freq[k]["entity_count"] += freq.get("entity_count", 0)
        if v is not None and v not in prop_variants[k]:
            prop_variants[k].append(v)

    for p_iri, dtype_counts in lit_counts.items():
        k, _ = prop_key(p_iri)
        for dt, c in dtype_counts.items():
            new_lit[k][dt] += c

    for p_iri, tc_counts in range_counts.items():
        k, _ = prop_key(p_iri)
        for t_iri, c in tc_counts.items():
            tk, tv = ent_key(t_iri)
            new_range[k][tk] += c
            if tv is not None and tv not in target_variants[k][tk]:
                target_variants[k][tk].append(tv)

    # canonicalize variant order
    for k in list(prop_variants.keys()):
        prop_variants[k] = sort_variants(prop_variants[k], prop_default)
    for k in list(target_variants.keys()):
        for tk in list(target_variants[k].keys()):
            target_variants[k][tk] = sort_variants(target_variants[k][tk], ent_default)

    return new_freq, new_lit, new_range, prop_variants, target_variants


def assemble_profile(
    class_iri: str,
    freq_map: dict[str, dict],
    lit_counts: dict[str, dict[str, int]],
    range_counts: dict[str, dict[str, int]],
    total_entities: int,
    shape_config: ShapeConfig,
    manager: KgManager,
) -> ClassProfile:
    freq_map, lit_counts, range_counts, prop_variants, target_variants = (
        group_by_normalized(freq_map, lit_counts, range_counts, manager)
    )

    common_props = sorted(
        freq_map.items(),
        key=lambda item: (
            property_rank(item[0], manager),
            -item[1].get("triple_count", 0),
        ),
    )

    properties: list[PropertyProfile] = []
    for p_iri, freq in common_props:
        triple_count = freq.get("triple_count", 0)
        targets: list[Target] = []

        for dt, c in lit_counts.get(p_iri, {}).items():
            targets.append(TargetLiteral(datatype=dt, triple_count=c))

        for t_iri, c in range_counts.get(p_iri, {}).items():
            targets.append(
                TargetClass(
                    iri=t_iri,
                    short_iri=manager.format_iri(t_iri, wrap=True),
                    variants=target_variants.get(p_iri, {}).get(t_iri, []),
                    triple_count=c,
                )
            )

        # gap between total triples and the typed buckets is "untyped IRI" traffic.
        # truncated/approximate count queries can produce small spurious gaps, so
        # require the gap to be both positive and a meaningful share of total.
        typed_sum = sum(t.triple_count for t in targets)
        gap = triple_count - typed_sum
        if (
            gap > 0
            and triple_count > 0
            and gap / triple_count >= shape_config.min_target_share
        ):
            targets.append(TargetIri(triple_count=gap))

        targets.sort(key=lambda t: t.triple_count, reverse=True)
        targets = targets[: shape_config.max_targets_per_property]

        properties.append(
            PropertyProfile(
                iri=p_iri,
                short_iri=manager.format_iri(p_iri, wrap=True),
                triple_count=triple_count,
                entity_count=freq.get("entity_count", 0),
                variants=prop_variants.get(p_iri, []),
                targets=targets,
            )
        )

    return ClassProfile(
        iri=class_iri,
        short_iri=manager.format_iri(class_iri, wrap=True),
        total_entities=total_entities,
        properties=properties,
    )


def compute_shape(
    class_iri: str,
    pattern: str,
    manager: KgManager,
    shape_config: ShapeConfig | None = None,
) -> ClassProfile:
    if shape_config is None:
        shape_config = ShapeConfig()

    def run(query: str) -> list[dict]:
        result = manager.execute_sparql(
            query,
            shape_config.request_timeout,
            shape_config.read_timeout,
            sparql_result_max_rows=shape_config.sparql_result_max_rows,
        )
        assert isinstance(result, SelectResult), "Expected SELECT query result"
        return [
            {var: row[var].value for var in result.variables} for row in result.rows()
        ]

    try:
        freq_rows = run(build_per_class_property_frequency_query(pattern, class_iri))
        lit_rows = run(build_per_class_literal_profile_query(pattern, class_iri))
        range_rows = run(build_per_class_range_profile_query(pattern, class_iri))
        total_rows = run(build_per_class_total_query(pattern, class_iri))
    except Exception as e:
        raise RuntimeError(
            f"One of the queries for computing the shape of '{class_iri}' failed:\n{e}"
        )

    total = int(total_rows[0]["totalEntities"]) if total_rows else 0

    freq_map: dict[str, dict] = {}
    for row in freq_rows:
        freq_map[row["p"]] = {
            "triple_count": int(row["tripleCount"]),
            "entity_count": int(row["entityCount"]),
        }

    lit_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in lit_rows:
        dt = manager.format_iri(row["datatype"], wrap=True)
        lit_counts[row["p"]][dt] += int(row["count"])

    range_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in range_rows:
        range_counts[row["p"]][row["targetClass"]] += int(row["count"])

    return assemble_profile(
        class_iri,
        freq_map,
        lit_counts,
        range_counts,
        total,
        shape_config,
        manager,
    )


def build_shapes(
    pattern: str,
    shapes_dir: str,
    manager: KgManager,
    shape_config: ShapeConfig = ShapeConfig(),
    max_classes: int = 500,
    log_level: str | int | None = None,
    request_timeout: float | tuple[float, float] | None = None,
    read_timeout: float | None = None,
) -> tuple[list[ShapeSample], int]:
    logger = get_logger("GRASP SHAPES BUILD", log_level)

    def run(query: str) -> list[dict[str, Any]]:
        logger.debug(f"Running query:\n{query}")
        result = manager.execute_sparql(
            query,
            request_timeout,
            read_timeout,
            sparql_result_max_rows=shape_config.sparql_result_max_rows,
        )
        assert isinstance(result, SelectResult), "Expected SELECT query result"
        return [
            {var: row[var].value for var in result.variables} for row in result.rows()
        ]

    logger.info("Discovering classes")
    try:
        total_rows = run(build_total_entities_query(pattern))
    except Exception as e:
        raise ValueError(
            f"Pattern validation failed: could not query classes ({e}). "
            "Check that the pattern correctly connects instances to class nodes."
        ) from e
    if not total_rows:
        raise ValueError(
            "Pattern validation failed: no classes found. "
            "Check that the pattern correctly connects instances to class nodes."
        )
    total_entities = {row["class"]: int(row["totalEntities"]) for row in total_rows}

    all_class_iris = list(total_entities.keys())[:max_classes]
    logger.info(f"Building shapes for {len(all_class_iris)} classes")

    samples = []
    skipped = 0
    for c_iri in tqdm(all_class_iris, desc="Profiling classes"):
        try:
            freq_rows = run(build_per_class_property_frequency_query(pattern, c_iri))
            lit_rows = run(build_per_class_literal_profile_query(pattern, c_iri))
            range_rows = run(build_per_class_range_profile_query(pattern, c_iri))
        except Exception as e:
            logger.warning(
                f"Skipping {c_iri}: profiling query failed ({e}). "
                "To retry, increase timeouts in the shape config and re-run with --overwrite."
            )
            skipped += 1
            continue

        freq_map: dict[str, dict] = {}
        for row in freq_rows:
            freq_map[row["p"]] = {
                "triple_count": int(row["tripleCount"]),
                "entity_count": int(row["entityCount"]),
            }

        lit_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for row in lit_rows:
            dt = manager.format_iri(row["datatype"], wrap=True)
            lit_counts[row["p"]][dt] += int(row["count"])

        range_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for row in range_rows:
            range_counts[row["p"]][row["targetClass"]] += int(row["count"])

        profile = assemble_profile(
            c_iri,
            freq_map,
            lit_counts,
            range_counts,
            total_entities[c_iri],
            shape_config,
            manager,
        )
        class_label, class_aliases = get_label_and_aliases(c_iri, "entities", manager)
        if class_label:
            derived = derive_label_from_iri(c_iri, manager.prefixes)
            if derived and derived.lower() == class_label.lower():
                class_label = None

        samples.append(
            ShapeSample(
                iri=c_iri,
                short_iri=profile.short_iri,
                profile=profile,
                label=class_label,
                aliases=class_aliases or [],
            )
        )

    if skipped:
        logger.warning(f"Skipped {skipped:,} class(es) due to query failures")

    os.makedirs(shapes_dir, exist_ok=True)
    pattern_file = os.path.join(shapes_dir, "pattern.sparql")
    dump_text(pattern, pattern_file)
    logger.info(f"Wrote {len(samples)} shapes and pattern to {shapes_dir}")
    return samples, len(total_entities)
