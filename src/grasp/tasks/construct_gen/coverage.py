import re

from grasp.manager import KgManager
from grasp.sparql.types import SelectResult

RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

_RDFS = "http://www.w3.org/2000/01/rdf-schema#"
_OWL = "http://www.w3.org/2002/07/owl#"
# predicates that only describe the source schema (its T-Box); to be skipped
SCHEMA_PREDICATES = {
    _RDFS + "domain",
    _RDFS + "range",
    _RDFS + "subClassOf",
    _RDFS + "subPropertyOf",
    _OWL + "inverseOf",
    _OWL + "equivalentClass",
    _OWL + "equivalentProperty",
    _OWL + "disjointWith",
    _OWL + "imports",
    _OWL + "versionInfo",
}

_PREFIX_RE = re.compile(r"PREFIX\s+([A-Za-z][\w.\-]*)?:\s*<([^>]*)>", re.IGNORECASE)
_FULL_IRI_RE = re.compile(r"<([^>\s]+)>")
# match a prefixed name
_PNAME_RE = re.compile(r"(?<![\w<>/#:])([A-Za-z][\w.\-]*)?:([A-Za-z0-9_\-%.]+)")
# standalone `a` is SPARQL shorthand for rdf:type
_TYPE_SHORTHAND_RE = re.compile(r"(?:^|[\s;,\[])a\s", re.MULTILINE)

_ALL_PREDICATES_SPARQL = """\
SELECT ?p (COUNT(*) AS ?count) WHERE { ?s ?p ?o } GROUP BY ?p ORDER BY DESC(?count)\
"""


def referenced_iris(sparql: str) -> set[str]:
    """
    Every IRI a single query mentions, with the query's own PREFIX
    declarations resolved.

    The whole query text is scanned, not just its WHERE clause. That keeps the
    check simple and it can only ever over-count and only when the output
    deliberately reuses source IRIs.
    """
    prefixes = {(name or ""): iri for name, iri in _PREFIX_RE.findall(sparql)}

    iris = set(_FULL_IRI_RE.findall(sparql))
    for name, local in _PNAME_RE.findall(sparql):
        base = prefixes.get(name or "")
        if base is not None:
            iris.add(base + local)

    if _TYPE_SHORTHAND_RE.search(sparql):
        iris.add(RDF_TYPE)

    return iris


def source_predicates(
    manager: KgManager,
    request_timeout: float | tuple[float, float] | None = None,
    read_timeout: float | None = None,
) -> list[tuple[str, int]]:
    """Every predicate in the source graph with its triple count, most frequent first."""
    result = manager.execute_sparql(
        _ALL_PREDICATES_SPARQL,
        request_timeout=request_timeout,
        read_timeout=read_timeout,
    )
    assert isinstance(result, SelectResult), "expected a SELECT result"

    predicates = []
    for row in result.data:
        if not row or "p" not in row:
            continue
        iri = row["p"]["value"]
        try:
            count = int(row["count"]["value"])
        except (KeyError, TypeError, ValueError):
            count = 0
        predicates.append((iri, count))
    return predicates


def compute_source_coverage(
    manager: KgManager,
    queries: list[dict],
    request_timeout: float | tuple[float, float] | None = None,
    read_timeout: float | None = None,
) -> dict:
    """
    Check which of the source graph's predicates the submitted queries mention.

    Returns {"total", "covered", "uncovered": [(iri, count), ...]}, or
    {"error": str} if the source predicates could not be listed.
    """
    try:
        predicates = source_predicates(manager, request_timeout, read_timeout)
    except Exception as e:
        return {"error": f"could not list source predicates: {e}"}

    predicates = [
        (iri, count) for iri, count in predicates if iri not in SCHEMA_PREDICATES
    ]
    if not predicates:
        return {"error": "the source knowledge graph reported no predicates"}

    referenced: set[str] = set()
    for query in queries:
        referenced |= referenced_iris(query.get("sparql", ""))

    uncovered = [(iri, count) for iri, count in predicates if iri not in referenced]
    return {
        "total": len(predicates),
        "covered": len(predicates) - len(uncovered),
        "uncovered": uncovered,
    }


def format_source_coverage(coverage: dict, max_listed: int = 50) -> str:
    """Render the coverage result as a section for the reviewer's prompt."""
    if "error" in coverage:
        return (
            "## Computed source coverage\n\n"
            f"Unavailable ({coverage['error']}). "
            "Verify SOURCE COVERAGE by querying the knowledge graph yourself."
        )

    total = coverage["total"]
    covered = coverage["covered"]
    uncovered = coverage["uncovered"]
    percent = 100 * covered / total if total else 0.0

    lines = [
        "## Computed source coverage",
        "",
        f"{covered} of {total} source predicates ({percent:.1f}%) are referenced "
        "by these queries.",
        "",
        "A predicate counts as referenced if its IRI appears anywhere in a query, "
        "so this is an upper bound: a predicate can be referenced and still have "
        "its values dropped.",
        "",
    ]

    if not uncovered:
        lines.append("Every source predicate is referenced by at least one query.")
        return "\n".join(lines)

    shown = uncovered[:max_listed]
    lines.append(
        f"The following {len(uncovered)} source predicate(s) are referenced by NO "
        "query, ordered by triple count:"
    )
    lines.append("")
    lines.extend(f"- <{iri}> ({count} triples)" for iri, count in shown)
    if len(uncovered) > len(shown):
        lines.append(f"- ... and {len(uncovered) - len(shown)} more")
    lines.append("")
    lines.append(
        "Each of these carries source data that the output graph will not contain. "
        "Report them under SOURCE COVERAGE, naming the ones that matter and what "
        "should be added to cover them, unless the data is genuinely represented "
        "some other way."
    )
    return "\n".join(lines)
