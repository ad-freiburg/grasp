from types import SimpleNamespace

import pytest

from grasp.manager.utils import find_longest_prefix, get_common_sparql_prefixes
from grasp.model import ToolCall
from grasp.provenance.trace import call_provenance, note_iris, trace_provenance
from grasp.sparql.utils import load_iri_and_literal_parser, load_sparql_parser

EX = "http://example.org/kg#"
REMOTE = "http://example.org/remote/"


@pytest.fixture(scope="module")
def manager():
    prefixes = {**get_common_sparql_prefixes(), "ex": EX}
    return SimpleNamespace(
        kg="example",
        prefixes=prefixes,
        sparql_parser=load_sparql_parser(),
        iri_literal_parser=load_iri_and_literal_parser(),
        find_longest_prefix=lambda iri: find_longest_prefix(iri, prefixes),
        normalize=lambda iri, index: None,
        get_iri_label=lambda iri: None,
        format_iri=lambda iri: iri,
    )


SERVICE_QUERY = f"""PREFIX remote: <{REMOTE}>
SELECT ?value WHERE {{
  ex:place_1 geo:hasGeometry/geo:asWKT ?wkt .
  SERVICE remote: {{
    _:config remote:endpoint <http://example.org/raster> ;
      remote:input ?wkt ; remote:output ?value .
  }}
}}"""


NOTES = [
    f"The raster service is called with PREFIX remote: <{REMOTE}> and "
    "SERVICE remote: { _:config remote:endpoint <http://example.org/raster> }.",
    "Places link via ex:teilfläche; links like https://en.wikipedia.org/wiki/<Title>.",
    "Use rdfs:label for labels, e.g. ex:place_<nr>, and remote:output for values.",
    "PREFIXES foo: <http://example.org/foo/> are no declaration, foo:bar is unknown.",
]


def test_note_iris(manager):
    iris = note_iris(manager, NOTES)
    assert REMOTE in iris
    assert iris[REMOTE + "endpoint"] == NOTES[0]
    assert "http://example.org/raster" in iris
    assert iris[EX + "teilfläche"] == NOTES[1]
    # prefixes declared in one note hold for the others
    assert iris[REMOTE + "output"] == NOTES[2]
    # no relative IRIs, undefined prefixes or common vocabulary
    assert "Title" not in iris
    assert not any(
        iri.startswith(("Places", "https:", "http://example.org/foo/b")) for iri in iris
    )
    assert "http://www.w3.org/2000/01/rdf-schema#label" not in iris


def table_call(id: str, sparql: str, shown: list[str]) -> dict:
    return {
        "id": id,
        "name": "execute",
        "args": {"kg": "example", "sparql": sparql},
        "result": {
            "type": "table",
            "rows": [{"s": {"type": "uri", "value": iri}} for iri in shown],
        },
        "new_known": shown,
    }


def assistant(*calls: dict) -> dict:
    return {"role": "assistant", "content": {"tool_calls": list(calls)}}


def test_trace_status_from_notes(manager):
    sparql = (
        f"PREFIX remote: <{REMOTE}>\n"
        "SELECT ?v WHERE { ex:place_1 ex:x ?v . ?v remote:endpoint ?e }"
    )
    call = table_call("c1", "SELECT * WHERE { ?s ?p ?o }", [EX + "place_1"])
    trace = trace_provenance(manager, sparql, [assistant(call)], NOTES)
    assert trace is not None
    items = {item["identifier"]: item for item in trace["items"]}
    assert items[EX + "place_1"]["status"] == "retrieved"
    assert items[EX + "place_1"]["shown_by"]["step"] == 1
    assert items[REMOTE + "endpoint"]["status"] == "noted"
    assert items[REMOTE + "endpoint"]["note"] == NOTES[0]
    assert items[EX + "x"]["status"] == "unverified"


def test_trace_shown_by_example(manager):
    sparql = f"PREFIX remote: <{REMOTE}>\nSELECT ?e WHERE {{ ?x remote:endpoint ?e }}"
    call = {
        "id": "c1",
        "name": "search_example",
        "args": {"kg": "example", "query": "raster"},
        "result": {
            "type": "sparql_qa_examples",
            "examples": [
                {"question": "Other?", "sparql": "SELECT * WHERE { ?s ?p ?o }"},
                {"question": "Raster?", "sparql": SERVICE_QUERY},
            ],
        },
        "new_known": [REMOTE + "endpoint"],
    }
    trace = trace_provenance(manager, sparql, [assistant(call)], [])
    assert trace is not None
    item = next(i for i in trace["items"] if i["identifier"] == REMOTE + "endpoint")
    assert item["status"] == "retrieved"
    assert item["shown_by"]["result_paths"] == ["examples.1"]
    shown = item["shown_by"]["shown"]
    assert shown["type"] == "example" and shown["question"] == "Raster?"
    assert "remote:endpoint" in shown["line"]


def test_trace_older_traces(manager):
    call = table_call("c1", "SELECT * WHERE { ?s ?p ?o }", [EX + "place_1"])
    del call["new_known"]
    sparql = "SELECT ?v WHERE { ex:place_1 ex:x ?v }"
    assert trace_provenance(manager, sparql, [assistant(call)], []) is None


def test_call_provenance(manager):
    # ex:x is used before it is shown, ex:place_1 shown before it is used
    first = table_call("c1", "SELECT * WHERE { ?s ex:x ?o }", [EX + "place_1"])
    second = table_call("c2", "SELECT * WHERE { ?s ?p ?o }", [EX + "x"])
    current = ToolCall(
        id="c3",
        name="execute",
        args={
            "kg": "example",
            "sparql": "SELECT ?v WHERE { ex:place_1 ex:x ?v . ?v ex:y ?w }",
        },
    )
    messages = [assistant(first, second, current.model_dump())]

    provenance = call_provenance([manager], messages, current, {}, [])
    assert provenance is not None
    items = {item["identifier"]: item for item in provenance["items"]}
    assert items[EX + "place_1"]["status"] == "retrieved"
    assert items[EX + "place_1"]["shown_by"]["id"] == "c1"
    assert items[EX + "place_1"]["spans"] == [[18, 28]]
    assert items[EX + "x"]["status"] == "verified"
    assert items[EX + "x"]["first_used_by"]["id"] == "c1"
    assert items[EX + "y"]["status"] == "unverified"


def test_call_provenance_args(manager):
    first = table_call("c1", "SELECT * WHERE { ?s ?p ?o }", [EX + "place_1"])
    current = ToolCall(
        id="c2",
        name="list",
        args={"kg": "example", "subject": "ex:place_1", "property": "ex:teilfläche"},
    )
    messages = [assistant(first, current.model_dump())]

    provenance = call_provenance([manager], messages, current, {"example": NOTES}, [])
    assert provenance is not None
    items = {item["identifier"]: item for item in provenance["items"]}
    assert items[EX + "place_1"]["args"] == ["subject"]
    assert items[EX + "place_1"]["status"] == "retrieved"
    assert items[EX + "teilfläche"]["args"] == ["property"]
    assert items[EX + "teilfläche"]["status"] == "noted"


def test_call_provenance_none(manager):
    answer = ToolCall(
        id="c1",
        name="answer",
        args={"kg": "example", "sparql": "SELECT ?v WHERE { ex:place_1 ex:x ?v }"},
    )
    messages = [assistant(answer.model_dump())]
    assert call_provenance([manager], messages, answer, {}, []) is None

    search = ToolCall(id="c2", name="search", args={"kg": "example", "query": "x"})
    messages = [assistant(search.model_dump())]
    assert call_provenance([manager], messages, search, {}, []) is None
