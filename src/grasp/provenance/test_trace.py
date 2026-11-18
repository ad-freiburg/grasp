from types import SimpleNamespace

import pytest

from grasp.manager.utils import find_longest_prefix, get_common_sparql_prefixes
from grasp.provenance.trace import instruction_iris, trace_provenance
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


INSTRUCTION = f"""You are a question answering assistant.
- Notes: the raster service is called with PREFIX remote: <{REMOTE}> and
SERVICE remote: {{ _:config remote:endpoint <http://example.org/raster> }}.
- Places link via ex:teilfläche; links like https://en.wikipedia.org/wiki/<Title>.
- Use rdfs:label for labels, e.g. ex:place_<nr>."""


def test_instruction_iris(manager):
    iris = instruction_iris(manager, [{"role": "system", "content": INSTRUCTION}])
    assert REMOTE in iris
    assert REMOTE + "endpoint" in iris
    assert "http://example.org/raster" in iris
    assert EX + "teilfläche" in iris
    assert iris[EX + "teilfläche"].startswith("- Places link")
    # no relative IRIs, undefined prefixes or common vocabulary
    assert "Title" not in iris
    assert not any(iri.startswith("Notes") for iri in iris)
    assert "http://www.w3.org/2000/01/rdf-schema#label" not in iris


def test_instruction_iris_only_system(manager):
    assert instruction_iris(manager, [{"role": "user", "content": INSTRUCTION}]) == {}


def test_trace_status_from_instruction(manager):
    sparql = (
        f"PREFIX remote: <{REMOTE}>\n"
        "SELECT ?v WHERE { ex:place_1 ex:x ?v . ?v remote:endpoint ?e }"
    )
    call = {
        "id": "c1",
        "name": "execute",
        "args": {"kg": "example", "sparql": "SELECT * WHERE { ?s ?p ?o }"},
        "result": {
            "type": "table",
            "rows": [{"s": {"type": "uri", "value": EX + "place_1"}}],
        },
        "new_known": [EX + "place_1"],
    }
    messages = [
        {"role": "system", "content": INSTRUCTION},
        {"role": "assistant", "content": {"tool_calls": [call]}},
    ]
    trace = trace_provenance(manager, sparql, messages)
    assert trace is not None
    status = {item["identifier"]: item["status"] for item in trace["items"]}
    assert status[EX + "place_1"] == "retrieved"
    assert status[REMOTE + "endpoint"] == "instructed"
    assert status[EX + "x"] == "unverified"
    item = next(i for i in trace["items"] if i["identifier"] == REMOTE + "endpoint")
    assert "remote:endpoint" in item["instruction"]["text"]


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
    messages = [{"role": "assistant", "content": {"tool_calls": [call]}}]
    trace = trace_provenance(manager, sparql, messages)
    assert trace is not None
    item = next(i for i in trace["items"] if i["identifier"] == REMOTE + "endpoint")
    assert item["status"] == "retrieved"
    assert item["shown_by"]["result_paths"] == ["examples.1"]
    shown = item["shown_by"]["shown"]
    assert shown["type"] == "example" and shown["question"] == "Raster?"
    assert "remote:endpoint" in shown["line"]
