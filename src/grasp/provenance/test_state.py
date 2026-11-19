from grasp.model import ToolCall
from grasp.provenance.state import Provenance
from grasp.provenance.test_trace import EX, NOTES, REMOTE, SERVICE_QUERY


def table_call(id: str, sparql: str, shown: list[str]) -> dict:
    return {
        "id": id,
        "name": "execute",
        "args": {"kg": "example", "sparql": sparql},
        "result": {
            "type": "table",
            "rows": [{"s": {"type": "uri", "value": iri}} for iri in shown],
        },
    }


def run(
    manager,
    provenance: Provenance,
    calls: list[tuple[dict, list[str]]],
) -> tuple[list[dict], list[dict | None]]:
    # each call with the items its result made known, as the functions would
    messages = []
    recorded = []
    for call, shown in calls:
        messages.append({"role": "assistant", "content": {"tool_calls": [call]}})
        provenance.known.update(shown)
        tool_call = ToolCall(id=call["id"], name=call["name"], args=call["args"])
        recorded.append(provenance.record([manager], tool_call, messages))
    return messages, recorded


def by_iri(provenance: dict | None) -> dict[str, dict]:
    assert provenance is not None
    return {item["identifier"]: item for item in provenance["items"]}


def test_query_status_from_notes(manager):
    provenance = Provenance({"example": NOTES}, [])
    call = table_call("c1", "SELECT * WHERE { ?s ?p ?o }", [EX + "place_1"])
    messages, _ = run(manager, provenance, [(call, [EX + "place_1"])])

    sparql = (
        f"PREFIX remote: <{REMOTE}>\n"
        "SELECT ?v WHERE { ex:place_1 ex:x ?v . ?v remote:endpoint ?e }"
    )
    items = by_iri(provenance.query(manager, sparql, messages))
    assert items[EX + "place_1"]["status"] == "retrieved"
    assert items[EX + "place_1"]["shown_by"]["step"] == 1
    assert items[REMOTE + "endpoint"]["status"] == "noted"
    assert items[REMOTE + "endpoint"]["note"] == NOTES[0]
    assert items[EX + "x"]["status"] == "unverified"


def test_query_shown_by_example(manager):
    provenance = Provenance({}, [])
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
    }
    messages, _ = run(manager, provenance, [(call, [REMOTE + "endpoint"])])

    sparql = f"PREFIX remote: <{REMOTE}>\nSELECT ?e WHERE {{ ?x remote:endpoint ?e }}"
    item = by_iri(provenance.query(manager, sparql, messages))[REMOTE + "endpoint"]
    assert item["status"] == "retrieved"
    assert item["shown_by"]["result_paths"] == ["examples.1"]
    shown = item["shown_by"]["shown"]
    assert shown["type"] == "example" and shown["question"] == "Raster?"
    assert "remote:endpoint" in shown["line"]


def test_record_call(manager):
    # ex:x is used before it is shown, ex:place_1 shown before it is used
    provenance = Provenance({}, [])
    calls = [
        (table_call("c1", "SELECT * WHERE { ?s ex:x ?o }", []), [EX + "place_1"]),
        (table_call("c2", "SELECT * WHERE { ?s ?p ?o }", []), [EX + "x"]),
        (
            table_call("c3", "SELECT ?v WHERE { ex:place_1 ex:x ?v . ?v ex:y ?w }", []),
            [],
        ),
    ]
    _, recorded = run(manager, provenance, calls)

    assert provenance.shown == {EX + "place_1": "c1", EX + "x": "c2"}
    assert provenance.used[EX + "x"] == "c1"
    items = by_iri(recorded[2])
    assert items[EX + "place_1"]["status"] == "retrieved"
    assert items[EX + "place_1"]["shown_by"]["id"] == "c1"
    assert items[EX + "place_1"]["spans"] == [[18, 28]]
    assert items[EX + "x"]["status"] == "verified"
    assert items[EX + "x"]["first_used_by"]["id"] == "c1"
    assert items[EX + "y"]["status"] == "unverified"


def test_record_args(manager):
    provenance = Provenance({"example": NOTES}, [])
    first = table_call("c1", "SELECT * WHERE { ?s ?p ?o }", [])
    current = {
        "id": "c2",
        "name": "list",
        "args": {"kg": "example", "subject": "ex:place_1", "property": "ex:teilfläche"},
    }
    _, recorded = run(manager, provenance, [(first, [EX + "place_1"]), (current, [])])

    items = by_iri(recorded[1])
    assert items[EX + "place_1"]["args"] == ["subject"]
    assert items[EX + "place_1"]["status"] == "retrieved"
    assert items[EX + "teilfläche"]["args"] == ["property"]
    assert items[EX + "teilfläche"]["status"] == "noted"


def test_record_without_items(manager):
    provenance = Provenance({}, [])
    answer = {
        "id": "c1",
        "name": "answer",
        "args": {"kg": "example", "sparql": "SELECT ?v WHERE { ex:place_1 ex:x ?v }"},
    }
    search = {"id": "c2", "name": "search", "args": {"kg": "example", "query": "x"}}
    _, recorded = run(manager, provenance, [(answer, []), (search, [])])
    assert recorded == [None, None]
    # the query of a stopping call is the output, it uses no items
    assert provenance.used == {}


def test_state_continues(manager):
    # known items of an older conversation without their calls, and items
    # known from the setup, count as shown
    provenance = Provenance({}, [], {"known": [EX + "old"]})
    provenance.known.add(EX + "setup")
    provenance.mark_seen()
    call = table_call("c1", "SELECT * WHERE { ?s ?p ?o }", [])
    messages, _ = run(manager, provenance, [(call, [EX + "new"])])
    assert provenance.dump() == {
        "known": [EX + "new", EX + "old", EX + "setup"],
        "shown": {EX + "new": "c1"},
        "used": {},
    }

    restored = Provenance({}, [], provenance.dump())
    sparql = "SELECT ?v WHERE { ex:old ex:setup ex:new }"
    items = by_iri(restored.query(manager, sparql, messages))
    assert items[EX + "old"]["status"] == "retrieved"
    assert "shown_by" not in items[EX + "old"]
    assert items[EX + "new"]["shown_by"]["id"] == "c1"
