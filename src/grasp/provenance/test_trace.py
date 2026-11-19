from grasp.provenance.trace import note_iris

EX = "http://example.org/kg#"
REMOTE = "http://example.org/remote/"


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
