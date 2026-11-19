from functools import cache
from types import SimpleNamespace

from grasp.manager.utils import find_longest_prefix, get_common_sparql_prefixes
from grasp.provenance.trace import note_iris
from grasp.sparql.utils import load_iri_and_literal_parser, load_sparql_parser

EX = "http://example.org/kg#"
REMOTE = "http://example.org/remote/"


# a knowledge graph manager with the ex prefix and no indices, also used by
# the tests of the provenance state
@cache
def example_manager() -> SimpleNamespace:
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


def test_note_iris():
    manager = example_manager()
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
