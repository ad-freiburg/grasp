from types import SimpleNamespace

import pytest

from grasp.functions import (
    check_known,
    order_triples,
    query_iris,
    row_iris,
    update_known_from_index_iris,
    update_known_from_selections,
)
from grasp.manager.normalizer import Normalizer
from grasp.manager.utils import find_longest_prefix, get_common_sparql_prefixes
from grasp.sparql.types import KgItem, Binding, TermKind, Selection
from grasp.sparql.utils import load_iri_and_literal_parser, load_sparql_parser
from grasp.utils import FunctionCallException


class VariantNormalizer(Normalizer):
    def denormalize(self, iri: str, variant: str | None) -> str | None:
        return f"{iri}/{variant}"


class PropertyNormalizer(Normalizer):
    def normalize(self, iri: str) -> tuple[str, str | None] | None:
        if not iri.startswith("ex:direct/"):
            return None
        return "ex:" + iri.removeprefix("ex:direct/"), "direct"


class Manager:
    def get_normalizer(self, name: str) -> Normalizer:
        if name == "classes":
            return VariantNormalizer()
        elif name == "properties":
            return PropertyNormalizer()
        return Normalizer()


def test_known_from_selections():
    selections = [
        Selection(KgItem(identifier="ex:e"), TermKind.ENTITY),
        Selection(KgItem(identifier="ex:c", variants=["v"]), TermKind.CLASS),
        Selection(KgItem(identifier="ex:p"), TermKind.PROPERTY),
    ]
    known = set()
    update_known_from_selections(known, selections, Manager())  # type: ignore
    assert known == {"ex:e", "ex:c", "ex:c/v", "ex:p"}


def test_known_from_rows_uses_all_indices():
    rows = (
        {"p": Binding("uri", "ex:direct/p"), "v": Binding("literal", "x")}
        for _ in range(2)
    )
    known = set()
    update_known_from_index_iris(known, row_iris(rows), Manager())  # type: ignore
    assert known == {"ex:direct/p", "ex:p"}


class Data:
    def __init__(self, identifiers: list[str]):
        self.ids = {identifier: i for i, identifier in enumerate(identifiers)}

    def id_from_identifier(self, identifier: str) -> int | None:
        return self.ids.get(identifier)


class TripleManager:
    # ids are popularity ranks, the classes are not in the entity index
    data = {
        "properties": Data(["ex:type", "ex:name", "ex:knows"]),
        "entities": Data(["ex:a", "ex:b", "ex:x"]),
        "classes": Data(["ex:C1", "ex:C2"]),
    }

    def normalize(self, iri: str, index_name: str):
        return iri, None

    def try_get_data(self, index_name: str) -> Data | None:
        return self.data.get(index_name)

    def check_identifier(self, identifier: str, index_name: str) -> bool:
        return identifier in self.data[index_name].ids

    def get_normalizer(self, name: str) -> Normalizer:
        return Normalizer()


def test_all_classes_of_an_entity_come_first():
    triples = [
        ("ex:x", "ex:knows", "ex:a"),
        ("ex:x", "ex:knows", "ex:b"),
        ("ex:x", "ex:name", "x"),
        ("ex:x", "ex:type", "ex:C2"),
        ("ex:x", "ex:type", "ex:C1"),
    ]
    rows = [
        {
            "s": Binding("uri", s),
            "p": Binding("uri", p),
            "o": Binding("literal" if o == "x" else "uri", o),
        }
        for s, p, o in triples
    ]
    order = order_triples(rows, TripleManager())  # type: ignore
    # types by class popularity, then the other properties, then repeats
    assert order == [4, 3, 2, 0, 1]


WALD = "https://wald.cs.uni-freiburg.de/ontology#"
QLPROXY = "https://qlever.cs.uni-freiburg.de/qlproxy/"


@pytest.fixture(scope="module")
def wald_manager():
    prefixes = {**get_common_sparql_prefixes(), "wald": WALD}
    return SimpleNamespace(
        kg="wald",
        prefixes=prefixes,
        sparql_parser=load_sparql_parser(),
        iri_literal_parser=load_iri_and_literal_parser(),
        find_longest_prefix=lambda iri: find_longest_prefix(iri, prefixes),
        format_iri=lambda iri: iri,
    )


SERVICE_QUERY = f"""PREFIX qlproxy: <{QLPROXY}>
SELECT ?value WHERE {{
  wald:Betrieb_1 geo:hasGeometry/geo:asWKT ?wkt .
  SERVICE qlproxy: {{
    _:config qlproxy:endpoint <https://qlever.dev/api/wald-tiff> ;
      qlproxy:input-filter_wkt ?wkt ; qlproxy:output-value ?value .
  }}
}}"""


def test_query_iris_resolves_query_prefixes(wald_manager):
    iris = query_iris(wald_manager, SERVICE_QUERY)
    assert iris == {
        WALD + "Betrieb_1",
        QLPROXY,
        QLPROXY + "endpoint",
        "https://qlever.dev/api/wald-tiff",
        QLPROXY + "input-filter_wkt",
        QLPROXY + "output-value",
    }


def test_query_iris_skips_service(wald_manager):
    iris = query_iris(wald_manager, SERVICE_QUERY, skip_service=True)
    assert iris == {WALD + "Betrieb_1"}


def test_check_known_ignores_service(wald_manager):
    check_known(wald_manager, SERVICE_QUERY, {WALD + "Betrieb_1"})
    with pytest.raises(FunctionCallException, match="without being known"):
        check_known(wald_manager, SERVICE_QUERY, set())


def test_check_known_undefined_prefix(wald_manager):
    sparql = "SELECT ?x WHERE { ?x foo:bar wald:Betrieb_1 . ?x foo:baz ?y }"
    with pytest.raises(FunctionCallException) as exc:
        check_known(wald_manager, sparql, {WALD + "Betrieb_1"})
    message = str(exc.value)
    assert "foo: (e.g. foo:bar)" in message
    assert "neither declared in the query nor common or wald prefixes" in message
