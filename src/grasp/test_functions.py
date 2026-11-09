from grasp.functions import (
    order_triples,
    row_iris,
    update_known_from_index_iris,
    update_known_from_selections,
)
from grasp.manager.normalizer import Normalizer
from grasp.sparql.types import Alternative, Binding, ObjType, Selection


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
        Selection(Alternative("ex:e"), ObjType.ENTITY),
        Selection(Alternative("ex:c", variants=["v"]), ObjType.CLASS),
        Selection(Alternative("ex:p"), ObjType.PROPERTY),
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
