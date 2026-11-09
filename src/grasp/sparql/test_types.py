from grasp.sparql.types import (
    STANDARD_INDICES,
    IRI_INDICES,
    ObjType,
    Position,
    iri_indices,
)


def test_iri_indices_are_the_standard_non_literal_indices():
    assert set(IRI_INDICES) == set(STANDARD_INDICES) - {ObjType.LITERAL}


def test_iri_lookup_order():
    assert iri_indices() == [ObjType.CLASS, ObjType.ENTITY, ObjType.PROPERTY]
    assert iri_indices(Position.OBJECT) == iri_indices()
    assert iri_indices(Position.PROPERTY) == [
        ObjType.PROPERTY,
        ObjType.CLASS,
        ObjType.ENTITY,
    ]
