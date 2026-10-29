from grasp.sparql.types import (
    STANDARD_INDICES,
    IRI_INDICES,
    TermKind,
    Position,
    iri_indices,
)


def test_iri_indices_are_the_standard_non_literal_indices():
    assert set(IRI_INDICES) == set(STANDARD_INDICES) - {TermKind.LITERAL}


def test_iri_lookup_order():
    assert iri_indices() == [TermKind.CLASS, TermKind.ENTITY, TermKind.PROPERTY]
    assert iri_indices(Position.OBJECT) == iri_indices()
    assert iri_indices(Position.PROPERTY) == [
        TermKind.PROPERTY,
        TermKind.CLASS,
        TermKind.ENTITY,
    ]
