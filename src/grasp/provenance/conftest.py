from types import SimpleNamespace

import pytest

from grasp.manager.utils import find_longest_prefix, get_common_sparql_prefixes
from grasp.sparql.utils import load_iri_and_literal_parser, load_sparql_parser

EX = "http://example.org/kg#"


# a knowledge graph manager with the ex prefix and no indices
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
