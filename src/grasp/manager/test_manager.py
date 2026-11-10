from unittest.mock import Mock

from grasp.manager import KgManager


# identifiers ending in "-v" are a variant of the identifier before it
def fake_manager(indices: dict[str, set[str]]) -> Mock:
    def normalizer(index_name: str) -> Mock:
        normalizer = Mock()
        normalizer.normalize.side_effect = lambda identifier: (
            (identifier.removesuffix("-v"), "v")
            if identifier.endswith("-v") and index_name == "entities"
            else None
        )
        return normalizer

    def data(index_name: str) -> Mock:
        data = Mock()
        data.id_from_identifier.side_effect = lambda identifier: (
            0 if identifier in indices[index_name] else None
        )
        return data

    manager = Mock()
    manager.get_normalizer.side_effect = normalizer
    manager.get_data.side_effect = data
    return manager


def test_candidate_identifiers_mapped_per_index():
    manager = fake_manager({"entities": {"e1", "c1"}, "classes": {"c1", "c2"}})
    identifiers = ["e1", "e1-v", "c1", "c2", "x"]
    assert KgManager.map_candidate_identifiers(manager, "entities", identifiers) == {
        "e1": ["v"],
        "c1": [],
    }
    assert KgManager.map_candidate_identifiers(manager, "classes", identifiers) == {
        "c1": [],
        "c2": [],
    }
