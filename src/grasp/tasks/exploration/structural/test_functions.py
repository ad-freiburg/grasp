from grasp.manager import KgManager
from grasp.manager.normalizer import Normalizer
from grasp.tasks.exploration.structural.functions import seed_index_name


class Manager:
    # ex:c is in both the entity and the class index
    contents = {
        "entities": {"ex:e": "entity", "ex:c": "entity"},
        "classes": {"ex:c": "class"},
        "properties": {"ex:p": "property"},
    }

    def normalize(self, iri: str, index_name: str):
        return Normalizer().normalize(iri)

    def check_identifier(self, identifier: str, index_name: str) -> bool:
        return identifier in self.contents[index_name]

    def get_label(self, identifier: str, index_name: str) -> str | None:
        return self.contents[index_name].get(identifier)


def test_seed_index_name():
    manager = Manager()
    assert seed_index_name("ex:e", manager) == "entities"  # type: ignore
    assert seed_index_name("ex:c", manager) == "classes"  # type: ignore
    assert seed_index_name("ex:p", manager) == "properties"  # type: ignore
    assert seed_index_name("ex:x", manager) == "entities"  # type: ignore


def test_iri_label_prefers_classes():
    manager = Manager()
    assert KgManager.get_iri_label(manager, "ex:c") == "class"  # type: ignore
    assert KgManager.get_iri_label(manager, "ex:p") == "property"  # type: ignore
    assert KgManager.get_iri_label(manager, "ex:x") is None  # type: ignore
