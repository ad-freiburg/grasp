import os
from dataclasses import dataclass, field
from threading import Lock
from typing import Literal

from cachetools import LRUCache
from pydantic import BaseModel, Field
from universal_ml_utils.io import (
    dump_json,
    dump_jsonl,
    load_json,
    load_jsonl,
    load_text,
)


class TargetClass(BaseModel):
    type: Literal["class"] = "class"
    iri: str
    short_iri: str
    variants: list[str] = Field(default_factory=list)
    triple_count: int = 0


class TargetLiteral(BaseModel):
    type: Literal["literal"] = "literal"
    datatype: str
    triple_count: int = 0


class TargetIri(BaseModel):
    type: Literal["iri"] = "iri"
    triple_count: int = 0


class TargetBnode(BaseModel):
    type: Literal["bnode"] = "bnode"
    triple_count: int = 0


Target = TargetClass | TargetLiteral | TargetIri | TargetBnode


class PropertyProfile(BaseModel):
    iri: str
    short_iri: str
    triple_count: int = 0
    entity_count: int = 0
    variants: list[str] = Field(default_factory=list)
    targets: list[Target] = Field(default_factory=list)


class ClassProfile(BaseModel):
    iri: str
    short_iri: str
    total_entities: int = 0
    # instances profiled if the class was sampled, 0 if profiled exhaustively
    profiled_entities: int = 0
    properties: list[PropertyProfile] = Field(default_factory=list)
    # inverse edges: properties for which instances of this class are the *object*.
    # reuses PropertyProfile, where `targets` holds the *source* classes rather
    # than the target ones, and `entity_count` counts distinct receiving instances.
    incoming: list[PropertyProfile] = Field(default_factory=list)

    # coverage counts are over the profiled instances, not the whole extension
    @property
    def share_base(self) -> int:
        return self.profiled_entities or self.total_entities


# precomputed profiles of the largest classes, computed on request for others
@dataclass
class Shapes:
    instance_pattern: str | None = None
    schema_pattern: str | None = None
    description: str | None = None
    profiles: dict[str, ClassProfile] = field(default_factory=dict)
    total_classes: int | None = None
    cache: LRUCache = field(default_factory=lambda: LRUCache(maxsize=256))
    lock: Lock = field(default_factory=Lock)

    def get(self, iri: str) -> ClassProfile | None:
        if iri in self.profiles:
            return self.profiles[iri]
        with self.lock:
            return self.cache.get(iri)

    def put(self, profile: ClassProfile) -> None:
        with self.lock:
            self.cache[profile.iri] = profile


def save_profiles(
    profiles: list[ClassProfile],
    shapes_dir: str,
    total_classes: int | None = None,
) -> None:
    dump_jsonl(
        (p.model_dump() for p in profiles),
        os.path.join(shapes_dir, "profiles.jsonl"),
    )
    info: dict = {"profiled_classes": len(profiles)}
    if total_classes is not None:
        info["total_classes"] = total_classes
    dump_json(info, os.path.join(shapes_dir, "info.json"))


def load_profiles(shapes_dir: str) -> tuple[list[ClassProfile], int | None]:
    path = os.path.join(shapes_dir, "profiles.jsonl")
    if not os.path.exists(path):
        return [], None

    profiles = [ClassProfile.model_validate(p) for p in load_jsonl(path)]
    info_path = os.path.join(shapes_dir, "info.json")
    total = (
        load_json(info_path).get("total_classes") if os.path.exists(info_path) else None
    )
    return profiles, total


def load_setup_description(shapes_dir: str) -> str | None:
    setup_path = os.path.join(shapes_dir, "setup.json")
    if not os.path.exists(setup_path):
        return None
    return load_json(setup_path).get("description")


def load_setup_patterns(shapes_dir: str) -> tuple[str | None, str | None]:
    def read(name: str) -> str | None:
        path = os.path.join(shapes_dir, name)
        return load_text(path) if os.path.exists(path) else None

    return read("instance_pattern.sparql"), read("schema_pattern.sparql")


def load_shapes(shapes_dir: str, cache_size: int = 256) -> Shapes | None:
    instance_pattern, schema_pattern = load_setup_patterns(shapes_dir)
    if instance_pattern is None and schema_pattern is None:
        return None

    profiles, total_classes = load_profiles(shapes_dir)
    return Shapes(
        instance_pattern=instance_pattern,
        schema_pattern=schema_pattern,
        description=load_setup_description(shapes_dir),
        profiles={p.iri: p for p in profiles},
        total_classes=total_classes,
        cache=LRUCache(maxsize=cache_size),
    )
