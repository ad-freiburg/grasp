from unittest.mock import Mock

from grasp.build.shapes import (
    ClassMaps,
    ClassProfile,
    DirectedMaps,
    PropertyFreq,
    PropertyProfile,
    assemble_profile,
    cardinality_tag,
    collect_iris,
    compute_shape,
    emit_pseudo_shex,
)
from grasp.configs import ShapeConfig
from grasp.manager import KgManager
from grasp.manager.normalizer import Normalizer, WikidataPropertyNormalizer
from grasp.shapes import ShapeSample, Target, TargetClass, TargetLiteral
from grasp.sparql.types import AskResult, SelectResult


def _attach_normalizers(
    m: Mock,
    property_normalizer: Normalizer | None = None,
    entity_normalizer: Normalizer | None = None,
) -> None:
    normalizers = {
        "properties": property_normalizer or Normalizer(),
        "entities": entity_normalizer or Normalizer(),
    }
    m.get_normalizer.side_effect = lambda idx: normalizers.get(idx, Normalizer())
    m.normalize.side_effect = lambda iri, idx: normalizers.get(
        idx, Normalizer()
    ).normalize(iri)
    m.denormalize.side_effect = lambda iri, variant, idx: normalizers.get(
        idx, Normalizer()
    ).denormalize(iri, variant)


def make_manager(
    property_normalizer: Normalizer | None = None,
    entity_normalizer: Normalizer | None = None,
) -> Mock:
    m = Mock(spec=KgManager)
    m.format_iri.side_effect = lambda iri, **_: iri
    m.get_label.return_value = None
    m.try_get_data.return_value = None
    m.prefixes = {}
    _attach_normalizers(m, property_normalizer, entity_normalizer)
    return m


def make_labelled_manager(
    entity_labels: dict[str, str] | None = None,
    property_labels: dict[str, str] | None = None,
    property_normalizer: Normalizer | None = None,
    entity_normalizer: Normalizer | None = None,
) -> Mock:
    m = Mock(spec=KgManager)
    m.format_iri.side_effect = lambda iri, **_: iri
    m.prefixes = {}

    _entity_labels = entity_labels or {}
    _property_labels = property_labels or {}

    def get_label(iri: str, index: str) -> str | None:
        if index == "entities":
            return _entity_labels.get(iri)
        if index == "properties":
            return _property_labels.get(iri)
        return None

    m.get_label.side_effect = get_label
    m.try_get_data.return_value = object()  # non-None signals index exists
    _attach_normalizers(m, property_normalizer, entity_normalizer)
    return m


def select(vars: list[str], bindings: list[dict]) -> SelectResult:
    return SelectResult.from_json(
        {"head": {"vars": vars}, "results": {"bindings": bindings}}
    )


def uri(value: str) -> dict:
    return {"type": "uri", "value": value}


def literal(value: str, datatype: str | None = None) -> dict:
    d: dict = {"type": "literal", "value": value}
    if datatype:
        d["datatype"] = datatype
    return d


def make_directed(
    freq: dict[str, dict] | None = None,
    literals: dict[str, dict[str, int]] | None = None,
    ranges: dict[str, dict[str, int]] | None = None,
    bnodes: dict[str, int] | None = None,
) -> DirectedMaps:
    return DirectedMaps(
        freq={p: PropertyFreq(**f) for p, f in (freq or {}).items()},
        literals=literals or {},
        classes=ranges or {},
        bnodes=bnodes or {},
    )


def make_maps(
    freq: dict[str, dict] | None = None,
    literals: dict[str, dict[str, int]] | None = None,
    ranges: dict[str, dict[str, int]] | None = None,
    bnodes: dict[str, int] | None = None,
    incoming: DirectedMaps | None = None,
) -> ClassMaps:
    return ClassMaps(
        out=make_directed(freq, literals, ranges, bnodes),
        inc=incoming or DirectedMaps(),
    )


class TestCardinality:
    def test_mandatory_single(self):
        assert cardinality_tag(0.95, 1.05) == ""

    def test_optional_single(self):
        assert cardinality_tag(0.50, 1.05) == "?"

    def test_mandatory_multi(self):
        assert cardinality_tag(0.95, 2.0) == "+"

    def test_optional_multi(self):
        assert cardinality_tag(0.50, 2.0) == "*"


class TestAssembleProfile:
    def test_basic_profile(self):
        manager = make_manager()
        freq_map = {
            "http://ex.org/name": {"triple_count": 900, "entity_count": 900},
            "http://ex.org/type": {"triple_count": 1000, "entity_count": 1000},
        }
        lit_counts = {"http://ex.org/name": {"xsd:string": 900}}
        range_counts = {"http://ex.org/type": {"http://ex.org/Class": 1000}}
        shape_config = ShapeConfig(min_property_share=0.01)

        profile = assemble_profile(
            "http://ex.org/Human",
            make_maps(freq_map, lit_counts, range_counts),
            total_entities=1000,
            shape_config=shape_config,
            manager=manager,
        )
        shex = emit_pseudo_shex(profile, manager, shape_config)

        assert shex == (
            "http://ex.org/Human {\n"
            "  http://ex.org/type http://ex.org/Class ;\n"
            "  http://ex.org/name xsd:string ;\n"
            "}"
        )

    def test_min_coverage_filters(self):
        manager = make_manager()
        freq_map = {
            "http://ex.org/rare": {"triple_count": 1, "entity_count": 1},
            "http://ex.org/common": {"triple_count": 900, "entity_count": 900},
        }
        shape_config = ShapeConfig(min_property_share=0.5)

        profile = assemble_profile(
            "http://ex.org/C",
            make_maps(freq_map),
            total_entities=1000,
            shape_config=shape_config,
            manager=manager,
        )
        shex = emit_pseudo_shex(profile, manager, shape_config)

        assert shex == (
            "http://ex.org/C {\n"
            "  http://ex.org/common IRI ;\n"
            "  ... 1 filtered (low coverage)\n"
            "}"
        )

    def test_cap_omits(self):
        manager = make_manager()
        freq_map = {
            f"http://ex.org/p{i}": {"triple_count": 100 - i, "entity_count": 100 - i}
            for i in range(5)
        }
        shape_config = ShapeConfig(max_properties_per_class=3, min_property_share=0.0)

        profile = assemble_profile(
            "http://ex.org/C",
            make_maps(freq_map),
            total_entities=100,
            shape_config=shape_config,
            manager=manager,
        )
        shex = emit_pseudo_shex(profile, manager, shape_config)

        assert "... 2 omitted (cap)" in shex
        assert "filtered" not in shex

    def test_cap_and_coverage_filter(self):
        manager = make_manager()
        # p0: common, p1: rare (high triples, low entities → filtered within cap),
        # p2: common, p3: omitted (beyond cap=3)
        freq_map = {
            "http://ex.org/p0": {"triple_count": 100, "entity_count": 100},
            "http://ex.org/p1": {"triple_count": 99, "entity_count": 1},
            "http://ex.org/p2": {"triple_count": 98, "entity_count": 98},
            "http://ex.org/p3": {"triple_count": 1, "entity_count": 1},
        }
        shape_config = ShapeConfig(max_properties_per_class=3, min_property_share=0.5)

        profile = assemble_profile(
            "http://ex.org/C",
            make_maps(freq_map),
            total_entities=100,
            shape_config=shape_config,
            manager=manager,
        )
        shex = emit_pseudo_shex(profile, manager, shape_config)

        assert "... 1 omitted (cap), 1 filtered (low coverage)" in shex

    def test_empty_freq_map(self):
        manager = make_manager()
        shape_config = ShapeConfig()
        profile = assemble_profile(
            "http://ex.org/X",
            make_maps(),
            total_entities=100,
            shape_config=shape_config,
            manager=manager,
        )
        shex = emit_pseudo_shex(profile, manager, shape_config)

        assert shex == "http://ex.org/X {\n  # no properties found for this class\n}"

    def test_hint_suppressed_when_only_incoming(self):
        # a class with no properties of its own but with inverse edges is not
        # empty, so the hint must not fire.
        manager = make_manager()
        shape_config = ShapeConfig()
        profile = assemble_profile(
            "http://ex.org/Decision",
            make_maps(
                incoming=make_directed(
                    freq={"http://ex.org/hasDecision": {"triple_count": 10}},
                    ranges={"http://ex.org/hasDecision": {"http://ex.org/Paper": 10}},
                )
            ),
            total_entities=0,
            shape_config=shape_config,
            manager=manager,
        )
        shex = emit_pseudo_shex(profile, manager, shape_config)

        assert shex == (
            "http://ex.org/Decision {\n"
            "  ^http://ex.org/hasDecision http://ex.org/Paper ;\n"
            "}"
        )

    def test_incoming_has_its_own_cap(self):
        manager = make_manager()
        shape_config = ShapeConfig(max_incoming_per_class=2)
        incoming = make_directed(
            freq={f"http://ex.org/in{i}": {"triple_count": 100 - i} for i in range(5)},
        )
        profile = assemble_profile(
            "http://ex.org/C",
            make_maps(
                freq={"http://ex.org/out": {"triple_count": 10, "entity_count": 10}},
                incoming=incoming,
            ),
            total_entities=10,
            shape_config=shape_config,
            manager=manager,
        )
        shex = emit_pseudo_shex(profile, manager, shape_config)

        # the outgoing property is untouched by the incoming cap
        assert "  http://ex.org/out IRI ;" in shex
        assert shex.count("  ^") == 2
        assert "... 3 incoming omitted (cap)" in shex

    def test_blank_node_objects_split_out_of_the_iri_gap(self):
        manager = make_manager()
        shape_config = ShapeConfig()
        profile = assemble_profile(
            "http://ex.org/C",
            make_maps(
                freq={"http://ex.org/p": {"triple_count": 100, "entity_count": 100}},
                ranges={"http://ex.org/p": {"http://ex.org/T": 40}},
                bnodes={"http://ex.org/p": 35},
            ),
            total_entities=100,
            shape_config=shape_config,
            manager=manager,
        )
        shex = emit_pseudo_shex(profile, manager, shape_config)

        # 40 typed + 35 blank + 25 untyped IRIs; blanks must not be swallowed
        # by the gap bucket, which would misreport them as IRI.
        assert shex == (
            "http://ex.org/C {\n  http://ex.org/p [ http://ex.org/T BNODE IRI ] ;\n}"
        )


class TestMixedTargets:
    def test_literal_class_and_iri_gap(self):
        manager = make_manager()
        freq_map = {
            "http://ex.org/author": {"triple_count": 100, "entity_count": 100},
        }
        lit_counts = {"http://ex.org/author": {"xsd:string": 60}}
        range_counts = {"http://ex.org/author": {"http://ex.org/Person": 30}}
        shape_config = ShapeConfig(min_property_share=0.0)

        profile = assemble_profile(
            "http://ex.org/Book",
            make_maps(freq_map, lit_counts, range_counts),
            total_entities=100,
            shape_config=shape_config,
            manager=manager,
        )

        assert len(profile.properties) == 1
        targets = profile.properties[0].targets
        # ordered by triple_count desc: xsd:string (60), Person (30), IRI (10)
        assert [type(t).__name__ for t in targets] == [
            "TargetLiteral",
            "TargetClass",
            "TargetIri",
        ]
        assert [t.triple_count for t in targets] == [60, 30, 10]

        shex = emit_pseudo_shex(profile, manager, shape_config)
        assert "[ xsd:string http://ex.org/Person IRI ]" in shex

    def test_iri_gap_below_threshold_is_dropped(self):
        manager = make_manager()
        freq_map = {
            "http://ex.org/p": {"triple_count": 1000, "entity_count": 1000},
        }
        # 999 typed, 1 untyped (0.1%) — below default 1% threshold
        lit_counts = {"http://ex.org/p": {"xsd:string": 999}}
        range_counts: dict[str, dict[str, int]] = {}
        shape_config = ShapeConfig(min_property_share=0.0)

        profile = assemble_profile(
            "http://ex.org/C",
            make_maps(freq_map, lit_counts, range_counts),
            total_entities=1000,
            shape_config=shape_config,
            manager=manager,
        )
        assert [type(t).__name__ for t in profile.properties[0].targets] == [
            "TargetLiteral",
        ]


class TestEmitRetrievalDoc:
    def test_full_output(self):
        manager = make_manager()
        shape_config = ShapeConfig()
        profile = ClassProfile(
            iri="http://ex.org/Human",
            short_iri="ex:Human",
            total_entities=500,
            properties=[
                PropertyProfile(
                    iri="http://ex.org/name",
                    short_iri="ex:name",
                    triple_count=500,
                    entity_count=500,
                )
            ],
        )
        shex = emit_pseudo_shex(profile, manager, shape_config)

        assert shex == "ex:Human {\n  ex:name IRI ;\n}"


class TestComputeShape:
    def test_basic(self):
        manager = make_manager()

        freq_result = select(
            ["p", "tripleCount", "entityCount"],
            [
                {
                    "p": uri("http://ex.org/name"),
                    "tripleCount": literal("800"),
                    "entityCount": literal("800"),
                }
            ],
        )
        lit_result = select(
            ["p", "datatype", "count"],
            [
                {
                    "p": uri("http://ex.org/name"),
                    "datatype": uri("http://www.w3.org/2001/XMLSchema#string"),
                    "count": literal("800"),
                }
            ],
        )
        range_result = select(["p", "targetClass", "count"], [])
        in_freq_result = select(
            ["p", "tripleCount", "entityCount", "bnodeCount"],
            [
                {
                    "p": uri("http://ex.org/knows"),
                    "tripleCount": literal("400"),
                    "entityCount": literal("400"),
                    "bnodeCount": literal("0"),
                }
            ],
        )
        in_source_result = select(
            ["p", "sourceClass", "count"],
            [
                {
                    "p": uri("http://ex.org/knows"),
                    "sourceClass": uri("http://ex.org/Human"),
                    "count": literal("400"),
                }
            ],
        )
        total_result = select(["totalEntities"], [{"totalEntities": literal("1000")}])
        manager.execute_sparql.side_effect = [
            freq_result,
            lit_result,
            range_result,
            in_freq_result,
            in_source_result,
            total_result,
        ]

        shape_config = ShapeConfig()
        profile = compute_shape(
            "http://ex.org/Human",
            manager,
            instance_pattern="?instance wdt:P31 {CLASS} .",
            shape_config=shape_config,
        )

        # inverse edges render with a leading ^ and carry no cardinality tag,
        # even though the incoming entity counts are known.
        assert emit_pseudo_shex(profile, manager, shape_config) == (
            "http://ex.org/Human {\n"
            "  http://ex.org/name http://www.w3.org/2001/XMLSchema#string ? ;\n"
            "  ^http://ex.org/knows http://ex.org/Human ;\n"
            "}"
        )

    def test_query_failure_raises(self):
        manager = make_manager()
        manager.execute_sparql.side_effect = Exception("timeout")

        try:
            compute_shape(
                "http://ex.org/X", manager, instance_pattern="?instance a {CLASS} ."
            )
            assert False, "Expected an exception"
        except RuntimeError as e:
            assert "timeout" in str(e)

    def test_missing_class_raises_not_found(self):
        # every profile query legitimately returns zero rows for an IRI that is
        # not in the graph; only the existence probe can tell the difference.
        manager = make_manager()
        manager.execute_sparql.side_effect = [
            select(["property", "target"], []),
            AskResult(boolean=False),
        ]

        try:
            compute_shape(
                "http://edas/#MadeUp",
                manager,
                schema_pattern="?property rdfs:domain {CLASS} .",
            )
            assert False, "Expected an exception"
        except RuntimeError as e:
            assert "http://edas/#MadeUp" in str(e)
            assert "does not exist" in str(e)

    def test_existing_class_without_properties_renders_hint(self):
        # same empty query results as above, but the IRI does occur in the graph
        manager = make_manager()
        manager.execute_sparql.side_effect = [
            select(["property", "target"], []),
            AskResult(boolean=True),
        ]

        shape_config = ShapeConfig()
        profile = compute_shape(
            "http://edas/#Acceptance",
            manager,
            schema_pattern="?property rdfs:domain {CLASS} .",
            shape_config=shape_config,
        )

        assert emit_pseudo_shex(profile, manager, shape_config) == (
            "http://edas/#Acceptance {\n  # no properties found for this class\n}"
        )

    def test_schema_pattern_classifies_targets(self):
        # mirrors EDAS: a single (?property, ?other, ?dir) query, no instances/counts.
        manager = make_manager()
        xsd_string = "http://www.w3.org/2001/XMLSchema#string"
        schema_result = select(
            ["property", "other", "dir"],
            [
                {
                    "property": uri("http://edas/#hasFirstName"),
                    "other": uri(xsd_string),
                    "dir": literal("out"),
                },
                {
                    "property": uri("http://edas/#attendeeAt"),
                    "other": uri("http://edas/#ConferenceEvent"),
                    "dir": literal("out"),
                },
                {
                    "property": uri("http://edas/#noRange"),
                    "dir": literal("out"),
                },  # optional peer unbound
            ],
        )
        manager.execute_sparql.side_effect = [schema_result]

        shape_config = ShapeConfig()
        profile = compute_shape(
            "http://edas/#Person",
            manager,
            schema_pattern=(
                "?property rdfs:domain {CLASS} . "
                "OPTIONAL { ?property rdfs:range ?other } "
                'BIND("out" AS ?dir)'
            ),
            shape_config=shape_config,
        )

        # one SELECT DISTINCT ?property ?other ?dir query, no instance/total queries
        assert manager.execute_sparql.call_count == 1
        assert profile.total_entities == 0  # no cardinality tags

        out = emit_pseudo_shex(profile, manager, shape_config)
        # xsd:* range -> literal datatype target; class IRI -> class target;
        # missing range -> bare IRI; total_entities == 0 so no cardinality tags.
        assert f"http://edas/#hasFirstName {xsd_string} ;" in out
        assert "http://edas/#attendeeAt http://edas/#ConferenceEvent ;" in out
        assert "http://edas/#noRange IRI ;" in out
        assert "?" not in out and "+" not in out and "*" not in out

    def test_schema_pattern_routes_both_directions(self):
        manager = make_manager()
        schema_result = select(
            ["property", "other", "dir"],
            [
                {
                    "property": uri("http://cmt/#hasDecision"),
                    "other": uri("http://cmt/#Decision"),
                    "dir": literal("out"),
                },
                {
                    "property": uri("http://cmt/#writePaper"),
                    "other": uri("http://cmt/#Author"),
                    "dir": literal("in"),
                },
            ],
        )
        manager.execute_sparql.side_effect = [schema_result]

        shape_config = ShapeConfig()
        profile = compute_shape(
            "http://cmt/#Paper",
            manager,
            schema_pattern="{ ?property rdfs:domain {CLASS} } UNION "
            "{ ?property rdfs:range {CLASS} }",
            shape_config=shape_config,
        )

        assert [p.iri for p in profile.properties] == ["http://cmt/#hasDecision"]
        assert [p.iri for p in profile.incoming] == ["http://cmt/#writePaper"]
        assert emit_pseudo_shex(profile, manager, shape_config) == (
            "http://cmt/#Paper {\n"
            "  http://cmt/#hasDecision http://cmt/#Decision ;\n"
            "  ^http://cmt/#writePaper http://cmt/#Author ;\n"
            "}"
        )

    def test_schema_row_without_dir_raises(self):
        manager = make_manager()
        manager.execute_sparql.side_effect = [
            select(
                ["property", "other", "dir"],
                [{"property": uri("http://ex.org/p"), "other": uri("http://ex.org/C")}],
            )
        ]

        try:
            compute_shape(
                "http://ex.org/X",
                manager,
                schema_pattern="?property rdfs:domain {CLASS} .",
            )
            assert False, "Expected an exception"
        except RuntimeError as e:
            assert "?dir" in str(e)

    def test_blank_node_peer_renders_as_bnode(self):
        manager = make_manager()
        schema_result = select(
            ["property", "other", "dir"],
            [
                {
                    "property": uri("http://cmt/#markConflictOfInterest"),
                    "other": {"type": "bnode", "value": "bn4"},
                    "dir": literal("in"),
                }
            ],
        )
        manager.execute_sparql.side_effect = [schema_result]

        shape_config = ShapeConfig()
        profile = compute_shape(
            "http://cmt/#Paper",
            manager,
            schema_pattern="?property rdfs:range {CLASS} .",
            shape_config=shape_config,
        )

        # an anonymous class expression is reported as BNODE, not as a bare IRI:
        # claiming IRI would tell the model to bind something that cannot match.
        assert emit_pseudo_shex(profile, manager, shape_config) == (
            "http://cmt/#Paper {\n  ^http://cmt/#markConflictOfInterest BNODE ;\n}"
        )

    def test_merged_keeps_schema_only_props_but_filters_rare_instance_props(self):
        # both patterns set: instance + schema bindings merge. A rare *instance*
        # property (low coverage) is filtered, but a pure schema-declared property
        # with no instance usage (entity_count == 0) is kept.
        manager = make_manager()
        freq_result = select(
            ["p", "tripleCount", "entityCount"],
            [
                {
                    "p": uri("http://ex.org/name"),
                    "tripleCount": literal("800"),
                    "entityCount": literal("800"),
                },
                {  # coverage 5/1000 = 0.005 < min_property_share (0.01) -> filtered
                    "p": uri("http://ex.org/rare"),
                    "tripleCount": literal("5"),
                    "entityCount": literal("5"),
                },
            ],
        )
        lit_result = select(
            ["p", "datatype", "count"],
            [
                {
                    "p": uri("http://ex.org/name"),
                    "datatype": uri("http://www.w3.org/2001/XMLSchema#string"),
                    "count": literal("800"),
                }
            ],
        )
        range_result = select(
            ["p", "targetClass", "count"],
            [
                {
                    "p": uri("http://ex.org/rare"),
                    "targetClass": uri("http://ex.org/Thing"),
                    "count": literal("5"),
                }
            ],
        )
        total_result = select(["totalEntities"], [{"totalEntities": literal("1000")}])
        empty_in_freq = select(["p", "tripleCount", "entityCount", "bnodeCount"], [])
        empty_in_source = select(["p", "sourceClass", "count"], [])
        schema_result = select(
            ["property", "other", "dir"],
            [
                {
                    "property": uri("http://ex.org/schemaOnly"),
                    "other": uri("http://ex.org/Target"),
                    "dir": literal("out"),
                }
            ],
        )
        manager.execute_sparql.side_effect = [
            freq_result,
            lit_result,
            range_result,
            empty_in_freq,
            empty_in_source,
            total_result,
            schema_result,
        ]

        shape_config = ShapeConfig()
        profile = compute_shape(
            "http://ex.org/C",
            manager,
            instance_pattern="?instance a {CLASS} .",
            schema_pattern="?property rdfs:domain {CLASS} . "
            'OPTIONAL { ?property rdfs:range ?other } BIND("out" AS ?dir)',
            shape_config=shape_config,
        )

        # 5 instance-side queries (freq, literals, ranges, incoming freq,
        # incoming sources) + the total + 1 schema query
        assert manager.execute_sparql.call_count == 7
        out = emit_pseudo_shex(profile, manager, shape_config)
        assert "http://ex.org/name" in out  # high coverage instance prop kept
        assert "http://ex.org/schemaOnly" in out  # schema-only prop kept (no usage)
        assert "http://ex.org/rare" not in out  # rare instance prop filtered


def _make_profile(with_target_iris: bool = True) -> ClassProfile:
    type_targets: list[Target] = (
        [TargetClass(iri="http://ex.org/Class", short_iri="ex:Class")]
        if with_target_iris
        else []
    )
    return ClassProfile(
        iri="http://ex.org/Human",
        short_iri="ex:Human",
        total_entities=0,
        properties=[
            PropertyProfile(
                iri="http://ex.org/type",
                short_iri="ex:type",
                triple_count=500,
                entity_count=500,
                targets=type_targets,
            ),
            PropertyProfile(
                iri="http://ex.org/name",
                short_iri="ex:name",
                triple_count=400,
                entity_count=400,
                targets=[TargetLiteral(datatype="xsd:string")],
            ),
        ],
    )


class TestEmitPseudoShexLabelled:
    def test_no_index_falls_back_to_short_iris(self):
        manager = make_manager()
        profile = _make_profile()
        shex = emit_pseudo_shex(profile, manager, ShapeConfig())
        assert shex == ("ex:Human {\n  ex:type ex:Class ;\n  ex:name xsd:string ;\n}")

    def test_labels_resolved_when_index_available(self):
        manager = make_labelled_manager(
            entity_labels={
                "http://ex.org/Human": "Human",
                "http://ex.org/Class": "MyClass",
            },
            property_labels={
                "http://ex.org/type": "type of",
                "http://ex.org/name": "name",
            },
        )
        profile = _make_profile()
        shex = emit_pseudo_shex(profile, manager, ShapeConfig())
        assert shex == (
            "ex:Human {\n"
            "  ex:type (type of) ex:Class (MyClass) ;\n"
            "  ex:name xsd:string ;\n"
            "}"
        )

    def test_partial_labels_fall_back_to_short_iri(self):
        manager = make_labelled_manager(
            entity_labels={"http://ex.org/Human": "Human"},
        )
        profile = _make_profile()
        shex = emit_pseudo_shex(profile, manager, ShapeConfig())
        assert shex == ("ex:Human {\n  ex:type ex:Class ;\n  ex:name xsd:string ;\n}")


class TestWikidataVariantGrouping:
    def test_property_variants_collapse_to_single_line(self):
        manager = make_manager(property_normalizer=WikidataPropertyNormalizer())
        freq_map = {
            "http://www.wikidata.org/prop/direct/P31": {
                "triple_count": 100,
                "entity_count": 100,
            },
            "http://www.wikidata.org/prop/P31": {
                "triple_count": 100,
                "entity_count": 100,
            },
            "http://www.wikidata.org/prop/statement/P31": {
                "triple_count": 100,
                "entity_count": 100,
            },
        }
        profile = assemble_profile(
            "http://ex.org/Human",
            make_maps(freq_map),
            total_entities=100,
            shape_config=ShapeConfig(min_property_share=0.0),
            manager=manager,
        )

        assert len(profile.properties) == 1
        prop = profile.properties[0]
        assert prop.iri == "http://www.wikidata.org/entity/P31"
        assert prop.variants == ["p", "ps", "wdt"]
        assert prop.triple_count == 300
        assert prop.entity_count == 300

        shex = emit_pseudo_shex(profile, manager, ShapeConfig(min_property_share=0.0))
        assert "as p/ps/wdt" in shex
        assert shex.count("P31") == 1

    def test_collect_iris_emits_all_variants(self):
        manager = make_manager(property_normalizer=WikidataPropertyNormalizer())
        freq_map = {
            "http://www.wikidata.org/prop/direct/P31": {
                "triple_count": 10,
                "entity_count": 10,
            },
            "http://www.wikidata.org/prop/P31": {
                "triple_count": 10,
                "entity_count": 10,
            },
        }
        profile = assemble_profile(
            "http://ex.org/Human",
            make_maps(freq_map),
            total_entities=10,
            shape_config=ShapeConfig(min_property_share=0.0),
            manager=manager,
        )
        iris = collect_iris(profile, manager, ShapeConfig(min_property_share=0.0))
        assert "http://www.wikidata.org/entity/P31" in iris
        assert "http://www.wikidata.org/prop/direct/P31" in iris
        assert "http://www.wikidata.org/prop/P31" in iris

    def test_collect_iris_covers_incoming(self):
        # inverse edges are rendered, so their IRIs must be marked known too or
        # know_before_use rejects the very properties the shape just showed.
        manager = make_manager()
        profile = assemble_profile(
            "http://ex.org/Decision",
            make_maps(
                incoming=make_directed(
                    freq={"http://ex.org/hasDecision": {"triple_count": 10}},
                    ranges={"http://ex.org/hasDecision": {"http://ex.org/Paper": 10}},
                )
            ),
            total_entities=0,
            shape_config=ShapeConfig(),
            manager=manager,
        )
        iris = collect_iris(profile, manager, ShapeConfig())
        assert "http://ex.org/hasDecision" in iris
        assert "http://ex.org/Paper" in iris


def _empty_profile(
    iri: str = "http://ex.org/Q5", short_iri: str = "wd:Q5"
) -> ClassProfile:
    return ClassProfile(iri=iri, short_iri=short_iri)


class TestShapeSampleQueries:
    def test_label_and_aliases(self):
        s = ShapeSample(
            iri="http://ex.org/Q5",
            short_iri="wd:Q5",
            profile=_empty_profile(),
            label="Human",
            aliases=["person", "human being"],
        )
        assert s.queries() == ["Human", "person", "human being", "wd:Q5"]

    def test_deduplicates(self):
        s = ShapeSample(
            iri="http://ex.org/Q5",
            short_iri="wd:Q5",
            profile=_empty_profile(),
            label="wd:Q5",  # same as short_iri
        )
        assert s.queries() == ["wd:Q5"]

    def test_missing_optional_fields(self):
        s = ShapeSample(
            iri="http://ex.org/Q5",
            short_iri="wd:Q5",
            profile=_empty_profile(),
        )
        assert s.queries() == ["wd:Q5"]

    def test_roundtrip_via_model_dump(self):
        s = ShapeSample(
            iri="http://ex.org/Q5",
            short_iri="wd:Q5",
            profile=_empty_profile(),
            label="Human",
            aliases=["person"],
        )
        s2 = ShapeSample.model_validate(s.model_dump())
        assert s2 == s
