from grasp.build.data import prepare_items


def uri(value: str) -> dict:
    return {"type": "uri", "value": value}


def literal(value: str) -> dict:
    return {"type": "literal", "value": value}


def test_unknown_tags_are_dropped():
    bindings = [
        {
            "id": uri("http://example.org/legalities"),
            "value": literal("legalities"),
            "tags": literal("main"),
        },
        {
            "id": uri("http://example.org/legalities"),
            "value": literal("legality"),
            "tags": literal("alias"),
        },
        {
            "id": uri("http://example.org/sets"),
            "value": literal("sets"),
            "tags": literal("main,alias"),
        },
    ]
    items = list(prepare_items(iter(bindings), {}, None))  # type: ignore
    assert items == [
        {
            "identifier": "http://example.org/legalities",
            "fields": [
                {"type": "text", "value": "legalities", "tags": ["main"]},
                {"type": "text", "value": "legality", "tags": []},
            ],
        },
        {
            "identifier": "http://example.org/sets",
            "fields": [{"type": "text", "value": "sets", "tags": ["main"]}],
        },
    ]


def test_ids_of_wrong_type_are_skipped():
    bindings = [
        {"id": uri("http://example.org/a"), "value": literal("a")},
        {"id": {"type": "bnode", "value": "b0"}, "value": literal("blank")},
        {"id": literal("x"), "value": literal("x")},
        {"id": uri("http://example.org/b"), "value": literal("b")},
    ]
    items = list(prepare_items(iter(bindings), {}, None, id_type="uri"))  # type: ignore
    assert [item["identifier"] for item in items] == [
        "http://example.org/a",
        "http://example.org/b",
    ]

    items = list(prepare_items(iter(bindings), {}, None, id_type="literal"))  # type: ignore
    assert [item["identifier"] for item in items] == ['"x"']
