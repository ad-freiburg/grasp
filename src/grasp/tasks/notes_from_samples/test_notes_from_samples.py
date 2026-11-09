from grasp.tasks.notes_from_samples import note_taking_instructions


def test_samples_name_their_knowledge_graph():
    samples = [
        {"kg": "wald", "input": "Which stands have maple?", "reference": "SELECT"},
        {"kg": "freebase", "input": "Who directed Inception?"},
    ]
    text = note_taking_instructions(samples)
    assert "## Sample 1 over the wald knowledge graph\n\nWhich stands" in text
    assert "### Reference output\n\nSELECT" in text
    assert "## Sample 2 over the freebase knowledge graph\n\nWho directed" in text
