import json
from pathlib import Path

import pytest


EXAMPLES = sorted((Path(__file__).resolve().parents[1] / "example_workflows").glob("*.json"))


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda path: path.stem)
def test_example_links_and_credentials(path, plugin):
    graph = json.loads(path.read_text())
    nodes = {node["id"]: node for node in graph["nodes"]}
    assert len(nodes) == len(graph["nodes"])
    for link_id, source, output, target, input_index, kind in graph["links"]:
        assert nodes[target]["inputs"][input_index]["link"] == link_id
        assert link_id in nodes[source]["outputs"][output]["links"]
        if nodes[source]["type"] in plugin.NODE_CLASS_MAPPINGS:
            cls = plugin.NODE_CLASS_MAPPINGS[nodes[source]["type"]]
            assert cls.RETURN_TYPES[output] == kind
    for node in nodes.values():
        if node["type"] in plugin.NODE_CLASS_MAPPINGS:
            assert "title" not in node  # New examples use the selected UI language.
        if node["type"] == "VisionAPIDirect":
            assert node["widgets_values"][2] == ""
    if path.stem == "08_ordered_model_release":
        unload = next(node for node in nodes.values() if node["type"] == "VisionUnload")
        dependency = next(port["link"] for port in unload["inputs"] if port["name"] == "after")
        link = next(link for link in graph["links"] if link[0] == dependency)
        assert nodes[link[1]]["type"] == "VisionChat" and link[2] == 0
