# test_scene_graph.py

from pprint import pprint

from src.loader import load_frame
from src.world_state import build_world_state
from src.vlm_scene import describe_scene

from src.object_memory import ObjectMemory

from src.scene_graph import (
    build_scene_graph,
    graph_summary
)

frame = load_frame("013342")

world = build_world_state(frame)

scene = describe_scene(
    frame,
    world
)

memory = ObjectMemory()

memory.update(
    scene["objects"],
    frame["frame_id"]
)

graph = build_scene_graph(
    world,
    scene,
    memory
)

pprint(graph)

print()
print(graph_summary(graph))