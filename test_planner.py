# test_planner.py

from src.loader import load_frame
from src.world_state import build_world_state
from src.reasoner import reason

from src.vlm_scene import describe_scene

from src.object_memory import ObjectMemory

from src.scene_graph import build_scene_graph

from src.planner import (
    plan,
    planner_summary
)

frame = load_frame("013342")

world = build_world_state(frame)

decision = reason(world)

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

robot_plan = plan(
    world,
    decision,
    scene,
    graph
)

print(robot_plan)

print()
print(planner_summary(robot_plan))