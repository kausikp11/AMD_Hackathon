from src.loader import load_frame
from src.world_state import build_world_state
from src.reasoner import reason
from src.vlm_scene import describe_scene
from src.object_memory import ObjectMemory
from src.scene_graph import build_scene_graph
from src.planner import plan
from src.copilot import generate_explanation

# Load frame
frame = load_frame("013342")

# Build world state
world = build_world_state(frame)

# Risk reasoning
decision = reason(world)

# Scene understanding
scene = describe_scene(
    frame,
    world
)

# Object memory
memory = ObjectMemory()

memory.update(
    scene["objects"],
    frame["frame_id"]
)

# Scene graph
graph = build_scene_graph(
    world,
    scene,
    memory
)

# Planner
robot_plan = plan(
    world,
    decision,
    scene,
    graph
)

# Copilot explanation
text = generate_explanation(
    world,
    scene,
    robot_plan
)

print(text)