from src.loader import load_frame

from src.world_state import build_world_state

from src.reasoner import reason

from src.vlm_scene import describe_scene

from src.object_memory import ObjectMemory

from src.scene_graph import build_scene_graph

from src.planner import plan

from src.copilot import generate_explanation


memory = ObjectMemory()


def run_pipeline(frame_id):

    frame = load_frame(frame_id)

    world = build_world_state(frame)

    decision = reason(world)

    scene = describe_scene(
        frame,
        world
    )

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

    explanation = generate_explanation(
        world,
        scene,
        robot_plan
    )

    return {

        "frame": frame,

        "world": world,

        "decision": decision,

        "scene": scene,

        "graph": graph,

        "plan": robot_plan,

        "explanation": explanation
    }