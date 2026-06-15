from pprint import pprint

from src.loader import load_frame
from src.world_state import build_world_state
from src.reasoner import reason
from src.scene_understanding import understand_scene
from src.object_memory import ObjectMemory
from src.scene_graph import build_scene_graph


def main():

    frame_id = "013342"

    print("=" * 60)
    print("LOADING FRAME")
    print("=" * 60)

    frame = load_frame(frame_id)

    print(f"Frame ID: {frame['frame_id']}")
    print()

    # --------------------------------------------------
    # WORLD STATE
    # --------------------------------------------------

    print("=" * 60)
    print("WORLD STATE")
    print("=" * 60)

    world = build_world_state(frame)

    pprint(world)

    print()

    # --------------------------------------------------
    # REASONER
    # --------------------------------------------------

    print("=" * 60)
    print("REASONER")
    print("=" * 60)

    decision = reason(world)

    pprint(decision)

    print()

    # --------------------------------------------------
    # SCENE UNDERSTANDING
    # --------------------------------------------------

    print("=" * 60)
    print("SCENE UNDERSTANDING")
    print("=" * 60)

    scene = understand_scene(frame)

    pprint(scene)

    print()

    # --------------------------------------------------
    # OBJECT MEMORY
    # --------------------------------------------------

    print("=" * 60)
    print("OBJECT MEMORY")
    print("=" * 60)

    memory = ObjectMemory()

    memory.update(
        scene["objects"],
        frame_id
    )

    pprint(memory.summary())

    print()

    # --------------------------------------------------
    # SCENE GRAPH
    # --------------------------------------------------

    print("=" * 60)
    print("SCENE GRAPH")
    print("=" * 60)

    graph = build_scene_graph(
        world,
        memory
    )

    pprint(graph)

    print()

    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()