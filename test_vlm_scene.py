# test_vlm_scene.py

from src.loader import load_frame
from src.world_state import build_world_state
from src.vlm_scene import describe_scene
from src.vlm_scene import scene_summary


frame = load_frame("013342")

world = build_world_state(frame)

scene = describe_scene(
    frame,
    world
)

print(scene)

print()
print(scene_summary(scene))