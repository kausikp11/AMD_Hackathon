from src.loader import load_frame
from src.world_state import build_world_state
from src.reasoner import reason
from src.scene_understanding import understand_scene

frame = load_frame("013342")

world = build_world_state(frame)

decision = reason(world)

scene = understand_scene(frame)

print("\nWORLD")
print(world)

print("\nDECISION")
print(decision)

print("\nSCENE")
print(scene)

print("\nTAGS")
print(world["scene_tags"])