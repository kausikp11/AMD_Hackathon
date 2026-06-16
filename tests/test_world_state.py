from src.loader import load_frame
from src.world_state import (
    build_world_state,
    world_state_summary
)

frame = load_frame("013342")

world = build_world_state(frame)

print(world)

print()
print(world_state_summary(world))