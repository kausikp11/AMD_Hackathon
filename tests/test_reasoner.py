# test_reasoner.py

from src.loader import load_frame
from src.world_state import build_world_state
from src.reasoner import reason, reason_summary

frame = load_frame("013342")

world = build_world_state(frame)

decision = reason(world)

print(world)
print()
print(reason_summary(decision))