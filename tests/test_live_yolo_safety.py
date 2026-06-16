import os

from src.loader import load_frame
import src.locate_anything as locator
import src.fusion as fusion
from src.world_state import build_world_state
from src.reasoner import reason
from src.copilot import generate_explanation


def fake_locate_objects(frame, object_names=None):
    return [
        {
            "label": "human",
            "bbox": {
                "x1": 10,
                "y1": 20,
                "x2": 110,
                "y2": 220
            },
            "distance": None,
            "confidence": 0.9,
            "source": "yolo_live"
        }
    ]


locator.locate_objects = fake_locate_objects
fusion.locate_objects = fake_locate_objects

os.environ["LOCATOR_BACKEND"] = "yolo_live"

frame = load_frame("013342")
frame["rgb_detections"] = []
frame["thermal_detections"] = []

world = build_world_state(frame)
decision = reason(world)
explanation = generate_explanation(
    world,
    {
        "environment_type": "machine_shop"
    },
    {
        "mode": "MONITOR",
        "action": decision["action"],
        "target_speed": 0.5
    }
)

assert world["human"]["present"] is True
assert world["human"]["distance"] is None
assert world["human"]["source"] == "yolo_live"
assert "unknown_human_distance" in world["scene_tags"]
assert decision["risk_level"] == "MEDIUM"
assert decision["action"] == "PROCEED_WITH_CAUTION"
assert "distance is unavailable" in explanation

print(world["human"])
print(decision)
print(explanation)
