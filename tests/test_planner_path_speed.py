from src.planner import apply_speed_profile
from src.planner import determine_navigation


def test_stop_speed_collapses_path_to_robot_point():
    path = [
        {"x": 320, "y": 490, "speed": 0.0},
        {"x": 300, "y": 390, "speed": 0.4}
    ]

    assert apply_speed_profile(path, 0.0) == [
        {"x": 320, "y": 490, "speed": 0.0}
    ]


def test_path_speed_is_capped_by_target_speed():
    path = [
        {"x": 320, "y": 490, "speed": 0.0},
        {"x": 300, "y": 390, "speed": 1.0}
    ]

    assert apply_speed_profile(path, 0.25)[1]["speed"] == 0.25


def test_navigation_preserves_path_and_floor_sources():
    scene = {
        "navigation": {
            "aisle_detected": True,
            "walkable_region": "center",
            "desired_path": [{"x": 1, "y": 2}],
            "desired_path_source": "qwen_vlm",
            "floor_region": [{"x": 0, "y": 1}],
            "floor_region_source": "qwen_vlm"
        }
    }

    nav = determine_navigation(scene, 0.5)

    assert nav["desired_path_source"] == "qwen_vlm"
    assert nav["floor_region_source"] == "qwen_vlm"
    assert nav["floor_region"] == [{"x": 0, "y": 1}]
