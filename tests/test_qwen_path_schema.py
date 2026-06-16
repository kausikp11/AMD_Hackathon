from src.qwen_vlm import normalize_scene_json


def test_qwen_navigation_desired_path_is_preserved():
    scene = normalize_scene_json({
        "navigation": {
            "aisle_detected": True,
            "walkable_region": "center",
            "obstacle_regions": ["left", "right"],
            "desired_path": [
                {"x": 320, "y": 470, "speed": 0},
                {"x": "300", "y": "260", "speed": "0.5"},
                {"bad": "point"}
            ]
        }
    })

    assert scene["navigation"]["desired_path"] == [
        {"x": 320.0, "y": 470.0, "speed": 0.0},
        {"x": 300.0, "y": 260.0, "speed": 0.5}
    ]
