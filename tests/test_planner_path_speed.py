from src.planner import apply_speed_profile


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
