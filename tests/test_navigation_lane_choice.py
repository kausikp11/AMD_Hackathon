from src.vlm_scene import choose_best_lane
from src.vlm_scene import estimate_desired_path


IMAGE_SIZE = (
    640,
    512
)


def test_clear_scene_prefers_straight_center_lane():
    assert choose_best_lane(
        [],
        IMAGE_SIZE
    ) == "center"


def test_center_obstacle_moves_to_free_left_lane():
    located = [
        {
            "label": "human",
            "bbox": {
                "x1": 270,
                "y1": 120,
                "x2": 370,
                "y2": 500
            }
        },
        {
            "label": "cart",
            "bbox": {
                "x1": 390,
                "y1": 220,
                "x2": 520,
                "y2": 500
            }
        }
    ]

    assert choose_best_lane(
        located,
        IMAGE_SIZE
    ) == "left"


def test_left_obstacle_prefers_center_when_center_is_clear():
    located = [
        {
            "label": "storage_box",
            "bbox": {
                "x1": 80,
                "y1": 260,
                "x2": 200,
                "y2": 510
            }
        }
    ]

    assert choose_best_lane(
        located,
        IMAGE_SIZE
    ) == "center"


def test_left_and_center_blocked_moves_right():
    located = [
        {
            "label": "storage_box",
            "bbox": {
                "x1": 70,
                "y1": 260,
                "x2": 245,
                "y2": 510
            }
        },
        {
            "label": "human",
            "bbox": {
                "x1": 250,
                "y1": 160,
                "x2": 405,
                "y2": 510
            }
        }
    ]

    assert choose_best_lane(
        located,
        IMAGE_SIZE
    ) == "right"


def test_desired_path_curves_left_when_left_is_freer(tmp_path):
    image_path = tmp_path / "frame.jpg"

    from PIL import Image

    Image.new(
        "RGB",
        IMAGE_SIZE
    ).save(
        image_path
    )

    located = [
        {
            "label": "human",
            "bbox": {
                "x1": 260,
                "y1": 120,
                "x2": 370,
                "y2": 500
            }
        },
        {
            "label": "cart",
            "bbox": {
                "x1": 390,
                "y1": 220,
                "x2": 520,
                "y2": 500
            }
        }
    ]

    path = estimate_desired_path(
        image_path,
        {
            "walkable_region":
                "center"
        },
        located
    )

    assert path[0]["x"] == 320
    assert path[-1]["x"] < 320
    assert path[1]["x"] > path[-1]["x"]
