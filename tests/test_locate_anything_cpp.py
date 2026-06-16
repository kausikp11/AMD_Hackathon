from src.locate_anything import locate_anything_cpp_detections
from src.locate_anything import locate_anything_cpp_prompt


def test_locate_anything_cpp_prompt_uses_category_separator():
    prompt = locate_anything_cpp_prompt([
        "human",
        "industrial machine",
        "floor"
    ])

    assert "human</c>industrial machine" in prompt
    assert "floor" not in prompt


def test_locate_anything_cpp_detections_parse_json_boxes():
    payload = {
        "detections": [
            {
                "label": "person",
                "box": [10, 20, 100, 180],
                "confidence": 0.9
            }
        ]
    }

    assert locate_anything_cpp_detections(payload, ["human"]) == [
        {
            "label": "human",
            "bbox": {
                "x1": 10.0,
                "y1": 20.0,
                "x2": 100.0,
                "y2": 180.0
            },
            "distance": None,
            "confidence": 0.9,
            "source": "locate_anything_cpp"
        }
    ]
