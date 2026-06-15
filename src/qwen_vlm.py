import base64
import json
import os
from pathlib import Path


DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"


class QwenVLM:

    def __init__(
        self,
        base_url=None,
        api_key=None,
        model=None
    ):

        from openai import OpenAI

        self.model = model or os.getenv(
            "QWEN_VLM_MODEL",
            DEFAULT_MODEL
        )

        self.client = OpenAI(
            base_url=base_url or os.getenv(
                "QWEN_VLM_BASE_URL",
                "http://localhost:8000/v1"
            ),
            api_key=api_key or os.getenv(
                "QWEN_VLM_API_KEY",
                "EMPTY"
            )
        )

    def describe(self, image_path):

        image_url = image_to_data_url(
            image_path
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an industrial robot scene understanding "
                        "model. Return only valid JSON."
                    )
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": scene_prompt()
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            }
                        }
                    ]
                }
            ],
            temperature=0,
            max_tokens=800
        )

        content = response.choices[0].message.content

        return normalize_scene_json(
            parse_json_response(content)
        )


def image_to_data_url(image_path):

    path = Path(image_path)
    suffix = path.suffix.lower()

    if suffix in {".jpg", ".jpeg"}:
        mime_type = "image/jpeg"
    elif suffix == ".png":
        mime_type = "image/png"
    else:
        mime_type = "application/octet-stream"

    encoded = base64.b64encode(
        path.read_bytes()
    ).decode("ascii")

    return f"data:{mime_type};base64,{encoded}"


def scene_prompt():

    return """
Analyze this industrial robot workspace.

Return JSON with this exact shape:

{
  "environment_type": "machine_shop|factory_floor|warehouse|outdoor_industrial|unknown",
  "objects": [
    {"label": "industrial_machine", "count": 1}
  ],
  "hazards": [
    "human_in_operational_zone"
  ],
  "navigation": {
    "aisle_detected": true,
    "walkable_region": "center|left|right|unknown",
    "obstacle_regions": ["left", "right"]
  },
  "located_objects": [
    {
      "label": "human",
      "bbox": {"x1": 10, "y1": 20, "x2": 100, "y2": 180},
      "confidence": 0.8
    }
  ]
}

Use snake_case labels. Focus on humans, robots, machines, workbenches,
carts, cabinets, boxes, forklifts, pipes, obstacles, and navigable aisles.
If you can estimate object boxes, include them in located_objects using pixel
coordinates relative to the input image. If boxes are uncertain, return an
empty located_objects list.
Return only JSON.
""".strip()


def parse_json_response(content):

    text = content.strip()

    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1 or end <= start:
            raise

        return json.loads(
            text[start:end + 1]
        )


def normalize_scene_json(scene):

    navigation = scene.get(
        "navigation",
        {}
    )

    return {
        "environment_type":
            scene.get(
                "environment_type",
                "unknown"
            ),

        "objects":
            normalize_objects(
                scene.get(
                    "objects",
                    []
                )
            ),

        "hazards":
            list(
                scene.get(
                    "hazards",
                    []
                )
            ),

        "navigation": {
            "aisle_detected":
                bool(
                    navigation.get(
                        "aisle_detected",
                        False
                    )
                ),

            "walkable_region":
                navigation.get(
                    "walkable_region",
                    "unknown"
                ),

            "obstacle_regions":
                list(
                    navigation.get(
                        "obstacle_regions",
                        []
                    )
                )
        },

        "located_objects":
            normalize_located_objects(
                scene.get(
                    "located_objects",
                    []
                )
            )
    }


def normalize_objects(objects):

    normalized = []

    for obj in objects:

        if isinstance(obj, str):
            normalized.append({
                "label":
                    obj.replace(
                        " ",
                        "_"
                    ),
                "count":
                    1
            })
            continue

        label = obj.get(
            "label",
            "unknown"
        ).replace(
            " ",
            "_"
        )

        normalized.append({
            "label":
                label,

            "count":
                int(
                    obj.get(
                        "count",
                        1
                    )
                )
        })

    return normalized


def normalize_located_objects(objects):

    normalized = []

    for obj in objects:

        if not isinstance(obj, dict):
            continue

        bbox = obj.get(
            "bbox"
        )

        if not bbox:
            continue

        normalized.append({
            "label":
                obj.get(
                    "label",
                    "unknown"
                ).replace(
                    " ",
                    "_"
                ),

            "bbox":
                bbox,

            "distance":
                obj.get(
                    "distance"
                ),

            "confidence":
                float(
                    obj.get(
                        "confidence",
                        obj.get(
                            "score",
                            1.0
                        )
                    )
                ),

            "source":
                "qwen_vlm"
        })

    return normalized
