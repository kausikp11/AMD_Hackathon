import base64
import json
import os
from pathlib import Path

from src.tracked_objects import load_tracked_objects


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
                        "model. Return only valid JSON. Do not return "
                        "markdown, comments, or explanatory text."
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

        try:
            scene = parse_json_response(
                content
            )
        except json.JSONDecodeError as exc:
            return fallback_scene(
                content,
                exc
            )

        return normalize_scene_json(
            scene
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

    prompts = "\n".join(
        f"- {prompt}"
        for prompt in load_tracked_objects()
    )

    return (
        """
Analyze this industrial robot workspace.

Use these GroundingDINO-style object prompts for localization:

"""
        + prompts
        + """

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
    "obstacle_regions": ["left", "right"],
    "floor_region": [
      {"x": 40, "y": 500},
      {"x": 600, "y": 500},
      {"x": 440, "y": 260},
      {"x": 200, "y": 260}
    ],
    "desired_path": [
      {"x": 320, "y": 490, "speed": 0.0},
      {"x": 280, "y": 390, "speed": 0.25},
      {"x": 240, "y": 270, "speed": 0.5}
    ]
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
Estimate desired_path as 3 to 6 pixel waypoints on the visible floor or ground
plane. The first point should be near the robot at the bottom of the image,
and later points should move through the safest visible aisle or free space.
Keep the path inside image bounds, below the floor horizon, and away from
visible humans and obstacles. Do not place path points on walls, machines, or
other vertical surfaces. Include speed at each point as a normalized target
speed from 0.0 stopped, through 0.5 cautious, to 1.0 normal speed.
Do not force the final waypoint to the image center. Continue the path along
the visible lane that gives the robot the most clearance from humans and
obstacles.
Estimate floor_region as a polygon around the visible traversable floor or
ground plane. Use 4 to 8 pixel points in image coordinates. Exclude machines,
walls, tables, people, shelves, and vertical surfaces. If the floor is unclear,
return an empty floor_region.
If you can estimate object boxes, include them in located_objects using pixel
coordinates relative to the input image. If boxes are uncertain, return an
empty located_objects list.
Return only a single valid JSON object. Use double quotes for every key and
string. Do not include markdown fences, bullets, comments, or text before or
after the JSON object.
"""
    ).strip()


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


def fallback_scene(content, error):

    return normalize_scene_json({
        "environment_type":
            "unknown",

        "objects":
            [],

        "hazards":
            [
                "qwen_json_parse_error"
            ],

        "navigation": {
            "aisle_detected":
                False,
            "walkable_region":
                "unknown",
            "obstacle_regions":
                []
        },

        "located_objects":
            [],

        "parse_error":
            str(
                error
            ),

        "raw_response_preview":
            content[:300]
    })


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
                ),

            "floor_region":
                normalize_path_points(
                    navigation.get(
                        "floor_region",
                        []
                    ),
                    include_speed=False
                ),

            "floor_region_source":
                (
                    "qwen_vlm"
                    if navigation.get(
                        "floor_region"
                    )
                    else None
                ),

            "desired_path":
                normalize_path_points(
                    navigation.get(
                        "desired_path",
                        []
                    ),
                    include_speed=True
                ),

            "desired_path_source":
                (
                    "qwen_vlm"
                    if navigation.get(
                        "desired_path"
                    )
                    else None
                )
        },

        "located_objects":
            normalize_located_objects(
                scene.get(
                    "located_objects",
                    []
                )
            ),

        "parse_error":
            scene.get(
                "parse_error"
            ),

        "raw_response_preview":
            scene.get(
                "raw_response_preview"
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

        label = normalize_label(
            obj.get(
                "label",
                "unknown"
            )
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
                normalize_label(
                    obj.get(
                        "label",
                        "unknown"
                    )
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


def normalize_path_points(points, include_speed=True):

    normalized = []

    for point in points:

        if not isinstance(point, dict):
            continue

        if "x" not in point or "y" not in point:
            continue

        try:
            x = float(
                point["x"]
            )
            y = float(
                point["y"]
            )
        except (TypeError, ValueError):
            continue

        normalized_point = {
            "x":
                x,

            "y":
                y
        }

        if include_speed:
            normalized_point["speed"] = normalize_speed(
                point.get(
                    "speed"
                )
            )

        normalized.append(
            normalized_point
        )

    return normalized


def normalize_speed(speed):

    if speed is None:
        return None

    try:
        value = float(
            speed
        )
    except (TypeError, ValueError):
        return None

    return max(
        0.0,
        min(
            1.0,
            value
        )
    )


def normalize_label(label):

    normalized = label.replace(
        " ",
        "_"
    )

    aliases = {
        "person":
            "human",
        "people":
            "human",
        "worker":
            "human",
        "operator":
            "human",
        "man":
            "human",
        "woman":
            "human"
    }

    return aliases.get(
        normalized,
        normalized
    )
