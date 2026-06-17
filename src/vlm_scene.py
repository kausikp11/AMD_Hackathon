from collections import Counter
import os
from PIL import Image

from src.locate_anything import locate_objects


def infer_environment(image_path=None, world=None):
    """
    Simple heuristic version.

    Later:
        Replace with Qwen2.5-VL.
    """

    if world is not None:

        weather = world["environment"]["weather"]

        if weather == "Indoor":
            return "machine_shop"

    return "industrial_factory"


def build_navigation(environment_type):

    if environment_type == "machine_shop":

        return {
            "aisle_detected": True,
            "walkable_region": "center",
            "obstacle_regions": [
                "left",
                "right"
            ],
            "floor_region": [],
            "floor_region_source": None,
            "desired_path": [],
            "desired_path_source": None
        }

    return {
        "aisle_detected": True,
        "walkable_region": "center",
        "obstacle_regions": [
            "left",
            "right"
        ],
        "floor_region": [],
        "floor_region_source": None,
        "desired_path": [],
        "desired_path_source": None
    }


def estimate_floor_region(image_path):

    with Image.open(image_path) as image:
        width, height = image.size

    return [
        {
            "x": width * 0.06,
            "y": height * 0.98
        },
        {
            "x": width * 0.94,
            "y": height * 0.98
        },
        {
            "x": width * 0.68,
            "y": height * 0.52
        },
        {
            "x": width * 0.32,
            "y": height * 0.52
        }
    ]


def estimate_desired_path(image_path, navigation, located_objects):

    with Image.open(image_path) as image:
        width, height = image.size

    lane = choose_best_lane(
        located_objects,
        (width, height)
    )

    route = navigation.get(
        "walkable_region",
        "center"
    )

    if route in {
        "left",
        "right"
    }:
        lane = route

    lane_x = {
        "left": width * 0.34,
        "center": width * 0.50,
        "right": width * 0.66
    }[lane]

    vanishing_x = width * 0.50
    start_x = vanishing_x
    control_x = (
        start_x * 0.35
        + lane_x * 0.65
    )
    end_x = (
        lane_x * 0.75
        + vanishing_x * 0.25
    )

    y_ratios = [
        0.96,
        0.84,
        0.74,
        0.66,
        0.58,
        0.52
    ]
    speeds = [
        0.0,
        0.25,
        0.45,
        0.55,
        0.65,
        0.75
    ]

    return [
        {
            "x": bezier_x(
                start_x,
                control_x,
                end_x,
                index / (len(y_ratios) - 1)
            ),
            "y": height * y_ratio,
            "speed": speeds[index]
        }
        for index, y_ratio in enumerate(
            y_ratios
        )
    ]


def bezier_x(start_x, control_x, end_x, ratio):

    inverse = 1.0 - ratio

    return (
        inverse
        * inverse
        * start_x
        + 2
        * inverse
        * ratio
        * control_x
        + ratio
        * ratio
        * end_x
    )


def choose_best_lane(located_objects, image_size):

    width, height = image_size
    lanes = {
        "left": (
            0.16,
            0.43
        ),
        "center": (
            0.34,
            0.66
        ),
        "right": (
            0.57,
            0.84
        )
    }
    scores = {
        "left": 0.0,
        "center": 0.0,
        "right": 0.0
    }

    for obj in located_objects:
        box = bbox_to_pixels(
            obj.get(
                "bbox"
            ),
            image_size
        )

        if not box or not blocks_navigation(
            obj
        ):
            continue

        x1, y1, x2, y2 = box
        x1_ratio = max(
            0.0,
            min(
                1.0,
                x1 / width
            )
        )
        x2_ratio = max(
            0.0,
            min(
                1.0,
                x2 / width
            )
        )
        vertical_weight = max(
            0.25,
            min(
                1.0,
                y2 / height
            )
        )
        label_weight = 1.8 if obj.get(
            "label"
        ) == "human" else 1.0

        for lane, bounds in lanes.items():
            overlap = interval_overlap(
                (
                    x1_ratio,
                    x2_ratio
                ),
                bounds
            )
            lane_width = bounds[1] - bounds[0]
            if overlap > 0:
                scores[lane] += (
                    overlap
                    / lane_width
                    * vertical_weight
                    * label_weight
                )

    return min(
        [
            "center",
            "left",
            "right"
        ],
        key=lambda lane: (
            scores[lane],
            lane != "center"
        )
    )


def interval_overlap(first, second):

    return max(
        0.0,
        min(
            first[1],
            second[1]
        )
        - max(
            first[0],
            second[0]
        )
    )


def blocks_navigation(obj):

    label = normalize_label(
        obj.get(
            "label",
            ""
        )
    )

    return label in {
        "human",
        "person",
        "industrial_machine",
        "machine",
        "workbench",
        "cart",
        "cabinet",
        "storage_box",
        "box",
        "crate",
        "pallet",
        "shelf",
        "rack",
        "safety_cone",
        "barrel",
        "toolbox",
        "forklift",
        "robot",
        "obstacle"
    }


def bbox_to_pixels(bbox, image_size):

    if not bbox:
        return None

    width, height = image_size

    if all(
        key in bbox
        for key in [
            "x1",
            "y1",
            "x2",
            "y2"
        ]
    ):
        x1 = float(
            bbox["x1"]
        )
        y1 = float(
            bbox["y1"]
        )
        x2 = float(
            bbox["x2"]
        )
        y2 = float(
            bbox["y2"]
        )
    elif all(
        key in bbox
        for key in [
            "cx",
            "cy",
            "w",
            "h"
        ]
    ):
        cx = float(
            bbox["cx"]
        ) * width
        cy = float(
            bbox["cy"]
        ) * height
        bw = float(
            bbox["w"]
        ) * width
        bh = float(
            bbox["h"]
        ) * height
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2
    else:
        return None

    return [
        x1,
        y1,
        x2,
        y2
    ]


def build_object_inventory(environment_type):

    if environment_type == "machine_shop":

        return [

            {
                "label": "industrial_machine",
                "count": 2
            },

            {
                "label": "workbench",
                "count": 1
            },

            {
                "label": "storage_box",
                "count": 2
            }
        ]

    return [

        {
            "label": "industrial_machine",
            "count": 4
        },

        {
            "label": "control_panel",
            "count": 2
        },

        {
            "label": "cart",
            "count": 1
        }
    ]


def normalize_label(label):

    return label.replace(
        " ",
        "_"
    )


def detection_source_items(frame):
    return locate_objects(frame)


def build_detected_objects(located_objects):

    source_counts = {}

    for obj in located_objects:

        source = obj["source"]

        if source not in source_counts:
            source_counts[source] = Counter()

        source_counts[source][obj["label"]] += 1

    labels = {
        label
        for counts in source_counts.values()
        for label in counts
        if label != "human"
    }

    return [
        {
            "label": label,
            "count": max(
                counts.get(
                    label,
                    0
                )
                for counts in source_counts.values()
            )
        }
        for label in sorted(
            labels
        )
    ]


def merge_objects(*object_groups):

    counts = Counter()

    for objects in object_groups:

        for obj in objects:

            counts[obj["label"]] += obj.get(
                "count",
                1
            )

    return [
        {
            "label": label,
            "count": count
        }
        for label, count in sorted(
            counts.items()
        )
    ]


def build_hazards(world):

    hazards = []

    human = world["human"]

    if human["present"]:

        if human["proximity"] == "very_near":
            hazards.append(
                "human_inside_safety_zone"
            )

        elif human["proximity"] == "near":
            hazards.append(
                "human_in_operational_zone"
            )

    return hazards


def describe_scene(frame, world):

    image_path = frame["rgb_image"]

    vlm_scene = describe_with_configured_vlm(
        image_path
    )

    environment_type = (
        vlm_scene.get(
            "environment_type"
        )
        if vlm_scene
        else infer_environment(
            image_path,
            world
        )
    )

    located_objects = detection_source_items(
        frame
    )

    qwen_located_objects = (
        vlm_scene.get(
            "located_objects",
            []
        )
        if vlm_scene
        else []
    )

    all_located_objects = (
        located_objects
        + qwen_located_objects
    )

    detected_objects = build_detected_objects(
        all_located_objects
    )

    inventory = (
        vlm_scene.get(
            "objects",
            []
        )
        if vlm_scene
        else build_object_inventory(
            environment_type
        )
    )

    navigation = (
        vlm_scene.get(
            "navigation"
        )
        if vlm_scene
        else build_navigation(
            environment_type
        )
    )

    if not navigation.get(
        "floor_region"
    ):
        navigation["floor_region"] = estimate_floor_region(
            image_path
        )
        navigation["floor_region_source"] = "heuristic_floor_projection"

    if not navigation.get(
        "desired_path"
    ):
        navigation["desired_path"] = estimate_desired_path(
            image_path,
            navigation,
            all_located_objects
        )
        navigation["desired_path_source"] = "heuristic_floor_projection"
    elif not navigation.get(
        "desired_path_source"
    ):
        navigation["desired_path_source"] = (
            "qwen_vlm"
            if vlm_scene
            else "heuristic_floor_projection"
        )

    if not navigation.get(
        "floor_region_source"
    ):
        navigation["floor_region_source"] = (
            "qwen_vlm"
            if vlm_scene
            else "heuristic_floor_projection"
        )

    scene = {
        "environment_type":
            environment_type,

        "objects":
            merge_objects(
                inventory,
                detected_objects
            ),

        "hazards":
            merge_hazards(
                vlm_scene.get(
                    "hazards",
                    []
                )
                if vlm_scene
                else [],
                build_hazards(world)
            ),

        "navigation":
            navigation,

        "located_objects":
            all_located_objects,

        "scene_source":
            "qwen_vlm"
            if vlm_scene
            else "heuristic"
    }

    return scene


def describe_with_configured_vlm(image_path):

    backend = os.getenv(
        "VLM_BACKEND",
        "heuristic"
    ).lower()

    if backend == "heuristic":
        return None

    if backend == "qwen":
        try:
            return describe_with_qwen(
                image_path
            )
        except Exception as exc:
            if os.getenv(
                "QWEN_STRICT",
                "0"
            ) == "1":
                raise

            return qwen_error_scene(
                exc
            )

    if backend == "auto":
        try:
            return describe_with_qwen(
                image_path
            )
        except Exception:
            return None

    return None


def describe_with_qwen(image_path):

    from src.qwen_vlm import QwenVLM

    return QwenVLM().describe(
        image_path
    )


def qwen_error_scene(exc):

    return {
        "environment_type":
            "unknown",

        "objects":
            [],

        "hazards":
            [
                "qwen_scene_error"
            ],

        "navigation": {
            "aisle_detected":
                False,
            "walkable_region":
                "unknown",
            "obstacle_regions":
                [],
            "desired_path":
                [],
            "desired_path_source":
                None,
            "floor_region":
                [],
            "floor_region_source":
                None
        },

        "located_objects":
            [],

        "parse_error":
            f"{type(exc).__name__}: {exc}"
    }


def merge_hazards(*hazard_groups):

    hazards = []
    seen = set()

    for group in hazard_groups:

        for hazard in group:

            if hazard in seen:
                continue

            hazards.append(
                hazard
            )
            seen.add(
                hazard
            )

    return hazards


def scene_summary(scene):

    lines = []

    lines.append(
        f"Environment: {scene['environment_type']}"
    )

    lines.append("\nObjects:")

    for obj in scene["objects"]:

        lines.append(
            f"- {obj['label']} "
            f"(x{obj['count']})"
        )

    lines.append("\nHazards:")

    if scene["hazards"]:

        for h in scene["hazards"]:

            lines.append(f"- {h}")

    else:

        lines.append("- none")

    nav = scene["navigation"]

    lines.append("\nNavigation:")

    lines.append(
        f"- Walkable: "
        f"{nav['walkable_region']}"
    )

    lines.append(
        f"- Path source: "
        f"{nav.get('desired_path_source', 'unknown')}"
    )

    lines.append(
        f"- Floor source: "
        f"{nav.get('floor_region_source', 'unknown')}"
    )

    lines.append(
        f"- Obstacles: "
        f"{', '.join(nav['obstacle_regions'])}"
    )

    return "\n".join(lines)
