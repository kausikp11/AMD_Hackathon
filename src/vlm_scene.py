from collections import Counter
import os

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
            ]
        }

    return {
        "aisle_detected": True,
        "walkable_region": "center",
        "obstacle_regions": [
            "left",
            "right"
        ]
    }


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
            (
                vlm_scene.get(
                    "navigation"
                )
                if vlm_scene
                else build_navigation(
                    environment_type
                )
            ),

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
                []
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
        f"- Obstacles: "
        f"{', '.join(nav['obstacle_regions'])}"
    )

    return "\n".join(lines)
