def determine_speed(risk_level):

    if risk_level == "CRITICAL":
        return 0.0

    if risk_level == "HIGH":
        return 0.25

    if risk_level == "MEDIUM":
        return 0.50

    return 1.0


def determine_mode(risk_level):

    if risk_level == "CRITICAL":
        return "EMERGENCY_STOP"

    if risk_level == "HIGH":
        return "CAUTIOUS"

    if risk_level == "MEDIUM":
        return "MONITOR"

    return "NORMAL"


def determine_goal(world):

    if world["human"]["present"]:
        return "continue_monitoring"

    return "continue_patrol"


def determine_navigation(scene, target_speed):

    nav = scene["navigation"]
    desired_path = apply_speed_profile(
        nav.get(
            "desired_path",
            []
        ),
        target_speed
    )

    if nav["aisle_detected"]:

        return {
            "route": nav["walkable_region"],
            "path_status": "available",
            "desired_path": desired_path
        }

    return {
        "route": None,
        "path_status": "unknown",
        "desired_path": desired_path
    }


def apply_speed_profile(path, target_speed):

    if not path:
        return []

    target_speed = max(
        0.0,
        min(
            1.0,
            float(
                target_speed
            )
        )
    )

    if target_speed == 0.0:
        return [
            {
                **path[0],
                "speed": 0.0
            }
        ]

    max_index = max(
        len(path) - 1,
        1
    )
    profiled = []

    for index, point in enumerate(path):
        ratio = index / max_index
        fallback_speed = target_speed * ratio
        speed = point.get(
            "speed",
            fallback_speed
        )

        if speed is None:
            speed = fallback_speed

        profiled.append({
            **point,
            "speed":
                max(
                    0.0,
                    min(
                        target_speed,
                        float(
                            speed
                        )
                    )
                )
        })

    return profiled


def plan(
    world,
    decision,
    scene,
    graph
):

    risk = decision["risk_level"]

    target_speed = determine_speed(risk)

    planner_output = {

        "mode":
            determine_mode(risk),

        "action":
            decision["action"],

        "target_speed":
            target_speed,

        "goal":
            determine_goal(world),

        "navigation":
            determine_navigation(
                scene,
                target_speed
            ),

        "reasoning":
            decision["reasoning"],

        "scene_tags":
            world["scene_tags"]
    }

    return planner_output

def planner_summary(plan):

    nav = plan["navigation"]

    return f"""
Mode:
- {plan['mode']}

Action:
- {plan['action']}

Target Speed:
- {plan['target_speed']}

Goal:
- {plan['goal']}

Navigation:
- Route: {nav['route']}
- Status: {nav['path_status']}

Reasoning:
- {plan['reasoning']}
""".strip()
