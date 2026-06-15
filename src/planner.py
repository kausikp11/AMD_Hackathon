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


def determine_navigation(scene):

    nav = scene["navigation"]

    if nav["aisle_detected"]:

        return {
            "route": nav["walkable_region"],
            "path_status": "available"
        }

    return {
        "route": None,
        "path_status": "unknown"
    }


def plan(
    world,
    decision,
    scene,
    graph
):

    risk = decision["risk_level"]

    planner_output = {

        "mode":
            determine_mode(risk),

        "action":
            decision["action"],

        "target_speed":
            determine_speed(risk),

        "goal":
            determine_goal(world),

        "navigation":
            determine_navigation(scene),

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

