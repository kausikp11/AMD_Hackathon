from src.fusion import fuse_frame

def classify_proximity(distance):

    if distance is None:
        return "unknown"

    if distance < 0.7:
        return "very_near"

    if distance < 1.95:
        return "near"

    if distance < 3.2:
        return "medium"

    return "far"

def classify_motion(motion_ratio):

    if motion_ratio < 0.15:
        return "low"

    if motion_ratio < 0.70:
        return "moderate"

    return "high"

def build_world_state(frame):

    state = fuse_frame(frame)

    human_distance = state["human_distance"]

    radar_summary = state.get("radar_summary", {})

    world = {

        "timestamp": state["frame_id"],

        "environment": {
            "season":
                frame["environment"]["season"],

            "weather":
                frame["environment"]["weather"],

            "lighting":
                frame["environment"]["lighting"]
        },

        "human": {

            "present":
                state["human_detected"],

            "distance":
                human_distance,

            "proximity":
                classify_proximity(
                    human_distance
                ),

            "thermal_confirmed":
                state["thermal_confirmed"],

            "source":
                state.get(
                    "human_source"
                )
        },

        "scene": {

            "radar_activity":
                state["radar_activity"],

            "motion_ratio":
                state["motion_ratio"],

            "motion_level":
                classify_motion(
                    state["motion_ratio"]
                ),

            "moving_points": 
            radar_summary.get("moving_points", 0),
            "max_velocity": 
            radar_summary.get("max_velocity", 0.0)
        },

        "confidence": {

            "rgb_detection":
                state["human_detected"],

            "thermal_detection":
                state["thermal_confirmed"],

            "multimodal_agreement":
                (
                    state["human_detected"]
                    and
                    state["thermal_confirmed"]
                )
        }
    }
    
    world["scene_tags"] = classify_scene(world)


    return world

def classify_scene(world):

    human = world["human"]
    scene = world["scene"]

    tags = []

    if human["present"]:
        tags.append("human_present")

    if (human["present"] and human["thermal_confirmed"]):
        tags.append("multimodal_confirmation")

    if human["proximity"] == "far":
        tags.append("safe_separation")

    if human["proximity"] == "medium":
        tags.append("monitor_distance")

    if human["proximity"] == "near":
        tags.append("caution_zone")

    if human["proximity"] == "very_near":
        tags.append("critical_proximity")

    if human["present"] and human["proximity"] == "unknown":
        tags.append("unknown_human_distance")

    if scene["motion_level"] == "high":
        tags.append("dynamic_environment")

    if scene["motion_level"] == "low":
        tags.append("static_environment")

    env = world["environment"]

    if env["lighting"] == "Dawn":
        tags.append("low_visibility")

    if env["lighting"] == "Day":
        tags.append("good_visibility")

    if env["weather"] == "Indoor":
        tags.append("indoor_environment")

    if env["weather"] == "Overcast":
        tags.append("reduced_visibility")    

    return tags

def world_state_summary(world):

    human = world["human"]
    scene = world["scene"]
    env = world["environment"]

    tags_text = "\n".join(f"- {tag}" for tag in world["scene_tags"])

    return f"""
Environment:
- {env['weather']}
- {env['lighting']}
- {env['season']}

Human:
- Present: {human['present']}
- Distance: {human['distance']}
- Proximity: {human['proximity']}
- Thermal Confirmed: {human['thermal_confirmed']}

Scene:
- Motion Level: {scene['motion_level']}
- Motion Ratio: {scene['motion_ratio']:.2f}
- Moving Radar Points: {scene['moving_points']}
- Max Velocity: {scene['max_velocity']:.2f}

Tags:
{tags_text}
""".strip()

def llm_context(world):

    return {
        "environment": world["environment"],
        "human": world["human"],
        "scene": {
            "motion_level": world["scene"]["motion_level"],
            "moving_points": world["scene"]["moving_points"],
            "max_velocity": world["scene"]["max_velocity"]
        },
        "scene_tags": world["scene_tags"]
    }
