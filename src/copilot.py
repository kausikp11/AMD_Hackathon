def generate_explanation(
    world,
    scene,
    plan
):

    human = world["human"]
    radar = world["scene"]

    lines = []

    # ----------------------------
    # Human
    # ----------------------------

    if human["present"]:

        if human["distance"] is None:

            lines.append(
                "Human detected, but distance is unavailable."
            )

        else:

            lines.append(

                f"Human detected at "
                f"{human['distance']:.2f} meters."
            )

        if human["thermal_confirmed"]:

            lines.append(
                "Thermal and RGB sensors "
                "agree on the detection."
            )

    else:

        lines.append(
            "No human detected."
        )

    # ----------------------------
    # Radar
    # ----------------------------

    if radar["radar_activity"]:

        lines.append(

            f"Radar indicates a "
            f"{radar['motion_level']} "
            f"activity environment."
        )

    # ----------------------------
    # Environment
    # ----------------------------

    lines.append(

        f"Operating in a "
        f"{scene['environment_type']}."
    )

    # ----------------------------
    # Plan
    # ----------------------------

    lines.append(

        f"Robot mode is "
        f"{plan['mode']}."
    )

    lines.append(

        f"Recommended action is "
        f"{plan['action']}."
    )

    lines.append(

        f"Target speed is "
        f"{plan['target_speed']}."
    )

    return "\n".join(lines)
