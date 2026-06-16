# src/reasoner.py

def reason(world):
    """
    Rule-based safety reasoner.

    Input:
        world = build_world_state(...)

    Output:
        {
            "risk_level": "...",
            "action": "...",
            "confidence": "...",
            "reasoning": "...",
            "triggered_tags": [...]
        }
    """

    tags = set(world["scene_tags"])

    risk_level = "LOW"
    action = "CONTINUE"
    confidence = "MEDIUM"

    reasons = []

    # --------------------------------------------------
    # CRITICAL
    # --------------------------------------------------

    if "critical_proximity" in tags:

        risk_level = "CRITICAL"
        action = "STOP"
        confidence = "HIGH"

        reasons.append(
            "Human detected within critical proximity zone."
        )

    # --------------------------------------------------
    # HIGH
    # --------------------------------------------------

    elif "caution_zone" in tags:

        risk_level = "HIGH"
        action = "SLOW_DOWN"
        confidence = (
            "HIGH"
            if "dynamic_environment" in tags
            else "MEDIUM"
        )

        reasons.append(
            "Human detected in caution zone."
        )

        if "dynamic_environment" in tags:

            reasons.append(
                "Radar indicates a dynamic environment."
            )

    # --------------------------------------------------
    # MEDIUM
    # --------------------------------------------------

    elif "monitor_distance" in tags:

        risk_level = "MEDIUM"
        action = "PROCEED_WITH_CAUTION"
        confidence = (
            "HIGH"
            if "dynamic_environment" in tags
            else "MEDIUM"
        )

        reasons.append(
            "Human detected at monitored distance."
        )

        if "dynamic_environment" in tags:

            reasons.append(
                "Radar indicates a dynamic environment."
            )

    elif "unknown_human_distance" in tags:

        risk_level = "MEDIUM"
        action = "PROCEED_WITH_CAUTION"
        confidence = "LOW"

        reasons.append(
            "Human detected, but distance is unavailable."
        )

        if "dynamic_environment" in tags:

            confidence = "MEDIUM"

            reasons.append(
                "Radar indicates a dynamic environment."
            )

    # --------------------------------------------------
    # LOW
    # --------------------------------------------------

    else:

        risk_level = "LOW"
        action = "CONTINUE"
        confidence = "MEDIUM"

        reasons.append(
            "No nearby human detected."
        )

    # --------------------------------------------------
    # Additional context
    # --------------------------------------------------

    if "multimodal_confirmation" in tags:

        reasons.append(
            "RGB and thermal sensors agree."
        )

    if "low_visibility" in tags:

        reasons.append(
            "Reduced visibility conditions."
        )

    if "indoor_environment" in tags:

        reasons.append(
            "Indoor operating environment."
        )

    return {

        "risk_level":
            risk_level,

        "action":
            action,

        "confidence":
            confidence,

        "reasoning":
            " ".join(reasons),

        "triggered_tags":
            sorted(list(tags))
    }


def reason_summary(decision):

    return f"""
Decision:
- Risk Level: {decision['risk_level']}
- Action: {decision['action']}
- Confidence: {decision['confidence']}

Reasoning:
{decision['reasoning']}
""".strip()
