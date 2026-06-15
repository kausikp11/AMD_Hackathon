ACTION_MAP = {

    "STOP": {
        "speed": 0.0
    },

    "SLOW_DOWN": {
        "speed": 0.3
    },

    "PROCEED_WITH_CAUTION": {
        "speed": 0.6
    },

    "CONTINUE": {
        "speed": 1.0
    }
}


def get_action_command(action):

    return ACTION_MAP[action]