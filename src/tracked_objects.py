import json
from functools import lru_cache
from pathlib import Path


DEFAULT_OBJECTS = [
    "human",
    "industrial machine",
    "workbench",
    "pipe",
    "cabinet",
    "storage box",
    "control panel",
    "forklift",
    "robot",
]


@lru_cache(maxsize=1)
def load_tracked_objects():

    path = Path("config/tracked_objects.json")

    if not path.exists():
        return DEFAULT_OBJECTS

    try:
        data = json.loads(
            path.read_text()
        )
    except json.JSONDecodeError:
        return DEFAULT_OBJECTS

    objects = data.get(
        "track_objects",
        []
    )

    cleaned = [
        str(obj).strip()
        for obj in objects
        if str(obj).strip()
    ]

    return cleaned or DEFAULT_OBJECTS
