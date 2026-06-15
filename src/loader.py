from pathlib import Path

# --------------------------------------------------
# Constants
# --------------------------------------------------

CLASS_MAP = {
    0: "human",
    1: "bicycle",
    2: "toy_car",
    3: "doll",
}


# --------------------------------------------------
# Detection Parser
# --------------------------------------------------

def load_detections(label_file):
    """
    Parse YOLO distance labels.

    Format:
    class cx cy w h distance
    """

    label_file = Path(label_file)

    detections = []

    if not label_file.exists():
        return detections

    with open(label_file, "r") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 6:
                continue

            cls_id = int(parts[0])

            detections.append({
                "class_id": cls_id,
                "class_name": CLASS_MAP.get(
                    cls_id,
                    f"class_{cls_id}"
                ),

                "bbox": {
                    "cx": float(parts[1]),
                    "cy": float(parts[2]),
                    "w": float(parts[3]),
                    "h": float(parts[4]),
                },

                "distance": float(parts[5])
            })

    return detections

def load_metadata(meta_file):

    meta_file = Path(meta_file)

    result = {
        "season": None,
        "weather": None,
        "lighting": None,
        "dataset": None
    }

    if not meta_file.exists():
        return result

    with open(meta_file, "r") as f:

        lines = [
            line.strip()
            for line in f.readlines()
            if line.strip()
        ]

    if len(lines) > 0:
        result["season"] = lines[0]

    if len(lines) > 1:
        result["weather"] = lines[1]

    if len(lines) > 2:
        result["lighting"] = lines[2]

    if len(lines) > 3:
        result["dataset"] = lines[3]

    return result

def find_image(folder, frame_id):

    candidates = [
        folder / "calibrated" / f"{frame_id}.jpg",
        folder / "images" / f"{frame_id}.jpg",
        folder / "calibrated" / f"{frame_id}.png",
        folder / "images" / f"{frame_id}.png",
    ]

    for path in candidates:
        if path.exists():
            return str(path)

    return None

def get_radar_files(root, frame_id):

    radar_root = root / "04_radar"

    return {

        "csv":
            str(
                radar_root /
                "pointcloud" /
                "csv" /
                f"{frame_id}.csv"
            ),

        "pcd":
            str(
                radar_root /
                "pointcloud" /
                "pcd" /
                f"{frame_id}.pcd"
            ),

        "doppler_abs":
            str(
                radar_root /
                "doppler_abs" /
                f"{frame_id}.png"
            )
    }

def load_frame(
    frame_id,
    data_root="data/industrial_subset"
):
    """
    Load complete multimodal state
    for a single frame.
    """

    root = Path(data_root)

    frame = {

        "frame_id": frame_id,

        "environment": {},

        "rgb_image": None,

        "thermal_image": None,

        "rgb_detections": [],

        "thermal_detections": [],

        "radar_files": {}
    }

    # Metadata
    frame["environment"] = load_metadata(
        root /
        "00_meta_data" /
        f"{frame_id}.txt"
    )

    # RGB detections
    frame["rgb_detections"] = load_detections(
        root /
        "02_yolo_rgb_distance" /
        f"{frame_id}.txt"
    )

    # Thermal detections
    frame["thermal_detections"] = load_detections(
        root /
        "02_yolo_thermal_distance" /
        f"{frame_id}.txt"
    )

    # Images
    frame["rgb_image"] = find_image(
        root / "05_rgb",
        frame_id
    )

    frame["thermal_image"] = find_image(
        root / "06_thermal",
        frame_id
    )

    frame["rgb_path"] = frame["rgb_image"]
    frame["thermal_path"] = frame["thermal_image"]

    # Radar
    frame["radar_files"] = get_radar_files(
        root,
        frame_id
    )

    frame["radar_csv"] = frame["radar_files"]["csv"]

    return frame

def nearest_detection(detections,class_name=None):

    humans = [d for d in detections if d["class_name"] == "human"]

    if not humans:
        return None

    return min(
        humans,
        key=lambda x: x["distance"]
    )

def nearest_human(frame):

    return nearest_detection(
        frame["rgb_detections"],
        "human"
    )

def human_present(detections):

    return any(
        d["class_name"] == "human"
        for d in detections
    )

def has_human(detections):

    return any(
        d["class_name"] == "human"
        for d in detections
    )

def frame_summary(frame):

    human = nearest_human(frame)

    return {
        "frame_id": frame["frame_id"],
        "season": frame["environment"]["season"],
        "lighting": frame["environment"]["lighting"],
        "human_present": human is not None,
        "nearest_human_distance":
            None if human is None
            else human["distance"]
    }
