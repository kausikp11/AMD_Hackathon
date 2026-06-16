import importlib.util
import os
import platform
from pathlib import Path


def has_module(name):

    return importlib.util.find_spec(name) is not None


def status(name, ok, detail=""):

    marker = "OK" if ok else "MISSING"
    suffix = f" - {detail}" if detail else ""
    print(f"{marker}: {name}{suffix}")


def main():

    root = Path(__file__).resolve().parents[1]
    data_root = root / "data" / "industrial_subset"
    frames_file = data_root / "frames.txt"

    print("Industrial Robot Copilot environment check")
    print(f"Python: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    print()

    status("dataset root", data_root.exists(), str(data_root))
    status("frames.txt", frames_file.exists(), str(frames_file))

    if frames_file.exists():
        frames = [
            line.strip()
            for line in frames_file.read_text().splitlines()
            if line.strip()
        ]
        print(f"Frame count: {len(frames)}")
        print(f"First frame: {frames[0] if frames else 'none'}")

    print()
    for module in [
        "gradio",
        "PIL",
        "pandas",
        "numpy",
        "openai"
    ]:
        status(f"python module {module}", has_module(module))

    print()
    status(
        "optional python module ultralytics",
        has_module("ultralytics"),
        "needed only for LOCATOR_BACKEND=yolo_live"
    )

    print()
    print("Runtime configuration:")
    for key in [
        "VLM_BACKEND",
        "LOCATOR_BACKEND",
        "YOLO_LIVE_MODEL",
        "YOLO_LIVE_DEVICE",
        "QWEN_VLM_BASE_URL",
        "QWEN_VLM_MODEL",
        "NVIDIA_LOCATE_ANYTHING_BASE_URL",
        "NVIDIA_LOCATE_ANYTHING_MODEL",
    ]:
        print(f"{key}={os.getenv(key, '')}")

    print()
    print("AMD final-demo recommendation:")
    print("VLM_BACKEND=qwen when Qwen is served on ROCm/vLLM.")
    print("LOCATOR_BACKEND=yolo_live when Ultralytics YOLO is installed.")
    print("LOCATOR_BACKEND=labels remains the deterministic fallback.")


if __name__ == "__main__":
    main()
