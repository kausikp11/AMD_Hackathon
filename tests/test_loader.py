# test_loader.py

from src.loader import load_frame,frame_summary

frame = load_frame("013342")


print(frame_summary(frame))

print(type(frame))
print(type(frame["rgb_detections"]))

print(frame["rgb_detections"][:2])

print(frame["environment"])
print(frame["rgb_image"])
print(frame["thermal_image"])

print(
    "RGB detections:",
    len(frame["rgb_detections"])
)

print(
    "Thermal detections:",
    len(frame["thermal_detections"])
)