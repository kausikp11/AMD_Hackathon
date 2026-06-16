from src.loader import load_frame
from src.scene_understanding import understand_scene
from src.object_memory import ObjectMemory

memory = ObjectMemory()

frame = load_frame("013342")

scene = understand_scene(frame)

memory.update(
    scene["objects"],
    frame["frame_id"]
)

print(memory.summary())