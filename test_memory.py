# test_memory.py

from src.object_memory import ObjectMemory

memory = ObjectMemory()

memory.update(
    [
        {"label": "machine"},
        {"label": "workbench"},
        {"label": "pipe"}
    ],
    frame_id="013342"
)

print(memory.summary())