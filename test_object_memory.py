from src.object_memory import ObjectMemory

memory = ObjectMemory()

memory.update(
    [
        {
            "label": "machine"
        },
        {
            "label": "pipe"
        },
        {
            "label": "workbench"
        }
    ],
    frame_id="013345"
)

print(memory.summary())