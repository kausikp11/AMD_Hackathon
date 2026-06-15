from src.loader import load_frame
from src.fusion import fuse_frame

frame = load_frame("013342")

state = fuse_frame(frame)

print(state)