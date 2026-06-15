import json
import os
import html as html_lib
from pathlib import Path
from urllib.parse import quote

import gradio as gr
from PIL import Image, ImageDraw

from src.pipeline import run_pipeline


DATA_ROOT = Path("data/industrial_subset")
FRAMES = [
    line.strip()
    for line in (DATA_ROOT / "frames.txt").read_text().splitlines()
    if line.strip()
]
DATA_ROOT_ABS = DATA_ROOT.resolve()


def find_image(folder, frame_id):

    candidates = [
        DATA_ROOT / folder / "calibrated" / f"{frame_id}.jpg",
        DATA_ROOT / folder / "images" / f"{frame_id}.jpg",
        DATA_ROOT / folder / "calibrated" / f"{frame_id}.png",
        DATA_ROOT / folder / "images" / f"{frame_id}.png",
    ]

    for path in candidates:
        if path.exists():
            return path.resolve()

    return None


def file_url(path):

    return "/gradio_api/file=" + quote(
        str(path),
        safe="/:"
    )


def frame_media():

    media = []

    for frame_id in FRAMES:

        rgb_path = find_image(
            "05_rgb",
            frame_id
        )
        thermal_path = find_image(
            "06_thermal",
            frame_id
        )

        if rgb_path is None or thermal_path is None:
            continue

        media.append({
            "id":
                frame_id,
            "rgb":
                file_url(rgb_path),
            "thermal":
                file_url(thermal_path)
        })

    return media


def live_stream_html():

    media_json = json.dumps(
        frame_media()
    )

    iframe = f"""
<!doctype html>
<html>
<head>
<style>
  body {{
    margin: 0;
    font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: #101418;
    color: #f5f7fa;
  }}
  .bar {{
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 12px;
    background: #171d23;
    border-bottom: 1px solid #2a333d;
  }}
  button {{
    height: 32px;
    padding: 0 12px;
    border: 1px solid #47515c;
    background: #222a32;
    color: #f5f7fa;
    cursor: pointer;
  }}
  button:hover {{
    background: #2d3742;
  }}
  input[type=number] {{
    width: 64px;
    height: 28px;
    background: #0f1419;
    color: #f5f7fa;
    border: 1px solid #47515c;
  }}
  input[type=range] {{
    flex: 1;
  }}
  .label {{
    min-width: 230px;
    font-size: 14px;
    font-variant-numeric: tabular-nums;
  }}
  .grid {{
    display: grid;
    grid-template-columns: 1fr 1fr 0.85fr;
    gap: 1px;
    background: #2a333d;
  }}
  .panel {{
    position: relative;
    background: #050608;
    height: 430px;
    overflow: hidden;
  }}
  .panel img {{
    width: 100%;
    height: 100%;
    object-fit: contain;
    display: block;
  }}
  .tag {{
    position: absolute;
    left: 10px;
    top: 10px;
    background: rgba(0, 0, 0, 0.65);
    padding: 4px 8px;
    font-size: 13px;
  }}
  .map {{
    position: relative;
    height: 430px;
    background:
      linear-gradient(90deg, rgba(255,255,255,0.06) 1px, transparent 1px),
      linear-gradient(rgba(255,255,255,0.06) 1px, transparent 1px),
      #0d1418;
    background-size: 48px 48px;
    overflow: hidden;
  }}
  .aisle {{
    position: absolute;
    left: 42%;
    top: 6%;
    width: 16%;
    height: 88%;
    background: rgba(86, 180, 233, 0.16);
    border: 1px solid rgba(86, 180, 233, 0.35);
  }}
  .machine {{
    position: absolute;
    width: 23%;
    height: 18%;
    background: rgba(230, 159, 0, 0.28);
    border: 1px solid rgba(230, 159, 0, 0.6);
  }}
  .robot {{
    position: absolute;
    width: 22px;
    height: 22px;
    border-radius: 50%;
    background: #00cc88;
    border: 3px solid #eafff7;
    box-shadow: 0 0 18px rgba(0, 204, 136, 0.75);
    transform: translate(-50%, -50%);
    transition: left 45ms linear, top 45ms linear;
  }}
  .path {{
    position: absolute;
    left: 49%;
    top: 10%;
    width: 2%;
    height: 80%;
    background: rgba(0, 204, 136, 0.28);
  }}
</style>
</head>
<body>
  <div class="bar">
    <button id="play">Play</button>
    <button id="pause">Pause</button>
    <button id="prev">Prev</button>
    <button id="next">Next</button>
    <label>FPS <input id="fps" type="number" min="1" max="30" value="20"></label>
    <div id="status" class="label"></div>
    <input id="timeline" type="range" min="0" value="0">
  </div>
  <div class="grid">
    <div class="panel">
      <div class="tag">RGB</div>
      <img id="rgb" alt="RGB stream">
    </div>
    <div class="panel">
      <div class="tag">Thermal</div>
      <img id="thermal" alt="Thermal stream">
    </div>
    <div class="map">
      <div class="tag">Robot Movement</div>
      <div class="aisle"></div>
      <div class="path"></div>
      <div class="machine" style="left:8%; top:12%;"></div>
      <div class="machine" style="right:8%; top:26%;"></div>
      <div class="machine" style="left:10%; bottom:12%;"></div>
      <div class="machine" style="right:9%; bottom:16%;"></div>
      <div id="robot" class="robot"></div>
    </div>
  </div>
  <script>
    const frames = {media_json};
    let index = 0;
    let timer = null;

    const rgb = document.getElementById("rgb");
    const thermal = document.getElementById("thermal");
    const robot = document.getElementById("robot");
    const status = document.getElementById("status");
    const timeline = document.getElementById("timeline");
    const fpsInput = document.getElementById("fps");

    timeline.max = Math.max(frames.length - 1, 0);

    function show(i) {{
      if (!frames.length) return;
      index = (i + frames.length) % frames.length;
      const frame = frames[index];
      rgb.src = frame.rgb;
      thermal.src = frame.thermal;
      timeline.value = index;
      status.textContent = `Frame ${{index + 1}}/${{frames.length}} | ID ${{frame.id}}`;
      const phase = frames.length > 1 ? index / (frames.length - 1) : 0;
      const y = 12 + phase * 76;
      const x = 50 + Math.sin(phase * Math.PI * 4) * 5;
      robot.style.left = `${{x}}%`;
      robot.style.top = `${{y}}%`;

      const next = frames[(index + 1) % frames.length];
      if (next) {{
        const preloadRgb = new Image();
        const preloadThermal = new Image();
        preloadRgb.src = next.rgb;
        preloadThermal.src = next.thermal;
      }}
    }}

    function intervalMs() {{
      const fps = Math.max(1, Math.min(30, Number(fpsInput.value) || 20));
      return 1000 / fps;
    }}

    function play() {{
      pause();
      timer = setInterval(() => show(index + 1), intervalMs());
    }}

    function pause() {{
      if (timer) clearInterval(timer);
      timer = null;
    }}

    document.getElementById("play").onclick = play;
    document.getElementById("pause").onclick = pause;
    document.getElementById("prev").onclick = () => show(index - 1);
    document.getElementById("next").onclick = () => show(index + 1);
    fpsInput.onchange = () => {{
      if (timer) play();
    }};
    timeline.oninput = () => show(Number(timeline.value));

    show(0);
    play();
  </script>
</body>
</html>
"""

    return (
        '<iframe style="width:100%; height:492px; border:0;" srcdoc="'
        + html_lib.escape(
            iframe,
            quote=True
        )
        + '"></iframe>'
    )


def clamp_index(index):

    return max(
        0,
        min(
            int(index),
            len(FRAMES) - 1
        )
    )


def frame_index(frame_id):

    try:
        return FRAMES.index(
            frame_id
        )
    except ValueError:
        return 0


def analyze_index(index):

    index = clamp_index(
        index
    )

    frame_id = FRAMES[index]
    result = run_pipeline(
        frame_id
    )

    frame = result["frame"]
    world = result["world"]
    decision = result["decision"]
    scene = result["scene"]
    plan = result["plan"]

    rgb = Image.open(
        frame["rgb_image"]
    ).convert("RGB")

    thermal = Image.open(
        frame["thermal_image"]
    ).convert("RGB")

    rgb = draw_detections(
        rgb,
        scene.get(
            "located_objects",
            []
        ),
        target="rgb"
    )

    thermal = draw_detections(
        thermal,
        scene.get(
            "located_objects",
            []
        ),
        target="thermal"
    )

    status = demo_status(
        index,
        frame_id,
        world,
        decision,
        plan,
        scene
    )

    return (
        index,
        index,
        frame_id,
        status,
        rgb,
        thermal,
        json.dumps(
            world,
            indent=2
        ),
        json.dumps(
            decision,
            indent=2
        ),
        json.dumps(
            scene,
            indent=2
        ),
        json.dumps(
            plan,
            indent=2
        ),
        result["explanation"]
    )


def draw_detections(image, detections, target):

    output = image.copy()
    draw = ImageDraw.Draw(output)

    for det in detections:

        source = det.get(
            "source",
            "unknown"
        )

        if target == "thermal" and source != "thermal":
            continue

        if target == "rgb" and source == "thermal":
            continue

        box = bbox_to_pixels(
            det.get(
                "bbox"
            ),
            output.size
        )

        if box is None:
            continue

        color = source_color(
            source
        )

        label = (
            f"{det.get('label', 'object')} "
            f"[{source_label(source)}]"
        )

        draw.rectangle(
            box,
            outline=color,
            width=4
        )

        text_box = draw.textbbox(
            (box[0], box[1]),
            label
        )

        pad = 4
        background = [
            text_box[0] - pad,
            text_box[1] - pad,
            text_box[2] + pad,
            text_box[3] + pad
        ]

        draw.rectangle(
            background,
            fill=color
        )

        draw.text(
            (box[0], box[1]),
            label,
            fill=(0, 0, 0)
        )

    return output


def bbox_to_pixels(bbox, image_size):

    if not bbox:
        return None

    width, height = image_size

    if all(
        key in bbox
        for key in [
            "x1",
            "y1",
            "x2",
            "y2"
        ]
    ):
        x1 = float(
            bbox["x1"]
        )
        y1 = float(
            bbox["y1"]
        )
        x2 = float(
            bbox["x2"]
        )
        y2 = float(
            bbox["y2"]
        )
    elif all(
        key in bbox
        for key in [
            "cx",
            "cy",
            "w",
            "h"
        ]
    ):
        cx = float(
            bbox["cx"]
        ) * width
        cy = float(
            bbox["cy"]
        ) * height
        bw = float(
            bbox["w"]
        ) * width
        bh = float(
            bbox["h"]
        ) * height
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2
    else:
        return None

    x1 = max(
        0,
        min(
            width - 1,
            x1
        )
    )
    y1 = max(
        0,
        min(
            height - 1,
            y1
        )
    )
    x2 = max(
        0,
        min(
            width - 1,
            x2
        )
    )
    y2 = max(
        0,
        min(
            height - 1,
            y2
        )
    )

    if x2 <= x1 or y2 <= y1:
        return None

    return [
        x1,
        y1,
        x2,
        y2
    ]


def source_color(source):

    if source == "grounding_dino":
        return (86, 180, 233)

    if source == "qwen_vlm":
        return (0, 204, 136)

    if source == "thermal":
        return (213, 94, 0)

    if source == "rgb":
        return (240, 228, 66)

    return (204, 121, 167)


def source_label(source):

    labels = {
        "grounding_dino": "DINO",
        "qwen_vlm": "Qwen",
        "rgb": "RGB label",
        "thermal": "Thermal label",
        "nvidia_locateanything_transformers": "LocateAnything",
        "nvidia_locateanything_vllm": "LocateAnything"
    }

    return labels.get(
        source,
        source
    )


def demo_status(
    index,
    frame_id,
    world,
    decision,
    plan,
    scene
):

    human = world["human"]
    distance = human["distance"]
    distance_text = (
        "n/a"
        if distance is None
        else f"{distance:.2f} m"
    )

    return (
        f"Frame {index + 1}/{len(FRAMES)} | "
        f"ID {frame_id} | "
        f"Risk {decision['risk_level']} | "
        f"Action {plan['action']} | "
        f"Human {human['present']} | "
        f"Distance {distance_text} | "
        f"Scene {scene.get('scene_source', 'unknown')}"
    )


def analyze_frame_id(frame_id):

    return analyze_index(
        frame_index(
            frame_id
        )
    )


def next_frame(index):

    return analyze_index(
        (clamp_index(index) + 1)
        % len(FRAMES)
    )


def previous_frame(index):

    return analyze_index(
        (clamp_index(index) - 1)
        % len(FRAMES)
    )


def jump_to_index(index):

    return analyze_index(
        index
    )


def timer_tick(index, playing):

    if not playing:
        return analyze_index(
            index
        )

    return next_frame(
        index
    )


def set_playing(value):

    return value


def timer_interval(speed):

    return gr.Timer(
        value=float(speed),
        active=True
    )


OUTPUTS = []


with gr.Blocks() as demo:

    gr.Markdown(
        "# Industrial Robot Copilot"
    )

    gr.HTML(
        live_stream_html()
    )

    gr.Markdown(
        "Use the live stream above for motion. Use the controls below to inspect one frame through the robot copilot pipeline."
    )

    frame_index_state = gr.State(
        value=0
    )

    status_box = gr.Textbox(
        label="Demo Status",
        interactive=False
    )

    with gr.Row():

        frame_id = gr.Dropdown(
            choices=FRAMES,
            value=FRAMES[0],
            label="Frame ID",
            scale=2
        )

        frame_slider = gr.Slider(
            minimum=0,
            maximum=len(FRAMES) - 1,
            value=0,
            step=1,
            label="Timeline",
            scale=4
        )

    with gr.Row():

        previous_btn = gr.Button(
            "Previous"
        )

        analyze_btn = gr.Button(
            "Analyze"
        )

        next_btn = gr.Button(
            "Next"
        )

    with gr.Row():

        rgb_image = gr.Image(
            label="RGB"
        )

        thermal_image = gr.Image(
            label="Thermal"
        )

    world_box = gr.Textbox(
        label="World State",
        lines=12
    )

    decision_box = gr.Textbox(
        label="Risk Decision",
        lines=7
    )

    scene_box = gr.Textbox(
        label="Scene Understanding",
        lines=10
    )

    plan_box = gr.Textbox(
        label="Robot Plan",
        lines=8
    )

    explanation_box = gr.Textbox(
        label="Copilot Explanation",
        lines=8
    )

    OUTPUTS = [
        frame_index_state,
        frame_slider,
        frame_id,
        status_box,
        rgb_image,
        thermal_image,
        world_box,
        decision_box,
        scene_box,
        plan_box,
        explanation_box
    ]

    demo.load(
        analyze_index,
        inputs=[
            frame_index_state
        ],
        outputs=OUTPUTS
    )

    frame_id.change(
        analyze_frame_id,
        inputs=[
            frame_id
        ],
        outputs=OUTPUTS
    )

    frame_slider.release(
        jump_to_index,
        inputs=[
            frame_slider
        ],
        outputs=OUTPUTS
    )

    analyze_btn.click(
        analyze_frame_id,
        inputs=[
            frame_id
        ],
        outputs=OUTPUTS
    )

    previous_btn.click(
        previous_frame,
        inputs=[
            frame_index_state
        ],
        outputs=OUTPUTS
    )

    next_btn.click(
        next_frame,
        inputs=[
            frame_index_state
        ],
        outputs=OUTPUTS
    )


demo.launch(
    server_name=os.getenv(
        "GRADIO_SERVER_NAME",
        "0.0.0.0"
    ),
    server_port=int(
        os.getenv(
            "GRADIO_SERVER_PORT",
            "7860"
        )
    ),
    allowed_paths=[
        str(DATA_ROOT_ABS)
    ], share=True
)
