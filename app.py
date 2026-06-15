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
DEMO_FRAME_COUNT = os.getenv(
    "DEMO_FRAME_COUNT",
    "10"
)
DEMO_FRAMES = (
    FRAMES
    if DEMO_FRAME_COUNT.lower() in {"all", "0", ""}
    else FRAMES[:int(DEMO_FRAME_COUNT)]
)
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

    for frame_id in DEMO_FRAMES:

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


def qwen_stream_records():

    cache_path = Path(
        os.getenv(
            "QWEN_STREAM_CACHE",
            "cache/qwen_stream.json"
        )
    )

    if not cache_path.exists():
        return {}

    try:
        payload = json.loads(
            cache_path.read_text()
        )
    except json.JSONDecodeError:
        return {}

    return {
        record["frame_id"]: record
        for record in payload.get(
            "records",
            []
        )
    }


def live_stream_html():

    media_json = json.dumps(
        frame_media()
    )
    qwen_json = json.dumps(
        qwen_stream_records()
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
  .qwen {{
    padding: 8px 12px;
    background: #101820;
    border-bottom: 1px solid #2a333d;
    color: #dfe7ef;
    font-size: 14px;
    font-variant-numeric: tabular-nums;
  }}
  .grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1px;
    background: #2a333d;
  }}
  .panel {{
    position: relative;
    background: #050608;
    height: 430px;
    overflow: hidden;
  }}
  .panel canvas {{
    width: 100%;
    height: 100%;
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
  <div id="qwenStatus" class="qwen">Qwen stream cache: not loaded</div>
  <div class="grid">
    <div class="panel">
      <div class="tag">RGB</div>
      <canvas id="rgb"></canvas>
    </div>
    <div class="panel">
      <div class="tag">Thermal</div>
      <canvas id="thermal"></canvas>
    </div>
  </div>
  <script>
    const frames = {media_json};
    const qwenRecords = {qwen_json};
    let index = 0;
    let playing = true;
    let lastFrameTime = 0;
    const cache = new Map();

    const rgbCanvas = document.getElementById("rgb");
    const thermalCanvas = document.getElementById("thermal");
    const rgbCtx = rgbCanvas.getContext("2d");
    const thermalCtx = thermalCanvas.getContext("2d");
    const status = document.getElementById("status");
    const qwenStatus = document.getElementById("qwenStatus");
    const timeline = document.getElementById("timeline");
    const fpsInput = document.getElementById("fps");

    timeline.max = Math.max(frames.length - 1, 0);

    function resizeCanvas(canvas) {{
      const rect = canvas.getBoundingClientRect();
      const scale = window.devicePixelRatio || 1;
      const width = Math.max(1, Math.floor(rect.width * scale));
      const height = Math.max(1, Math.floor(rect.height * scale));
      if (canvas.width !== width || canvas.height !== height) {{
        canvas.width = width;
        canvas.height = height;
      }}
    }}

    function loadImage(url) {{
      if (cache.has(url)) return cache.get(url);
      const img = new Image();
      img.decoding = "async";
      img.src = url;
      cache.set(url, img);
      return img;
    }}

    function preloadAround(i) {{
      for (let offset = 0; offset < 8; offset++) {{
        const frame = frames[(i + offset) % frames.length];
        loadImage(frame.rgb);
        loadImage(frame.thermal);
      }}
    }}

    function drawContain(ctx, canvas, img) {{
      resizeCanvas(canvas);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#050608";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      if (!img.complete || !img.naturalWidth) return;

      const scale = Math.min(
        canvas.width / img.naturalWidth,
        canvas.height / img.naturalHeight
      );
      const width = img.naturalWidth * scale;
      const height = img.naturalHeight * scale;
      const x = (canvas.width - width) / 2;
      const y = (canvas.height - height) / 2;
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(img, x, y, width, height);
    }}

    function show(i) {{
      if (!frames.length) return;
      index = (i + frames.length) % frames.length;
      const frame = frames[index];
      const rgb = loadImage(frame.rgb);
      const thermal = loadImage(frame.thermal);

      drawContain(rgbCtx, rgbCanvas, rgb);
      drawContain(thermalCtx, thermalCanvas, thermal);

      if (!rgb.complete) rgb.onload = () => drawContain(rgbCtx, rgbCanvas, rgb);
      if (!thermal.complete) thermal.onload = () => drawContain(thermalCtx, thermalCanvas, thermal);

      timeline.value = index;
      status.textContent = `Realtime stream | Frame ${{index + 1}}/${{frames.length}} | ID ${{frame.id}}`;
      const qwen = qwenRecords[frame.id];
      if (qwen) {{
        const distance = qwen.human_distance === null || qwen.human_distance === undefined
          ? "n/a"
          : `${{Number(qwen.human_distance).toFixed(2)}} m`;
        qwenStatus.textContent =
          `Qwen/cache | Scene ${{qwen.scene_source}} | Risk ${{qwen.risk_level}} | ` +
          `Action ${{qwen.action}} | Human ${{qwen.human_present}} | ` +
          `Distance ${{distance}} | Objects ${{qwen.located_count}}`;
      }} else {{
        qwenStatus.textContent =
          "Qwen/cache | no cached output for this frame. Run scripts/cache_qwen_outputs.py to sync model output with the stream.";
      }}
      preloadAround(index + 1);
    }}

    function intervalMs() {{
      const fps = Math.max(1, Math.min(30, Number(fpsInput.value) || 20));
      return 1000 / fps;
    }}

    function play() {{
      playing = true;
    }}

    function pause() {{
      playing = false;
    }}

    function loop(timestamp) {{
      if (playing && timestamp - lastFrameTime >= intervalMs()) {{
        show(index + 1);
        lastFrameTime = timestamp;
      }}
      requestAnimationFrame(loop);
    }}

    document.getElementById("play").onclick = play;
    document.getElementById("pause").onclick = pause;
    document.getElementById("prev").onclick = () => show(index - 1);
    document.getElementById("next").onclick = () => show(index + 1);
    timeline.oninput = () => show(Number(timeline.value));
    window.onresize = () => show(index);

    show(0);
    requestAnimationFrame(loop);
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
            len(DEMO_FRAMES) - 1
        )
    )


def frame_index(frame_id):

    try:
        return DEMO_FRAMES.index(
            frame_id
        )
    except ValueError:
        return 0


def analyze_index(index):

    index = clamp_index(
        index
    )

    frame_id = DEMO_FRAMES[index]
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

    robot_command = robot_command_text(
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
        robot_command,
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


def error_outputs(index, message):

    index = clamp_index(
        index
    )
    frame_id = DEMO_FRAMES[index]

    return (
        index,
        index,
        frame_id,
        f"Frame {index + 1}/{len(DEMO_FRAMES)} | ID {frame_id} | ERROR",
        "Decision source: unavailable\n"
        "Scene source: unavailable\n"
        f"Reason: {message}",
        None,
        None,
        "",
        "",
        "",
        "",
        message
    )


def draw_detections(image, detections, target):

    output = image.copy()
    draw = ImageDraw.Draw(output)

    for det in detections:

        source = det.get(
            "source",
            "unknown"
        )

        if target == "thermal" and source not in {
            "thermal",
            "thermal_fallback",
            "yolo_thermal",
            "yolo_thermal_fallback"
        }:
            continue

        if target == "rgb" and source in {
            "thermal",
            "thermal_fallback",
            "yolo_thermal",
            "yolo_thermal_fallback"
        }:
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
        return (204, 121, 255)

    if source in {
        "thermal",
        "yolo_thermal"
    }:
        return (213, 94, 0)

    if source in {
        "thermal_fallback",
        "yolo_thermal_fallback"
    }:
        return (213, 94, 0)

    if source in {
        "rgb",
        "yolo_rgb"
    }:
        return (240, 228, 66)

    if source in {
        "rgb_fallback",
        "yolo_rgb_fallback"
    }:
        return (240, 228, 66)

    return (204, 121, 167)


def source_label(source):

    labels = {
        "grounding_dino": "DINO",
        "qwen_vlm": "Qwen",
        "rgb": "RGB label",
        "rgb_fallback": "RGB fallback",
        "thermal": "Thermal label",
        "thermal_fallback": "Thermal fallback",
        "yolo_rgb": "YOLO RGB",
        "yolo_rgb_fallback": "YOLO RGB fallback",
        "yolo_thermal": "YOLO thermal",
        "yolo_thermal_fallback": "YOLO thermal fallback",
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
        f"Frame {index + 1}/{len(DEMO_FRAMES)} | "
        f"ID {frame_id} | "
        f"Risk {decision['risk_level']} | "
        f"Action {plan['action']} | "
        f"Human {human['present']} | "
        f"Distance {distance_text} | "
        f"Scene {scene.get('scene_source', 'unknown')}"
    )


def robot_command_text(world, decision, plan, scene):

    action = plan["action"]
    human = world["human"]
    located = scene.get(
        "located_objects",
        []
    )

    if action == "STOP":
        command = "STOPPING"
        steering = "hold position"
    elif action == "SLOW_DOWN":
        command = "SLOWING DOWN"
        steering = steering_from_human_box(
            located
        )
    elif action == "PROCEED_WITH_CAUTION":
        command = "CAUTIOUS FORWARD"
        steering = steering_from_human_box(
            located
        )
    else:
        command = "MOVING FORWARD"
        steering = "center aisle"

    if human["present"] and action != "STOP":
        steering = steering_from_human_box(
            located
        )

    return (
        "Decision source: rule-based control algorithm\n"
        f"Scene source: {scene.get('scene_source', 'unknown')}\n"
        f"Command: {command}\n"
        f"Steering: {steering}\n"
        f"Target speed: {plan['target_speed']}\n"
        f"Risk: {decision['risk_level']}\n"
        f"Reason: {decision['reasoning']}"
    )


def steering_from_human_box(located_objects):

    for obj in located_objects:

        if obj.get("label") != "human":
            continue

        bbox = obj.get(
            "bbox",
            {}
        )

        if "cx" in bbox:
            cx = float(
                bbox["cx"]
            )
        elif "x1" in bbox and "x2" in bbox:
            cx = (
                float(
                    bbox["x1"]
                )
                +
                float(
                    bbox["x2"]
                )
            ) / 2
            if cx > 1:
                return "turn left around obstacle"
        else:
            continue

        if cx > 0.58:
            return "turn left around human"

        if cx < 0.42:
            return "turn right around human"

        return "slow center approach"

    return "center aisle"


def analyze_frame_id(frame_id):

    return analyze_index(
        frame_index(
            frame_id
        )
    )


def next_frame(index):

    return analyze_index(
        (clamp_index(index) + 1)
        % len(DEMO_FRAMES)
    )


def previous_frame(index):

    return analyze_index(
        (clamp_index(index) - 1)
        % len(DEMO_FRAMES)
    )


def jump_to_index(index):

    return analyze_index(
        index
    )


def play_model_synced(index):

    start = clamp_index(
        index
    )

    for offset in range(
        len(DEMO_FRAMES)
    ):

        current = (
            start
            + offset
        ) % len(DEMO_FRAMES)

        try:
            yield analyze_index(
                current
            )
        except Exception as exc:
            yield error_outputs(
                current,
                f"{type(exc).__name__}: {exc}"
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
        "The top stream plays the dataset smoothly in the browser. Use the controls below to inspect one frame, or use model-synced playback to render each frame only after Qwen/control output is ready."
    )

    frame_index_state = gr.State(
        value=0
    )

    status_box = gr.Textbox(
        label="Demo Status",
        interactive=False
    )

    command_box = gr.Textbox(
        label="Robot Command / Decision Source",
        lines=6,
        interactive=False
    )

    with gr.Row():

        frame_id = gr.Dropdown(
            choices=DEMO_FRAMES,
            value=DEMO_FRAMES[0],
            label="Frame ID",
            scale=2
        )

        frame_slider = gr.Slider(
            minimum=0,
            maximum=len(DEMO_FRAMES) - 1,
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

        model_play_btn = gr.Button(
            "Play Model-Synced"
        )

        model_pause_btn = gr.Button(
            "Stop Model-Synced"
        )

    with gr.Row():

        rgb_image = gr.Image(
            label="RGB With Boxes"
        )

        thermal_image = gr.Image(
            label="Thermal With Boxes"
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
        command_box,
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

    model_play_event = model_play_btn.click(
        play_model_synced,
        inputs=[
            frame_index_state
        ],
        outputs=OUTPUTS
    )

    model_pause_btn.click(
        None,
        None,
        None,
        cancels=[
            model_play_event
        ]
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
    ],
    share=os.getenv(
        "GRADIO_SHARE",
        "0"
    ) == "1"
)
