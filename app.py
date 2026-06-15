import json
import os
from pathlib import Path

import gradio as gr
from PIL import Image

from src.pipeline import run_pipeline


DATA_ROOT = Path("data/industrial_subset")
FRAMES = [
    line.strip()
    for line in (DATA_ROOT / "frames.txt").read_text().splitlines()
    if line.strip()
]


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
    )

    thermal = Image.open(
        frame["thermal_image"]
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

    frame_index_state = gr.State(
        value=0
    )

    playing_state = gr.State(
        value=False
    )

    playback_timer = gr.Timer(
        value=0.75,
        active=True
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

        speed = gr.Slider(
            minimum=0.25,
            maximum=3.0,
            value=0.75,
            step=0.25,
            label="Seconds / Frame",
            scale=2
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

        play_btn = gr.Button(
            "Play"
        )

        pause_btn = gr.Button(
            "Pause"
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

    play_btn.click(
        set_playing,
        inputs=[
            gr.State(
                value=True
            )
        ],
        outputs=[
            playing_state
        ]
    )

    pause_btn.click(
        set_playing,
        inputs=[
            gr.State(
                value=False
            )
        ],
        outputs=[
            playing_state
        ]
    )

    speed.change(
        timer_interval,
        inputs=[
            speed
        ],
        outputs=[
            playback_timer
        ]
    )

    playback_timer.tick(
        timer_tick,
        inputs=[
            frame_index_state,
            playing_state
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
    )
)
