# Industrial Robot Copilot - Project Specification

## Goal

Build a multimodal industrial robot copilot system that combines:

* RGB camera
* Thermal camera
* Radar point cloud
* Metadata

to create:

* World State
* Scene Understanding
* Risk Assessment
* Robot Planning
* Explainable AI Copilot

The system should support future integration of:

* Qwen2.5-VL
* NVIDIA Locate Anything
* GroundingDINO
* ROS2
* Isaac Sim

---

# Dataset Structure

Dataset root:

data/industrial_subset/

Contains:

00_meta_data/
02_yolo_rgb_distance/
02_yolo_thermal_distance/

05_rgb/
├── calibrated/
└── images/

06_thermal/
├── calibrated/
└── images/

radar/
├── azimuth_abs/
├── azimuth_phase/
├── doppler_abs/
├── doppler_phase/
├── elevation_abs/
├── elevation_phase/
└── pointcloud/
├── csv/
└── pcd/

frames.txt

Each frame id is a six-digit identifier.

Example:

013342

---

# Architecture

Pipeline:

Loader
↓
Fusion
↓
World State
↓
Risk Reasoner
↓
VLM Scene Understanding
↓
Object Memory
↓
Scene Graph
↓
Planner
↓
Copilot
↓
Gradio Dashboard

---

# Module Requirements

## src/loader.py

Responsibilities:

Load a frame by frame_id.

Return:

{
"frame_id": str,

```
"environment": {
    "season": str,
    "weather": str,
    "lighting": str,
    "dataset": str
},

"rgb_path": str,
"thermal_path": str,

"rgb_detections": [],
"thermal_detections": [],

"radar_csv": str
```

}

Functions:

load_frame(frame_id)

nearest_human(frame)

frame_summary(frame)

---

## src/fusion.py

Responsibilities:

Fuse RGB, thermal and radar observations.

Return:

{
"frame_id": str,

```
"human_detected": bool,

"human_distance": float | None,

"thermal_confirmed": bool,

"radar_activity": bool,

"motion_ratio": float,

"radar_summary": {
    "point_count": int,
    "moving_points": int,
    "max_velocity": float,
    "mean_velocity": float
}
```

}

Functions:

summarize_radar()

fuse_frame()

---

## src/world_state.py

Responsibilities:

Convert sensor fusion into semantic world model.

Output:

{
"timestamp": str,

```
"environment": {},

"human": {},

"scene": {},

"confidence": {},

"scene_tags": []
```

}

Scene tags examples:

human_present
multimodal_confirmation
critical_proximity
caution_zone
monitor_distance
safe_separation
dynamic_environment
static_environment
indoor_environment
good_visibility
low_visibility

Functions:

build_world_state()

classify_proximity()

classify_motion()

classify_scene()

llm_context()

world_state_summary()

---

## src/reasoner.py

Responsibilities:

Convert world state into risk assessment.

Output:

{
"risk_level": "LOW|MEDIUM|HIGH|CRITICAL",

```
"action":
    "CONTINUE|
     PROCEED_WITH_CAUTION|
     SLOW_DOWN|
     STOP",

"confidence": str,

"reasoning": str,

"triggered_tags": []
```

}

Rules:

very_near -> CRITICAL

near -> HIGH

medium -> MEDIUM

far -> LOW

Dynamic environment should increase confidence.

Functions:

reason()

---

## src/vlm_backend.py

Abstract backend.

Interface:

class VLMBackend

describe(image_path)

Used for all future VLM models.

---

## src/qwen_vlm.py

Responsibilities:

Connect to local vLLM server.

Model:

Qwen2.5-VL-7B-Instruct

Input:

RGB image

Output JSON:

{
"environment_type": "",

"objects": [],

"hazards": [],

"navigation": {

```
  "aisle_detected": true,

  "walkable_region": "",

  "obstacle_regions": []
```

}
}

Must support image input through OpenAI-compatible API.

---

## src/locate_anything.py

Responsibilities:

Object localization.

Input:

image
object names

Output:

[
{
"label": "",
"bbox": [x1,y1,x2,y2],
"confidence": float
}
]

Initially create a mock implementation.

Later support NVIDIA Locate Anything.

---

## src/vlm_scene.py

Responsibilities:

Combine:

Qwen scene understanding

*

Locate Anything detections

Output:

{
"environment_type": "",

```
"objects": [],

"hazards": [],

"navigation": {},

"located_objects": []
```

}

Tracked objects:

human
machine
workbench
pipe
cabinet
storage box
control panel
forklift
robot

---

## src/object_memory.py

Responsibilities:

Track persistent objects.

Output:

{
"total_objects": int,

```
"objects": [...]
```

}

Methods:

update()

snapshot()

---

## src/object_schema.py

Object categories:

person -> agent

robot -> agent

machine -> obstacle

workbench -> obstacle

cabinet -> obstacle

storage box -> obstacle

pipe -> obstacle

forklift -> dynamic_obstacle

door -> landmark

aisle -> navigable_region

---

## src/scene_graph.py

Responsibilities:

Build graph representation.

Nodes:

environment

robot

human

objects

risk

planner state

Relations:

inside

near

left_of

right_of

occupies

blocks

can_navigate

observes

executes

Output:

{
"frame_id": str,

```
"nodes": [],

"relations": []
```

}

Functions:

build_scene_graph()

scene_graph_summary()

---

## src/planner.py

Responsibilities:

Generate robot behavior.

Input:

world
decision
scene
graph

Output:

{
"mode":
"NORMAL|
MONITOR|
CAUTIOUS|
EMERGENCY_STOP",

```
"action": str,

"target_speed": float,

"goal": str,

"navigation": {},

"reasoning": str
```

}

Functions:

plan()

planner_summary()

---

## src/copilot.py

Responsibilities:

Generate explainable narrative.

Input:

world
scene
plan

Output:

Natural language explanation.

Example:

Human detected at 1.49 meters.

Thermal and RGB sensors agree on the detection.

Radar indicates a dynamic environment.

Robot has entered CAUTIOUS mode and reduced speed.

Functions:

generate_explanation()

---

## src/pipeline.py

Single orchestration layer.

Function:

run_pipeline(frame_id)

Returns:

{
"frame": {},
"world": {},
"decision": {},
"scene": {},
"graph": {},
"plan": {},
"explanation": str
}

---

# Dashboard

File:

app.py

Framework:

Gradio

Features:

Frame selector

RGB image

Thermal image

World state

Risk decision

Scene understanding

Planner output

Copilot explanation

Support:

Single frame mode

Future timeline playback mode

---

# Future Roadmap

Phase 1

✓ Loader
✓ Fusion
✓ World State
✓ Reasoner
✓ Planner

Phase 2

Qwen2.5-VL integration

Locate Anything integration

Scene Graph improvements

Phase 3

Isaac Sim

ROS2

Robot control

Path planning

Phase 4

Industrial AI Copilot

Natural language queries

Mission planning

Autonomous navigation recommendations

---

# Deployment

Target Hardware:

AMD MI300

Inference:

vLLM

Model:

Qwen2.5-VL-7B-Instruct

API:

OpenAI-compatible endpoint

Default:

http://localhost:8000/v1

---

# Success Criteria

System demonstrates:

1. Multimodal perception

2. Sensor fusion

3. World modeling

4. Risk assessment

5. Explainable robot planning

6. Scene understanding using VLMs

7. Interactive Gradio dashboard

8. Future compatibility with simulation and ROS2

