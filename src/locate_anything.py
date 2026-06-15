import base64
import os
import re
from functools import lru_cache
from pathlib import Path

from PIL import Image


DEFAULT_MODEL = "nvidia/LocateAnything-3B"

DEFAULT_OBJECTS = [
    "human",
    "industrial machine",
    "workbench",
    "pipe",
    "cabinet",
    "storage box",
    "control panel",
    "forklift",
    "robot",
]


def locate_objects(frame, object_names=None):
    """
    Locate tracked objects.

    Backends:
    - labels: use dataset RGB/thermal labels.
    - grounding_dino: use Hugging Face zero-shot object detection.
    - nvidia_vllm: call nvidia/LocateAnything-3B through vLLM/OpenAI API.
    - nvidia_transformers: load nvidia/LocateAnything-3B locally from HF.
    - nvidia: use vLLM if configured, otherwise local Transformers.
    - auto: try NVIDIA LocateAnything, then fall back to labels.
    """

    backend = os.getenv(
        "LOCATOR_BACKEND",
        "labels"
    ).lower()

    names = object_names or DEFAULT_OBJECTS

    if backend == "grounding_dino":
        try:
            return locate_with_grounding_dino(
                frame,
                names
            )
        except Exception as exc:
            return mark_fallback(
                locate_from_labels(
                    frame,
                    names
                ),
                "grounding_dino",
                exc
            )

    if backend == "nvidia_vllm":
        return locate_with_nvidia_vllm(
            frame,
            names
        )

    if backend == "nvidia_transformers":
        return locate_with_nvidia_transformers(
            frame,
            names
        )

    if backend == "nvidia":
        return locate_with_nvidia(
            frame,
            names
        )

    if backend == "auto":
        try:
            return locate_with_grounding_dino(
                frame,
                names
            )
        except Exception:
            pass

        try:
            return locate_with_nvidia(
                frame,
                names
            )
        except Exception:
            return locate_from_labels(
                frame,
                names
            )

    return locate_from_labels(
        frame,
        names
    )


def locate_with_nvidia(frame, object_names=None):

    if os.getenv("NVIDIA_LOCATE_ANYTHING_BASE_URL"):
        return locate_with_nvidia_vllm(
            frame,
            object_names
        )

    return locate_with_nvidia_transformers(
        frame,
        object_names
    )


def locate_with_nvidia_vllm(frame, object_names=None):

    from openai import OpenAI

    image_path = require_rgb_image(
        frame
    )

    image = Image.open(
        image_path
    ).convert("RGB")

    client = OpenAI(
        base_url=os.getenv(
            "NVIDIA_LOCATE_ANYTHING_BASE_URL",
            "http://localhost:8000/v1"
        ),
        api_key=os.getenv(
            "NVIDIA_LOCATE_ANYTHING_API_KEY",
            "EMPTY"
        )
    )

    model = os.getenv(
        "NVIDIA_LOCATE_ANYTHING_MODEL",
        DEFAULT_MODEL
    )

    located = []

    for name in object_names or DEFAULT_OBJECTS:

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": multi_grounding_prompt(
                                name
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_to_data_url(
                                    image_path
                                )
                            }
                        }
                    ]
                }
            ],
            temperature=0,
            max_tokens=int(
                os.getenv(
                    "NVIDIA_LOCATE_ANYTHING_MAX_TOKENS",
                    "2048"
                )
            )
        )

        answer = response.choices[0].message.content

        located.extend(
            boxes_to_detections(
                parse_boxes(
                    answer,
                    image.width,
                    image.height
                ),
                name,
                "nvidia_locateanything_vllm"
            )
        )

    return located


def locate_with_nvidia_transformers(frame, object_names=None):

    image_path = require_rgb_image(
        frame
    )

    image = Image.open(
        image_path
    ).convert("RGB")

    worker = get_transformers_worker()

    located = []

    for name in object_names or DEFAULT_OBJECTS:

        result = worker.ground_multi(
            image,
            name,
            generation_mode=os.getenv(
                "NVIDIA_LOCATE_ANYTHING_GENERATION_MODE",
                "hybrid"
            ),
            max_new_tokens=int(
                os.getenv(
                    "NVIDIA_LOCATE_ANYTHING_MAX_TOKENS",
                    "2048"
                )
            ),
            temperature=float(
                os.getenv(
                    "NVIDIA_LOCATE_ANYTHING_TEMPERATURE",
                    "0.7"
                )
            ),
            verbose=False
        )

        located.extend(
            boxes_to_detections(
                parse_boxes(
                    result["answer"],
                    image.width,
                    image.height
                ),
                name,
                "nvidia_locateanything_transformers"
            )
        )

    return located


def locate_with_grounding_dino(frame, object_names=None):

    image_path = require_rgb_image(
        frame
    )

    image = Image.open(
        image_path
    ).convert("RGB")

    detector = get_grounding_dino_detector()

    threshold = float(
        os.getenv(
            "GROUNDING_DINO_THRESHOLD",
            "0.30"
        )
    )

    results = detector(
        image,
        candidate_labels=list(
            object_names or DEFAULT_OBJECTS
        ),
        threshold=threshold
    )

    located = []

    for result in results:

        box = result.get(
            "box",
            {}
        )

        located.append({
            "label":
                normalize_label(
                    result.get(
                        "label",
                        "unknown"
                    )
                ),

            "bbox": {
                "x1":
                    float(
                        box.get(
                            "xmin",
                            0
                        )
                    ),
                "y1":
                    float(
                        box.get(
                            "ymin",
                            0
                        )
                    ),
                "x2":
                    float(
                        box.get(
                            "xmax",
                            0
                        )
                    ),
                "y2":
                    float(
                        box.get(
                            "ymax",
                            0
                        )
                    )
            },

            "distance":
                None,

            "confidence":
                float(
                    result.get(
                        "score",
                        1.0
                    )
                ),

            "source":
                "grounding_dino"
        })

    return located


def mark_fallback(detections, requested_backend, exc):

    reason = (
        f"{type(exc).__name__}: {exc}"
    )

    for det in detections:
        det["requested_backend"] = requested_backend
        det["fallback_reason"] = reason
        det["source"] = (
            det.get(
                "source",
                "labels"
            )
            + "_fallback"
        )

    return detections


@lru_cache(maxsize=1)
def get_grounding_dino_detector():

    try:
        from transformers import pipeline
    except Exception as exc:
        raise RuntimeError(
            "GroundingDINO backend requires a Transformers install with "
            "`pipeline` support. Use LOCATOR_BACKEND=labels for the stable "
            "demo, or fix the Transformers environment."
        ) from exc

    kwargs = {
        "task":
            "zero-shot-object-detection",
        "model":
            os.getenv(
                "GROUNDING_DINO_MODEL",
                "IDEA-Research/grounding-dino-tiny"
            )
    }

    device = os.getenv(
        "GROUNDING_DINO_DEVICE"
    )

    if device:
        kwargs["device"] = device

    return pipeline(
        **kwargs
    )


@lru_cache(maxsize=1)
def get_transformers_worker():

    import torch
    from transformers import AutoModel, AutoProcessor, AutoTokenizer

    model_path = os.getenv(
        "NVIDIA_LOCATE_ANYTHING_MODEL",
        DEFAULT_MODEL
    )

    device = os.getenv(
        "NVIDIA_LOCATE_ANYTHING_DEVICE",
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dtype_name = os.getenv(
        "NVIDIA_LOCATE_ANYTHING_DTYPE",
        "bfloat16"
    )

    dtype = getattr(
        torch,
        dtype_name
    )

    attn_implementation = os.getenv(
        "NVIDIA_LOCATE_ANYTHING_ATTN_IMPLEMENTATION",
        "sdpa"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation=attn_implementation
    )

    force_attention_implementation(
        model,
        attn_implementation
    )

    model = model.to(device).eval()

    return LocateAnythingWorker(
        model,
        processor,
        tokenizer,
        device,
        dtype
    )


def force_attention_implementation(model, attn_implementation):

    for obj in attention_targets(model):

        if hasattr(obj, "_attn_implementation"):
            setattr(
                obj,
                "_attn_implementation",
                attn_implementation
            )

        if hasattr(obj, "attn_implementation"):
            setattr(
                obj,
                "attn_implementation",
                attn_implementation
            )


def attention_targets(model):

    targets = [
        model
    ]

    for attr in [
        "config",
        "language_model",
        "vision_model",
        "model"
    ]:

        if hasattr(model, attr):
            targets.append(
                getattr(
                    model,
                    attr
                )
            )

    if hasattr(model, "modules"):
        targets.extend(
            list(
                model.modules()
            )
        )

    for target in list(targets):
        if hasattr(target, "config"):
            targets.append(
                target.config
            )

    return targets


class LocateAnythingWorker:

    def __init__(
        self,
        model,
        processor,
        tokenizer,
        device,
        dtype
    ):

        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype

    def predict(
        self,
        image,
        question,
        generation_mode="hybrid",
        max_new_tokens=2048,
        temperature=0.7,
        verbose=False
    ):

        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image
                    },
                    {
                        "type": "text",
                        "text": question
                    }
                ]
            }
        ]

        text = self.processor.py_apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        images, videos = self.processor.process_vision_info(
            messages
        )

        inputs = self.processor(
            text=[text],
            images=images,
            videos=videos,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            response = self.model.generate(
                pixel_values=inputs["pixel_values"].to(
                    self.dtype
                ),
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_grid_hws=inputs.get(
                    "image_grid_hws",
                    None
                ),
                tokenizer=self.tokenizer,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                generation_mode=generation_mode,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                verbose=verbose
            )

        return {
            "answer":
                response[0]
                if isinstance(response, tuple)
                else response
        }

    def ground_multi(self, image, phrase, **kwargs):

        return self.predict(
            image,
            multi_grounding_prompt(
                phrase
            ),
            **kwargs
        )


def multi_grounding_prompt(name):

    return (
        "Locate all the instances that match the following description: "
        f"{name}."
    )


def parse_boxes(answer, image_width, image_height):

    boxes = []

    for match in re.finditer(
        r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>",
        answer
    ):

        x1, y1, x2, y2 = [
            int(group)
            for group in match.groups()
        ]

        boxes.append({
            "x1":
                x1 / 1000 * image_width,
            "y1":
                y1 / 1000 * image_height,
            "x2":
                x2 / 1000 * image_width,
            "y2":
                y2 / 1000 * image_height
        })

    return boxes


def boxes_to_detections(boxes, label, source):

    return [
        {
            "label":
                normalize_label(
                    label
                ),

            "bbox":
                box,

            "distance":
                None,

            "confidence":
                1.0,

            "source":
                source
        }
        for box in boxes
    ]


def image_to_data_url(image_path):

    path = Path(image_path)
    suffix = path.suffix.lower()

    if suffix in {".jpg", ".jpeg"}:
        mime_type = "image/jpeg"
    elif suffix == ".png":
        mime_type = "image/png"
    else:
        mime_type = "application/octet-stream"

    encoded = base64.b64encode(
        path.read_bytes()
    ).decode("ascii")

    return f"data:{mime_type};base64,{encoded}"


def require_rgb_image(frame):

    image_path = frame.get(
        "rgb_image"
    )

    if not image_path:
        raise ValueError(
            "RGB image path is required for NVIDIA LocateAnything."
        )

    return image_path


def locate_from_labels(frame, object_names=None):

    tracked = normalize_tracked_names(
        object_names
    )

    located = []

    sources = [
        (
            "rgb",
            frame.get(
                "rgb_detections",
                []
            )
        ),
        (
            "thermal",
            frame.get(
                "thermal_detections",
                []
            )
        )
    ]

    for source, detections in sources:

        for det in detections:

            label = normalize_label(
                det["class_name"]
            )

            if tracked is not None and label not in tracked:
                continue

            located.append({
                "label":
                    label,

                "bbox":
                    det["bbox"],

                "distance":
                    det["distance"],

                "confidence":
                    1.0,

                "source":
                    source
            })

    return located


def normalize_label(label):

    return label.replace(
        " ",
        "_"
    )


def normalize_tracked_names(object_names):

    if object_names is None:
        return None

    return {
        normalize_label(name)
        for name in object_names
    }
