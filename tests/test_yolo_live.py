from src.loader import load_frame
import src.locate_anything as locator


class FakeScalar:

    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeTensor:

    def __init__(self, values):
        self.values = values

    def __getitem__(self, index):
        return self

    def item(self):
        return self.values[0]

    def tolist(self):
        return self.values


class FakeBox:

    cls = [
        FakeScalar(0)
    ]

    conf = [
        FakeScalar(0.91)
    ]

    xyxy = [
        FakeTensor(
            [
                10,
                20,
                110,
                220
            ]
        )
    ]


class FakeResult:

    names = {
        0: "person"
    }

    boxes = [
        FakeBox()
    ]


class FakeYolo:

    def __call__(self, image, **kwargs):
        return [
            FakeResult()
        ]


def fake_yolo_model():
    return FakeYolo()


locator.get_yolo_model = fake_yolo_model

frame = load_frame("013342")
detections = locator.locate_with_yolo_live(
    frame,
    [
        "human"
    ]
)

assert detections == [
    {
        "label": "human",
        "bbox": {
            "x1": 10.0,
            "y1": 20.0,
            "x2": 110.0,
            "y2": 220.0
        },
        "distance": None,
        "confidence": 0.91,
        "source": "yolo_live"
    }
]

print(detections)
