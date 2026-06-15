from collections import defaultdict


class ObjectMemory:

    def __init__(self):

        self.objects = {}

        self.type_counter = defaultdict(int)

    def update(self, detections, frame_id):

        """
        detections:

        [
            {
                "label": "machine",
                "bbox": [x1,y1,x2,y2]
            }
        ]
        """

        for det in detections:

            label = det["label"]

            obj_id = self._create_id(label)

            self.objects[obj_id] = {

                "id": obj_id,

                "type": label,

                "bbox": det.get("bbox"),

                "last_seen": frame_id,

                "status": "visible"
            }

    def _create_id(self, label):

        self.type_counter[label] += 1

        return f"{label}_{self.type_counter[label]}"

    def get_objects(self):

        return self.objects

    def summary(self):

        return {

            "total_objects":
                len(self.objects),

            "objects":
                list(self.objects.values())
        }

    def snapshot(self):

        return self.summary()
