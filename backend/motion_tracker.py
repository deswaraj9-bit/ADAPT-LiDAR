import math
import time


class MotionTracker:
    def __init__(self, max_match_distance=3.0):
        self.max_match_distance = max_match_distance
        self.previous_objects = {}
        self.next_id = 1
        self.last_timestamp = None

    @staticmethod
    def distance(a, b):
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        dz = a[2] - b[2]

        return math.sqrt(
            dx * dx +
            dy * dy +
            dz * dz
        )

    def update(self, objects, timestamp=None):

        if timestamp is None:
            timestamp = time.time()

        # Calculate time difference
        if self.last_timestamp is None:
            dt = 0.1
        else:
            dt = timestamp - self.last_timestamp
            dt = max(0.01, min(dt, 1.0))

        self.last_timestamp = timestamp

        current_objects = {}
        results = []

        for obj in objects:

            class_id = int(obj["class_id"])
            centroid = tuple(obj["centroid"])

            best_id = None
            best_distance = float("inf")

            # Find closest previous object of the same class
            for object_id, previous in self.previous_objects.items():

                if previous["class_id"] != class_id:
                    continue

                distance = self.distance(
                    centroid,
                    previous["centroid"]
                )

                if (
                    distance < best_distance
                    and distance <= self.max_match_distance
                ):
                    best_distance = distance
                    best_id = object_id

            # New object
            if best_id is None:

                object_id = self.next_id
                self.next_id += 1

                speed = 0.0

            # Existing object
            else:

                object_id = best_id

                previous = self.previous_objects[object_id]

                displacement = self.distance(
                    centroid,
                    previous["centroid"]
                )

                speed = displacement / dt

            tracked_object = {
                "object_id": object_id,
                "class_id": class_id,
                "centroid": centroid,
                "speed_mps": speed
            }

            current_objects[object_id] = tracked_object
            results.append(tracked_object)

        self.previous_objects = current_objects

        return results