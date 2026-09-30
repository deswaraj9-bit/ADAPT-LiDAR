import os
import sys

import numpy as np
import torch

# Make backend imports available
BACKEND_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

if BACKEND_DIR not in sys.path:
    sys.path.append(BACKEND_DIR)

from ml.pointnet_model import PointNetSegmentation


class LidarSemanticModel:

    CLASS_NAMES = {
        0: "ground",
        1: "vehicle",
        2: "pedestrian",
        3: "static_obstacle"
    }

    def __init__(
        self,
        model_path="models/pointnet_lidar.pth"
    ):

        self.model_path = model_path

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )

        # CPU is used for reliable FastAPI inference.
        self.device = torch.device("cpu")

        checkpoint = torch.load(
            model_path,
            map_location=self.device
        )

        self.model = PointNetSegmentation(
            num_classes=checkpoint.get(
                "num_classes",
                4
            )
        )

        self.model.load_state_dict(
            checkpoint["model_state"]
        )

        self.model.to(self.device)

        self.model.eval()

        self.x_scale = checkpoint.get(
            "normalization",
            {}
        ).get("x_scale", 100.0)

        self.y_scale = checkpoint.get(
            "normalization",
            {}
        ).get("y_scale", 100.0)

        self.z_scale = checkpoint.get(
            "normalization",
            {}
        ).get("z_scale", 5.0)

        self.validation_accuracy = checkpoint.get(
            "validation_accuracy",
            None
        )

        print(
            "✓ PointNet semantic model loaded"
        )

        print(
            f"✓ Model validation accuracy: "
            f"{self.validation_accuracy * 100:.2f}%"
            if self.validation_accuracy is not None
            else "✓ Validation accuracy unavailable"
        )

    # --------------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------------

    def normalize_points(self, points):

        points = np.asarray(
            points,
            dtype=np.float32
        ).copy()

        points[:, 0] /= self.x_scale
        points[:, 1] /= self.y_scale
        points[:, 2] /= self.z_scale

        return points

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    def predict(
        self,
        points,
        chunk_size=4096
    ):

        points = np.asarray(
            points,
            dtype=np.float32
        )

        total_points = len(points)

        all_labels = []
        all_confidences = []

        with torch.no_grad():

            for start in range(
                0,
                total_points,
                chunk_size
            ):

                end = min(
                    start + chunk_size,
                    total_points
                )

                chunk = points[start:end]

                normalized = self.normalize_points(
                    chunk
                )

                # [N, 3] → [1, 3, N]
                tensor = torch.from_numpy(
                    normalized
                ).transpose(
                    0,
                    1
                ).unsqueeze(0)

                tensor = tensor.to(
                    self.device
                )

                logits = self.model(
                    tensor
                )

                probabilities = torch.softmax(
                    logits,
                    dim=1
                )

                confidence, labels = torch.max(
                    probabilities,
                    dim=1
                )

                labels = labels.squeeze(0).cpu().numpy()

                confidence = (
                    confidence
                    .squeeze(0)
                    .cpu()
                    .numpy()
                )

                all_labels.append(labels)
                all_confidences.append(confidence)

        labels = np.concatenate(
            all_labels
        )

        confidences = np.concatenate(
            all_confidences
        )

        return labels, confidences

    # --------------------------------------------------------
    # CLASS COUNTS
    # --------------------------------------------------------

    def get_class_counts(
        self,
        labels
    ):

        counts = {}

        for class_id, name in self.CLASS_NAMES.items():

            counts[name] = int(
                np.sum(labels == class_id)
            )

        return counts