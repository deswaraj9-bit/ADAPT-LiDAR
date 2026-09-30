import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from lidar_simulator import LidarSimulator
from ml.pointnet_model import PointNetSegmentation


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLASSES = 4
NUM_POINTS = 2048

TRAIN_SAMPLES = 120
VAL_SAMPLES = 30

BATCH_SIZE = 8
EPOCHS = 10

LEARNING_RATE = 0.001

MODEL_DIR = "models"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "pointnet_lidar.pth"
)

CLASS_NAMES = [
    "ground",
    "vehicle",
    "pedestrian",
    "static_obstacle"
]


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("=" * 60)
print("ADAPT-LiDAR PointNet Training")
print("=" * 60)
print("Device:", DEVICE)
print("Training samples:", TRAIN_SAMPLES)
print("Validation samples:", VAL_SAMPLES)
print("Points/sample:", NUM_POINTS)
print("Epochs:", EPOCHS)
print("=" * 60)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# ============================================================
# POINT NORMALIZATION
# ============================================================

def normalize_points(points):
    """
    Keep the same coordinate system during training and inference.

    X = left/right
    Y = forward distance
    Z = height
    """

    points = points.astype(np.float32).copy()

    # Fixed physical scaling.
    # This preserves the scene geometry while keeping
    # neural-network input values numerically stable.

    points[:, 0] = points[:, 0] / 100.0
    points[:, 1] = points[:, 1] / 100.0
    points[:, 2] = points[:, 2] / 5.0

    return points


# ============================================================
# DATASET
# ============================================================

class SyntheticLidarDataset(Dataset):

    def __init__(
        self,
        num_samples,
        num_points=2048,
        seed=42
    ):

        self.num_samples = num_samples
        self.num_points = num_points

        self.simulator = LidarSimulator()

        self.rng = np.random.default_rng(seed)

        self.samples = []

        print(
            f"\nGenerating {num_samples} synthetic LiDAR scenes..."
        )

        for i in range(num_samples):

            points, labels = self.simulator.generate_frame()

            points = np.asarray(
                points,
                dtype=np.float32
            )

            labels = np.asarray(
                labels,
                dtype=np.int64
            )

            # ------------------------------------------------
            # Balanced-ish sampling
            # ------------------------------------------------
            #
            # Ground has many more points than other classes.
            # We increase the probability of selecting
            # minority classes.
            #

            class_counts = np.bincount(
                labels,
                minlength=NUM_CLASSES
            ).astype(np.float64)

            class_weights = np.zeros(NUM_CLASSES)

            for c in range(NUM_CLASSES):

                if class_counts[c] > 0:
                    class_weights[c] = 1.0 / class_counts[c]

            point_probability = class_weights[labels]

            point_probability /= point_probability.sum()

            # Sample without replacement.
            selected = self.rng.choice(
                len(points),
                size=min(
                    num_points,
                    len(points)
                ),
                replace=False,
                p=point_probability
            )

            sampled_points = points[selected]
            sampled_labels = labels[selected]

            sampled_points = normalize_points(
                sampled_points
            )

            self.samples.append(
                (
                    sampled_points,
                    sampled_labels
                )
            )

            if (i + 1) % 10 == 0:
                print(
                    f"Generated {i + 1}/{num_samples}"
                )

        print("Dataset generation complete.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        points, labels = self.samples[index]

        # Model expects:
        # [3, N]

        points = torch.tensor(
            points,
            dtype=torch.float32
        ).transpose(0, 1)

        labels = torch.tensor(
            labels,
            dtype=torch.long
        )

        return points, labels


# ============================================================
# DATASETS
# ============================================================

train_dataset = SyntheticLidarDataset(
    num_samples=TRAIN_SAMPLES,
    num_points=NUM_POINTS,
    seed=42
)

val_dataset = SyntheticLidarDataset(
    num_samples=VAL_SAMPLES,
    num_points=NUM_POINTS,
    seed=1000
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# MODEL
# ============================================================

model = PointNetSegmentation(
    num_classes=NUM_CLASSES
).to(DEVICE)


# ============================================================
# CLASS WEIGHTS
# ============================================================

class_counts = np.zeros(
    NUM_CLASSES,
    dtype=np.float64
)

for _, labels in train_dataset.samples:

    for c in range(NUM_CLASSES):

        class_counts[c] += np.sum(
            labels == c
        )


print("\nTraining class distribution:")

for i, name in enumerate(CLASS_NAMES):

    print(
        f"{name:20s}: "
        f"{int(class_counts[i])}"
    )


# Inverse-frequency weighting
class_weights = 1.0 / (
    class_counts + 1e-6
)

class_weights = (
    class_weights /
    class_weights.mean()
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32,
    device=DEVICE
)


print("\nClass weights:")

for i, name in enumerate(CLASS_NAMES):

    print(
        f"{name:20s}: "
        f"{class_weights[i].item():.4f}"
    )


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=5,
    gamma=0.5
)


# ============================================================
# VALIDATION
# ============================================================

def validate():

    model.eval()

    total_correct = 0
    total_points = 0

    class_correct = np.zeros(
        NUM_CLASSES,
        dtype=np.int64
    )

    class_total = np.zeros(
        NUM_CLASSES,
        dtype=np.int64
    )

    total_loss = 0.0

    with torch.no_grad():

        for points, labels in val_loader:

            points = points.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(points)

            loss = criterion(
                logits,
                labels
            )

            total_loss += loss.item()

            predictions = torch.argmax(
                logits,
                dim=1
            )

            total_correct += (
                predictions == labels
            ).sum().item()

            total_points += labels.numel()

            for c in range(NUM_CLASSES):

                mask = labels == c

                class_total[c] += mask.sum().item()

                class_correct[c] += (
                    (predictions == labels) &
                    mask
                ).sum().item()

    accuracy = (
        total_correct /
        max(total_points, 1)
    )

    class_accuracy = []

    for c in range(NUM_CLASSES):

        if class_total[c] > 0:

            acc = (
                class_correct[c] /
                class_total[c]
            )

        else:

            acc = 0.0

        class_accuracy.append(acc)

    return (
        total_loss / len(val_loader),
        accuracy,
        class_accuracy
    )


# ============================================================
# TRAINING
# ============================================================

best_accuracy = 0.0

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

print("\nStarting training...\n")


for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    total_correct = 0
    total_points = 0

    for batch_index, (points, labels) in enumerate(
        train_loader
    ):

        points = points.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        logits = model(points)

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        predictions = torch.argmax(
            logits,
            dim=1
        )

        total_correct += (
            predictions == labels
        ).sum().item()

        total_points += labels.numel()

    scheduler.step()

    train_loss = (
        running_loss /
        len(train_loader)
    )

    train_accuracy = (
        total_correct /
        max(total_points, 1)
    )

    val_loss, val_accuracy, class_accuracy = validate()

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy * 100:.2f}% | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy * 100:.2f}%"
    )

    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        checkpoint = {
            "model_state": model.state_dict(),
            "num_classes": NUM_CLASSES,
            "class_names": CLASS_NAMES,
            "input_features": 3,
            "num_points": NUM_POINTS,
            "normalization": {
                "x_scale": 100.0,
                "y_scale": 100.0,
                "z_scale": 5.0
            },
            "validation_accuracy": best_accuracy
        }

        torch.save(
            checkpoint,
            MODEL_PATH
        )

        print(
            f"  ✓ Best model saved "
            f"({best_accuracy * 100:.2f}%)"
        )


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(
    f"Best validation accuracy: "
    f"{best_accuracy * 100:.2f}%"
)

print(
    f"Model saved to: {MODEL_PATH}"
)

print("\nPer-class validation accuracy:")

_, _, final_class_accuracy = validate()

for i, name in enumerate(CLASS_NAMES):

    print(
        f"{name:20s}: "
        f"{final_class_accuracy[i] * 100:.2f}%"
    )

print("=" * 60)