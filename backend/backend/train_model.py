import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Allow importing files from backend/
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from lidar_simulator import LidarSimulator
from pointnet_model import PointNetSegmentation


NUM_CLASSES = 4
NUM_POINTS = 2048
EPOCHS = 15
STEPS_PER_EPOCH = 20
LEARNING_RATE = 0.001


def sample_points(points, labels, num_points=2048):

    if len(points) >= num_points:

        indices = np.random.choice(
            len(points),
            num_points,
            replace=False
        )

    else:

        indices = np.random.choice(
            len(points),
            num_points,
            replace=True
        )

    return (
        points[indices],
        labels[indices]
    )


def normalize_points(points):

    points = points.copy().astype(
        np.float32
    )

    points[:, 0] /= 50.0
    points[:, 1] /= 100.0
    points[:, 2] /= 5.0

    return points


def get_device():

    if torch.backends.mps.is_available():

        return torch.device("mps")

    return torch.device("cpu")


def main():

    device = get_device()

    print("=" * 60)
    print("ADAPT-LiDAR PointNet Training")
    print("=" * 60)

    print(
        f"Device: {device}"
    )

    print(
        f"Classes: {NUM_CLASSES}"
    )

    print(
        f"Points per sample: {NUM_POINTS}"
    )

    print(
        f"Epochs: {EPOCHS}"
    )

    print("=" * 60)

    simulator = LidarSimulator()

    model = PointNetSegmentation(
        num_classes=NUM_CLASSES
    )

    model = model.to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    criterion = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(EPOCHS):

        total_loss = 0.0
        total_correct = 0
        total_points = 0

        for step in range(STEPS_PER_EPOCH):

            # Generate synthetic LiDAR frame
            points, labels = (
                simulator.generate_frame()
            )

            # Sample manageable number of points
            points, labels = sample_points(
                points,
                labels,
                NUM_POINTS
            )

            # Normalize
            points = normalize_points(
                points
            )

            # Convert to tensors
            x = torch.tensor(
                points,
                dtype=torch.float32
            )

            y = torch.tensor(
                labels,
                dtype=torch.long
            )

            # [N,3] -> [1,3,N]
            x = x.transpose(
                0,
                1
            ).unsqueeze(0)

            # [N] -> [1,N]
            y = y.unsqueeze(0)

            x = x.to(device)
            y = y.to(device)

            # Clear gradients
            optimizer.zero_grad()

            # Forward pass
            logits = model(x)

            # Calculate loss
            loss = criterion(
                logits,
                y
            )

            # Backpropagation
            loss.backward()

            # Update weights
            optimizer.step()

            total_loss += loss.item()

            predictions = (
                logits.argmax(
                    dim=1
                )
            )

            total_correct += (
                predictions == y
            ).sum().item()

            total_points += y.numel()

        average_loss = (
            total_loss /
            STEPS_PER_EPOCH
        )

        accuracy = (
            total_correct /
            total_points
        )

        print(
            f"Epoch "
            f"{epoch + 1:02d}/{EPOCHS} | "
            f"Loss: {average_loss:.4f} | "
            f"Accuracy: {accuracy * 100:.2f}%"
        )

    # Create model directory
    os.makedirs(
        "models",
        exist_ok=True
    )

    model_path = (
        "models/pointnet_lidar.pth"
    )

    # Save trained model
    torch.save(
        model.state_dict(),
        model_path
    )

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(
        f"Model saved to: {model_path}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()