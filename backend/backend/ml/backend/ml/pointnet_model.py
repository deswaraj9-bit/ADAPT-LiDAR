import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetSegmentation(nn.Module):
    """
    PointNet-based semantic segmentation model.

    Classes:
        0 = Ground
        1 = Vehicle
        2 = Pedestrian
        3 = Static Obstacle
    """

    def __init__(self, num_classes=4):
        super().__init__()

        # Point-wise feature extraction
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 256, 1)
        self.bn3 = nn.BatchNorm1d(256)

        # Global feature
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn4 = nn.BatchNorm1d(512)

        # Segmentation head
        self.seg1 = nn.Conv1d(256 + 512, 256, 1)
        self.seg_bn1 = nn.BatchNorm1d(256)

        self.seg2 = nn.Conv1d(256, 128, 1)
        self.seg_bn2 = nn.BatchNorm1d(128)

        self.seg3 = nn.Conv1d(128, num_classes, 1)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        """
        Input:
            x: [B, 3, N]

        Output:
            logits: [B, num_classes, N]
        """

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        local_features = F.relu(
            self.bn3(self.conv3(x))
        )

        global_features = F.relu(
            self.bn4(self.conv4(local_features))
        )

        # Maximum pooling across all points
        global_feature = torch.max(
            global_features,
            dim=2,
            keepdim=True
        )[0]

        # Repeat global feature for every point
        global_feature = global_feature.repeat(
            1,
            1,
            local_features.shape[2]
        )

        # Combine local + global information
        combined = torch.cat(
            [local_features, global_feature],
            dim=1
        )

        x = F.relu(
            self.seg_bn1(
                self.seg1(combined)
            )
        )

        x = self.dropout(x)

        x = F.relu(
            self.seg_bn2(
                self.seg2(x)
            )
        )

        logits = self.seg3(x)

        return logits


def predict(model, points, device="cpu"):
    """
    Perform semantic segmentation.

    points:
        numpy array [N, 3]

    Returns:
        labels
        confidence
    """

    import numpy as np

    model.eval()

    # Normalize coordinates
    normalized = points.copy().astype(np.float32)

    normalized[:, 0] /= 50.0
    normalized[:, 1] /= 100.0
    normalized[:, 2] /= 5.0

    tensor = torch.from_numpy(
        normalized
    ).float()

    tensor = tensor.transpose(0, 1)

    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(device)

    with torch.no_grad():

        logits = model(tensor)

        probabilities = F.softmax(
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

    return labels, confidence