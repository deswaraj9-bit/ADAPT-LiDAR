import torch
import torch.nn as nn


class PointNetSegmentation(nn.Module):
    """
    Lightweight PointNet-style semantic segmentation model.

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

        # Segmentation head
        # 64 + 128 + 256 local features + 256 global feature
        self.seg1 = nn.Conv1d(704, 256, 1)
        self.seg_bn1 = nn.BatchNorm1d(256)

        self.seg2 = nn.Conv1d(256, 128, 1)
        self.seg_bn2 = nn.BatchNorm1d(128)

        self.dropout = nn.Dropout(0.2)

        self.classifier = nn.Conv1d(
            128,
            num_classes,
            1
        )

    def forward(self, x):
        """
        Input:
            x: [B, 3, N]

        Output:
            logits: [B, num_classes, N]
        """

        # Local features
        f1 = torch.relu(self.bn1(self.conv1(x)))
        f2 = torch.relu(self.bn2(self.conv2(f1)))
        f3 = torch.relu(self.bn3(self.conv3(f2)))

        # Global feature
        global_feature = torch.max(
            f3,
            dim=2,
            keepdim=True
        )[0]

        global_feature = global_feature.expand(
            -1,
            -1,
            f3.shape[2]
        )

        # Combine local + global information
        features = torch.cat(
            [f1, f2, f3, global_feature],
            dim=1
        )

        # Segmentation head
        x = torch.relu(
            self.seg_bn1(
                self.seg1(features)
            )
        )

        x = torch.relu(
            self.seg_bn2(
                self.seg2(x)
            )
        )

        x = self.dropout(x)

        logits = self.classifier(x)

        return logits


if __name__ == "__main__":
    model = PointNetSegmentation(num_classes=4)

    dummy = torch.randn(2, 3, 2048)

    output = model(dummy)

    print("Input shape :", dummy.shape)
    print("Output shape:", output.shape)
    print("Model parameters:", sum(
        p.numel() for p in model.parameters()
    ))