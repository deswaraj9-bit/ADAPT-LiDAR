import numpy as np


class LidarSimulator:
    """
    Simulates a LiDAR sensor for ADAPT-LiDAR.

    Coordinate system:
        X = left/right
        Y = forward distance
        Z = height

    Semantic classes:
        0 = Ground
        1 = Vehicle
        2 = Pedestrian
        3 = Static Obstacle
    """

    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)

        self.max_range = 100.0
        self.min_range = 0.5

        self.classes = {
            0: "ground",
            1: "vehicle",
            2: "pedestrian",
            3: "static_obstacle",
        }

    def generate_ground(self, num_points=70000):
        """Generate road/ground LiDAR points."""

        x = self.rng.uniform(-50, 50, num_points)
        y = self.rng.uniform(1, 100, num_points)

        # Slightly uneven terrain
        z = (
            0.02 * np.sin(x * 0.15)
            + 0.015 * np.cos(y * 0.1)
            + self.rng.normal(0, 0.015, num_points)
        )

        labels = np.zeros(num_points, dtype=np.int32)

        return np.column_stack((x, y, z)), labels

    def generate_vehicle(
        self,
        center_x,
        center_y,
        width=2.0,
        length=4.5,
        height=1.6,
        num_points=5000,
    ):
        """Generate a box-shaped vehicle."""

        x = self.rng.uniform(
            center_x - width / 2,
            center_x + width / 2,
            num_points,
        )

        y = self.rng.uniform(
            center_y - length / 2,
            center_y + length / 2,
            num_points,
        )

        z = self.rng.uniform(0.05, height, num_points)

        # Add measurement noise
        x += self.rng.normal(0, 0.025, num_points)
        y += self.rng.normal(0, 0.025, num_points)
        z += self.rng.normal(0, 0.02, num_points)

        labels = np.full(num_points, 1, dtype=np.int32)

        return np.column_stack((x, y, z)), labels

    def generate_pedestrian(
        self,
        center_x,
        center_y,
        height=1.75,
        num_points=1800,
    ):
        """Generate a simple pedestrian-shaped point cluster."""

        x = self.rng.normal(center_x, 0.22, num_points)
        y = self.rng.normal(center_y, 0.22, num_points)

        z = self.rng.uniform(0.05, height, num_points)

        # Make the upper body slightly narrower
        scale = 1.0 - 0.25 * (z / height)

        x = center_x + (x - center_x) * scale
        y = center_y + (y - center_y) * scale

        x += self.rng.normal(0, 0.015, num_points)
        y += self.rng.normal(0, 0.015, num_points)
        z += self.rng.normal(0, 0.015, num_points)

        labels = np.full(num_points, 2, dtype=np.int32)

        return np.column_stack((x, y, z)), labels

    def generate_static_obstacle(
        self,
        center_x,
        center_y,
        width=1.2,
        depth=1.2,
        height=3.0,
        num_points=2500,
    ):
        """Generate a static obstacle/pole-like object."""

        x = self.rng.uniform(
            center_x - width / 2,
            center_x + width / 2,
            num_points,
        )

        y = self.rng.uniform(
            center_y - depth / 2,
            center_y + depth / 2,
            num_points,
        )

        z = self.rng.uniform(0.05, height, num_points)

        x += self.rng.normal(0, 0.02, num_points)
        y += self.rng.normal(0, 0.02, num_points)
        z += self.rng.normal(0, 0.02, num_points)

        labels = np.full(num_points, 3, dtype=np.int32)

        return np.column_stack((x, y, z)), labels

    def generate_noise(self, num_points=2500):
        """Generate random LiDAR noise/outlier points."""

        x = self.rng.uniform(-50, 50, num_points)
        y = self.rng.uniform(1, 100, num_points)
        z = self.rng.uniform(-0.5, 5, num_points)

        labels = np.zeros(num_points, dtype=np.int32)

        return np.column_stack((x, y, z)), labels

    def generate_frame(self):
        """
        Generate one complete LiDAR frame.

        Returns:
            points: Nx3 numpy array
            labels: N numpy array
        """

        point_clouds = []
        label_arrays = []

        # Ground
        points, labels = self.generate_ground()
        point_clouds.append(points)
        label_arrays.append(labels)

        # Vehicles at different distances
        vehicles = [
            (3.0, 12.0),
            (-6.0, 28.0),
            (8.0, 42.0),
            (-12.0, 65.0),
        ]

        for x, y in vehicles:
            points, labels = self.generate_vehicle(x, y)
            point_clouds.append(points)
            label_arrays.append(labels)

        # Pedestrians
        pedestrians = [
            (-2.0, 8.0),
            (4.0, 18.0),
            (-8.0, 35.0),
        ]

        for x, y in pedestrians:
            points, labels = self.generate_pedestrian(x, y)
            point_clouds.append(points)
            label_arrays.append(labels)

        # Static obstacles
        obstacles = [
            (-15.0, 20.0),
            (15.0, 45.0),
            (-20.0, 75.0),
        ]

        for x, y in obstacles:
            points, labels = self.generate_static_obstacle(x, y)
            point_clouds.append(points)
            label_arrays.append(labels)

        # Sensor noise
        points, labels = self.generate_noise()
        point_clouds.append(points)
        label_arrays.append(labels)

        points = np.vstack(point_clouds)
        labels = np.concatenate(label_arrays)

        # Keep only points inside sensor range
        distance = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)

        mask = (
            (distance >= self.min_range)
            & (distance <= self.max_range)
        )

        points = points[mask]
        labels = labels[mask]

        return points, labels


if __name__ == "__main__":

    simulator = LidarSimulator()

    points, labels = simulator.generate_frame()

    print("=" * 50)
    print("ADAPT-LiDAR Point Cloud Simulator")
    print("=" * 50)

    print(f"Total points: {len(points):,}")

    print("\nSemantic distribution:")

    for class_id, class_name in simulator.classes.items():
        count = np.sum(labels == class_id)
        print(f"{class_name:20s}: {count:,}")

    print("\nPoint cloud shape:")
    print(points.shape)

    print("\nFirst 5 points:")
    print(points[:5])

    print("\nDistance range:")

    distances = np.sqrt(
        points[:, 0] ** 2 +
        points[:, 1] ** 2
    )

    print(f"Minimum: {distances.min():.2f} m")
    print(f"Maximum: {distances.max():.2f} m")