import numpy as np
from collections import defaultdict


class AdaptiveGrid:
    """
    Distance-adaptive 2.5D grid for ADAPT-LiDAR.

    Resolution:
        0–10 m    -> 5 cm
        10–25 m   -> 10 cm
        25–50 m   -> 25 cm
        50–100 m  -> 50 cm
    """

    ZONES = [
        (0.0, 10.0, 0.05),
        (10.0, 25.0, 0.10),
        (25.0, 50.0, 0.25),
        (50.0, 100.0, 0.50),
    ]

    CLASS_NAMES = {
        0: "ground",
        1: "vehicle",
        2: "pedestrian",
        3: "static_obstacle",
    }

    def get_resolution(self, distance):
        """Return the grid resolution for a given distance."""

        for min_distance, max_distance, resolution in self.ZONES:
            if min_distance <= distance < max_distance:
                return resolution

        # Include exactly 100 m in the final zone.
        if distance == 100.0:
            return 0.50

        return None

    def get_zone(self, distance):
        """Return the distance zone name."""

        if distance < 10:
            return "0-10m"

        if distance < 25:
            return "10-25m"

        if distance < 50:
            return "25-50m"

        if distance <= 100:
            return "50-100m"

        return "outside"

    def build_grid(self, points, labels):
        """
        Convert LiDAR points into an adaptive 2.5D grid.

        Each cell stores:
            x/y location
            elevation
            point count
            dominant semantic class
            confidence
            resolution
            distance zone
        """

        if len(points) != len(labels):
            raise ValueError(
                "Points and labels must have the same length."
            )

        cells = {}

        # Group points according to their distance zone.
        for zone_min, zone_max, resolution in self.ZONES:

            distance = np.sqrt(
                points[:, 0] ** 2 +
                points[:, 1] ** 2
            )

            if zone_max == 100.0:
                mask = (
                    (distance >= zone_min) &
                    (distance <= zone_max)
                )
            else:
                mask = (
                    (distance >= zone_min) &
                    (distance < zone_max)
                )

            zone_points = points[mask]
            zone_labels = labels[mask]

            if len(zone_points) == 0:
                continue

            # Grid indices.
            grid_x = np.floor(
                zone_points[:, 0] / resolution
            ).astype(np.int64)

            grid_y = np.floor(
                zone_points[:, 1] / resolution
            ).astype(np.int64)

            # Aggregate each occupied cell.
            unique_cells = defaultdict(list)

            for index, (gx, gy) in enumerate(
                zip(grid_x, grid_y)
            ):
                unique_cells[(gx, gy)].append(index)

            for (gx, gy), indices in unique_cells.items():

                cell_points = zone_points[indices]
                cell_labels = zone_labels[indices]

                # Mean elevation gives the 2.5D height.
                elevation = float(
                    np.mean(cell_points[:, 2])
                )

                point_count = len(indices)

                # Find dominant semantic class.
                class_counts = np.bincount(
                    cell_labels,
                    minlength=len(self.CLASS_NAMES)
                )

                dominant_class = int(
                    np.argmax(class_counts)
                )

                confidence = float(
                    class_counts[dominant_class]
                    / point_count
                )

                center_x = (
                    (gx + 0.5) * resolution
                )

                center_y = (
                    (gy + 0.5) * resolution
                )

                distance_from_sensor = float(
                    np.sqrt(
                        center_x ** 2 +
                        center_y ** 2
                    )
                )

                key = (
                    int(gx),
                    int(gy),
                    resolution
                )

                cells[key] = {
                    "x": round(center_x, 3),
                    "y": round(center_y, 3),
                    "elevation": round(elevation, 4),
                    "point_count": point_count,
                    "class_id": dominant_class,
                    "class_name": self.CLASS_NAMES[
                        dominant_class
                    ],
                    "confidence": round(
                        confidence, 3
                    ),
                    "resolution": resolution,
                    "zone": self.get_zone(
                        distance_from_sensor
                    ),
                }

        return cells

    def get_statistics(self, cells):
        """Calculate adaptive grid statistics."""

        statistics = {
            "total_cells": len(cells),
            "zones": {},
            "classes": {},
        }

        # Zone statistics
        for _, _, resolution in self.ZONES:

            zone_cells = [
                cell
                for cell in cells.values()
                if cell["resolution"] == resolution
            ]

            zone_name = {
                0.05: "0-10m",
                0.10: "10-25m",
                0.25: "25-50m",
                0.50: "50-100m",
            }[resolution]

            statistics["zones"][zone_name] = {
                "resolution": resolution,
                "cells": len(zone_cells),
                "points": sum(
                    cell["point_count"]
                    for cell in zone_cells
                ),
            }

        # Semantic class statistics
        for class_id, class_name in self.CLASS_NAMES.items():

            class_cells = [
                cell
                for cell in cells.values()
                if cell["class_id"] == class_id
            ]

            statistics["classes"][class_name] = len(
                class_cells
            )

        return statistics


if __name__ == "__main__":

    from lidar_simulator import LidarSimulator

    print("=" * 60)
    print("ADAPT-LiDAR Adaptive Grid Engine")
    print("=" * 60)

    # Generate LiDAR data
    simulator = LidarSimulator()

    points, labels = simulator.generate_frame()

    print(f"\nInput LiDAR points: {len(points):,}")

    # Build adaptive grid
    grid = AdaptiveGrid()

    cells = grid.build_grid(
        points,
        labels
    )

    statistics = grid.get_statistics(cells)

    print(
        f"Adaptive grid cells: "
        f"{statistics['total_cells']:,}"
    )

    print("\nResolution zones:")

    for zone, data in statistics["zones"].items():

        print(
            f"{zone:10s} | "
            f"{data['resolution'] * 100:5.1f} cm | "
            f"cells: {data['cells']:,} | "
            f"points: {data['points']:,}"
        )

    print("\nSemantic cells:")

    for class_name, count in (
        statistics["classes"].items()
    ):

        print(
            f"{class_name:20s}: "
            f"{count:,}"
        )

    print("\nSample cells:")

    for i, cell in enumerate(
        cells.values()
    ):

        if i >= 5:
            break

        print(cell)

    print("\nAdaptive grid generation complete.")