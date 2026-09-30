from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import math
import time
import inspect

from lidar_simulator import LidarSimulator
from adaptive_grid import AdaptiveGrid
from ml_model import LidarSemanticModel
from risk_controller import RiskAwareResolutionController
from motion_tracker import MotionTracker


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="ADAPT-LiDAR",
    description="Risk-Aware Adaptive Variable Resolution 2.5D LiDAR Mapping",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# COMPONENTS
# ============================================================

simulator = LidarSimulator()
grid_engine = AdaptiveGrid()
semantic_model = LidarSemanticModel()
risk_controller = RiskAwareResolutionController()
motion_tracker = MotionTracker(max_match_distance=3.0)

frame_id = 0


# ============================================================
# CLASSES
# ============================================================

CLASS_NAMES = {
    0: "ground",
    1: "vehicle",
    2: "pedestrian",
    3: "static_obstacle",
}


# ============================================================
# HELPERS
# ============================================================

def distance_xy(x, y):
    return math.sqrt(x * x + y * y)


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def normalize_resolution(value):
    value = safe_float(value, 0.50)

    if value <= 0.05:
        return 0.05
    if value <= 0.10:
        return 0.10
    if value <= 0.25:
        return 0.25

    return 0.50


# ============================================================
# SAFE RISK CONTROLLER
# ============================================================

def calculate_risk(
    distance,
    class_name,
    confidence,
    speed,
    terrain_variation=0.10,
    object_id=None
):
    """
    Calls the user's existing RiskAwareResolutionController
    while protecting the API from signature differences.
    """

    try:

        method = risk_controller.get_resolution

        signature = inspect.signature(method)

        parameters = signature.parameters

        kwargs = {}

        possible_values = {
            "distance": distance,
            "distance_m": distance,

            "class_name": class_name,
            "class": class_name,

            "confidence": confidence,

            "speed_mps": speed,
            "speed": speed,

            "terrain_variation": terrain_variation,
            "geometry_risk": terrain_variation,

            "object_id": object_id,
        }

        for name in parameters:

            if name in possible_values:
                kwargs[name] = possible_values[name]

        result = method(**kwargs)

        if not isinstance(result, dict):
            result = {}

        total_risk = safe_float(
            result.get(
                "total_risk",
                result.get("risk", 0.0)
            )
        )

        normal_resolution = normalize_resolution(
            result.get(
                "normal_resolution",
                result.get(
                    "base_resolution",
                    0.50
                )
            )
        )

        desired_resolution = normalize_resolution(
            result.get(
                "desired_resolution",
                normal_resolution
            )
        )

        risk_resolution = normalize_resolution(
            result.get(
                "risk_aware_resolution",
                result.get(
                    "resolution",
                    desired_resolution
                )
            )
        )

        return {
            "risk": round(total_risk, 3),
            "normal_resolution": normal_resolution,
            "desired_resolution": desired_resolution,
            "risk_aware_resolution": risk_resolution,

            "distance_risk": safe_float(
                result.get("distance_risk", 0.0)
            ),

            "semantic_risk": safe_float(
                result.get("semantic_risk", 0.0)
            ),

            "uncertainty_risk": safe_float(
                result.get("uncertainty_risk", 0.0)
            ),

            "motion_risk": safe_float(
                result.get("motion_risk", 0.0)
            ),

            "geometry_risk": safe_float(
                result.get("geometry_risk", terrain_variation)
            ),

            "controller_status": "active",
        }

    except Exception as error:

        # Safety fallback.
        # The feature remains active using the same
        # risk-aware principle.

        class_risk = {
            "ground": 0.05,
            "vehicle": 0.70,
            "pedestrian": 0.90,
            "static_obstacle": 0.60,
        }.get(class_name, 0.20)

        distance_risk = max(
            0.0,
            min(
                1.0,
                1.0 - distance / 100.0
            )
        )

        uncertainty_risk = max(
            0.0,
            min(
                1.0,
                1.0 - confidence
            )
        )

        motion_risk = max(
            0.0,
            min(
                1.0,
                speed / 10.0
            )
        )

        risk = (
            0.20 * distance_risk +
            0.30 * class_risk +
            0.20 * uncertainty_risk +
            0.20 * motion_risk +
            0.10 * terrain_variation
        )

        if distance <= 10:
            normal = 0.05
        elif distance <= 25:
            normal = 0.10
        elif distance <= 50:
            normal = 0.25
        else:
            normal = 0.50

        if risk >= 0.75:
            resolution = 0.05
        elif risk >= 0.55:
            resolution = min(normal, 0.10)
        elif risk >= 0.30:
            resolution = min(normal, 0.25)
        else:
            resolution = normal

        return {
            "risk": round(risk, 3),
            "normal_resolution": normal,
            "desired_resolution": resolution,
            "risk_aware_resolution": resolution,
            "distance_risk": round(distance_risk, 3),
            "semantic_risk": round(class_risk, 3),
            "uncertainty_risk": round(
                uncertainty_risk, 3
            ),
            "motion_risk": round(
                motion_risk, 3
            ),
            "geometry_risk": round(
                terrain_variation, 3
            ),
            "controller_status": "fallback_active",
        }


# ============================================================
# OBJECT EXTRACTION
# ============================================================

def extract_objects(
    points,
    labels,
    confidences
):
    """
    Lightweight spatial object extraction.
    No sklearn/scipy required.
    """

    groups = {}

    bin_size = 1.5

    for i in range(len(points)):

        class_id = int(labels[i])

        if class_id == 0:
            continue

        x = float(points[i][0])
        y = float(points[i][1])
        z = float(points[i][2])

        distance = distance_xy(x, y)

        if distance > 100:
            continue

        bx = int(math.floor(x / bin_size))
        by = int(math.floor(y / bin_size))

        key = (
            class_id,
            bx,
            by
        )

        if key not in groups:
            groups[key] = {
                "class_id": class_id,
                "points": [],
                "confidences": [],
            }

        groups[key]["points"].append(
            (x, y, z)
        )

        groups[key]["confidences"].append(
            float(confidences[i])
        )

    objects = []

    for group in groups.values():

        if len(group["points"]) < 8:
            continue

        pts = np.asarray(
            group["points"],
            dtype=np.float32
        )

        centroid = pts.mean(axis=0)

        confidence = float(
            np.mean(
                group["confidences"]
            )
        )

        x = float(centroid[0])
        y = float(centroid[1])
        z = float(centroid[2])

        distance = distance_xy(x, y)

        class_id = group["class_id"]

        objects.append({
            "class_id": class_id,
            "class": CLASS_NAMES.get(
                class_id,
                "unknown"
            ),
            "centroid": (
                round(x, 3),
                round(y, 3),
                round(z, 3)
            ),
            "distance": round(
                distance,
                3
            ),
            "confidence": round(
                confidence,
                3
            ),
            "points": len(
                group["points"]
            )
        })

    objects.sort(
        key=lambda x: x["distance"]
    )

    return objects[:30]


# ============================================================
# DISPLAY POINTS
# ============================================================

def make_display_points(
    points,
    labels,
    confidences,
    maximum=6000
):

    total = len(points)

    if total <= maximum:

        indices = np.arange(total)

    else:

        indices = np.linspace(
            0,
            total - 1,
            maximum,
            dtype=int
        )

    result = []

    for i in indices:

        p = points[i]

        class_id = int(labels[i])

        result.append({
            "x": round(float(p[0]), 3),
            "y": round(float(p[1]), 3),
            "z": round(float(p[2]), 3),
            "label": class_id,
            "class": CLASS_NAMES.get(
                class_id,
                "unknown"
            ),
            "confidence": round(
                float(confidences[i]),
                3
            ),
            "distance": round(
                distance_xy(
                    float(p[0]),
                    float(p[1])
                ),
                3
            )
        })

    return result


# ============================================================
# DISPLAY GRID
# ============================================================

def make_display_cells(cells, maximum=8000):

    if not isinstance(cells, list):
        return []

    if len(cells) <= maximum:
        selected = cells
    else:
        indices = np.linspace(
            0,
            len(cells) - 1,
            maximum,
            dtype=int
        )

        selected = [
            cells[i]
            for i in indices
        ]

    result = []

    for index, cell in enumerate(selected):

        if not isinstance(cell, dict):
            continue

        # Try common coordinate names used by grid engines.
        x = cell.get(
            "x",
            cell.get(
                "center_x",
                cell.get(
                    "grid_x",
                    0
                )
            )
        )

        y = cell.get(
            "y",
            cell.get(
                "center_y",
                cell.get(
                    "grid_y",
                    0
                )
            )
        )

        resolution = cell.get(
            "resolution",
            cell.get(
                "cell_size",
                0.50
            )
        )

        elevation = cell.get(
            "elevation",
            cell.get(
                "z",
                cell.get(
                    "mean_z",
                    0
                )
            )
        )

        result.append({
            "id": index,
            "x": safe_float(x),
            "y": safe_float(y),
            "resolution": normalize_resolution(
                resolution
            ),
            "elevation": safe_float(
                elevation
            ),
            "class": cell.get(
                "class",
                cell.get(
                    "semantic_class",
                    "ground"
                )
            )
        })

    return result


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "project": "ADAPT-LiDAR",
        "problem_statement": "SIH26053",
        "status": "running",
        "version": "3.0.0",
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/api/health")
def health():

    return {
        "status": "healthy",
        "lidar": "ready",
        "pointnet": "ready",
        "motion_tracker": "ready",
        "risk_controller": "ready",
        "adaptive_grid": "ready",
    }


# ============================================================
# FULL PERCEPTION PIPELINE
# ============================================================

@app.get("/api/lidar/perception")
def perception():

    global frame_id

    start = time.perf_counter()

    frame_id += 1

    # --------------------------------------------------------
    # 1. RAW LIDAR
    # --------------------------------------------------------

    points, ground_truth = (
        simulator.generate_frame()
    )

    # --------------------------------------------------------
    # 2. POINTNET
    # --------------------------------------------------------

    predicted_labels, confidences = (
        semantic_model.predict(points)
    )

    # --------------------------------------------------------
    # 3. SEMANTIC COUNTS
    # --------------------------------------------------------

    class_counts = (
        semantic_model.get_class_counts(
            predicted_labels
        )
    )

    # --------------------------------------------------------
    # 4. SYNTHETIC AGREEMENT
    # --------------------------------------------------------

    agreement = (
        predicted_labels ==
        ground_truth
    )

    synthetic_accuracy = float(
        np.mean(agreement)
    )

    # --------------------------------------------------------
    # 5. OBJECT EXTRACTION
    # --------------------------------------------------------

    detected_objects = extract_objects(
        points,
        predicted_labels,
        confidences
    )

    # --------------------------------------------------------
    # 6. MOTION TRACKING
    # --------------------------------------------------------

    tracked = motion_tracker.update(
        detected_objects
    )

    # --------------------------------------------------------
    # 7. RISK ANALYSIS
    # --------------------------------------------------------

    final_objects = []

    for obj in tracked:

        object_id = obj[
            "object_id"
        ]

        class_id = obj[
            "class_id"
        ]

        class_name = CLASS_NAMES.get(
            class_id,
            "static_obstacle"
        )

        centroid = obj[
            "centroid"
        ]

        speed = safe_float(
            obj.get(
                "speed_mps",
                0
            )
        )

        confidence = 0.5

        for detected in detected_objects:

            if (
                detected["centroid"]
                == centroid
            ):

                confidence = (
                    detected[
                        "confidence"
                    ]
                )

                break

        distance = distance_xy(
            centroid[0],
            centroid[1]
        )

        # Geometry complexity for the
        # prototype risk model.
        geometry_risk = 0.10

        risk = calculate_risk(
            distance=distance,
            class_name=class_name,
            confidence=confidence,
            speed=speed,
            terrain_variation=geometry_risk,
            object_id=object_id
        )

        final_objects.append({

            "object_id": object_id,

            "class_id": class_id,

            "class": class_name,

            "x": centroid[0],

            "y": centroid[1],

            "z": centroid[2],

            "centroid": list(
                centroid
            ),

            "distance": round(
                distance,
                2
            ),

            "confidence": round(
                confidence,
                3
            ),

            "speed_mps": round(
                speed,
                3
            ),

            "risk": risk["risk"],

            "normal_resolution": (
                risk[
                    "normal_resolution"
                ]
            ),

            "risk_aware_resolution": (
                risk[
                    "risk_aware_resolution"
                ]
            ),

            "distance_risk": (
                risk[
                    "distance_risk"
                ]
            ),

            "semantic_risk": (
                risk[
                    "semantic_risk"
                ]
            ),

            "uncertainty_risk": (
                risk[
                    "uncertainty_risk"
                ]
            ),

            "motion_risk": (
                risk[
                    "motion_risk"
                ]
            ),

            "controller": risk[
                "controller_status"
            ]
        })

    # --------------------------------------------------------
    # 8. ADAPTIVE GRID
    # --------------------------------------------------------

    grid_cells = grid_engine.build_grid(
        points,
        predicted_labels
    )

    grid_statistics = (
        grid_engine.get_statistics(
            grid_cells
        )
    )

    display_cells = make_display_cells(
        grid_cells
    )

    # --------------------------------------------------------
    # 9. POINT CLOUD FOR FRONTEND
    # --------------------------------------------------------

    display_points = make_display_points(
        points,
        predicted_labels,
        confidences
    )

    # --------------------------------------------------------
    # 10. GLOBAL RISK
    # --------------------------------------------------------

    if final_objects:

        global_risk = max(
            x["risk"]
            for x in final_objects
        )

    else:

        global_risk = 0.0

    # --------------------------------------------------------
    # 11. RESOLUTION DISTRIBUTION
    # --------------------------------------------------------

    resolution_distribution = {
        "5cm": 0,
        "10cm": 0,
        "25cm": 0,
        "50cm": 0
    }

    for obj in final_objects:

        r = obj[
            "risk_aware_resolution"
        ]

        if r <= 0.05:
            resolution_distribution[
                "5cm"
            ] += 1

        elif r <= 0.10:
            resolution_distribution[
                "10cm"
            ] += 1

        elif r <= 0.25:
            resolution_distribution[
                "25cm"
            ] += 1

        else:
            resolution_distribution[
                "50cm"
            ] += 1

    # --------------------------------------------------------
    # 12. PERFORMANCE
    # --------------------------------------------------------

    latency = (
        time.perf_counter() -
        start
    ) * 1000

    fps = (
        1000 / latency
        if latency > 0
        else 0
    )

    # --------------------------------------------------------
    # FINAL RESPONSE
    # --------------------------------------------------------

    return {

        "project": "ADAPT-LiDAR",

        "problem_statement": "SIH26053",

        "frame": {

            "id": frame_id,

            "total_points": int(
                len(points)
            ),

            "display_points": len(
                display_points
            ),

            "range_m": 100
        },

        "semantic": {

            "model": "Lightweight PointNet",

            "classes": class_counts,

            "mean_confidence": round(
                float(
                    np.mean(
                        confidences
                    )
                ),
                3
            ),

            "synthetic_ground_truth_agreement": round(
                synthetic_accuracy,
                4
            ),

            "validation_accuracy": (
                semantic_model.validation_accuracy
            )
        },

        "objects": final_objects,

        "risk": {

            "global_risk": round(
                global_risk,
                3
            ),

            "object_count": len(
                final_objects
            )
        },

        "resolution": {

            "distance_zones": {
                "0-10m": "5cm",
                "10-25m": "10cm",
                "25-50m": "25cm",
                "50-100m": "50cm"
            },

            "risk_distribution": (
                resolution_distribution
            ),

            "principle": (
                "Resolution follows risk, "
                "not distance alone"
            )
        },

        "grid": {

            "total_cells": len(
                grid_cells
            ),

            "display_cells": len(
                display_cells
            ),

            "statistics": grid_statistics,

            "cells": display_cells
        },

        "points": display_points,

        "metrics": {

            "latency_ms": round(
                latency,
                2
            ),

            "fps": round(
                fps,
                2
            )
        },

        "pipeline": {

            "raw_lidar": True,

            "pointnet": True,

            "semantic_segmentation": True,

            "object_extraction": True,

            "motion_tracking": True,

            "risk_aware_controller": True,

            "temporal_hysteresis": True,

            "adaptive_2_5d_grid": True,

            "real_time_metrics": True
        }
    }


# ============================================================
# OLD FRAME ENDPOINT
# ============================================================

@app.get("/api/lidar/frame")
def lidar_frame():

    points, labels = (
        simulator.generate_frame()
    )

    distances = np.sqrt(
        points[:, 0] ** 2 +
        points[:, 1] ** 2
    )

    cells = grid_engine.build_grid(
        points,
        labels
    )

    statistics = (
        grid_engine.get_statistics(
            cells
        )
    )

    return {

        "total_points": int(
            len(points)
        ),

        "min_distance": round(
            float(
                distances.min()
            ),
            2
        ),

        "max_distance": round(
            float(
                distances.max()
            ),
            2
        ),

        "grid_cells": len(
            cells
        ),

        "grid_statistics": statistics
    }