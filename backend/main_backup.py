from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from lidar_simulator import LidarSimulator
from adaptive_grid import AdaptiveGrid


app = FastAPI(
    title="ADAPT-LiDAR API",
    description="Adaptive Variable Resolution 2.5D LiDAR Mapping",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Create simulator and grid engine once.
simulator = LidarSimulator()
grid_engine = AdaptiveGrid()


@app.get("/")
def root():
    return {
        "project": "ADAPT-LiDAR",
        "problem_statement": "SIH26053",
        "status": "running"
    }


@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "lidar": "simulation",
        "grid_engine": "ready"
    }


@app.get("/api/lidar/frame")
def lidar_frame():

    # Generate LiDAR frame
    points, labels = simulator.generate_frame()

    # Calculate distances
    distances = (
        points[:, 0] ** 2 +
        points[:, 1] ** 2
    ) ** 0.5

    # Generate adaptive grid
    cells = grid_engine.build_grid(
        points,
        labels
    )

    # Calculate statistics
    statistics = grid_engine.get_statistics(
        cells
    )

    return {
        "total_points": int(len(points)),
        "min_distance": round(
            float(distances.min()), 2
        ),
        "max_distance": round(
            float(distances.max()), 2
        ),
        "grid_cells": int(len(cells)),
        "grid_statistics": statistics
    }


@app.get("/api/grid/stats")
def grid_stats():

    points, labels = simulator.generate_frame()

    cells = grid_engine.build_grid(
        points,
        labels
    )

    statistics = grid_engine.get_statistics(
        cells
    )

    return statistics