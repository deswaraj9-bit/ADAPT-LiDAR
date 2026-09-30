from pydantic import BaseModel
from typing import Dict, List, Any


class ZoneStats(BaseModel):
    resolution: float
    cells: int
    points: int


class GridStats(BaseModel):
    total_cells: int
    zones: Dict[str, ZoneStats]
    classes: Dict[str, int]


class LidarFrameResponse(BaseModel):
    total_points: int
    min_distance: float
    max_distance: float
    grid_cells: int
    grid_statistics: GridStats