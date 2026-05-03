"""
Map Builder
Builds a 2D top-down occupancy/exploration map from rover's visited targets.
Implements a simplified occupancy grid updated from detections.
Visualizes as a minimap overlay on the frame.
"""

import cv2
import numpy as np
import json
import logging
from pathlib import Path
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass, field

log = logging.getLogger("MapBuilder")

# Cell types
FREE      = 0
OBSTACLE  = 1
EXPLORED  = 2
TARGET    = 3
ROVER     = 4

CELL_COLORS = {
    FREE:     (30,  30,  30),
    OBSTACLE: (0,   0,  200),
    EXPLORED: (0,  120,  80),
    TARGET:   (0,  220, 255),
    ROVER:    (0,  255,  80),
}


@dataclass
class MapEvent:
    x: float    # world x (m)
    y: float    # world y (m)
    cell_type: int
    label: str = ""


class MapBuilder:
    """
    Maintains a 2D grid map of the rover's exploration area.
    Grid is updated each time a target is selected or an obstacle is detected.
    Renders as a minimap in the corner of the frame.
    """

    def __init__(self,
                 grid_size: int = 100,
                 world_range_m: float = 20.0,
                 minimap_px: int = 180):
        self.grid_size   = grid_size
        self.world_range = world_range_m
        self.minimap_px  = minimap_px
        self.cell_m      = world_range_m / grid_size

        self._grid       = np.zeros((grid_size, grid_size), dtype=np.uint8)
        self._rover_pos  = (grid_size // 2, grid_size // 2)   # start at center
        self._heading    = 0.0    # degrees, 0 = north
        self._events: List[MapEvent] = []

        # Mark rover start
        self._grid[self._rover_pos] = ROVER

    # ── World ↔ Grid conversion ───────────────────────────────────────────────
    def _world_to_grid(self, wx: float, wy: float) -> Tuple[int, int]:
        origin = self.grid_size // 2
        gx = int(np.clip(origin + wx / self.cell_m, 0, self.grid_size - 1))
        gy = int(np.clip(origin - wy / self.cell_m, 0, self.grid_size - 1))
        return gx, gy

    # ── Update ────────────────────────────────────────────────────────────────
    def update_rover(self, dx: float = 0.0, dy: float = 0.1):
        """Move rover by (dx, dy) in world meters."""
        rx, ry = self._rover_pos
        # Clear old rover pos
        if self._grid[ry, rx] == ROVER:
            self._grid[ry, rx] = EXPLORED

        # Move
        nx = int(np.clip(rx + dx / self.cell_m, 0, self.grid_size - 1))
        ny = int(np.clip(ry - dy / self.cell_m, 0, self.grid_size - 1))
        self._rover_pos = (nx, ny)
        self._grid[ny, nx] = ROVER

        # Mark path as explored
        cv2.line(self._grid,
                 (rx, ry), (nx, ny), EXPLORED, 1)

    def mark_target(self, distance_m: float, angle_deg: float = 0.0,
                    label: str = "target"):
        """Mark a detected target at (distance, angle) from rover."""
        rad = np.radians(self._heading + angle_deg)
        wx  = distance_m * np.sin(rad)
        wy  = distance_m * np.cos(rad)
        gx, gy = self._world_to_grid(wx, wy)
        self._grid[gy, gx] = TARGET
        self._events.append(MapEvent(wx, wy, TARGET, label))
        log.debug(f"Target marked: {label} at ({wx:.1f},{wy:.1f})m → grid({gx},{gy})")

    def mark_obstacle(self, distance_m: float, angle_deg: float = 0.0):
        """Mark obstacle cell."""
        rad = np.radians(self._heading + angle_deg)
        wx  = distance_m * np.sin(rad)
        wy  = distance_m * np.cos(rad)
        gx, gy = self._world_to_grid(wx, wy)
        self._grid[gy, gx] = OBSTACLE

    # ── Minimap render ────────────────────────────────────────────────────────
    def render_minimap(self) -> np.ndarray:
        mm = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        for cell_type, color in CELL_COLORS.items():
            mask = self._grid == cell_type
            mm[mask] = color

        # Scale to minimap_px
        mm = cv2.resize(mm, (self.minimap_px, self.minimap_px),
                        interpolation=cv2.INTER_NEAREST)

        # Border
        cv2.rectangle(mm, (0, 0), (self.minimap_px - 1, self.minimap_px - 1),
                      (100, 100, 100), 1)
        cv2.putText(mm, "MAP", (4, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        return mm

    def overlay_minimap(self, frame: np.ndarray,
                        corner: str = "top_right") -> np.ndarray:
        mm = self.render_minimap()
        h, w = frame.shape[:2]
        mh, mw = mm.shape[:2]

        if corner == "top_right":
            x1, y1 = w - mw - 10, 10
        elif corner == "top_left":
            x1, y1 = 10, 10
        elif corner == "bottom_right":
            x1, y1 = w - mw - 10, h - mh - 10
        else:
            x1, y1 = 10, h - mh - 10

        frame[y1:y1 + mh, x1:x1 + mw] = mm
        return frame

    # ── Exploration coverage ───────────────────────────────────────────────────
    def coverage_pct(self) -> float:
        explored = np.sum(self._grid > 0)
        total    = self.grid_size ** 2
        return round(100.0 * explored / total, 2)

    def save(self, path: str = "logs/map.json"):
        with open(path, "w") as f:
            json.dump({
                "grid_size":    self.grid_size,
                "world_range_m": self.world_range,
                "coverage_pct": self.coverage_pct(),
                "events": [{"x": e.x, "y": e.y,
                             "label": e.label} for e in self._events],
            }, f, indent=2)
