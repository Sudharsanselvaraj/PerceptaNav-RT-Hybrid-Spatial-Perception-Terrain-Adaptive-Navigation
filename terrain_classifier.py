"""
Terrain Classifier
Multi-class terrain segmentation and traversability scoring.
Goes beyond binary rock/no-rock — estimates slope, roughness, traversal risk.
"""

import cv2
import numpy as np
import logging
from dataclasses import dataclass
from typing import Tuple, Dict, List

log = logging.getLogger("TerrainClassifier")


@dataclass
class TerrainPatch:
    label: str
    traversability: float   # 0=blocked, 1=fully safe
    roughness: float        # 0=flat, 1=very rough
    slope_deg: float
    region: Tuple[int, int, int, int]  # x1,y1,x2,y2


# Traversability scores per terrain class
TRAVERSABILITY = {
    "flat_soil":  0.95,
    "sand":       0.80,
    "gravel":     0.65,
    "rock":       0.20,
    "dense_rock": 0.05,
    "shadow":     0.50,
    "unknown":    0.40,
}


class TerrainClassifier:
    """
    Divides frame into a grid, analyses each cell for terrain type
    and traversability. Used by PathPlanner and DecisionEngine.
    """

    def __init__(self, grid_rows: int = 4, grid_cols: int = 6):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols

    # ── Per-patch analysis ────────────────────────────────────────────────────
    def _analyse_patch(self, patch: np.ndarray) -> TerrainPatch:
        gray  = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        h, w  = patch.shape[:2]

        # Roughness via Laplacian variance
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        roughness = float(np.clip(lap_var / 500.0, 0, 1))

        # Slope proxy via Sobel gradient magnitude
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag  = np.sqrt(sx**2 + sy**2).mean()
        slope_deg = float(np.clip(grad_mag / 5.0, 0, 45))

        # Color features (BGR mean)
        b, g, r = [patch[:, :, i].mean() for i in range(3)]

        # Heuristic label assignment
        if roughness > 0.7:
            label = "dense_rock"
        elif roughness > 0.4:
            label = "rock"
        elif roughness > 0.2:
            label = "gravel"
        elif r > 120 and g < 100:
            label = "sand"
        elif b < 80 and g < 80 and r < 80:
            label = "shadow"
        else:
            label = "flat_soil"

        traversability = TRAVERSABILITY.get(label, 0.4) * (1 - slope_deg / 90)

        return TerrainPatch(
            label=label,
            traversability=float(np.clip(traversability, 0, 1)),
            roughness=roughness,
            slope_deg=slope_deg,
            region=(0, 0, w, h)
        )

    # ── Full frame grid analysis ───────────────────────────────────────────────
    def analyse(self, frame: np.ndarray) -> List[List[TerrainPatch]]:
        h, w  = frame.shape[:2]
        ph    = h // self.grid_rows
        pw    = w // self.grid_cols
        grid  = []

        for r in range(self.grid_rows):
            row = []
            for c in range(self.grid_cols):
                y1, y2 = r * ph, (r + 1) * ph
                x1, x2 = c * pw, (c + 1) * pw
                patch  = frame[y1:y2, x1:x2]
                tp     = self._analyse_patch(patch)
                tp.region = (x1, y1, x2, y2)
                row.append(tp)
            grid.append(row)

        return grid

    # ── Overlay ───────────────────────────────────────────────────────────────
    def draw_grid(self, frame: np.ndarray,
                  grid: List[List[TerrainPatch]]) -> np.ndarray:
        overlay = frame.copy()
        COLOR_MAP = {
            "flat_soil":  (0, 200, 80),
            "sand":       (0, 180, 220),
            "gravel":     (0, 140, 255),
            "rock":       (0, 60, 255),
            "dense_rock": (0, 0, 200),
            "shadow":     (100, 100, 100),
            "unknown":    (180, 180, 180),
        }
        for row in grid:
            for tp in row:
                x1, y1, x2, y2 = tp.region
                color = COLOR_MAP.get(tp.label, (200, 200, 200))
                alpha = 0.3 * (1 - tp.traversability) + 0.05
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

        for row in grid:
            for tp in row:
                x1, y1, x2, y2 = tp.region
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.putText(frame, f"{tp.traversability:.2f}",
                            (x1 + 4, y1 + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        return frame

    def safest_corridor(self, grid: List[List[TerrainPatch]]) -> int:
        """Returns column index with highest avg traversability (for steering)."""
        col_scores = []
        for c in range(self.grid_cols):
            score = np.mean([grid[r][c].traversability
                             for r in range(self.grid_rows)])
            col_scores.append(score)
        return int(np.argmax(col_scores))
