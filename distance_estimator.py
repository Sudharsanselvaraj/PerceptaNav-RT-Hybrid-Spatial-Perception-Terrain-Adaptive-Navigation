"""
Distance Estimator
Monocular distance estimation using:
1. Bounding box apparent size method (known object size assumption)
2. Vertical position in frame (perspective geometry)
3. Calibration-based focal length method
No depth sensor required — works with single ESP32-CAM.
"""
import pandas as pd 
import cv2
import numpy as np
import logging
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

log = logging.getLogger("DistanceEstimator")

# Reference object sizes (real-world height in meters)
REFERENCE_HEIGHTS: Dict[str, float] = {
    "rock":     0.15,    # ~15 cm typical target rock
    "mineral":  0.08,
    "gravel":   0.05,
    "soil":     0.10,
    "person":   1.70,
    "default":  0.15,
}


@dataclass
class DistanceEstimate:
    label: str
    distance_m: float
    method: str
    confidence: float


class DistanceEstimator:
    """
    Estimates target distance using monocular camera geometry.

    Calibration: provide focal length (pixels) or run calibrate()
    with a known object at a known distance.

    Formula: D = (H_real × f) / H_pixel
    where H_real = real object height, f = focal length, H_pixel = bbox height
    """

    def __init__(self,
                 focal_length_px: float = 750.0,
                 frame_h: int = 720,
                 frame_w: int = 960):
        self.f      = focal_length_px
        self.fh     = frame_h
        self.fw     = frame_w
        self._calib_log: list = []

    # ── Bounding-box method ────────────────────────────────────────────────────
    def from_bbox(self, label: str,
                  box: Tuple[float, float, float, float]) -> DistanceEstimate:
        """
        D = (H_real × f) / H_pixel
        """
        x1, y1, x2, y2 = box
        h_pixel = max(abs(y2 - y1), 1.0)
        h_real  = REFERENCE_HEIGHTS.get(label.lower(),
                                         REFERENCE_HEIGHTS["default"])
        dist    = (h_real * self.f) / h_pixel
        dist    = float(np.clip(dist, 0.1, 30.0))
        conf    = float(np.clip(1.0 - abs(h_pixel - 80) / 400, 0.3, 0.95))
        return DistanceEstimate(label, round(dist, 2), "bbox_focal", conf)

    # ── Vertical position method ───────────────────────────────────────────────
    def from_vertical_position(self, label: str,
                                cy: float,
                                horizon_y: float = None) -> DistanceEstimate:
        """
        Objects near horizon → far. Objects near bottom → close.
        Linear mapping: horizon → max_range, bottom → min_range.
        """
        horizon_y = horizon_y or self.fh * 0.55
        min_d, max_d = 0.3, 10.0
        frac = float(np.clip((cy - horizon_y) / (self.fh - horizon_y), 0, 1))
        dist = max_d - frac * (max_d - min_d)
        return DistanceEstimate(label, round(dist, 2), "vertical_pos", 0.6)

    # ── Combined estimate ─────────────────────────────────────────────────────
    def estimate(self, label: str,
                 box: Tuple[float, float, float, float]) -> DistanceEstimate:
        x1, y1, x2, y2 = box
        cy = (y1 + y2) / 2.0

        d_bbox = self.from_bbox(label, box)
        d_vert = self.from_vertical_position(label, cy)

        # Weighted average: higher weight to bbox method
        w1, w2 = d_bbox.confidence, d_vert.confidence
        dist   = (d_bbox.distance_m * w1 + d_vert.distance_m * w2) / (w1 + w2)
        conf   = (w1 + w2) / 2.0

        return DistanceEstimate(label, round(dist, 2), "combined", round(conf, 3))

    # ── Calibration ───────────────────────────────────────────────────────────
    def calibrate(self, label: str,
                  box: Tuple[float, float, float, float],
                  true_distance_m: float):
        """
        Given a known true distance, back-compute focal length and update.
        """
        x1, y1, x2, y2 = box
        h_pixel = max(abs(y2 - y1), 1.0)
        h_real  = REFERENCE_HEIGHTS.get(label.lower(),
                                         REFERENCE_HEIGHTS["default"])
        f_new   = (true_distance_m * h_pixel) / h_real
        old_f   = self.f
        self.f  = 0.8 * self.f + 0.2 * f_new   # EMA update
        self._calib_log.append({
            "label": label, "true_m": true_distance_m,
            "f_old": round(old_f, 1), "f_new": round(self.f, 1)
        })
        log.info(f"Calibrated: f {old_f:.1f} → {self.f:.1f} px")

    # ── Draw ─────────────────────────────────────────────────────────────────
    def draw(self, frame: np.ndarray,
             estimate: DistanceEstimate,
             box: Tuple[float, float, float, float]) -> np.ndarray:
        x1, y1 = int(box[0]), int(box[1])
        cv2.putText(frame, f"{estimate.distance_m:.1f}m",
                    (x1, max(y1 - 22, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 220, 0), 1)
        return frame
