"""
Obstacle Detector
Real-time obstacle detection and avoidance zone marking.
Uses optical flow + edge density to estimate obstacle proximity
without a depth sensor (monocular proxy).
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass

log = logging.getLogger("ObstacleDetector")


@dataclass
class Obstacle:
    cx: int
    cy: int
    radius: int
    threat_level: float   # 0=safe, 1=critical
    zone: str             # 'left', 'center', 'right'


class ObstacleDetector:
    """
    Detects obstacles via:
    1. Edge density (high Canny density → cluttered/rocky)
    2. Optical flow magnitude (fast motion → close object)
    3. Contour-based blob detection on edge map
    """

    def __init__(self, frame_w: int = 960, frame_h: int = 720,
                 threat_threshold: float = 0.45):
        self.fw  = frame_w
        self.fh  = frame_h
        self.thr = threat_threshold
        self._prev_gray: Optional[np.ndarray] = None
        self._flow_mag  = np.zeros((frame_h, frame_w), dtype=np.float32)

    # ── Optical flow magnitude ────────────────────────────────────────────────
    def _update_flow(self, gray: np.ndarray):
        if self._prev_gray is not None and \
                self._prev_gray.shape == gray.shape:
            flow = cv2.calcOpticalFlowFarneback(
                self._prev_gray, gray, None,
                0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            self._flow_mag = cv2.resize(mag, (self.fw, self.fh))
        self._prev_gray = gray.copy()

    # ── Edge density map ──────────────────────────────────────────────────────
    def _edge_density(self, gray: np.ndarray) -> np.ndarray:
        blur  = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        # Dilate edges into regions
        kernel = np.ones((15, 15), np.uint8)
        dense  = cv2.dilate(edges, kernel)
        return dense

    # ── Blob-based obstacle extraction ───────────────────────────────────────
    def _extract_obstacles(self, density_map: np.ndarray,
                           gray: np.ndarray) -> List[Obstacle]:
        contours, _ = cv2.findContours(density_map, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        obstacles = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 800:  # ignore noise
                continue

            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            cx, cy, radius   = int(cx), int(cy), int(radius)

            # Threat = area fraction + flow magnitude in region
            area_frac = area / (self.fw * self.fh)
            flow_patch = self._flow_mag[
                max(0, cy-radius):cy+radius,
                max(0, cx-radius):cx+radius]
            flow_score = float(flow_patch.mean()) / 10.0 if flow_patch.size else 0

            threat = float(np.clip(area_frac * 15 + flow_score, 0, 1))
            if threat < 0.1:
                continue

            # Zone
            third = self.fw // 3
            if cx < third:
                zone = "left"
            elif cx < 2 * third:
                zone = "center"
            else:
                zone = "right"

            obstacles.append(Obstacle(cx, cy, radius, threat, zone))

        return sorted(obstacles, key=lambda o: -o.threat_level)

    # ── Main detect ───────────────────────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> List[Obstacle]:
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._update_flow(gray)
        density = self._edge_density(gray)
        return self._extract_obstacles(density, gray)

    # ── Draw ─────────────────────────────────────────────────────────────────
    def draw(self, frame: np.ndarray,
             obstacles: List[Obstacle]) -> np.ndarray:
        for obs in obstacles:
            if obs.threat_level < self.thr:
                color = (0, 200, 100)
            elif obs.threat_level < 0.7:
                color = (0, 165, 255)
            else:
                color = (0, 0, 255)

            cv2.circle(frame, (obs.cx, obs.cy), obs.radius, color, 2)
            cv2.putText(frame,
                        f"OBS {obs.threat_level:.2f}",
                        (obs.cx - 30, obs.cy - obs.radius - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # Zone danger summary
        zones = {"left": 0.0, "center": 0.0, "right": 0.0}
        for obs in obstacles:
            zones[obs.zone] = max(zones[obs.zone], obs.threat_level)

        y_bar = frame.shape[0] - 20
        third = frame.shape[1] // 3
        zone_colors = {z: (0, 0, 255) if v > 0.7
                         else (0, 165, 255) if v > 0.4
                         else (0, 200, 80)
                       for z, v in zones.items()}
        for i, (z, col) in enumerate(zone_colors.items()):
            x1 = i * third
            cv2.rectangle(frame, (x1, y_bar), (x1 + third, y_bar + 10), col, -1)
            cv2.putText(frame, z.upper(), (x1 + 5, y_bar + 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)

        return frame

    def avoidance_command(self, obstacles: List[Obstacle]) -> str:
        """Simple rule: if center is blocked, go to clearer side."""
        zones = {"left": 0.0, "center": 0.0, "right": 0.0}
        for obs in obstacles:
            zones[obs.zone] = max(zones[obs.zone], obs.threat_level)

        if zones["center"] > self.thr:
            return "LEFT" if zones["left"] < zones["right"] else "RIGHT"
        if max(zones.values()) > 0.8:
            return "STOP"
        return "FORWARD"
