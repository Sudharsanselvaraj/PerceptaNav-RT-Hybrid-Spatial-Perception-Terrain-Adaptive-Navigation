"""
Rover Configuration
All tunable parameters in one place.
"""

from dataclasses import dataclass


@dataclass
class RoverConfig:

    # ── Stream ─────────────────────────────────────────────────────────────────
    STREAM_URL:    str   = "http://10.15.140.40:81/stream"
    WINDOW_WIDTH:  int   = 960
    WINDOW_HEIGHT: int   = 720

    # ── Models ─────────────────────────────────────────────────────────────────
    DETECTOR_MODEL:   str = "yolov8n.pt"
    CLASSIFIER_MODEL: str = "runs/classify/train/weights/best.pt"

    # ── Energy ────────────────────────────────────────────────────────────────
    BATTERY_CAPACITY_WH: float = 50.0    # Etotal (Wh)
    IDLE_POWER_W:        float = 2.5     # baseline draw
    MOVE_POWER_W:        float = 8.0     # Pmove
    SAMPLE_POWER_W:      float = 3.0     # Psample
    SAMPLE_TIME_S:       float = 30.0    # tsample
    CRITICAL_ENERGY_WH:  float = 2.0     # low-power threshold

    # ── Rover motion ──────────────────────────────────────────────────────────
    ROVER_SPEED_MPS: float = 0.05        # ~5 cm/s (typical planetary rover)
    MAX_RANGE_M:     float = 10.0        # max detection range estimate

    # ── Decision ─────────────────────────────────────────────────────────────
    SAFETY_MARGIN: float = 0.05          # 5% energy buffer
