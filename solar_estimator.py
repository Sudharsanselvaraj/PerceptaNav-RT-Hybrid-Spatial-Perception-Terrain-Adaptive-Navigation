"""
Solar Energy Estimator
Models solar irradiance and charging power from frame brightness.
On Mars, dust storms and sun angle dramatically affect solar input.
Feeds back into EnergyModel as charging offset.
"""

import cv2
import numpy as np
import logging
import time
from collections import deque
from typing import Tuple

log = logging.getLogger("SolarEstimator")

# Mars solar constants
SOLAR_CONSTANT_MARS_W_M2 = 590.0    # ~43% of Earth's at Mars distance
PANEL_AREA_M2             = 0.5     # rover panel area
PANEL_EFFICIENCY          = 0.22    # typical solar cell efficiency
MAX_PANEL_POWER_W         = SOLAR_CONSTANT_MARS_W_M2 * PANEL_AREA_M2 * PANEL_EFFICIENCY


class SolarEstimator:
    """
    Estimates solar charging power from:
    1. Frame luminance (sky/ambient brightness)
    2. Time-of-day simulation (cosine solar angle model)
    3. Dust opacity factor (histogram-based)

    Output: estimated charging power in Watts
    """

    def __init__(self, sol_duration_s: float = 88775.0):
        """
        sol_duration_s: length of one Martian sol in seconds (~24h 37m)
        """
        self.sol_duration = sol_duration_s
        self._start_time  = time.time()
        self._power_hist  = deque(maxlen=100)

    # ── Solar angle model ─────────────────────────────────────────────────────
    def _solar_elevation(self) -> float:
        """
        Simulates solar elevation angle over a Martian sol.
        Returns sin(elevation) ∈ [0, 1], 0 = night/horizon, 1 = zenith.
        """
        elapsed  = (time.time() - self._start_time) % self.sol_duration
        phase    = elapsed / self.sol_duration          # 0→1 over one sol
        # Peak at noon (phase=0.5), zero at dawn/dusk
        elev_sin = float(np.clip(np.sin(np.pi * phase), 0, 1))
        return elev_sin

    # ── Dust opacity from frame ───────────────────────────────────────────────
    def _dust_factor(self, frame: np.ndarray) -> float:
        """
        Low contrast + washed-out sky → dust storm → lower solar input.
        Returns factor ∈ [0.3, 1.0].
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Top 20% of frame = sky region
        sky  = gray[:int(frame.shape[0] * 0.2), :]
        std  = sky.std()
        # Low std = uniform hazy sky = dust
        dust_opacity = float(np.clip(1.0 - std / 60.0, 0, 0.7))
        return float(np.clip(1.0 - dust_opacity, 0.3, 1.0))

    # ── Luminance-based irradiance proxy ─────────────────────────────────────
    def _luminance_factor(self, frame: np.ndarray) -> float:
        """
        Overall frame brightness as proxy for ambient irradiance.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lum  = gray.mean() / 255.0
        return float(np.clip(lum * 1.4, 0, 1))

    # ── Main estimate ─────────────────────────────────────────────────────────
    def estimate(self, frame: np.ndarray) -> Tuple[float, dict]:
        """
        Returns (charging_power_W, debug_info).
        """
        elev   = self._solar_elevation()
        dust   = self._dust_factor(frame)
        lum    = self._luminance_factor(frame)

        # Combined: P_solar = P_max × sin(elev) × dust_factor × luminance
        power  = MAX_PANEL_POWER_W * elev * dust * lum
        power  = float(np.clip(power, 0, MAX_PANEL_POWER_W))

        self._power_hist.append(power)

        info = {
            "solar_elevation_sin": round(elev, 3),
            "dust_factor":         round(dust, 3),
            "luminance_factor":    round(lum, 3),
            "charging_power_w":    round(power, 3),
            "avg_charging_w":      round(np.mean(self._power_hist), 3),
        }
        return power, info

    def draw_solar_indicator(self, frame: np.ndarray,
                             power: float, info: dict) -> np.ndarray:
        h, w = frame.shape[:2]
        x = w - 160
        y = h - 120

        # Panel
        pct = power / max(MAX_PANEL_POWER_W, 0.01)
        bar_h = 60
        fill  = int(bar_h * pct)
        col   = (0, 220, 255) if pct > 0.5 else (0, 140, 255) if pct > 0.2 else (60, 60, 180)

        cv2.rectangle(frame, (x, y), (x + 20, y + bar_h), (40, 40, 40), -1)
        cv2.rectangle(frame, (x, y + bar_h - fill), (x + 20, y + bar_h), col, -1)
        cv2.rectangle(frame, (x, y), (x + 20, y + bar_h), (120, 120, 120), 1)

        cv2.putText(frame, "SOLAR", (x + 24, y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, f"{power:.2f}W", (x + 24, y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
        cv2.putText(frame, f"dust={info['dust_factor']:.2f}", (x + 24, y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1)
        return frame
