"""
Energy Model — Section 2.2 of paper
P(t) = V(t)·I(t)
E = ∫P(t)dt
Ei = Pmove·ti + Psample·tsample
Erem = Etotal − Eused
"""

import time
import numpy as np
from collections import deque
from config import RoverConfig

cfg = RoverConfig()


class EnergyModel:
    """
    Real-time energy tracking + predictive cost estimation per target.
    """

    def __init__(self, total_wh: float):
        self.total_wh  = total_wh          # Etotal
        self.used_wh   = 0.0               # Eused
        self._last_t   = time.time()
        self._history  = deque(maxlen=200) # rolling power window

    # ── Real-time tracking ─────────────────────────────────────────────────────
    def update(self, power_watts: float = None,
               voltage: float = None, current: float = None):
        """
        Call each frame. Pass either power_watts directly
        or (voltage, current) from INA219 sensor.
        """
        now = time.time()
        dt  = now - self._last_t
        self._last_t = now

        if power_watts is None and voltage is not None and current is not None:
            power_watts = voltage * current          # P(t) = V(t)·I(t)

        if power_watts is None:
            power_watts = cfg.IDLE_POWER_W

        # E += P·dt  (convert W·s → Wh)
        self.used_wh += power_watts * dt / 3600.0
        self._history.append(power_watts)

    def remaining_wh(self) -> float:
        return max(0.0, self.total_wh - self.used_wh)   # Erem

    def remaining_pct(self) -> float:
        return 100.0 * self.remaining_wh() / self.total_wh

    def avg_power(self) -> float:
        if not self._history:
            return cfg.IDLE_POWER_W
        return float(np.mean(self._history))

    def reset(self):
        self.used_wh = 0.0
        self._last_t = time.time()
        self._history.clear()

    # ── Predictive estimation ──────────────────────────────────────────────────
    def predict_cost(self, distance_m: float,
                     velocity_mps: float = None,
                     sample: bool = True) -> float:
        """
        Ei = Pmove·ti + Psample·tsample   (Eq. 7)
        """
        v  = velocity_mps or cfg.ROVER_SPEED_MPS
        ti = distance_m / max(v, 0.01)                  # traversal time (s)
        e_move   = cfg.MOVE_POWER_W   * ti / 3600.0     # Wh
        e_sample = (cfg.SAMPLE_POWER_W * cfg.SAMPLE_TIME_S / 3600.0
                    if sample else 0.0)
        return e_move + e_sample

    def is_feasible(self, cost_wh: float, safety_margin: float = 0.05) -> bool:
        """Ei < Erem with a safety buffer."""
        return cost_wh < self.remaining_wh() * (1 - safety_margin)

    # ── Status dict ───────────────────────────────────────────────────────────
    def status(self) -> dict:
        return {
            "total_wh":     round(self.total_wh, 3),
            "used_wh":      round(self.used_wh, 3),
            "remaining_wh": round(self.remaining_wh(), 3),
            "remaining_pct": round(self.remaining_pct(), 1),
            "avg_power_w":  round(self.avg_power(), 2),
        }

    def __repr__(self):
        s = self.status()
        return (f"EnergyModel(remaining={s['remaining_pct']}% | "
                f"used={s['used_wh']}Wh | avg={s['avg_power_w']}W)")
