"""
Rover State Machine
Finite State Machine (FSM) controlling high-level rover operating modes.
States: EXPLORE → APPROACH → SAMPLE → AVOID → LOW_POWER → SAFE_MODE
Transitions driven by energy level, obstacle detection, and target selection.
"""

import logging
import time
from enum import Enum, auto
from typing import Optional, Dict, Callable

log = logging.getLogger("RoverFSM")


class RoverState(Enum):
    EXPLORE   = auto()   # Default: roving, looking for targets
    APPROACH  = auto()   # Target selected, moving toward it
    SAMPLE    = auto()   # Close enough to sample (stationary analysis)
    AVOID     = auto()   # Obstacle blocking path, rerouting
    LOW_POWER = auto()   # Energy < 20%, conservative mode
    SAFE_MODE = auto()   # Energy < 5%, halt all non-essential ops


# Allowed transitions
TRANSITIONS: Dict[RoverState, list] = {
    RoverState.EXPLORE:   [RoverState.APPROACH, RoverState.AVOID,
                           RoverState.LOW_POWER, RoverState.SAFE_MODE],
    RoverState.APPROACH:  [RoverState.SAMPLE, RoverState.AVOID,
                           RoverState.EXPLORE, RoverState.LOW_POWER,
                           RoverState.SAFE_MODE],
    RoverState.SAMPLE:    [RoverState.EXPLORE, RoverState.LOW_POWER,
                           RoverState.SAFE_MODE],
    RoverState.AVOID:     [RoverState.EXPLORE, RoverState.APPROACH,
                           RoverState.LOW_POWER, RoverState.SAFE_MODE],
    RoverState.LOW_POWER: [RoverState.EXPLORE, RoverState.SAFE_MODE],
    RoverState.SAFE_MODE: [RoverState.LOW_POWER],   # Only charge out
}

# Power consumption multipliers per state
POWER_MULTIPLIER: Dict[RoverState, float] = {
    RoverState.EXPLORE:   1.0,
    RoverState.APPROACH:  1.4,
    RoverState.SAMPLE:    0.7,
    RoverState.AVOID:     1.6,
    RoverState.LOW_POWER: 0.4,
    RoverState.SAFE_MODE: 0.1,
}


class RoverFSM:
    """
    Controls rover operating mode via FSM.
    Drives energy_model power multiplier and decision_engine behaviour.
    """

    def __init__(self,
                 low_power_threshold: float = 20.0,
                 safe_mode_threshold: float = 5.0,
                 sample_distance_m:   float = 1.0,
                 sample_duration_s:   float = 10.0):
        self.state      = RoverState.EXPLORE
        self.prev_state = None
        self._low_pwr   = low_power_threshold
        self._safe_pwr  = safe_mode_threshold
        self._sample_d  = sample_distance_m
        self._sample_dur = sample_duration_s
        self._state_entry_time = time.time()
        self._history: list = []
        self._callbacks: Dict[RoverState, Callable] = {}

    # ── Transition ─────────────────────────────────────────────────────────────
    def transition(self, new_state: RoverState) -> bool:
        if new_state == self.state:
            return False
        if new_state not in TRANSITIONS.get(self.state, []):
            log.warning(f"Invalid transition: {self.state.name} → {new_state.name}")
            return False

        log.info(f"FSM: {self.state.name} → {new_state.name}")
        self.prev_state       = self.state
        self.state            = new_state
        self._state_entry_time = time.time()
        self._history.append({
            "from": self.prev_state.name,
            "to":   self.state.name,
            "ts":   time.time()
        })

        if new_state in self._callbacks:
            self._callbacks[new_state]()

        return True

    def on_enter(self, state: RoverState, fn: Callable):
        """Register callback for state entry."""
        self._callbacks[state] = fn

    # ── Auto-update logic ──────────────────────────────────────────────────────
    def update(self,
               energy_pct:       float,
               has_target:       bool,
               target_distance:  float = 99.0,
               obstacle_blocked: bool  = False) -> RoverState:
        """
        Call every frame. Automatically drives transitions based on inputs.
        """
        # ── Safety overrides (highest priority) ──
        if energy_pct <= self._safe_pwr:
            self.transition(RoverState.SAFE_MODE)
            return self.state

        if energy_pct <= self._low_pwr and self.state not in (
                RoverState.LOW_POWER, RoverState.SAFE_MODE):
            self.transition(RoverState.LOW_POWER)
            return self.state

        if self.state == RoverState.SAFE_MODE and energy_pct > self._safe_pwr + 2:
            self.transition(RoverState.LOW_POWER)
            return self.state

        if self.state == RoverState.LOW_POWER and energy_pct > self._low_pwr + 5:
            self.transition(RoverState.EXPLORE)
            return self.state

        # ── Obstacle handling ──
        if obstacle_blocked and self.state in (RoverState.EXPLORE,
                                               RoverState.APPROACH):
            self.transition(RoverState.AVOID)
            return self.state

        if self.state == RoverState.AVOID and not obstacle_blocked:
            self.transition(RoverState.EXPLORE)
            return self.state

        # ── Target approach / sampling ──
        if self.state == RoverState.EXPLORE and has_target:
            self.transition(RoverState.APPROACH)
            return self.state

        if self.state == RoverState.APPROACH:
            if not has_target:
                self.transition(RoverState.EXPLORE)
            elif target_distance < self._sample_d:
                self.transition(RoverState.SAMPLE)
            return self.state

        if self.state == RoverState.SAMPLE:
            elapsed = time.time() - self._state_entry_time
            if elapsed > self._sample_dur:
                log.info("Sampling complete.")
                self.transition(RoverState.EXPLORE)
            return self.state

        return self.state

    # ── Power multiplier ───────────────────────────────────────────────────────
    def power_multiplier(self) -> float:
        return POWER_MULTIPLIER.get(self.state, 1.0)

    # ── Status ────────────────────────────────────────────────────────────────
    def status(self) -> Dict:
        return {
            "state":           self.state.name,
            "power_mult":      self.power_multiplier(),
            "time_in_state_s": round(time.time() - self._state_entry_time, 1),
            "transitions":     len(self._history),
        }

    def draw_state(self, frame, x: int = 10, y: int = 70):
        import cv2
        STATE_COLORS = {
            RoverState.EXPLORE:   (0, 200, 80),
            RoverState.APPROACH:  (0, 200, 255),
            RoverState.SAMPLE:    (255, 200, 0),
            RoverState.AVOID:     (0, 100, 255),
            RoverState.LOW_POWER: (0, 165, 255),
            RoverState.SAFE_MODE: (0, 0, 220),
        }
        color = STATE_COLORS.get(self.state, (200, 200, 200))
        label = f"MODE: {self.state.name}"
        cv2.putText(frame, label, (x, y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, color, 2)
        return frame
