"""
Decision Engine — Section 2.3 of paper
Ui = Si / Ei
i* = argmax(Ui)  subject to Ei < Erem
"""

import logging
from typing import List, Optional, Dict
from energy_model import EnergyModel
from config import RoverConfig

log = logging.getLogger("DecisionEngine")
cfg = RoverConfig()


# ── Scientific scoring ─────────────────────────────────────────────────────────
SCIENCE_SCORES: Dict[str, float] = {
    # High value
    "rock":       0.90,
    "stone":      0.85,
    "mineral":    0.95,
    "crystal":    1.00,
    # Medium
    "soil":       0.50,
    "sand":       0.40,
    "gravel":     0.55,
    "dust":       0.30,
    # Low / navigate around
    "person":     0.05,
    "obstacle":   0.02,
    "default":    0.20,
}

def scientific_score(label: str) -> float:
    return SCIENCE_SCORES.get(label.lower(), SCIENCE_SCORES["default"])


class DecisionEngine:
    """
    Utility-driven autonomous target selection.
    Implements Eq. 4, 9, 10, 11 from the paper.
    """

    def __init__(self, energy_model: EnergyModel):
        self.em = energy_model
        self._last_selected: Optional[Dict] = None
        self._low_power_mode = False

    def select(self, candidates: List[Dict]) -> Optional[Dict]:
        """
        Evaluate candidates, return target with highest feasible utility.
        Returns None if no feasible target or energy critically low.
        """
        erem = self.em.remaining_wh()

        # ── Low-power fallback ─────────────────────────────────────────────────
        if erem < cfg.CRITICAL_ENERGY_WH:
            if not self._low_power_mode:
                log.warning(f"CRITICAL ENERGY {erem:.3f}Wh — entering low-power mode")
                self._low_power_mode = True
            return None

        self._low_power_mode = False

        if not candidates:
            return None

        best = None
        best_utility = -1.0

        for c in candidates:
            dist     = c.get("distance", cfg.MAX_RANGE_M / 2)
            label    = c.get("label", "default")
            conf     = c.get("confidence", 1.0)

            # Scientific score Si — weighted by detection confidence
            si = scientific_score(label) * conf

            # Energy cost Ei = Pmove·ti + Psample·tsample
            ei = self.em.predict_cost(dist)

            # Feasibility check: Ei < Erem
            if not self.em.is_feasible(ei):
                log.debug(f"Skip {label} at {dist:.1f}m — infeasible cost {ei:.4f}Wh")
                continue

            # Utility Ui = Si / Ei
            ui = si / max(ei, 1e-6)

            if ui > best_utility:
                best_utility = ui
                best = {**c,
                        "scientific_score": round(si, 4),
                        "energy_cost_wh":   round(ei, 4),
                        "utility":          round(ui, 4)}

        if best and best != self._last_selected:
            log.info(f"New optimal target: {best['label']} "
                     f"Ui={best['utility']} dist={best.get('distance',0):.1f}m")
        self._last_selected = best
        return best

    def status(self) -> Dict:
        return {
            "low_power_mode": self._low_power_mode,
            "last_target":    self._last_selected,
            "energy":         self.em.status(),
        }
