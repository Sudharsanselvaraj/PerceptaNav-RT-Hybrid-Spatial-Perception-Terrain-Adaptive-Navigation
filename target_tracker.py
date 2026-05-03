"""
Target Tracker
Maintains a session history of explored targets and their scientific scores.
"""

import time
import json
import logging
from collections import defaultdict
from typing import Optional, Dict, List

log = logging.getLogger("TargetTracker")


class TargetTracker:
    """
    Tracks targets across frames.
    Accumulates scientific scores for mission summary (∑Si).
    Suppresses re-selection of recently visited targets.
    """

    def __init__(self, cooldown_s: float = 10.0):
        self.cooldown_s  = cooldown_s
        self._visited: Dict[str, float] = {}   # label → last_visited_ts
        self._scores:  Dict[str, List]  = defaultdict(list)
        self._total_si  = 0.0
        self._total_ei  = 0.0
        self._count     = 0

    def is_on_cooldown(self, label: str) -> bool:
        last = self._visited.get(label, 0)
        return (time.time() - last) < self.cooldown_s

    def update(self, target: Dict):
        label = target.get("label", "unknown")
        si    = target.get("scientific_score", 0.0)
        ei    = target.get("energy_cost_wh", 0.0)

        self._visited[label] = time.time()
        self._scores[label].append(si)
        self._total_si += si
        self._total_ei += ei
        self._count    += 1

        log.debug(f"Tracked: {label} si={si:.3f} | ΣSi={self._total_si:.2f}")

    def efficiency(self) -> float:
        """η = ΣSi / Eused"""
        if self._total_ei < 1e-6:
            return 0.0
        return self._total_si / self._total_ei

    def summary(self) -> Dict:
        return {
            "targets_visited": self._count,
            "unique_labels":   list(self._scores.keys()),
            "total_si":        round(self._total_si, 4),
            "total_ei_wh":     round(self._total_ei, 4),
            "efficiency_eta":  round(self.efficiency(), 4),
            "breakdown":       {k: {"count": len(v), "avg_si": round(sum(v)/len(v), 3)}
                                for k, v in self._scores.items()},
        }

    def save(self, path: str = "logs/tracker_summary.json"):
        with open(path, "w") as f:
            json.dump(self.summary(), f, indent=2)
        log.info(f"Tracker summary saved → {path}")

    def __repr__(self):
        s = self.summary()
        return (f"TargetTracker(visited={s['targets_visited']} "
                f"ΣSi={s['total_si']} η={s['efficiency_eta']})")
