"""
Rover Simulator
Runs the full decision pipeline without a physical camera.
Generates synthetic detections → tests energy model + decision engine.
Usage: python simulate.py
"""

import random
import time
import json
import logging
from energy_model import EnergyModel
from decision_engine import DecisionEngine
from target_tracker import TargetTracker
from config import RoverConfig

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("Simulator")

cfg     = RoverConfig()
em      = EnergyModel(total_wh=cfg.BATTERY_CAPACITY_WH)
de      = DecisionEngine(em)
tracker = TargetTracker()

LABELS = ["rock", "soil", "sand", "gravel", "mineral", "obstacle"]
N_STEPS = 200


def synthetic_detections():
    n = random.randint(0, 4)
    return [
        {
            "label":      random.choice(LABELS),
            "confidence": random.uniform(0.5, 1.0),
            "distance":   random.uniform(0.5, cfg.MAX_RANGE_M),
        }
        for _ in range(n)
    ]


def run():
    log.info("═══ Rover Simulator ═══")
    log.info(f"Battery: {cfg.BATTERY_CAPACITY_WH}Wh | Steps: {N_STEPS}")
    results = []

    for step in range(N_STEPS):
        # Simulate power draw
        power = cfg.IDLE_POWER_W + random.uniform(-0.5, 1.0)
        em.update(power_watts=power)

        candidates = synthetic_detections()
        selected   = de.select(candidates)

        if selected:
            tracker.update(selected)

        if step % 20 == 0:
            status = em.status()
            log.info(f"Step {step:>3} | {status['remaining_pct']:.1f}% | "
                     f"target={selected['label'] if selected else 'None'}")
            results.append({
                "step": step,
                "energy_pct": status["remaining_pct"],
                "target": selected["label"] if selected else None,
                "utility": selected["utility"] if selected else 0,
            })

        time.sleep(0.05)

    # ── Summary ──
    summary = tracker.summary()
    log.info("═══ Simulation Complete ═══")
    log.info(f"Targets visited:  {summary['targets_visited']}")
    log.info(f"ΣSi:              {summary['total_si']}")
    log.info(f"Efficiency η:     {summary['efficiency_eta']}")

    with open("logs/sim_results.json", "w") as f:
        json.dump({"summary": summary, "steps": results}, f, indent=2)
    log.info("Results → logs/sim_results.json")


if __name__ == "__main__":
    run()
