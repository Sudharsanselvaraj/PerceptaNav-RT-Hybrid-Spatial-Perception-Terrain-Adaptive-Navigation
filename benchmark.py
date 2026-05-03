"""
Benchmark
Reproduces the comparison experiments from the paper (Tables 1, 2, 3).
Runs N episodes of both Conventional and Proposed approach.
Prints statistical summary matching paper metrics.

Conventional approach: selects nearest target regardless of scientific value.
Proposed approach:     selects target with highest Ui = Si / Ei.

Usage: python benchmark.py --episodes 50
"""

import random
import time
import argparse
import logging
import json
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from energy_model import EnergyModel
from decision_engine import DecisionEngine, scientific_score
from config import RoverConfig

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("Benchmark")

cfg = RoverConfig()
LABELS = ["rock", "soil", "sand", "gravel", "mineral", "obstacle", "dust"]
N_TARGETS_PER_EPISODE = 8


@dataclass
class Episode:
    method: str
    energy_used_wh:  float
    targets_visited: int
    total_si:        float
    efficiency:      float
    traversal_steps: int


def generate_candidates(n: int) -> List[Dict]:
    return [{
        "label":      random.choice(LABELS),
        "confidence": random.uniform(0.6, 1.0),
        "distance":   random.uniform(0.5, 8.0),
    } for _ in range(n)]


def run_conventional(candidates: List[Dict], em: EnergyModel) -> Optional[Dict]:
    """Selects nearest feasible target regardless of Si."""
    feasible = [c for c in candidates
                if em.is_feasible(em.predict_cost(c["distance"]))]
    if not feasible:
        return None
    return min(feasible, key=lambda c: c["distance"])


def run_proposed(candidates: List[Dict], de: DecisionEngine) -> Optional[Dict]:
    return de.select(candidates)


def run_episode(method: str, seed: int) -> Episode:
    random.seed(seed)
    em  = EnergyModel(total_wh=cfg.BATTERY_CAPACITY_WH)
    de  = DecisionEngine(em)

    total_si    = 0.0
    n_visited   = 0
    steps       = 0
    max_steps   = 150

    while steps < max_steps and em.remaining_pct() > 3.0:
        candidates = generate_candidates(random.randint(1, N_TARGETS_PER_EPISODE))

        if method == "conventional":
            selected = run_conventional(candidates, em)
            if selected:
                si = scientific_score(selected["label"]) * selected["confidence"]
                ei = em.predict_cost(selected["distance"])
                selected = {**selected,
                            "scientific_score": si,
                            "energy_cost_wh": ei,
                            "utility": si / max(ei, 1e-6)}
        else:
            selected = run_proposed(candidates, de)

        if selected:
            si = selected.get("scientific_score",
                              scientific_score(selected["label"]))
            ei = selected.get("energy_cost_wh",
                              em.predict_cost(selected.get("distance", 2.0)))
            em.update(power_watts=cfg.MOVE_POWER_W)
            time.sleep(0.0)  # simulate time
            em.used_wh += ei
            total_si   += si
            n_visited  += 1
        else:
            em.update(power_watts=cfg.IDLE_POWER_W)

        em.update(power_watts=cfg.IDLE_POWER_W)
        steps += 1

    used = em.used_wh
    eta  = total_si / max(used, 1e-6)

    return Episode(
        method=method,
        energy_used_wh=round(used, 4),
        targets_visited=n_visited,
        total_si=round(total_si, 4),
        efficiency=round(eta, 4),
        traversal_steps=steps,
    )


def run_benchmark(n_episodes: int = 30):
    print(f"\n{'═'*64}")
    print(f"  BENCHMARK  —  {n_episodes} episodes per method")
    print(f"{'═'*64}")

    results = {"conventional": [], "proposed": []}

    for ep in range(n_episodes):
        for method in ("conventional", "proposed"):
            r = run_episode(method, seed=ep * 7 + 13)
            results[method].append(r)

    def stats(episodes: List[Episode], key: str):
        vals = [getattr(e, key) for e in episodes]
        return (round(float(np.mean(vals)), 4),
                round(float(np.std(vals)),  4))

    metrics = ["energy_used_wh", "targets_visited", "total_si",
               "efficiency", "traversal_steps"]
    labels  = ["Energy Used (Wh)", "Targets Visited", "ΣSi",
                "Efficiency η", "Steps"]

    print(f"\n  {'Metric':<22} {'Conventional':>18} {'Proposed':>18} {'Δ%':>10}")
    print(f"  {'─'*22} {'─'*18} {'─'*18} {'─'*10}")

    summary = {}
    for metric, label in zip(metrics, labels):
        c_mean, c_std = stats(results["conventional"], metric)
        p_mean, p_std = stats(results["proposed"],     metric)
        delta = (p_mean - c_mean) / max(abs(c_mean), 1e-6) * 100
        sign  = "↑" if delta > 0 else "↓"
        print(f"  {label:<22} {c_mean:>9.4f} ±{c_std:<6.3f}  "
              f"{p_mean:>9.4f} ±{p_std:<6.3f}  {sign}{abs(delta):>7.1f}%")
        summary[metric] = {"conv": c_mean, "prop": p_mean, "delta_pct": round(delta, 2)}

    print(f"\n{'═'*64}\n")

    with open("logs/benchmark_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("  Results saved → logs/benchmark_results.json\n")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30)
    args = parser.parse_args()
    import os; os.makedirs("logs", exist_ok=True)
    run_benchmark(args.episodes)
