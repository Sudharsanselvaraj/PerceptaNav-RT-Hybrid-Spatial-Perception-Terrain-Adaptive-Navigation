"""
Mission Analytics Dashboard
Reads mission_log.json → prints summary table + efficiency metrics from paper.
Run after a mission: python dashboard.py
"""

import json
import sys
import os
from pathlib import Path
from collections import defaultdict
from datetime import datetime

LOG_FILE = "logs/mission_log.json"


def load_log(path: str):
    if not Path(path).exists():
        print(f"[ERROR] Log not found: {path}")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def print_separator(char="─", width=60):
    print(char * width)


def analyze(events):
    if not events:
        print("No events in log.")
        return

    total_si    = sum(e.get("utility", 0) for e in events)
    n_targets   = len(events)
    labels      = defaultdict(int)
    energy_drop = events[0]["energy_pct"] - events[-1]["energy_pct"]
    energy_used_pct = energy_drop

    for e in events:
        labels[e["target"]] += 1

    # Efficiency η = ΣSi / Eused  (normalized by %)
    efficiency  = total_si / max(energy_used_pct, 0.01)

    ts_start = datetime.fromisoformat(events[0]["ts"])
    ts_end   = datetime.fromisoformat(events[-1]["ts"])
    duration = (ts_end - ts_start).total_seconds()

    print()
    print_separator("═")
    print("  MISSION ANALYTICS REPORT")
    print_separator("═")
    print(f"  Start:       {events[0]['ts']}")
    print(f"  End:         {events[-1]['ts']}")
    print(f"  Duration:    {duration:.1f}s  ({duration/60:.1f} min)")
    print_separator()
    print(f"  Targets visited:    {n_targets}")
    print(f"  Unique types:       {len(labels)}")
    print(f"  ΣUi (proxy ΣSi):    {total_si:.4f}")
    print(f"  Energy used:        {energy_used_pct:.1f}%")
    print(f"  Efficiency η:       {efficiency:.4f}")
    print_separator()
    print("  TARGET BREAKDOWN")
    print_separator()
    for label, count in sorted(labels.items(), key=lambda x: -x[1]):
        bar = "█" * count
        print(f"  {label:<15} {count:>3}  {bar}")
    print_separator()

    # Compare vs conventional baseline (from paper: η=0.35)
    baseline_eta = 0.35
    if efficiency > 0:
        delta = (efficiency - baseline_eta) / baseline_eta * 100
        print(f"  vs Conventional (η=0.35):  Δη = {delta:+.1f}%")
    print_separator("═")
    print()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else LOG_FILE
    events = load_log(path)
    analyze(events)
