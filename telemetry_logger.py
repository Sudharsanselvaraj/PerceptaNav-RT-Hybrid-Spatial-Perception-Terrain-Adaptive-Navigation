"""
Telemetry Logger
Records all rover state per frame to CSV and JSON.
Mimics actual planetary rover downlink telemetry format.
"""

import csv
import json
import time
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

log = logging.getLogger("TelemetryLogger")

TELEMETRY_FIELDS = [
    "timestamp", "frame_id", "sol", "local_time",
    # Energy
    "energy_pct", "used_wh", "remaining_wh", "avg_power_w", "solar_w",
    # Detection
    "terrain_label", "terrain_conf",
    "n_candidates", "selected_target", "selected_utility", "selected_si",
    # Navigation
    "steering_cmd", "obstacle_cmd", "safest_corridor",
    # Efficiency
    "cumulative_si", "efficiency_eta",
]


class TelemetryLogger:
    """
    Writes one row per frame to CSV.
    Also maintains a rolling JSON buffer for real-time dashboard use.
    """

    def __init__(self, log_dir: str = "logs",
                 session_id: str = None,
                 buffer_size: int = 500):
        self.log_dir    = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.buffer_size = buffer_size

        self._csv_path  = self.log_dir / f"telemetry_{self.session_id}.csv"
        self._json_path = self.log_dir / f"telemetry_{self.session_id}.json"

        self._buffer: List[Dict] = []
        self._frame_id = 0
        self._start_ts = time.time()
        self._csv_file = open(self._csv_path, "w", newline="")
        self._writer   = csv.DictWriter(self._csv_file, fieldnames=TELEMETRY_FIELDS)
        self._writer.writeheader()

        log.info(f"Telemetry → {self._csv_path}")

    def _sol_time(self) -> Dict[str, float]:
        """Simulated Martian sol number and local solar time."""
        elapsed = time.time() - self._start_ts
        SOL_S   = 88775.0
        sol     = elapsed / SOL_S
        local   = (sol % 1.0) * 24.0   # local time in hours
        return {"sol": round(sol, 6), "local_time": round(local, 4)}

    def log(self, **kwargs):
        """
        Log one frame of telemetry.
        Missing fields are filled with empty string.
        """
        self._frame_id += 1
        sol_info = self._sol_time()

        row = {f: "" for f in TELEMETRY_FIELDS}
        row.update({
            "timestamp":  datetime.now().isoformat(),
            "frame_id":   self._frame_id,
            "sol":        sol_info["sol"],
            "local_time": sol_info["local_time"],
        })
        row.update({k: v for k, v in kwargs.items() if k in TELEMETRY_FIELDS})

        self._writer.writerow(row)

        self._buffer.append(row)
        if len(self._buffer) > self.buffer_size:
            self._buffer.pop(0)

        # Flush every 30 frames
        if self._frame_id % 30 == 0:
            self._csv_file.flush()
            self._flush_json()

    def _flush_json(self):
        with open(self._json_path, "w") as f:
            json.dump(self._buffer[-100:], f, indent=2)

    def close(self):
        self._csv_file.flush()
        self._csv_file.close()
        self._flush_json()
        log.info(f"Telemetry closed. {self._frame_id} frames logged.")

    # ── Stats ─────────────────────────────────────────────────────────────────
    def session_stats(self) -> Dict[str, Any]:
        if not self._buffer:
            return {}
        energies = [float(r["energy_pct"]) for r in self._buffer
                    if r["energy_pct"] != ""]
        targets  = [r["selected_target"] for r in self._buffer
                    if r["selected_target"]]
        return {
            "session_id":    self.session_id,
            "total_frames":  self._frame_id,
            "min_energy_pct": round(min(energies), 2) if energies else None,
            "unique_targets": list(set(targets)),
            "target_events":  len(targets),
        }

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
