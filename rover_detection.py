"""
Rover Detection & Navigation System
Integrates: YOLOv8 detection + classification + energy-aware target selection
Paper: Adaptive Target Selection in Energy-Limited Rover Systems
"""

import cv2
import numpy as np
import time
import logging
import json
from datetime import datetime
from ultralytics import YOLO
from energy_model import EnergyModel
from decision_engine import DecisionEngine
from target_tracker import TargetTracker
from config import RoverConfig

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/rover.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("RoverDetection")

# ── Models ─────────────────────────────────────────────────────────────────────
cfg = RoverConfig()

detector   = YOLO(cfg.DETECTOR_MODEL)
classifier = YOLO(cfg.CLASSIFIER_MODEL)

energy_model   = EnergyModel(cfg.BATTERY_CAPACITY_WH)
decision_engine = DecisionEngine(energy_model)
tracker        = TargetTracker()

# ── Stream ─────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(cfg.STREAM_URL, cv2.CAP_FFMPEG)
if not cap.isOpened():
    log.warning("Primary stream failed, retrying...")
    cap = cv2.VideoCapture(cfg.STREAM_URL)

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# ── Helpers ────────────────────────────────────────────────────────────────────
def draw_3d_mesh(frame, color=(0, 255, 0)):
    h, w = frame.shape[:2]
    horizon = int(h * 0.6)
    for i in range(1, 15):
        y = int(horizon + (i ** 1.5) * 10)
        if y >= h:
            break
        cv2.line(frame, (0, y), (w, y), color, max(1, int(i / 4)))
    center = w // 2
    for i in range(-8, 9):
        cv2.line(frame,
                 (center + i * 25, horizon),
                 (center + i * 70, h),
                 color, 1)
    return frame


def draw_hud(frame, label, confidence, energy_pct, selected_target, fps):
    h, w = frame.shape[:2]

    # ── Top bar ──
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # ── Classification label ──
    color = (0, 0, 255) if label == "rock" else (0, 255, 80)
    cv2.putText(frame, f"TERRAIN: {label.upper()}  {confidence:.2f}",
                (15, 38), cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2)

    # ── FPS ──
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 130, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

    # ── Energy bar ──
    bar_x, bar_y, bar_w, bar_h = 15, h - 50, 200, 18
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
    fill = int(bar_w * energy_pct / 100)
    bar_color = (0, 255, 80) if energy_pct > 40 else (0, 165, 255) if energy_pct > 20 else (0, 0, 255)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), bar_color, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (120, 120, 120), 1)
    cv2.putText(frame, f"ENERGY {energy_pct:.1f}%", (bar_x, bar_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # ── Selected target ──
    if selected_target:
        t = selected_target
        info = (f"TARGET: {t['label'].upper()}  "
                f"Si={t['scientific_score']:.2f}  "
                f"Ui={t['utility']:.3f}")
        cv2.putText(frame, info, (15, h - 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)

    return frame


def draw_target_box(frame, label, color):
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    bw, bh = int(w * 0.4), int(h * 0.4)
    x1, y1 = cx - bw // 2, cy - bh // 2
    x2, y2 = cx + bw // 2, cy + bh // 2

    # Corner brackets
    L = 20
    for px, py, dx, dy in [(x1, y1, 1, 1), (x2, y1, -1, 1),
                             (x1, y2, 1, -1), (x2, y2, -1, -1)]:
        cv2.line(frame, (px, py), (px + dx * L, py), color, 3)
        cv2.line(frame, (px, py), (px, py + dy * L), color, 3)

    cv2.putText(frame, f"[TARGET: {label.upper()}]",
                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


# ── Main loop ──────────────────────────────────────────────────────────────────
cv2.namedWindow("Rover Navigation", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Rover Navigation", cfg.WINDOW_WIDTH, cfg.WINDOW_HEIGHT)

frame_count = 0
fps = 0.0
t0 = time.time()
mission_log = []

log.info("Rover navigation started.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            log.warning("Stream lost. Reconnecting...")
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(cfg.STREAM_URL, cv2.CAP_FFMPEG)
            continue

        frame = cv2.resize(frame, (cfg.WINDOW_WIDTH, cfg.WINDOW_HEIGHT))
        frame_count += 1

        # ── FPS ──
        if frame_count % 10 == 0:
            fps = 10 / (time.time() - t0)
            t0 = time.time()

        # ── Detection ──
        det_results = detector(frame, verbose=False)
        annotated = det_results[0].plot()

        # ── Classification ──
        cls_results = classifier(frame, verbose=False)
        label      = cls_results[0].names[cls_results[0].probs.top1]
        confidence = float(cls_results[0].probs.top1conf)

        # ── Energy update (simulate INA219 reading) ──
        energy_model.update(power_watts=cfg.IDLE_POWER_W)
        energy_pct = energy_model.remaining_pct()

        # ── Candidate targets from detections ──
        boxes = det_results[0].boxes
        candidates = []
        if boxes is not None and len(boxes):
            for box in boxes:
                cls_id = int(box.cls[0])
                obj_label = det_results[0].names[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                # Estimate distance from bounding box size (proxy)
                box_area = (x2 - x1) * (y2 - y1)
                frame_area = cfg.WINDOW_WIDTH * cfg.WINDOW_HEIGHT
                dist_est = cfg.MAX_RANGE_M * (1 - box_area / frame_area) + 0.5
                candidates.append({
                    "label":      obj_label,
                    "confidence": conf,
                    "distance":   dist_est,
                    "cx": cx, "cy": cy,
                    "box": (x1, y1, x2, y2)
                })

        # ── Decision engine ──
        selected = decision_engine.select(candidates)

        # ── Track & log ──
        if selected:
            tracker.update(selected)
            mission_log.append({
                "ts":     datetime.now().isoformat(),
                "target": selected["label"],
                "utility": selected["utility"],
                "energy_pct": energy_pct
            })

        # ── Draw ──
        if label == "rock":
            annotated = draw_target_box(annotated, label, (0, 0, 255))
        else:
            annotated = draw_3d_mesh(annotated)

        # Highlight selected candidate
        if selected and "box" in selected:
            x1, y1, x2, y2 = [int(v) for v in selected["box"]]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 255), 2)
            cv2.putText(annotated, f"OPTIMAL Ui={selected['utility']:.3f}",
                        (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)

        annotated = draw_hud(annotated, label, confidence,
                             energy_pct, selected, fps)

        cv2.imshow("Rover Navigation", annotated)

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break
        elif key == ord('s'):
            fname = f"logs/snapshot_{frame_count}.jpg"
            cv2.imwrite(fname, annotated)
            log.info(f"Snapshot saved: {fname}")
        elif key == ord('r'):
            energy_model.reset()
            log.info("Energy model reset.")

finally:
    cap.release()
    cv2.destroyAllWindows()

    # Save mission log
    with open("logs/mission_log.json", "w") as f:
        json.dump(mission_log, f, indent=2)
    log.info(f"Mission ended. {len(mission_log)} events logged.")
