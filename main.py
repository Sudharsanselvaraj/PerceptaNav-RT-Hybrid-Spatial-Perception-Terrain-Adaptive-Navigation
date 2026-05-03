"""
Rover Main — Full Integration
Ties together ALL modules:
  rover_detection + terrain_classifier + obstacle_detector + path_planner
  + science_scorer + distance_estimator + solar_estimator
  + rover_fsm + map_builder + telemetry_logger

Run this for the complete system.
Usage: python main.py [--stream URL] [--no-map] [--headless]
"""

import cv2
import time
import argparse
import logging
from datetime import datetime

from ultralytics import YOLO

from config           import RoverConfig
from energy_model     import EnergyModel
from decision_engine  import DecisionEngine
from target_tracker   import TargetTracker
from terrain_classifier import TerrainClassifier
from obstacle_detector  import ObstacleDetector
from path_planner       import PathPlanner
from science_scorer     import ScienceScorer
from distance_estimator import DistanceEstimator
from solar_estimator    import SolarEstimator
from rover_fsm          import RoverFSM, RoverState
from map_builder        import MapBuilder
from telemetry_logger   import TelemetryLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.FileHandler("logs/main.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("Main")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stream",   default=None)
    p.add_argument("--no-map",   action="store_true")
    p.add_argument("--headless", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = RoverConfig()
    if args.stream:
        cfg.STREAM_URL = args.stream

    log.info("═══ Rover System Initialising ═══")

    # ── Models ────────────────────────────────────────────────────────────────
    detector   = YOLO(cfg.DETECTOR_MODEL)
    classifier = YOLO(cfg.CLASSIFIER_MODEL)

    # ── Modules ───────────────────────────────────────────────────────────────
    em       = EnergyModel(cfg.BATTERY_CAPACITY_WH)
    de       = DecisionEngine(em)
    tracker  = TargetTracker()
    terrain  = TerrainClassifier()
    obstacle = ObstacleDetector(cfg.WINDOW_WIDTH, cfg.WINDOW_HEIGHT)
    planner  = PathPlanner()
    scorer   = ScienceScorer()
    dist_est = DistanceEstimator()
    solar    = SolarEstimator()
    fsm      = RoverFSM()
    mapper   = MapBuilder() if not args.no_map else None
    telem    = TelemetryLogger()

    # ── FSM callbacks ─────────────────────────────────────────────────────────
    fsm.on_enter(RoverState.SAMPLE,    lambda: log.info(">>> SAMPLING TARGET"))
    fsm.on_enter(RoverState.SAFE_MODE, lambda: log.warning(">>> SAFE MODE ACTIVE"))

    # ── Stream ────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(cfg.STREAM_URL, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cfg.STREAM_URL)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not args.headless:
        cv2.namedWindow("Rover — Full System", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Rover — Full System", cfg.WINDOW_WIDTH, cfg.WINDOW_HEIGHT)

    frame_id = 0
    fps, t0  = 0.0, time.time()
    cumulative_si = 0.0

    log.info("Rover running. ESC=quit | S=snapshot | R=reset energy | M=save map")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                log.warning("Stream lost — reconnecting")
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(cfg.STREAM_URL, cv2.CAP_FFMPEG)
                continue

            frame    = cv2.resize(frame, (cfg.WINDOW_WIDTH, cfg.WINDOW_HEIGHT))
            frame_id += 1

            if frame_id % 10 == 0:
                fps = 10 / max(time.time() - t0, 0.001)
                t0  = time.time()

            # ── Solar charging ────────────────────────────────────────────────
            solar_w, solar_info = solar.estimate(frame)

            # ── Energy update ─────────────────────────────────────────────────
            base_power = cfg.IDLE_POWER_W * fsm.power_multiplier()
            net_power  = max(0, base_power - solar_w)
            em.update(power_watts=net_power)

            # ── Detection ─────────────────────────────────────────────────────
            det_results = detector(frame, verbose=False)
            annotated   = det_results[0].plot()

            # ── Classification ────────────────────────────────────────────────
            cls_results = classifier(frame, verbose=False)
            cls_label   = cls_results[0].names[cls_results[0].probs.top1]
            cls_conf    = float(cls_results[0].probs.top1conf)

            # ── Terrain ───────────────────────────────────────────────────────
            terrain_grid = terrain.analyse(frame)
            safe_col     = terrain.safest_corridor(terrain_grid)
            annotated    = terrain.draw_grid(annotated, terrain_grid)

            # ── Obstacles ────────────────────────────────────────────────────
            obstacles    = obstacle.detect(frame)
            annotated    = obstacle.draw(annotated, obstacles)
            obs_cmd      = obstacle.avoidance_command(obstacles)
            blocked      = obs_cmd in ("LEFT", "RIGHT", "STOP")

            # ── Candidate targets with scored Si ─────────────────────────────
            boxes      = det_results[0].boxes
            candidates = []
            if boxes is not None:
                for box in boxes:
                    cls_id    = int(box.cls[0])
                    lbl       = det_results[0].names[cls_id]
                    conf      = float(box.conf[0])
                    xyxy      = box.xyxy[0].tolist()
                    d_est     = dist_est.estimate(lbl, tuple(xyxy))
                    si        = scorer.score_from_box(frame, tuple(xyxy), lbl, conf)
                    candidates.append({
                        "label":            lbl,
                        "confidence":       conf,
                        "distance":         d_est.distance_m,
                        "scientific_score": si,
                        "box":              xyxy,
                    })
                    annotated = dist_est.draw(annotated, d_est, tuple(xyxy))

            # ── Decision ──────────────────────────────────────────────────────
            selected = de.select(candidates)

            # ── Path planning ─────────────────────────────────────────────────
            steer_cmd = obs_cmd
            if selected and not blocked:
                target_dist = selected.get("distance", 5.0)
                path = planner.plan(terrain_grid, safe_col)
                if path:
                    annotated = planner.draw_path(annotated, terrain_grid, path)
                    steer_cmd = planner.steering_command(path)

            # ── FSM update ────────────────────────────────────────────────────
            e_pct = em.remaining_pct()
            t_dist = selected.get("distance", 99.0) if selected else 99.0
            fsm.update(e_pct, selected is not None, t_dist, blocked)
            annotated = fsm.draw_state(annotated)

            # ── Tracking ─────────────────────────────────────────────────────
            if selected:
                tracker.update(selected)
                cumulative_si += selected.get("scientific_score", 0)
                if mapper:
                    mapper.mark_target(selected.get("distance", 2.0),
                                       label=selected["label"])

            if obstacles and mapper:
                mapper.mark_obstacle(min(o.threat_level * 3 for o in obstacles))
            if mapper:
                mapper.update_rover(dy=0.02)

            # ── Solar indicator ───────────────────────────────────────────────
            annotated = solar.draw_solar_indicator(annotated, solar_w, solar_info)

            # ── Minimap ───────────────────────────────────────────────────────
            if mapper:
                annotated = mapper.overlay_minimap(annotated)

            # ── Telemetry ─────────────────────────────────────────────────────
            telem.log(
                energy_pct       = round(e_pct, 2),
                used_wh          = round(em.used_wh, 3),
                remaining_wh     = round(em.remaining_wh(), 3),
                avg_power_w      = round(em.avg_power(), 2),
                solar_w          = round(solar_w, 3),
                terrain_label    = cls_label,
                terrain_conf     = round(cls_conf, 3),
                n_candidates     = len(candidates),
                selected_target  = selected["label"] if selected else "",
                selected_utility = selected.get("utility", 0) if selected else 0,
                selected_si      = selected.get("scientific_score", 0) if selected else 0,
                steering_cmd     = steer_cmd,
                obstacle_cmd     = obs_cmd,
                safest_corridor  = safe_col,
                cumulative_si    = round(cumulative_si, 4),
                efficiency_eta   = round(tracker.efficiency(), 4),
            )

            # ── Display ───────────────────────────────────────────────────────
            if not args.headless:
                # FPS overlay
                cv2.putText(annotated, f"FPS:{fps:.1f}", (5, 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
                cv2.imshow("Rover — Full System", annotated)

                key = cv2.waitKey(1)
                if key == 27:
                    break
                elif key == ord("s"):
                    cv2.imwrite(f"logs/snap_{frame_id}.jpg", annotated)
                elif key == ord("r"):
                    em.reset()
                    log.info("Energy reset.")
                elif key == ord("m") and mapper:
                    mapper.save()
                    log.info("Map saved.")

    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()
        telem.close()
        tracker.save()
        if mapper:
            mapper.save()
        log.info("Rover shutdown complete.")
        log.info(f"Final: ΣSi={cumulative_si:.3f}  η={tracker.efficiency():.4f}")


if __name__ == "__main__":
    main()
