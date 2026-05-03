# Adaptive Target Selection — Energy-Limited Rover System

> Full implementation of: *Adaptive Target Selection in Energy-Limited Rover Systems*

## Files (19 Python modules)

| File | Role |
|------|------|
| `main.py` | ★ Full integration — run this |
| `rover_detection.py` | Standalone detection loop |
| `trainmodel.py` | Train YOLOv8 classifier |
| `simulate.py` | Offline simulation (no camera) |
| `benchmark.py` | Reproduce paper Tables 1–3 |
| `dashboard.py` | Post-mission analytics |
| `prepare_dataset.py` | Dataset train/val split |
| `config.py` | All parameters |
| `energy_model.py` | P=VI, Ei, Erem — §2.2 |
| `decision_engine.py` | Ui=Si/Ei, argmax — §2.3 |
| `science_scorer.py` | Multi-factor Si — §2.1 |
| `distance_estimator.py` | Monocular distance |
| `terrain_classifier.py` | Traversability grid |
| `obstacle_detector.py` | Optical flow obstacles |
| `path_planner.py` | A* path planning |
| `solar_estimator.py` | Solar charging from frame |
| `rover_fsm.py` | EXPLORE→APPROACH→SAMPLE FSM |
| `map_builder.py` | 2D occupancy minimap |
| `target_tracker.py` | ΣSi, η tracker |
| `telemetry_logger.py` | CSV + JSON telemetry |

## Quick Start

```bash
pip install -r requirements.txt
python prepare_dataset.py --source raw_images
python trainmodel.py
python main.py --stream http://10.15.140.40:81/stream
python benchmark.py --episodes 50
```

## Keys: ESC=quit | S=snapshot | R=reset energy | M=save map
