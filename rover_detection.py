import cv2
import numpy as np
from ultralytics import YOLO

# -----------------------------
# ESP32 IP
# -----------------------------
ESP32_IP = "172.20.10.2"

# -----------------------------
# LOAD YOLO MODEL (MULTI-CLASS)
# -----------------------------
model = YOLO("yolov8n.pt")  # detects ~80 classes

# -----------------------------
# STREAM URL
# -----------------------------
stream_url = f"http://{ESP32_IP}:81/stream"

cap = cv2.VideoCapture(stream_url)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

WINDOW_WIDTH = 960
WINDOW_HEIGHT = 720

# -----------------------------
# 3D MESH FUNCTION
# -----------------------------
def draw_3d_mesh(frame):

    h, w = frame.shape[:2]
    horizon = int(h * 0.6)

    color = (0, 255, 0)

    for i in range(1, 15):
        y = int(horizon + (i ** 1.5) * 10)
        if y >= h:
            break
        thickness = max(1, int(i / 4))
        cv2.line(frame, (0, y), (w, y), color, thickness)

    center = w // 2
    for i in range(-8, 9):
        x_top = center + i * 25
        x_bottom = center + i * 70
        cv2.line(frame, (x_top, horizon), (x_bottom, h), color, 1)

    return frame

# -----------------------------
# WINDOW
# -----------------------------
cv2.namedWindow("Rover Navigation", cv2.WINDOW_NORMAL)

# -----------------------------
# MAIN LOOP
# -----------------------------
while True:

    ret, frame = cap.read()

    if not ret:
        print("Reconnecting...")
        cap.release()
        cap = cv2.VideoCapture(stream_url)
        continue

    frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))

    # -------------------------
    # YOLO DETECTION
    # -------------------------
    results = model(frame)

    annotated = frame.copy()

    person_detected = False

    for box in results[0].boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = model.names[cls]

        text = f"{label} {conf:.2f}"

        # Highlight humans differently
        if label == "person":
            color = (0, 0, 255)   # RED for human
            person_detected = True
        else:
            color = (255, 0, 0)   # BLUE for others

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    # -------------------------
    # MESH (only if safe)
    # -------------------------
    if not person_detected:
        annotated = draw_3d_mesh(annotated)

    # -------------------------
    # DISPLAY
    # -------------------------
    cv2.imshow("Rover Navigation", annotated)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()