import cv2
import numpy as np
import requests
from ultralytics import YOLO

# -----------------------------
# ESP32 IP (IMPORTANT: NO http)
# -----------------------------
ESP32_IP = "172.20.10.2"

# -----------------------------
# TURN FLASH LIGHT ON
# -----------------------------
def turn_flash_on(ip):

    commands = [
        f"http://{ip}/control?var=flash&val=255",
        f"http://{ip}/control?var=led_intensity&val=255",
        f"http://{ip}/control?var=led&val=255",
        f"http://{ip}/control?var=led_gpio&val=255",
        f"http://{ip}/control?var=torch&val=1"
    ]

    for cmd in commands:
        try:
            r = requests.get(cmd, timeout=2)
            if r.status_code == 200:
                print("Flash turned ON using:", cmd)
                return
        except:
            pass

    print("Flash control not supported by firmware")

# Turn ON flash once
turn_flash_on(ESP32_IP)

# -----------------------------
# LOAD MODELS
# -----------------------------
detector = YOLO("yolov8n.pt")   # object detection
classifier = YOLO("runs/classify/train/weights/best.pt")  # rock classifier

# -----------------------------
# STREAM URL
# -----------------------------
stream_url = f"http://{ESP32_IP}:81/stream"

# Use default backend (more stable on Mac)
cap = cv2.VideoCapture(stream_url)

# Reduce buffer delay
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

    # horizontal lines
    for i in range(1, 15):
        y = int(horizon + (i ** 1.5) * 10)
        if y >= h:
            break
        thickness = max(1, int(i / 4))
        cv2.line(frame, (0, y), (w, y), color, thickness)

    # vertical perspective lines
    center = w // 2
    for i in range(-8, 9):
        x_top = center + i * 25
        x_bottom = center + i * 70
        cv2.line(frame, (x_top, horizon), (x_bottom, h), color, 1)

    return frame

# -----------------------------
# WINDOW SETUP
# -----------------------------
cv2.namedWindow("Rover Navigation", cv2.WINDOW_NORMAL)

# -----------------------------
# MAIN LOOP
# -----------------------------
while True:

    ret, frame = cap.read()

    # Reconnect if stream fails
    if not ret:
        print("Reconnecting...")
        cap.release()
        cap = cv2.VideoCapture(stream_url)
        continue

    # Resize frame
    frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))

    # -------------------------
    # YOLO OBJECT DETECTION
    # -------------------------
    det_results = detector(frame)

    # Draw boxes manually
    annotated = frame.copy()

    for box in det_results[0].boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = detector.names[cls]

        text = f"{label} {conf:.2f}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(annotated, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # -------------------------
    # ROCK CLASSIFICATION
    # -------------------------
    cls_results = classifier(frame)

    label = cls_results[0].names[cls_results[0].probs.top1]
    confidence = cls_results[0].probs.top1conf

    text = f"{label} {confidence:.2f}"

    h, w = annotated.shape[:2]

    # -------------------------
    # VISUAL LOGIC
    # -------------------------
    if label == "rock":

        # red warning box
        cv2.rectangle(
            annotated,
            (int(w * 0.3), int(h * 0.3)),
            (int(w * 0.7), int(h * 0.7)),
            (0, 0, 255),
            3
        )

        color = (0, 0, 255)

    else:
        # draw mesh when safe
        annotated = draw_3d_mesh(annotated)
        color = (0, 255, 0)

    # -------------------------
    # TEXT DISPLAY
    # -------------------------
    cv2.putText(
        annotated,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    # -------------------------
    # SHOW OUTPUT
    # -------------------------
    cv2.imshow("Rover Navigation", annotated)

    if cv2.waitKey(1) == 27:
        break

# -----------------------------
# CLEANUP
# -----------------------------
cap.release()
cv2.destroyAllWindows()