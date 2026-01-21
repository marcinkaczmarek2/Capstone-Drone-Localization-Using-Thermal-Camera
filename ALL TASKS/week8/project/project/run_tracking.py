import cv2
import numpy as np
import os
from tracker import DroneTracker
from visualize import draw_trajectory, draw_prediction, draw_uncertainty, draw_trajectory_alpha, draw_prediction_alpha

VIDEO = r"C:\Users\szymo\DroneLocalization\YOLOv5-kalman\input.mp4"
FPS = 24
WIDTH, HEIGHT = 640, 512

print("CWD:", os.getcwd())
print("VIDEO:", VIDEO)
print("EXISTS:", os.path.exists(VIDEO))

LABELS_DIR = r"C:\Users\szymo\DroneLocalization\YOLOv5-kalman\detections\run6\labels"

def get_detection(frame_idx):
    txt_file = os.path.join(LABELS_DIR, f"{frame_idx+1:06d}.txt")
    if not os.path.exists(txt_file):
        return None

    with open(txt_file, "r") as f:
        line = f.readline().strip()
        if not line:
            return None
        parts = line.split()
        x_norm = float(parts[1])
        y_norm = float(parts[2])
        x_px = int(x_norm * WIDTH)
        y_px = int(y_norm * HEIGHT)
        return (x_px, y_px)

cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    raise ValueError(f"Nie można otworzyć pliku {VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS) or FPS
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or WIDTH
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or HEIGHT

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output_predicted.mp4", fourcc, fps, (width, height))

tracker = DroneTracker(FPS)
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detection = get_detection(frame_idx)
    if detection is None and not tracker.initialized:
        print("detection is None and not tracker.initialized")
        frame_idx += 1
        continue
    pos, history, pred, cov = tracker.step(detection)

    draw_trajectory_alpha(frame, history, color=(0, 255, 0), thickness=2, alpha=0.5)
    if pred is not None:
        draw_prediction_alpha(frame, pred, cov, color=(255, 0, 0), thickness=2, alpha=0.5)

    out.write(frame)
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) == 27:
        break

    frame_idx += 1

cap.release()
out.release()
cv2.destroyAllWindows()
