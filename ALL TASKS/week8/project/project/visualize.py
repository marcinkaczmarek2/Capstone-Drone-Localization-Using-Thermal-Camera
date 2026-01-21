import cv2
import numpy as np


def draw_trajectory(frame, history):
    for i in range(1, len(history)):
        cv2.line(frame, history[i - 1], history[i], (0, 255, 0), 2)


def draw_prediction(frame, pred):
    x, y = int(pred[0, 0]), int(pred[1, 0])
    cv2.circle(frame, (x, y), 6, (255, 0, 0), -1)


def draw_uncertainty(frame, cov, center):
    cov_xy = cov[:2, :2]
    eigvals, eigvecs = np.linalg.eig(cov_xy)

    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    axes = (int(np.sqrt(eigvals[0]) * 3), int(np.sqrt(eigvals[1]) * 3))

    cv2.ellipse(frame, center, axes, angle, 0, 360, (255, 0, 0), 2)


def draw_trajectory_alpha(frame, history, color=(0,255,0), thickness=2, alpha=0.5):
    overlay = frame.copy()
    for i in range(1, len(history)):
        cv2.line(overlay, history[i-1], history[i], color, thickness)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_prediction_alpha(frame, pred, cov, color=(255,0,0), thickness=2, alpha=0.5):
    overlay = frame.copy()
    x, y = int(pred[0,0]), int(pred[1,0])
    cv2.circle(overlay, (x,y), 4, color, -1)
    cov_xy = cov[:2, :2]
    eigvals, eigvecs = np.linalg.eig(cov_xy)
    angle = np.degrees(np.arctan2(eigvecs[1,0], eigvecs[0,0]))
    axes = (int(np.sqrt(eigvals[0])*2), int(np.sqrt(eigvals[1])*2))
    cv2.ellipse(overlay, (x,y), axes, angle, 0, 360, color, 1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)