import numpy as np
from ekf import EKF


class DroneTracker:
    def __init__(self, fps, history_len=30):
        self.dt = 1.0 / fps
        self.ekf = EKF(self.dt)
        self.initialized = False
        self.history = []
        self.history_len = history_len

    def step(self, detection):
        if detection is not None:
            z = np.array([[detection[0]], [detection[1]]])
            if not self.initialized:
                self.ekf.x[0, 0] = detection[0]
                self.ekf.x[1, 0] = detection[1]
                self.initialized = True
            self.ekf.update(z)

        if len(self.history) >= 2:
            N = min(self.history_len, len(self.history)-1)
            dx = self.history[-1][0] - self.history[-1-N][0]
            dy = self.history[-1][1] - self.history[-1-N][1]
            v = np.sqrt(dx**2 + dy**2) / (self.dt * N)
            yaw = np.arctan2(dy, dx)
            self.ekf.x[2, 0] = v
            self.ekf.x[3, 0] = yaw
            self.ekf.x[4, 0] = 0

        self.ekf.predict()

        x, y = self.ekf.x[0, 0], self.ekf.x[1, 0]
        self.history.append((int(x), int(y)))

        steps = int(2 / self.dt)
        pred, cov = self.ekf.predict_future(steps)

        return (x, y), self.history, pred, cov
