import numpy as np


class EKF:
    def __init__(self, dt):
        self.dt = dt

        # stan: [x, y, v, yaw, yaw_rate]
        self.x = np.zeros((5, 1))

        self.P = np.eye(5) * 500
        self.Q = np.diag([1, 1, 10, 0.1, 0.1])
        self.R = np.diag([5, 5])

    def predict(self):
        x, y, v, yaw, w = self.x.flatten()
        dt = self.dt

        if abs(w) < 1e-4:
            x += v * np.cos(yaw) * dt
            y += v * np.sin(yaw) * dt
        else:
            x += (v / w) * (np.sin(yaw + w * dt) - np.sin(yaw))
            y += (v / w) * (-np.cos(yaw + w * dt) + np.cos(yaw))
            yaw += w * dt

        self.x = np.array([[x], [y], [v], [yaw], [w]])
        self.P = self.P + self.Q

    def update(self, z):
        H = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0]
        ])

        y = z - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(5) - K @ H) @ self.P

    def predict_future(self, steps):
        x_backup = self.x.copy()
        P_backup = self.P.copy()

        for _ in range(steps):
            self.predict()

        pred = self.x.copy()
        cov = self.P.copy()

        self.x = x_backup
        self.P = P_backup

        return pred, cov
