import numpy as np

class PID:
    def __init__(self, kp=None, ki=None, kd=None):
        # Base PID parameters
        self.kpx = kp if kp else 0.95
        self.kix = ki if ki else 0.01
        self.kdx = kd if kd else 0.05

        self.kpy = kp if kp else 0.95
        self.kiy = ki if ki else 0.01
        self.kdy = kd if kd else 0.05

        self.prev_dx = 0.0
        self.prev_dy = 0.0
        self.prev_ix = 0.0
        self.prev_iy = 0.0

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        if X is None:
            return None
            
        gx, gy = float(X[0]), float(X[1])
        bx, by = float(X[2]), float(X[3])
        dt = float(X[6]) if len(X) > 6 else 1.0
        if dt <= 0.0: dt = 1e-4

        blue_x = gx + bx
        blue_y = gy + by

        px = self.kpx * bx
        self.prev_ix += bx * dt
        self.prev_ix = max(-1000.0, min(1000.0, self.prev_ix))
        ix = self.kix * self.prev_ix
        dfx = self.kdx * (bx - self.prev_dx)

        py = self.kpy * by
        self.prev_iy += by * dt
        self.prev_iy = max(-1000.0, min(1000.0, self.prev_iy))
        iy = self.kiy * self.prev_iy
        dfy = self.kdy * (by - self.prev_dy)

        ex = px + ix + dfx
        ey = py + iy + dfy

        action_x = (bx - ex) / 100.0
        action_y = (by - ey) / 100.0

        self.prev_dx = bx
        self.prev_dy = by

        return np.array([action_x, action_y])

    def learn(self, reward):
        return 0.0