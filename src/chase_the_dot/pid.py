import numpy as np

class PID:
    def __init__(self):
        # Base PID parameters
        self.kpx = 0.95
        self.kix = 0.01
        self.kdx = 0.05

        self.kpy = 0.95
        self.kiy = 0.01
        self.kdy = 0.05

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
        ix = self.kix * self.prev_ix
        dfx = self.kdx * (bx - self.prev_dx)

        py = self.kpy * by
        self.prev_iy += by * dt
        iy = self.kiy * self.prev_iy
        dfy = self.kdy * (by - self.prev_dy)

        ex = px + ix + dfx
        ey = py + iy + dfy

        action_x = max(-50, min(950, int(round(blue_x - ex))))
        action_y = max(-50, min(750, int(round(blue_y - ey))))

        self.prev_dx = bx
        self.prev_dy = by

        return np.array([action_x, action_y])

    def learn(self, reward):
        return 0.0