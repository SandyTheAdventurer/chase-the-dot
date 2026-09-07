import numpy as np

class PID:
    def __init__(self, kp=None, ki=None, kd=None):
        self.kpx = self.kpy = 0.10 if kp is None else float(kp)
        self.kix = self.kiy = 0.00 if ki is None else float(ki)
        self.kdx = self.kdy = 0.00 if kd is None else float(kd)
        self.integral = np.zeros(2, dtype=np.float32)
        self.prev_err = None

    def __call__(self, X): return self.forward(X)

    def forward(self, X):
        if X is None: return None
        err = np.array([X[2], X[3]], dtype=np.float32)
        dt = max(float(X[4]) if len(X) == 9 else (float(X[6]) if len(X) > 6 else 0.016), 1e-4)
        self.integral = np.clip(self.integral + err * dt, -100.0, 100.0)
        d_err = (err - self.prev_err) if self.prev_err is not None else np.zeros(2, dtype=np.float32)
        self.prev_err = err
        e = self.kpx * err + self.kix * self.integral + self.kdx * d_err
        return np.clip(- e / 20.0, -1.0, 1.0)

    def learn(self, reward): return 0.0
    def save(self, path): pass
    def load(self, path): pass