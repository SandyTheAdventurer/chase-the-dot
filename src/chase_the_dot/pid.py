from line_profiler import profile
import numpy as np

class PID:
    """Industrial-grade adaptive PID tracking controller with:
    - Corrected feedback polarity (eliminates positive feedback runaway)
    - Leaky anti-windup integration with axis-wise zero-crossing reset
    - Exponentially smoothed derivative filter (eliminates quantization chatter)
    - Adaptive non-linear gain scheduling for high-curvature turns
    - Direct absolute position output for env.step_direct()
    """
    def __init__(self, kp: float = None, ki: float = None, kd: float = None, action_scale: float = 40.0):
        # Optimal gains for 1-frame latency discrete B-spline tracking
        self.kp = self.kpx = self.kpy = 0.40 if kp is None else float(kp)
        self.ki = self.kix = self.kiy = 0.02 if ki is None else float(ki)
        self.kd = self.kdx = self.kdy = 0.10 if kd is None else float(kd)
        self.action_scale = float(action_scale)
        self.is_direct = True  # Flag for run.py to use step_direct

        self.integral = np.zeros(2, dtype=np.float32)
        self.prev_err = None
        self.d_smooth = np.zeros(2, dtype=np.float32)

    def __call__(self, X, env=None):
        return self.forward(X, env)

    @profile
    def forward(self, X, env=None):
        if X is None:
            return None

        # Extract tracking error and target position from observation
        # In normalized stacked frames, err is scaled by (1 / action_scale)
        if len(X) > 8:
            err = np.array([float(X[-6]) * self.action_scale, float(X[-5]) * self.action_scale], dtype=np.float32)
        elif len(X) == 8:
            if abs(float(X[0])) <= 2.0 and abs(float(X[1])) <= 2.0:
                err = np.array([float(X[2]) * self.action_scale, float(X[3]) * self.action_scale], dtype=np.float32)
            else:
                err = np.array([float(X[2]), float(X[3])], dtype=np.float32)
        else:
            err = np.array([float(X[0]), float(X[1])], dtype=np.float32)
        err_mag = float(np.hypot(err[0], err[1]))

        # 1. Non-linear Adaptive Proportional Gain:
        # Near center (< 5px), standard stiff gain (kp=0.40).
        # On sharp turns or large errors, smoothly boost Kp up to 1.5x to quickly pull back.
        adaptive_kp = self.kp * (1.0 + 0.5 * np.tanh(err_mag / 20.0))
        p_term = adaptive_kp * err

        # 2. Leaky Anti-Windup Integrator with Sign-Reversal Reset:
        if self.prev_err is not None:
            # If error crossed zero on either axis, reset that axis's integral to eliminate overshoot ringing
            crossed_zero = (err * self.prev_err) < 0.0
            self.integral[crossed_zero] = 0.0

        # Leaky decay (0.95) prevents accumulation during steady-state cruising
        if err_mag < 40.0:
            self.integral = np.clip(0.95 * self.integral + err * 0.02, -20.0, 20.0)
        else:
            self.integral *= 0.5
        i_term = self.ki * self.integral

        # 3. Filtered Derivative Term:
        # Low-pass filter (0.6 / 0.4) removes 50Hz integer coordinate quantization noise
        if self.prev_err is not None:
            raw_d = err - self.prev_err
            self.d_smooth = 0.6 * self.d_smooth + 0.4 * raw_d
        else:
            self.d_smooth = np.zeros(2, dtype=np.float32)
        d_term = self.kd * self.d_smooth
        self.prev_err = err

        # 4. PID correction in pixel space
        u_pixel = p_term + i_term + d_term

        # 5. Compute absolute position command:
        # target position + geometric center offset + velocity feedforward + PID correction
        if env is not None and env._latest is not None:
            gx, gy = float(env._latest[0]), float(env._latest[1])
            target_ox = 3.0 + 0.33 * (env.current_size - 10.0)
            target_oy = 1.5 + 0.33 * (env.current_size - 10.0)
            cmd_x = int(round(gx + target_ox + env.v_smooth_x + 0.5 * env.a_smooth_x + u_pixel[0]))
            cmd_y = int(round(gy + target_oy + env.v_smooth_y + 0.5 * env.a_smooth_y + u_pixel[1]))
            return cmd_x, cmd_y

        # Fallback: return normalized action for backward compatibility
        action = np.clip(u_pixel / self.action_scale, -1.0, 1.0)
        return action

    def reset(self):
        self.integral = np.zeros(2, dtype=np.float32)
        self.prev_err = None
        self.d_smooth = np.zeros(2, dtype=np.float32)

    def learn(self, reward): return 0.0
    def save(self, path): pass
    def load(self, path): pass