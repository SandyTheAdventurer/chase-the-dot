import math, socket, struct, threading
from typing import Optional, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np

_RX = struct.Struct(">iiii??2s")
_P = struct.Struct(">cbbii2s")
_C = struct.Struct(">cbbId2s")

def normalize(s: np.ndarray) -> np.ndarray:
    """Normalize a raw pixel state to the observation range [-2.0, 2.0].
    Idempotent: if s is already in normalized observation space, returns it directly.
    """
    s = np.asarray(s, dtype=np.float32)
    if s.ndim == 1:
        if s.shape[0] % 8 == 0:
            if abs(float(s[0])) <= 2.0 and abs(float(s[1])) <= 2.0:
                return s
            if s.shape[0] == 8:
                out = np.empty(8, dtype=np.float32)
                out[0] = s[0] * 0.001
                out[1] = s[1] * 0.001
                out[2] = max(-2.0, min(2.0, float(s[2]) * 0.02))
                out[3] = max(-2.0, min(2.0, float(s[3]) * 0.02))
                out[4] = max(-2.0, min(2.0, float(s[4]) * 0.05))
                out[5] = max(-2.0, min(2.0, float(s[5]) * 0.05))
                out[6] = max(-2.0, min(2.0, float(s[6]) * 0.1))
                out[7] = max(-2.0, min(2.0, float(s[7]) * 0.1))
                return out
            out = np.empty_like(s)
            for i in range(0, s.shape[0], 8):
                out[i:i+8] = normalize(s[i:i+8])
            return out
    elif s.ndim == 2:
        if s.shape[1] % 8 == 0:
            if np.all(np.abs(s[:, :2]) <= 2.0):
                return s
            if s.shape[1] == 8:
                out = np.empty_like(s)
                out[:, :2] = s[:, :2] * 0.001
                out[:, 2:4] = np.clip(s[:, 2:4], -100.0, 100.0) * 0.02
                out[:, 4:6] = np.clip(s[:, 4:6], -40.0, 40.0) * 0.05
                out[:, 6:8] = np.clip(s[:, 6:8], -20.0, 20.0) * 0.1
                return out
            out = np.empty_like(s)
            for i in range(0, s.shape[1], 8):
                out[:, i:i+8] = normalize(s[:, i:i+8])
            return out
    return s

class ChaseTheDotEnv(gym.Env):
    """Gymnasium TCP Client Environment for Chase the Dot."""
    metadata = {"render_modes": []}

    def __init__(self, host: str = "127.0.0.1", port: int = 6102, timeout: float = 5.0, action_scale: float = 50.0, oob_penalty: float = 1.0, frame_stack: int = 4, **kwargs) -> None:
        super().__init__()
        self.host, self.port, self.timeout = host, port, timeout
        self.current_size = 50.0
        self.action_scale = action_scale
        self.oob_penalty = float(oob_penalty)
        self.frame_stack = max(1, int(frame_stack))
        from collections import deque
        self._frames = deque(maxlen=self.frame_stack)
        self.prev_gx = self.prev_gy = None
        self.v_smooth_x = self.v_smooth_y = 0.0
        self.a_smooth_x = self.a_smooth_y = 0.0
        self.prev_dx = self.prev_dy = None
        self.error_x = self.error_y = False
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.observation_space = spaces.Box(
            np.array([-2.] * (8 * self.frame_stack), np.float32),
            np.array([2.] * (8 * self.frame_stack), np.float32),
            (8 * self.frame_stack,), np.float32,
        )
        self.socket: Optional[socket.socket] = None
        self._latest = None
        self._new_data, self._stop = threading.Event(), threading.Event()

    def connect(self, host: Optional[str] = None, port: Optional[int] = None) -> None:
        self.host, self.port = host or self.host, port or self.port
        if self.socket: self.close()
        self.socket = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._stop.clear()
        threading.Thread(target=self._drain, name="ChaseTheDot-Drain", daemon=True).start()

    def _drain(self) -> None:
        buf = bytearray()
        while not self._stop.is_set() and self.socket:
            try:
                chunk = self.socket.recv(4096)
                if not chunk: break
                buf.extend(chunk)
                while len(buf) >= 20:
                    if buf[18:20] == b"\r\n":
                        gx, gy, bx, by, ex, ey, _ = _RX.unpack(buf[:20])
                        if self.prev_gx is not None:
                            dx, dy = float(gx - self.prev_gx), float(gy - self.prev_gy)
                            jump = float(np.hypot(dx, dy))
                            if jump > 40.0:
                                self.v_smooth_x, self.v_smooth_y = 0.0, 0.0
                                self.a_smooth_x, self.a_smooth_y = 0.0, 0.0
                                self.prev_dx = self.prev_dy = None
                            else:
                                self.v_smooth_x = dx
                                self.v_smooth_y = dy
                                if self.prev_dx is not None:
                                    ax = dx - self.prev_dx
                                    ay = dy - self.prev_dy
                                    self.a_smooth_x = ax
                                    self.a_smooth_y = ay
                                self.prev_dx, self.prev_dy = dx, dy
                        self.prev_gx, self.prev_gy = gx, gy
                        self.error_x, self.error_y = bool(ex), bool(ey)
                        target_ox = 3.0 + 0.33 * (self.current_size - 10.0)
                        target_oy = 1.5 + 0.33 * (self.current_size - 10.0)
                        err_x = target_ox - bx
                        err_y = target_oy - by
                        self._latest = np.array([gx, gy, err_x, err_y, self.v_smooth_x, self.v_smooth_y, self.a_smooth_x, self.a_smooth_y], dtype=np.float32)
                        del buf[:20]
                        self._new_data.set()
                    else:
                        idx = buf.find(b"\r\n")
                        del buf[:len(buf) if idx == -1 else idx + 2]
            except Exception: break

    def configure(self, speed: int, size: float) -> None:
        self.current_size = float(size)
        if self.socket: self.socket.sendall(_C.pack(b"C", 13, 10, int(speed), float(size), b"\r\n"))

    def receive_state(self, wait_for_new: bool = False, timeout: Optional[float] = None) -> np.ndarray:
        t = timeout or self.timeout
        if wait_for_new:
            self._new_data.clear()
            if not self._new_data.wait(t): raise TimeoutError("Timed out waiting for state")
        elif self._latest is None and not self._new_data.wait(t):
            raise TimeoutError("Timed out waiting for initial state")
        return self._latest

    def _metrics(self, s: np.ndarray) -> Tuple[np.ndarray, float, dict]:
        dist = math.hypot(float(s[2]), float(s[3]))
        num_flags = int(self.error_x) + int(self.error_y)
        in_bounds = (num_flags == 0)

        if in_bounds:
            target_radius = max(5.0, self.current_size * 0.5)
            norm_dist = dist / target_radius
            centering_bonus = math.exp(-2.0 * norm_dist * norm_dist)
            reward = 1.0 + centering_bonus
        else:
            reward = -self.oob_penalty * num_flags - (dist * 0.04)

        info = {"distance": dist, "in_bounds": in_bounds, "error_x": self.error_x, "error_y": self.error_y, "state": s}
        return normalize(s), reward, info

    def step(self, action, wait_for_new: bool = True) -> Tuple[np.ndarray, float, bool, bool, dict]:
        gx, gy = (self._latest[0], self._latest[1]) if self._latest is not None else (0.0, 0.0)
        target_ox = 3.0 + 0.33 * (self.current_size - 10.0)
        target_oy = 1.5 + 0.33 * (self.current_size - 10.0)
        x = int(np.clip(round(gx + target_ox + self.v_smooth_x + 0.5 * self.a_smooth_x + float(action[0]) * self.action_scale), -50, 950))
        y = int(np.clip(round(gy + target_oy + self.v_smooth_y + 0.5 * self.a_smooth_y + float(action[1]) * self.action_scale), -50, 950))
        if self.socket:
            self.socket.sendall(_P.pack(b"P", 13, 10, x, y, b"\r\n"))
        s = self.receive_state(wait_for_new=wait_for_new)
        obs, reward, info = self._metrics(s)
        self._frames.append(obs)
        stacked_obs = np.concatenate(list(self._frames), axis=-1)
        return stacked_obs, reward, False, False, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        obs, _, info = self._metrics(self.receive_state(wait_for_new=False))
        self._frames.clear()
        for _ in range(self.frame_stack):
            self._frames.append(obs)
        stacked_obs = np.concatenate(list(self._frames), axis=-1)
        return stacked_obs, info

    def close(self) -> None:
        self._stop.set()
        if self.socket:
            try: self.socket.shutdown(socket.SHUT_RDWR)
            except OSError: pass
            self.socket.close()
            self.socket = None

    def __enter__(self) -> "ChaseTheDotEnv":
        self.connect()
        return self

    def __exit__(self, *args) -> None: self.close()

Environment = ChaseTheDotEnv