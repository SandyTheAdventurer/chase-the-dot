import math, socket, struct, threading
from typing import Optional, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np

_RX = struct.Struct(">iiii??2s")
_P = struct.Struct(">cbbii2s")
_C = struct.Struct(">cbbId2s")

_SCALES = np.array([0.001, 0.001, 1.0 / 40.0, 1.0 / 40.0, 0.05, 0.05, 0.1, 0.1], dtype=np.float32)
_SCALES_40_CACHE = {40.0: _SCALES}

def normalize(s: np.ndarray, action_scale: float = 40.0) -> np.ndarray:
    """Normalize a raw pixel state to the observation range [-2.0, 2.0].
    Idempotent: if s is already in normalized observation space, returns it directly.
    """
    if isinstance(s, np.ndarray) and s.dtype == np.float32:
        pass
    else:
        s = np.asarray(s, dtype=np.float32)
    if s.ndim == 1:
        n = s.shape[0]
        if n % 8 == 0:
            if s[0] >= -2.0 and s[0] <= 2.0 and s[1] >= -2.0 and s[1] <= 2.0:
                return s
            if n == 8:
                scales = _SCALES_40_CACHE.get(action_scale)
                if scales is None:
                    scales = _SCALES.copy()
                    scales[2] = 1.0 / action_scale
                    scales[3] = 1.0 / action_scale
                    _SCALES_40_CACHE[action_scale] = scales
                return np.clip(s * scales, -2.0, 2.0)
            out = np.empty_like(s)
            scales = _SCALES_40_CACHE.get(action_scale)
            if scales is None:
                scales = _SCALES.copy()
                scales[2] = 1.0 / action_scale
                scales[3] = 1.0 / action_scale
                _SCALES_40_CACHE[action_scale] = scales
            for i in range(0, n, 8):
                out[i:i+8] = np.clip(s[i:i+8] * scales, -2.0, 2.0)
            return out
    elif s.ndim == 2:
        if s.shape[1] % 8 == 0:
            if s[0, 0] >= -2.0 and s[0, 0] <= 2.0 and s[0, 1] >= -2.0 and s[0, 1] <= 2.0:
                return s
            scales = _SCALES_40_CACHE.get(action_scale)
            if scales is None:
                scales = _SCALES.copy()
                scales[2] = 1.0 / action_scale
                scales[3] = 1.0 / action_scale
                _SCALES_40_CACHE[action_scale] = scales
            if s.shape[1] == 8:
                return np.clip(s * scales, -2.0, 2.0)
            out = np.empty_like(s)
            for i in range(0, s.shape[1], 8):
                out[:, i:i+8] = np.clip(s[:, i:i+8] * scales, -2.0, 2.0)
            return out
    return s

class ChaseTheDotEnv(gym.Env):
    """Gymnasium TCP Client Environment for Chase the Dot."""
    metadata = {"render_modes": []}

    def __init__(self, host: str = "127.0.0.1", port: int = 6102, timeout: float = 5.0, action_scale: float = 40.0, oob_penalty: float = 1.0, frame_stack: int = 12, **kwargs) -> None:
        super().__init__()
        self.host, self.port, self.timeout = host, port, timeout
        self.current_size = 50.0
        self.action_scale = action_scale
        self.oob_penalty = float(oob_penalty)
        self.frame_stack = max(1, int(frame_stack))
        self._frame_stack_arr = np.empty((self.frame_stack, 8), dtype=np.float32)
        self._frame_ptr = 0
        self._frame_count = 0
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
        self._latest_buf = np.zeros(8, dtype=np.float32)

    def normalize(self, s: np.ndarray) -> np.ndarray:
        """Normalize a raw pixel state using this env's action_scale."""
        return normalize(s, self.action_scale)

    def connect(self, host: Optional[str] = None, port: Optional[int] = None) -> None:
        self.host, self.port = host or self.host, port or self.port
        if self.socket: self.close()
        self.socket = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._stop.clear()
        threading.Thread(target=self._drain, name="ChaseTheDot-Drain", daemon=True).start()

    def _drain(self) -> None:
        buf = bytearray()
        _local = self._latest_buf
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
                            jump = math.sqrt(dx * dx + dy * dy)
                            if jump > 150.0:
                                self.v_smooth_x, self.v_smooth_y = 0.0, 0.0
                                self.a_smooth_x, self.a_smooth_y = 0.0, 0.0
                                self.prev_dx = self.prev_dy = None
                            else:
                                self.v_smooth_x = dx
                                self.v_smooth_y = dy
                                if self.prev_dx is not None:
                                    self.a_smooth_x = dx - self.prev_dx
                                    self.a_smooth_y = dy - self.prev_dy
                                self.prev_dx, self.prev_dy = dx, dy
                        self.prev_gx, self.prev_gy = gx, gy
                        self.error_x, self.error_y = bool(ex), bool(ey)
                        target_ox = 3.0 + 0.33 * (self.current_size - 10.0)
                        target_oy = 1.5 + 0.33 * (self.current_size - 10.0)
                        _local[0] = gx
                        _local[1] = gy
                        _local[2] = target_ox - bx
                        _local[3] = target_oy - by
                        _local[4] = self.v_smooth_x
                        _local[5] = self.v_smooth_y
                        _local[6] = self.a_smooth_x
                        _local[7] = self.a_smooth_y
                        self._latest = _local
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
        dist = math.sqrt(float(s[2]) * float(s[2]) + float(s[3]) * float(s[3]))
        num_flags = int(self.error_x) + int(self.error_y)
        in_bounds = (num_flags == 0)

        reward = 1.0 if in_bounds else 0.0

        info = {"distance": dist, "in_bounds": in_bounds, "error_x": self.error_x, "error_y": self.error_y, "state": s}
        return self.normalize(s), reward, info

    def step(self, action, wait_for_new: bool = True) -> Tuple[np.ndarray, float, bool, bool, dict]:
        gx, gy = (self._latest[0], self._latest[1]) if self._latest is not None else (0.0, 0.0)
        target_ox = 3.0 + 0.33 * (self.current_size - 10.0)
        target_oy = 1.5 + 0.33 * (self.current_size - 10.0)
        x = int(max(-50, min(950, round(gx - target_ox + self.v_smooth_x + 0.5 * self.a_smooth_x + float(action[0]) * self.action_scale))))
        y = int(max(-50, min(950, round(gy - target_oy + self.v_smooth_y + 0.5 * self.a_smooth_y + float(action[1]) * self.action_scale))))
        if self.socket:
            self.socket.sendall(_P.pack(b"P", 13, 10, x, y, b"\r\n"))
        s = self.receive_state(wait_for_new=wait_for_new)
        obs, reward, info = self._metrics(s)
        self._frame_stack_arr[self._frame_ptr] = obs
        self._frame_ptr = (self._frame_ptr + 1) % self.frame_stack
        self._frame_count = min(self._frame_count + 1, self.frame_stack)
        stacked_obs = self._frame_stack_arr[:self._frame_count].ravel()
        return stacked_obs, reward, False, False, info

    def step_direct(self, x: int, y: int, wait_for_new: bool = True) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Send an absolute pixel position command without feedforward compensation.
        Intended for standalone controllers (e.g. PID) that compute their own position.
        """
        x -= 3.0 + 0.33 * (self.current_size - 10.0)
        y -= 1.5 + 0.33 * (self.current_size - 10.0)
        x = int(max(-50, min(950, x)))
        y = int(max(-50, min(950, y)))
        if self.socket:
            self.socket.sendall(_P.pack(b"P", 13, 10, x, y, b"\r\n"))
        s = self.receive_state(wait_for_new=wait_for_new)
        obs, reward, info = self._metrics(s)
        self._frame_stack_arr[self._frame_ptr] = obs
        self._frame_ptr = (self._frame_ptr + 1) % self.frame_stack
        self._frame_count = min(self._frame_count + 1, self.frame_stack)
        stacked_obs = self._frame_stack_arr[:self._frame_count].ravel()
        return stacked_obs, reward, False, False, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        obs, _, info = self._metrics(self.receive_state(wait_for_new=False))
        self._frame_stack_arr[:] = obs
        self._frame_ptr = 0
        self._frame_count = self.frame_stack
        stacked_obs = self._frame_stack_arr[:self._frame_count].ravel()
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