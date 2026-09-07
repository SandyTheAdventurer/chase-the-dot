import socket, struct, threading, time
from typing import Optional, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np

_RX = struct.Struct(">iiii??2s")
_P = struct.Struct(">cbbii2s")
_C = struct.Struct(">cbbId2s")

def normalize(s: np.ndarray) -> np.ndarray:
    """Normalize a raw pixel state to the observation range."""
    s = np.asarray(s, dtype=np.float32)
    if s.shape[-1] == 9:
        is_1d = s.ndim == 1
        if is_1d: s = s[np.newaxis, :]
        norm = np.column_stack([
            s[:, :2] / 1000.0,
            np.clip(s[:, 2:4], -100.0, 100.0) / 50.0,
            s[:, 4:5],
            np.clip(s[:, 5:7], -300.0, 300.0) / 300.0,
            np.clip(s[:, 7:9], -10000.0, 10000.0) / 10000.0
        ])
        return norm[0] if is_1d else norm
    if s.shape[-1] == 11:
        is_1d = s.ndim == 1
        if is_1d: s = s[np.newaxis, :]
        norm = np.column_stack([
            s[:, :2] / 1000.0,
            np.clip(s[:, 2:4], -100.0, 100.0) / 50.0,
            s[:, 6:7],
            np.clip(s[:, 7:9], -300.0, 300.0) / 300.0,
            np.clip(s[:, 9:11], -10000.0, 10000.0) / 10000.0
        ])
        return norm[0] if is_1d else norm
    return s

class ChaseTheDotEnv(gym.Env):
    """Gymnasium TCP Client Environment for Chase the Dot."""
    metadata = {"render_modes": []}

    def __init__(self, host: str = "127.0.0.1", port: int = 6102, timeout: float = 5.0, action_scale: float = 20.0, **kwargs) -> None:
        super().__init__()
        self.host, self.port, self.timeout = host, port, timeout
        self.current_size = 50.0
        self.action_scale = action_scale
        self.prev_gx = self.prev_gy = None
        self.vx = self.vy = 0.0
        self.error_x = self.error_y = False
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.observation_space = spaces.Box(
            np.array([-2., -2., -2., -2., 0., -2., -2., -2., -2.], np.float32),
            np.array([2., 2., 2., 2., 1., 2., 2., 2., 2.], np.float32),
            (9,), np.float32,
        )
        self.socket: Optional[socket.socket] = None
        self._latest = self._last_time = None
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
                        now = time.time()
                        dt = now - self._last_time if self._last_time else 0.016
                        self._last_time = now
                        if self.prev_gx is not None and dt > 0:
                            dx, dy = gx - self.prev_gx, gy - self.prev_gy
                            jump = float(np.hypot(dx, dy))
                            if jump > 40.0:
                                self.vx, self.vy = 0.0, 0.0
                            elif jump > 0.0:
                                self.vx = 0.8 * self.vx + 0.2 * (dx / dt)
                                self.vy = 0.8 * self.vy + 0.2 * (dy / dt)
                        self.prev_gx, self.prev_gy = gx, gy
                        self.error_x, self.error_y = bool(ex), bool(ey)
                        self._latest = np.array([gx, gy, bx, by, dt, self.vx, self.vy, 0.0, 0.0], dtype=np.float32)
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
        dist = float(np.hypot(s[2], s[3]))
        in_bounds = not (bool(s[4]) or bool(s[5])) if s.shape[-1] == 11 else not (self.error_x or self.error_y)
        reward = 1.0 - (dist / 100.0) if in_bounds else -0.2 - (dist / 50.0)
        dt = float(s[6]) if s.shape[-1] == 11 else float(s[4])
        err_x = bool(s[4]) if s.shape[-1] == 11 else self.error_x
        err_y = bool(s[5]) if s.shape[-1] == 11 else self.error_y
        info = {"distance": dist, "in_bounds": in_bounds, "error_x": err_x, "error_y": err_y, "dt": dt, "state": s}
        return normalize(s), reward, info

    def step(self, action, wait_for_new: bool = True) -> Tuple[np.ndarray, float, bool, bool, dict]:
        gx, gy = (self._latest[0], self._latest[1]) if self._latest is not None else (0.0, 0.0)
        offset = self.current_size / (2.0 * np.pi)
        dt = float(self._latest[6]) if (self._latest is not None and len(self._latest) == 11) else (float(self._latest[4]) if self._latest is not None else 0.016)
        x = int(np.clip(round(gx + offset + self.vx * dt + float(action[0]) * self.action_scale), -50, 950))
        y = int(np.clip(round(gy + offset + self.vy * dt + float(action[1]) * self.action_scale), -50, 950))
        if self.socket:
            self.socket.sendall(_P.pack(b"P", 13, 10, x, y, b"\r\n"))
        s = self.receive_state(wait_for_new=wait_for_new)
        obs, reward, info = self._metrics(s)
        return obs, reward, False, False, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        obs, _, info = self._metrics(self.receive_state(wait_for_new=False))
        return obs, info

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