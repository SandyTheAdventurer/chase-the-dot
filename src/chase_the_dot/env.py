import socket, struct, threading, time
from typing import Any, List, Optional, Sequence, Tuple, Union
import gymnasium as gym
from gymnasium import spaces
import numpy as np

_RX = struct.Struct(">iiii??2s")
_P = struct.Struct(">cbbii2s")
_C = struct.Struct(">cbbId2s")
def normalize(state) -> np.ndarray:
    """Normalize a raw pixel state to the observation range.

    If the state is already normalized (first element <= 2.0), it is
    returned as-is.  Otherwise the first four elements (green x/y and
    tracking error x/y) are divided by 1000 to bring them into a
    small range suitable for neural networks.

    Works with numpy arrays, lists, or torch tensors.
    """
    s = np.asarray(state, dtype=np.float32)
    if s[0] > 2.0:
        return np.array([
            s[0] / 1000., s[1] / 1000.,
            s[2] / 1000., s[3] / 1000.,
            s[4], s[5], s[6],
        ], dtype=np.float32)
    return s


class ChaseTheDotEnv(gym.Env):
    """Gymnasium TCP Client Environment for Chase the Dot.

    Action is ``[x, y]`` absolute pixel coordinates for the Blue dot.
    action_space: Box([-50, -50], [950, 750])
    """
    metadata = {"render_modes": []}
    normalize = staticmethod(normalize)

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 6102,
        timeout: float = 5.0,
    ) -> None:
        super().__init__()
        self.host, self.port, self.timeout = host, port, timeout

        self.action_space = spaces.Box(
            np.array([-50., -50.], np.float32),
            np.array([950., 750.], np.float32),
            (2,), np.float32,
        )

        self.observation_space = spaces.Box(
            np.array([-2., -2., -2., -2., 0., 0., 0.], np.float32),
            np.array([2., 2., 2., 2., 1., 1., 1.], np.float32),
            (7,), np.float32,
        )
        self.socket: Optional[socket.socket] = None
        self._latest: Optional[np.ndarray] = None
        self._last_time: Optional[float] = None
        self._new_data = threading.Event()
        self._stop = threading.Event()

    def connect(self, host: Optional[str] = None, port: Optional[int] = None) -> None:
        self.host, self.port = host or self.host, port or self.port
        if self.socket: self.close()
        self.socket = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self._stop.clear()
        threading.Thread(target=self._drain, name="ChaseTheDot-Drain", daemon=True).start()

    @property
    def is_connected(self) -> bool: return self.socket is not None

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
                        current_time = time.time()
                        dt = current_time - self._last_time if self._last_time else 0.0
                        self._last_time = current_time
                        self._latest = np.array([gx, gy, bx, by, float(ex), float(ey), dt], dtype=np.float32)
                        del buf[:20]
                        self._new_data.set()
                    else:
                        idx = buf.find(b"\r\n")
                        del buf[:len(buf) if idx == -1 else idx + 2]
            except Exception: break

    def send_position(self, x: int, y: int) -> None:
        if self.socket: self.socket.sendall(_P.pack(b"P", 13, 10, int(x), int(y), b"\r\n"))

    def configure(self, speed: int, size: float) -> None:
        if self.socket: self.socket.sendall(_C.pack(b"C", 13, 10, int(speed), float(size), b"\r\n"))

    def receive_state(self, wait_for_new: bool = False, timeout: Optional[float] = None) -> np.ndarray:
        t = timeout or self.timeout
        if wait_for_new:
            self._new_data.clear()
            if not self._new_data.wait(t): raise TimeoutError("Timed out waiting for state")
        elif self._latest is None:
            if not self._new_data.wait(t): raise TimeoutError("Timed out waiting for initial state")
        return self._latest

    def _metrics(self, s: np.ndarray) -> Tuple[np.ndarray, float, dict]:
        dist = float((s[2]**2 + s[3]**2)**0.5)
        in_bounds = not (bool(s[4]) or bool(s[5]))
        reward = 1.0 - min(0.5, dist / 100.) if in_bounds else -0.2 - min(2.0, dist / 50.)
        obs = np.array([s[0]/1000., s[1]/1000., s[2]/1000., s[3]/1000., s[4], s[5], s[6]], dtype=np.float32)
        info = {"distance": dist, "in_bounds": in_bounds, "error_x": bool(s[4]), "error_y": bool(s[5]), "dt": float(s[6]), "state": s}
        return obs, reward, info

    def _resolve_action(self, action, y: Optional[int] = None) -> Tuple[int, int]:
        """Convert any supported action format to absolute (x, y) pixel coordinates."""
        # Legacy: step(x_int, y_int)
        if y is not None:
            x, y = int(action), int(y)
        else:
            x, y = float(action[0]), float(action[1])

        # Clamp to screen bounds
        x = max(-50, min(950, int(round(x))))
        y = max(-50, min(750, int(round(y))))
        return x, y

    def step(self, action, y: Optional[int] = None, wait_for_new: bool = True) -> Tuple[np.ndarray, float, bool, bool, dict]:
        x, y = self._resolve_action(action, y)
        self.send_position(x, y)
        s = self.receive_state(wait_for_new=wait_for_new)
        obs, reward, info = self._metrics(s)
        return obs, reward, False, False, info

    step_rl = step

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        s = self.receive_state(wait_for_new=False)
        obs, _, info = self._metrics(s)
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

    def __exit__(self, *args) -> None:
        self.close()

Environment = ChaseTheDotEnv