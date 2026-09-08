# Chase the Dot

Real-time TCP tracking agent and Reinforcement Learning environment for the **CynLr "Chase the Dot"** control benchmark.

---

## 1. Problem Statement & System Understanding

Based on the official [problem-statement.pdf](problem-statement.pdf) and reverse-engineering of the live binary application (`Cy_RL_PS.exe`):

### 1.1 Objective
The benchmark challenges an external agent to control a **Blue Dot** in real time by sending target position commands over TCP. The goal is to track an autonomously moving **Green Dot** as closely as possible, minimizing position error $(\Delta X, \Delta Y)$ and keeping the error percentage below the required target ($< 1\%$ error rate).

### 1.2 Environment Details
- **Coordinate Space Bounds**: $X \in [-50, 950]$ and $Y \in [-50, 950]$. Target commands must be clamped to these bounds to avoid the "Blue out of screen bound" fault.
- **Observation State (`state` variable)**: The `receive_state()` function returns a 9-dimensional `float32` numpy array (boolean boundary error flags `error_x` and `error_y` are processed internally for reward and diagnostics, but omitted from the observation):
  1. `state[0]`: **Green X** (Target position)
  2. `state[1]`: **Green Y** 
  3. `state[2]`: **Tracking Error X** (`Green_X - Blue_X`). Positive means Green is to the right of Blue (Blue is to the left of Green).
  4. `state[3]`: **Tracking Error Y** (`Green_Y - Blue_Y`). Positive means Green is below Blue (Blue is above Green).
  5. `state[4]`: **Delta Time (dt)** in seconds since the last packet received.
  6. `state[5]`: **Green Velocity X (vx)** in pixels/sec.
  7. `state[6]`: **Green Velocity Y (vy)** in pixels/sec.
  8. `state[7]`: **Green Acceleration X (ax)** in pixels/sec².
  9. `state[8]`: **Green Acceleration Y (ay)** in pixels/sec².
- **RL Observation Space**: `env.step()` and `env.reset()` return a 9-dimensional normalized observation vector (`obs`):
  `[gx/1000, gy/1000, clip(bx, -100, 100)/50, clip(by, -100, 100)/50, dt, clip(vx, -300, 300)/300, clip(vy, -300, 300)/300, clip(ax, -10000, 10000)/10000, clip(ay, -10000, 10000)/10000]`.

### 1.3 Operating Modes
- **Testing Mode**: The Green Dot follows a dynamically generated random parametric path on each run.
- **Learning Mode**: The Green Dot follows a fixed path. Enables **Save** and **Load** options so the same trajectory can be repeated on loop (requires launching the application in **Administrator Mode** to access file saving/loading).

### 1.3 Application On-Screen Controls & Diagnostics
- **Object Speed (Samples/s)**: Configurable from `100` to `500` (counter-intuitively, `500` is the slowest rate, whereas `100` is the fastest).
- **Object Size (%)**: Configurable from `10` to `100` (`10` represents the smallest target radius/tolerance, `100` is largest).
- **Goal / Status Panel**:
  - Colored indicator dot: Green when within acceptable bounds, Red when out of bounds.
  - `Error %`: Percentage of sample cycles where the Blue Dot was outside the bounds of the Green Dot.
- **Plots & Panels**:
  - `Paths Traced`: Real-time coordinate graph of recent paths ($X \in [-100, 700]$, $Y \in [0, 500]$).
  - `Pos_Error Watch`: Live graph of $X$ and $Y$ tracking error over time.
  - `System Messages` & `TCP Packets`: Network diagnostics and raw communication packet logs.

---

## 2. TCP Protocol Specification

Communication is established over a single bi-directional TCP socket (default port: `6102`).

### 2.1 Inbound Data Stream (Application $\to$ Agent: 20 Bytes / packet)
The application operates in **continuous auto-drain streaming mode**, pushing 20-byte binary packets at high frequency:

| Byte Range | Field Name | Data Type | Format | Notes |
|:---|:---|:---|:---|:---|
| `0 .. 3` | `green_x` | `i32` (4B) | Big-Endian signed int | Autonomous Green Dot X position |
| `4 .. 7` | `green_y` | `i32` (4B) | Big-Endian signed int | Autonomous Green Dot Y position |
| `8 .. 11` | `blue_x` | `i32` (4B) | Big-Endian signed int | Blue Dot tracking error $\Delta X$ (0 when on target) |
| `12 .. 15` | `blue_y` | `i32` (4B) | Big-Endian signed int | Blue Dot tracking error $\Delta Y$ (0 when on target) |
| `16` | `error_x` | `bool` (1B) | `0x00` = In Bounds, `0x01` = Out of Bounds | X error flag |
| `17` | `error_y` | `bool` (1B) | `0x00` = In Bounds, `0x01` = Out of Bounds | Y error flag |
| `18` | CR (`\r`) | `char` (1B) | ASCII `0x0D` | Delimiter byte 1 |
| `19` | LF (`\n`) | `char` (1B) | ASCII `0x0A` | Delimiter byte 2 |

### 2.2 Outbound Commands (Agent $\to$ Application)

#### Position Mode (`P` Mode: 13 Bytes)
Moves the Blue Dot to target coordinates $(X, Y)$:
- Header: `b'P\r\n'` (3 bytes)
- `target_x`: `i32` Big-Endian (4 bytes)
- `target_y`: `i32` Big-Endian (4 bytes)
- Delimiter: `b'\r\n'` (2 bytes)

#### Configuration Mode (`C` Mode: 17 Bytes)
Sets simulation speed and size parameters:
- Header: `b'C\r\n'` (3 bytes)
- `speed`: `u32` Big-Endian (4 bytes, range `100`–`500`)
- `size`: `double` Big-Endian (8 bytes, range `10.0`–`100.0`)
- Delimiter: `b'\r\n'` (2 bytes)

---
## 3. Usage

### 3.1 Training
Train any supported algorithm:
```bash
uv run chase-the-dot --algo sac --timesteps 200000
uv run chase-the-dot --algo td3 --timesteps 200000
uv run chase-the-dot --algo ppo --timesteps 200000
```

### 3.2 Curriculum Learning with Domain Expansion (`--domain-expansion`)
Train an agent that masters the hardest configuration (`speed=100`, `size=10.0`) while maintaining generalization across the entire spectrum:
```bash
uv run chase-the-dot --algo sac --timesteps 200000 --domain-expansion
uv run chase-the-dot --algo td3 --timesteps 200000 --domain-expansion
```
- **Domain Expansion Schedule**: Starts at the easiest domain (`speed=500`, `size=70.0`) and gradually unlocks harder speeds and smaller target sizes (down to `speed=100`, `size=10.0`) over the first 70% of timesteps (customizable via `--curriculum-steps`).
- **Mastery + Generalization Mix**: Every interval (`--curriculum-interval 5000`), samples with 50% probability (`--curriculum-hard-ratio 0.5`) directly at the hardest unlocked frontier, and 50% uniformly across the full unlocked domain to prevent catastrophic forgetting.

### 3.3 Evaluation Mode (`--eval`)
Run any trained checkpoint deterministically (disables exploration noise and learning updates):
```bash
uv run chase-the-dot --algo sac --eval --timesteps 2000
uv run chase-the-dot --algo td3 --eval --timesteps 2000
uv run chase-the-dot --algo ppo --eval --timesteps 2000
```
You can also specify a custom checkpoint:
```bash
uv run chase-the-dot --algo sac --eval --model-path models/sac_latest.pt
```

---
## 4. Future Plans

### Algorithms
- ~~PID~~ (Completed)
- ~~VPG~~ (Completed)
- ~~A2C~~ (Completed)
- ~~PPO~~ (Completed)
- ~~DDPG~~ (Completed)
- ~~TD3~~ (Completed)
- ~~SAC~~ (Completed)

### Experimentational Future Plans
- **Detailed Logging & Visualization:** Expand logging metrics beyond basic rewards/losses to include advanced diagnostics like KL divergence, entropy loss, and critic explained variance for better algorithm debugging.
- **Frame Stacking:** Stack the $N$ most recent observations to give the agent a sense of velocity and acceleration.
- **Multi-step Action Prediction:** Train the network to predict a sequence of future actions to compensate for inference latency.
- **Hindsight Experience Replay (HER):** Combine HER with off-policy algorithms to massively improve sample efficiency by learning from failures via goal relabeling.
- **DreamerV3:** Explore world models and latent dynamics planning for a continuous tracking task.
- **Genetic Algorithms (GA):** Use symbolic regression or GA to evolve an explicit closed-form mathematical equation that deterministically solves the pathing problem.
- **Parallel Environments:** Wrap the TCP socket architecture in a vectorized environment (e.g., `SubprocVecEnv`) to gather experience from multiple application instances running on different ports simultaneously, massively increasing sample efficiency.

---
## 5. Issues & Troubleshooting

### Resolved Issues
- **Finding Environment Bounds:** The strict coordinate boundaries were unknown. **Solution:** Discovered the bounds by analyzing the environment outputs ($X \in [-50, 950]$ and $Y \in [-50, 950]$).
- **Observation Decoding:** The binary payload structure was undocumented. **Solution:** Successfully decoded the packet structure by cross-referencing the problem statement.
- **Uncorrelated Actions & Observations:** Actions appeared to have no immediate effect on the observations. **Solution:** Discovered the TCP connection operates in an open-drain streaming mode, meaning delayed receiving caused old states to pile up. Fixed by implementing a buffer-draining thread to ensure the agent always acts on the freshest state.
- **Misinterpreted `blue_x` / `blue_y` variables:** Initially assumed these represented the absolute screen coordinates of the Blue Dot. **Solution:** After analyzing the live values, determined they actually represent the *tracking error* ($\Delta X, \Delta Y$) between the Blue and Green dots.
- **Single Instance Limitation:** The LabVIEW environment executable (`Cy_RL_PS.exe`) natively restricted itself to a single instance, preventing parallel training across different ports. **Solution:** Discovered that appending `allowmultipleinstances = TRUE` to the adjacent `Cy_RL_PS.ini` configuration file overrides the LabVIEW runtime engine, successfully enabling multiple application instances and parallel training!

- **LabVIEW Error 56 / Inference Latency Drops:** The environment stream would routinely crash with a TCP Timeout (Error 56) due to perceived inference latency delaying the action payloads. **Solution:** Discovered that Python's default networking behavior (Nagle's Algorithm) was artificially buffering and delaying the tiny 15-byte action packets. Setting `socket.TCP_NODELAY` instantly transmitted the actions and completely eliminated the timeouts.
- **Tracking Offsets & Artificial Radius Misconception:** Previously, an artificial `size / (2 * pi)` offset was added to outgoing position commands. Socket probing on `Cy_RL_PS.exe` proved that LabVIEW evaluates tracking tolerances directly from exact mathematical coordinates: commanding `(gx, gy)` yields `0` error and 100% in-bounds rate, while adding `size / (2 * pi)` injected an artificial +8px error that triggered boundary faults (especially at small target sizes). Removed the artificial offset from `env.step()`.
- **SAC Floating Above Green Dot / Reward Inversion:** SAC policies previously exhibited an issue where the Blue Dot floated persistently above the Green Dot. This was caused by an inverted reward incentive: a discrete velocity penalty (`vel_dist / 300`) spiked due to 50 Hz packet jitter, severely penalizing in-bounds states (-1.5 to -118) compared to out-of-bounds states (-0.6). The agent learned that floating outside the boundary yielded higher return. Combined with screen space coordinate conventions ($Y$ increases downwards, error is positive when Blue is above Green), the agent parked above the target. Fixed by establishing a clean monotonic reward function ($1.0 - dist / 100$ when in-bounds, $-0.2 - dist / 50$ when out-of-bounds) and correcting SAC temperature tuning ($\alpha$), action squashing (`tanh`), and gradient clipping.
- **PyTorch Training Loop Bottlenecks:** The algorithms were suffering from PyTorch backend overhead. **Solution:** Used `line_profiler` and `kernprof` to identify three core bottlenecks and eliminated them:
  1. **Soft Updates:** Replaced slow parameter loops with `torch._foreach_lerp_` and eliminated the overhead of repeatedly calling `.parameters()` every step by pre-caching the parameter `list()` in `__init__`.
  2. **Optimizers:** Initialized Adam optimizers with `foreach=True` to utilize fused C++ vector operations for the backward pass.
  3. **Rollout Buffers (On-Policy):** Eliminated the massive overhead of `torch.stack()` on Python lists during rollouts by pre-allocating zero-allocation fixed-size tensor buffers (e.g. `obs_buffer = torch.zeros((64, *obs_shape))`) and filling them in-place during the environment loop.

### Unresolved Problems
- *(Currently none! All core architectural roadblocks have been conquered!)*
