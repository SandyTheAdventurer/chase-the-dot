# Chase the Dot

Real-time TCP tracking agent and Reinforcement Learning environment for the **CynLr "Chase the Dot"** control benchmark.

---

## 1. Problem Statement & System Understanding

Based on the official [problem-statement.pdf](problem-statement.pdf) and reverse-engineering of the live binary application (`Cy_RL_PS.exe`):

### 1.1 Objective
The benchmark challenges an external agent to control a **Blue Dot** in real time by sending target position commands over TCP. The goal is to track an autonomously moving **Green Dot** as closely as possible, minimizing position error $(\Delta X, \Delta Y)$ and keeping the error percentage below the required target ($< 1\%$ error rate).

### 1.2 Environment Details
- **Coordinate Space Bounds**: $X \in [-50, 950]$ and $Y \in [-50, 750]$. Target commands must be clamped to these bounds to avoid the "Blue out of screen bound" fault.
- **Observation State (`state` variable)**: The `receive_state()` function returns a 7-dimensional `float32` numpy array:
  1. `state[0]`: **Green X** (Target to follow)
  2. `state[1]`: **Green Y** 
  3. `state[2]`: **Tracking Error X** (`Blue_X - Green_X`). Positive means Blue is to the right of Green.
  4. `state[3]`: **Tracking Error Y** (`Blue_Y - Green_Y`). Positive means Blue is below Green.
  5. `state[4]`: **Error Flag X** (1.0 if Out of Bounds, 0.0 if In Bounds)
  6. `state[5]`: **Error Flag Y** (1.0 if Out of Bounds, 0.0 if In Bounds)
  7. `state[6]`: **Delta Time (dt)** in seconds since the last packet received.
- **RL Observation Space**: `env.step()` and `env.reset()` return an observation where the first 4 elements are scaled down by `1000.0` to normalize them for neural networks (e.g. VPG/DQN).

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
## 3. Future Plans

### Algorithms
- ~~PID~~ (Completed)
- ~~VPG~~ (Completed)
- PPO
- DDPG
- TD3
- SAC

### Experimentational Future Plans
- **DreamerV3:** Explore world models and latent dynamics planning for a continuous tracking task.
- **Genetic Algorithms (GA):** Use symbolic regression or GA to evolve an explicit closed-form mathematical equation that deterministically solves the pathing problem.
- **Parallel Environments:** Wrap the TCP socket architecture in a vectorized environment (e.g., `SubprocVecEnv`) to gather experience from multiple application instances running on different ports simultaneously, massively increasing sample efficiency.
