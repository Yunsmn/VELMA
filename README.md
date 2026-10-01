# VELMA

Vision-Enabled LLM Manipulation Arm. VELMA is a tabletop SO-101 arm that runs a full
pick-and-place from vision, in MuJoCo or on the real arm. A fixed side camera locates the object to within a centimetre or so,
and the wrist camera orbits that seed over five views and triangulates it to about 1 mm. The arm
grasps and places from those coordinates, and can be driven by an LLM over MCP, or by hand from a
terminal control panel. The same server also drives a physical SO-101 through lerobot (see
[Real SO-101 arm](#real-so-101-arm)).

## What's here

```
so101-Models/
  main.py            entry point: loads config.yaml, starts the MCP server (or hardware backend)
  config.yaml        runtime config (backend, viewer on/off, port, LLM provider)
  requirements.txt   robot venv deps
  server/            the MCP server (see "Tools" below)
  perception/        find_object (coarse) + wrist triangulation + the sidecar client
    requirements.txt   perception sidecar venv deps (torch CPU + ultralytics + transformers)
  percept_venv/      torch/SAM/Depth-Anything sidecar interpreter (CPU)
  wrist_refine.py    wrist-camera hover/state helpers used by perception/wrist_triangulate.py
  robot/, kinematics/, models/   backends (simulation + hardware), IK, MuJoCo scenes
  poses.json         named arm poses taught on the real arm (save_pose / goto_pose)
  control_panel.py   terminal UI to drive the arm by hand through the MCP server
  agent.py           thin MCP client that connects any LLM (Ollama/OpenAI-compatible) to the arm
  SKILL.md           system-prompt guide for an LLM driving the arm (fed in by agent.py)
  .mcp.json          MCP server entry for CLI tools (Claude Code, Codex CLI, ...)
  venv/              robot interpreter (MuJoCo, no torch)
```

## Install

The two virtual environments are the only heavy pieces. Recreate them from the pinned deps
(run from inside `so101-Models/`):

```bash
# robot venv (renders MuJoCo, runs the MCP server — no torch)
python3 -m venv venv && ./venv/bin/pip install -r requirements.txt

# perception sidecar venv (torch CPU + ultralytics + transformers; needs Python 3.11/3.12)
python3.11 -m venv percept_venv && ./percept_venv/bin/pip install -r perception/requirements.txt
```

Then, optionally, run the setup wizard to write `config.yaml` (backend, LLM connection, server
port) — it's a config helper, not part of the install:

```bash
./venv/bin/python setup.py
```

A committed `config.yaml` with sane defaults already ships in this repo, so `setup.py` is only
needed to change something (e.g. switch to the hardware backend, or add an API key).

## Run the server (headless — required for perception)

Perception needs offscreen rendering, so run the server **without** the viewer:

```bash
SO101_VIEWER=0 MUJOCO_GL=egl ./venv/bin/python main.py
```

`SO101_SCENE=models/so101/<scene>.xml` overrides the scene (default `pick_and_place_scene.xml`).
Set `SO101_VIEWER=1` to watch the arm instead — but perception is then disabled by design (the
offscreen and on-screen renderers cannot coexist on this box).

## Drive it by hand

In a second terminal:

```bash
./venv/bin/python control_panel.py
```

It connects to the running server, lists the tools with their live schemas, and lets you call
any of them with guided input. **Any MCP client must set a long read timeout** (the panel uses
`read_timeout_seconds=300`) — `grasp`/`place`/`stop_recording` take several seconds, and a client
that times out mid-call takes the server down with it.

## Connect an LLM

With the headless server running (above), pick one of two paths:

**A CLI tool that speaks MCP** (Claude Code, Codex CLI, Gemini CLI, Cursor, ...): this repo's
`.mcp.json` already points at `http://127.0.0.1:3001/mcp` — just run the CLI tool from inside
`so101-Models/` (e.g. `claude`). If you changed `config.yaml`'s `server.port`, re-run
`./venv/bin/python setup.py` (Server settings) to regenerate `.mcp.json` to match.

**Any Ollama / OpenAI-compatible model**, via the bundled thin agent:

```bash
./venv/bin/python agent.py --model gemma3:4b --task "pick up the red cube and place it in the container"
```

`agent.py` feeds the model `SKILL.md` as a system prompt plus the *live* tool schemas from the
server (never hard-coded, so it can't drift from what the server actually exposes), and relays
tool calls both ways — it doesn't inject a procedure or second-guess the model. See `agent.py
--help` for endpoint/model flags (`LLM_BASE_URL`, `LLM_API_KEY`, `--mode auto|native|text`).

## The pick-and-place path (what an LLM calls)

The object's position is **measured**, never given:

1. `find_object("the red cube")` — coarse `(x, y)` from the fixed side camera (~cm, enough to aim).
2. `triangulate("the red cube", x, y)` — refine to a grasp-ready 3D point with the wrist camera
   (~1 mm). If `confidence < 0.6` the views disagree — re-run `find_object` or decline.
3. `grasp(x, y, z, trust_coords=true, grip_width_mm=width_mm+10)` — gentle, never-bat top grasp.
4. `find_object("the container")` — the container's position is measured too, not assumed.
5. `place(container_x, container_y, 0.05)` — put it down; check `cube_in_container`.

## Tools

Fifteen tools work on both backends:

Perception: `find_object`, `triangulate`. State: `get_robot_state`, `capture_cameras`,
`get_initial_instructions`. Manipulation: `grasp`, `place`. Motion: `move_to_position`,
`set_joint_angles`. Gripper: `set_gripper`, `open_gripper`, `close_gripper`. Harness:
`reset_scene`, `start_recording`, `stop_recording`.

Eighteen more are for the real arm and answer `unsupported` in simulation:

Motion: `move_to_xyz`, `check_reachable`, `goto_servo_angles`, `jog_joint`,
`sweep_joint_to_limit`, `joint_limits`. Poses: `save_pose`, `goto_pose`, `list_poses`.
Hand guiding: `set_torque`, `observe_hand_motion`. Gripper: `close_on_object`, `release_object`,
`clear_gripper_overload`. Wrist camera: `find_object_3d`, `detect_in_wrist_view`,
`center_object_in_view`, `capture_with_pose`.

## Real SO-101 arm

The hardware backend (`robot/backends/hardware.py`) drives a physical SO-101 follower through
[lerobot](https://github.com/huggingface/lerobot). The MuJoCo model is still loaded, but only for
kinematics. Joint signs and offsets are measured, so base-frame coordinates mean the same thing on
both backends: x forward, y left, z up, table at z = 0.

**Install.** lerobot needs torch, which has no Python 3.14 wheels, so the hardware backend runs
under Python 3.12 in its own environment:

```bash
uv venv --python 3.12 hw_venv && source hw_venv/bin/activate
uv pip install -r requirements.txt "lerobot[core_scripts]"
```

Install only `[core_scripts]`. `lerobot[all]` pulls in training dependencies worth several GB.

**Serial access.** On Arch the port belongs to the `uucp` group; on Debian/Ubuntu it is `dialout`.
Add yourself (`sudo usermod -aG uucp $USER`) and log out and back in.

**Calibrate once**, by robot id (lerobot stores it under `~/.cache/huggingface/lerobot/`):

```bash
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_follower_arm
```

Never run lerobot with `sudo`. It then looks in `/root/.cache`, finds no calibration and silently
recalibrates. A working calibration is worth backing up, since a bad sweep can leave a joint, most
often the gripper, with an unusable range.

**Configure and run.** Run `python setup.py`, choose *Real robot*, and give the port and robot id.
A `/dev/serial/by-id/...` path survives replugging; a plain `/dev/ttyACM0` can be renamed by the
kernel. Then start the server as usual with `python main.py`.

**Things that differ from simulation, on purpose:**

- The arm does not move at startup or on `reset_scene`. Bring it to a known posture with a pose
  you taught (`save_pose`, then `goto_pose`).
- Torque stays on when the server stops, so the arm does not drop. Use `set_torque(false)` from a
  low pose to let it go limp.
- Multi-joint moves drive one axis at a time, with a few passes and a small droop correction,
  because the loaded joints stall when every joint is commanded at once. Expect a staircase path
  rather than a straight line.
- The column directly above the base (x ≤ 0.10 m, |y| ≤ 0.05 m, z ≥ 0.05 m) is unreachable. Keep
  Cartesian targets at x ≥ 0.15 m and call `check_reachable` before `move_to_xyz`.

**Wrist camera grounding.** `find_object_3d` uses Falcon, which runs in a separate CUDA
environment reached through the perception sidecar:

```bash
python3.11 -m venv percept_gpu_venv && ./percept_gpu_venv/bin/pip install -r perception/requirements-gpu.txt
```

Set `SO101_WRIST_CAM=<index>` if the wrist camera is not found automatically.

**Tuning** (environment variables, defaults in brackets): `SO101_MOVE_TIMEOUT_S` [60],
`SO101_SEQUENTIAL_PASSES` [4], `SO101_DROOP_ROUNDS` [3], `SO101_DROOP_MAX_DEG` [10],
`SO101_MAX_REL_TARGET_DEG` [20], `SO101_POSITION_P` / `_I` / `_D` [32 / 0 / 32].

The hardware path is newer than the simulation and less tested. Grasping is reliable in simulation;
on the real arm, verify each step with the robot in view.

## Honesty note

The perception path (`find_object` → `triangulate`) is measurement-only — nothing in it reads
ground truth. `get_robot_state` and every tool's returned state deliberately **omit** the cube's
and container's true positions, so an LLM cannot shortcut perception by reading the answer out of
state (a benchmark harness that needs ground truth for scoring can opt back in by starting the
server with `SO101_EXPOSE_TRUTH=1` — never set this when an LLM is connected).

Two internal reads of ground truth remain inside `grasp` itself (object shape for the never-bat
decline, and a position drift-abort guard) and are pending removal. Full audit, exact locations,
and severity are recorded in the project's research notes, outside this repository.
