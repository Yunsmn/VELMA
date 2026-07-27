# VELMA

Vision-Enabled LLM Manipulation Arm. VELMA is a tabletop SO-101 arm in MuJoCo that runs a full
pick-and-place from vision. A fixed side camera locates the object to within a centimetre or so,
and the wrist camera orbits that seed over five views and triangulates it to about 1 mm. The arm
grasps and places from those coordinates, and can be driven by an LLM over MCP, or by hand from a
terminal control panel.

## What's here

```
so101-Models/
  main.py            entry point: loads config.yaml, starts the MCP server (or hardware backend)
  config.yaml        runtime config (backend, viewer on/off, port, LLM provider)
  requirements.txt   robot venv deps
  server/            the MCP server — 15 tools (see below)
  perception/        find_object (coarse) + wrist triangulation + the sidecar client
    requirements.txt   perception sidecar venv deps (torch CPU + ultralytics + transformers)
  percept_venv/      torch/SAM/Depth-Anything sidecar interpreter (CPU)
  wrist_refine.py    wrist-camera hover/state helpers used by perception/wrist_triangulate.py
  robot/, kinematics/, models/   backend, IK, MuJoCo scenes
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

It connects to the running server, lists the 15 tools with their live schemas, and lets you call
any of them with guided input. **Any MCP client must set a long read timeout** (the panel uses
`read_timeout_seconds=300`) — `grasp`/`place`/`stop_recording` take several seconds, and a client
that times out mid-call takes the server down with it.

## Connect an LLM

With the headless server running (above), pick one of two paths:

**A CLI tool that speaks MCP** (Claude Code, Codex CLI, Gemini CLI, Cursor, ...): this repo's
`.mcp.json` already points at `http://localhost:3001/mcp` — just run the CLI tool from inside
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

## The 15 tools

Perception: `find_object`, `triangulate`. State: `get_robot_state`, `capture_cameras`,
`get_initial_instructions`. Manipulation: `grasp`, `place`. Motion: `move_to_position`,
`set_joint_angles`. Gripper: `set_gripper`, `open_gripper`, `close_gripper`. Harness:
`reset_scene`, `start_recording`, `stop_recording`.

## Honesty note

The perception path (`find_object` → `triangulate`) is measurement-only — nothing in it reads
ground truth. `get_robot_state` and every tool's returned state deliberately **omit** the cube's
and container's true positions, so an LLM cannot shortcut perception by reading the answer out of
state (a benchmark harness that needs ground truth for scoring can opt back in by starting the
server with `SO101_EXPOSE_TRUTH=1` — never set this when an LLM is connected).

Two internal reads of ground truth remain inside `grasp` itself (object shape for the never-bat
decline, and a position drift-abort guard) and are pending removal. Full audit, exact locations,
and severity: [`research/HONESTY_REVIEW.md`](../research/HONESTY_REVIEW.md).
