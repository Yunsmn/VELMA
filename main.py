"""Start the SO-101 MCP server with the configured backend."""
from __future__ import annotations
import logging
import os
import sys
import threading
import time
from pathlib import Path

# Add the project venv's site-packages so the script works without activating first.
# Only when it matches the RUNNING interpreter: the hardware backend needs lerobot
# (and so torch), which has no Python 3.14 wheels, so that backend runs under a 3.12
# interpreter instead. Injecting a 3.14 site-packages there would put extension
# modules built for the wrong ABI ahead of the real ones on sys.path.
_site = Path(__file__).parent / (
    f"venv/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
)
if _site.is_dir() and str(_site) not in sys.path:
    sys.path.insert(0, str(_site))

# The offscreen renderer (render_side/render_wrist, find_object, the run recorder)
# needs an EGL GL context when running headless. setdefault so an explicit
# MUJOCO_GL (e.g. 'glfw' for a desktop with the interactive viewer) still wins.
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import mujoco.viewer
import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

# Keep uvicorn/fastmcp logs simple — their default Rich handler wraps badly.
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

console = Console()


def _load_config() -> dict:
    cfg_path = Path("config.yaml")
    if not cfg_path.exists():
        console.print("[red]config.yaml not found. Run [bold]python setup.py[/bold] first.[/red]")
        sys.exit(1)
    return yaml.safe_load(cfg_path.read_text())


def main() -> None:
    load_dotenv()
    cfg = _load_config()

    backend_cfg = cfg["backend"]
    server_cfg  = cfg.get("server", {})
    llm_cfg     = cfg.get("llm", {})

    backend_type = backend_cfg.get("type", "simulation")
    # The MuJoCo model differs by backend and must not be shared. In simulation it
    # is the scene being manipulated; on hardware it is used for kinematics only,
    # so it has to be the arm alone — a scene model would populate the kinematic
    # world with a cube and container the real table does not contain.
    if backend_type == "hardware":
        model_path = backend_cfg.get("hardware", {}).get(
            "model", "models/so101/so101_new_calib.xml")
    else:
        model_path = backend_cfg.get("simulation", {}).get(
            "model", "models/so101/pick_and_place_scene.xml")
    # Allow a test harness to pick the scene without editing config.yaml.
    model_path   = os.environ.get("SO101_SCENE", model_path)
    open_viewer  = backend_type == "simulation" and backend_cfg.get("simulation", {}).get("viewer", False)
    # Per-run override, so switching between "watch the arm" and "run perception" does not
    # mean editing config.yaml (or re-running setup.py) every time. The two are mutually
    # exclusive on this machine: the viewer owns the GL context and the offscreen cameras
    # come back corrupted, so perception tools are refused while it is open.
    #   SO101_VIEWER=0  -> headless: find_object / refine_grasp_point work, record to mp4
    #   SO101_VIEWER=1  -> viewer: watch live, perception disabled
    _viewer_env = os.environ.get("SO101_VIEWER")
    if _viewer_env is not None:
        open_viewer = _viewer_env.strip() not in ("0", "false", "False", "no", "")

    model = mujoco.MjModel.from_xml_path(model_path)
    data  = mujoco.MjData(model)

    from kinematics.ik import IKController
    ik = IKController(model, data, end_effector_site="gripperframe")

    if backend_type == "simulation":
        from robot.backends.simulation import SimulationBackend
        backend = SimulationBackend(model, data)
    else:
        hw = backend_cfg.get("hardware", {})
        from robot.backends.hardware import HardwareBackend
        backend = HardwareBackend(model, data,
                                  port=hw.get("port", "/dev/ttyACM0"),
                                  robot_id=hw.get("id", "my_follower_arm"))

    backend.reset()

    from robot.controller import RobotController
    controller = RobotController(backend, ik)

    port      = int(server_cfg.get("port", 3001))
    # Per-run override (mirrors SO101_VIEWER/SO101_SCENE above), so a script that needs
    # its own dedicated port (e.g. a benchmark harness that spawns its own server
    # instance) doesn't have to edit config.yaml.
    port      = int(os.environ.get("SO101_PORT", port))
    transport = server_cfg.get("transport", "http")

    from server.server import create_server
    mcp_server = create_server(controller, port=port)

    provider = llm_cfg.get("provider", "cli")
    model_name = llm_cfg.get("model", "") or "—"
    llm_label = "CLI tool" if provider == "cli" else f"{provider} / {model_name}"

    console.print(Panel.fit(
        f"[bold]SO-101 Robot Controller[/bold]\n\n"
        f"  Backend   : [cyan]{backend_type}{'  (viewer on)' if open_viewer else ''}[/cyan]\n"
        f"  Transport : [cyan]{transport}"
        + (f"  →  http://127.0.0.1:{port}/mcp" if transport == "http" else "")
        + f"[/cyan]\n"
        f"  LLM       : [cyan]{llm_label}[/cyan]\n\n"
        f"[dim]Waiting for connections…  Ctrl+C to stop.[/dim]",
        title="[green]●[/green] Running",
        border_style="green",
    ))

    def _run_server() -> None:
        if transport == "http":
            mcp_server.run(transport="streamable-http")
        else:
            mcp_server.run(transport="stdio")

    if open_viewer:
        # MCP server runs in a background thread; viewer owns the main thread.
        t = threading.Thread(target=_run_server, daemon=True)
        t.start()

        # The old line here was `backend._renderer = None`, intended to stop the viewer
        # and the offscreen renderer fighting over a GL context. It did nothing:
        # `_renderer` is the generic capture renderer and is ALREADY None by default
        # (simulation.py), while the perception cameras use a SEPARATE `_offscreen`
        # renderer that the line never touched. So perception was not actually disabled
        # under the viewer — it was merely untested, and a context clash would surface as
        # a crash on the server's worker thread.
        #
        # MEASURED on this machine (Wayland + glfw): with the viewer running, the offscreen
        # renderer does NOT raise — it silently returns BLANK frames (mean pixel 0.6 vs 90.5
        # headless). Perception therefore reports "nothing detected" and looks broken when
        # it simply never received a picture. simulation._check_frame now detects that and
        # says so once, rather than letting it look like a detector failure.
        #
        # Perception is left enabled (it may work on other drivers, e.g. X11), but the
        # warning below sets expectations. SO101_NO_OFFSCREEN=1 forces it off.
        # The viewer and the offscreen renderer cannot share the GL context on this setup, and
        # the failure is SILENT: frames come back black, or dim with viewer overlay bleeding
        # in, and perception then reports "no detection" as if the detector were at fault.
        # So the cameras are refused outright while the viewer runs, with an explicit message,
        # instead of returning images that merely look plausible.
        type(backend)._viewer_active = True
        console.print(
            "[yellow]NOTE:[/yellow] perception cameras ([cyan]find_object[/cyan], "
            "[cyan]refine_grasp_point[/cyan], [cyan]capture_cameras[/cyan]) are "
            "DISABLED while the viewer is running —\n"
            "      the viewer owns the GL context and offscreen renders come back corrupted. "
            "They will report a clear error.\n"
            "      For perception, restart headless: "
            "[dim]config.yaml viewer: false  +  MUJOCO_GL=egl[/dim]")

        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                with backend._step_lock:
                    viewer.sync()
                time.sleep(0.005)  # ~200 Hz
    else:
        _run_server()


if __name__ == "__main__":
    main()
