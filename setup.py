"""Interactive setup wizard for SO-101 robot controller."""
from __future__ import annotations
import os
import re
import subprocess
import sys
from pathlib import Path

# Add the project venv's site-packages so the script works without activating first.
_site = next(Path(__file__).parent.glob("venv/lib/python*/site-packages"), None)
if _site and str(_site) not in sys.path:
    sys.path.insert(0, str(_site))
from pathlib import Path

import questionary
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
CONFIG_PATH = Path("config.yaml")
ENV_PATH = Path(".env")

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULTS = {
    "backend": {
        "type": "simulation",
        "simulation": {"model": "models/so101/pick_and_place_scene.xml", "viewer": False},
        "hardware": {"port": "/dev/ttyACM0", "id": "my_follower_arm"},
    },
    "llm": {"provider": "none", "model": "", "api_key_env": ""},
    "server": {"transport": "http", "port": 3001},
}

# CLI tools that handle their own auth — our server just needs to be running.
# API providers — we store the key and the user connects programmatically.
CLI_TOOLS = {
    "Claude Code":  "Already configured via .mcp.json — just run: claude",
    "Codex CLI":    "Run: codex  (picks up .mcp.json automatically)",
    "Gemini CLI":   "Run: gemini  (add server URL in ~/.gemini/config if needed)",
    "Cursor":       "Add server URL in Cursor → Settings → MCP Servers",
    "Other / any":  "Point your client at http://localhost:{port}/mcp",
}

API_PROVIDERS = {
    "Claude (Anthropic)": ("claude", "claude-sonnet-4-6", "ANTHROPIC_API_KEY"),
    "OpenAI / compatible": ("openai", "gpt-4o",           "OPENAI_API_KEY"),
    "Ollama (local)":      ("ollama", "llama3.2",          None),
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load() -> dict:
    if CONFIG_PATH.exists():
        return yaml.safe_load(CONFIG_PATH.read_text()) or {}
    return {}


def _save(cfg: dict) -> None:
    CONFIG_PATH.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))


def _write_env(key: str, value: str) -> None:
    lines = ENV_PATH.read_text().splitlines() if ENV_PATH.exists() else []
    lines = [l for l in lines if not l.startswith(f"{key}=")]
    lines.append(f"{key}={value}")
    ENV_PATH.write_text("\n".join(lines) + "\n")
    ENV_PATH.chmod(0o600)


def _masked(key: str) -> str:
    val = os.getenv(key, "")
    return val[:6] + "…" if len(val) > 6 else ("(not set)" if not val else val)


def _get(cfg: dict, *path, default=None):
    for k in path:
        if not isinstance(cfg, dict):
            return default
        cfg = cfg.get(k, {})
    return cfg if cfg != {} else default


# ── Section configurators ─────────────────────────────────────────────────────

def configure_backend(cfg: dict) -> dict:
    console.rule("[bold]Backend")

    choice = questionary.select(
        "Choose backend:",
        choices=["Simulation (MuJoCo)", "Real robot (SO-101 hardware)"],
        default="Simulation (MuJoCo)" if _get(cfg, "backend", "type") != "hardware" else "Real robot (SO-101 hardware)",
    ).ask()

    if choice is None:
        return cfg

    if "Simulation" in choice:
        current_model = _get(cfg, "backend", "simulation", "model",
                             default=DEFAULTS["backend"]["simulation"]["model"])
        current_viewer = _get(cfg, "backend", "simulation", "viewer", default=False)

        model = questionary.text("Scene XML path:", default=current_model).ask()
        viewer = questionary.confirm("Open live viewer window?", default=current_viewer).ask()

        cfg.setdefault("backend", {})
        cfg["backend"]["type"] = "simulation"
        cfg["backend"].setdefault("simulation", {})
        cfg["backend"]["simulation"]["model"] = model
        cfg["backend"]["simulation"]["viewer"] = viewer
    else:
        current_port = _get(cfg, "backend", "hardware", "port",
                            default=DEFAULTS["backend"]["hardware"]["port"])
        current_id = _get(cfg, "backend", "hardware", "id",
                          default=DEFAULTS["backend"]["hardware"]["id"])

        port = questionary.text("Serial port:", default=current_port).ask()
        robot_id = questionary.text(
            "Robot id (the --robot.id you calibrated with):", default=current_id).ask()

        console.print(
            f"  Calibration is loaded by lerobot from its cache for id '{robot_id}'. Run "
            f"`lerobot-calibrate --robot.type=so101_follower --robot.port={port} "
            f"--robot.id={robot_id}` first if you have not.")

        cfg.setdefault("backend", {})
        cfg["backend"]["type"] = "hardware"
        cfg["backend"].setdefault("hardware", {})
        cfg["backend"]["hardware"]["port"] = port
        cfg["backend"]["hardware"]["id"] = robot_id

    return cfg


def configure_llm(cfg: dict) -> dict:
    console.rule("[bold]LLM Connection")

    mode = questionary.select(
        "How will you connect an LLM?",
        choices=[
            "CLI tool  (Claude Code, Codex CLI, Gemini CLI, Cursor…)  — no key needed",
            "API  (provide a key, connect programmatically)",
        ],
        default="CLI tool  (Claude Code, Codex CLI, Gemini CLI, Cursor…)  — no key needed"
        if _get(cfg, "llm", "provider", default="cli") in ("cli", "none", "")
        else "API  (provide a key, connect programmatically)",
    ).ask()

    if mode is None:
        return cfg

    cfg.setdefault("llm", {})

    if "CLI" in mode:
        cfg["llm"]["provider"] = "cli"
        cfg["llm"]["model"] = ""

        port = _get(cfg, "server", "port", default=3001)
        cli = questionary.select(
            "Which CLI tool? (just for setup instructions — any MCP client works)",
            choices=list(CLI_TOOLS.keys()),
        ).ask()

        if cli:
            instruction = CLI_TOOLS[cli].replace("{port}", str(port))
            console.print(f"\n  [green]OK[/green] {instruction}\n")
        return cfg

    # API path
    choice = questionary.select(
        "Choose API provider:",
        choices=list(API_PROVIDERS.keys()),
        default=next(
            (k for k, (p, _, _) in API_PROVIDERS.items()
             if p == _get(cfg, "llm", "provider", default="")),
            list(API_PROVIDERS.keys())[0],
        ),
    ).ask()

    if choice is None:
        return cfg

    provider, default_model, env_key = API_PROVIDERS[choice]
    cfg["llm"]["provider"] = provider

    current_model = _get(cfg, "llm", "model", default=default_model) or default_model
    model = questionary.text("Model name:", default=current_model).ask()
    cfg["llm"]["model"] = model

    if provider == "ollama":
        current_url = _get(cfg, "llm", "base_url", default="http://localhost:11434")
        url = questionary.text("Ollama base URL:", default=current_url).ask()
        cfg["llm"]["base_url"] = url
        return cfg

    if provider == "openai":
        current_url = _get(cfg, "llm", "base_url", default="https://api.openai.com/v1")
        url = questionary.text("API base URL:", default=current_url).ask()
        cfg["llm"]["base_url"] = url

    cfg["llm"]["api_key_env"] = env_key
    console.print(f"\n  Current [dim]{env_key}[/dim]: {_masked(env_key)}")
    key_action = questionary.select(
        "API key:",
        choices=["Enter / update key", f"Keep existing ({env_key})", "Skip"],
    ).ask()

    if key_action == "Enter / update key":
        key = questionary.password(f"{env_key}:").ask()
        if key:
            _write_env(env_key, key)
            console.print("  [green]OK[/green] Saved to .env")

    return cfg


def configure_server(cfg: dict) -> dict:
    console.rule("[bold]Server")

    current_transport = _get(cfg, "server", "transport", default="http")
    current_port = _get(cfg, "server", "port", default=3001)

    transport = questionary.select(
        "Transport:",
        choices=["http", "stdio"],
        default=current_transport,
    ).ask()

    if transport is None:
        return cfg

    cfg.setdefault("server", {})
    cfg["server"]["transport"] = transport

    if transport == "http":
        port = questionary.text("Port:", default=str(current_port)).ask()
        cfg["server"]["port"] = int(port)
        _update_mcp_json(int(port))

    return cfg


def _update_mcp_json(port: int) -> None:
    import json
    mcp_path = Path(".mcp.json")
    content = {"mcpServers": {"so101-sim": {"type": "http", "url": f"http://localhost:{port}/mcp"}}}
    mcp_path.write_text(json.dumps(content, indent=2) + "\n")


# ── Summary ───────────────────────────────────────────────────────────────────

def _show_summary(cfg: dict) -> None:
    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_column(style="dim")
    t.add_column()

    bt = _get(cfg, "backend", "type", default="simulation")
    if bt == "simulation":
        bval = f"Simulation — {_get(cfg, 'backend', 'simulation', 'model', default='—')}"
        if _get(cfg, "backend", "simulation", "viewer"):
            bval += "  [dim](viewer on)[/dim]"
    else:
        bval = f"Hardware — {_get(cfg, 'backend', 'hardware', 'port', default='—')}"

    provider = _get(cfg, "llm", "provider", default="cli")
    model = _get(cfg, "llm", "model", default="") or ""
    if provider == "cli":
        llm_val = "CLI tool (Claude Code / Codex / Gemini / …)"
    elif provider in ("none", ""):
        llm_val = "none — connect manually"
    else:
        llm_val = f"{provider} / {model}" if model else provider

    transport = _get(cfg, "server", "transport", default="http")
    port = _get(cfg, "server", "port", default=3001)
    srv_val = f"{transport}" + (f"  →  http://localhost:{port}/mcp" if transport == "http" else "")

    t.add_row("Backend", bval)
    t.add_row("LLM", llm_val)
    t.add_row("Server", srv_val)

    console.print(Panel(t, title="[green]Config saved[/green]", border_style="green"))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    console.print(Panel.fit(
        "[bold]SO-101 Robot Controller — Setup[/bold]",
        border_style="blue",
    ))

    cfg = _load()

    if cfg:
        t = Table(show_header=False, box=None, padding=(0, 2))
        t.add_column(style="dim")
        t.add_column()
        bt = _get(cfg, "backend", "type", default="—")
        t.add_row("Backend", bt)
        _p = _get(cfg, "llm", "provider", default="cli")
        _m = _get(cfg, "llm", "model", default="") or ""
        _llm = "CLI tool" if _p == "cli" else (f"{_p} / {_m}" if _m else _p or "—")
        t.add_row("LLM", _llm)
        t.add_row("Server", f"{_get(cfg, 'server', 'transport', default='—')} / port {_get(cfg, 'server', 'port', default='—')}")
        console.print(Panel(t, title="Current config", border_style="dim"))

        section = questionary.select(
            "What would you like to configure?",
            choices=[
                "Everything",
                "Backend only",
                "LLM provider only",
                "Server settings only",
                "Nothing — exit",
            ],
        ).ask()

        if section is None or section == "Nothing — exit":
            return

        if section in ("Everything", "Backend only"):
            cfg = configure_backend(cfg)
        if section in ("Everything", "LLM provider only"):
            cfg = configure_llm(cfg)
        if section in ("Everything", "Server settings only"):
            cfg = configure_server(cfg)
    else:
        cfg = configure_backend(cfg)
        cfg = configure_llm(cfg)
        cfg = configure_server(cfg)

    _save(cfg)
    _show_summary(cfg)

    if questionary.confirm("\nStart MCP server now?", default=True).ask():
        subprocess.run([sys.executable, "main.py"])


if __name__ == "__main__":
    main()
