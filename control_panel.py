"""SO-101 MCP control panel (TUI).

Fire any MCP tool by hand, or hand a task to an LLM. The tool list is pulled live from the
server (`list_tools`), so it always reflects whatever tools the server exposes: names,
descriptions, and each parameter's type/description come straight from the MCP schema,
nothing is hard-coded here.

Run the server HEADLESS in one terminal (required for find_object/triangulate/capture_cameras):
    cd so101-Models && SO101_VIEWER=0 MUJOCO_GL=egl ./venv/bin/python main.py
then this panel in another:
    cd so101-Models && ./venv/bin/python control_panel.py

If you'd rather watch the arm move live and don't need perception, SO101_VIEWER=1 opens a
viewer window instead — but the perception tools are then disabled by design (the on-screen
and offscreen renderers can't share a GL context on this box); see README.md.
"""
from __future__ import annotations
import asyncio, base64, json, os, subprocess, sys, traceback
from datetime import timedelta
from pathlib import Path

# make the venv importable without activating
_site = next(Path(__file__).parent.glob("venv/lib/python*/site-packages"), None)
if _site and str(_site) not in sys.path:
    sys.path.insert(0, str(_site))

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

console = Console()
MCP_URL = os.environ.get("MCP_URL", "http://localhost:3001/mcp")
# Long enough for the slowest tool (a full grasp, or a wrist orbit with CPU
# inference). Overridable for debugging.
READ_TIMEOUT = timedelta(seconds=float(os.environ.get("MCP_READ_TIMEOUT", "300")))


def _params(tool):
    schema = getattr(tool, "inputSchema", None) or {}
    return schema.get("properties", {}) or {}, set(schema.get("required", []) or [])


def _describe(tool):
    props, required = _params(tool)
    console.print(Panel.fit(f"[bold]{tool.name}[/bold]\n{(tool.description or '').strip()}",
                            title="tool", border_style="cyan"))
    if props:
        t = Table(show_header=True, header_style="bold cyan", box=None, pad_edge=False)
        for col in ("param", "type", "req", "default", "description"):
            t.add_column(col)
        for name, spec in props.items():
            t.add_row(name, str(spec.get("type", "")), "•" if name in required else "",
                      "" if spec.get("default") is None else str(spec.get("default")),
                      str(spec.get("description", ""))[:70])
        console.print(t)


async def _ask_args(tool):
    # NOTE: every prompt below MUST use questionary's async `.ask_async()`. These run
    # inside the MCP client's running event loop, and the synchronous `.ask()` cannot
    # drive prompt_toolkit from within one — it returns an un-awaited coroutine and the
    # panel dies with "coroutine 'Application.run_async' was never awaited".
    props, required = _params(tool)
    args = {}
    for name, spec in props.items():
        typ = spec.get("type", "string")
        req = name in required
        default = spec.get("default")
        label = f"{name}" + (" *" if req else "") + (f"  [{typ}]" if typ else "")
        if "enum" in spec:
            val = await questionary.select(label, choices=[str(e) for e in spec["enum"]]).ask_async()
            if val is None:
                return None
        elif typ == "boolean":
            val = await questionary.confirm(label, default=bool(default)).ask_async()
        else:
            raw = await questionary.text(
                label, default="" if default is None else str(default)).ask_async()
            if raw is None:
                return None
            if raw.strip() == "":
                if req:
                    console.print(f"[yellow]{name} is required — using 0/empty[/yellow]")
                else:
                    continue
            if typ in ("number", "integer") and raw.strip() != "":
                try:
                    val = int(float(raw)) if typ == "integer" else float(raw)
                except ValueError:
                    console.print(f"[red]bad number for {name}[/red]"); return None
            else:
                val = raw
        args[name] = val
    return args


def _show(res):
    imgs = 0
    for c in getattr(res, "content", []):
        if hasattr(c, "text"):
            try:
                console.print_json(c.text)
            except Exception:
                console.print(c.text[:1500])
        elif hasattr(c, "data"):
            out = f"/tmp/cp_cam{imgs}.jpg"
            try:
                open(out, "wb").write(base64.b64decode(c.data)); imgs += 1
                console.print(f"[dim]camera image saved -> {out}[/dim]")
            except Exception:
                pass


async def _run_llm():
    task = await questionary.text("Task for the LLM:").ask_async()
    if not task:
        return
    model = os.environ.get("LLM_MODEL", "gemma3:4b")
    console.print(f"[cyan]LLM ({model}) driving via agent.py:[/cyan] {task}")
    subprocess.run([sys.executable, str(Path(__file__).parent / "agent.py"), "--model", model, "--task", task])


async def loop(session, tools):
    while True:
        choice = await questionary.select(
            "MCP tool to run:",
            choices=[t.name for t in tools] + ["— refresh tools", "— LLM mode", "— quit"],
        ).ask_async()
        if choice in (None, "— quit"):
            break
        if choice == "— refresh tools":
            tools = (await session.list_tools()).tools
            continue
        if choice == "— LLM mode":
            await _run_llm()
            continue
        tool = next(t for t in tools if t.name == choice)
        _describe(tool)
        args = await _ask_args(tool)
        if args is None:
            continue
        console.print(f"[dim]→ {tool.name}({json.dumps(args)})[/dim]")
        try:
            _show(await session.call_tool(tool.name, args))
        except Exception as e:
            console.print(f"[red]call failed:[/red] {e}")


def _is_server_gone(exc) -> bool:
    """True if this exception means the SERVER died, not that the panel misbehaved.

    A dead server surfaces as a connect/read error buried inside anyio's ExceptionGroup,
    so the whole tree has to be walked — the top-level type alone says nothing useful.
    """
    seen, stack = set(), [exc]
    while stack:
        e = stack.pop()
        if id(e) in seen:
            continue
        seen.add(id(e))
        name = type(e).__name__
        if name in ("ConnectError", "ReadError", "RemoteProtocolError", "ReadTimeout",
                    "ConnectTimeout", "ClosedResourceError", "IncompleteRead"):
            return True
        stack.extend(getattr(e, "exceptions", None) or [])
        if getattr(e, "__cause__", None) is not None:
            stack.append(e.__cause__)
        if getattr(e, "__context__", None) is not None:
            stack.append(e.__context__)
    return False


async def main():
    console.print(Panel.fit("SO-101 · MCP Control Panel", style="bold green"))
    # `connected` distinguishes a genuine connection failure from a crash INSIDE the
    # session. Previously one try/except wrapped both, so any bug in the menu was
    # reported as "could not reach the MCP server" — pointing at a healthy server
    # while the real fault was local.
    connected = False
    try:
        async with streamablehttp_client(MCP_URL) as (r, w, _):
            # A generous read timeout is REQUIRED, not a nicety. Several tools run for many
            # seconds — grasp drives a whole descent/close/test-lift, refine_grasp_point flies
            # a 5-pose orbit with CPU inference, stop_recording encodes an mp4 — and the
            # library default is far shorter. Worse, a client that gives up mid-call takes the
            # server down with it, so a timeout does not just fail the call, it kills the
            # session and loses the result. Both were hit in practice.
            async with ClientSession(r, w, read_timeout_seconds=READ_TIMEOUT) as s:
                await s.initialize()
                tools = (await s.list_tools()).tools
                connected = True
                console.print(f"[green]connected[/green] · {len(tools)} tools · {MCP_URL}\n")
                await loop(s, tools)
    except Exception as e:
        if connected and _is_server_gone(e):
            # The server DIED mid-session. Saying "the server is fine" here (as this branch
            # used to, unconditionally) sends you hunting for a panel bug while the process is
            # actually dead — it fired twice during a grasp and a stop_recording that had in
            # fact taken the server down.
            console.print(Panel.fit(
                f"[red]The MCP server went away mid-session.[/red]\n"
                f"{type(e).__name__}\n\n"
                "The server process died while a call was in flight — usually a long tool\n"
                "(grasp, refine_grasp_point, stop_recording) outliving the client timeout.\n"
                f"Current read timeout: {READ_TIMEOUT.total_seconds():.0f}s "
                "(raise with MCP_READ_TIMEOUT).\n\n"
                "Restart it:  [dim]SO101_VIEWER=0 MUJOCO_GL=egl ./venv/bin/python main.py[/dim]",
                title="server gone", border_style="red"))
        elif connected:
            console.print(Panel.fit(
                f"[red]The panel hit an error after connecting successfully.[/red]\n"
                f"{type(e).__name__}: {e}\n\n"
                "The server was reachable — this looks like a fault in the panel itself.",
                title="panel error", border_style="red"))
            traceback.print_exc()
        else:
            console.print(Panel.fit(
                f"[red]Could not reach the MCP server at {MCP_URL}[/red]\n{type(e).__name__}: {e}\n\n"
                "Start it first in another terminal:\n"
                "  cd so101-Models && ./venv/bin/python main.py\n"
                "For the perception tools, load the perception scene:\n"
                "  SO101_SCENE=models/so101/perception_test_scene.xml ./venv/bin/python main.py",
                title="not connected", border_style="red"))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[dim]bye[/dim]")
