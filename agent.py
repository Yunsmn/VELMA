"""Model-agnostic MCP agent for the SO-101 robot arm.

One thin driver that connects an LLM to the SO-101 MCP server and lets the model
control the arm. It is deliberately *thin*: it relays the model's tool calls to
the server and the results back, and otherwise stays out of the way. The model
keeps control of what to do — this script does not inject a procedure, plan the
grasp, or second-guess the model's moves.

Works with any model / endpoint:
  - **Native tool-calling** (OpenAI-compatible `/chat/completions`, or Ollama
    `/api/chat` with models that support `tools`) — the model emits structured
    tool_calls and we run them.
  - **Text tool-calls** (local / weak models with no native tool support) — the
    model writes `<tool_call>{"name": ..., "arguments": {...}}</tool_call>` and a
    tolerant parser extracts it (brace-repair for truncated JSON).
  - **auto** (default) tries native and transparently accepts text calls too, so
    you don't have to know in advance what the model can do.

Endpoints:
  - Ollama:            base_url like http://localhost:11434   (uses /api/chat)
  - OpenAI-compatible: base_url like https://...              (uses /chat/completions)
    e.g. GitHub Models, OpenAI, OpenRouter, vLLM, llama.cpp server, Ollama /v1.

Config via env or flags:
  LLM_MODEL / --model        e.g. gemma4-cpu  or  openai/gpt-5-mini
  LLM_BASE_URL / --base-url  default http://localhost:11434  (Ollama)
  LLM_API_KEY / --api-key    token if the endpoint needs one (local: leave empty)
  --mode auto|native|text    default auto
  --mcp-url                  default http://localhost:3001/mcp
  --task "..."               one-shot task; omit for an interactive REPL

Examples:
  ./venv/bin/python agent.py --model gemma4-cpu --task "pick up the cube and place it in the container"
  LLM_BASE_URL=https://models.github.ai/inference LLM_API_KEY=$GH_TOKEN \\
      ./venv/bin/python agent.py --model openai/gpt-5-mini --task "..."
"""
from __future__ import annotations
import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

# Run without activating the venv first.
_site = next(Path(__file__).parent.glob("venv/lib/python*/site-packages"), None)
if _site and str(_site) not in sys.path:
    sys.path.insert(0, str(_site))

import requests
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

MCP_URL        = os.environ.get("MCP_URL", "http://localhost:3001/mcp")
DEFAULT_BASE   = os.environ.get("LLM_BASE_URL", "http://localhost:11434")
# Co-located with this file so a standalone so101-Models checkout is self-contained
# (this used to point at ../skills/so101-arm/SKILL.md, one level ABOVE this project,
# which only existed in the full monorepo checkout). skills/so101-arm/SKILL.md in the
# monorepo is now a symlink to this file, so both locations stay in sync.
SKILL_PATH     = Path(__file__).parent / "SKILL.md"
MAX_TOOL_TURNS = 35

# Patterns for extracting a text tool-call from models without native support.
_PATTERNS = [
    re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL),    # <tool_call>{}</tool_call>
    re.compile(r"<tool_call>\s*(\{.*)", re.DOTALL),                   # <tool_call>{ ... (no close)
    re.compile(r"```tool_call\s*(.*?)\s*```", re.DOTALL),             # ```tool_call\n{}\n```
    re.compile(r"```json\s*(\{.*?\"name\".*?\})\s*```", re.DOTALL),   # ```json\n{"name":...}\n```
    re.compile(r"(\{[^{}]*\"name\"\s*:\s*\"[a-z_]+\".*)", re.DOTALL), # bare {"name": ...} anywhere
]


# ── Skill / system prompt ──────────────────────────────────────────────────────

def _build_system(skill: str, tools, mode: str = "auto") -> str:
    """Compose the system prompt: the skill text + how to call tools + the tool
    list. The 'how to call' note works for both native and text models so the
    same prompt drives any runtime."""
    if mode == "native":
        howto = ("\n## Calling tools\n"
                 "Use your native tool/function calling to invoke the tools below. "
                 "Do NOT print tool calls as text. Work step by step: call one tool, "
                 "read its result, then decide the next call.\n")
    elif mode == "text":
        howto = ("\n## Calling tools\n"
                 "Emit EXACTLY one tool call per reply, as:\n"
                 "<tool_call>{\"name\": \"TOOL_NAME\", \"arguments\": {\"arg\": value}}</tool_call>\n"
                 "Nothing else. After you see the result, send your next call.\n")
    else:  # auto — works either way
        howto = ("\n## Calling tools\n"
                 "If your runtime supports tool/function calls, use them. Otherwise emit "
                 "exactly one tool call as text:\n"
                 "<tool_call>{\"name\": \"TOOL_NAME\", \"arguments\": {\"arg\": value}}</tool_call>\n"
                 "Work step by step: one tool at a time, read the result, then decide the "
                 "next call. When the task is done, give a short final answer.\n")
    lines = [skill, howto, "## Available tools\n"]
    for t in tools:
        props = (t.inputSchema or {}).get("properties", {})
        args = ", ".join(
            f'{k} ({v.get("type", "any")}): {v.get("description", "")}'
            for k, v in props.items()
        )
        lines.append(f"- **{t.name}**: {t.description or ''}")
        if args:
            lines.append(f"  args: {args}")
    return "\n".join(lines)


# ── Tolerant text tool-call parsing (helps weak models, restricts nothing) ─────

def _repair_json(blob: str) -> dict | None:
    """Best-effort parse of possibly-truncated/unbalanced JSON from weak models."""
    blob = blob.strip()
    start = blob.find("{")
    if start == -1:
        return None
    blob = blob[start:]
    try:
        obj = json.loads(blob)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    depth_curly = depth_square = 0
    in_str = esc = False
    end = None
    for i, ch in enumerate(blob):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth_curly += 1
        elif ch == "}":
            depth_curly -= 1
            if depth_curly == 0:
                end = i + 1
                break
        elif ch == "[":
            depth_square += 1
        elif ch == "]":
            depth_square -= 1
    if end is not None:
        try:
            obj = json.loads(blob[:end])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            pass
    candidate = blob
    if in_str:
        candidate += '"'
    candidate = re.sub(r"[,:\s]+$", "", candidate)
    candidate += "]" * max(depth_square, 0)
    candidate += "}" * max(depth_curly, 0)
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _parse_call(text: str) -> dict | None:
    """Extract a {"name", "arguments"} tool call from model text, tolerantly."""
    for pattern in _PATTERNS:
        m = pattern.search(text)
        if not m:
            continue
        obj = _repair_json(m.group(1))
        if obj and isinstance(obj.get("name"), str):
            return obj
    return None


def _tool_schemas(tools) -> dict:
    return {t.name: set((t.inputSchema or {}).get("properties", {}).keys()) for t in tools}


def _filter_args(name: str, args, schemas: dict) -> tuple[dict, list]:
    """Drop args the tool doesn't accept (a weak model sometimes invents some) and
    report which were dropped — we tell the model, so this stays transparent rather
    than silently changing its intent."""
    if not isinstance(args, dict):
        return {}, []
    allowed = schemas.get(name)
    if allowed is None:
        return args, []
    clean = {k: v for k, v in args.items() if k in allowed}
    dropped = [k for k in args if k not in allowed]
    return clean, dropped


# ── Endpoint adapters (Ollama / OpenAI-compatible) ─────────────────────────────

def _detect_flavor(base_url: str) -> str:
    b = base_url.rstrip("/")
    if "/v1" in b:
        return "openai"
    if "11434" in b or b.endswith("/api"):
        return "ollama"
    return "openai"


def _normalize_tool_calls(msg: dict) -> list:
    """Return [{id, name, arguments(dict)}] from either provider's tool_calls."""
    tcs = msg.get("tool_calls") or []
    out = []
    for i, tc in enumerate(tcs):
        fn = tc.get("function", tc)
        name = fn.get("name")
        args = fn.get("arguments")
        if isinstance(args, str):
            args = _repair_json(args) or {}
        if not isinstance(args, dict):
            args = {}
        out.append({"id": tc.get("id") or f"call_{i}", "name": name, "arguments": args})
    return [c for c in out if c["name"]]


def make_chat(base_url: str, api_key: str, flavor: str | None = None):
    """Return chat(model, messages, tools_param, timeout) -> provider message dict.

    tools_param is the provider tool list (or None to suppress native tools).
    The returned dict has at least 'content'; it may carry 'tool_calls'.
    """
    flavor = flavor or _detect_flavor(base_url)
    base = base_url.rstrip("/")

    def chat(model, messages, tools_param, timeout=180):
        if flavor == "ollama":
            body = {"model": model, "messages": messages, "stream": False}
            if tools_param:
                body["tools"] = tools_param
            r = requests.post(base + "/api/chat", json=body, timeout=timeout)
            if not r.ok:
                raise RuntimeError(f"Ollama {r.status_code}: {r.text[:300]}")
            return r.json()["message"]
        # openai-compatible
        body = {"model": model, "messages": messages, "temperature": 0}
        if tools_param:
            body["tools"] = tools_param
            body["tool_choice"] = "auto"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        r = requests.post(base + "/chat/completions", headers=headers, json=body, timeout=timeout)
        if not r.ok:
            raise RuntimeError(f"LLM {r.status_code}: {r.text[:300]}")
        return r.json()["choices"][0]["message"]

    chat.flavor = flavor
    return chat


def _mcp_tools_to_provider(tools, flavor: str) -> list:
    """MCP tool schemas -> the provider's `tools` array (same shape for both)."""
    out = []
    for t in tools:
        schema = t.inputSchema or {"type": "object", "properties": {}}
        if schema.get("type") != "object":
            schema = {"type": "object", "properties": {}}
        out.append({"type": "function", "function": {
            "name": t.name,
            "description": (t.description or "")[:1024],
            "parameters": schema,
        }})
    return out


# ── The driver loop ────────────────────────────────────────────────────────────

async def run_task(session, model, system, task, tools,
                   chat=None, provider_tools=None, mode="auto",
                   max_turns=MAX_TOOL_TURNS, per_call_timeout=900,
                   max_idle_nudges=3, verbose=False,
                   progress=None, stop_condition=None) -> dict:
    """Drive one task to completion. Endpoint- and mode-agnostic.

    `chat` is a callable from make_chat(); if omitted, an Ollama chat against
    DEFAULT_BASE is built (keeps the simple local-Ollama default working).
    `mode`: 'native' (only structured tool_calls), 'text' (only <tool_call> text),
    or 'auto' (offer native tools and also accept text calls; fall back to text if
    the endpoint rejects the tools param).

    Returns a dict: success_text, n_turns, n_tool_calls, n_parse_fail,
    n_idle_nudges, tool_trace, final_reply, stopped. If `progress` is a dict it is
    mirrored in place each turn so a wall-clock-cancelled coroutine keeps partial
    progress. If `stop_condition(name, result_text)->bool` is given, the loop ends
    the moment it returns True (stopped='task_done')."""
    if chat is None:
        chat = make_chat(DEFAULT_BASE, os.environ.get("LLM_API_KEY", ""), "ollama")
    if provider_tools is None:
        provider_tools = _mcp_tools_to_provider(tools, getattr(chat, "flavor", "openai"))

    valid_names = {t.name for t in tools}
    schemas = _tool_schemas(tools)
    native_on = mode in ("auto", "native")
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": task},
    ]
    n_turns = n_tool_calls = n_parse_fail = n_idle_nudges = 0
    tool_trace: list = []
    final_reply = ""
    stopped = "turn_limit"

    def _mirror():
        if progress is not None:
            progress.update(n_turns=n_turns, n_tool_calls=n_tool_calls,
                            n_parse_fail=n_parse_fail, n_idle_nudges=n_idle_nudges,
                            tool_trace=tool_trace, final_reply=final_reply, stopped=stopped)
    _mirror()

    async def _dispatch(name, raw_args):
        """Run one tool on the MCP server; return (clean_args, dropped, ok, text)."""
        nonlocal n_tool_calls
        args, dropped = _filter_args(name, raw_args, schemas)
        if verbose:
            extra = f"  (ignored {dropped})" if dropped else ""
            print(f"  -> {name}({args}){extra}")
        try:
            res = await session.call_tool(name, args)
            text = " | ".join(c.text for c in res.content if hasattr(c, "text"))
            ok = True
        except Exception as e:  # noqa: BLE001
            text = f"[tool error: {e}]"
            ok = False
        if dropped:  # tell the model what we ignored — keep it in control
            text = f"(ignored unexpected args {dropped}) " + text
        n_tool_calls += 1
        tool_trace.append({"name": name, "args": args, "dropped": dropped,
                           "ok": ok, "result_excerpt": text[:200]})
        if verbose:
            print(f"  <- {text[:200]}")
        return args, dropped, ok, text

    for _ in range(max_turns):
        n_turns += 1
        try:
            msg = chat(model, messages, provider_tools if native_on else None, per_call_timeout)
        except Exception as e:  # noqa: BLE001
            emsg = str(e)
            # Endpoint rejected the tools param? drop to text-only and retry once.
            if native_on and mode == "auto" and "tool" in emsg.lower():
                native_on = False
                n_turns -= 1
                continue
            final_reply = f"[llm error: {e}]"
            stopped = "llm_error"
            break

        content = msg.get("content") or ""
        final_reply = content
        _mirror()
        calls = _normalize_tool_calls(msg) if native_on else []

        # ── Native tool calls ───────────────────────────────────────────────
        if calls:
            messages.append(msg)  # provider's assistant msg (carries tool_calls)
            done = False
            for c in calls:
                if c["name"] not in valid_names:
                    n_parse_fail += 1
                    result_text = (f"There is no tool named '{c['name']}'. "
                                   f"Valid tools: {sorted(valid_names)}.")
                    ok = False
                else:
                    _, _, ok, result_text = await _dispatch(c["name"], c["arguments"])
                messages.append({"role": "tool", "tool_call_id": c["id"],
                                 "content": result_text or "(no output)"})
                _mirror()
                if stop_condition and ok and stop_condition(c["name"], result_text):
                    stopped = "task_done"
                    done = True
                    break
            if done:
                break
            continue

        # ── Text tool call ──────────────────────────────────────────────────
        parsed = _parse_call(content)
        if parsed is not None:
            name = parsed.get("name", "")
            if name not in valid_names:
                n_parse_fail += 1
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content":
                    f"There is no tool named '{name}'. Valid tools: {sorted(valid_names)}. "
                    "Call a valid tool."})
                continue
            _, _, ok, result_text = await _dispatch(name, parsed.get("arguments") or {})
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": f"[tool result for {name}]: {result_text}"})
            _mirror()
            if stop_condition and ok and stop_condition(name, result_text):
                stopped = "task_done"
                break
            continue

        # ── No tool call shaped output ──────────────────────────────────────
        if "tool_call" in content or '"name"' in content:
            # Looked like an attempt we couldn't parse — ask once for clean form.
            n_parse_fail += 1
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content":
                "That tool call could not be parsed. Reply with exactly one "
                "<tool_call>{\"name\": ..., \"arguments\": {...}}</tool_call> and nothing else."})
            continue
        if n_idle_nudges < max_idle_nudges and n_tool_calls == 0:
            # A stalling model that hasn't acted yet — one neutral nudge, no hints.
            n_idle_nudges += 1
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content":
                "You have not called any tool yet. Make your next tool call now."})
            continue
        stopped = "gave_up" if n_tool_calls == 0 else "final_answer"
        break

    success_text = final_reply if stopped in ("final_answer", "task_done") else ""
    return {
        "success_text": success_text,
        "n_turns": n_turns, "n_tool_calls": n_tool_calls,
        "n_parse_fail": n_parse_fail, "n_idle_nudges": n_idle_nudges,
        "tool_trace": tool_trace, "final_reply": final_reply, "stopped": stopped,
    }


# ── Entry point ────────────────────────────────────────────────────────────────

async def _connect_and_run(model, base_url, api_key, mode, mcp_url, task, verbose,
                           per_call_timeout=900):
    skill = SKILL_PATH.read_text() if SKILL_PATH.exists() else ""
    if not skill:
        print(f"[warn] skill not found at {SKILL_PATH}", file=sys.stderr)

    chat = make_chat(base_url, api_key)
    print(f"Endpoint: {base_url}  ({chat.flavor})  |  model: {model}  |  mode: {mode}")

    async with streamablehttp_client(mcp_url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools
            provider_tools = _mcp_tools_to_provider(tools, chat.flavor)
            print(f"Connected to MCP — {len(tools)} tools available.\n")

            system = _build_system(skill, tools, mode)

            async def _one(t):
                r = await run_task(session, model, system, t, tools,
                                   chat=chat, provider_tools=provider_tools,
                                   mode=mode, verbose=verbose,
                                   per_call_timeout=per_call_timeout)
                if r["success_text"]:
                    print(f"\nAssistant: {r['success_text'].strip()}\n")
                else:
                    print(f"\n(stopped: {r['stopped']}; {r['n_tool_calls']} tool calls, "
                          f"{r['n_parse_fail']} parse fails)\n")

            if task:
                await _one(task)
            else:
                print("Interactive — type a task, or 'quit' to exit.\n")
                while True:
                    try:
                        line = input("You: ").strip()
                    except EOFError:
                        break
                    if line.lower() in ("quit", "exit", "q"):
                        break
                    if line:
                        await _one(line)


def main():
    ap = argparse.ArgumentParser(description="Model-agnostic MCP agent for the SO-101 arm.")
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL", "gemma4-cpu"))
    ap.add_argument("--base-url", default=DEFAULT_BASE)
    ap.add_argument("--api-key", default=os.environ.get("LLM_API_KEY", ""))
    ap.add_argument("--mode", choices=["auto", "native", "text"], default="auto")
    ap.add_argument("--mcp-url", default=MCP_URL)
    ap.add_argument("--task", default=None, help="one-shot task; omit for interactive REPL")
    ap.add_argument("--quiet", action="store_true", help="don't print each tool call")
    ap.add_argument("--timeout", type=int, default=int(os.environ.get("LLM_TIMEOUT", "900")),
                    help="per-LLM-call timeout in seconds (CPU-only models need 600-900+)")
    args = ap.parse_args()

    asyncio.run(_connect_and_run(args.model, args.base_url, args.api_key, args.mode,
                                 args.mcp_url, args.task, verbose=not args.quiet,
                                 per_call_timeout=args.timeout))


if __name__ == "__main__":
    main()
