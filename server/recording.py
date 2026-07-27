"""Titled MP4 run recorder for MCP-driven pick-and-place sessions.

A background daemon thread grabs frames from a DEDICATED offscreen renderer (its
OWN ``mujoco.Renderer`` — never the backend's shared servo ``_offscreen``, which
is not thread-safe) at a fixed fps and, on ``stop()``, writes
``recordings/<title>.mp4`` via ``imageio`` (the same dependency the training
``MultiCameraVideoRecorder`` uses).

Recording spans a whole verb, not a single tool call: one ``grasp()`` runs
hundreds of internal sim steps, so frames are pulled on a timer thread rather
than per tool call. Every frame grab holds ``backend._step_lock`` so it never
races ``mj_step``. All GL work (renderer create/render/close) happens on the
daemon thread so the EGL context is owned by a single thread.
"""
from __future__ import annotations

import re
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np

# recordings/ lives at the PROJECT ROOT: this file is
# so101-Models/server/recording.py -> parents[2] == Internship/.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_REC_DIR = _PROJECT_ROOT / "recordings"

_REC_W, _REC_H = 640, 480
_DEFAULT_FPS = 20.0
_READY_TIMEOUT_S = 20.0   # max wait for the capture thread's GL context
_JOIN_TIMEOUT_S = 10.0
_MAX_TITLE_LEN = 80

# Overview free camera (diagonal look, mirrors SimulationBackend.render's wide view).
_CAM_LOOKAT = (0.20, 0.10, 0.05)
_CAM_DIST = 0.85
_CAM_AZ = 135.0
_CAM_EL = -25.0

_UNSAFE_TITLE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize_title(title: str) -> str:
    """Path-traversal-safe filename: collapse every unsafe char (incl. '/', '\\',
    ':', spaces) to '_' so no separator survives, then trim leading/trailing
    '._-' (kills leading '..') and cap the length. The result is always a single
    filename that lands inside recordings/, never an escape."""
    base = _UNSAFE_TITLE_CHARS.sub("_", str(title)).strip("._-")
    return base[:_MAX_TITLE_LEN] or "run"


class RunRecorder:
    """One-at-a-time titled MP4 recorder driven by a daemon frame-grab thread.

    Usage: ``start(title)`` begins capture; ``stop()`` finalises the file and
    returns its path. A second ``start`` while recording finalises the in-flight
    clip first (``replace=True``, default) or is rejected (``replace=False``).
    """

    def __init__(self, backend, fps: float = _DEFAULT_FPS):
        self.backend = backend
        self.fps = float(fps)
        self._thread: Optional[threading.Thread] = None
        self._stop_evt: Optional[threading.Event] = None
        self._frames: list[np.ndarray] = []
        self._title: Optional[str] = None
        self._lock = threading.Lock()            # guards start/stop transitions

    @property
    def is_recording(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _make_cam(self) -> "mujoco.MjvCamera":
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = _CAM_LOOKAT
        cam.distance = _CAM_DIST
        cam.azimuth = _CAM_AZ
        cam.elevation = _CAM_EL
        return cam

    def _grab_loop(self, stop_evt: threading.Event,
                   ready_evt: Optional[threading.Event] = None) -> None:
        """Own the renderer for the whole recording, on THIS thread (EGL is
        single-threaded). Append a frame every 1/fps s until stopped."""
        period = 1.0 / max(self.fps, 1.0)
        renderer = None
        try:
            try:
                renderer = mujoco.Renderer(self.backend.model, height=_REC_H, width=_REC_W)
                cam = self._make_cam()
                # Private MjData to render FROM. The previous version held _step_lock across
                # update_scene + render — roughly 30 ms — while the control loop takes and
                # releases that same lock on every physics step. Python locks are not fair, so
                # a thread that releases and immediately re-acquires wins nearly every time,
                # which would starve this one during any sustained motion. Holding the lock
                # only for a cheap state copy, and rendering outside it, removes that risk.
                scratch = mujoco.MjData(self.backend.model)
            except Exception as exc:
                print(f"[recording] could not start renderer: "
                      f"{type(exc).__name__}: {exc}", file=sys.stderr)
                return
            finally:
                # Signal readiness (or failure) BEFORE the capture loop, so start() does not
                # return until this thread can actually grab frames. Building the Renderer
                # creates a GL context and is slow; a short motion could otherwise finish and
                # stop the recording before the first frame was ever taken, reporting
                # "no frames captured" with no error — which reads as a broken renderer
                # rather than the race it is.
                if ready_evt is not None:
                    ready_evt.set()

            consecutive_failures = 0
            while not stop_evt.is_set():
                t0 = time.monotonic()
                try:
                    with self.backend._step_lock:
                        scratch.qpos[:] = self.backend.data.qpos
                        scratch.qvel[:] = self.backend.data.qvel
                        scratch.ctrl[:] = self.backend.data.ctrl
                        scratch.time = self.backend.data.time
                    mujoco.mj_forward(self.backend.model, scratch)
                    renderer.update_scene(scratch, camera=cam)
                    self._frames.append(renderer.render().copy())
                    consecutive_failures = 0
                except Exception as exc:
                    # A transient failure must not kill the recording, but silently
                    # swallowing EVERY failure is how the starvation above stayed invisible:
                    # stop() reported "no frames captured" with no clue why. Report the first
                    # one and give up after a persistent run of them.
                    consecutive_failures += 1
                    if consecutive_failures == 1:
                        print(f"[recording] frame capture failed: "
                              f"{type(exc).__name__}: {exc}", file=sys.stderr)
                    if consecutive_failures >= 20:
                        print("[recording] giving up after 20 consecutive failures",
                              file=sys.stderr)
                        break
                dt = time.monotonic() - t0
                stop_evt.wait(max(0.0, period - dt))
        finally:
            if renderer is not None:
                try:
                    renderer.close()
                except Exception:
                    pass

    def start(self, title: str, replace: bool = True) -> dict:
        with self._lock:
            if self.is_recording:
                if not replace:
                    return {"ok": False, "recording": True, "title": self._title,
                            "error": "a recording is already running; stop it first"}
                self._stop_locked()              # finalise the in-flight clip first
            self._title = _sanitize_title(title)
            self._frames = []
            self._stop_evt = threading.Event()
            ready_evt = threading.Event()
            self._thread = threading.Thread(
                target=self._grab_loop, args=(self._stop_evt, ready_evt),
                name=f"run-recorder:{self._title}", daemon=True)
            self._thread.start()
            # Block until the capture thread is actually able to render. Without this,
            # start() returns while the GL context is still being built and any motion
            # shorter than that setup is recorded as zero frames.
            if not ready_evt.wait(timeout=_READY_TIMEOUT_S):
                print("[recording] renderer still not ready after "
                      f"{_READY_TIMEOUT_S:.0f}s — recording may miss the start",
                      file=sys.stderr)
            return {"ok": True, "recording": True, "title": self._title,
                    "fps": self.fps}

    def _stop_locked(self) -> dict:
        """Stop the thread and write the file. Caller must hold ``self._lock``."""
        if self._thread is None:
            return {"ok": False, "recording": False,
                    "error": "no recording in progress"}
        if self._stop_evt is not None:
            self._stop_evt.set()
        self._thread.join(timeout=_JOIN_TIMEOUT_S)   # join => frames list is settled
        self._thread = None
        self._stop_evt = None
        title = self._title or "run"
        frames = self._frames
        self._frames = []
        if not frames:
            return {"ok": False, "recording": False, "title": title,
                    "n_frames": 0, "error": "no frames captured "
                    "(offscreen renderer unavailable? launch with MUJOCO_GL=egl)"}
        _REC_DIR.mkdir(parents=True, exist_ok=True)
        path = _REC_DIR / f"{title}.mp4"

        # Encode OFF the request thread. Encoding hundreds of frames takes seconds, and doing
        # it inline held the stop_recording call open long enough for the client to time out —
        # which does not merely fail the call, it takes the server down and loses the clip
        # entirely. The tool now returns at once and the file appears when it is complete.
        #
        # Written to <title>.mp4.part and renamed only on success, so the final filename
        # NEVER exists in a half-written state: if <title>.mp4 is there, it is playable.
        # Non-daemon thread on purpose — the process should wait for an encode in flight
        # rather than exit and discard it.
        # NOTE the ".part.mp4" ordering: imageio picks its backend from the FINAL extension,
        # so a ".mp4.part" temp name fails with "could not find a backend ... iomode wI".
        tmp = path.with_name(f"{path.stem}.part.mp4")
        fps = self.fps

        def _encode():
            try:
                import imageio
                imageio.mimsave(str(tmp), np.asarray(frames), fps=fps)
                tmp.replace(path)
                print(f"[recording] wrote {path} ({len(frames)} frames)", file=sys.stderr)
            except Exception as exc:
                print(f"[recording] encode FAILED for {path.name}: "
                      f"{type(exc).__name__}: {exc}", file=sys.stderr)
                try:
                    tmp.unlink(missing_ok=True)
                except Exception:
                    pass

        threading.Thread(target=_encode, name=f"run-encoder:{title}",
                         daemon=False).start()
        return {"ok": True, "recording": False, "title": title,
                "n_frames": len(frames), "path": str(path), "encoding": True,
                "note": "encoding in the background; the file appears when complete"}

    def stop(self) -> dict:
        with self._lock:
            return self._stop_locked()
