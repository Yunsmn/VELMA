"""Named arm poses, recorded in SERVO degrees and persisted to disk.

Teach-and-repeat needs somewhere to keep positions between sessions, and servo
degrees are the right unit to keep them in: they are read straight off the
encoders and do not depend on the model<->servo offsets, which are still
unmeasured. A pose saved today therefore replays correctly even if the
kinematic mapping is fixed later.

Each entry also carries the gripper opening at the time, so a pose can record
"holding the object" as distinct from "open above it", plus a free-text note.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "poses.json"


class PoseStore:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else DEFAULT_PATH
        self._poses: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            self._poses = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            # A corrupt store must not take the server down, but silently
            # starting empty would quietly lose taught positions.
            raise RuntimeError(f"pose store at {self.path} is unreadable: {e}") from e

    def _save(self) -> None:
        # Write via a temporary file in the same directory, then replace: a crash
        # mid-write would otherwise leave a truncated file and lose every pose.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(self._poses, fh, indent=2, sort_keys=True)
                fh.write("\n")
            os.replace(tmp, self.path)
        except Exception:
            Path(tmp).unlink(missing_ok=True)
            raise

    def save(self, name: str, angles_deg: dict[str, float], note: str = "") -> dict:
        entry = {
            "angles_deg": {k: round(float(v), 2) for k, v in angles_deg.items()},
            "note": note,
            "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self._poses[name] = entry
        self._save()
        return {"name": name, **entry}

    def get(self, name: str) -> dict:
        if name not in self._poses:
            raise KeyError(f"no pose named {name!r}; known: {sorted(self._poses)}")
        return self._poses[name]

    def delete(self, name: str) -> None:
        self._poses.pop(name, None)
        self._save()

    def list(self) -> dict[str, dict]:
        return dict(self._poses)
