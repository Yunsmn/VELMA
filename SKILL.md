---
name: so101-arm
description: >
  Drive an SO-101 6-DOF robot arm with a parallel-jaw gripper in a MuJoCo
  tabletop simulation, through an MCP server. Use whenever a task means
  physically manipulating an object with the arm — pick up, move, place, or
  clear an obstacle. Connects over MCP and works with any LLM runtime,
  whether it has native tool/function calling or only emits text tool-calls.
---

# Driving the SO-101 Arm

You are operating a real-feeling 6-DOF robot arm with a two-finger (parallel-jaw)
gripper, in a physics simulation. The tools listed below are your hands, your
eyes, and your only way to know where anything is — nothing happens unless you
call one, and **no tool ever hands you an object's coordinates for free.** You
find the object yourself, with the cameras. This skill is here to make you good
at that decision, not to script it. Treat it as advice from someone who has
driven this arm a lot, not as a fixed procedure you must replay.

The running example throughout is pick-and-place (find an object, lift it, drop
it in a container), because that exercises every part of the arm.

---

## 1. Connecting to the arm (MCP)

The arm lives behind an **MCP server**. Until you're connected to it, you have no
tools; once you are, the tools below appear in your runtime like any other.

**Start the server** (once, from the project's robot package directory, e.g.
`so101-Models/`): run the server entry point, headless (perception needs
offscreen rendering — see the project README). When it's up it serves MCP at:

```
http://127.0.0.1:3001/mcp        (transport: streamable-http)
```

**Connect your client** to that URL:

- If your client/runtime is configured by a file (an `.mcp.json`-style config),
  add a server entry pointing at it:
  ```json
  { "mcpServers": { "so101-sim": { "type": "streamable-http",
                                    "url": "http://127.0.0.1:3001/mcp" } } }
  ```
- If you're writing the client yourself, open a streamable-http MCP session to
  the same URL, initialize it, then list tools. (In Python that's
  `streamablehttp_client("http://127.0.0.1:3001/mcp")` → `ClientSession` →
  `await session.initialize()` → `await session.list_tools()`.) Set a **long
  read timeout** (300s) — `grasp`/`triangulate`/`stop_recording` take several
  seconds, and a client that times out mid-call can take the server down.

**First call, every session:** `get_initial_instructions()`. It returns the live,
authoritative description of the scene and tools straight from the server — if it
ever disagrees with this document, trust the server.

> If tools aren't showing up, the server isn't running or your client is pointed
> at the wrong URL/transport — it must be `streamable-http` at the `/mcp` path,
> not plain HTTP at the root.

---

## 2. The mental model

- **Coordinates** are metres from the arm's base: **x = forward, y = left (+) /
  right (−), z = up**. The table is at `z = 0`; a typical small object sits around
  `z ≈ 0.015`. Angles are in degrees.
- **Reach** is roughly a `hypot(x, y) ≲ 0.30 m` bubble in front of the base, with
  base rotation about ±110°. The arm can't reach straight behind itself, and the
  far corners of that bubble are weak. If a target sits near the edge, expect it
  to be harder and be ready to accept that some spots are simply out of reach.
- **Nothing hands you the object's position — you find it.** `get_robot_state()`
  tells you about the ROBOT (joints, gripper, end-effector, whether you're
  holding something) but deliberately does not know or reveal where the cube or
  container are. That is perception's job: `find_object` (coarse, a fixed
  camera) then `triangulate` (precise, the wrist camera). This mirrors a real
  arm, which has no ground-truth sensor for "where is the cube" either.
- **The loop that works:** *perceive → decide → act → verify.* Locate the
  object with the cameras, reason about one move, make it, then check the
  result (`is_grasping`, `cube_in_container`) before committing to the next
  move. The arm is in a physics sim — momentum, gravity sag, and contact are
  real, so the result of a move is not always exactly what you predicted.

### The one principle worth internalising: be gentle

The single thing that separates a good operator of this arm from a bad one is
**not shoving the object around.** A grasp that knocks the object skittering
across the table is a failed grasp, even if the gripper ends up closed. So the
thing you're really optimising for, on every move near the object, is *leaving it
where it is until you have a secure hold.*

This isn't a rule with a magic number you have to enforce by hand — it's a goal.
In practice it means: approach slowly, and trust the high-level `grasp` tool,
which already declines gently rather than forcing a bad grip instead of pushing
through.

---

## 3. The toolbox (15 tools)

### Perceive (find the object — never guess, never assume you already know)
| Tool | Good for |
|---|---|
| `get_initial_instructions()` | The server's own canonical brief. Call first. |
| `find_object(prompt)` | Coarse `(x, y, z)` of an object named by a short phrase (e.g. `"the red cube"`, `"the blue container"`) from a fixed side camera. Centimetre-scale — enough to aim, not enough to grasp. |
| `triangulate(prompt, x_m, y_m)` | Takes the coarse `(x, y)` `find_object` returned and refines it to a grasp-ready point (~1 mm) using the wrist camera, which orbits above it. Also returns a **measured** `width_mm` for jaw sizing. `confidence < 0.6` means the views disagreed — re-run `find_object` or decline rather than trust it. |
| `get_robot_state()` | Joint angles, end-effector position, gripper openness, `is_grasping`, `cube_in_container`, camera images. Your "how am I doing" call — it does **not** contain object coordinates. |
| `capture_cameras()` | Save the current side + wrist camera frames to disk and return the paths — useful to visually sanity-check what the cameras are seeing. |

### Manipulation
| Tool | Good for |
|---|---|
| `grasp(x_m, y_m, z_m, trust_coords=true, grip_width_mm=null, object_width_mm=null, ...)` | A complete gentle top-down pickup at the point you measured: reads the object live, lines the jaws up, descends without batting it, closes, self-checks. `trust_coords=true` tells it to act on the coordinate YOU pass (from `triangulate`) rather than any internal shortcut. Pass `grip_width_mm=width_mm+10` from `triangulate`'s measured width. Returns `is_grasping`. |
| `place(x_m, y_m, z_m)` | A gentle put-down at a target: move above, descend, release, withdraw. Use after a successful grasp. Returns `cube_in_container`. |

### Motion
| Tool | Good for |
|---|---|
| `move_to_position(x_m, y_m, z_m, gain=0.5, lock_wrist=true)` | Send the gripper to an absolute point (IK). Lower `gain` (≈0.1–0.3) for slow, careful moves, especially while carrying. `lock_wrist` keeps the wrist orientation steady. Use this for waypoints, e.g. lifting clear of an obstacle before crossing it. |
| `set_joint_angles(shoulder_pan_deg?, shoulder_lift_deg?, elbow_flex_deg?, wrist_flex_deg?, wrist_roll_deg?)` | Snap one or more arm joints to an absolute angle. Good for jumping to a known pose to pre-position or recover from a tangle. (Gripper is separate — use `set_gripper`.) |

### Gripper
| Tool | Effect |
|---|---|
| `set_gripper(percent)` | 0 = closed, ~65 = open-for-approach, 100 = wide open. Closing uses an IK-hold so the arm doesn't drift while the jaws move. |
| `open_gripper()` / `close_gripper()` | shorthands for `set_gripper(100)` / `set_gripper(0)`. |

### Harness only (testing/eval, not manipulation moves)
| Tool | Effect |
|---|---|
| `reset_scene(cube_x?, cube_y?, cube_z?, container_x?, container_y?, container_z?)` | Reposition objects for a trial. |
| `start_recording(title)` / `stop_recording()` | Record the run to `recordings/<title>.mp4`. Only on an explicit go-ahead. |

You don't have to use every tool, and not in a fixed order — but the perception
tools (`find_object` → `triangulate`) are not optional busywork: they are the
*only* way you learn where an object is, so a pick-and-place always starts there.

---

## 4. How to think about a pick-and-place

This is a way of reasoning, not a checklist to obey.

1. **Find it.** `find_object("the red cube")` for a coarse `(x, y, z)`. If it
   isn't found, try re-phrasing the prompt (colour + noun works best) — don't
   invent a position.
2. **Refine it.** `triangulate("the red cube", x, y)` using the coarse `x, y`.
   Check `confidence` — below 0.6, the wrist camera's views disagreed; re-run
   `find_object` (the object may have moved, or the coarse aim was too far off)
   rather than grasp on a point you can't trust.
3. **Reason about the grip.** `triangulate` gives you a *measured* `width_mm` —
   use `grip_width_mm = width_mm + 10` so the jaws open a bit wider than the
   object. The gripper is two flat pads: it holds box-like objects roughly
   30–40 mm wide well, and honestly declines (rather than forcing a hold) on
   round objects or ones outside that band — see §6.
4. **Grasp.** `grasp(x_m, y_m, z_m, trust_coords=true, grip_width_mm=width_mm+10)`
   using the coordinate `triangulate` returned. Check `is_grasping`.
5. **Find the target, then place.** If the container's position isn't already
   known, `find_object("the container")` for it too (a single coarse read is
   normally enough for a place — you don't need to grasp it). Then
   `place(cx, cy, 0.05)` and confirm `cube_in_container`.

A worked sketch (yours will differ — these are illustrative values, not a script):

```
find_object("the red cube")                  → found (0.24, 0.00, 0.02), conf 0.8
triangulate("the red cube", 0.24, 0.00)       → (0.241, -0.003, 0.016), width 31mm, conf 0.94
grasp(0.241, -0.003, 0.016, trust_coords=true, grip_width_mm=41)  → is_grasping: true
find_object("the container")                  → found (0.18, 0.30, 0.0)
place(0.18, 0.30, 0.05)                        → cube_in_container: true
```

---

## 5. When it doesn't go cleanly — reading feedback and recovering

Things go sideways; staying gentle while you recover is the whole game.

- **`find_object` reports `found: false`.** The object isn't visible from the
  fixed camera at all, or the prompt didn't match anything by colour/shape. Try
  a clearer prompt, or accept that this object currently isn't reachable by
  perception (don't fall back to a guessed coordinate).
- **`triangulate` returns `confidence < 0.6`.** The wrist views disagreed —
  usually because the coarse aim missed, or something moved. Re-run
  `find_object` from scratch rather than grasping on the low-confidence point.
- **`grasp` came back `is_grasping: false`.** It declined to force a bad grip
  (good — that's it being gentle, not failing you). Re-`triangulate` (the
  object may have shifted slightly), reconsider the grip width, and try once
  more. If it keeps declining the *same* object in the *same* way, that's
  usually telling you the shape is outside what the pads can hold (§6), not
  that the next attempt will be the lucky one.
- **The object moved when you didn't want it to.** `grasp` already backs off
  gently and re-homes on its own rather than chasing it. Re-run
  `find_object` → `triangulate` on the object's new position before trying again.
- **Carrying / placing dropped it.** Re-check `is_grasping` before committing to
  the carry; `place` already moves and descends carefully, so let it do the
  descent rather than improvising one with `move_to_position`.
- **The arm is in a weird pose or facing the wrong way.** `set_joint_angles` back
  to a sane configuration, then re-navigate. The base pan that faces a point at
  `(x, y)` is roughly `atan2(-y, x)` in degrees — note the minus on `y`.

The high-level `grasp`/`place` already encapsulate the careful version of all
this — most recovery is "re-perceive, then retry the high-level call," not
hand-driving individual joints.

---

## 6. Know your gripper — what flat pads can and can't hold

This is hardware reality, not a limitation to fight. The SO-101's two flat parallel
pads grip by squeezing opposite faces:

- **They hold box-like objects in roughly the 30–40 mm range well** — a face for
  each pad to press, the right size for the jaws. A default ~30 mm cube is the
  sweet spot, and a taller flat-faced box held high on its body is fine.
- **They struggle outside that band**, and the honest move is to *decline gently
  rather than bat the object*:
  - *Narrower than ~30 mm* → closing the wide jaws onto it tends to shove it.
  - *Short and wide* → an off-centre pad levers the wide face sideways.
  - *Round (spheres, cylinders)* → smooth surfaces roll out from between flat pads.
- A well-behaved attempt at one of these will **decline up front, before touching**,
  rather than chase a hold it can't get. `triangulate`'s measured `width_mm` (and
  its uncertainty) is what lets `grasp` make this call honestly, from what the
  camera actually saw — not from assumed geometry.

So: when you're picking the object, prefer a box-like target near cube scale, or
present a tall object to be gripped high. If the task hands you a sphere, it's
reasonable to report that this gripper can't securely hold it.

---

## 7. When the scene has several objects, or something is in the way

Nothing about the arm changes when the table is busy — the same gentleness, the
same gripper, the same verbs. What changes is that you have to *ask the cameras*
rather than assume:

- **There is no "list everything" tool.** `find_object` takes a prompt, not a
  survey — call it once per object you need, with a distinguishing phrase
  ("the red cube," "the blue box," "the small container"). If a prompt is
  ambiguous, say what you think it refers to and proceed on that.
- **Judge each object for yourself.** Whether something is a good grip is the
  same question as always (§6) — `triangulate` gives you the measured width to
  reason with. Decide per object; don't assume everything is graspable, or that
  nothing is.
- **Mind what's in the path.** If an obstacle sits between you and where an
  object needs to go, lift clear and route *over or around* it with
  `move_to_position` waypoints — raise `z` to a safe height, move across, then
  descend — rather than dragging straight through it. `place` already carries
  high; if you drive the move yourself, give the obstacle the same room.
- **One thing at a time.** Finish an object (find → triangulate → grasp → carry
  → place → confirm) before starting the next, and re-`find_object` if the
  scene may have shifted (a previous grasp can nudge a neighbour).

This is a general method — it reads the same whatever the objects, colours, or
layout turn out to be.

---

## 8. Composing other motions

There's no special tool for these — you build them by sequencing the verbs you
already have, with `move_to_position` and `set_joint_angles` as your primitives:

- **Move something over a path:** a sequence of `move_to_position` waypoints
  while holding it (keep `gain` low, e.g. 0.1–0.2, while carrying).
- **Avoid an obstacle:** raise `z` to clear it, route through safe intermediate
  `move_to_position` waypoints, then descend.
- **Re-orient without translating:** `set_joint_angles(wrist_roll_deg=...)`
  changes the gripper's facing without moving the end-effector.

If a task calls for a motion this toolbox doesn't cleanly express, say so rather
than forcing it through the wrong primitive.
