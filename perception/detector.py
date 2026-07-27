"""Honest, color-free object detection for the SO-101 perception pipeline.

Runs in the SCRATCH venv (torch + ultralytics). NO HSV / color threshold and NO
object qpos. FastSAM everything-mode produces class-agnostic masks; a geometric
"compact object resting in the reachable workspace" selector picks the target's
mask, and its centroid is returned in pixels. The 3D lift (triangulation /
plane / depth) is done by the caller from the camera FK pose.

Open-vocab note: FastSAM also accepts a CLIP text prompt (`texts=`), but on the
~30 px synthetic cube at side-cam range CLIP mis-selects (validated: it latched
onto a far region ~130 px off the true centroid). So the delivered primary is
the geometric on-table selector; `name` is carried through for API parity and a
future Grounding-DINO/OWL-ViT grounding layer (see the log).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image

# Workspace envelope (metres) — the reachable tabletop the object can sit in.
# Used only as a geometric gate on candidate masks (no object truth).
WS_X_MIN, WS_X_MAX = 0.15, 0.34
WS_Y_ABS = 0.34
WS_R_MAX = 0.36

# Mask geometry gates.
AREA_MIN_PX = 25
AREA_MAX_FRAC = 0.02     # reject table/arm/background/container (large masks)
BORDER_MARGIN = 3        # reject masks touching the frame edge (arm/table)
MIN_FILL = 0.68          # mask area / bbox area — compact objects only
CLUSTER_MM = 25.0        # deprojected-xy radius that groups masks of one object
COLOR_TIE_RGB = 45.0     # mean-RGB distance treated as "same colour" when picking by target_rgb
# Absolute colour-match ceiling. Without this, target_rgb selection returns the LEAST-BAD
# candidate even when the target is not in frame at all — a blue box confidently reported
# as "red cube". Measured separation on the sim palette: a true match scores 20-65 while
# the nearest wrong-colour object scores 227+, so 120 splits them with wide margin and
# turns "target absent" into an honest no-detection.
COLOR_MAX_RGB = 120.0


def _ray_plane(u, v, f, cx, cy, cam_pos, R_cw, z_plane):
    d_c = np.array([(u - cx) / f, -(v - cy) / f, -1.0])
    d_w = R_cw @ d_c
    if abs(d_w[2]) < 1e-9:
        return None
    t = (z_plane - cam_pos[2]) / d_w[2]
    if t <= 0.0:
        return None
    hit = cam_pos + t * d_w
    return float(hit[0]), float(hit[1])


class FastSAMDetector:
    """Loads FastSAM-s once; returns a color-free mask centroid per frame."""

    def __init__(self, weights: str = "FastSAM-s.pt", imgsz: int = 1024):
        from ultralytics import FastSAM
        self.model = FastSAM(weights)
        self.imgsz = imgsz

    # Close-range (wrist) gates. At ~0.11 m standoff a 30 mm object is ~92 px wide
    # (~8.5k px area) — far over the side-view AREA_MAX_FRAC, and the near-top-down
    # silhouette of a cylinder/sphere is less box-filling. So the wrist view needs its
    # own envelope; these are passed explicitly by the wrist stage, never guessed here.
    WRIST_GATES = {"area_max_frac": 0.25, "area_min_px": 400, "min_fill": 0.55}

    def _masks(self, img_rgb: np.ndarray):
        res = self.model.predict(img_rgb, device="cpu", retina_masks=True,
                                 imgsz=self.imgsz, conf=0.25, iou=0.9,
                                 verbose=False)
        r = res[0]
        if r.masks is None:
            return []
        data = r.masks.data.cpu().numpy()
        H, W = img_rgb.shape[:2]
        out = []
        for m in data:
            mm = m > 0.5
            if mm.shape != (H, W):
                mm = np.array(Image.fromarray(mm.astype(np.uint8) * 255)
                              .resize((W, H))) > 127
            out.append(mm)
        return out

    def detect(self, img_rgb: np.ndarray, cam, z_plane: float = 0.015,
               prefer_center: bool = False,
               near_xy: Optional[tuple] = None,
               target_rgb: Optional[tuple] = None,
               area_max_frac: float = AREA_MAX_FRAC,
               area_min_px: int = AREA_MIN_PX,
               min_fill: float = MIN_FILL,
               in_workspace: bool = True,
               debug: bool = False) -> Optional[dict]:
        """Return {uv, area, fill, xy_plane, votes} for the selected object.

        cam = (f, cx, cy, cam_pos, R_cw). z_plane is the support plane used only
        to gate candidate masks into the reachable workspace (geometry, not
        truth). FastSAM everything-mode over-segments: the target object shows up
        as a tight CLUSTER of overlapping compact masks (object core + shadow +
        sub-faces), while spurious blobs are isolated. So:

          1. keep compact, small, non-border, in-workspace masks;
          2. cluster them by deprojected (x,y);
          3. pick the winning cluster:
             * near_xy given (wrist view / disambiguation): the cluster whose
               deprojected support-plane xy is CLOSEST to near_xy. On-table
               objects deproject near the prior; the gripper (well above the
               table) deprojects far, so this rejects it WITHOUT using colour.
             * else: the cluster with the most member masks (a consensus vote).
          4. within the cluster pick a member:
             * near_xy given: the highest-fill member (cleanest silhouette);
             * side view: the TOPMOST-centroid member (min v) — the ground
               shadow attaches BELOW the object, so the highest member has the
               least shadow bleed.
        """
        f, cx, cy, cam_pos, R_cw = cam
        cam_pos = np.asarray(cam_pos)
        R_cw = np.asarray(R_cw)
        H, W = img_rgb.shape[:2]
        area_max = area_max_frac * H * W
        cand = []
        rejected: list[dict] = []
        for mm in self._masks(img_rgb):
            area = int(mm.sum())
            if area < area_min_px or area > area_max:
                if debug:
                    rejected.append({"area": area, "why": "area"})
                continue
            ys, xs = np.where(mm)
            x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
            if (x0 <= BORDER_MARGIN or y0 <= BORDER_MARGIN
                    or x1 >= W - 1 - BORDER_MARGIN or y1 >= H - 1 - BORDER_MARGIN):
                if debug:
                    rejected.append({"area": area, "why": "border"})
                continue
            bbox_area = (x1 - x0 + 1) * (y1 - y0 + 1)
            fill = area / max(bbox_area, 1)
            if fill < min_fill:
                if debug:
                    rejected.append({"area": area, "fill": round(fill, 3), "why": "fill"})
                continue
            u, v = float(xs.mean()), float(ys.mean())
            xy = _ray_plane(u, v, f, cx, cy, cam_pos, R_cw, z_plane)
            if xy is None:
                if debug:
                    rejected.append({"area": area, "why": "ray"})
                continue
            r = (xy[0] ** 2 + xy[1] ** 2) ** 0.5
            if in_workspace and not (WS_X_MIN < xy[0] < WS_X_MAX and abs(xy[1]) < WS_Y_ABS
                                     and r < WS_R_MAX):
                if debug:
                    rejected.append({"area": area, "why": "workspace",
                                     "xy": [round(xy[0], 3), round(xy[1], 3)]})
                continue
            rgb = img_rgb[mm].reshape(-1, 3).mean(axis=0)
            # bottom-centre pixel of the mask -> floor contact (the shape-free (x,y) locator)
            vb = int(ys.max()); ub = float(xs[ys >= vb - 2].mean())
            xy_floor = _ray_plane(ub, vb, f, cx, cy, cam_pos, R_cw, 0.0)
            cand.append({"uv": [u, v], "area": area, "fill": fill,
                         "bbox": [int(x0), int(y0), int(x1), int(y1)],
                         "xy_plane": [xy[0], xy[1]],
                         "xy_floor": None if xy_floor is None else [xy_floor[0], xy_floor[1]],
                         "rgb": [float(c) for c in rgb]})
        if not cand:
            return {"uv": None, "candidates": [], "rejected": rejected} if debug else None

        def _dbg(pick: dict) -> dict:
            if debug:
                pick["candidates"] = [{k: c[k] for k in ("uv", "area", "fill", "rgb", "xy_plane")}
                                      for c in cand]
                pick["rejected"] = rejected
            return pick

        # Colour-match selection: the candidates are already compact, in-workspace, non-border
        # object masks (floor/arm excluded by the gates above), so picking the closest-colour one
        # is safe — colour only chooses WHICH object, it never has to fight the background.
        if target_rgb is not None:
            t = np.asarray(target_rgb, float)
            def cdist(c):
                return float(np.linalg.norm(np.asarray(c["rgb"]) - t))
            ordered = sorted(cand, key=cdist)
            best_d = cdist(ordered[0])
            if best_d > COLOR_MAX_RGB:
                # Nothing in frame is the requested colour -> say so, rather than
                # returning whichever object happened to be least unlike it.
                return _dbg({"uv": None, "color_dist": round(best_d, 1),
                             "stage": "no_colour_match"}) if debug else None
            near = [c for c in ordered if cdist(c) < best_d + COLOR_TIE_RGB]
            if near_xy is not None:
                # Colour narrows to the right OBJECT; the spatial prior then picks the right
                # INSTANCE (and, in the wrist view, rejects a same-colour blob elsewhere in
                # frame). "Topmost" is a side-view shadow heuristic and is meaningless in a
                # near-top-down view, so proximity wins whenever a prior is available.
                nx, ny = near_xy
                pick = min(near, key=lambda m: (m["xy_plane"][0] - nx) ** 2
                           + (m["xy_plane"][1] - ny) ** 2)
            else:
                pick = min(near, key=lambda m: m["uv"][1])  # topmost => least ground-shadow bleed
            pick = dict(pick); pick["votes"] = len(near); pick["color_dist"] = round(best_d, 1)
            return _dbg(pick)

        # Greedy clustering by deprojected (x,y).
        clusters: list[list[dict]] = []
        for c in cand:
            cx_m, cy_m = c["xy_plane"]
            placed = False
            for cl in clusters:
                mx = np.mean([m["xy_plane"][0] for m in cl])
                my = np.mean([m["xy_plane"][1] for m in cl])
                if ((cx_m - mx) ** 2 + (cy_m - my) ** 2) ** 0.5 * 1000 < CLUSTER_MM:
                    cl.append(c)
                    placed = True
                    break
            if not placed:
                clusters.append([c])

        if near_xy is not None:
            nx, ny = near_xy

            def cl_dist(cl):
                mx = np.mean([m["xy_plane"][0] for m in cl])
                my = np.mean([m["xy_plane"][1] for m in cl])
                return (mx - nx) ** 2 + (my - ny) ** 2

            best = min(clusters, key=cl_dist)
            pick = max(best, key=lambda m: m["fill"])     # cleanest silhouette
        else:
            # Consensus: most-voted cluster; tie-break by best single fill.
            clusters.sort(key=lambda cl: (len(cl), max(m["fill"] for m in cl)),
                          reverse=True)
            best = clusters[0]
            pick = min(best, key=lambda m: m["uv"][1])    # topmost => least shadow
        pick = dict(pick)
        pick["votes"] = len(best)
        return _dbg(pick)
