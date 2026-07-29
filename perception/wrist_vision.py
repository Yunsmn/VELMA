"""Object detection in the real wrist camera, for the hardware backend.

The simulation perception path (find_object_depth + the torch sidecar) renders a
fixed side camera and a MuJoCo depth model. On the physical rig neither exists:
there is one USB camera on the wrist, and the model<->servo mapping is not yet
accurate enough to turn a pixel into a trustworthy metric coordinate.

So this module deliberately stops at IMAGE SPACE. It reports where a thing is in
the frame and how big it looks, and says nothing about metres. That is enough to
servo the gripper onto a target visually, and it is honest about what one
uncalibrated view can actually support. Apparent radius doubles as a relative
distance cue — it grows as the camera closes in — without claiming to be a depth
measurement.

Pure OpenCV: no torch, so it runs in the same interpreter as the MCP server.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

# The gripper's own jaws occupy the bottom of the wrist view and are pale
# plastic; excluded so they are never picked as the target.
GRIPPER_BAND_FRAC = 0.62
MIN_AREA_PX = 800
MAX_AREA_FRAC = 0.35
BACKGROUND_PERCENTILE = 96.0
MAX_ASPECT = 3.0          # rejects the power cable and other long thin streaks

# Circle fit (the adapter is round; other targets simply won't fit one).
HOUGH_DP = 1.0
HOUGH_MIN_DIST = 120
HOUGH_PARAM1 = 60
HOUGH_PARAM2 = 45
HOUGH_MIN_R = 30
HOUGH_MAX_R = 300


@dataclass(frozen=True)
class Detection:
    u: float
    v: float
    area_px: float
    bbox: tuple[int, int, int, int]
    radius_px: float
    circular: bool
    score: float
    extras: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "u": round(self.u, 1),
            "v": round(self.v, 1),
            "area_px": int(self.area_px),
            "bbox": [int(v) for v in self.bbox],
            "radius_px": round(self.radius_px, 1),
            "circular": self.circular,
            "score": round(self.score, 3),
            **self.extras,
        }


def background_distance(bgr: np.ndarray) -> np.ndarray:
    """Per-pixel distance from the dominant background colour.

    The table is a large uniform region, so its colour is the image median. An
    object differs from it either in hue/saturation or in brightness; taking the
    larger of the two catches both a coloured object and a dark or pale one,
    without hard-coding what colour the target is.
    """
    blurred = cv2.GaussianBlur(bgr, (9, 9), 0)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB).astype(np.float32)
    background = np.median(lab.reshape(-1, 3), axis=0)
    return np.linalg.norm(lab - background, axis=2)


def detect(bgr: np.ndarray, max_results: int = 5) -> list[Detection]:
    """Rank things on the table that are not the table."""
    height, width = bgr.shape[:2]
    distance = background_distance(bgr)

    search = distance.copy()
    search[int(height * GRIPPER_BAND_FRAC):, :] = 0.0

    threshold = np.percentile(search, BACKGROUND_PERCENTILE)
    mask = (search > threshold).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circles = _hough_circles(bgr)

    found: list[Detection] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < MIN_AREA_PX or area > MAX_AREA_FRAC * height * width:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        aspect = max(w, h) / max(1.0, min(w, h))
        if aspect > MAX_ASPECT:
            continue

        moments = cv2.moments(contour)
        if moments["m00"] <= 0:
            continue
        cu = moments["m10"] / moments["m00"]
        cv_ = moments["m01"] / moments["m00"]

        circle = _nearest_circle(circles, cu, cv_)
        if circle is not None:
            cu, cv_, radius = circle
            circular = True
        else:
            radius = 0.5 * (w + h) / 2
            circular = False

        # Mean background distance over the blob: how confidently it is "not table".
        blob = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(blob, [contour], -1, 255, -1)
        contrast = float(distance[blob > 0].mean())

        found.append(Detection(
            u=cu, v=cv_, area_px=area, bbox=(x, y, w, h),
            radius_px=float(radius), circular=circular,
            score=contrast,
            extras={"aspect": round(aspect, 2)},
        ))

    found.sort(key=lambda d: d.score * np.sqrt(d.area_px), reverse=True)
    return found[:max_results]


def _hough_circles(bgr: np.ndarray) -> Optional[np.ndarray]:
    grey = cv2.medianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), 5)
    circles = cv2.HoughCircles(
        grey, cv2.HOUGH_GRADIENT, dp=HOUGH_DP, minDist=HOUGH_MIN_DIST,
        param1=HOUGH_PARAM1, param2=HOUGH_PARAM2,
        minRadius=HOUGH_MIN_R, maxRadius=HOUGH_MAX_R,
    )
    return None if circles is None else np.round(circles[0]).astype(float)


def _nearest_circle(circles: Optional[np.ndarray], u: float, v: float,
                    tolerance_px: float = 90.0):
    """A circle fit is only accepted if it agrees with the blob it claims to be."""
    if circles is None:
        return None
    best, best_distance = None, tolerance_px
    for cu, cv_, radius in circles:
        distance = float(np.hypot(cu - u, cv_ - v))
        if distance < best_distance:
            best, best_distance = (cu, cv_, radius), distance
    return best


def annotate(bgr: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Draw detections and the frame centre, for eyeballing what was found."""
    vis = bgr.copy()
    height, width = vis.shape[:2]
    cv2.drawMarker(vis, (width // 2, height // 2), (255, 0, 0),
                   cv2.MARKER_TILTED_CROSS, 34, 2)
    for rank, det in enumerate(detections):
        colour = (0, 0, 255) if rank == 0 else (0, 165, 255)
        x, y, w, h = det.bbox
        cv2.rectangle(vis, (x, y), (x + w, y + h), colour, 2)
        if det.circular:
            cv2.circle(vis, (int(det.u), int(det.v)), int(det.radius_px), colour, 2)
        cv2.drawMarker(vis, (int(det.u), int(det.v)), (0, 255, 0),
                       cv2.MARKER_CROSS, 30, 2)
        cv2.putText(vis, f"#{rank} s={det.score:.0f}", (x, max(18, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
    return vis
