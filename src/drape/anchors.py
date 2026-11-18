"""Where each kind of item sits on the body.

Pure geometry: takes landmark arrays in PIXELS (from MediaPipe face / pose / hand landmarkers) and returns
placements: the target point, the width the item should span, and its rotation. No MediaPipe import here,
so this is easy to test.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

# MediaPipe Face Landmarker indices (478-point mesh)
F_FOREHEAD, F_CHIN = 10, 152
F_EYE_R_OUT, F_EYE_L_OUT = 33, 263      # subject's right / left outer eye corners
F_BRIDGE = 168                          # between the eyes, top of the nose bridge
F_TRAGUS_R, F_TRAGUS_L = 234, 454       # face edge at ear level (subject's right / left)

# MediaPipe Pose Landmarker indices
P_SHOULDER_L, P_SHOULDER_R = 11, 12

# MediaPipe Hand Landmarker indices
H_WRIST, H_INDEX_MCP, H_MIDDLE_MCP, H_RING_MCP, H_RING_PIP, H_PINKY_MCP = 0, 5, 9, 13, 14, 17

KINDS = ("necklace", "earrings", "glasses", "ring", "wrist")


@dataclass
class Placement:
    x: float            # target point (pixels) where the item's anchor goes
    y: float
    width: float        # pixels the item's width should span
    angle: float        # degrees, counter-clockwise, 0 = upright
    mirror: bool = False


def _pt(a: np.ndarray, i: int) -> np.ndarray:
    return np.asarray(a[i][:2], dtype=float)


def _angle(p: np.ndarray, q: np.ndarray) -> float:
    """Angle of the line p->q relative to horizontal, in degrees CCW (image y points down)."""
    return -math.degrees(math.atan2(q[1] - p[1], q[0] - p[0]))


def _face_up(face: np.ndarray) -> np.ndarray:
    v = _pt(face, F_FOREHEAD) - _pt(face, F_CHIN)
    return v / (np.linalg.norm(v) or 1.0)


def glasses(face: np.ndarray) -> Placement:
    r, l = _pt(face, F_TRAGUS_R), _pt(face, F_TRAGUS_L)
    eye_r, eye_l = _pt(face, F_EYE_R_OUT), _pt(face, F_EYE_L_OUT)
    b = _pt(face, F_BRIDGE)
    return Placement(b[0], b[1], 1.1 * float(np.linalg.norm(l - r)), _angle(eye_r, eye_l))


def earrings(face: np.ndarray) -> List[Placement]:
    """One placement per ear lobe: below the tragus along the face's down axis, nudged outward."""
    up = _face_up(face)
    r, l = _pt(face, F_TRAGUS_R), _pt(face, F_TRAGUS_L)
    face_h = float(np.linalg.norm(_pt(face, F_FOREHEAD) - _pt(face, F_CHIN)))
    face_w = float(np.linalg.norm(l - r))
    ang = _angle(_pt(face, F_EYE_R_OUT), _pt(face, F_EYE_L_OUT))
    out_dir = (l - r) / (face_w or 1.0)
    width = 0.13 * face_w
    lobe_r = r - up * 0.16 * face_h - out_dir * 0.03 * face_w
    lobe_l = l - up * 0.16 * face_h + out_dir * 0.03 * face_w
    return [Placement(lobe_r[0], lobe_r[1], width, ang), Placement(lobe_l[0], lobe_l[1], width, ang, mirror=True)]


def necklace(pose: np.ndarray, face: Optional[np.ndarray] = None) -> Placement:
    """Top of the necklace sits at the base of the neck: part-way from chin to the shoulder midpoint."""
    sl, sr = _pt(pose, P_SHOULDER_L), _pt(pose, P_SHOULDER_R)
    mid = (sl + sr) / 2
    span = float(np.linalg.norm(sl - sr))
    if face is not None:
        chin = _pt(face, F_CHIN)
        top = chin + 0.45 * (mid - chin)
    else:
        top = mid - np.array([0.0, 0.22 * span])
    return Placement(top[0], top[1], 0.46 * span, _angle(sr, sl))


def ring(hand: np.ndarray) -> Placement:
    mcp, pip = _pt(hand, H_RING_MCP), _pt(hand, H_RING_PIP)
    c = mcp + 0.38 * (pip - mcp)
    finger_w = 0.9 * float(np.linalg.norm(_pt(hand, H_RING_MCP) - _pt(hand, H_MIDDLE_MCP)))
    along = _angle(mcp, pip)            # finger axis; the ring band lies across it
    return Placement(c[0], c[1], finger_w, along - 90.0 if along > -90 else along + 270.0)


def wrist(hand: np.ndarray) -> Placement:
    w, m = _pt(hand, H_WRIST), _pt(hand, H_MIDDLE_MCP)
    palm_w = float(np.linalg.norm(_pt(hand, H_INDEX_MCP) - _pt(hand, H_PINKY_MCP)))
    axis = (m - w) / (np.linalg.norm(m - w) or 1.0)
    c = w - axis * 0.18 * palm_w          # a little down the forearm from the wrist joint
    along = _angle(w, m)
    return Placement(c[0], c[1], 1.05 * palm_w, along - 90.0)


def place(kind: str, face=None, pose=None, hands: Optional[List[np.ndarray]] = None) -> List[Placement]:
    """All placements for an item kind given whatever landmarks were found (empty list = can't place)."""
    hands = hands or []
    if kind == "glasses" and face is not None:
        return [glasses(face)]
    if kind == "earrings" and face is not None:
        return earrings(face)
    if kind == "necklace" and pose is not None:
        return [necklace(pose, face)]
    if kind == "ring":
        return [ring(h) for h in hands]
    if kind == "wrist":
        return [wrist(h) for h in hands]
    return []


class Smoother:
    """Exponential smoothing of placements per slot, so overlays don't jitter frame to frame."""

    def __init__(self, alpha: float = 0.45) -> None:
        self.alpha = alpha
        self._prev: Dict[tuple, Placement] = {}

    def __call__(self, key: tuple, p: Placement) -> Placement:
        q = self._prev.get(key)
        if q is not None:
            a = self.alpha
            da = (p.angle - q.angle + 180) % 360 - 180
            p = Placement(q.x + a * (p.x - q.x), q.y + a * (p.y - q.y), q.width + a * (p.width - q.width),
                          q.angle + a * da, p.mirror)
        self._prev[key] = p
        return p

    def reset(self) -> None:
        self._prev.clear()
