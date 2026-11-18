"""Where each kind of item sits on the body, in 3D.

Pure geometry: takes landmark arrays in PIXELS, (N, 3) with z in the same pixel scale as x (MediaPipe's z is
relative depth; smaller = closer to the camera), and returns placements. No MediaPipe import here, so this
is easy to test.

A flat (photo) item is placed as a plane: an origin where its attach point goes, two unit 3D vectors for the
item's right and down directions, and its width. Projecting that plane gives a perspective-correct quad, so
glasses foreshorten when you turn your head and earrings stay hanging straight down while turning with it.

A 3D (mesh) item gets a full pose: rotation, per-axis scale, and "occluders" (finger / wrist cylinders that
hide the parts of the band that go behind them).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# MediaPipe Face Landmarker indices (478-point mesh)
F_FOREHEAD, F_CHIN = 10, 152
F_EYE_R_OUT, F_EYE_L_OUT = 33, 263      # subject's right / left outer eye corners
F_BRIDGE = 168                          # between the eyes, top of the nose bridge
F_TRAGUS_R, F_TRAGUS_L = 234, 454       # face edge at ear level (subject's right / left)
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
             176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# MediaPipe Pose Landmarker indices
P_SHOULDER_L, P_SHOULDER_R = 11, 12

# MediaPipe Hand Landmarker indices
H_WRIST, H_INDEX_MCP, H_MIDDLE_MCP, H_MIDDLE_PIP, H_RING_MCP, H_RING_PIP, H_PINKY_MCP, H_PINKY_PIP = \
    0, 5, 9, 10, 13, 14, 17, 18

KINDS = ("necklace", "earrings", "glasses", "ring", "wrist")
DOWN = np.array([0.0, 1.0, 0.0])


# ---------------------------------------------------------------------------------------------- types
@dataclass
class Capsule:
    """A cylinder that hides whatever of a 3D item passes behind it (a finger, the wrist)."""
    p0: np.ndarray
    p1: np.ndarray
    radius: float


@dataclass
class Pose3D:
    """Where a 3D item goes: its canonical +X/+Y/+Z axes (columns of R) placed at origin, scaled per axis.
    Canonical item frame: the finger / wrist runs along +Z, +Y is the back of the hand (stone, watch dial),
    and the band's inner radius is 1."""
    origin: np.ndarray
    R: np.ndarray                   # 3x3, columns = item X, Y, Z in pixel space
    scale: np.ndarray               # (sx, sy, sz)
    occluder_scale: np.ndarray      # the finger/wrist inside the band, in the same frame (unit cylinder along Z)
    occluder_length: float          # px, along Z
    others: List[Capsule] = field(default_factory=list)   # neighbouring fingers


@dataclass
class Placement:
    origin: np.ndarray              # (3,) px; the item's attach point goes here
    ex: np.ndarray                  # (3,) unit: the item's right
    ey: np.ndarray                  # (3,) unit: the item's down
    width: float                    # px the item's width spans (in 3D, before foreshortening)
    mirror: bool = False
    opacity: float = 1.0
    pose3d: Optional[Pose3D] = None
    occlude: Tuple[str, ...] = ()   # masks that may hide it: "hair", "face"

    @property
    def x(self) -> float:
        return float(self.origin[0])

    @property
    def y(self) -> float:
        return float(self.origin[1])

    @property
    def angle(self) -> float:
        """In-image rotation in degrees CCW (0 = upright), from the projected right vector."""
        return -math.degrees(math.atan2(self.ex[1], self.ex[0]))

    def quad(self, w: float, h: float, ax: float, ay: float) -> np.ndarray:
        """Image corners (TL, TR, BR, BL) of an item w x h px whose attach point is (ax, ay)."""
        s = self.width / max(w, 1e-6)
        pts = []
        for u, v in ((0, 0), (w, 0), (w, h), (0, h)):
            p = self.origin + (u - ax) * s * self.ex + (v - ay) * s * self.ey
            pts.append(p[:2])
        return np.array(pts, dtype=np.float32)


def flat(x: float, y: float, width: float, angle: float, **kw) -> Placement:
    """A placement in the image plane, rotated `angle` degrees CCW."""
    t = math.radians(angle)
    return Placement(np.array([x, y, 0.0]), np.array([math.cos(t), -math.sin(t), 0.0]),
                     np.array([math.sin(t), math.cos(t), 0.0]), width, **kw)


# ---------------------------------------------------------------------------------------------- helpers
def _p(a: np.ndarray, i: int) -> np.ndarray:
    v = np.zeros(3)
    row = np.asarray(a[i], dtype=float)
    v[: min(3, len(row))] = row[:3]
    return v


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def _ortho(v: np.ndarray, against: np.ndarray) -> np.ndarray:
    return _unit(v - np.dot(v, against) * against)


def _angle2d(p: np.ndarray, q: np.ndarray) -> float:
    return -math.degrees(math.atan2(q[1] - p[1], q[0] - p[0]))


def face_frame(face: np.ndarray):
    """Head axes in pixel space: right (subject's right ear -> left ear, i.e. image left -> right), down, and the
    3D face width/height."""
    r, l = _p(face, F_TRAGUS_R), _p(face, F_TRAGUS_L)
    ex = _unit(l - r)
    up = _ortho(_p(face, F_FOREHEAD) - _p(face, F_CHIN), ex)
    return ex, -up, float(np.linalg.norm(l - r)), float(np.linalg.norm(_p(face, F_FOREHEAD) - _p(face, F_CHIN)))


# ---------------------------------------------------------------------------------------------- face items
def glasses(face: np.ndarray) -> Placement:
    ex, ey, fw, _ = face_frame(face)
    return Placement(_p(face, F_BRIDGE), ex, ey, 1.1 * fw)


def earrings(face: np.ndarray) -> List[Placement]:
    """One per ear lobe. Earrings hang straight down (gravity) but turn with the head; the one on the ear that
    turns away from the camera fades out as the head hides it."""
    ex, ey, fw, fh = face_frame(face)
    r, l = _p(face, F_TRAGUS_R), _p(face, F_TRAGUS_L)
    lobe_r = r + ey * 0.16 * fh - ex * 0.03 * fw
    lobe_l = l + ey * 0.16 * fh + ex * 0.03 * fw
    side = _ortho(np.array([ex[0], 0.0, ex[2]]), DOWN) if abs(ex[0]) + abs(ex[2]) > 1e-6 else np.array([1.0, 0, 0])
    width = 0.13 * fw

    def vis(away: float) -> float:          # away = sin(yaw) toward hiding this ear
        return float(np.clip((0.5 - away) / 0.25, 0.0, 1.0))

    return [Placement(lobe_r, side, DOWN.copy(), width, opacity=vis(-ex[2]), occlude=("hair",)),
            Placement(lobe_l, side, DOWN.copy(), width, mirror=True, opacity=vis(ex[2]), occlude=("hair",))]


def necklace(pose: np.ndarray, face: Optional[np.ndarray] = None) -> Placement:
    """Top of the necklace sits at the base of the neck: part-way from chin to the shoulder midpoint. The chin
    and hair can cover it."""
    sl, sr = _p(pose, P_SHOULDER_L), _p(pose, P_SHOULDER_R)
    sl[2] *= 0.5                                     # pose depth is noisy; damp it
    sr[2] *= 0.5
    mid = (sl + sr) / 2
    span = float(np.linalg.norm((sl - sr)[:2]))
    if face is not None:
        chin = _p(face, F_CHIN)
        top = chin + 0.45 * (mid - chin)
        top[2] = mid[2]
    else:
        top = mid - np.array([0.0, 0.22 * span, 0.0])
    ex = _unit(sl - sr)
    return Placement(top, ex, _ortho(DOWN, ex), 0.46 * float(np.linalg.norm(sl - sr)), occlude=("hair", "face"))


# ---------------------------------------------------------------------------------------------- hand items
def dorsal(hand: np.ndarray, label: str = "") -> np.ndarray:
    """Unit vector out of the BACK of the hand. MediaPipe's handedness label describes how the hand looks in
    the image ('Right' = looks like a left hand would in a non-mirrored photo), which fixes the sign."""
    w, i, p = _p(hand, H_WRIST), _p(hand, H_INDEX_MCP), _p(hand, H_PINKY_MCP)
    n = _unit(np.cross(i - w, p - w))   # out of the BACK of the hand for a hand MediaPipe labels 'Left'
    return -n if label == "Right" else n


def _frame(origin, along, back) -> np.ndarray:
    z = _unit(along)
    y = _ortho(back, z)
    x = np.cross(y, z)
    return np.stack([x, y, z], axis=1)


def ring(hand: np.ndarray, label: str = "") -> Placement:
    mcp, pip = _p(hand, H_RING_MCP), _p(hand, H_RING_PIP)
    c = mcp + 0.38 * (pip - mcp)
    finger_w = 0.9 * float(np.linalg.norm(_p(hand, H_RING_MCP) - _p(hand, H_MIDDLE_MCP)))
    r = finger_w / 2
    R = _frame(c, pip - mcp, dorsal(hand, label))
    seg = float(np.linalg.norm(pip - mcp))
    others = [Capsule(_p(hand, a), _p(hand, b), 0.95 * r)
              for a, b in ((H_MIDDLE_MCP, H_MIDDLE_PIP), (H_PINKY_MCP, H_PINKY_PIP))]
    pose = Pose3D(c, R, np.array([r, r, r]), np.array([0.97 * r, 0.97 * r]), 1.6 * seg, others)
    ex = R[:, 0] if R[0, 0] >= 0 else -R[:, 0]            # flat photo rings: lie across the finger, upright
    return Placement(c, ex, _ortho(_unit(pip - mcp) * -1, ex), finger_w, pose3d=pose)


def wrist(hand: np.ndarray, label: str = "") -> Placement:
    w, m = _p(hand, H_WRIST), _p(hand, H_MIDDLE_MCP)
    palm_w = float(np.linalg.norm(_p(hand, H_INDEX_MCP) - _p(hand, H_PINKY_MCP)))
    axis = _unit(m - w)
    c = w - axis * 0.18 * palm_w          # a little down the forearm from the wrist joint
    R = _frame(c, axis, dorsal(hand, label))
    rx, ry = 0.42 * palm_w, 0.30 * palm_w  # the wrist is wider than it is thick
    pose = Pose3D(c, R, np.array([rx, ry, rx]), np.array([0.97 * rx, 0.97 * ry]), 2.5 * palm_w)
    ex = R[:, 0] if R[0, 0] >= 0 else -R[:, 0]
    return Placement(c, ex, _ortho(-axis, ex), 1.05 * palm_w, pose3d=pose)


def place(kind: str, face=None, pose=None, hands: Optional[Sequence[np.ndarray]] = None,
          hand_labels: Optional[Sequence[str]] = None) -> List[Placement]:
    """All placements for an item kind given whatever landmarks were found (empty list = can't place)."""
    hands = list(hands or [])
    labels = list(hand_labels or []) + [""] * len(hands)
    if kind == "glasses" and face is not None:
        return [glasses(face)]
    if kind == "earrings" and face is not None:
        return earrings(face)
    if kind == "necklace" and pose is not None:
        return [necklace(pose, face)]
    if kind == "ring":
        return [ring(h, lb) for h, lb in zip(hands, labels)]
    if kind == "wrist":
        return [wrist(h, lb) for h, lb in zip(hands, labels)]
    return []


# ---------------------------------------------------------------------------------------------- smoothing
class Smoother:
    """Exponential smoothing of placements per slot, so overlays don't jitter frame to frame."""

    def __init__(self, alpha: float = 0.45) -> None:
        self.alpha = alpha
        self._prev: Dict[tuple, Placement] = {}

    def __call__(self, key: tuple, p: Placement) -> Placement:
        q = self._prev.get(key)
        if q is not None:
            a = self.alpha
            origin = q.origin + a * (p.origin - q.origin)
            ex = _unit(q.ex + a * (p.ex - q.ex))
            ey = _ortho(q.ey + a * (p.ey - q.ey), ex)
            pose3d = p.pose3d
            if p.pose3d is not None and q.pose3d is not None:
                P, Q = p.pose3d, q.pose3d
                o3 = Q.origin + a * (P.origin - Q.origin)
                z = _unit(Q.R[:, 2] + a * (P.R[:, 2] - Q.R[:, 2]))
                y = _ortho(Q.R[:, 1] + a * (P.R[:, 1] - Q.R[:, 1]), z)
                shift = o3 - P.origin                     # move the neighbour fingers with the item
                pose3d = replace(P, origin=o3, R=np.stack([np.cross(y, z), y, z], axis=1),
                                 scale=Q.scale + a * (P.scale - Q.scale),
                                 others=[Capsule(c.p0 + shift, c.p1 + shift, c.radius) for c in P.others])
            p = replace(p, origin=origin, ex=ex, ey=ey, width=q.width + a * (p.width - q.width),
                        opacity=q.opacity + a * (p.opacity - q.opacity), pose3d=pose3d)
        self._prev[key] = p
        return p

    def reset(self) -> None:
        self._prev.clear()
