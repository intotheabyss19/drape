"""Where each kind of item sits on the body, in 3D.

Pure geometry: takes landmark arrays in PIXELS, (N, 3) with z in the same pixel scale as x (MediaPipe's z is
relative depth; smaller = closer to the camera), and returns placements. No MediaPipe import here, so this
is easy to test.

Sizes are real-world: the face's width gives pixels per metre, so a 22 mm hoop or 15 cm glasses come out the
right size on anyone at any distance.

A photo item is placed as a plane (origin, right and down vectors, width), drawn in perspective. A 3D item
gets a full pose (rotation, scale) plus "occluders": simple shapes for the head, neck, finger or wrist that
hide the parts of the item behind them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

# MediaPipe Face Landmarker indices (478-point mesh)
F_FOREHEAD, F_CHIN = 10, 152
F_EYE_R_OUT, F_EYE_L_OUT = 33, 263      # subject's right / left outer eye corners
F_EYE_R_IN, F_EYE_L_IN = 133, 362
F_BRIDGE = 168                          # between the eyes, top of the nose bridge
F_TRAGUS_R, F_TRAGUS_L = 234, 454       # face edge at eye level (subject's right / left)
F_JAW_R, F_JAW_L = 132, 361             # face edge at the level of the ear lobes
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
             176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# MediaPipe Pose Landmarker indices
P_SHOULDER_L, P_SHOULDER_R = 11, 12

# MediaPipe Hand Landmarker indices
H_WRIST, H_INDEX_MCP, H_MIDDLE_MCP, H_MIDDLE_PIP, H_RING_MCP, H_RING_PIP, H_PINKY_MCP, H_PINKY_PIP = \
    0, 5, 9, 10, 13, 14, 17, 18

KINDS = ("necklace", "earrings", "glasses", "ring", "wrist")
DOWN = np.array([0.0, 1.0, 0.0])

# Typical adult dimensions (metres)
HEAD_W = 0.140          # between face-mesh points 234 and 454 (the sides of the face at eye level)
SHOULDER_W = 0.37       # between the pose shoulder points
NECK_R = 0.062          # where a necklace rests round the base of the neck
GLASSES_GAP = 0.004     # glasses' front is this much wider than the face

# ---------------------------------------------------------------------------------------------- types
@dataclass
class Pose3D:
    """Where a 3D item goes: its model axes (columns of R) placed at origin, scaled per axis (pixels per model
    unit). occluders: (shape, 4x4 matrix) for unit "cylinder" (radius 1, length 1 along Z) or "sphere" shapes,
    drawn into the depth buffer first so they hide what's behind them."""
    origin: np.ndarray
    R: np.ndarray                   # 3x3, columns = model X, Y, Z in pixel space
    scale: np.ndarray               # (3,)
    occluders: List[Tuple[str, np.ndarray]] = field(default_factory=list)


def affine(axes: np.ndarray, scale, origin) -> np.ndarray:
    """4x4 matrix placing a unit shape: columns of `axes` scaled by `scale`, at `origin`."""
    m = np.eye(4)
    m[:3, :3] = np.asarray(axes, float) * np.asarray(scale, float)[None, :]
    m[:3, 3] = origin
    return m


def capsule(p0: np.ndarray, p1: np.ndarray, r: float) -> Tuple[str, np.ndarray]:
    """A cylinder occluder from p0 to p1 (a finger)."""
    z = p1 - p0
    L = float(np.linalg.norm(z)) or 1.0
    z = z / L
    a = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
    x = _unit(np.cross(a, z))
    return "cylinder", affine(np.stack([x, np.cross(z, x), z], axis=1), (r, r, L * 1.15), (p0 + p1) / 2)


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


def head(face: np.ndarray):
    """(R, ppm): head axes as columns (x right, y down, z into the head) and pixels per metre."""
    ex, ey, fw, _ = face_frame(face)
    return np.stack([ex, ey, np.cross(ex, ey)], axis=1), fw / HEAD_W


def head_occluder(face: np.ndarray) -> Tuple[str, np.ndarray]:
    """An ellipsoid for the skull, a little narrower than the face so the ears and the glasses' arms stay
    outside it, and well behind the eyes so the lenses stay in front of it."""
    R, ppm = head(face)
    mid = (_p(face, F_TRAGUS_R) + _p(face, F_TRAGUS_L)) / 2
    c = mid + R[:, 2] * 0.012 * ppm - R[:, 1] * 0.01 * ppm
    return "sphere", affine(R, (0.93 * HEAD_W / 2 * ppm, 0.125 * ppm, 0.098 * ppm), c)


def glasses(face: np.ndarray) -> Placement:
    R, ppm = head(face)
    ex, ey, ez = R.T
    eyes = np.mean([_p(face, i) for i in (F_EYE_R_OUT, F_EYE_R_IN, F_EYE_L_OUT, F_EYE_L_IN)], axis=0)
    # lens centres in front of the eyes; the frame front a little in front of the top of the nose bridge
    o = eyes + ez * (np.dot(_p(face, F_BRIDGE) - eyes, ez) - 0.009 * ppm) + ey * 0.002 * ppm
    pose = Pose3D(o, R, np.full(3, ppm), [head_occluder(face)])
    return Placement(o, ex, ey, (HEAD_W + GLASSES_GAP) * ppm, pose3d=pose)


def lobes(face: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Ear-lobe piercing points: just outside and behind the jaw line at lobe height."""
    R, ppm = head(face)
    ex, _, ez = R.T
    return (_p(face, F_JAW_R) - ex * 0.004 * ppm + ez * 0.014 * ppm,
            _p(face, F_JAW_L) + ex * 0.004 * ppm + ez * 0.014 * ppm)


def earrings(face: np.ndarray) -> List[Placement]:
    """One per ear lobe. Earrings hang straight down (gravity) and turn with the head; a 3D earring on the far
    side disappears behind the head, a photo one fades out as that ear turns away."""
    R, ppm = head(face)
    ex, ey, ez = R.T
    lobe_r, lobe_l = lobes(face)
    occ = [head_occluder(face)]
    out = []
    for lobe, outward, mirror, away in ((lobe_r, -ex, False, -ex[2]), (lobe_l, ex, True, ex[2])):
        # the earring's plane: mostly along the side of the head, turned 40 degrees towards the camera
        x = ez * math.cos(math.radians(40)) + outward * math.sin(math.radians(40))
        x = _unit(np.array([x[0], 0.0, x[2]])) if abs(x[0]) + abs(x[2]) > 1e-6 else np.array([1.0, 0, 0])
        Re = np.stack([x, DOWN, np.cross(x, DOWN)], axis=1)
        side = _ortho(np.array([ex[0], 0.0, ex[2]]), DOWN) if abs(ex[0]) + abs(ex[2]) > 1e-6 else np.array([1.0, 0, 0])
        opacity = float(np.clip((0.5 - away) / 0.25, 0.0, 1.0))
        out.append(Placement(lobe, side, DOWN.copy(), 0.018 * ppm, mirror=mirror, opacity=opacity,
                             pose3d=Pose3D(lobe, Re, np.full(3, ppm), occ), occlude=("head", "cover")))
    return out


def necklace(pose: Optional[np.ndarray], face: Optional[np.ndarray] = None) -> Optional[Placement]:
    """Round the base of the neck. The body's sideways direction comes from the shoulders (if seen), the size
    and the neck's position from the face (more reliable close up), else from the shoulders."""
    if pose is None and face is None:
        return None
    if pose is not None:
        sl, sr = _p(pose, P_SHOULDER_L), _p(pose, P_SHOULDER_R)
        d = sl - sr
        d[2] *= 0.3                                      # pose depth is noisy; damp it
        bx = _unit(d)
    else:
        ex = face_frame(face)[0]
        bx = _unit(np.array([ex[0], ex[1], 0.3 * ex[2]]))
    by = _ortho(DOWN, bx)
    bz = np.cross(bx, by)
    if face is not None:
        _, ppm = head(face)
        neck = (_p(face, F_TRAGUS_R) + _p(face, F_TRAGUS_L)) / 2 + by * 0.155 * ppm
    else:
        span = float(np.linalg.norm((sl - sr)[:2]))
        ppm = span / SHOULDER_W
        neck = (sl + sr) / 2 - by * 0.05 * ppm
    R = np.stack([bx, by, bz], axis=1)
    L = 0.175 * ppm
    occ = [("cylinder", affine(np.stack([bx, bz, by], axis=1), (0.057 * ppm, 0.057 * ppm, L),
                               neck - by * (0.5 * L - 0.015 * ppm)))]
    top = neck - bz * NECK_R * ppm + by * 0.02 * ppm      # photo necklaces: top centre on the collarbones
    return Placement(top, bx, by, 0.15 * ppm, pose3d=Pose3D(neck, R, np.full(3, ppm), occ),
                     occlude=("hair", "face"))


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
    occ = [("cylinder", affine(R, (0.97 * r, 0.97 * r, 1.6 * seg), c))]
    occ += [capsule(_p(hand, a), _p(hand, b), 0.95 * r)
            for a, b in ((H_MIDDLE_MCP, H_MIDDLE_PIP), (H_PINKY_MCP, H_PINKY_PIP))]
    pose = Pose3D(c, R, np.array([r, r, r]), occ)
    ex = R[:, 0] if R[0, 0] >= 0 else -R[:, 0]            # flat photo rings: lie across the finger, upright
    return Placement(c, ex, _ortho(_unit(pip - mcp) * -1, ex), finger_w, pose3d=pose)


def wrist(hand: np.ndarray, label: str = "") -> Placement:
    w, m = _p(hand, H_WRIST), _p(hand, H_MIDDLE_MCP)
    palm_w = float(np.linalg.norm(_p(hand, H_INDEX_MCP) - _p(hand, H_PINKY_MCP)))
    axis = _unit(m - w)
    c = w - axis * 0.18 * palm_w          # a little down the forearm from the wrist joint
    R = _frame(c, axis, dorsal(hand, label))
    rx, ry = 0.42 * palm_w, 0.30 * palm_w  # the wrist is wider than it is thick
    pose = Pose3D(c, R, np.array([rx, ry, rx]), [("cylinder", affine(R, (0.97 * rx, 0.97 * ry, 2.5 * palm_w), c))])
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
    if kind == "necklace":
        p = necklace(pose, face)
        return [p] if p is not None else []
    if kind == "ring":
        return [ring(h, lb) for h, lb in zip(hands, labels)]
    if kind == "wrist":
        return [wrist(h, lb) for h, lb in zip(hands, labels)]
    return []


# ---------------------------------------------------------------------------------------------- smoothing
class OneEuro:
    """One Euro filter (Casiez et al., 2012) on landmark arrays: heavy smoothing when still (no jitter), light
    smoothing when moving fast (no lag). Speeds are per landmark, in pixels per second."""

    def __init__(self, min_cutoff: float = 1.2, beta: float = 0.03, d_cutoff: float = 2.0) -> None:
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self._x: Optional[np.ndarray] = None
        self._dx: Optional[np.ndarray] = None
        self._t = 0.0

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, float)
        if self._x is None or self._x.shape != x.shape or t - self._t > 0.5:
            self._x, self._dx, self._t = x.copy(), np.zeros_like(x), t
            return x
        dt = max(t - self._t, 1e-3)
        dx = (x - self._x) / dt
        self._dx = self._dx + self._alpha(self.d_cutoff, dt) * (dx - self._dx)
        speed = np.linalg.norm(self._dx[..., :2], axis=-1, keepdims=True)
        a = self._alpha(self.min_cutoff + self.beta * speed, dt)
        self._x = self._x + a * (x - self._x)
        self._t = t
        return self._x.copy()

    def reset(self) -> None:
        self._x = None
