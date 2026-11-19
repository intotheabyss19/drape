"""Armour built by drape: a sci-fi helmet and a pair of gauntlets (original designs).

The helmet is an ordinary metric model (origin at the centre of the head, +X right, +Y down, +Z away from the
viewer, metres): a brushed-steel shell, a dark chevron visor in a polished frame, a gunmetal jaw and ear pods.

The gauntlets follow your fingers. Armour is made of rigid plates, so no skinned model is needed: every frame
each plate (one per finger bone, the back of the hand, knuckle guards, the cuff) is placed between the hand
landmarks it belongs to. Each plate is a superellipsoid (a rounded box or pill), so the whole glove is one
fixed mesh whose vertices move.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .mesh3d import Material, Model3D, Primitive, prim
from .jewelry import tube, torus

STEEL = Material(base=(0.58, 0.58, 0.60, 1.0), metallic=1.0, roughness=0.30, name="brushed steel")
POLISHED = Material(base=(0.66, 0.66, 0.68, 1.0), metallic=1.0, roughness=0.14, name="polished steel")
GUNMETAL = Material(base=(0.20, 0.21, 0.23, 1.0), metallic=1.0, roughness=0.38, name="gunmetal")
LEATHER = Material(base=(0.035, 0.033, 0.032, 1.0), metallic=0.0, roughness=0.70, name="leather")
PAINT = Material(base=(0.20, 0.215, 0.235, 1.0), metallic=0.0, roughness=0.28, name="armour paint")
VISOR = Material(base=(0.012, 0.016, 0.020, 1.0), metallic=0.0, roughness=0.03, name="visor")


def _spow(x, p):
    return np.sign(x) * np.abs(x) ** p


# ---------------------------------------------------------------------------------------------- helmet
def _shell_radius(d: np.ndarray, p: float = 2.6) -> np.ndarray:
    """Distance from the head centre to the shell along unit directions d: a superellipsoid a little bigger
    than a head (20 cm wide, 30 cm tall to the chin, 24 cm front to back)."""
    ax = np.full(len(d), 0.098)
    ay = np.where(d[:, 1] < 0, 0.145, 0.150)
    az = np.where(d[:, 2] < 0, 0.122, 0.118)
    s = (np.abs(d[:, 0]) / ax) ** p + (np.abs(d[:, 1]) / ay) ** p + (np.abs(d[:, 2]) / az) ** p
    return s ** (-1.0 / p)


def _shell_point(d: np.ndarray) -> np.ndarray:
    P = _shell_radius(d)[:, None] * d
    low = np.clip(P[:, 1] / 0.14, 0, 1) ** 1.3                   # the jaw is narrower than the skull
    P[:, 0] *= 1 - 0.22 * low
    return P


def _submesh(v: np.ndarray, f: np.ndarray, keep: np.ndarray, mat: Material) -> Primitive:
    faces = f[keep]
    idx = np.unique(faces)
    remap = -np.ones(len(v), int)
    remap[idx] = np.arange(len(idx))
    return prim(v[idx], remap[faces], mat)


def _dir(phi, th) -> np.ndarray:
    """Unit directions from the head centre: phi from the crown (0) down, th round the head (0 = ahead)."""
    phi, th = np.broadcast_arrays(np.asarray(phi, float), np.asarray(th, float))
    return np.stack([np.sin(phi) * np.sin(th), -np.cos(phi), -np.sin(phi) * np.cos(th)], axis=-1)


def _at_height(th: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The shell point straight round at angle th whose height is y (searched down each meridian)."""
    phis = np.linspace(0.05, 0.80 * np.pi, 400)
    out = []
    for t, yy in zip(np.atleast_1d(th), np.atleast_1d(y)):
        P = _shell_point(_dir(phis, np.full_like(phis, t)))
        out.append(_shell_point(_dir([np.interp(yy, P[:, 1], phis)], [t]))[0])
    return np.array(out)


# the visor: a chevron, deepest in the middle, narrowing towards the temples
VISOR_TH = 1.12


def _visor_top(th):
    u = np.abs(th) / VISOR_TH
    return -0.046 + 0.020 * u ** 1.5


def _visor_bottom(th):
    u = np.abs(th) / VISOR_TH
    return 0.036 - 0.040 * u


def helmet() -> Model3D:
    n_phi, n_th = 110, 140
    phi = np.linspace(0, 0.82 * np.pi, n_phi)                    # from the crown down to the neck opening
    th = np.linspace(-np.pi, np.pi, n_th, endpoint=False)
    PH, TH = np.meshgrid(phi, th, indexing="ij")
    v = _shell_point(_dir(PH, TH).reshape(-1, 3))
    f = []
    for i in range(n_phi - 1):
        for j in range(n_th):
            j2 = (j + 1) % n_th
            a, b, c, e = i * n_th + j, i * n_th + j2, (i + 1) * n_th + j, (i + 1) * n_th + j2
            f += [[a, c, b], [b, c, e]]
    f = np.array(f)
    cen = v[f].mean(axis=1)
    cth = np.arctan2(cen[:, 0], -cen[:, 2])
    y = cen[:, 1]
    visor = (np.abs(cth) < VISOR_TH) & (y > _visor_top(cth)) & (y < _visor_bottom(cth)) & (cen[:, 2] < 0)
    jaw = (y > 0.050) & (np.abs(cth) < 1.75) & ~visor
    parts = [_submesh(v, f, ~visor & ~jaw, PAINT), _submesh(v, f, jaw, GUNMETAL)]

    # the visor as its own smooth surface, recessed 3 mm, and a polished frame round it hiding the cut
    tv = np.linspace(-VISOR_TH, VISOR_TH, 90)
    rows = []
    for sfrac in np.linspace(-0.08, 1.08, 16):
        yy = _visor_top(tv) + sfrac * (_visor_bottom(tv) - _visor_top(tv))
        rows.append(_at_height(tv, yy) * 0.975)
    V = np.concatenate(rows)
    nr, nc = len(rows), len(tv)
    fv = [[i * nc + j, (i + 1) * nc + j, i * nc + j + 1] for i in range(nr - 1) for j in range(nc - 1)]
    fv += [[i * nc + j + 1, (i + 1) * nc + j, (i + 1) * nc + j + 1] for i in range(nr - 1) for j in range(nc - 1)]
    parts.append(prim(V, fv, VISOR))
    loop = np.concatenate([_at_height(tv, _visor_top(tv)), _at_height(tv[::-1], _visor_bottom(tv[::-1]))])
    parts.append(tube(loop * 1.004, 0.0038, POLISHED, sides=10, closed=True))
    # the line where the painted shell meets the gunmetal jaw
    tj = np.linspace(-1.75, 1.75, 120)
    seam = _at_height(tj, np.full_like(tj, 0.050))
    keep = ~((np.abs(tj) < VISOR_TH) & (0.050 < _visor_bottom(tj)))
    if keep.any():
        for run in np.split(np.arange(len(tj))[keep], np.where(np.diff(np.arange(len(tj))[keep]) > 1)[0] + 1):
            if len(run) > 2:
                parts.append(tube(seam[run] * 1.003, 0.0022, POLISHED, sides=8))
    # a rolled rim round the neck opening
    parts.append(tube(v[(n_phi - 1) * n_th:], 0.006, GUNMETAL, sides=10, closed=True))
    # ear pods, set into the shell
    for side in (-1.0, 1.0):
        dn = np.array([side, 0.03, 0.06])
        dn /= np.linalg.norm(dn)
        P = _shell_point(dn[None])[0]
        n = np.array([side, 0.0, 0.0])
        parts.append(tube(np.array([P - n * 0.008, P + n * 0.006]), 0.028, GUNMETAL, sides=40))
        parts.append(torus(0.028, 0.003, POLISHED, center=P + n * 0.006,
                           axes=np.array([[0, 1.0, 0], [0, 0, 1.0], [1.0, 0, 0]]), seg=40, sides=10))
    return Model3D(parts, "metric", {"helmet_width": 0.196})


# ---------------------------------------------------------------------------------------------- gauntlets
def _pill_template(n_phi: int = 12, n_th: int = 18):
    """Unit superellipsoid sample directions (cx, cy, cz) on a grid; cz runs along the plate's axis."""
    phi = np.linspace(0, np.pi, n_phi)
    th = np.linspace(-np.pi, np.pi, n_th, endpoint=False)
    PH, TH = np.meshgrid(phi, th, indexing="ij")
    f = []
    for i in range(n_phi - 1):
        for j in range(n_th):
            j2 = (j + 1) % n_th
            a, b, c, e = i * n_th + j, i * n_th + j2, (i + 1) * n_th + j, (i + 1) * n_th + j2
            f += [[a, b, c], [b, e, c]]
    return PH, TH, np.array(f)


# plates: (name, landmark from, landmark to, radius x palm width, flatness, taper, exponent, ends exponent,
#          top material, underside material)
_FINGERS = [(5, 6, 7, 8, 0.150), (9, 10, 11, 12, 0.155), (13, 14, 15, 16, 0.145), (17, 18, 19, 20, 0.125)]
_THUMB = (1, 2, 3, 4, 0.185)


class Gauntlet:
    """Armour for one or two hands. `posed(hand, label, slot)` returns this frame's model in pixel space."""

    def __init__(self) -> None:
        self.PH, self.TH, self.f = _pill_template()
        self.mats = [POLISHED, PAINT, GUNMETAL, LEATHER]
        self.plates: List[Tuple[int, int, float, float]] = []    # (top mat, under mat, exponent, ends exponent)
        for _ in range(5 * 3):                                    # finger and thumb bones
            self.plates.append((1, 3, 0.55, 0.55))
        self.plates.append((1, 3, 0.45, 0.55))                    # back of the hand / palm
        for _ in range(4):
            self.plates.append((0, 0, 1.0, 1.0))                  # knuckle guards
        self.plates.append((1, 1, 0.8, 0.25))                     # cuff
        self.plates.append((0, 0, 0.8, 0.30))                     # band round the cuff
        self.nv = self.PH.size
        cy = (np.sin(self.PH) * np.sin(self.TH)).reshape(-1)
        under = cy[self.f].mean(axis=1) < -0.05
        groups: Dict[int, List[np.ndarray]] = {}
        for k, (top, und, _, _) in enumerate(self.plates):
            off = k * self.nv
            for m, sel in ((top, ~under), (und, under)):
                groups.setdefault(m, []).append(self.f[sel] + off)
        self.groups = {m: np.concatenate(fs) for m, fs in groups.items()}
        self.idx = {m: np.unique(fs) for m, fs in self.groups.items()}
        self.local_f = {}
        for m, fs in self.groups.items():
            remap = np.zeros(len(self.plates) * self.nv, int)
            remap[self.idx[m]] = np.arange(len(self.idx[m]))
            self.local_f[m] = remap[fs].astype(np.uint32)
        self._slots: Dict[int, List[Primitive]] = {}
        self.info: Dict[str, float] = {}

    # the template as a static model (for listings and tests): a flat open hand
    @property
    def prims(self) -> List[Primitive]:
        return self.posed(_REST_HAND, "Left", -1).prims

    def vertices(self) -> np.ndarray:
        return self.posed(_REST_HAND, "Left", -1).vertices()

    def _plate_params(self, hand: np.ndarray, label: str, scale: float):
        from .anchors import dorsal

        h = np.asarray(hand, float)
        # the hand's size: knuckle width, or palm length when the hand is side-on (its depth is a guess then)
        palm = max(float(np.linalg.norm(h[5] - h[17])), 0.8 * float(np.linalg.norm(h[9] - h[0]))) * scale
        up = dorsal(h, label)
        along = h[9] - h[0]
        along /= np.linalg.norm(along) + 1e-9
        C, Z, Yh, A, B, L, T = [], [], [], [], [], [], []

        def add(c, z, yhint, a, b, half, taper):
            C.append(c), Z.append(z), Yh.append(yhint), A.append(a), B.append(b), L.append(half), T.append(taper)

        for chain in _FINGERS + [_THUMB]:
            *ids, rf = chain
            for k in range(3):
                p0, p1 = h[ids[k]], h[ids[k + 1]]
                d = p1 - p0
                n = float(np.linalg.norm(d)) + 1e-9
                z = d / n
                r = rf * palm * (1.0, 0.93, 0.86)[k]
                start = p0 - z * (0.35 if k else 0.2) * r
                end = p1 + z * (0.25 if k < 2 else 0.15) * r
                add((start + end) / 2, z, up, r, 0.85 * r, np.linalg.norm(end - start) / 2, -0.12 if k == 2 else -0.06)
        start = h[0] - along * 0.05 * palm
        end = (h[5] + h[17]) / 2 + along * 0.08 * palm
        add((start + end) / 2, along, up, 0.55 * palm, 0.20 * palm, np.linalg.norm(end - start) / 2, 0.12)
        for m, p in ((5, 6), (9, 10), (13, 14), (17, 18)):
            fd = h[p] - h[m]
            fd /= np.linalg.norm(fd) + 1e-9
            z = along + fd
            z /= np.linalg.norm(z) + 1e-9
            yv = up - z * np.dot(up, z)
            yv /= np.linalg.norm(yv) + 1e-9
            add(h[m] + yv * 0.19 * palm - along * 0.02 * palm, z, up, 0.13 * palm, 0.06 * palm, 0.11 * palm, 0.0)
        add(h[0] - along * 0.46 * palm, -along, up, 0.46 * palm, 0.33 * palm, 0.42 * palm, 0.10)
        add(h[0] - along * 0.08 * palm, -along, up, 0.50 * palm, 0.37 * palm, 0.06 * palm, 0.0)
        return [np.array(x, float) for x in (C, Z, Yh, A, B, L, T)]

    def _vertices(self, hand, label, scale):
        C, Z, Yh, A, B, L, T = self._plate_params(hand, label, scale)
        Y = Yh - Z * np.sum(Yh * Z, axis=1, keepdims=True)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True) + 1e-9
        X = np.cross(Y, Z)
        e = np.array([p[2] for p in self.plates])[:, None, None]
        ez = np.array([p[3] for p in self.plates])[:, None, None]
        sp, cp = np.sin(self.PH)[None], np.cos(self.PH)[None]
        cx = _spow(sp, e) * _spow(np.cos(self.TH)[None], e)
        cy = _spow(sp, e) * _spow(np.sin(self.TH)[None], e)
        cz = _spow(cp, ez)
        taper = 1 + T[:, None, None] * cz
        P = (C[:, None, None, :] + X[:, None, None, :] * (A[:, None, None] * taper * cx)[..., None]
             + Y[:, None, None, :] * (B[:, None, None] * taper * cy)[..., None]
             + Z[:, None, None, :] * (L[:, None, None] * cz)[..., None])           # (plates, n_phi, n_th, 3)
        d_phi = np.gradient(P, axis=1)
        d_th = np.roll(P, -1, axis=2) - np.roll(P, 1, axis=2)
        N = np.cross(d_th, d_phi)
        N[:, 0] = Z[:, None, :]                                    # the poles: straight along the axis
        N[:, -1] = -Z[:, None, :]
        N /= np.linalg.norm(N, axis=-1, keepdims=True) + 1e-12
        return P.reshape(-1, 3).astype(np.float32), N.reshape(-1, 3).astype(np.float32)

    def posed(self, hand: np.ndarray, label: str = "", slot: int = 0, scale: float = 1.0) -> Model3D:
        V, N = self._vertices(hand, label, scale)
        if slot not in self._slots:
            self._slots[slot] = [Primitive(V[self.idx[m]], N[self.idx[m]], self.local_f[m], self.mats[m])
                                 for m in sorted(self.groups)]
        for p, m in zip(self._slots[slot], sorted(self.groups)):
            p.v, p.n = V[self.idx[m]], N[self.idx[m]]
            p.version += 1
        return Model3D(self._slots[slot], "pixels")


# a flat left hand, back towards the viewer, palm width ~100 px (for tests and listings)
_REST_HAND = np.array([
    [0, 0, 0], [-40, -25, 0], [-62, -55, 0], [-80, -80, 0], [-95, -100, 0],
    [-45, -100, 0], [-50, -145, 0], [-53, -172, 0], [-55, -195, 0],
    [-15, -105, 0], [-15, -155, 0], [-15, -185, 0], [-15, -210, 0],
    [15, -100, 0], [17, -147, 0], [18, -175, 0], [19, -197, 0],
    [42, -90, 0], [48, -125, 0], [51, -147, 0], [53, -165, 0]], float) + np.array([300, 400, 0])


SAMPLES = [("armour helmet", "helmet", helmet), ("armour gauntlets", "gloves", Gauntlet)]
