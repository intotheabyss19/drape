"""Jewellery built from real-world dimensions: a diamond solitaire ring, pearl drop and gold hoop earrings, a
pearl strand and a gold chain with a pendant.

There are no good openly licensed jewellery models to ship, and jewellery is mostly simple shapes (bands,
spheres, links, a cut stone): what makes it look real is the material and the light, which the renderer does
physically (metal reflections, a dispersive diamond, pearl lustre). Dimensions are typical retail sizes.

Rings are in the "band" frame (finger along +Z, stone towards +Y, inner radius 1); everything else is in
metres in the "metric" frame (+X right, +Y down, +Z away from the viewer), see mesh3d.
"""
from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np

from .mesh3d import Material, Model3D, Primitive, flat_shaded, prim, uv_sphere

# metals and gems (linear base colours: measured reflectances of the metals)
GOLD = Material(base=(1.0, 0.71, 0.29, 1.0), metallic=1.0, roughness=0.16, name="gold")
ROSE_GOLD = Material(base=(0.98, 0.64, 0.52, 1.0), metallic=1.0, roughness=0.18, name="rose gold")
SILVER = Material(base=(0.95, 0.93, 0.88, 1.0), metallic=1.0, roughness=0.14, name="silver")
DIAMOND = Material(base=(1.0, 1.0, 1.0, 1.0), roughness=0.0, shading="gem", name="diamond")
PEARL = Material(base=(0.86, 0.82, 0.76, 1.0), roughness=0.28, shading="pearl", name="pearl")


# ---------------------------------------------------------------------------------------------- shapes
def _sphere(c, r, mat, rings=14, segs=20) -> Primitive:
    v, f = uv_sphere(rings, segs)
    return prim(v * r + np.asarray(c, float), f, mat, n=v)


def _frames(path: np.ndarray, closed: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tangent and two normals along a polyline (parallel transport, so tubes don't twist)."""
    t = np.gradient(path, axis=0)
    if closed:
        t = np.roll(path, -1, axis=0) - np.roll(path, 1, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-12
    n = np.zeros_like(path)
    ref = np.array([0.0, 0.0, 1.0]) if abs(t[0, 2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    n[0] = np.cross(t[0], ref)
    n[0] /= np.linalg.norm(n[0])
    for i in range(1, len(path)):
        m = n[i - 1] - np.dot(n[i - 1], t[i]) * t[i]
        n[i] = m / (np.linalg.norm(m) + 1e-12)
    return t, n, np.cross(t, n)


def tube(path, radius, mat, sides=12, closed=False) -> Primitive:
    path = np.asarray(path, float)
    _, n, b = _frames(path, closed)
    a = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    rad = np.asarray(radius, float) * np.ones(len(path))
    dirs = np.cos(a)[None, :, None] * n[:, None, :] + np.sin(a)[None, :, None] * b[:, None, :]
    v = (path[:, None, :] + rad[:, None, None] * dirs).reshape(-1, 3)
    f = []
    rows = len(path) if closed else len(path) - 1
    for i in range(rows):
        i2 = (i + 1) % len(path)
        for j in range(sides):
            j2 = (j + 1) % sides
            p, q, r, s = i * sides + j, i * sides + j2, i2 * sides + j, i2 * sides + j2
            f += [[p, r, q], [q, r, s]]
    if not closed:                                            # cap the ends
        for end, sign in ((0, -1), (len(path) - 1, 1)):
            c = len(v)
            v = np.vstack([v, path[end]])
            for j in range(sides):
                j2 = (j + 1) % sides
                f.append([c, end * sides + j, end * sides + j2] if sign < 0 else [c, end * sides + j2, end * sides + j])
    return prim(v, f, mat, n=None if not closed else dirs.reshape(-1, 3))


def torus(R, r, mat, center=(0, 0, 0), axes=np.eye(3), seg=32, sides=12, stretch=1.0) -> Primitive:
    """Ring in the plane of axes[0], axes[1]; stretch elongates it along axes[0] (chain links)."""
    a = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    path = np.stack([R * stretch * np.cos(a), R * np.sin(a), np.zeros(seg)], axis=1)
    p = tube(path, r, mat, sides, closed=True)
    A = np.asarray(axes, float).T
    p.v = (p.v @ A.T + np.asarray(center, float)).astype(np.float32)
    p.n = (p.n @ A.T).astype(np.float32)
    return p


def brilliant(diameter: float, mat: Material = DIAMOND) -> Primitive:
    """A round brilliant cut, table up (-Y in the metric frame is up, so the table is at the most negative y).
    Ideal proportions: table 57%, crown 15%, girdle 3%, pavilion 43% of the diameter."""
    from scipy.spatial import ConvexHull

    D = diameter
    r = D / 2
    pts = []
    for k in range(8):                                        # table
        a = 2 * np.pi * k / 8
        pts.append([0.57 * r * np.cos(a), -0.15 * D - 0.015 * D, 0.57 * r * np.sin(a)])
    for k in range(8):                                        # crown star facets' outer points
        a = 2 * np.pi * (k + 0.5) / 8
        pts.append([0.80 * r * np.cos(a), -0.075 * D - 0.015 * D, 0.80 * r * np.sin(a)])
    for k in range(16):                                       # girdle, top and bottom
        a = 2 * np.pi * k / 16
        pts.append([r * np.cos(a), -0.015 * D, r * np.sin(a)])
        pts.append([r * np.cos(a), 0.015 * D, r * np.sin(a)])
    for k in range(8):                                        # pavilion mains
        a = 2 * np.pi * (k + 0.5) / 8
        pts.append([0.55 * r * np.cos(a), 0.015 * D + 0.2 * D, 0.55 * r * np.sin(a)])
    pts.append([0, 0.015 * D + 0.43 * D, 0])                  # culet
    pts = np.array(pts)
    hull = ConvexHull(pts)
    f = hull.simplices.copy()
    c = pts.mean(axis=0)
    for i, (a, b, d) in enumerate(f):                         # orient outward
        if np.dot(np.cross(pts[b] - pts[a], pts[d] - pts[a]), pts[a] - c) < 0:
            f[i] = [a, d, b]
    v, n, ff = flat_shaded(pts, f)
    return Primitive(v, n, ff, mat)


def _move(p: Primitive, R=np.eye(3), t=(0, 0, 0)) -> Primitive:
    R = np.asarray(R, float)
    p.v = (p.v @ R.T + np.asarray(t, float)).astype(np.float32)
    p.n = (p.n @ R.T).astype(np.float32)
    return p


def _rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


# ---------------------------------------------------------------------------------------------- rings (band frame)
def solitaire_ring(metal: Material = GOLD) -> Model3D:
    """A 2.2 mm comfort-fit band with a 0.5 ct (5.1 mm) round brilliant in a four-prong setting, for a finger
    about 18 mm across (1 unit = 9 mm)."""
    u = 1 / 9.0                                                # mm -> band units
    a = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    ring_path = np.stack([np.cos(a), np.sin(a), np.zeros_like(a)], axis=1)
    # band: a rounded profile swept round the finger (inner radius 1, 1.6 mm thick, 2.2 mm wide)
    th, wd = 1.6 * u, 2.2 * u
    prof_a = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    prof = np.stack([0.5 * th * np.abs(np.cos(prof_a)) ** 0.6 * np.sign(np.cos(prof_a)),
                     0.5 * wd * np.abs(np.sin(prof_a)) ** 0.6 * np.sign(np.sin(prof_a))], axis=1)
    rad_dir = ring_path
    zdir = np.array([0, 0, 1.0])
    v = (ring_path[:, None, :] * (1 + 0.5 * th) + prof[None, :, 0:1] * rad_dir[:, None, :]
         + prof[None, :, 1:2] * zdir).reshape(-1, 3)
    f = []
    S = len(prof)
    for i in range(len(a)):
        i2 = (i + 1) % len(a)
        for j in range(S):
            j2 = (j + 1) % S
            p, q, r, s = i * S + j, i * S + j2, i2 * S + j, i2 * S + j2
            f += [[p, q, r], [q, s, r]]
    band = prim(v, f, metal)

    D = 5.1 * u
    top = 1 + th                                               # outside of the band at the top
    girdle_y = top + 0.9 * D * 0.45 + 0.6 * u                  # the stone sits up in its head
    stone = brilliant(D)
    # brilliant() has its table at -y (metric "up"); in the band frame "up" (the back of the hand) is +y
    stone = _move(stone, np.diag([1, -1, 1]), (0, girdle_y, 0))
    stone.f = stone.f[:, ::-1].copy()
    parts: List[Primitive] = [band, stone]
    # four prongs from the band up round the girdle, plus a small basket ring under the stone
    for k in range(4):
        ang = np.pi / 4 + k * np.pi / 2
        dx, dz = np.cos(ang), np.sin(ang)
        p0 = np.array([0.18 * D * dx, top - 0.2 * u, 0.18 * D * dz])
        p1 = np.array([0.53 * D * dx, girdle_y + 0.05 * D, 0.53 * D * dz])
        p2 = np.array([0.42 * D * dx, girdle_y - 0.12 * D, 0.42 * D * dz])
        path = np.vstack([p0 + (p1 - p0) * t for t in np.linspace(0, 1, 8)] + [p2])
        parts.append(tube(path, 0.45 * u, metal, sides=10))
        parts.append(_sphere(p2, 0.5 * u, metal, 8, 12))
    parts.append(torus(0.36 * D, 0.35 * u, metal, center=(0, girdle_y - 0.2 * D, 0),
                       axes=np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]]), seg=32, sides=8))
    return Model3D(parts, "band")


# ---------------------------------------------------------------------------------------------- earrings (metric)
def pearl_drop_earring(metal: Material = GOLD) -> Model3D:
    """A 4 mm gold ball stud at the lobe, a 9 mm wire and an 8 mm pearl with a small cap."""
    mm = 0.001
    parts = [_sphere((0, 0, 0), 2.0 * mm, metal),
             tube(np.array([[0, 1.5, 0], [0, 6, 0], [0, 10.5, 0]]) * mm, 0.45 * mm, metal, sides=8)]
    pearl_c = np.array([0, 15.0, 0]) * mm
    parts.append(_sphere(pearl_c, 4.0 * mm, PEARL, 20, 28))
    cap = _sphere(pearl_c, 4.25 * mm, metal, 8, 20)              # a small dome over the pearl's top
    top = (cap.v[:, 1] - pearl_c[1]) / (4.25 * mm) < -0.6
    idx = np.where(top)[0]
    remap = -np.ones(len(cap.v), int)
    remap[idx] = np.arange(len(idx))
    faces = cap.f[np.all(top[cap.f], axis=1)]
    parts.append(Primitive(cap.v[idx], cap.n[idx], remap[faces].astype(np.uint32), metal))
    return Model3D(parts, "metric")


def hoop_earring(diameter_mm: float = 22.0, metal: Material = GOLD) -> Model3D:
    """A classic 2 mm thick hoop, hanging from the piercing at the origin, in the X-Y plane."""
    mm = 0.001
    R = diameter_mm / 2 * mm
    a = np.linspace(-np.pi / 2, 3 * np.pi / 2, 96, endpoint=False)
    path = np.stack([R * np.cos(a), R + R * np.sin(a), np.zeros_like(a)], axis=1)
    return Model3D([tube(path, 1.0 * mm, metal, sides=14, closed=True)], "metric")


# ---------------------------------------------------------------------------------------------- necklaces (metric)
NECK_R = 0.062        # where a chain rests round the base of the neck (radius, m)


def necklace_path(length: float, n: int = 600, neck_r: float = NECK_R) -> np.ndarray:
    """A closed loop of the given length round the base of the neck: behind the neck it hugs a circle (slightly
    higher at the back), in front it falls over the collarbones into a U on the chest, which leans forward."""
    half_w = neck_r * 1.18

    def build(D):
        th = np.linspace(0, np.pi, n // 2)                      # behind the neck, right side round to left
        back = np.stack([neck_r * np.cos(th), -0.018 * np.sin(th), neck_r * np.sin(th)], axis=1)
        ph = np.linspace(np.pi, 2 * np.pi, n // 2 + 1)[1:-1]    # in front, left side down and back up
        s = -np.sin(ph)                                         # 0 at the sides, 1 at the bottom
        x = (neck_r + (half_w - neck_r) * s ** 0.5) * np.cos(ph) * (1 - 0.15 * s)
        y = D * s ** 1.15
        z = -(neck_r + 0.012) * s ** 0.5 - 0.22 * y
        return np.vstack([back, np.stack([x, y, z], axis=1)])

    def L(D):
        p = build(D)
        return float(np.linalg.norm(np.diff(np.vstack([p, p[:1]]), axis=0), axis=1).sum())

    lo, hi = 0.0, 0.5
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if L(mid) < length else (lo, mid)
    return build((lo + hi) / 2)


def _resample(path: np.ndarray, step: float) -> Tuple[np.ndarray, np.ndarray]:
    """Points every `step` metres along a closed path, and the unit tangent there."""
    loop = np.vstack([path, path[:1]])
    seg = np.linalg.norm(np.diff(loop, axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    k = int(s[-1] // step)
    ts = np.arange(k) * (s[-1] / k)
    pts = np.stack([np.interp(ts, s, loop[:, i]) for i in range(3)], axis=1)
    tan = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
    return pts, tan / (np.linalg.norm(tan, axis=1, keepdims=True) + 1e-12)


def pearl_strand(length: float = 0.45, pearl_mm: float = 7.0) -> Model3D:
    """A princess-length (45 cm) strand of 7 mm pearls."""
    path = necklace_path(length)
    pts, _ = _resample(path, (pearl_mm + 0.4) / 1000)
    v1, f1 = uv_sphere(12, 18)
    r = pearl_mm / 2000
    v = np.concatenate([v1 * r + p for p in pts])
    n = np.concatenate([v1] * len(pts))
    f = np.concatenate([f1 + i * len(v1) for i in range(len(pts))])
    return Model3D([Primitive(v.astype(np.float32), n.astype(np.float32), f.astype(np.uint32), PEARL)], "metric")


def chain_necklace(length: float = 0.46, link_mm: float = 4.2, wire_mm: float = 0.9,
                   metal: Material = GOLD, pendant: bool = True) -> Model3D:
    """A cable chain (alternating links at right angles) with an optional 6 mm diamond pendant."""
    path = necklace_path(length)
    pts, tan = _resample(path, link_mm * 0.72 / 1000)
    vs, ns, fs, off = [], [], [], 0
    R = (link_mm / 2 - wire_mm / 2) / 1000 / 1.35
    for i, (p, t) in enumerate(zip(pts, tan)):
        out = np.array([0.0, 0.0, -1.0]) - np.dot([0.0, 0.0, -1.0], t) * t   # towards the viewer
        out /= np.linalg.norm(out) + 1e-12
        side = np.cross(t, out)
        axes = np.stack([t, out if i % 2 else side, np.cross(t, out if i % 2 else side)])
        link = torus(R, wire_mm / 2000, metal, center=p, axes=axes, seg=16, sides=6, stretch=1.35)
        vs.append(link.v)
        ns.append(link.n)
        fs.append(link.f + off)
        off += len(link.v)
    parts = [Primitive(np.concatenate(vs), np.concatenate(ns), np.concatenate(fs).astype(np.uint32), metal)]
    if pendant:
        bottom = path[np.argmax(path[:, 1])]
        lean = _rot_x(np.arctan(0.22) + np.pi / 2)              # face the viewer, tilted with the chest
        bail_c = bottom + np.array([0, 2.2, -0.4]) / 1000
        parts.append(torus(1.6 / 1000, 0.5 / 1000, metal, center=bail_c,
                           axes=np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]), seg=20, sides=8))
        stone_c = bottom + np.array([0, 7.6, -1.4]) / 1000
        stone = _move(brilliant(6.0 / 1000), lean, stone_c)
        parts.append(stone)
        for k in range(4):                                       # a four-claw cup holding it
            ang = np.pi / 4 + k * np.pi / 2
            claw = np.array([np.cos(ang), np.sin(ang), 0.0]) * 3.05 / 1000
            parts.append(_sphere(stone_c + claw @ np.eye(3) + np.array([0, 0, -0.6]) / 1000, 0.55 / 1000, metal, 6, 10))
    return Model3D(parts, "metric")


# ---------------------------------------------------------------------------------------------- catalogue
SAMPLES: List[Tuple[str, str, Callable[[], Model3D]]] = [
    ("diamond solitaire ring", "ring", solitaire_ring),
    ("pearl drop earrings", "earrings", pearl_drop_earring),
    ("gold hoop earrings", "earrings", hoop_earring),
    ("pearl necklace", "necklace", pearl_strand),
    ("gold chain with diamond", "necklace", chain_necklace),
]
