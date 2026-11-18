"""3D items (rings, watches, bracelets as .glb / .gltf / .obj / .stl / .ply meshes).

The mesh is put into a canonical frame (the finger or wrist runs along +Z, +Y is the back of the hand where a
ring's stone or a watch's dial sits, the band's inner radius is 1), then posed on the hand every frame and
rendered with OpenGL (moderngl, offscreen). Before the item, a depth-only cylinder for the finger / wrist and
capsules for the neighbouring fingers are drawn, so the back of the band disappears behind the finger and a
finger in front hides the ring, the way it does in real life.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .anchors import Pose3D

MESH_EXT = {".glb", ".gltf", ".obj", ".stl", ".ply"}


@dataclass
class MeshData:
    vertices: np.ndarray       # (N, 3) float32, canonical frame
    normals: np.ndarray        # (N, 3)
    colors: np.ndarray         # (N, 3) float32 0..1 (RGB)
    faces: np.ndarray          # (M, 3) uint32
    metallic: float = 0.8


# ---------------------------------------------------------------------------------------------- loading
def canonicalize(v: np.ndarray, axis=None, up=None, inner_radius=None) -> Tuple[np.ndarray, np.ndarray]:
    """Rotation/scale R (rows) and centre c putting a ring/band mesh into the canonical frame: (v - c) @ R.T.
    Guesses what isn't given: the band axis is the thinnest bounding-box direction; 'up' is the side that sticks
    out more (the stone or the dial); the inner radius is the 3rd percentile of distance from the axis."""
    v = np.asarray(v, dtype=np.float64)
    lo, hi = v.min(axis=0), v.max(axis=0)
    if axis is None:
        axis = np.eye(3)[int(np.argmin(hi - lo))]
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    c = (lo + hi) / 2
    if up is None:
        # The band is a circle; the stone/dial sticks out on one side. 'up' is the longer of the two directions
        # across the axis, and its sign is whichever end gives a centre that puts the band's points on a circle.
        e1, e2 = [e for e in np.eye(3) if abs(np.dot(e, axis)) < 0.9][:2]
        p1, p2 = (v - c) @ e1, (v - c) @ e2
        e, pe, half = (e1, p1, np.ptp(p2) / 2) if np.ptp(p1) >= np.ptp(p2) else (e2, p2, np.ptp(p1) / 2)
        other = e2 if e is e1 else e1
        po = (v - c) @ other
        po = po - (po.max() + po.min()) / 2

        def circle_fit(center_e: float) -> float:
            rad = np.hypot(pe - center_e, po)
            return float(np.mean((rad > 0.75 * half) & (rad < 1.02 * half)))

        up = e if circle_fit(pe.min() + half) >= circle_fit(pe.max() - half) else -e
    up = np.asarray(up, float)
    up = up - np.dot(up, axis) * axis
    up /= np.linalg.norm(up)
    side = np.cross(up, axis)
    # the band is symmetric side to side, and its bottom is opposite the stone/dial: its centre is half the
    # side-to-side width above its lowest point
    ps, pu, pa = (v - c) @ side, (v - c) @ up, (v - c) @ axis
    half = (ps.max() - ps.min()) / 2
    c = c + side * (ps.max() + ps.min()) / 2 + up * (pu.min() + half) + axis * (pa.max() + pa.min()) / 2
    if inner_radius is None:
        d = v - c
        radial = np.linalg.norm(d - np.outer(d @ axis, axis), axis=1)
        inner_radius = float(np.percentile(radial, 3)) or half * 0.85
    R = np.stack([side, up, axis]) / float(inner_radius)
    return R, c


def from_trimesh(mesh, axis=None, up=None, inner_radius=None, metallic: float = 0.8) -> MeshData:
    R, c = canonicalize(mesh.vertices, axis, up, inner_radius)
    v = (np.asarray(mesh.vertices) - c) @ R.T
    rot = R / np.linalg.norm(R[0])
    n = np.asarray(mesh.vertex_normals) @ rot.T
    col = None
    vis = mesh.visual
    try:
        if getattr(vis, "kind", None) == "vertex":
            col = np.asarray(vis.vertex_colors)[:, :3] / 255.0
        elif getattr(vis, "kind", None) == "face":
            fc = np.asarray(vis.face_colors)[:, :3] / 255.0           # spread face colours to their vertices
            col = np.zeros((len(v), 3))
            cnt = np.zeros(len(v))
            for k in range(3):
                np.add.at(col, mesh.faces[:, k], fc)
                np.add.at(cnt, mesh.faces[:, k], 1)
            col /= np.maximum(cnt, 1)[:, None]
        else:
            col = np.asarray(vis.to_color().vertex_colors)[:, :3] / 255.0
    except Exception:
        col = None
    if col is None or len(col) != len(v):
        col = np.full((len(v), 3), 0.8)
    if len(col) != len(v):
        col = np.full((len(v), 3), 0.8)
    return MeshData(v.astype(np.float32), n.astype(np.float32), col.astype(np.float32),
                    np.asarray(mesh.faces, dtype=np.uint32), metallic)


def load_mesh(path: Path, meta: Optional[dict] = None) -> MeshData:
    import trimesh

    meta = meta or {}
    mesh = trimesh.load(str(path), force="mesh", process=True)
    metallic = meta.get("metallic")
    if metallic is None:
        mat = getattr(getattr(mesh.visual, "material", None), "metallicFactor", None)
        metallic = 0.8 if mat is None else float(mat)
    return from_trimesh(mesh, meta.get("axis"), meta.get("up"), meta.get("inner_radius"), float(metallic))


# ---------------------------------------------------------------------------------------------- rendering
_VS = """
#version 330
uniform mat4 proj;
uniform mat4 model;
uniform mat3 nmat;
in vec3 in_pos;
in vec3 in_nrm;
in vec3 in_col;
out vec3 v_n;
out vec3 v_c;
void main() {
    gl_Position = proj * model * vec4(in_pos, 1.0);
    v_n = normalize(nmat * in_nrm);
    v_c = in_col;
}
"""
_FS = """
#version 330
uniform vec3 to_light;      // unit, pixel space (x right, y down, z away from camera)
uniform float gain;
uniform vec3 tint;
uniform float metallic;
in vec3 v_n;
in vec3 v_c;
out vec4 frag;
void main() {
    vec3 n = normalize(v_n);
    vec3 V = vec3(0.0, 0.0, -1.0);                  // towards the camera
    if (dot(n, V) < 0.0) n = -n;
    float diff = max(dot(n, to_light), 0.0);
    vec3 H = normalize(to_light + V);
    float spec = pow(max(dot(n, H), 0.0), 64.0);
    float env = 0.5 + 0.5 * clamp(-n.y, -1.0, 1.0);  // brighter where the surface faces up (sky / ceiling)
    vec3 base = v_c;
    vec3 specc = mix(vec3(1.0), base, metallic);
    vec3 col = base * (0.28 + (1.0 - 0.5 * metallic) * 0.75 * diff)
             + specc * (1.3 * spec + metallic * 0.45 * env);
    frag = vec4(col * gain * (0.75 + 0.25 * tint), 1.0);
}
"""
_VS_DEPTH = """
#version 330
uniform mat4 proj;
uniform mat4 model;
in vec3 in_pos;
void main() { gl_Position = proj * model * vec4(in_pos, 1.0); }
"""
_FS_DEPTH = """
#version 330
out vec4 frag;
void main() { frag = vec4(0.0); }
"""


def _unit_cylinder(sections: int = 32):
    import trimesh

    return trimesh.creation.cylinder(radius=1.0, height=1.0, sections=sections)   # along Z, centred


class MeshRenderer:
    """Offscreen OpenGL renderer for posed 3D items, with finger/wrist occluders."""

    def __init__(self) -> None:
        import moderngl

        self.mgl = moderngl
        self.ctx = moderngl.create_standalone_context()
        self.prog = self.ctx.program(vertex_shader=_VS, fragment_shader=_FS)
        self.prog_d = self.ctx.program(vertex_shader=_VS_DEPTH, fragment_shader=_FS_DEPTH)
        self.size: Tuple[int, int] = (0, 0)
        self._vaos = {}
        cyl = _unit_cylinder()
        self._cyl = self._depth_vao(np.asarray(cyl.vertices, np.float32), np.asarray(cyl.faces, np.uint32))
        self.samples = min(4, self.ctx.max_samples)

    def _resize(self, w: int, h: int) -> None:
        if (w, h) == self.size:
            return
        ctx = self.ctx
        self.fbo_ms = ctx.framebuffer(color_attachments=[ctx.renderbuffer((w, h), 4, samples=self.samples)],
                                      depth_attachment=ctx.depth_renderbuffer((w, h), samples=self.samples))
        self.fbo = ctx.framebuffer(color_attachments=[ctx.renderbuffer((w, h), 4)])
        self.size = (w, h)

    def _depth_vao(self, v: np.ndarray, f: np.ndarray):
        vbo = self.ctx.buffer(v.astype("f4").tobytes())
        ibo = self.ctx.buffer(f.astype("u4").tobytes())
        return self.ctx.vertex_array(self.prog_d, [(vbo, "3f", "in_pos")], index_buffer=ibo, index_element_size=4)

    def _item_vao(self, mesh: MeshData):
        key = id(mesh)
        if key not in self._vaos:
            data = np.hstack([mesh.vertices, mesh.normals, mesh.colors]).astype("f4")
            vbo = self.ctx.buffer(data.tobytes())
            ibo = self.ctx.buffer(mesh.faces.astype("u4").tobytes())
            self._vaos[key] = (mesh, self.ctx.vertex_array(
                self.prog, [(vbo, "3f 3f 3f", "in_pos", "in_nrm", "in_col")], index_buffer=ibo,
                index_element_size=4))
        return self._vaos[key][1]

    @staticmethod
    def _proj(w: int, h: int) -> np.ndarray:
        d = float(max(w, h)) * 2.0
        return np.array([[2 / w, 0, 0, -1], [0, -2 / h, 0, 1], [0, 0, 1 / d, 0], [0, 0, 0, 1]], np.float32)

    @staticmethod
    def _model(R: np.ndarray, scale, origin) -> np.ndarray:
        m = np.eye(4, dtype=np.float32)
        m[:3, :3] = R * np.asarray(scale, float)[None, :]
        m[:3, 3] = origin
        return m

    @staticmethod
    def _capsule_model(p0: np.ndarray, p1: np.ndarray, r: float) -> np.ndarray:
        z = p1 - p0
        L = float(np.linalg.norm(z)) or 1.0
        z = z / L
        a = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
        x = np.cross(a, z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        return MeshRenderer._model(np.stack([x, y, z], axis=1), (r, r, L * 1.15), (p0 + p1) / 2)

    def render(self, frame_shape, mesh: MeshData, pose: Pose3D, scale: float = 1.0,
               gain: float = 1.0, tint=(1.0, 1.0, 1.0)):
        """Returns (bgra_patch, x0, y0) for the item's on-screen box, or None if it's off-screen."""
        fh, fw = frame_shape[:2]
        self._resize(fw, fh)
        model = self._model(pose.R, pose.scale * scale, pose.origin)
        pts = mesh.vertices @ model[:3, :3].T + model[:3, 3]
        x0, y0 = np.floor(pts[:, :2].min(axis=0)).astype(int) - 2
        x1, y1 = np.ceil(pts[:, :2].max(axis=0)).astype(int) + 2
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, fw), min(y1, fh)
        if x1 - x0 < 2 or y1 - y0 < 2:
            return None
        proj = self._proj(fw, fh)
        ctx = self.ctx
        self.fbo_ms.use()
        ctx.viewport = (0, 0, fw, fh)
        ctx.enable(self.mgl.DEPTH_TEST)
        ctx.disable(self.mgl.CULL_FACE)
        self.fbo_ms.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)

        # occluders: depth only
        self.fbo_ms.color_mask = (False, False, False, False)
        self.prog_d["proj"].write(proj.T.tobytes())
        occ = self._model(pose.R, (pose.occluder_scale[0], pose.occluder_scale[1], pose.occluder_length),
                          pose.origin)
        self.prog_d["model"].write(occ.T.astype("f4").tobytes())
        self._cyl.render()
        for c in pose.others:
            self.prog_d["model"].write(self._capsule_model(c.p0, c.p1, c.radius).T.astype("f4").tobytes())
            self._cyl.render()
        self.fbo_ms.color_mask = (True, True, True, True)

        # the item
        lin = model[:3, :3]
        nmat = np.linalg.inv(lin).T
        nmat /= np.linalg.norm(nmat, axis=0, keepdims=True) + 1e-9
        p = self.prog
        p["proj"].write(proj.T.tobytes())
        p["model"].write(model.T.astype("f4").tobytes())
        p["nmat"].write(nmat.T.astype("f4").tobytes())
        to_light = np.array([-0.35, -0.75, -0.55])
        p["to_light"].value = tuple(to_light / np.linalg.norm(to_light))
        p["gain"].value = float(gain)
        p["tint"].value = tuple(float(t) for t in np.asarray(tint)[::-1])   # BGR -> RGB
        p["metallic"].value = float(mesh.metallic)
        self._item_vao(mesh).render()

        ctx.copy_framebuffer(self.fbo, self.fbo_ms)
        gl_y = fh - y1
        raw = self.fbo.read(viewport=(x0, gl_y, x1 - x0, y1 - y0), components=4)
        rgba = np.frombuffer(raw, np.uint8).reshape(y1 - y0, x1 - x0, 4)[::-1]
        bgra = rgba[:, :, [2, 1, 0, 3]].copy()
        return bgra, x0, y0

    def close(self) -> None:
        self.ctx.release()


# ---------------------------------------------------------------------------------------------- samples
def sample_ring():
    """A gold band with a round stone, in the canonical frame."""
    import trimesh

    band = trimesh.creation.annulus(r_min=1.0, r_max=1.16, height=0.32, sections=96)
    band.visual.vertex_colors = [228, 178, 70, 255]
    seat = trimesh.creation.cylinder(radius=0.2, height=0.22, sections=24)
    seat.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    seat.apply_translation([0, 1.2, 0])
    seat.visual.vertex_colors = [228, 178, 70, 255]
    stone = trimesh.creation.icosphere(subdivisions=1, radius=0.24)
    stone.apply_scale([1.0, 0.8, 1.0])
    stone.apply_translation([0, 1.38, 0])
    stone.visual.vertex_colors = [225, 240, 255, 255]
    return trimesh.util.concatenate([band, seat, stone])


def sample_watch():
    """A leather strap with a steel case and dark dial, in the canonical frame."""
    import trimesh

    strap = trimesh.creation.annulus(r_min=1.0, r_max=1.07, height=0.5, sections=96)
    strap.visual.vertex_colors = [70, 45, 30, 255]
    parts = [strap]

    def disc(radius, height, y, color):
        d = trimesh.creation.cylinder(radius=radius, height=height, sections=48)
        d.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
        d.apply_translation([0, y, 0])
        d.visual.vertex_colors = color
        return d

    parts += [disc(0.46, 0.22, 1.14, [205, 205, 215, 255]), disc(0.39, 0.02, 1.26, [25, 28, 34, 255])]
    for ang, length in ((0.0, 0.3), (1.9, 0.22)):
        hand = trimesh.creation.box([0.03, 0.01, length])
        hand.apply_translation([0, 0, length / 2])
        hand.apply_transform(trimesh.transformations.rotation_matrix(ang, [0, 1, 0]))
        hand.apply_translation([0, 1.28, 0])
        hand.visual.vertex_colors = [235, 235, 235, 255]
        parts.append(hand)
    return trimesh.util.concatenate(parts)


def write_samples(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for name, fn, kind, metal in (("gold_ring_3d", sample_ring, "ring", 0.85),
                                  ("leather_watch_3d", sample_watch, "wrist", 0.5)):
        fn().export(str(folder / f"{name}.glb"))
        (folder / f"{name}.json").write_text(json.dumps(
            {"kind": kind, "axis": [0, 0, 1], "up": [0, 1, 0], "inner_radius": 1.0, "metallic": metal}))
