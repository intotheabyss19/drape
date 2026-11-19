"""3D items: loading models, and rendering them with physically based shading.

Two kinds of model frame:

- "band" (rings, watches, bracelets): the finger / wrist runs along +Z, +Y is the back of the hand (where a
  ring's stone or a watch's dial sits), and the band's inner radius is 1. Any model is put into this frame
  automatically (or with hints from a sidecar json), so it fits the finger whatever its units.
- "metric" (glasses, earrings, necklaces): real-world metres, +X to the viewer's right, +Y down, +Z away from
  the viewer, with the attach point at the origin (glasses: the front of the frame at lens-centre height;
  earrings: the piercing; necklaces: the centre of the neck where the chain goes round).

Rendering is OpenGL (moderngl, offscreen) with image-based lighting: a studio environment map is pre-blurred
for a few roughness levels, metals and gems reflect it, and the result is tone-mapped and matched to the
camera's brightness and colour. Before the item, depth-only "occluders" (the finger, the neck, the head) are
drawn, so the parts of the item behind them disappear the way they do in real life.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .anchors import Pose3D

MESH_EXT = {".glb", ".gltf", ".obj", ".stl", ".ply"}
METRIC_KINDS = {"glasses", "earrings", "necklace", "helmet"}
GLTF_TO_OURS = np.diag([1.0, -1.0, -1.0])   # glTF: +Y up, +Z towards the viewer -> ours: +Y down, +Z away


# ---------------------------------------------------------------------------------------------- data
@dataclass
class Material:
    base: Tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0)   # linear RGBA
    base_tex: Optional[np.ndarray] = None       # (H, W, 4) uint8 sRGB RGBA
    mr_tex: Optional[np.ndarray] = None         # (H, W, 4) uint8: G = roughness, B = metallic (glTF)
    metallic: float = 0.0
    roughness: float = 0.5
    blend: bool = False                          # see-through (glass lenses)
    shading: str = "pbr"                         # "pbr", "gem" (diamond), "pearl"
    name: str = ""


@dataclass
class Primitive:
    v: np.ndarray                    # (N, 3) float32
    n: np.ndarray                    # (N, 3) float32 unit normals
    f: np.ndarray                    # (M, 3) uint32
    mat: Material
    uv: Optional[np.ndarray] = None  # (N, 2)
    col: Optional[np.ndarray] = None # (N, 3) linear vertex colours
    version: int = 0                 # bump after changing v/n in place (same vertex count): re-uploaded


@dataclass
class Model3D:
    prims: List[Primitive]
    frame: str = "band"              # "band", "metric", or "pixels" (already placed in the frame: gauntlets)
    info: Dict[str, float] = field(default_factory=dict)   # e.g. glasses: front "width" in metres

    def vertices(self) -> np.ndarray:
        return np.concatenate([p.v for p in self.prims]) if self.prims else np.zeros((0, 3), np.float32)

    def transform(self, M: np.ndarray, t=(0.0, 0.0, 0.0)) -> "Model3D":
        """v -> M v + t for every vertex (normals follow), in place."""
        M = np.asarray(M, float)
        Ninv = np.linalg.inv(M).T
        flip = np.linalg.det(M) < 0
        for p in self.prims:
            p.v = (p.v @ M.T + np.asarray(t, float)).astype(np.float32)
            n = p.n @ Ninv.T
            p.n = (n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)).astype(np.float32)
            if flip:
                p.f = p.f[:, ::-1].copy()
        return self


# ---------------------------------------------------------------------------------------------- geometry helpers
def smooth_normals(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    n = np.zeros_like(v, dtype=np.float64)
    for k in range(3):
        np.add.at(n, f[:, k], fn)
    return (n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)).astype(np.float32)


def flat_shaded(v: np.ndarray, f: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unshare vertices so every face gets its own normal (crisp facets for gems)."""
    vv = v[f.reshape(-1)]
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
    return vv.astype(np.float32), np.repeat(fn, 3, axis=0).astype(np.float32), \
        np.arange(len(vv), dtype=np.uint32).reshape(-1, 3)


def prim(v, f, mat: Material, n=None) -> Primitive:
    v = np.asarray(v, np.float32)
    f = np.asarray(f, np.uint32)
    return Primitive(v, smooth_normals(v, f) if n is None else np.asarray(n, np.float32), f, mat)


# ---------------------------------------------------------------------------------------------- loading
def vertex_weights(model: "Model3D") -> np.ndarray:
    """Surface area around each vertex, so shape guesses aren't swayed by how finely each part is meshed (a
    faceted stone has many more vertices per square millimetre than a smooth band)."""
    out = []
    for p in model.prims:
        a = 0.5 * np.linalg.norm(np.cross(p.v[p.f[:, 1]] - p.v[p.f[:, 0]], p.v[p.f[:, 2]] - p.v[p.f[:, 0]]), axis=1)
        w = np.zeros(len(p.v))
        for k in range(3):
            np.add.at(w, p.f[:, k], a / 3)
        out.append(w)
    return np.concatenate(out)


def _wpercentile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    i = np.argsort(x)
    cw = np.cumsum(w[i])
    return float(x[i][np.searchsorted(cw, q / 100 * cw[-1])])


def canonicalize(v: np.ndarray, axis=None, up=None, inner_radius=None,
                 weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Rotation/scale R (rows) and centre c putting a ring/band mesh into the band frame: (v - c) @ R.T.
    Guesses what isn't given: the band axis is the thinnest bounding-box direction; 'up' is the side that sticks
    out more (the stone or the dial); the inner radius is the 3rd percentile of distance from the axis
    (weighted by surface area when `weights` are given)."""
    v = np.asarray(v, dtype=np.float64)
    wts = np.ones(len(v)) if weights is None else np.asarray(weights, float) + 1e-12
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
            return float(np.average((rad > 0.75 * half) & (rad < 1.02 * half), weights=wts))

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
        inner_radius = _wpercentile(radial, wts, 3) or half * 0.85
    R = np.stack([side, up, axis]) / float(inner_radius)
    return R, c


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _img(pil) -> Optional[np.ndarray]:
    if pil is None:
        return None
    try:
        a = np.asarray(pil.convert("RGBA"))
    except Exception:
        return None
    if max(a.shape[:2]) > 1024:                       # keep GPU memory and upload time small
        import cv2
        s = 1024 / max(a.shape[:2])
        a = cv2.resize(a, (int(a.shape[1] * s), int(a.shape[0] * s)), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(a)


def _prim_from_trimesh(m) -> Optional[Primitive]:
    v = np.asarray(m.vertices, np.float32)
    f = np.asarray(m.faces, np.uint32)
    if len(v) == 0 or len(f) == 0:
        return None
    mat = Material()
    uv = col = None
    vis = m.visual
    kind = getattr(vis, "kind", None)
    pbr = getattr(vis, "material", None)
    if pbr is not None:
        mat.name = str(getattr(pbr, "name", "") or "")
        bc = getattr(pbr, "baseColorFactor", None)       # glTF: linear (trimesh stores it as 0..255)
        dif = getattr(pbr, "diffuse", None)              # OBJ SimpleMaterial: sRGB 0..255
        if bc is not None or dif is not None:
            raw = np.asarray(bc if bc is not None else dif)
            c = raw.astype(float).reshape(-1) / (255.0 if raw.dtype.kind in "ui" or raw.max() > 1.0 else 1.0)
            rgb = c[:3] if bc is not None else _srgb_to_linear(c[:3])
            mat.base = (*[float(x) for x in rgb], float(c[3]) if len(c) > 3 else 1.0)
        elif type(pbr).__name__ == "PBRMaterial":
            mat.base = (1.0, 1.0, 1.0, 1.0)                      # glTF default
        mat.base_tex = _img(getattr(pbr, "baseColorTexture", None) or getattr(pbr, "image", None))
        mat.mr_tex = _img(getattr(pbr, "metallicRoughnessTexture", None))
        mf, rf = getattr(pbr, "metallicFactor", None), getattr(pbr, "roughnessFactor", None)
        is_pbr = type(pbr).__name__ == "PBRMaterial"           # glTF defaults: metallic 1, roughness 1
        mat.metallic = float(mf) if mf is not None else (1.0 if is_pbr else 0.0)
        mat.roughness = float(rf) if rf is not None else (1.0 if is_pbr else 0.5)
        mat.blend = str(getattr(pbr, "alphaMode", "") or "").upper() == "BLEND"
    if getattr(vis, "uv", None) is not None and len(vis.uv) == len(v):
        uv = np.asarray(vis.uv, np.float32)
    if kind == "vertex":
        col = _srgb_to_linear(np.asarray(vis.vertex_colors)[:, :3] / 255.0).astype(np.float32)
    elif kind == "face":
        fc = _srgb_to_linear(np.asarray(vis.face_colors)[:, :3] / 255.0)
        col = np.zeros((len(v), 3))
        cnt = np.zeros(len(v))
        for k in range(3):
            np.add.at(col, f[:, k], fc)
            np.add.at(cnt, f[:, k], 1)
        col = (col / np.maximum(cnt, 1)[:, None]).astype(np.float32)
    try:
        n = np.asarray(m.vertex_normals, np.float32)
    except Exception:
        n = smooth_normals(v, f)
    if n.shape != v.shape:
        n = smooth_normals(v, f)
    return Primitive(v, n, f, mat, uv, col)


def _gltf_extensions(path: Path) -> Dict[str, dict]:
    """Material name -> its glTF extensions (trimesh drops them; we need transmission for glass)."""
    import json
    import struct

    try:
        raw = Path(path).read_bytes()
        if raw[:4] == b"glTF":
            n = struct.unpack("<I", raw[12:16])[0]
            doc = json.loads(raw[20:20 + n])
        else:
            doc = json.loads(raw)
        return {m.get("name", ""): m.get("extensions", {}) for m in doc.get("materials", [])}
    except Exception:
        return {}


def _glass(mat: Material, transmission: float) -> None:
    """Approximate a transmissive material (glass, lenses) as see-through: clear glass lets most light through,
    a dark tinted lens very little."""
    tint = float(np.mean(mat.base[:3]))
    alpha = float(np.clip(1.0 - transmission * tint, 0.06, 1.0))
    mat.base = (*mat.base[:3], alpha)
    mat.blend = True
    mat.metallic = 0.0


def load_model(path: Path, kind: str, meta: Optional[dict] = None) -> Model3D:
    """Load a model file and put it into the right frame for `kind` (see the module docstring)."""
    import trimesh

    meta = meta or {}
    obj = trimesh.load(str(path), process=False)
    meshes = obj.dump(concatenate=False) if isinstance(obj, trimesh.Scene) else [obj]
    prims = [p for p in (_prim_from_trimesh(m) for m in meshes) if p is not None]
    if not prims:
        raise ValueError(f"{path}: no triangles in this model")
    if Path(path).suffix.lower() in (".glb", ".gltf"):
        ext = _gltf_extensions(Path(path))
        for p in prims:
            t = ext.get(p.mat.name, {}).get("KHR_materials_transmission", {}).get("transmissionFactor", 0.0)
            if t > 0:
                _glass(p.mat, float(t))
    for p in prims:                                    # sidecar overrides for every material
        if "metallic" in meta:
            p.mat.metallic = float(meta["metallic"])
        if "roughness" in meta:
            p.mat.roughness = float(meta["roughness"])
    model = Model3D(prims)
    if kind not in METRIC_KINDS:
        R, c = canonicalize(model.vertices(), meta.get("axis"), meta.get("up"), meta.get("inner_radius"),
                            vertex_weights(model))
        model.transform(R, -(R @ c))
        model.frame = "band"
        return model
    # metric: glTF/OBJ convention (+Y up, +Z front) to ours, in metres
    model.transform(GLTF_TO_OURS)
    units = meta.get("units")
    ext = float(np.ptp(model.vertices(), axis=0).max())
    scale = {"m": 1.0, "cm": 0.01, "mm": 0.001}.get(units) if units else (0.001 if ext > 2.0 else 1.0)
    model.transform(np.eye(3) * scale)
    v = model.vertices()
    if kind == "glasses":
        front = v[v[:, 2] <= v[:, 2].min() + 0.012]                 # the frame front (nearest the viewer)
        c = np.array([(v[:, 0].min() + v[:, 0].max()) / 2, (front[:, 1].min() + front[:, 1].max()) / 2,
                      v[:, 2].min()])
        model.info["width"] = float(np.ptp(front[:, 0]))
    elif kind == "earrings":
        top = v[v[:, 1] <= v[:, 1].min() + 0.002]
        c = np.array([top[:, 0].mean(), v[:, 1].min(), top[:, 2].mean()])
    elif kind == "helmet":                                         # the middle of its box ~ the head's centre
        c = (v.min(axis=0) + v.max(axis=0)) / 2
        model.info["helmet_width"] = float(np.ptp(v[:, 0]))
    else:
        c = np.zeros(3)
    if "attach" in meta:
        c = np.asarray(meta["attach"], float) * np.array([1, -1, -1]) * scale
    model.transform(np.eye(3), -c)
    model.frame = "metric"
    return model


# ---------------------------------------------------------------------------------------------- environment
def _blur_wrap(img: np.ndarray, sigma_px: float) -> np.ndarray:
    import cv2

    if sigma_px < 0.3:
        return img
    pad = int(3 * sigma_px) + 1
    pad = min(pad, img.shape[1])
    big = np.concatenate([img[:, -pad:], img, img[:, :pad]], axis=1)
    big = cv2.copyMakeBorder(big, pad, pad, 0, 0, cv2.BORDER_REFLECT)
    big = cv2.GaussianBlur(big, (0, 0), sigma_px)
    return big[pad:-pad, pad:-pad]


ROUGH_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)


def studio_env(w: int = 256, h: int = 128) -> np.ndarray:
    """A procedural studio environment (linear RGB, equirectangular, row 0 = straight up): a soft grey room
    with a big softbox above and in front, two strip lights at the sides and a darker floor."""
    lon = (np.arange(w) + 0.5) / w * 2 * np.pi - np.pi       # 0 = towards the viewer's side of the object
    lat = np.pi / 2 - (np.arange(h) + 0.5) / h * np.pi       # +pi/2 up
    LON, LAT = np.meshgrid(lon, lat)
    env = 0.32 + 0.30 * np.clip(np.sin(LAT), 0, 1) - 0.18 * np.clip(-np.sin(LAT), 0, 1)

    def box(lon0, lat0, dlon, dlat, power):
        d = np.exp(-(((LON - lon0 + np.pi) % (2 * np.pi) - np.pi) / dlon) ** 8 - ((LAT - lat0) / dlat) ** 8)
        return power * d

    env = env + box(0.0, 0.55, 0.55, 0.25, 9.0) + box(-1.5, 0.15, 0.12, 0.6, 5.0) + box(1.6, 0.2, 0.12, 0.6, 4.0) \
        + box(np.pi, 0.4, 0.6, 0.3, 2.5)
    return np.repeat(env[:, :, None], 3, axis=2).astype(np.float32)


def prefilter(env: np.ndarray) -> np.ndarray:
    """(levels, H, W, 3) float32: the environment blurred for each roughness level, plus a last, very blurred
    level used as diffuse light."""
    import cv2

    base = cv2.resize(env, (256, 128), interpolation=cv2.INTER_AREA).astype(np.float32)
    px_per_rad = 256 / (2 * np.pi)
    out = [_blur_wrap(base, (0.02 + 0.55 * r * r) * px_per_rad) for r in ROUGH_LEVELS]
    out.append(_blur_wrap(_blur_wrap(base, 0.9 * px_per_rad), 0.6 * px_per_rad))
    return np.stack(out).astype(np.float32)


def load_hdr(path: Path) -> Optional[np.ndarray]:
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    if img is None:
        return None
    return np.ascontiguousarray(img[:, :, ::-1].astype(np.float32))


# ---------------------------------------------------------------------------------------------- shaders
_VS = """
#version 330
uniform mat4 proj;
uniform mat4 model;
uniform mat3 nmat;
in vec3 in_pos;
in vec3 in_nrm;
in vec2 in_uv;
in vec3 in_col;
out vec3 v_n;
out vec2 v_uv;
out vec3 v_c;
void main() {
    gl_Position = proj * model * vec4(in_pos, 1.0);
    v_n = normalize(nmat * in_nrm);
    v_uv = in_uv;
    v_c = in_col;
}
"""
_FS = """
#version 330
uniform sampler2DArray env;     // prefiltered: layers 0..4 = roughness 0..1, layer 5 = diffuse
uniform sampler2D base_tex;
uniform sampler2D mr_tex;
uniform bool has_base;
uniform bool has_mr;
uniform vec4 base_factor;
uniform float metal_factor;
uniform float rough_factor;
uniform int shading;            // 0 pbr, 1 gem, 2 pearl
uniform bool blend;
uniform float exposure;
uniform vec3 light;             // camera-matched gain * tint (linear RGB)
uniform float spec_gain;        // reflections dim further in a dim room (our studio isn't your room)
uniform float env_yaw;
in vec3 v_n;
in vec2 v_uv;
in vec3 v_c;
out vec4 frag;

const float PI = 3.14159265;
vec3 envAt(vec3 d, float layer) {
    // our frame: x right, y down, z away from the viewer
    float lon = atan(d.x, -d.z) + env_yaw;
    float u = lon / (2.0 * PI) + 0.5;
    float v = acos(clamp(-d.y, -1.0, 1.0)) / PI;
    return texture(env, vec3(u, v, layer)).rgb;
}
vec3 envRough(vec3 d, float r) {
    float l = clamp(r, 0.0, 1.0) * 4.0;
    float l0 = floor(l);
    return mix(envAt(d, l0), envAt(d, min(l0 + 1.0, 4.0)), l - l0);
}
vec2 envBRDF(float r, float NoV) {          // Karis' analytic fit of the split-sum BRDF
    const vec4 c0 = vec4(-1.0, -0.0275, -0.572, 0.022);
    const vec4 c1 = vec4(1.0, 0.0425, 1.04, -0.04);
    vec4 q = r * c0 + c1;
    float a004 = min(q.x * q.x, exp2(-9.28 * NoV)) * q.x + q.y;
    return vec2(-1.04, 1.04) * a004 + q.zw;
}
vec3 aces(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}
void main() {
    vec3 N = normalize(v_n);
    vec3 I = vec3(0.0, 0.0, 1.0);           // view ray (from the viewer into the scene)
    vec3 V = -I;
    if (dot(N, V) < 0.0) N = -N;
    float NoV = max(dot(N, V), 1e-3);
    vec4 base = base_factor * vec4(v_c, 1.0);
    if (has_base) {
        vec4 t = texture(base_tex, v_uv);
        base *= vec4(pow(t.rgb, vec3(2.2)), t.a);
    }
    float metal = metal_factor, rough = rough_factor;
    if (has_mr) {
        vec4 t = texture(mr_tex, v_uv);
        rough *= t.g;
        metal *= t.b;
    }
    rough = clamp(rough, 0.02, 1.0);
    vec3 R = reflect(I, N);
    vec3 diff, spec;
    if (shading == 1) {                     // gem: strong reflections plus dispersed light through it
        // A well-cut diamond sends light entering the top back out towards the viewer (total internal
        // reflection off the pavilion), so each facet shows a different, magnified bit of the room behind the
        // viewer, split slightly by colour (fire).
        float F = 0.17 + 0.83 * pow(1.0 - NoV, 5.0);
        vec3 tr;
        for (int k = 0; k < 3; k++) {
            float eta = 1.0 / (2.38 + 0.04 * float(k));
            vec3 T = refract(I, N, eta);
            vec3 back = normalize(vec3(T.x * 4.0, T.y * 4.0 - 0.3, -abs(T.z)));
            tr[k] = envAt(back, 0.0)[k];
        }
        spec = F * envAt(R, 0.0);
        diff = (1.0 - F) * pow(tr, vec3(1.6)) * 1.9 * base.rgb;
    } else {
        vec3 F0 = mix(vec3(0.04), base.rgb, metal);
        vec2 ab = envBRDF(rough, NoV);
        spec = envRough(R, rough) * (F0 * ab.x + ab.y);
        diff = envAt(N, 5.0) * base.rgb * (1.0 - metal);
        if (shading == 2) {                 // pearl: soft lustre and a faint pink-green orient at the rim
            float rim = pow(1.0 - NoV, 2.0);
            diff += envAt(N, 5.0) * rim * vec3(0.10, 0.06, 0.09);
            spec += envRough(R, 0.12) * 0.06 * (1.0 - rim);
        }
    }
    float a = blend ? base.a : 1.0;
    spec *= spec_gain;
    vec3 lit = (diff * a + spec) * exposure * light;
    vec3 c = pow(aces(lit), vec3(1.0 / 2.2));
    float lum = dot(pow(aces(spec * exposure * light), vec3(1.0 / 2.2)), vec3(0.3, 0.6, 0.1));
    float A = blend ? clamp(a + lum, 0.0, 1.0) : 1.0;
    frag = vec4(c, A);                      // premultiplied
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


def _unit_cylinder(sections: int = 40):
    """Radius 1, along Z from -0.5 to 0.5, closed."""
    a = np.linspace(0, 2 * np.pi, sections, endpoint=False)
    ring = np.stack([np.cos(a), np.sin(a)], axis=1)
    v = np.concatenate([np.c_[ring, np.full(sections, -0.5)], np.c_[ring, np.full(sections, 0.5)],
                        [[0, 0, -0.5], [0, 0, 0.5]]])
    f = []
    for i in range(sections):
        j = (i + 1) % sections
        f += [[i, j, sections + j], [i, sections + j, sections + i], [2 * sections, j, i],
              [2 * sections + 1, sections + i, sections + j]]
    return v.astype(np.float32), np.array(f, np.uint32)


def uv_sphere(rings: int = 16, segs: int = 24):
    th = np.linspace(0, np.pi, rings + 1)
    ph = np.linspace(0, 2 * np.pi, segs, endpoint=False)
    T, P = np.meshgrid(th, ph, indexing="ij")
    v = np.stack([np.sin(T) * np.cos(P), np.cos(T), np.sin(T) * np.sin(P)], axis=-1).reshape(-1, 3)
    f = []
    for i in range(rings):
        for j in range(segs):
            a, b = i * segs + j, i * segs + (j + 1) % segs
            c, d = a + segs, b + segs
            f += [[a, c, b], [b, c, d]]
    return v.astype(np.float32), np.array(f, np.uint32)


# ---------------------------------------------------------------------------------------------- renderer
class MeshRenderer:
    """Offscreen OpenGL renderer for posed 3D items, with depth-only occluders."""

    def __init__(self, env: Optional[np.ndarray] = None) -> None:
        import moderngl

        self.mgl = moderngl
        self.ctx = moderngl.create_standalone_context()
        self.prog = self.ctx.program(vertex_shader=_VS, fragment_shader=_FS)
        self.prog_d = self.ctx.program(vertex_shader=_VS_DEPTH, fragment_shader=_FS_DEPTH)
        self.size: Tuple[int, int] = (0, 0)
        self._vaos: Dict[int, list] = {}
        self._tex: Dict[int, object] = {}
        self._occ = {"cylinder": self._depth_vao(*_unit_cylinder()), "sphere": self._depth_vao(*uv_sphere(20, 32))}
        self.samples = min(4, self.ctx.max_samples)
        self.exposure = 1.0
        self.env_yaw = 0.0
        self.set_env(env if env is not None else studio_env())
        white = np.full((1, 1, 4), 255, np.uint8)
        self._white = self.ctx.texture((1, 1), 4, white.tobytes())

    def set_env(self, env: np.ndarray, peak: float = 20.0) -> None:
        # Cap the studio lights at `peak` x the average: a real studio's softbox is ~200x, and mirrored in a lens
        # it becomes a white sticker in any room. Capped, metals still glint and lenses get a soft sheen.
        env = np.minimum(env, peak * float(env.mean()))
        pre = prefilter(env)
        L, H, W, _ = pre.shape
        tex = self.ctx.texture_array((W, H, L), 3, pre.astype("f4").tobytes(), dtype="f4")
        tex.filter = (self.mgl.LINEAR, self.mgl.LINEAR)
        tex.repeat_x, tex.repeat_y = True, False
        self.env = tex
        # normalise brightness: a mid-grey diffuse surface lit by this environment should come out mid-grey
        self.exposure = float(0.55 / max(pre[-1].mean(), 1e-4))

    def _resize(self, w: int, h: int) -> None:
        if (w, h) == self.size:
            return
        ctx = self.ctx
        self.fbo_ms = ctx.framebuffer(color_attachments=[ctx.renderbuffer((w, h), 4, samples=self.samples)],
                                      depth_attachment=ctx.depth_renderbuffer((w, h), samples=self.samples))
        self.fbo = ctx.framebuffer(color_attachments=[ctx.renderbuffer((w, h), 4)])
        self.size = (w, h)

    def _depth_vao(self, v: np.ndarray, f: np.ndarray):
        vbo = self.ctx.buffer(np.asarray(v, "f4").tobytes())
        ibo = self.ctx.buffer(np.asarray(f, "u4").tobytes())
        return self.ctx.vertex_array(self.prog_d, [(vbo, "3f", "in_pos")], index_buffer=ibo, index_element_size=4)

    def _texture(self, img: Optional[np.ndarray]):
        if img is None:
            return self._white
        key = id(img)
        if key not in self._tex:
            t = self.ctx.texture((img.shape[1], img.shape[0]), 4, np.ascontiguousarray(img).tobytes())
            t.build_mipmaps()
            t.filter = (self.mgl.LINEAR_MIPMAP_LINEAR, self.mgl.LINEAR)
            t.repeat_x = t.repeat_y = True
            self._tex[key] = (img, t)
        return self._tex[key][1]

    @staticmethod
    def _vertex_data(p: Primitive) -> bytes:
        n = len(p.v)
        uv = p.uv if p.uv is not None else np.zeros((n, 2), np.float32)
        col = p.col if p.col is not None else np.ones((n, 3), np.float32)
        return np.hstack([p.v, p.n, uv, col]).astype("f4").tobytes()

    def _prim_vao(self, p: Primitive):
        key = id(p)
        if key not in self._vaos:
            vbo = self.ctx.buffer(self._vertex_data(p))
            ibo = self.ctx.buffer(np.asarray(p.f, "u4").tobytes())
            vao = self.ctx.vertex_array(self.prog, [(vbo, "3f 3f 2f 3f", "in_pos", "in_nrm", "in_uv", "in_col")],
                                        index_buffer=ibo, index_element_size=4)
            self._vaos[key] = [p, vao, vbo, p.version]  # keep p alive so id() stays unique
        entry = self._vaos[key]
        if entry[3] != p.version:                       # moved since last upload (gauntlets, every frame)
            entry[2].write(self._vertex_data(p))
            entry[3] = p.version
        return entry[1]

    @staticmethod
    def _proj(w: int, h: int) -> np.ndarray:
        d = float(max(w, h)) * 4.0
        return np.array([[2 / w, 0, 0, -1], [0, -2 / h, 0, 1], [0, 0, 1 / d, 0], [0, 0, 0, 1]], np.float32)

    @staticmethod
    def model_matrix(R: np.ndarray, scale, origin) -> np.ndarray:
        m = np.eye(4, dtype=np.float32)
        m[:3, :3] = np.asarray(R, float) * np.asarray(scale, float)[None, :]
        m[:3, 3] = origin
        return m

    def render(self, frame_shape, model: Model3D, pose: Pose3D, scale: float = 1.0,
               gain: float = 1.0, tint=(1.0, 1.0, 1.0)):
        """Returns (premultiplied BGRA patch, x0, y0) for the item's on-screen box, or None if off-screen."""
        fh, fw = frame_shape[:2]
        self._resize(fw, fh)
        M = self.model_matrix(pose.R, np.asarray(pose.scale, float) * scale, pose.origin)
        pts = model.vertices() @ M[:3, :3].T + M[:3, 3]
        x0, y0 = np.floor(pts[:, :2].min(axis=0)).astype(int) - 2
        x1, y1 = np.ceil(pts[:, :2].max(axis=0)).astype(int) + 2
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, fw), min(y1, fh)
        if x1 - x0 < 2 or y1 - y0 < 2:
            return None
        proj = self._proj(fw, fh)
        ctx, mgl = self.ctx, self.mgl
        self.fbo_ms.use()
        ctx.viewport = (0, 0, fw, fh)
        ctx.enable(mgl.DEPTH_TEST)
        ctx.disable(mgl.CULL_FACE)
        ctx.disable(mgl.BLEND)
        ctx.scissor = (x0, fh - y1, x1 - x0, y1 - y0)
        self.fbo_ms.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)

        # occluders: depth only
        self.fbo_ms.color_mask = (False, False, False, False)
        self.prog_d["proj"].write(proj.T.tobytes())
        for shape, OM in pose.occluders:
            self.prog_d["model"].write(np.asarray(OM, "f4").T.tobytes())
            self._occ[shape].render()
        self.fbo_ms.color_mask = (True, True, True, True)

        # the item: opaque parts, then see-through parts over them
        lin = M[:3, :3].astype(float)
        nmat = np.linalg.inv(lin).T
        nmat /= np.linalg.norm(nmat, axis=0, keepdims=True) + 1e-12
        p = self.prog
        p["proj"].write(proj.T.tobytes())
        p["model"].write(M.T.astype("f4").tobytes())
        p["nmat"].write(nmat.T.astype("f4").tobytes())
        p["exposure"].value = self.exposure
        p["env_yaw"].value = self.env_yaw
        tint = np.asarray(tint, float)[::-1]                     # BGR -> RGB
        p["light"].value = tuple(float(gain) * (0.7 + 0.3 * tint))
        p["spec_gain"].value = float(min(gain, 1.0))
        self.env.use(0)
        p["env"].value = 0
        p["base_tex"].value = 1
        p["mr_tex"].value = 2
        for blend in (False, True):
            if blend:
                ctx.enable(mgl.BLEND)
                ctx.blend_func = mgl.ONE, mgl.ONE_MINUS_SRC_ALPHA
                ctx.depth_mask = False
            for prim_ in model.prims:
                m = prim_.mat
                if m.blend != blend:
                    continue
                self._texture(m.base_tex).use(1)
                self._texture(m.mr_tex).use(2)
                p["has_base"].value = m.base_tex is not None
                p["has_mr"].value = m.mr_tex is not None
                p["base_factor"].value = tuple(float(x) for x in m.base)
                p["metal_factor"].value = float(m.metallic)
                p["rough_factor"].value = float(m.roughness)
                p["shading"].value = {"pbr": 0, "gem": 1, "pearl": 2}.get(m.shading, 0)
                p["blend"].value = bool(blend)
                self._prim_vao(prim_).render()
        ctx.depth_mask = True
        ctx.disable(mgl.BLEND)
        ctx.scissor = None

        ctx.copy_framebuffer(self.fbo, self.fbo_ms)
        raw = self.fbo.read(viewport=(x0, fh - y1, x1 - x0, y1 - y0), components=4)
        rgba = np.frombuffer(raw, np.uint8).reshape(y1 - y0, x1 - x0, 4)[::-1]
        return rgba[:, :, [2, 1, 0, 3]].copy(), x0, y0

    def close(self) -> None:
        self.ctx.release()
