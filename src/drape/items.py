"""Wearable items: load a photo, cut the item out of its background, and know where it attaches.

A photo with transparency (PNG with alpha) is used as is. Otherwise the background is removed:
- with `rembg` if it's installed (`pip install "drape[rembg]"`), which handles busy backgrounds;
- else by flood-filling a plain background in from the image border (fine for product shots on white/grey).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .anchors import KINDS

# Where on the cut-out item the body anchor goes, as fractions of its width/height.
DEFAULT_ANCHOR = {
    "necklace": (0.5, 0.0),   # top centre of the chain sits at the base of the neck
    "earrings": (0.5, 0.0),   # the hook sits on the lobe
    "glasses": (0.5, 0.38),   # the bridge sits between the eyes
    "ring": (0.5, 0.5),
    "wrist": (0.5, 0.5),
}

_KIND_WORDS = [
    ("earrings", r"ear-?rings?|studs?|jhumkas?|hoops?"),
    ("necklace", r"necklaces?|chains?|pendants?|chokers?|lockets?|mangalsutra|haar"),
    ("glasses", r"glasses|sunglasses|spectacles|specs|shades|eyewear|frames?"),
    ("ring", r"rings?"),
    ("wrist", r"watch(?:es)?|bracelets?|bangles?|kadas?|wristbands?|cuffs?"),
]
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".glb", ".gltf", ".obj", ".stl", ".ply"}


def guess_kind(name: str) -> Optional[str]:
    n = name.lower()
    for kind, pat in _KIND_WORDS:
        if re.search(rf"(^|[^a-z]){pat}([^a-z]|$)", n):
            return kind
    return None


@dataclass
class Item:
    name: str
    kind: str
    rgba: Optional[np.ndarray]             # cut-out, cropped, BGRA uint8 (None for a 3D item)
    anchor: Tuple[float, float] = (0.5, 0.5)
    scale: float = 1.0                     # user tweak on top of the automatic size
    path: Optional[Path] = field(default=None, repr=False)
    mesh: Optional[object] = field(default=None, repr=False)   # mesh3d.MeshData for 3D items

    @property
    def is_3d(self) -> bool:
        return self.mesh is not None

    @property
    def anchor_px(self) -> Tuple[float, float]:
        h, w = self.rgba.shape[:2]
        return self.anchor[0] * w, self.anchor[1] * h


def _rembg(bgr: np.ndarray) -> Optional[np.ndarray]:
    try:
        from rembg import remove  # type: ignore
    except ImportError:
        return None
    out = remove(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    out = np.asarray(out)
    return cv2.cvtColor(out, cv2.COLOR_RGBA2BGRA) if out.shape[2] == 4 else None


def remove_plain_background(bgr: np.ndarray, tol: int = 18) -> np.ndarray:
    """Alpha mask by flood-filling from every border pixel whose colour is close to the border's median."""
    h, w = bgr.shape[:2]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    border = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(lab.astype(np.float32) - bg, axis=2)
    near = (dist < tol).astype(np.uint8)
    # keep only the background region connected to the border (so the item's own light parts survive)
    n, labels = cv2.connectedComponents(near, connectivity=4)
    edge_labels = set(np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))) - {0}
    bg_mask = np.isin(labels, list(edge_labels)) & (near > 0)
    alpha = np.where(bg_mask, 0, 255).astype(np.uint8)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)          # soften the cut edge
    return np.dstack([bgr, alpha])


def crop_to_content(rgba: np.ndarray, pad: int = 2) -> np.ndarray:
    ys, xs = np.nonzero(rgba[:, :, 3] > 8)
    if len(xs) == 0:
        raise ValueError("the item is empty after removing the background")
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, rgba.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, rgba.shape[1])
    return rgba[y0:y1, x0:x1].copy()


def cutout(img: np.ndarray) -> np.ndarray:
    """BGR or BGRA image -> cropped BGRA cut-out of the item."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4 and (img[:, :, 3] < 250).mean() > 0.02:
        rgba = img
    else:
        bgr = img[:, :, :3]
        rgba = _rembg(bgr)
        if rgba is None:
            rgba = remove_plain_background(bgr)
    return crop_to_content(rgba)


def load(path: str | Path, kind: Optional[str] = None) -> Item:
    """Load an item photo or 3D model. Kind comes from the argument, a sidecar `<name>.json`, or the file name."""
    from .mesh3d import MESH_EXT, load_mesh

    path = Path(path).expanduser()
    meta = {}
    side = path.with_suffix(".json")
    if side.exists():
        meta = json.loads(side.read_text())
    kind = kind or meta.get("kind") or guess_kind(path.stem)
    if kind not in KINDS:
        raise ValueError(f"don't know what kind of item '{path.name}' is; pass a kind ({', '.join(KINDS)}) "
                         "or put it in the file name, e.g. gold_necklace.png")
    if path.suffix.lower() in MESH_EXT:
        if kind not in ("ring", "wrist"):
            raise ValueError(f"3D models are supported for rings and watches/bracelets, not {kind}")
        return Item(name=meta.get("name", path.stem), kind=kind, rgba=None, scale=float(meta.get("scale", 1.0)),
                    path=path, mesh=load_mesh(path, meta))
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"can't read image {path}")
    anchor = tuple(meta.get("anchor", DEFAULT_ANCHOR[kind]))
    return Item(name=meta.get("name", path.stem), kind=kind, rgba=cutout(img), anchor=anchor,
                scale=float(meta.get("scale", 1.0)), path=path)


def load_many(specs: List[str]) -> List[Item]:
    """Each spec is a file, a folder of images, or 'file:kind'."""
    out: List[Item] = []
    for spec in specs:
        kind = None
        m = re.match(r"^(.*):(" + "|".join(KINDS) + r")$", spec)
        if m:
            spec, kind = m.group(1), m.group(2)
        p = Path(spec).expanduser()
        if p.is_dir():
            for f in sorted(p.iterdir()):
                if f.suffix.lower() in IMAGE_EXT:
                    try:
                        out.append(load(f, kind))
                    except ValueError as e:
                        print(f"skipping {f.name}: {e}")
        else:
            out.append(load(p, kind))
    return out
