"""drape's built-in items: jewellery built from real dimensions (see jewelry.py), plus openly licensed 3D models
and a studio lighting environment, downloaded once into ~/.cache/drape/assets.

Downloaded models (all free to use and share, with credit where the licence asks for it):
- Sunglasses and Chronograph Watch: Khronos glTF Sample Assets, by Eric Chadwick / Darmstadt Graphics Group,
  CC-BY 4.0 (they carry the Khronos and 3D Commerce logos, which are trademarks of the Khronos Group).
- Round Spectacles: Poly Haven, CC0.
- Lighting: Poly Haven's "Studio Small 09" HDRI, CC0.
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import List, Optional

import numpy as np

KHRONOS = "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/"
PH = "https://dl.polyhaven.org/file/ph-assets/"
_UA = {"User-Agent": "drape (open-source try-on app)"}


def _ph_model(slug: str) -> dict:
    t = f"{PH}Models/jpg/1k/{slug}/{slug}_"
    return {f"{slug}.gltf": f"{PH}Models/gltf/1k/{slug}/{slug}_1k.gltf",
            f"{slug}.bin": f"{PH}Models/gltf/8k/{slug}/{slug}.bin",
            **{f"textures/{slug}_{m}_1k.jpg": f"{t}{m}_1k.jpg" for m in ("diff", "arm", "nor_gl")}}


DOWNLOADS = {
    "sunglasses": {
        "kind": "glasses", "main": "SunglassesKhronos.glb",
        "files": {"SunglassesKhronos.glb": KHRONOS + "SunglassesKhronos/glTF-Binary/SunglassesKhronos.glb"},
        "credit": "Sunglasses by Eric Chadwick / Darmstadt Graphics Group (Khronos glTF Sample Assets, CC-BY 4.0)",
    },
    "round spectacles": {
        "kind": "glasses", "main": "round_spectacles.gltf", "files": _ph_model("round_spectacles"),
        "credit": "Round Spectacles (Poly Haven, CC0)",
    },
    "chronograph watch": {
        "kind": "wrist", "main": "ChronographWatch.glb",
        "files": {"ChronographWatch.glb": KHRONOS + "ChronographWatch/glTF-Binary/ChronographWatch.glb"},
        "meta": {"axis": [1, 0, 0], "up": [0, 0, 1]},
        "credit": "Chronograph Watch by Eric Chadwick / Darmstadt Graphics Group, from graphiccompressor's "
                  "model (Khronos glTF Sample Assets, CC-BY 4.0)",
    },
}
HDRI = ("studio_small_09_1k.hdr", PH + "HDRIs/hdr/1k/studio_small_09_1k.hdr")


def asset_dir() -> Path:
    return Path(os.environ.get("DRAPE_ASSETS") or Path.home() / ".cache" / "drape" / "assets")


def _get(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 16):
            f.write(chunk)
    tmp.replace(dest)


def fetch(name: str) -> Path:
    """Download a model (once) and return the path of its main file."""
    spec = DOWNLOADS[name]
    folder = asset_dir() / name.replace(" ", "_")
    missing = [(rel, url) for rel, url in spec["files"].items() if not (folder / rel).exists()]
    if missing:
        print(f"downloading {name} (one time)…")
        for rel, url in missing:
            _get(url, folder / rel)
    return folder / spec["main"]


def environment() -> Optional[np.ndarray]:
    """The studio lighting environment (linear RGB equirectangular), or None if it can't be had."""
    from .mesh3d import load_hdr

    dest = asset_dir() / HDRI[0]
    try:
        if not dest.exists():
            _get(HDRI[1], dest)
        return load_hdr(dest)
    except Exception as e:                                   # offline: the built-in studio is used instead
        print(f"(no studio lighting download: {e}; using the built-in one)")
        return None


def samples(download: bool = True) -> list:
    """The built-in items, in a sensible order (one of each kind is worn by default)."""
    from .armour import SAMPLES as ARMOUR
    from .items import Item
    from .jewelry import SAMPLES
    from .mesh3d import load_model

    built = {name: Item(name=name, kind=kind, rgba=None, mesh=fn()) for name, kind, fn in SAMPLES + ARMOUR}
    got = {}
    if download:
        for name, spec in DOWNLOADS.items():
            try:
                path = fetch(name)
                got[name] = Item(name=name, kind=spec["kind"], rgba=None, path=path, credit=spec["credit"],
                                 mesh=load_model(path, spec["kind"], spec.get("meta")))
            except Exception as e:
                print(f"skipping {name}: {e}")
    order = ["sunglasses", "pearl drop earrings", "gold chain with diamond", "diamond solitaire ring",
             "chronograph watch", "armour helmet", "armour gauntlets", "round spectacles", "gold hoop earrings",
             "pearl necklace"]
    items: List = [x for n in order for x in [built.get(n) or got.get(n)] if x is not None]
    return items + [x for x in list(built.values()) + list(got.values()) if x not in items]
