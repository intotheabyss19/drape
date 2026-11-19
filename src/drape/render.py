"""Draw a flat (photo) item onto a frame: perspective warp onto its 3D placement, hide the parts that are behind
hair or the chin, match the scene's brightness and colour, and drop a soft contact shadow."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .anchors import Placement
from .items import Item


class Light:
    """Estimates the scene's light from the whole frame ("grey world"), smoothed over frames.

    gain scales the item's brightness toward the scene's (a dim room darkens a bright product photo); tint
    nudges its colour toward the light's colour (warm bulb -> warmer item). Measuring the whole frame rather
    than the spot behind the item matters: a gold chain on a black shirt is not in a dark room."""

    def __init__(self, alpha: float = 0.15) -> None:
        self.alpha = alpha
        self.gain, self.tint = 1.0, np.ones(3, np.float32)
        self._seen = False

    @staticmethod
    def measure(frame: np.ndarray) -> Tuple[float, np.ndarray]:
        small = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA).reshape(-1, 3).astype(np.float32)
        mean = small.mean(axis=0) + 1.0                                           # BGR
        lum = float(0.114 * mean[0] + 0.587 * mean[1] + 0.299 * mean[2]) / 255.0
        gain = float(np.clip((lum / 0.42) ** 0.75, 0.35, 1.1))
        tint = np.clip(mean / mean.mean(), 0.85, 1.2)
        return gain, tint

    def update(self, frame: np.ndarray) -> Tuple[float, np.ndarray]:
        g, t = self.measure(frame)
        if self._seen:
            a = self.alpha
            g, t = self.gain + a * (g - self.gain), self.tint + a * (t - self.tint)
        self.gain, self.tint, self._seen = g, t, True
        return g, t


def _item_at_scale(item: Item, target_w: float, mirror: bool):
    """The item image, pre-shrunk with area averaging when it will be drawn much smaller (avoids jaggies),
    plus its attach point in that image."""
    rgba = item.rgba
    h, w = rgba.shape[:2]
    ax, ay = item.anchor_px
    s = target_w / max(w, 1)
    if s < 0.5:
        f = min(1.0, 1.6 * s)
        rgba = cv2.resize(rgba, (max(2, int(w * f)), max(2, int(h * f))), interpolation=cv2.INTER_AREA)
        ax, ay = ax * rgba.shape[1] / w, ay * rgba.shape[0] / h
    if mirror:
        rgba = rgba[:, ::-1]
        ax = rgba.shape[1] - ax
    return rgba, ax, ay


def overlay(frame: np.ndarray, item: Item, p: Placement, masks: Optional[Dict[str, np.ndarray]] = None,
            light: Optional[Light] = None, shadow: bool = True) -> np.ndarray:
    """Blend `item` onto `frame` (BGR, modified in place and returned)."""
    if p.opacity <= 0.01 or item.rgba is None:
        return frame
    target_w = p.width * item.scale
    rgba, ax, ay = _item_at_scale(item, target_w, p.mirror)
    h, w = rgba.shape[:2]
    scaled = Placement(p.origin, p.ex, p.ey, target_w)
    quad = scaled.quad(w, h, ax, ay)
    if not np.all(np.isfinite(quad)):
        return frame

    fh, fw = frame.shape[:2]
    pad = int(0.08 * target_w) + 3                     # room for the shadow
    x0, y0 = np.floor(quad.min(axis=0)).astype(int) - pad
    x1, y1 = np.ceil(quad.max(axis=0)).astype(int) + pad
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, fw), min(y1, fh)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return frame
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    M = cv2.getPerspectiveTransform(src, (quad - [x0, y0]).astype(np.float32))
    warped = cv2.warpPerspective(rgba, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    alpha = warped[:, :, 3].astype(np.float32) * (p.opacity / 255.0)
    for name in p.occlude:                              # hair / chin in front of the item
        m = (masks or {}).get(name)
        if m is not None:
            alpha *= 1.0 - m[y0:y1, x0:x1]

    rgb = warped[:, :, :3].astype(np.float32)
    if light is not None:
        rgb *= light.gain * (0.75 + 0.25 * light.tint)
    roi = frame[y0:y1, x0:x1].astype(np.float32)

    if shadow:
        k = max(3, int(0.05 * target_w) | 1)
        dx, dy = 0.01 * target_w, 0.03 * target_w
        sh = cv2.warpAffine(alpha, np.float32([[1, 0, dx], [0, 1, dy]]), (alpha.shape[1], alpha.shape[0]))
        sh = cv2.GaussianBlur(sh, (k, k), 0)
        roi *= (1.0 - 0.28 * sh)[:, :, None]

    a = alpha[:, :, None]
    frame[y0:y1, x0:x1] = np.clip(rgb * a + roi * (1.0 - a), 0, 255).astype(np.uint8)
    return frame


def composite_premultiplied(frame: np.ndarray, bgra: np.ndarray, x0: int, y0: int,
                            mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Blend a premultiplied-alpha BGRA patch (a rendered 3D item) at (x0, y0); `mask` (1 = something in front
    of the item, full-frame) hides parts of it."""
    h, w = bgra.shape[:2]
    roi = frame[y0:y0 + h, x0:x0 + w]
    a = bgra[:, :, 3:4].astype(np.float32) / 255.0
    rgb = bgra[:, :, :3].astype(np.float32)
    if mask is not None:
        keep = 1.0 - mask[y0:y0 + h, x0:x0 + w, None]
        a, rgb = a * keep, rgb * keep
    roi[:] = np.clip(rgb + roi.astype(np.float32) * (1 - a), 0, 255).astype(np.uint8)
    return frame
