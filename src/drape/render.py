"""Draw a cut-out item onto a frame at a placement (rotate + scale about its anchor, then alpha-blend)."""
from __future__ import annotations

import math

import cv2
import numpy as np

from .anchors import Placement
from .items import Item


def overlay(frame: np.ndarray, item: Item, p: Placement) -> np.ndarray:
    """Blend `item` onto `frame` (BGR, modified in place and returned)."""
    rgba = item.rgba[:, ::-1] if p.mirror else item.rgba
    h, w = rgba.shape[:2]
    ax, ay = item.anchor_px
    if p.mirror:
        ax = w - ax
    s = (p.width * item.scale) / max(w, 1)
    if s <= 0 or not math.isfinite(s):
        return frame
    # affine: item anchor -> target point, scaled by s, rotated by angle (CCW in image terms)
    M = cv2.getRotationMatrix2D((ax, ay), p.angle, s)
    M[0, 2] += p.x - ax
    M[1, 2] += p.y - ay

    # warp only into the bounding box of the transformed item (fast)
    corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], dtype=float) @ M.T
    fh, fw = frame.shape[:2]
    x0, y0 = np.floor(corners.min(axis=0)).astype(int)
    x1, y1 = np.ceil(corners.max(axis=0)).astype(int)
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, fw), min(y1, fh)
    if x1 <= x0 or y1 <= y0:
        return frame
    M2 = M.copy()
    M2[0, 2] -= x0
    M2[1, 2] -= y0
    warped = cv2.warpAffine(rgba, M2, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    a = warped[:, :, 3:4].astype(np.float32) / 255.0
    roi = frame[y0:y1, x0:x1]
    roi[:] = (warped[:, :, :3].astype(np.float32) * a + roi.astype(np.float32) * (1 - a)).astype(np.uint8)
    return frame
