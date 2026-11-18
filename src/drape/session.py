"""A try-on session: the items you're wearing, the tracker, and drawing them onto each frame."""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import cv2
import numpy as np

from . import anchors
from .items import Item
from .render import Light, composite_premultiplied, overlay
from .tracker import SEG_HAIR, Landmarks, Tracker, parts_for

SEG_OTHER = 5          # MediaPipe multiclass segmenter: accessories (headphones, hats, ...)


class Session:
    def __init__(self, items: List[Item], video: bool = True, wearing: Optional[List[int]] = None,
                 realism: bool = True) -> None:
        if not items:
            raise ValueError("no items to try on")
        self.items = items
        self.video = video
        self.realism = realism            # occlusion, scene lighting and shadows
        if wearing is None:               # default: wear the first item of each kind
            seen, wearing = set(), []
            for i, it in enumerate(items):
                if it.kind not in seen:
                    seen.add(it.kind)
                    wearing.append(i)
        self.wearing: List[int] = list(wearing)
        self.filters: Dict[str, anchors.OneEuro] = {}
        self.light = Light(alpha=0.15 if video else 1.0)
        parts = parts_for({it.kind for it in items})
        if not realism:
            parts.discard("seg")
        self.tracker = Tracker(parts, video=video)
        self._mesh = None
        self.last: Landmarks = Landmarks()
        self.timing: Dict[str, float] = {}

    # ---- what's worn ------------------------------------------------------------------------------------------
    def toggle(self, i: int) -> None:
        if not 0 <= i < len(self.items):
            return
        if i in self.wearing:
            self.wearing.remove(i)
            return
        kind = self.items[i].kind                       # one item per body slot: swap out the same kind
        self.wearing = [j for j in self.wearing if self.items[j].kind != kind] + [i]

    def cycle(self, step: int = 1) -> None:
        """Swap the most recently put-on item for the next one of the same kind."""
        if not self.wearing:
            self.wearing = [0]
            return
        cur = self.wearing[-1]
        kind = self.items[cur].kind
        same = [i for i, it in enumerate(self.items) if it.kind == kind]
        nxt = same[(same.index(cur) + step) % len(same)]
        if len(same) == 1:                               # nothing else of this kind: move on to the next item
            nxt = (cur + step) % len(self.items)
        self.toggle(nxt)

    def resize(self, factor: float) -> None:
        if self.wearing:
            it = self.items[self.wearing[-1]]
            it.scale = float(np.clip(it.scale * factor, 0.3, 3.0))

    def worn(self) -> List[Item]:
        return [self.items[i] for i in self.wearing]

    def reset_smoothing(self) -> None:
        self.filters.clear()

    # ---- per frame --------------------------------------------------------------------------------------------
    def _smooth(self, lm: Landmarks) -> Landmarks:
        """Smooth the landmarks themselves (not each item), so everything worn moves together, steadily."""
        if not self.video:
            return lm
        t = time.monotonic()

        def f(key, arr):
            if arr is None:
                return None
            flt = self.filters.setdefault(key, anchors.OneEuro())
            return flt(arr, t).astype(np.float32)

        lm.face = f("face", lm.face)
        lm.pose = f("pose", lm.pose)
        seen = {}
        hands = []
        for h, lb in zip(lm.hands, lm.hand_labels + [""] * len(lm.hands)):
            k = f"hand:{lb}:{seen.get(lb, 0)}"
            seen[lb] = seen.get(lb, 0) + 1
            hands.append(f(k, h))
        lm.hands = hands
        return lm

    def _masks(self, frame: np.ndarray, lm: Landmarks, need: set) -> Dict[str, np.ndarray]:
        """Float masks (1 = something in front): hair and accessories from segmentation, the face/chin."""
        masks: Dict[str, np.ndarray] = {}
        if not self.realism:
            return masks
        seg = None if lm.seg is None else np.squeeze(lm.seg)
        if "hair" in need and seg is not None:
            masks["hair"] = cv2.GaussianBlur((seg == SEG_HAIR).astype(np.float32), (5, 5), 0)
        if "cover" in need and seg is not None:
            masks["cover"] = cv2.GaussianBlur(((seg == SEG_HAIR) | (seg == SEG_OTHER)).astype(np.float32), (5, 5), 0)
        if "face" in need and lm.face is not None:
            m = np.zeros(frame.shape[:2], np.float32)
            cv2.fillPoly(m, [lm.face[anchors.FACE_OVAL, :2].astype(np.int32)], 1.0, cv2.LINE_AA)
            masks["face"] = cv2.GaussianBlur(m, (5, 5), 0)
        return masks

    @staticmethod
    def _covered(masks, p: anchors.Placement, r: int) -> bool:
        """Is the ear lobe itself under hair or headphones? Then the earring hangs beneath them; otherwise the
        hair seen around it is behind it and must not hide it."""
        m = masks.get("cover")
        if m is None:
            return False
        x, y = int(p.x), int(p.y)
        patch = m[max(y - r, 0):y + r + 1, max(x - r, 0):x + r + 1]
        return patch.size > 0 and float(patch.mean()) > 0.5

    def render(self, frame: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        need = parts_for({self.items[i].kind for i in self.wearing})
        lm = self._smooth(self.tracker(frame, need)) if need else Landmarks()
        self.last = lm
        t1 = time.perf_counter()
        plan = []
        for idx in self.wearing:
            it = self.items[idx]
            for slot, p in enumerate(anchors.place(it.kind, face=lm.face, pose=lm.pose, hands=lm.hands,
                                                   hand_labels=lm.hand_labels)):
                plan.append((it, slot, p))
        masks = self._masks(frame, lm, {o for _, _, p in plan for o in p.occlude})
        if self.realism and plan:
            self.light.update(frame)
        for it, slot, p in plan:
            occ = p.occlude
            if it.kind == "earrings":
                r = max(2, int(0.15 * p.width))
                occ = ("cover",) if self._covered(masks, p, r) else ()
            mask = None
            for name in occ:
                m = masks.get(name)
                if m is not None:
                    mask = m if mask is None else np.maximum(mask, m)
            if it.is_3d and p.pose3d is not None:
                self._draw_3d(frame, it, p, mask)
            else:
                overlay(frame, it, anchors.replace(p, occlude=occ), masks=masks,
                        light=self.light if self.realism else None, shadow=self.realism)
        self.timing = {"track": (t1 - t0) * 1000, "draw": (time.perf_counter() - t1) * 1000}
        return frame

    def _renderer(self):
        if self._mesh is None:
            from .assets import environment
            from .mesh3d import MeshRenderer
            self._mesh = MeshRenderer(environment())
        return self._mesh

    def _draw_3d(self, frame: np.ndarray, it: Item, p: anchors.Placement, mask) -> None:
        model = it.mesh
        fit = 1.0
        if it.kind == "glasses" and model.info.get("width"):
            fit = (anchors.HEAD_W + anchors.GLASSES_GAP) / model.info["width"]
        gain, tint = (self.light.gain, self.light.tint) if self.realism else (1.0, np.ones(3))
        out = self._renderer().render(frame.shape, model, p.pose3d, fit * it.scale, gain, tint)
        if out is not None:
            patch, x0, y0 = out
            composite_premultiplied(frame, patch, x0, y0, mask)

    def close(self) -> None:
        self.tracker.close()
        if self._mesh is not None:
            self._mesh.close()
