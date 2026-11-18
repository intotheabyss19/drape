"""A try-on session: the items you're wearing, the tracker, and drawing them onto each frame."""
from __future__ import annotations

from typing import Dict, List, Optional

import cv2
import numpy as np

from . import anchors
from .items import Item
from .render import Light, composite_rgba, overlay
from .tracker import SEG_HAIR, Landmarks, Tracker, parts_for


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
        self.smooth = anchors.Smoother() if video else None
        self.light = Light(alpha=0.15 if video else 1.0)
        parts = parts_for({it.kind for it in items})
        if not realism:
            parts.discard("seg")
        self.tracker = Tracker(parts, video=video)
        self._mesh = None
        self.last: Landmarks = Landmarks()

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

    # ---- per frame --------------------------------------------------------------------------------------------
    def _masks(self, frame: np.ndarray, lm: Landmarks, need: set) -> Dict[str, np.ndarray]:
        """Float masks (1 = something in front): hair from segmentation, the face/chin from the face mesh."""
        masks: Dict[str, np.ndarray] = {}
        if not self.realism:
            return masks
        if "hair" in need and lm.seg is not None:
            hair = (lm.seg == SEG_HAIR).astype(np.float32)
            masks["hair"] = cv2.GaussianBlur(hair, (5, 5), 0)
        if "face" in need and lm.face is not None:
            m = np.zeros(frame.shape[:2], np.float32)
            cv2.fillPoly(m, [lm.face[anchors.FACE_OVAL, :2].astype(np.int32)], 1.0, cv2.LINE_AA)
            masks["face"] = cv2.GaussianBlur(m, (5, 5), 0)
        return masks

    @staticmethod
    def _hair_over(masks, p: anchors.Placement) -> bool:
        """Is the ear lobe itself under hair? Then the earring hangs beneath the hair; otherwise the hair seen
        around it is behind it, and must not hide it."""
        hair = masks.get("hair")
        if hair is None:
            return False
        r = max(2, int(0.25 * p.width))
        x, y = int(p.x), int(p.y)
        patch = hair[max(y - r, 0):y + r, max(x - r, 0):x + r]
        return patch.size > 0 and float(patch.mean()) > 0.4

    def render(self, frame: np.ndarray) -> np.ndarray:
        lm = self.tracker(frame)
        self.last = lm
        plan = []
        for idx in self.wearing:
            it = self.items[idx]
            for slot, p in enumerate(anchors.place(it.kind, face=lm.face, pose=lm.pose, hands=lm.hands,
                                                   hand_labels=lm.hand_labels)):
                if self.smooth is not None:
                    p = self.smooth((it.kind, slot), p)
                plan.append((it, slot, p))
        masks = self._masks(frame, lm, {o for _, _, p in plan for o in p.occlude})
        if self.realism and plan:
            self.light.update(frame)
        for it, slot, p in plan:
            if it.kind == "earrings" and not self._hair_over(masks, p):
                p = anchors.replace(p, occlude=())
            if it.is_3d and p.pose3d is not None:
                self._draw_3d(frame, it, slot, p)
            else:
                overlay(frame, it, p, masks=masks, light=self.light if self.realism else None,
                        shadow=self.realism)
        return frame

    def _draw_3d(self, frame: np.ndarray, it: Item, slot: int, p: anchors.Placement) -> None:
        if self._mesh is None:
            from .mesh3d import MeshRenderer
            self._mesh = MeshRenderer()
        gain, tint = (self.light.gain, self.light.tint) if self.realism else (1.0, np.ones(3))
        out = self._mesh.render(frame.shape, it.mesh, p.pose3d, it.scale, gain, tint)
        if out is not None:
            patch, x0, y0 = out
            composite_rgba(frame, patch, x0, y0)

    def close(self) -> None:
        self.tracker.close()
        if self._mesh is not None:
            self._mesh.close()
