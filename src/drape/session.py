"""A try-on session: the items you're wearing, the tracker, and drawing them onto each frame."""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from . import anchors
from .items import Item
from .render import overlay
from .tracker import Landmarks, Tracker, parts_for


class Session:
    def __init__(self, items: List[Item], video: bool = True, wearing: Optional[List[int]] = None) -> None:
        if not items:
            raise ValueError("no items to try on")
        self.items = items
        self.video = video
        # default: wear the first item of each kind
        if wearing is None:
            seen, wearing = set(), []
            for i, it in enumerate(items):
                if it.kind not in seen:
                    seen.add(it.kind)
                    wearing.append(i)
        self.wearing: List[int] = list(wearing)
        self.smooth = anchors.Smoother() if video else None
        self.tracker = Tracker(parts_for({it.kind for it in items}), video=video)
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
    def render(self, frame: np.ndarray) -> np.ndarray:
        lm = self.tracker(frame)
        self.last = lm
        for idx in self.wearing:
            it = self.items[idx]
            for slot, p in enumerate(anchors.place(it.kind, face=lm.face, pose=lm.pose, hands=lm.hands)):
                if self.smooth is not None:
                    p = self.smooth((it.kind, slot), p)
                overlay(frame, it, p)
        return frame

    def close(self) -> None:
        self.tracker.close()
