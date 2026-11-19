"""drape: try on jewellery, glasses and watches, live on your webcam: 3D models or your own photos of items.

  drape live  [ITEM ...]               live camera try-on (no items: the built-in 3D samples; items: .glb/.gltf
                                       models, photos, folders, or file:kind)
  drape photo PHOTO [ITEM ...] -o OUT  try items on a still photo
  drape samples                        download the built-in models and list them (with credits)
  drape kinds                          list the kinds of items drape understands

Live keys: 1-9 put on / take off item, n / p next / previous item, + / - bigger / smaller,
           s save a snapshot, f flip mirror view, h show/hide help, q or Esc quit.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

from . import __version__
from .anchors import KINDS
from .items import load_many

HELP = ["1-9 wear/remove", "n/p next/prev", "+/- size", "s snapshot", "f mirror", "h help", "q quit"]


def _snap_dir() -> Path:
    d = Path.home() / "Pictures" / "drape"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cmd_photo(a) -> int:
    import cv2

    from .session import Session

    frame = cv2.imread(a.photo)
    if frame is None:
        print(f"can't read {a.photo}", file=sys.stderr)
        return 2
    items = load_many(a.items)
    s = Session(items, video=False, wearing=list(range(len(items))) if a.all else None, realism=not a.simple)
    out = s.render(frame)
    s.close()
    lm = s.last
    found = [n for n, v in (("face", lm.face), ("body", lm.pose)) if v is not None] + \
            ([f"{len(lm.hands)} hand(s)"] if lm.hands else [])
    print("found:", ", ".join(found) or "nothing", "| wearing:", ", ".join(i.name for i in s.worn()))
    dest = a.output or str(_snap_dir() / f"drape-{time.strftime('%Y%m%d-%H%M%S')}.jpg")
    cv2.imwrite(dest, out)
    print("saved", dest)
    return 0


def _panel(frame, lines) -> None:
    """Help text on a translucent dark panel (one pass of white text: crisp at any OpenCV version)."""
    import cv2

    font, scale, lh = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 22
    w = max(cv2.getTextSize(line, font, scale, 1)[0][0] for line in lines) + 24
    h = lh * len(lines) + 14
    roi = frame[:h, :w]
    roi[:] = (roi.astype(np.float32) * 0.45).astype(np.uint8)
    for k, line in enumerate(lines):
        cv2.putText(frame, line, (12, 22 + lh * k), font, scale, (255, 255, 255), 1, cv2.LINE_AA)


class _Latest:
    """Runs `produce` over and over on its own thread and hands out the newest result. The live view chains two:
    the camera, and tracking + drawing. So reading the camera, processing a frame and showing the previous one
    overlap instead of taking turns (taking turns, 27 ms of processing plus ~13 ms for the window turned a
    30 fps camera into 15 fps). `produce` returns None when there's nothing more."""

    def __init__(self, produce) -> None:
        self.produce, self._val, self._n, self._seen, self.ok = produce, None, 0, 0, True
        self.error: Optional[BaseException] = None
        self._cv = threading.Condition()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self) -> None:
        while self.ok:
            try:
                v = self.produce()
            except BaseException as e:  # noqa: BLE001 (re-raised by read() instead of dying silently)
                v, self.error = None, e
            with self._cv:
                if v is None:
                    self.ok = False
                else:
                    self._val, self._n = v, self._n + 1
                self._cv.notify_all()

    def read(self, timeout: float = 2.0):
        """The newest result not handed out yet (waits for it), or None when the producer has stopped."""
        with self._cv:
            self._cv.wait_for(lambda: self._n != self._seen or not self.ok, timeout=timeout)
            if self._n == self._seen:
                if self.error is not None:
                    raise self.error
                return None
            self._seen = self._n
            return self._val

    def close(self) -> None:
        self.ok = False
        self._t.join(timeout=2.0)


def _camera(cap) -> _Latest:
    def grab():
        ok, frame = cap.read()
        return frame if ok else None
    return _Latest(grab)


def cmd_samples(a) -> int:
    from .assets import asset_dir, samples

    for it in samples():
        print(f"{it.name:28s} {it.kind:9s} {it.credit or 'built by drape (Apache-2.0)'}")
    print(f"\ndownloaded models are in {asset_dir()}")
    return 0


def cmd_live(a) -> int:
    import cv2

    from .session import Session

    items = load_many(a.items)
    s = Session(items, video=True, realism=not a.simple)
    cap = cv2.VideoCapture(a.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        print("can't open the camera (on macOS, allow camera access for your terminal app in "
              "System Settings > Privacy & Security > Camera)", file=sys.stderr)
        return 2
    show_help, fps, t_prev = True, 0.0, time.monotonic()
    state = {"mirror": True}
    t_start, t_rec, n_rec, log = t_prev, 0.0, 0, None
    if a.record:                                   # a frame a second, raw and dressed, for bug reports
        rec = Path(a.record).expanduser()
        rec.mkdir(parents=True, exist_ok=True)
        log = open(rec / "log.jsonl", "a")  # noqa: SIM115 (closed in the finally below)
        print("recording to", rec)
    win = "drape"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cam = _camera(cap)

    def process():                                 # on the worker thread (which also owns the OpenGL context)
        frame = cam.read()
        if frame is None:
            return None
        if state["mirror"]:
            frame = cv2.flip(frame, 1)
        raw = frame.copy() if log else None
        s.render(frame)
        lm = s.last
        return frame, raw, dict(s.timing), list(s.wearing), lm

    work = _Latest(process)
    try:
        while True:
            got = work.read()
            if got is None:
                break
            frame, raw, timing, wearing, lm = got
            clean = frame.copy()                       # snapshot without the help text
            now = time.monotonic()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-3))
            t_prev = now
            if show_help:
                lines = [(f"{fps:4.1f} fps  (tracking {timing.get('track', 0):.0f} ms, "
                          f"drawing {timing.get('draw', 0):.0f} ms)")]
                lines += [("* " if i in wearing else "  ") + f"{i + 1}. {it.name}" for i, it in enumerate(items[:9])]
                lines += ["  ".join(HELP)]
                _panel(frame, lines)
            if log and now - t_rec >= a.every:
                t_rec, n_rec = now, n_rec + 1
                cv2.imwrite(str(rec / f"{n_rec:04d}_raw.jpg"), raw, [cv2.IMWRITE_JPEG_QUALITY, 92])
                cv2.imwrite(str(rec / f"{n_rec:04d}_out.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                log.write(json.dumps({"n": n_rec, "t": round(now - t_start, 2), "fps": round(fps, 1),
                                      **{k: round(v, 1) for k, v in timing.items()},
                                      "wearing": [items[i].name for i in wearing], "face": lm.face is not None,
                                      "pose": lm.pose is not None, "hands": lm.hand_labels}) + "\n")
                log.flush()
            if a.seconds and now - t_start > a.seconds:
                break
            cv2.imshow(win, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if ord("1") <= key <= ord("9"):
                s.toggle(key - ord("1"))
            elif key == ord("n"):
                s.cycle(1)
            elif key == ord("p"):
                s.cycle(-1)
            elif key in (ord("+"), ord("=")):
                s.resize(1.08)
            elif key == ord("-"):
                s.resize(1 / 1.08)
            elif key == ord("f"):
                state["mirror"] = not state["mirror"]
                s.reset_smoothing()
            elif key == ord("h"):
                show_help = not show_help
            elif key == ord("s"):
                path = _snap_dir() / f"drape-{time.strftime('%Y%m%d-%H%M%S')}.jpg"
                cv2.imwrite(str(path), clean)
                print("saved", path)
    finally:
        work.close()
        cam.close()
        cap.release()
        cv2.destroyAllWindows()
        if log:
            log.close()
        s.close()
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="drape", description="Try on wearables from your own photos, locally.")
    p.add_argument("--version", action="version", version=f"drape {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    lv = sub.add_parser("live", help="live webcam try-on")
    lv.add_argument("items", nargs="*", help="3D models, item photos, folders of them, or path:kind "
                                              "(default: the built-in samples)")
    lv.add_argument("--camera", type=int, default=0)
    lv.add_argument("--width", type=int, default=1280)
    lv.add_argument("--height", type=int, default=720)
    lv.add_argument("--simple", action="store_true", help="no occlusion, scene lighting or shadows (faster)")
    lv.add_argument("--record", metavar="DIR", help="save the camera frame and the result every --every "
                                                  "seconds, with a log, to DIR (for bug reports)")
    lv.add_argument("--every", type=float, default=1.0, help="seconds between recorded frames (default 1)")
    lv.add_argument("--seconds", type=float, default=0, help="stop after this many seconds")
    lv.set_defaults(fn=cmd_live)

    ph = sub.add_parser("photo", help="try items on a still photo")
    ph.add_argument("photo")
    ph.add_argument("items", nargs="*")
    ph.add_argument("-o", "--output")
    ph.add_argument("--all", action="store_true", help="wear every item given, not one per kind")
    ph.add_argument("--simple", action="store_true", help="no occlusion, scene lighting or shadows")
    ph.set_defaults(fn=cmd_photo)

    sm = sub.add_parser("samples", help="download the built-in models and list them")
    sm.set_defaults(fn=cmd_samples)

    kd = sub.add_parser("kinds", help="list item kinds")
    kd.set_defaults(fn=lambda a: print("\n".join(KINDS)) or 0)

    a = p.parse_args(argv)
    try:
        return a.fn(a)
    except ValueError as e:
        print(f"drape: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
