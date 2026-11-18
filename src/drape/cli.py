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
import sys
import time
from pathlib import Path

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
    mirror, show_help, fps, t_prev = True, True, 0.0, time.monotonic()
    win = "drape"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if mirror:
                frame = cv2.flip(frame, 1)
            s.render(frame)
            clean = frame.copy()                       # snapshot without the help text
            now = time.monotonic()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-3))
            t_prev = now
            if show_help:
                t = s.timing
                lines = [f"{fps:4.1f} fps  (tracking {t.get('track', 0):.0f} ms, drawing {t.get('draw', 0):.0f} ms)"]
                lines += [("* " if i in s.wearing else "  ") + f"{i + 1}. {it.name}" for i, it in enumerate(items[:9])]
                lines += ["  ".join(HELP)]
                _panel(frame, lines)
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
                mirror = not mirror
                s.reset_smoothing()
            elif key == ord("h"):
                show_help = not show_help
            elif key == ord("s"):
                path = _snap_dir() / f"drape-{time.strftime('%Y%m%d-%H%M%S')}.jpg"
                cv2.imwrite(str(path), clean)
                print("saved", path)
    finally:
        cap.release()
        cv2.destroyAllWindows()
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
