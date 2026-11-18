"""drape: try on jewellery, glasses and watches from your own photos or 3D models, live on your webcam.

  drape live  ITEM [ITEM ...]          live camera try-on (items: images, .glb models, folders, or file:kind)
  drape photo PHOTO ITEM [...] -o OUT  try items on a still photo
  drape kinds                          list the kinds of items drape understands

Live keys: 1-9 put on / take off item, n / p next / previous item, + / - bigger / smaller,
           s save a snapshot, f flip mirror view, h show/hide help, q or Esc quit.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

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
    s = Session(load_many(a.items), video=False, wearing=list(range(len(a.items))) if a.all else None,
                realism=not a.simple)
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


def cmd_live(a) -> int:
    import cv2

    from .session import Session

    items = load_many(a.items)
    s = Session(items, video=True, realism=not a.simple)
    cap = cv2.VideoCapture(a.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
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
                lines = [f"wearing: {', '.join(i.name for i in s.worn()) or 'nothing'}   {fps:4.1f} fps"]
                lines += [f"{i + 1}. {it.name} ({it.kind})" for i, it in enumerate(items[:9])] + ["  ".join(HELP)]
                for k, line in enumerate(lines):
                    y = 28 + 24 * k
                    cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
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
                if s.smooth:
                    s.smooth.reset()
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
    lv.add_argument("items", nargs="+", help="item images, folders of them, or path:kind")
    lv.add_argument("--camera", type=int, default=0)
    lv.add_argument("--width", type=int, default=1280)
    lv.add_argument("--height", type=int, default=720)
    lv.add_argument("--simple", action="store_true", help="no occlusion, scene lighting or shadows (faster)")
    lv.set_defaults(fn=cmd_live)

    ph = sub.add_parser("photo", help="try items on a still photo")
    ph.add_argument("photo")
    ph.add_argument("items", nargs="+")
    ph.add_argument("-o", "--output")
    ph.add_argument("--all", action="store_true", help="wear every item given, not one per kind")
    ph.add_argument("--simple", action="store_true", help="no occlusion, scene lighting or shadows")
    ph.set_defaults(fn=cmd_photo)

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
