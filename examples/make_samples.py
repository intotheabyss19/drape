"""Draw the sample items in examples/items/ (simple, original artwork so the repo ships with no licensed images).
Saved as JPEGs on a white background on purpose, to exercise drape's background removal. Also writes a 3D
ring and watch (.glb) built from simple shapes.

    uv run python examples/make_samples.py
"""
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).parent / "items"
GOLD, GOLD_D, SILVER, PEARL = (40, 175, 225), (20, 120, 170), (200, 200, 205), (235, 240, 245)


def canvas(w, h):
    return np.full((h, w, 3), 255, np.uint8)


def necklace():
    img = canvas(600, 420)
    pts = []
    for t in np.linspace(0, np.pi, 120):        # a U-shaped chain hanging from the two top corners
        pts.append((int(300 - 270 * np.cos(t)), int(10 + 300 * np.sin(t) ** 1.3)))
    for i, (x, y) in enumerate(pts):
        cv2.circle(img, (x, y), 7, GOLD_D if i % 2 else GOLD, -1, cv2.LINE_AA)
        cv2.circle(img, (x - 2, y - 2), 2, (180, 240, 255), -1, cv2.LINE_AA)
    cx, cy = pts[60]
    cv2.ellipse(img, (cx, cy + 48), (34, 46), 0, 0, 360, GOLD, -1, cv2.LINE_AA)       # pendant
    cv2.ellipse(img, (cx, cy + 48), (20, 30), 0, 0, 360, (60, 40, 180), -1, cv2.LINE_AA)  # ruby
    cv2.ellipse(img, (cx - 6, cy + 38), (6, 9), 0, 0, 360, (150, 140, 255), -1, cv2.LINE_AA)
    return img


def earring():
    img = canvas(120, 260)
    cv2.circle(img, (60, 14), 10, GOLD, 3, cv2.LINE_AA)                                # hook
    cv2.line(img, (60, 24), (60, 70), GOLD, 4, cv2.LINE_AA)
    for k, y in enumerate((90, 135, 185)):
        cv2.circle(img, (60, y), 20 + 4 * k, PEARL, -1, cv2.LINE_AA)
        cv2.circle(img, (60, y), 20 + 4 * k, (190, 190, 200), 2, cv2.LINE_AA)
        cv2.circle(img, (53, y - 7), 5, (255, 255, 255), -1, cv2.LINE_AA)
    return img


def glasses():
    img = canvas(640, 220)
    for cx in (175, 465):
        cv2.ellipse(img, (cx, 110), (120, 82), 0, 0, 360, (70, 60, 50), -1, cv2.LINE_AA)   # tinted lens
        cv2.ellipse(img, (cx, 110), (120, 82), 0, 0, 360, (20, 20, 20), 14, cv2.LINE_AA)   # frame
        cv2.ellipse(img, (cx - 40, 85), (28, 12), -20, 0, 360, (130, 120, 110), -1, cv2.LINE_AA)
    cv2.ellipse(img, (320, 98), (55, 30), 0, 200, 340, (20, 20, 20), 12, cv2.LINE_AA)      # bridge
    cv2.line(img, (55, 95), (8, 85), (20, 20, 20), 12, cv2.LINE_AA)
    cv2.line(img, (585, 95), (632, 85), (20, 20, 20), 12, cv2.LINE_AA)
    return img


def ring():
    img = canvas(220, 200)
    cv2.ellipse(img, (110, 120), (90, 46), 0, 0, 360, SILVER, 16, cv2.LINE_AA)
    cv2.ellipse(img, (110, 120), (90, 46), 0, 200, 340, (240, 240, 245), 4, cv2.LINE_AA)
    pts = np.array([[110, 30], [140, 62], [110, 92], [80, 62]], np.int32)               # diamond
    cv2.fillPoly(img, [pts], (240, 220, 190), cv2.LINE_AA)
    cv2.polylines(img, [pts], True, (170, 150, 120), 3, cv2.LINE_AA)
    return img


def watch():
    img = canvas(420, 180)
    cv2.rectangle(img, (10, 55), (410, 125), (40, 40, 45), -1, cv2.LINE_AA)               # strap
    cv2.circle(img, (210, 90), 78, SILVER, -1, cv2.LINE_AA)
    cv2.circle(img, (210, 90), 64, (30, 30, 35), -1, cv2.LINE_AA)
    for k in range(12):
        a = k * np.pi / 6
        cv2.line(img, (int(210 + 52 * np.cos(a)), int(90 + 52 * np.sin(a))),
                 (int(210 + 60 * np.cos(a)), int(90 + 60 * np.sin(a))), (230, 230, 230), 3, cv2.LINE_AA)
    cv2.line(img, (210, 90), (210, 48), (230, 230, 230), 4, cv2.LINE_AA)
    cv2.line(img, (210, 90), (240, 104), (230, 230, 230), 4, cv2.LINE_AA)
    return img


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name, fn in [("gold_necklace", necklace), ("pearl_earrings", earring), ("black_sunglasses", glasses),
                     ("diamond_ring", ring), ("steel_watch", watch)]:
        cv2.imwrite(str(OUT / f"{name}.jpg"), fn(), [cv2.IMWRITE_JPEG_QUALITY, 92])
        print("wrote", OUT / f"{name}.jpg")
    from drape.mesh3d import write_samples          # 3D ring and watch (.glb + settings .json)
    write_samples(OUT)
    print("wrote 3D samples: gold_ring_3d.glb, leather_watch_3d.glb")
