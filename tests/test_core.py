"""Tests that need no camera and no MediaPipe models: geometry, cut-outs, drawing, item loading."""
from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from drape import anchors
from drape.anchors import Placement
from drape.items import Item, crop_to_content, cutout, guess_kind, load, load_many, remove_plain_background
from drape.render import overlay


# ---------------------------------------------------------------------------------------------- fake landmarks
def fake_face(cx=320.0, cy=240.0, w=160.0, h=200.0, tilt_deg=0.0) -> np.ndarray:
    f = np.zeros((478, 3), np.float32)
    pts = {
        anchors.F_FOREHEAD: (0, -h / 2), anchors.F_CHIN: (0, h / 2),
        anchors.F_EYE_R_OUT: (-w * 0.35, -h * 0.1), anchors.F_EYE_L_OUT: (w * 0.35, -h * 0.1),
        anchors.F_BRIDGE: (0, -h * 0.1), anchors.F_TRAGUS_R: (-w / 2, 0), anchors.F_TRAGUS_L: (w / 2, 0),
    }
    t = np.radians(tilt_deg)
    rot = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])   # CCW on screen (y down)
    for i, (dx, dy) in pts.items():
        f[i, :2] = rot @ np.array([dx, dy]) + (cx, cy)
    return f


def fake_pose(cx=320.0, y=400.0, span=260.0) -> np.ndarray:
    p = np.zeros((33, 3), np.float32)
    p[anchors.P_SHOULDER_L, :2] = (cx + span / 2, y)       # subject's left appears on image right (mirror view)
    p[anchors.P_SHOULDER_R, :2] = (cx - span / 2, y)
    return p


def fake_hand_up(x=300.0, wrist_y=400.0) -> np.ndarray:
    """An upright hand: fingers point up the screen."""
    hnd = np.zeros((21, 3), np.float32)
    hnd[anchors.H_WRIST, :2] = (x, wrist_y)
    hnd[anchors.H_INDEX_MCP, :2] = (x - 40, wrist_y - 90)
    hnd[anchors.H_MIDDLE_MCP, :2] = (x - 12, wrist_y - 95)
    hnd[anchors.H_RING_MCP, :2] = (x + 14, wrist_y - 92)
    hnd[anchors.H_RING_PIP, :2] = (x + 16, wrist_y - 140)
    hnd[anchors.H_PINKY_MCP, :2] = (x + 38, wrist_y - 82)
    return hnd


# ---------------------------------------------------------------------------------------------- anchors
def test_glasses_sit_on_the_bridge_and_follow_head_tilt():
    p = anchors.glasses(fake_face())
    assert (p.x, p.y) == pytest.approx((320, 220), abs=0.5)
    assert p.width == pytest.approx(1.1 * 160, rel=1e-3)
    assert p.angle == pytest.approx(0, abs=1e-3)
    assert anchors.glasses(fake_face(tilt_deg=15)).angle == pytest.approx(15, abs=0.5)


def test_earrings_hang_below_each_ear_and_the_left_is_mirrored():
    r, l = anchors.earrings(fake_face())
    assert r.x < 320 - 80 and l.x > 320 + 80          # outside the face edges
    assert r.y > 240 and l.y > 240                     # below ear level
    assert not r.mirror and l.mirror


def test_necklace_sits_between_chin_and_shoulders():
    p = anchors.necklace(fake_pose(), fake_face())
    chin_y, shoulder_y = 340, 400
    assert chin_y < p.y < shoulder_y and p.x == pytest.approx(320)
    assert p.width == pytest.approx(0.46 * 260)
    assert abs(p.angle) < 1e-6
    assert anchors.necklace(fake_pose()).y < 400       # works without a face too


def test_ring_and_watch_lie_across_an_upright_hand():
    hnd = fake_hand_up()
    ring, watch = anchors.ring(hnd), anchors.wrist(hnd)
    assert 400 - 140 < ring.y < 400 - 92                # between the ring finger's knuckles
    assert abs(ring.angle) < 10 and abs(watch.angle) < 15
    assert watch.y > 400                                # a little down the forearm


def test_place_needs_the_right_landmarks():
    assert anchors.place("glasses") == []
    assert anchors.place("necklace", face=fake_face()) == []
    assert len(anchors.place("earrings", face=fake_face())) == 2
    assert len(anchors.place("ring", hands=[fake_hand_up(), fake_hand_up(500)])) == 2


def test_smoother_moves_part_way_and_wraps_angles():
    s = anchors.Smoother(alpha=0.5)
    s("k", Placement(0, 0, 100, 170))
    p = s("k", Placement(10, 0, 100, -170))
    assert p.x == pytest.approx(5) and p.angle == pytest.approx(180)   # 170 -> -170 goes the short way


# ---------------------------------------------------------------------------------------------- items
def _white_item(w=200, h=120):
    img = np.full((h, w, 3), 255, np.uint8)
    cv2.circle(img, (w // 2, h // 2), 40, (30, 160, 220), -1)
    cv2.circle(img, (w // 2, h // 2), 12, (250, 250, 250), -1)        # a light spot INSIDE the item stays
    return img


def test_plain_background_is_removed_but_light_parts_inside_survive():
    rgba = remove_plain_background(_white_item())
    assert rgba[0, 0, 3] == 0 and rgba[60, 60 + 40, 3] > 200
    assert rgba[60, 100, 3] > 200                      # centre light spot isn't connected to the border


def test_cutout_crops_to_the_item_and_respects_existing_alpha():
    c = cutout(_white_item())
    assert c.shape[0] < 100 and c.shape[1] < 100
    rgba = np.zeros((50, 80, 4), np.uint8)
    rgba[10:20, 30:60] = (0, 0, 255, 255)
    assert cutout(rgba).shape[:2] == (14, 34)          # 10x30 plus 2px padding each side
    with pytest.raises(ValueError):
        crop_to_content(np.zeros((5, 5, 4), np.uint8))


@pytest.mark.parametrize("name,kind", [
    ("gold_necklace", "necklace"), ("Pearl-Earrings", "earrings"), ("ray ban sunglasses", "glasses"),
    ("diamond_ring", "ring"), ("steel_watch", "wrist"), ("bangle", "wrist"), ("jhumka earring", "earrings"),
    ("earring", "earrings"), ("holiday photo", None),
])
def test_guess_kind(name, kind):
    assert guess_kind(name) == kind


def test_load_uses_sidecar_json_and_folder_specs(tmp_path):
    cv2.imwrite(str(tmp_path / "thing.png"), _white_item())
    (tmp_path / "thing.json").write_text(json.dumps({"kind": "ring", "anchor": [0.5, 0.2], "scale": 1.3}))
    it = load(tmp_path / "thing.png")
    assert it.kind == "ring" and it.anchor == (0.5, 0.2) and it.scale == 1.3
    cv2.imwrite(str(tmp_path / "x_necklace.jpg"), _white_item())
    cv2.imwrite(str(tmp_path / "mystery.jpg"), _white_item())
    items = load_many([str(tmp_path)])
    assert sorted(i.kind for i in items) == ["necklace", "ring"]      # mystery.jpg skipped
    assert load_many([str(tmp_path / "mystery.jpg") + ":glasses"])[0].kind == "glasses"
    with pytest.raises(ValueError):
        load(tmp_path / "mystery.jpg")


# ---------------------------------------------------------------------------------------------- render
def test_overlay_puts_the_anchor_on_the_target_and_blends():
    rgba = np.zeros((20, 40, 4), np.uint8)
    rgba[:, :] = (0, 0, 255, 255)                      # solid red bar
    it = Item("bar", "necklace", rgba, anchor=(0.5, 0.0))
    frame = np.zeros((200, 200, 3), np.uint8)
    overlay(frame, it, Placement(100, 50, 80, 0))      # scale 2: 80x40, top centre at (100, 50)
    assert frame[60, 100, 2] == 255 and frame[60, 100, 0] == 0
    assert frame[45, 100].sum() == 0                   # nothing above the anchor
    assert frame[60, 145].sum() == 0                   # nothing past the right edge (x 60..140)
    overlay(frame, it, Placement(-500, -500, 80, 30))  # fully off-screen is a no-op, not an error
