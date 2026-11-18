"""Tests that need no camera and no MediaPipe models: geometry, cut-outs, drawing, item loading, 3D."""
from __future__ import annotations

import json
import math

import cv2
import numpy as np
import pytest

from drape import anchors
from drape.anchors import Placement, flat
from drape.items import Item, crop_to_content, cutout, guess_kind, load, load_many, remove_plain_background
from drape.render import Light, overlay


# ---------------------------------------------------------------------------------------------- fake landmarks
def _rot_y(deg: float) -> np.ndarray:
    """Turn the head: rotation about the image's vertical axis (positive brings the subject's left ear, on the
    image right, towards the camera and turns the right ear away)."""
    t = math.radians(deg)
    return np.array([[math.cos(t), 0, math.sin(t)], [0, 1, 0], [-math.sin(t), 0, math.cos(t)]])


def _rot_z(deg: float) -> np.ndarray:
    """Tilt the head in the image plane, CCW on screen (y points down)."""
    t = math.radians(deg)
    return np.array([[math.cos(t), math.sin(t), 0], [-math.sin(t), math.cos(t), 0], [0, 0, 1]])


def fake_face(cx=320.0, cy=240.0, w=160.0, h=200.0, tilt_deg=0.0, yaw_deg=0.0) -> np.ndarray:
    f = np.zeros((478, 3), np.float32)
    pts = {
        anchors.F_FOREHEAD: (0, -h / 2, 0), anchors.F_CHIN: (0, h / 2, 0),
        anchors.F_EYE_R_OUT: (-w * 0.35, -h * 0.1, -20), anchors.F_EYE_L_OUT: (w * 0.35, -h * 0.1, -20),
        anchors.F_BRIDGE: (0, -h * 0.1, -30), anchors.F_TRAGUS_R: (-w / 2, 0, 0), anchors.F_TRAGUS_L: (w / 2, 0, 0),
    }
    rot = _rot_z(tilt_deg) @ _rot_y(yaw_deg)
    for i, p in pts.items():
        f[i] = rot @ np.array(p, float) + (cx, cy, 0)
    oval = np.array([[cx + w / 2 * math.cos(a), cy + h / 2 * math.sin(a), 0]
                     for a in np.linspace(0, 2 * math.pi, len(anchors.FACE_OVAL), endpoint=False)])
    for k, i in enumerate(anchors.FACE_OVAL):
        if i not in pts:
            f[i] = oval[k]
    return f


def fake_pose(cx=320.0, y=400.0, span=260.0) -> np.ndarray:
    p = np.zeros((33, 3), np.float32)
    p[anchors.P_SHOULDER_L] = (cx + span / 2, y, 0)
    p[anchors.P_SHOULDER_R] = (cx - span / 2, y, 0)
    return p


def fake_hand_up(x=300.0, wrist_y=400.0, back_to_camera=True) -> tuple:
    """An upright hand (fingers up the screen), index finger on the image left. Returns (landmarks, label).
    This is how MediaPipe sees a hand it labels 'Right' showing its back (checked on a real photo); mirroring it
    across x with the same label gives a palm."""
    h = np.zeros((21, 3), np.float32)
    s = 1 if back_to_camera else -1          # mirror the hand across x to swap back/palm for the same label
    h[anchors.H_WRIST] = (x, wrist_y, 0)
    h[anchors.H_INDEX_MCP] = (x - s * 40, wrist_y - 90, -5)
    h[anchors.H_MIDDLE_MCP] = (x - s * 12, wrist_y - 95, -5)
    h[anchors.H_MIDDLE_PIP] = (x - s * 13, wrist_y - 140, -8)
    h[anchors.H_RING_MCP] = (x + s * 14, wrist_y - 92, -5)
    h[anchors.H_RING_PIP] = (x + s * 16, wrist_y - 140, -8)
    h[anchors.H_PINKY_MCP] = (x + s * 38, wrist_y - 82, -5)
    h[anchors.H_PINKY_PIP] = (x + s * 42, wrist_y - 120, -8)
    return h, "Right"


# ---------------------------------------------------------------------------------------------- face items
def test_glasses_sit_on_the_bridge_and_follow_head_tilt():
    p = anchors.glasses(fake_face())
    assert (p.x, p.y) == pytest.approx((320, 220), abs=0.5)
    assert p.width == pytest.approx(1.1 * 160, rel=1e-3)
    assert p.angle == pytest.approx(0, abs=1e-3)
    assert anchors.glasses(fake_face(tilt_deg=15)).angle == pytest.approx(15, abs=0.5)


def test_glasses_foreshorten_when_the_head_turns():
    def screen_width(yaw):
        q = anchors.glasses(fake_face(yaw_deg=yaw)).quad(200, 60, 100, 20)
        return float(q[1, 0] - q[0, 0])
    assert screen_width(0) == pytest.approx(1.1 * 160, rel=1e-3)
    assert screen_width(40) == pytest.approx(screen_width(0) * math.cos(math.radians(40)), rel=0.02)


def test_earrings_hang_down_even_when_the_head_tilts():
    r, l = anchors.earrings(fake_face(tilt_deg=20))
    for p in (r, l):
        assert p.ey == pytest.approx([0, 1, 0])
    assert not r.mirror and l.mirror
    r, l = anchors.earrings(fake_face())
    assert r.x < 320 - 80 and l.x > 320 + 80 and r.y > 240 and l.y > 240


def test_the_earring_on_the_ear_turning_away_fades_out():
    r, l = anchors.earrings(fake_face(yaw_deg=10))
    assert r.opacity == 1 and l.opacity == 1
    r, l = anchors.earrings(fake_face(yaw_deg=40))          # right ear turned away
    assert r.opacity == 0 and l.opacity == 1
    r, l = anchors.earrings(fake_face(yaw_deg=-40))
    assert r.opacity == 1 and l.opacity == 0


def test_necklace_sits_between_chin_and_shoulders_and_can_be_covered():
    p = anchors.necklace(fake_pose(), fake_face())
    assert 340 < p.y < 400 and p.x == pytest.approx(320)
    assert p.width == pytest.approx(0.46 * 260) and abs(p.angle) < 1e-6
    assert set(p.occlude) == {"hair", "face"}
    assert anchors.necklace(fake_pose()).y < 400


# ---------------------------------------------------------------------------------------------- hand items
def test_dorsal_points_out_of_the_back_of_the_hand():
    back, label = fake_hand_up(back_to_camera=True)
    palm, _ = fake_hand_up(back_to_camera=False)
    assert anchors.dorsal(back, label)[2] < -0.5        # towards the camera (smaller z)
    assert anchors.dorsal(palm, label)[2] > 0.5
    assert anchors.dorsal(back, "Left")[2] > 0.5        # the label flips the sign


def test_ring_pose_runs_along_the_finger_with_the_stone_on_the_back():
    h, label = fake_hand_up()
    p = anchors.ring(h, label)
    R = p.pose3d.R
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-6) and np.linalg.det(R) == pytest.approx(1)
    assert R[1, 2] < -0.5                               # finger axis points up the screen
    assert R[2, 1] < -0.5                               # stone towards the camera
    assert 400 - 140 < p.y < 400 - 92
    assert len(p.pose3d.others) == 2                    # middle and little finger can cover it


def test_watch_band_is_wider_than_thick():
    p = anchors.wrist(*fake_hand_up())
    sx, sy, _ = p.pose3d.scale
    assert sx > sy and p.y > 400


def test_place_needs_the_right_landmarks():
    assert anchors.place("glasses") == []
    assert anchors.place("necklace", face=fake_face()) == []
    assert len(anchors.place("earrings", face=fake_face())) == 2
    h, lb = fake_hand_up()
    assert len(anchors.place("ring", hands=[h, h], hand_labels=[lb, lb])) == 2


def test_smoother_moves_part_way_and_keeps_axes_orthonormal():
    s = anchors.Smoother(alpha=0.5)
    s("k", flat(0, 0, 100, 0))
    p = s("k", flat(10, 0, 100, 90))
    assert p.x == pytest.approx(5)
    assert np.linalg.norm(p.ex) == pytest.approx(1) and abs(np.dot(p.ex, p.ey)) < 1e-6
    assert 30 < p.angle < 60


# ---------------------------------------------------------------------------------------------- items
def _white_item(w=200, h=120):
    img = np.full((h, w, 3), 255, np.uint8)
    cv2.circle(img, (w // 2, h // 2), 40, (30, 160, 220), -1)
    cv2.circle(img, (w // 2, h // 2), 12, (250, 250, 250), -1)        # a light spot INSIDE the item stays
    return img


def test_plain_background_is_removed_but_light_parts_inside_survive():
    rgba = remove_plain_background(_white_item())
    assert rgba[0, 0, 3] == 0 and rgba[60, 60 + 40, 3] > 200
    assert rgba[60, 100, 3] > 200


def test_cutout_crops_to_the_item_and_respects_existing_alpha():
    c = cutout(_white_item())
    assert c.shape[0] < 100 and c.shape[1] < 100
    rgba = np.zeros((50, 80, 4), np.uint8)
    rgba[10:20, 30:60] = (0, 0, 255, 255)
    assert cutout(rgba).shape[:2] == (14, 34)
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
    assert sorted(i.kind for i in items) == ["necklace", "ring"]
    assert load_many([str(tmp_path / "mystery.jpg") + ":glasses"])[0].kind == "glasses"
    with pytest.raises(ValueError):
        load(tmp_path / "mystery.jpg")


# ---------------------------------------------------------------------------------------------- render
def _bar_item():
    rgba = np.zeros((20, 40, 4), np.uint8)
    rgba[:, :] = (0, 0, 255, 255)
    return Item("bar", "necklace", rgba, anchor=(0.5, 0.0))


def test_overlay_puts_the_anchor_on_the_target_and_blends():
    frame = np.zeros((200, 200, 3), np.uint8)
    overlay(frame, _bar_item(), flat(100, 50, 80, 0), shadow=False)    # 80x40, top centre at (100, 50)
    assert frame[60, 100, 2] == 255 and frame[60, 100, 0] == 0
    assert frame[45, 100].sum() == 0 and frame[60, 145].sum() == 0
    overlay(frame, _bar_item(), flat(-500, -500, 80, 30))              # off-screen: no-op, not an error


def test_masks_hide_the_item_behind_hair():
    frame = np.zeros((200, 200, 3), np.uint8)
    hair = np.zeros((200, 200), np.float32)
    hair[:, 100:] = 1.0
    p = replace_occlude(flat(100, 50, 80, 0), ("hair",))
    overlay(frame, _bar_item(), p, masks={"hair": hair}, shadow=False)
    assert frame[60, 80, 2] == 255 and frame[60, 120].sum() == 0


def replace_occlude(p, occ):
    from dataclasses import replace
    return replace(p, occlude=occ)


def test_shadow_darkens_just_below_the_item():
    frame = np.full((200, 200, 3), 200, np.uint8)
    overlay(frame, _bar_item(), flat(100, 50, 80, 0), shadow=True)
    assert frame[93, 100].mean() < 200                  # under the bottom edge (y 50..90)
    assert frame[30, 100].mean() == 200                 # nothing above


def test_light_follows_the_scene():
    dark, bright = np.full((90, 160, 3), 30, np.uint8), np.full((90, 160, 3), 200, np.uint8)
    warm = np.dstack([np.full((90, 160), 90), np.full((90, 160), 130), np.full((90, 160), 190)]).astype(np.uint8)
    assert Light.measure(dark)[0] < Light.measure(bright)[0]
    t = Light.measure(warm)[1]
    assert t[2] > 1 > t[0]                              # BGR: red up, blue down


# ---------------------------------------------------------------------------------------------- 3D
def test_canonicalize_recovers_any_axis_aligned_orientation():
    trimesh = pytest.importorskip("trimesh")
    from drape.mesh3d import from_trimesh, sample_ring

    rng = np.random.default_rng(1)
    for _ in range(6):
        m = sample_ring()
        R = np.eye(3)[rng.permutation(3)] * rng.choice([-1, 1], 3)[:, None]
        if np.linalg.det(R) < 0:
            R[0] *= -1
        T = np.eye(4)
        T[:3, :3] = R
        m.apply_transform(T)
        m.apply_scale(rng.uniform(0.5, 20))
        m.apply_translation(rng.normal(size=3) * 10)
        v = from_trimesh(m).vertices
        assert v.min(0) == pytest.approx([-1.16, -1.16, -0.24], abs=0.02)
        assert v.max(0) == pytest.approx([1.16, 1.57, 0.24], abs=0.02)


@pytest.fixture(scope="module")
def renderer():
    pytest.importorskip("moderngl")
    from drape.mesh3d import MeshRenderer
    try:
        r = MeshRenderer()
    except Exception as e:                               # no OpenGL (e.g. a headless CI box)
        pytest.skip(f"no OpenGL: {e}")
    yield r
    r.close()


def _ring_frame(back_to_camera: bool, renderer):
    import trimesh  # noqa: F401
    from drape.mesh3d import from_trimesh, sample_ring

    h, label = fake_hand_up(back_to_camera=back_to_camera)
    p = anchors.ring(h, label)
    out = renderer.render((480, 640, 3), from_trimesh(sample_ring(), [0, 0, 1], [0, 1, 0], 1.0), p.pose3d)
    assert out is not None
    patch, x0, y0 = out
    return patch, x0, y0, p


def test_3d_ring_shows_its_stone_on_the_back_of_the_hand_only(renderer):
    def stone_pixels(back):
        patch, x0, y0, p = _ring_frame(back, renderer)
        b, g, r, a = [patch[:, :, i].astype(int) for i in range(4)]
        whiteish = (a > 200) & (b > 150) & (np.abs(r - b) < 60)     # the pale stone, not the gold band
        return int(whiteish.sum())
    assert stone_pixels(True) > 10
    assert stone_pixels(False) < stone_pixels(True) / 4


def test_3d_ring_band_is_hidden_behind_the_finger(renderer):
    patch, x0, y0, p = _ring_frame(True, renderer)
    alpha = patch[:, :, 3] > 128
    # the band's front arc is visible but the ring isn't drawn as a full closed loop over the finger
    assert alpha.sum() > 30
    cy, cx = int(p.y) - y0, int(p.x) - x0
    assert alpha[cy, cx]                                  # the front of the band crosses the finger centre
