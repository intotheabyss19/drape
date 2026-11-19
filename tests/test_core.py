"""Tests that need no camera and no MediaPipe models: geometry, cut-outs, drawing, item loading, 3D."""
from __future__ import annotations

import json
import math

import cv2
import numpy as np
import pytest

from drape import anchors
from drape.anchors import flat
from drape.items import Item, crop_to_content, cutout, guess_kind, load, load_many, remove_plain_background
from drape.render import Light, composite_premultiplied, overlay


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
        anchors.F_EYE_R_IN: (-w * 0.12, -h * 0.1, -22), anchors.F_EYE_L_IN: (w * 0.12, -h * 0.1, -22),
        anchors.F_JAW_R: (-w * 0.47, h * 0.2, 0), anchors.F_JAW_L: (w * 0.47, h * 0.2, 0),
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
def test_glasses_sit_in_front_of_the_eyes_and_follow_head_tilt():
    p = anchors.glasses(fake_face())
    assert (p.x, p.y) == pytest.approx((320, 220), abs=3)
    assert p.origin[2] < -30                            # in front of the nose bridge
    assert p.width == pytest.approx((anchors.HEAD_W + anchors.GLASSES_GAP) / anchors.HEAD_W * 160, rel=1e-3)
    assert p.angle == pytest.approx(0, abs=1e-3)
    assert anchors.glasses(fake_face(tilt_deg=15)).angle == pytest.approx(15, abs=0.5)


def test_glasses_foreshorten_when_the_head_turns():
    def screen_width(yaw):
        q = anchors.glasses(fake_face(yaw_deg=yaw)).quad(200, 60, 100, 20)
        return float(q[1, 0] - q[0, 0])
    assert screen_width(0) == pytest.approx(anchors.glasses(fake_face()).width, rel=1e-3)
    assert screen_width(40) == pytest.approx(screen_width(0) * math.cos(math.radians(40)), rel=0.02)


def test_earrings_hang_down_even_when_the_head_tilts():
    r, l = anchors.earrings(fake_face(tilt_deg=20))
    for p in (r, l):
        assert p.ey == pytest.approx([0, 1, 0])
    assert not r.mirror and l.mirror
    r, l = anchors.earrings(fake_face())
    assert r.x < 320 - 70 and l.x > 320 + 70 and r.y > 240 and l.y > 240


def test_the_earring_on_the_ear_turning_away_fades_out():
    r, l = anchors.earrings(fake_face(yaw_deg=10))
    assert r.opacity == 1 and l.opacity == 1
    r, l = anchors.earrings(fake_face(yaw_deg=40))          # right ear turned away
    assert r.opacity == 0 and l.opacity == 1
    r, l = anchors.earrings(fake_face(yaw_deg=-40))
    assert r.opacity == 1 and l.opacity == 0


def test_necklace_goes_round_the_base_of_the_neck():
    p = anchors.necklace(fake_pose(), fake_face())
    ppm = 160 / anchors.HEAD_W
    neck = p.pose3d.origin
    assert neck[0] == pytest.approx(320) and neck[1] == pytest.approx(240 + 0.155 * ppm, abs=1)
    assert p.pose3d.scale[0] == pytest.approx(ppm)
    assert set(p.occlude) == {"hair", "face"}
    assert p.pose3d.occluders[0][0] == "cylinder"
    only_body = anchors.necklace(fake_pose(), None).pose3d
    assert only_body.origin[1] < 400 and only_body.scale[0] == pytest.approx(260 / anchors.SHOULDER_W)
    assert anchors.necklace(None, None) is None


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
    assert len(p.pose3d.occluders) == 3                 # the finger, and the middle and little fingers


def test_watch_band_is_wider_than_thick():
    p = anchors.wrist(*fake_hand_up())
    sx, sy, _ = p.pose3d.scale
    assert sx > sy and p.y > 400


def test_place_needs_the_right_landmarks():
    assert anchors.place("glasses") == []
    assert anchors.place("necklace") == []
    assert len(anchors.place("necklace", face=fake_face())) == 1          # the face alone is enough
    assert len(anchors.place("earrings", face=fake_face())) == 2
    h, lb = fake_hand_up()
    assert len(anchors.place("ring", hands=[h, h], hand_labels=[lb, lb])) == 2


def test_one_euro_steadies_jitter_but_follows_real_movement():
    rng = np.random.default_rng(0)
    f = anchors.OneEuro()
    still = [f(np.array([[100.0, 100, 0]]) + rng.normal(0, 2, (1, 3)), k / 30)[0, 0] for k in range(60)]
    assert np.std(still[20:]) < 1.0                     # jitter of 2 px mostly removed
    f = anchors.OneEuro()
    for k in range(30):
        out = f(np.array([[100.0 + 20 * k, 100, 0]]), k / 30)  # moving 600 px/s
    assert out[0, 0] > 100 + 20 * 29 - 40               # lags by less than 2 frames
    f.reset()
    assert f(np.array([[5.0, 5, 5]]), 10.0)[0, 0] == 5.0


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


def test_premultiplied_composite_and_mask():
    frame = np.full((10, 10, 3), 100, np.uint8)
    patch = np.zeros((4, 4, 4), np.uint8)
    patch[:, :2] = (0, 0, 128, 128)                     # half-transparent red, premultiplied
    mask = np.zeros((10, 10), np.float32)
    mask[:, 3:] = 1.0                                   # something in front of the right half
    composite_premultiplied(frame, patch, 2, 2, mask)
    assert frame[3, 2].tolist() == pytest.approx([50, 50, 178], abs=1)   # 128 + 100 * (1 - 0.5)
    assert tuple(frame[3, 3]) == (100, 100, 100)        # hidden by the mask
    assert tuple(frame[3, 5]) == (100, 100, 100)


# ---------------------------------------------------------------------------------------------- 3D
def test_canonicalize_recovers_any_axis_aligned_orientation():
    from drape.jewelry import solitaire_ring
    from drape.mesh3d import canonicalize, vertex_weights

    model = solitaire_ring()
    ref, wts = model.vertices(), vertex_weights(model)
    lo, hi = ref.min(0), ref.max(0)
    rng = np.random.default_rng(1)
    for _ in range(6):
        R = np.eye(3)[rng.permutation(3)] * rng.choice([-1, 1], 3)[:, None]
        if np.linalg.det(R) < 0:
            R[0] *= -1
        v = ref @ R.T * rng.uniform(0.5, 20) + rng.normal(size=3) * 10
        Rc, c = canonicalize(v, weights=wts)
        w = (v - c) @ Rc.T
        assert w.min(0) == pytest.approx(lo, abs=0.03) and w.max(0) == pytest.approx(hi, abs=0.03)


def test_jewellery_is_built_to_real_dimensions():
    from drape import jewelry

    hoop = jewelry.hoop_earring(22).vertices()
    assert np.ptp(hoop[:, 0]) == pytest.approx(0.024, abs=0.001)         # 22 mm + the 2 mm wire
    assert hoop[:, 1].min() == pytest.approx(-0.001, abs=0.0005)          # hangs from the piercing
    for L in (0.45, 0.50, 0.60):
        p = jewelry.necklace_path(L)
        assert np.linalg.norm(np.diff(np.vstack([p, p[:1]]), axis=0), axis=1).sum() == pytest.approx(L, rel=0.01)
    assert jewelry.necklace_path(0.60)[:, 1].max() > jewelry.necklace_path(0.45)[:, 1].max()   # longer hangs lower
    stone = jewelry.brilliant(0.006)
    assert np.ptp(stone.v[:, 0]) == pytest.approx(0.006, rel=0.02)
    assert np.ptp(stone.v[:, 1]) == pytest.approx(0.006 * 0.61, rel=0.03)  # crown + girdle + pavilion
    ring = jewelry.solitaire_ring().vertices()
    r = np.hypot(ring[:, 0], ring[:, 1])
    assert r.min() == pytest.approx(1.0, abs=0.01) and ring[:, 1].max() > 1.5   # stone on the back (+Y)


def _write_glasses_gltf(path, transmission=True):
    """A crude pair of 'glasses' in glTF convention (+Y up, +Z towards the viewer), with a glass lens."""
    trimesh = pytest.importorskip("trimesh")
    front = trimesh.creation.box([0.14, 0.04, 0.004])
    lens = trimesh.creation.box([0.05, 0.03, 0.002])
    lens.apply_translation([0.03, 0.0, 0.004])
    arm = trimesh.creation.box([0.004, 0.004, 0.13])
    arm.apply_translation([0.068, 0.01, -0.065])
    scene = trimesh.Scene()
    for name, m in (("front", front), ("lens", lens), ("arm", arm)):
        m.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
            name=name, baseColorFactor=[200, 200, 200, 255], metallicFactor=0.0))
        scene.add_geometry(m, geom_name=name)
    scene.export(str(path))
    if transmission:                                    # add the extension trimesh doesn't write
        import struct
        raw = bytearray(path.read_bytes())
        n = struct.unpack("<I", raw[12:16])[0]
        doc = json.loads(raw[20:20 + n])
        for m in doc["materials"]:
            if m["name"] == "lens":
                m["extensions"] = {"KHR_materials_transmission": {"transmissionFactor": 1.0}}
        js = json.dumps(doc).encode()
        js += b" " * (-len(js) % 4)
        rest = raw[20 + n:]
        out = raw[:12] + struct.pack("<I", len(js)) + raw[16:20] + js + rest
        out[8:12] = struct.pack("<I", len(out))
        path.write_bytes(bytes(out))


def test_glasses_model_is_put_in_the_metric_frame(tmp_path):
    from drape.mesh3d import load_model

    f = tmp_path / "specs.glb"
    _write_glasses_gltf(f)
    m = load_model(f, "glasses")
    v = m.vertices()
    assert m.frame == "metric" and m.info["width"] == pytest.approx(0.14, abs=0.002)
    assert v[:, 2].min() == pytest.approx(0.0, abs=1e-6)       # attach point at the front...
    assert v[:, 2].max() > 0.12                                # ...and the arms go back, away from the viewer
    lens = [p for p in m.prims if p.mat.name == "lens"][0]
    assert lens.mat.blend and lens.mat.base[3] < 0.3           # glass is see-through


def test_load_any_kind_of_3d_item(tmp_path):
    from drape.items import load

    f = tmp_path / "specs.glb"
    _write_glasses_gltf(f, transmission=False)
    assert load(f, "glasses").is_3d
    (tmp_path / "specs.json").write_text(json.dumps({"kind": "glasses", "credit": "me"}))
    it = load(f)
    assert it.kind == "glasses" and it.credit == "me"


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


def _ring_patch(back_to_camera: bool, renderer):
    from drape.jewelry import solitaire_ring

    h, label = fake_hand_up(back_to_camera=back_to_camera)
    p = anchors.ring(h, label)
    out = renderer.render((480, 640, 3), solitaire_ring(), p.pose3d)
    assert out is not None
    return out[0], out[1], out[2], p


def test_3d_ring_shows_its_stone_on_the_back_of_the_hand_only(renderer):
    def stone_pixels(back):
        patch, x0, y0, p = _ring_patch(back, renderer)
        b, g, r, a = [patch[:, :, i].astype(int) for i in range(4)]
        neutral = (a > 200) & (np.abs(r - b) < 40)                # the colourless stone, not the gold
        return int(neutral.sum())
    assert stone_pixels(True) > 10
    assert stone_pixels(False) < stone_pixels(True) / 4


def test_3d_ring_band_crosses_the_front_of_the_finger(renderer):
    patch, x0, y0, p = _ring_patch(True, renderer)
    alpha = patch[:, :, 3] > 128
    assert alpha.sum() > 30
    assert alpha[int(p.y) - y0, int(p.x) - x0]


def test_the_head_hides_the_far_earring(renderer):
    from drape.jewelry import hoop_earring

    hoop = hoop_earring()

    def visible(yaw):
        r, l = anchors.earrings(fake_face(w=300, h=380, yaw_deg=yaw))
        out = []
        for p in (r, l):
            res = renderer.render((480, 640, 3), hoop, p.pose3d)
            out.append(0 if res is None else int((res[0][:, :, 3] > 128).sum()))
        return out
    r0, l0 = visible(0)
    assert r0 > 20 and l0 > 20
    r1, l1 = visible(50)                                 # right ear turned away
    assert r1 < r0 / 3 and l1 > 20


def test_camera_thread_hands_out_new_frames_then_stops():
    from drape.cli import _camera

    class FakeCap:
        def __init__(self):
            self.n = 0

        def read(self):
            self.n += 1
            if self.n > 5:
                return False, None
            return True, np.full((2, 2, 3), self.n, np.uint8)

    cam = _camera(FakeCap())
    got = []
    while (f := cam.read()) is not None:
        got.append(int(f[0, 0, 0]))
    cam.close()
    assert got and got == sorted(set(got)) and got[-1] == 5     # never the same frame twice, ends with the last


# ---------------------------------------------------------------------------------------------- helmet and gloves
def test_helmet_model_is_head_sized_and_centred():
    from drape.armour import helmet

    m = helmet()
    v = m.vertices()
    assert m.frame == "metric"
    width = float(np.ptp(v[:, 0]))
    assert 0.18 < width < 0.25                                    # a little wider than a head
    assert v[:, 1].min() < -0.13 and v[:, 1].max() > 0.12         # crown above, chin below the centre
    assert abs(float(v[:, 0].mean())) < 0.005                     # symmetric
    assert any(p.mat.name == "visor" for p in m.prims)
    visor = next(p for p in m.prims if p.mat.name == "visor")
    assert visor.v[:, 2].max() < 0                                # the visor is on the front


def test_helmet_follows_the_head_and_hides_its_inside():
    face = fake_face(yaw_deg=25)
    p = anchors.helmet(face)
    R, ppm = anchors.head(face)
    assert np.allclose(p.pose3d.R, R) and np.allclose(p.pose3d.scale, ppm)
    assert [s for s, _ in p.pose3d.occluders] == ["sphere", "cylinder"]   # skull and neck
    assert "hands" in p.occlude
    assert anchors.place("helmet", face=face) and not anchors.place("helmet")


def test_gauntlet_is_one_mesh_that_moves_with_the_hand():
    from drape.armour import Gauntlet

    g = Gauntlet()
    h, label = fake_hand_up()
    m1 = g.posed(h, label, slot=0)
    v1 = m1.vertices().copy()
    versions = [p.version for p in m1.prims]
    m2 = g.posed(h + np.array([50.0, 0, 0]), label, slot=0)
    assert [p.version for p in m2.prims] == [v + 1 for v in versions]   # same buffers, re-uploaded
    assert np.allclose(m2.vertices(), v1 + np.array([50.0, 0, 0]), atol=1e-3)
    lo, hi = v1.min(axis=0), v1.max(axis=0)                       # covers every landmark of the hand
    assert np.all(h[:, :2] >= lo[:2]) and np.all(h[:, :2] <= hi[:2])
    assert m1.prims is not g.posed(h, label, slot=1).prims        # a second hand gets its own buffers


def test_gauntlet_keeps_its_size_on_a_side_on_hand():
    from drape.armour import Gauntlet

    g = Gauntlet()
    h, label = fake_hand_up()
    side = h.copy()
    side[:, 0] = h[0, 0] + (h[:, 0] - h[0, 0]) * 0.05             # seen edge-on: the knuckles line up
    w_front = np.ptp(g.posed(h, label).vertices()[:, 1])
    w_side = np.ptp(g.posed(side, label).vertices()[:, 1])
    assert w_side > 0.7 * w_front


def test_gloves_and_helmets_kinds_from_names_and_files(tmp_path):
    from drape.items import guess_kind, load

    assert guess_kind("space helmet") == "helmet" and guess_kind("party mask") == "helmet"
    assert guess_kind("leather gloves") == "gloves" and guess_kind("steel gauntlet") == "gloves"
    f = tmp_path / "my_gloves.png"
    f.write_bytes(b"x")
    with pytest.raises(ValueError, match="built-in gauntlets"):
        load(f)


def test_3d_gauntlet_renders_over_the_hand(renderer):
    from drape.armour import Gauntlet

    h, label = fake_hand_up()
    p = anchors.gloves(h, label)
    out = renderer.render((480, 640, 3), Gauntlet().posed(h, label), p.pose3d)
    assert out is not None
    patch, x0, y0 = out
    covered = 0
    for x, y in h[:, :2].astype(int):
        if 0 <= y - y0 < patch.shape[0] and 0 <= x - x0 < patch.shape[1]:
            covered += patch[y - y0, x - x0, 3] > 200
    assert covered >= 19                                          # nearly every joint is under armour
