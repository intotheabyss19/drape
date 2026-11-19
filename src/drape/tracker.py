"""Body tracking with MediaPipe (face, pose and hand landmarkers), running locally on CPU/GPU.

Models are downloaded once from Google's MediaPipe model storage into ~/.cache/drape/models.
Only the trackers an item needs are run each frame.
"""
from __future__ import annotations

import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set

import cv2
import numpy as np

os.environ.setdefault("GLOG_minloglevel", "2")       # quiet MediaPipe / TFLite start-up chatter
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

MODELS = {
    "face": "face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    "pose": "pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "hand": "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "seg": "image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite",
}
SEG_HAIR, SEG_BODY_SKIN, SEG_FACE_SKIN, SEG_CLOTHES = 1, 2, 3, 4
BASE = "https://storage.googleapis.com/mediapipe-models/"
NEEDS = {"glasses": {"face"}, "earrings": {"face", "seg"}, "necklace": {"pose", "face", "seg"},
         "ring": {"hand"}, "wrist": {"hand"}, "helmet": {"face", "hand"}, "gloves": {"hand"}}


def model_dir() -> Path:
    return Path(os.environ.get("DRAPE_MODELS") or Path.home() / ".cache" / "drape" / "models")


def model_path(name: str) -> Path:
    p = model_dir() / Path(MODELS[name]).name
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading MediaPipe {name} model (one time)…")
        tmp = p.with_suffix(".part")
        urllib.request.urlretrieve(BASE + MODELS[name], tmp)
        tmp.replace(p)
    return p


SHOULDERS = (11, 12)


@dataclass
class Landmarks:
    face: Optional[np.ndarray] = None          # (478, 3) pixels; z (relative depth) in the same pixel scale
    pose: Optional[np.ndarray] = None          # (33, 3)
    hands: List[np.ndarray] = field(default_factory=list)   # each (21, 3)
    hand_labels: List[str] = field(default_factory=list)    # MediaPipe handedness per hand ("Left"/"Right")
    seg: Optional[np.ndarray] = None           # (H, W) uint8 classes: 0 background, 1 hair, 2 body skin, ...


def _px(lms, w: int, h: int) -> np.ndarray:
    return np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in lms], dtype=np.float32)


class Tracker:
    """`video=True` for a live camera (uses temporal tracking), False for single photos."""

    def __init__(self, parts: Set[str], video: bool = True, max_hands: int = 2) -> None:
        from mediapipe.tasks.python import BaseOptions, vision

        mode = vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE
        self.video = video
        self._t0 = time.monotonic()
        self._last_ts = -1
        self.face = self.pose = self.hand = self.seg = None
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="drape-track")
        if "face" in parts:
            self.face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path("face"))), running_mode=mode, num_faces=1))
        if "pose" in parts:
            self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path("pose"))), running_mode=mode, num_poses=1))
        if "hand" in parts:
            self.hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path("hand"))), running_mode=mode,
                num_hands=max_hands))
        if "seg" in parts:
            self.seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
                base_options=BaseOptions(model_asset_path=str(model_path("seg"))), running_mode=mode,
                output_category_mask=True, output_confidence_masks=False))

    def __call__(self, bgr: np.ndarray, need: Optional[Set[str]] = None) -> Landmarks:
        """Run the trackers in `need` (default: all that were loaded)."""
        import mediapipe as mp

        h, w = bgr.shape[:2]
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        ts = int((time.monotonic() - self._t0) * 1000)
        ts = max(ts, self._last_ts + 1)
        self._last_ts = ts

        def run(det):
            return det.detect_for_video(img, ts) if self.video else det.detect(img)

        jobs = {}                                    # the models are independent: run them at the same time
        for name, det in (("face", self.face), ("pose", self.pose), ("hand", self.hand)):
            if det is not None and (need is None or name in need):
                jobs[name] = self._pool.submit(run, det)
        if self.seg is not None and (need is None or "seg" in need):
            jobs["seg"] = self._pool.submit(
                lambda: self.seg.segment_for_video(img, ts) if self.video else self.seg.segment(img))
        res = {k: f.result() for k, f in jobs.items()}

        out = Landmarks()
        r = res.get("face")
        if r is not None and r.face_landmarks:
            out.face = _px(r.face_landmarks[0], w, h)
        r = res.get("pose")
        if r is not None and r.pose_landmarks:
            lms = r.pose_landmarks[0]
            if all((lms[i].visibility or 0) > 0.5 and -0.05 < lms[i].x < 1.05 and lms[i].y < 1.05
                   for i in SHOULDERS):              # else the model is guessing (it keeps "tracking" a body that
                out.pose = _px(lms, w, h)            # has left the frame, and the necklace lands on your head)
        r = res.get("hand")
        if r is not None:
            out.hands = [_px(lms, w, h) for lms in (r.hand_landmarks or [])]
            out.hand_labels = [(c[0].category_name if c else "") for c in (r.handedness or [])]
        r = res.get("seg")
        if r is not None and r.category_mask is not None:
            m = np.squeeze(r.category_mask.numpy_view())
            out.seg = m if m.shape[:2] == (h, w) else cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        return out

    def close(self) -> None:
        self._pool.shutdown(wait=True)
        for d in (self.face, self.pose, self.hand, self.seg):
            if d is not None:
                d.close()


def parts_for(kinds) -> Set[str]:
    s: Set[str] = set()
    for k in kinds:
        s |= NEEDS.get(k, set())
    return s
