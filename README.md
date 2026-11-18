# drape

Try on jewellery, glasses and watches from **your own photos**, live on your webcam. Everything runs on your
machine: no account, no upload, no cloud.

![drape on a test portrait: necklace and earrings, then sunglasses](docs/demo.jpg)

<sub>Test portrait: NASA astronaut Eileen Collins (public domain). The items are drape's own sample drawings.</sub>

## What it does

Give drape a photo of an item (a product shot, or a picture you took on a plain background). It cuts the
item out and pins it to the right part of you in the camera view, following you as you move.

| Item kind | Sits on | Tracked with |
|---|---|---|
| `necklace` | base of the neck | body pose + face |
| `earrings` | both ear lobes (mirrored for the left) | face mesh |
| `glasses` | nose bridge, tilting with your head | face mesh |
| `ring` | ring finger, on every visible hand | hand landmarks |
| `wrist` (watch, bracelet, bangle) | just below the wrist | hand landmarks |

Tracking uses Google's [MediaPipe](https://ai.google.dev/edge/mediapipe) landmarkers (Apache-2.0), which run
in real time on a laptop CPU. The models (~20 MB) download once into `~/.cache/drape/models`.

## Install

Needs Python 3.10+ and a webcam. Developed on macOS (Apple Silicon); Linux and Windows should work too.

```bash
git clone https://github.com/intotheabyss19/drape && cd drape
uv sync                                  # or: pip install -e .
uv run python examples/make_samples.py   # draws a few sample items into examples/items/
```

## Use

```bash
uv run drape live examples/items/                    # every item in a folder
uv run drape live ~/Downloads/kundan_necklace.jpg my_specs.png:glasses
uv run drape photo me.jpg examples/items/gold_necklace.jpg -o me_with_necklace.jpg
```

Live keys: **1-9** put on / take off an item, **n / p** next / previous item of the same kind, **+ / -**
bigger / smaller, **s** save a snapshot to `~/Pictures/drape`, **f** flip the mirror view, **h** hide the help,
**q** quit. You can wear one item per kind at a time (say a necklace, earrings and glasses together).

On macOS the first live run asks for camera access for your terminal app.

### Your own items

- **Kind**: put it in the file name (`gold_necklace.jpg`, `ray-ban sunglasses.png`, `steel watch.jpg`), add
  `:kind` after the path, or use a sidecar file.
- **Background**: a PNG with transparency is used as is. Otherwise drape removes a plain background (white or
  grey product shots work well). For busy backgrounds, install the optional remover: `uv sync --extra rembg`.
- **Fine-tuning**: a sidecar `<image name>.json` next to the image can set the kind, the attach point and the
  size:

  ```json
  {"kind": "earrings", "anchor": [0.5, 0.05], "scale": 1.2}
  ```

  `anchor` is where on the item (fractions of its width and height) the body point goes: for earrings the
  hook, for a necklace the top centre of the chain, for glasses the bridge.

## Roadmap

Contributions are very welcome, especially on these:

- [ ] **Gloves**: warp a glove image over the whole hand (piecewise warp on the hand landmarks).
- [ ] **Occlusion**: hide parts of an item behind hair, the jaw or fingers (MediaPipe segmentation).
- [ ] **Clothes, live preview**: a 2D garment warp driven by body pose.
- [ ] **Clothes, realistic still**: freeze a frame and render it with a try-on diffusion model.
  [FASHN VTON 1.5](https://github.com/fashn-AI/fashn-vton-1.5) is Apache-2.0; other popular models
  (IDM-VTON, CatVTON, OOTDiffusion) are non-commercial and would only ever be optional downloads.
- [ ] **3D items** (glTF) for rings and watches that turn with your hand.
- [ ] **Control API** (CLI + MCP server) so voice assistants can say "try the gold one on me".
- [ ] A small GUI for picking items and saving looks.

## Develop

```bash
uv sync
uv run pytest
```

The pieces are small and separate: `anchors.py` (where items sit; pure geometry, unit-tested with fake
landmarks), `items.py` (loading and cutting out photos), `render.py` (warp and blend), `tracker.py`
(MediaPipe), `session.py` (what's worn) and `cli.py`.

To add a new item kind: add its placement function to `anchors.py`, its default attach point to
`items.DEFAULT_ANCHOR`, the trackers it needs to `tracker.NEEDS`, file-name words to `items._KIND_WORDS`, and
a test.

## Licence

[Apache-2.0](LICENSE). MediaPipe and its models are Apache-2.0 too.
