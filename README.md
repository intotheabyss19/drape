# drape

Try on jewellery, glasses, watches, helmets and gauntlets live on your webcam: realistic 3D models, or photos
of your own items.
Everything runs on your machine: no account, no upload, no cloud.

![drape: pearl drops and a gold chain; aviators, gold hoops and pearls; a diamond ring and chronograph on the back of a hand, and from the palm side](docs/demo.jpg)

<sub>Portrait: NASA astronaut Eileen Collins (public domain). Hands: MediaPipe's test images (Apache-2.0).
Sunglasses and watch: Eric Chadwick / Darmstadt Graphics Group, Khronos glTF Sample Assets (CC-BY 4.0). The ring,
earrings and necklaces are built by drape. Note the far side of the ring band hidden by the finger, the stone and
dial only on the back of the hand, and the earrings hanging straight down.</sub>

## What it does

`drape live` opens your camera and puts the items on you: glasses on your nose in 3D, earrings hanging from
your ear lobes, a necklace round your neck, a ring on your ring finger and a watch on your wrist. The armour
(press 6 and 7) is a full helmet that turns with your head and gauntlets that follow every finger. They follow
you as you move and turn, at real-world sizes.

Built-in items (more welcome):

| Item | Kind | Source |
|---|---|---|
| Sunglasses (aviators) | glasses | Khronos glTF Sample Assets, CC-BY 4.0, downloaded on first run |
| Round spectacles | glasses | Poly Haven, CC0, downloaded on first run |
| Pearl drop earrings, gold hoops | earrings | built by drape from real dimensions (8 mm pearl, 22 mm hoop) |
| Gold chain with a diamond pendant, pearl strand | necklace | built by drape (46 cm cable chain, 45 cm strand of 7 mm pearls) |
| Diamond solitaire | ring | built by drape (2.2 mm band, 0.5 ct round brilliant, four prongs) |
| Chronograph watch | wrist | Khronos glTF Sample Assets, CC-BY 4.0, downloaded on first run |
| Armour helmet | helmet | built by drape: painted shell, chevron visor, gunmetal jaw, ear pods |
| Armour gauntlets | gloves | built by drape: a plate on every finger bone, so they bend with your fingers |

Your own items can be **3D models** (`.glb`, `.gltf`, `.obj`, `.stl`, `.ply`; most realistic) or **photos**
(a product shot on a plain background; quick, but flat).

Runs in real time: the camera's full 30 frames a second with glasses, earrings, a necklace, and a ring and
a watch on each hand, at 1280x720 on an Apple M5 Pro (tracking ~18 ms, drawing ~16 ms, overlapped with reading
the camera and showing the previous frame).

## How it works

- **Tracking (learned):** Google's [MediaPipe](https://ai.google.dev/edge/mediapipe) models find 478 points on
  your face, 33 on your body and 21 on each hand, with depth, plus which pixels are hair, skin, clothes or
  accessories. Only the trackers the worn items need run, in parallel. The models (~30 MB, Apache-2.0)
  download once into `~/.cache/drape/models`.
- **Steadiness:** the tracked points are smoothed with a One Euro filter: steady when you're still, no lag when
  you move.
- **Placement (geometry, real units):** the width of your face gives pixels per metre, so items come out at
  their real size. Each kind has a rule for where it sits (glasses in front of your eyes, earrings at the lobes,
  a necklace round the base of the neck, a ring between the knuckle and the next joint), turned with your head
  or hand in 3D. The rules' proportions are average adult measurements, not learned; measured improvements are
  very welcome.
- **Things in front:** before drawing an item, invisible stand-ins for your head (an ellipsoid), neck and
  fingers (cylinders) are drawn into the depth buffer, so the arms of the glasses go behind your head, the far
  earring disappears when you turn, and the back of a ring band is hidden by the finger. Hair and headphones
  (from the segmentation) cover an earring when they cover the lobe, and hair and your chin can cover a
  necklace.
- **Materials and light:** physically based shading with image-based lighting from a studio environment
  (Poly Haven, CC0): metals reflect it, the diamond splits it into colours, pearls get a soft lustre, glass
  lenses are see-through. glTF base colour and metal/roughness textures are supported. The result is matched to
  your camera's brightness and colour.

## Install

Needs Python 3.10+ and a webcam. Developed on macOS (Apple Silicon); Linux and Windows should work too.

```bash
git clone https://github.com/intotheabyss19/drape && cd drape
uv sync                                  # or: pip install -e .
```

## Use

```bash
uv run drape live                                    # the built-in items
uv run drape live ~/models/ my_specs.glb:glasses kundan_necklace.jpg samples
uv run drape photo me.jpg -o me_dressed.jpg          # try the built-in items on a photo
uv run drape samples                                 # list the built-in items and their credits
uv run drape live --record ~/drape-rec --seconds 60  # save a camera frame and the result each second
```

If something looks wrong, `--record` saves what the camera saw next to what drape drew, with a log of timings
and what was tracked: attach a few of those frames to an issue.

Live keys: **1-9** and **0** put on / take off an item, **n / p** next / previous item of the same kind, **+ / -**
bigger / smaller, **s** save a snapshot to `~/Pictures/drape`, **f** flip the mirror view, **h** hide the help,
**q** quit. You can wear one item per kind at a time (say a necklace, earrings and glasses together); glasses and earrings
go away while you wear a helmet.
The panel shows the frame rate and which items you're wearing. `--simple` turns off occlusion, scene lighting
and shadows.

On macOS the first live run asks for camera access for your terminal app.

### Your own items

- **Kind**: put it in the file name (`gold_necklace.glb`, `ray-ban sunglasses.png`, `steel watch.jpg`), add
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

### 3D models

`.glb`, `.gltf`, `.obj`, `.stl` or `.ply`, with glTF materials (base colour, metal/roughness, textures,
see-through glass) or vertex colours.

- **Rings, watches, bracelets**: drape works out how the model sits on its own: the band's axis is its
  thinnest direction, the stone or dial is the side that sticks out, and the inner radius sets the size, so it
  fits the finger or wrist whatever units the model uses. If it guesses wrong, a sidecar `.json` can say
  `{"kind": "ring", "axis": [0, 0, 1], "up": [0, 1, 0], "inner_radius": 9.0}` in the model's own units.
- **Glasses**: modelled facing +Z with +Y up (the glTF convention), in metres or millimetres. They are scaled
  to the width of your face.
- **Earrings**: +Y up, real size; they hang from their topmost point (or set `"attach": [x, y, z]`).
- **Necklaces**: +Y up, real size, modelled round a neck whose centre is the origin (or set `"attach"`).
- **Helmets and masks**: +Y up, facing +Z, any units: scaled to about 20 cm wide, and the middle of the
  model goes on the middle of the head (or set `"attach"`). A photo of a mask works too (flat, on the face).
- **Gloves**: only the built-in gauntlets for now. A glove has to bend with every finger, which needs a rigged
  (skinned) model; support for those is on the roadmap.

Every material can be overridden with `"metallic"` and `"roughness"` in the sidecar, and `"credit"` records
where the model came from.

## Roadmap

Contributions are very welcome, especially on these:

- [x] Helmets and gauntlets (rigid plates per finger bone).
- [ ] **Rigged gloves**: glTF skinned glove models driven by the hand landmarks, and glove photos warped over
  the hand.
- [x] Occlusion by head, neck, fingers, hair and headphones; 3D head pose; physically based materials and
  lighting; 3D models for every kind.
- [ ] **Measured placement**: a small labelled photo set to fit the placement proportions and report accuracy.
- [ ] **More models**: openly licensed earrings, necklaces, bracelets and frames (glTF, real size).
- [ ] **Normal maps** and real light estimation from the camera (direction, not just brightness and colour).
- [ ] **Clothes, live preview**: a 2D garment warp driven by body pose.
- [ ] **Clothes, realistic still**: freeze a frame and render it with a try-on diffusion model.
  [FASHN VTON 1.5](https://github.com/fashn-AI/fashn-vton-1.5) is Apache-2.0; other popular models
  (IDM-VTON, CatVTON, OOTDiffusion) are non-commercial and would only ever be optional downloads.
- [ ] **Control API** (CLI + MCP server) so voice assistants can say "try the gold one on me".
- [ ] A small GUI for picking items and saving looks.

## Develop

```bash
uv sync
uv run pytest
```

The pieces are small and separate: `anchors.py` (where items sit and what hides them; pure 3D geometry,
unit-tested with fake landmarks), `mesh3d.py` (loading models, physically based OpenGL rendering),
`jewelry.py` (the built-in jewellery), `armour.py` (the helmet and gauntlets), `assets.py` (downloads and credits), `items.py` (loading items, cutting
out photos), `render.py` (photo items, lighting match, compositing), `tracker.py` (MediaPipe), `session.py`
(what's worn, smoothing, masks) and `cli.py`. 3D tests skip themselves where there's no OpenGL.

To add a new item kind: add its placement function to `anchors.py`, its default attach point to
`items.DEFAULT_ANCHOR`, the trackers it needs to `tracker.NEEDS`, file-name words to `items._KIND_WORDS`, and
a test.

## Licence

[Apache-2.0](LICENSE). MediaPipe and its models are Apache-2.0 too. The downloaded models keep their own
licences (see the table above): the Khronos sunglasses and watch are CC-BY 4.0 (credit Eric Chadwick /
Darmstadt Graphics Group; the watch is based on "Chronograph Watch Mudmaster" by graphiccompressor) and carry the
Khronos and 3D Commerce logos, trademarks of the Khronos Group; Poly Haven assets are CC0.
