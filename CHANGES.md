# Changes: Arbitrary Bit Depth & Channel Count Support

This document summarises every source-code change made to add support for
**arbitrary image bit depths (8 / 12 / 14 / 16-bit)**, **arbitrary channel
counts (1, 2, 3, 4, 5, 10 …)**, and a new **`augment_train` training flag**.

All changes are backward-compatible: the default `bit_depth = 8` reproduces
exactly the original behaviour.

---

## New concept: `max_pixel_value`

`max_pixel_value = (1 << bit_depth) - 1`

| bit_depth | max_pixel_value |
|-----------|-----------------|
| 8  | 255.0   |
| 12 | 4095.0  |
| 14 | 16383.0 |
| 16 | 65535.0 |

Every hardcoded `/255` in the pipeline is replaced by `/ max_pixel_value`.

---

## Phase 1 — Configuration & Helper

### `ultralytics/data/utils.py`

- Added module-level constant:
  ```python
  _VALID_BIT_DEPTHS = {8, 12, 14, 16}
  ```
- Added helper function:
  ```python
  def max_pixel_value(bit_depth: int = 8) -> float:
      """Return max pixel value for a given bit depth (255, 4095, 16383, 65535)."""
  ```
- In `check_det_dataset()` (YAML parsing):
  ```python
  data["channels"] = data.get("channels", 3)
  data["bit_depth"] = data.get("bit_depth", 8)   # NEW
  ```
- In `check_cls_dataset()` return dict: added `"bit_depth": 8`.

---

## Phase 2 — Image Loading

### `ultralytics/data/base.py`

Added `bit_depth: int = 8` to `BaseDataset.__init__`.

Replaced the two-way cv2 flag with a three-way selection:

```python
self.bit_depth = bit_depth
self.max_pixel_value = float((1 << bit_depth) - 1)

if channels == 1 and bit_depth == 8:
    self.cv2_flag = cv2.IMREAD_GRAYSCALE          # original 1-ch path
elif channels == 3 and bit_depth == 8:
    self.cv2_flag = cv2.IMREAD_COLOR              # original 3-ch path
else:
    self.cv2_flag = cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED  # NEW: multi-ch / high-bit
```

### `ultralytics/data/dataset.py`

- `YOLODataset.__init__`: passes `bit_depth=self.data.get("bit_depth", 8)` to
  `super().__init__()`.
- `GroundingDataset.__init__`: fixed hardcoded `data={"channels": 3}` →
  `data={"channels": 3, "bit_depth": 8}`.

### `ultralytics/utils/patches.py`

`imread()` inner function previously called
`cv2.imdecodemulti(file_bytes, cv2.IMREAD_UNCHANGED)` (hardcoded).
Changed to `cv2.imdecodemulti(file_bytes, flags)` to forward the caller's
flags (e.g. `IMREAD_ANYDEPTH | IMREAD_UNCHANGED` for 16-bit TIFFs).

### `ultralytics/data/loaders.py`

All three loader classes (`LoadStreams`, `LoadScreenshots`, `LoadImages`)
had a two-way cv2 flag selection.  Replaced with the same three-way logic
as `BaseDataset` above.

---

## Phase 3 — Normalisation (replacing `/255`)

### `ultralytics/models/yolo/detect/train.py`

- Added import: `from ultralytics.data.utils import max_pixel_value`
- `preprocess_batch`:
  ```python
  # was:  batch["img"].float() / 255
  batch["img"] = batch["img"].float() / max_pixel_value(self.data.get("bit_depth", 8))
  ```
  Cascades automatically to `SegmentationTrainer`, `PoseTrainer`,
  `OBBTrainer`, `WorldTrainer`, `YOLOETrainer`.

### `ultralytics/models/yolo/detect/val.py`

- Added import: `from ultralytics.data.utils import max_pixel_value`
- `preprocess`:
  ```python
  # was:  batch["img"].float() / 255
  batch["img"] = (...) / max_pixel_value(self.data.get("bit_depth", 8))
  ```
  Cascades to all task validators.

### `ultralytics/engine/predictor.py`

- `preprocess`:
  ```python
  # was:  im /= 255
  im /= getattr(self.model, "max_pixel_value", 255.0)
  ```

### `ultralytics/data/augment.py` — `ToTensor`

- Added `max_pixel_value: float = 255.0` parameter to `__init__`.
- `__call__`: `im /= self.max_pixel_value` (was `im /= 255.0`).

---

## Phase 4 — Model Metadata

### `ultralytics/nn/tasks.py`

Every model class now accepts a `bit_depth` constructor parameter and
stores `max_pixel_value` as an instance attribute.

**`DetectionModel.__init__`** (base for all detection-family models):
```python
def __init__(self, cfg="yolo11n.yaml", ch=3, nc=None, bit_depth=8, verbose=True):
    ...
    self.yaml["channels"] = ch
    self.yaml["bit_depth"] = self.yaml.get("bit_depth", bit_depth)  # NEW
    self.max_pixel_value = float((1 << self.yaml["bit_depth"]) - 1)  # NEW
```

**`ClassificationModel.__init__` / `_from_yaml`**: same additions (independent
inheritance chain).

**Subclasses updated** — each gets `bit_depth=8` parameter forwarded to
`super().__init__()`:
- `OBBModel`
- `SegmentationModel`
- `PoseModel`
- `WorldModel`
- `YOLOEModel`
- `YOLOESegModel`
- `RTDETRDetectionModel`

### Trainer `get_model` — 9 sites

Each trainer previously called the model constructor with only `ch=…`.
Added `bit_depth=self.data.get("bit_depth", 8)` to all:

| File | Model class |
|------|-------------|
| `models/yolo/detect/train.py` | `DetectionModel` |
| `models/yolo/segment/train.py` | `SegmentationModel` |
| `models/yolo/obb/train.py` | `OBBModel` |
| `models/yolo/pose/train.py` | `PoseModel` |
| `models/yolo/classify/train.py` | `ClassificationModel` |
| `models/yolo/world/train.py` | `WorldModel` |
| `models/yolo/yoloe/train.py` | `YOLOEModel` (2 sites) |
| `models/yolo/yoloe/train_seg.py` | `YOLOESegModel` (2 sites) |
| `models/rtdetr/train.py` | `RTDETRDetectionModel` |

### `ultralytics/nn/autobackend.py`

- `stride, ch = 32, 3` → `stride, ch, bit_depth = 32, 3, 8`
- Added `"bit_depth"` to the set of keys that are auto-cast to `int` when
  reading metadata.
- Reads `bit_depth` from `model.yaml` (PyTorch) and from `metadata` dict
  (ONNX / TFLite / CoreML / etc.).
- After `self.__dict__.update(locals())`:
  ```python
  self.max_pixel_value = float((1 << self.bit_depth) - 1)
  ```

---

## Phase 5 — Export Pipelines

### `ultralytics/engine/exporter.py`

- Metadata dict: added `"bit_depth": model.yaml.get("bit_depth", 8)` so
  exported models carry the bit-depth of their training dataset.
- **OpenVINO**: `scale_values` computed as
  `[float((1 << self.model.yaml.get("bit_depth", 8)) - 1)]` (was `[255]`).
- **CoreML**: `scale` computed as `1 / _mpv` where `_mpv` is the model's
  max pixel value (was hardcoded `1/255.0`).

### `ultralytics/utils/export/engine.py`

- `EngineCalibrator.__init__`: added `max_pixel_value: float = 255.0`
  parameter.
- `get_batch`: divides by `self.max_pixel_value` (was `/ 255.0`).
- Calibrator instantiation: passes
  `max_pixel_value=float((1 << _bd) - 1)` derived from export metadata.

### `ultralytics/utils/export/imx.py`

- `_max_pixel_value = getattr(model, "max_pixel_value", 255.0)` before the
  representative-dataset generator closure.
- Closure uses `_max_pixel_value` (was `/ 255.0`).

---

## Phase 6 — Augmentation Safety Guards

### `ultralytics/data/augment.py`

**Mosaic** (3-tile, 4-tile, 9-tile canvas — 3 sites):
```python
# was:  np.full(..., 114, dtype=np.uint8)
_fill = int(114 / 255.0 * getattr(self.dataset, "max_pixel_value", 255.0))
img4 = np.full((...), _fill, dtype=img.dtype)   # dtype follows input
```

**MixUp blend** (`_mix_transform`):
```python
# was:  .astype(np.uint8)
labels["img"] = (...).astype(labels["img"].dtype)
```

**RandomHSV** (`__call__`):
```python
if img.shape[-1] != 3 or img.dtype != np.uint8:
    return labels   # skip — LUT only valid for uint8
```

**Albumentations wrapper** (`__call__`):
```python
if im.shape[2] != 3 or im.dtype != np.uint8:
    return labels   # Albumentations expects uint8 RGB
```

---

## Phase 7 — `augment_train` Flag

### `ultralytics/cfg/default.yaml`

Added near the existing `close_mosaic` flag:
```yaml
augment_train: True  # (bool) enable v8-style training augmentation (mosaic, mixup, HSV, etc); False = LetterBox only
```

### `ultralytics/cfg/__init__.py`

Added `"augment_train"` to `CFG_BOOL_KEYS` (so it is accepted and
type-validated as a boolean argument).

### `ultralytics/data/build.py` — 2 sites

```python
# was:  augment=mode == "train"
augment = mode == "train" and getattr(cfg, "augment_train", True)
```

### `ultralytics/models/rtdetr/train.py`

```python
augment = mode == "train" and getattr(self.args, "augment_train", True)
```

### `ultralytics/models/yolo/classify/train.py`

```python
augment = mode == "train" and getattr(self.args, "augment_train", True)
```

---

## Phase 9 — Converter Fix

### `ultralytics/data/converter.py` — `convert_to_multispectral`

Added `bit_depth: int = 8` parameter.

```python
# was:  np.clip(multispectral, 0, 255).astype(np.uint8)
max_val = (1 << bit_depth) - 1
out_dtype = np.uint8 if bit_depth == 8 else np.uint16
np.clip(multispectral, 0, max_val).astype(out_dtype)
```

---

## Files Changed — Quick Reference

| File | Change category |
|------|-----------------|
| `ultralytics/data/utils.py` | Helper + YAML parsing |
| `ultralytics/data/base.py` | cv2_flag, bit_depth, max_pixel_value |
| `ultralytics/data/dataset.py` | Propagate bit_depth to super |
| `ultralytics/utils/patches.py` | Forward imread flags |
| `ultralytics/data/loaders.py` | 3-way cv2_flag (3 sites) |
| `ultralytics/models/yolo/detect/train.py` | Normalisation + get_model |
| `ultralytics/models/yolo/detect/val.py` | Normalisation |
| `ultralytics/engine/predictor.py` | Normalisation |
| `ultralytics/data/augment.py` | ToTensor, Mosaic, MixUp, RandomHSV, Albumentations |
| `ultralytics/nn/tasks.py` | All model constructors (9 classes) |
| `ultralytics/nn/autobackend.py` | bit_depth → max_pixel_value on loaded model |
| `ultralytics/engine/exporter.py` | Metadata + OpenVINO/CoreML scale |
| `ultralytics/utils/export/engine.py` | TRT INT8 calibrator |
| `ultralytics/utils/export/imx.py` | IMX representative dataset |
| `ultralytics/cfg/default.yaml` | augment_train flag |
| `ultralytics/cfg/__init__.py` | CFG_BOOL_KEYS |
| `ultralytics/data/build.py` | augment_train check (2 sites) |
| `ultralytics/models/rtdetr/train.py` | bit_depth + augment_train |
| `ultralytics/models/yolo/classify/train.py` | bit_depth + augment_train |
| `ultralytics/models/yolo/segment/train.py` | bit_depth in get_model |
| `ultralytics/models/yolo/obb/train.py` | bit_depth in get_model |
| `ultralytics/models/yolo/pose/train.py` | bit_depth in get_model |
| `ultralytics/models/yolo/world/train.py` | bit_depth in get_model |
| `ultralytics/models/yolo/yoloe/train.py` | bit_depth in get_model (2 sites) |
| `ultralytics/models/yolo/yoloe/train_seg.py` | bit_depth in get_model (2 sites) |
| `ultralytics/data/converter.py` | bit_depth-aware output dtype |
