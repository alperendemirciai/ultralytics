# Usage Guide: Extended Bit Depth & Channel Count Support

This guide explains how to use the extended Ultralytics repository that
supports **arbitrary bit depths** (8, 12, 14, 16-bit) and **arbitrary
channel counts** (grayscale, RGB, multispectral, hyperspectral).

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Dataset YAML Configuration](#dataset-yaml-configuration)
3. [Training](#training)
4. [Validation](#validation)
5. [Inference / Prediction](#inference--prediction)
6. [Exporting Models](#exporting-models)
7. [Disabling Training Augmentation](#disabling-training-augmentation)
8. [Converting Multispectral Images](#converting-multispectral-images)
9. [Supported Configurations](#supported-configurations)
10. [Backward Compatibility](#backward-compatibility)
11. [Running the Tests](#running-the-tests)

---

## Quick Start

Standard 8-bit RGB training works **identically** to before — no YAML changes
needed.  Just add `bit_depth` (and/or `channels`) to your dataset YAML to
unlock the new functionality.

```yaml
# my_16bit_dataset.yaml
path: /data/my_dataset
train: images/train
val:   images/val
nc: 3
names: {0: cat, 1: dog, 2: person}

channels: 3      # number of image channels (default: 3)
bit_depth: 16    # valid: 8 (default), 12, 14, 16
```

```bash
yolo train model=yolo11n.yaml data=my_16bit_dataset.yaml epochs=100 imgsz=640
```

---

## Dataset YAML Configuration

Two new optional fields are recognised in every dataset YAML:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `channels` | int | `3` | Number of image channels |
| `bit_depth` | int | `8` | Bit depth per channel (8, 12, 14, or 16) |

### Examples

#### Grayscale 8-bit (1 channel)
```yaml
channels: 1
bit_depth: 8
```

#### 16-bit single-channel (e.g. depth maps, thermal)
```yaml
channels: 1
bit_depth: 16
```

#### 16-bit RGB (e.g. medical / scientific cameras)
```yaml
channels: 3
bit_depth: 16
```

#### 5-channel multispectral
```yaml
channels: 5
bit_depth: 16
```

#### 12-bit RAW-style (e.g. drone imagery)
```yaml
channels: 3
bit_depth: 12
```

> **Note**: Valid bit depths are **8, 12, 14, 16**.  Any other value raises
> `ValueError` at dataset load time.

---

## Training

### CLI

```bash
# 16-bit YOLO detection
yolo train model=yolo11n.yaml data=my_16bit_dataset.yaml epochs=100 imgsz=640

# 16-bit RTDETR
yolo train model=rtdetr-l.yaml data=my_16bit_dataset.yaml epochs=50 imgsz=640

# 5-channel multispectral segmentation
yolo train task=segment model=yolo11n-seg.yaml data=my_multispectral.yaml epochs=50

# OBB with 14-bit depth
yolo train task=obb model=yolo11n-obb.yaml data=my_14bit.yaml epochs=100

# Classification with 16-bit images
yolo train task=classify model=yolo11n-cls.yaml data=my_16bit_cls/ epochs=50
```

### Python API

```python
from ultralytics import YOLO

model = YOLO("yolo11n.yaml")   # or "yolo11n.pt" for fine-tuning
results = model.train(data="my_16bit_dataset.yaml", epochs=100, imgsz=640)
```

### What happens internally

1. `check_det_dataset()` reads `bit_depth` from the YAML (defaults to `8`).
2. `YOLODataset` passes `bit_depth` to `BaseDataset`, which sets:
   - `self.bit_depth`
   - `self.max_pixel_value = float((1 << bit_depth) - 1)`
   - The correct `cv2` imread flag (`IMREAD_ANYDEPTH | IMREAD_UNCHANGED`
     for anything other than standard 8-bit 1ch/3ch).
3. `DetectionTrainer.preprocess_batch` divides pixel values by
   `max_pixel_value(bit_depth)` instead of `255`.
4. The model constructor receives `bit_depth` and stores `max_pixel_value`
   so it can be used at inference time.

---

## Validation

Validation uses the same YAML as training, so `bit_depth` is automatically
picked up:

```bash
yolo val model=runs/detect/train/weights/best.pt data=my_16bit_dataset.yaml
```

```python
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")
metrics = model.val(data="my_16bit_dataset.yaml")
print(metrics.box.map)   # mAP50-95
```

---

## Inference / Prediction

For inference the model carries `max_pixel_value` internally (stored in its
weights file), so you do **not** need to supply `bit_depth` at prediction
time — it is read directly from the loaded model:

```bash
# Using a PyTorch checkpoint trained on 16-bit data
yolo predict model=runs/detect/train/weights/best.pt source=my_16bit_image.tif

# Using an exported ONNX (bit_depth is embedded in metadata)
yolo predict model=best.onnx source=my_16bit_image.tif
```

```python
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")
results = model.predict("my_16bit_image.tif")
for r in results:
    r.show()
```

### Manual override (advanced)

If you load a model that was trained before this change (no `bit_depth` in
its metadata), it defaults to `max_pixel_value = 255.0`.  You can override:

```python
model.model.max_pixel_value = 65535.0   # treat loaded model as 16-bit
```

---

## Exporting Models

```bash
yolo export model=runs/detect/train/weights/best.pt format=onnx
yolo export model=runs/detect/train/weights/best.pt format=openvino
yolo export model=runs/detect/train/weights/best.pt format=coreml
yolo export model=runs/detect/train/weights/best.pt format=engine  # TensorRT
```

The exported model automatically embeds `bit_depth` in its metadata, so the
correct `max_pixel_value` is used at inference for all formats.

---

## Disabling Training Augmentation

A new boolean flag `augment_train` (default: `True`) controls whether the
full v8-style augmentation pipeline (Mosaic, MixUp, RandomHSV, Random
Perspective, …) is applied during training.

Setting `augment_train: False` reduces training augmentation to **LetterBox
only** — the same pre-processing used at validation time.  This is useful for:

- Multispectral or hyperspectral data where HSV augmentation is meaningless.
- Debugging / sanity-checking (does the model overfit on unaugmented data?).
- Datasets where augmentation hurts rather than helps.

### CLI

```bash
# Disable all v8 training augmentation
yolo train model=yolo11n.yaml data=coco8.yaml epochs=10 augment_train=false

# Works for RTDETR too
yolo train model=rtdetr-l.yaml data=coco8.yaml epochs=10 augment_train=false
```

### Python API

```python
from ultralytics import YOLO

model = YOLO("yolo11n.yaml")
model.train(data="coco8.yaml", epochs=10, augment_train=False)
```

### In dataset YAML

```yaml
# my_dataset.yaml
path: /data/my_dataset
train: images/train
val:   images/val
nc: 1
names: {0: cell}
channels: 5
bit_depth: 16

augment_train: false   # no Mosaic / HSV / MixUp for this dataset
```

> **Note**: `augment_train` is entirely separate from the existing `augment`
> flag, which controls **test-time augmentation (TTA)** during prediction.

---

## Converting Multispectral Images

The `convert_to_multispectral` function in `ultralytics/data/converter.py`
now accepts a `bit_depth` parameter:

```python
from ultralytics.data.converter import convert_to_multispectral

# Convert to 16-bit multispectral (output saved as uint16 .tif)
convert_to_multispectral(
    input_path="path/to/raw_bands/",
    output_path="path/to/output/",
    bit_depth=16,   # NEW — default is 8
)
```

When `bit_depth > 8`, the output dtype is `np.uint16` and pixel values are
clipped to `[0, (1 << bit_depth) - 1]`.

---

## Supported Configurations

| channels | bit_depth | imread flag | dtype | Notes |
|----------|-----------|-------------|-------|-------|
| 1 | 8 | `IMREAD_GRAYSCALE` | uint8 | Original grayscale path |
| 3 | 8 | `IMREAD_COLOR` | uint8 | Original RGB/BGR path |
| 1 | 12/14/16 | `IMREAD_ANYDEPTH\|IMREAD_UNCHANGED` | uint16 | 16-bit grayscale |
| 3 | 12/14/16 | `IMREAD_ANYDEPTH\|IMREAD_UNCHANGED` | uint16 | 16-bit colour |
| N | 8/16 | `IMREAD_ANYDEPTH\|IMREAD_UNCHANGED` | uint8/16 | Multispectral |

### Augmentation behaviour by image type

| Augmentation | uint8 3ch | uint16 3ch | N-channel |
|-------------|-----------|------------|-----------|
| Mosaic | ✅ | ✅ (scaled fill) | ✅ (scaled fill) |
| MixUp | ✅ | ✅ (dtype preserved) | ✅ |
| RandomHSV | ✅ | ⛔ skipped | ⛔ skipped |
| Albumentations | ✅ | ⛔ skipped | ⛔ skipped |
| LetterBox | ✅ | ✅ | ✅ |
| RandomFlip | ✅ | ✅ | ✅ |

---

## Backward Compatibility

- All defaults are `bit_depth=8`, `channels=3` — existing code and datasets
  work without any changes.
- Existing `.pt` weights do not contain `bit_depth` metadata and will be
  treated as 8-bit (`max_pixel_value=255.0`) automatically.
- All task types are supported: detect, segment, pose, obb, classify,
  RTDETR, YOLOE, YOLOEWorld.

---

## Running the Tests

Unit and integration tests for the new functionality live in:

```
tests/test_bit_depth.py
```

Run fast unit tests only (no GPU, no download):

```bash
pytest tests/test_bit_depth.py -v
```

Run including slow integration tests (requires model weights):

```bash
pytest tests/test_bit_depth.py -v --slow
```

Run the full existing test suite to verify backward compatibility:

```bash
pytest tests/ -v --ignore=tests/test_bit_depth.py
```

### What the tests cover

| Test | Description |
|------|-------------|
| `test_max_pixel_value_*` | Helper returns correct floats; invalid depths raise |
| `test_valid_bit_depths_constant` | `_VALID_BIT_DEPTHS == {8,12,14,16}` |
| `test_base_dataset_cv2_flag` | 6 channel/depth combos → correct cv2 flag |
| `test_base_dataset_max_pixel_value` | `max_pixel_value` attribute set on dataset |
| `test_base_dataset_default_is_8bit_3ch` | Backward-compatible defaults |
| `test_to_tensor_normalization` | ToTensor produces `[0,1]` range for each bit depth |
| `test_to_tensor_half_precision` | `half=True` returns float16 |
| `test_detection_model_bit_depth` | Model stores `bit_depth` in yaml + `max_pixel_value` |
| `test_classification_model_bit_depth` | Same for ClassificationModel |
| `test_random_hsv_skips_uint16` | RandomHSV skips non-uint8 images |
| `test_random_hsv_skips_non_3ch` | RandomHSV skips non-3-channel images |
| `test_mixup_preserves_uint16_dtype` | MixUp output dtype matches input |
| `test_augment_train_in_cfg_bool_keys` | Flag registered as boolean |
| `test_augment_train_in_default_yaml` | Flag present and defaults to `True` |
| `test_augment_train_false_accepted_by_config` | `False` value passes type check |
| `test_yaml_parsing_default_bit_depth` | Missing `bit_depth` → defaults to 8 |
| `test_yaml_parsing_custom_bit_depth` | `bit_depth: 16` preserved through parsing |
| `test_train_8bit_backward_compat` *(slow)* | Full 1-epoch train unchanged |
| `test_train_augment_train_false` *(slow)* | `augment_train=False` trains OK |
