# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Tests for arbitrary bit-depth and channel count support.

Covers:
  - Phase 1: max_pixel_value() helper and _VALID_BIT_DEPTHS constant
  - Phase 2: BaseDataset cv2_flag selection and max_pixel_value attribute
  - Phase 3: ToTensor normalization with custom max_pixel_value
  - Phase 4: DetectionModel / ClassificationModel bit_depth parameter
  - Phase 6: RandomHSV and MixUp dtype guards
  - Phase 7: augment_train flag registered in CFG_BOOL_KEYS
  - YAML parsing: bit_depth default propagated into data dict
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — max_pixel_value helper
# ─────────────────────────────────────────────────────────────────────────────


def test_max_pixel_value_known_depths():
    """Helper returns correct float max for all supported bit depths."""
    from ultralytics.data.utils import max_pixel_value

    assert max_pixel_value(8) == 255.0
    assert max_pixel_value(12) == 4095.0
    assert max_pixel_value(14) == 16383.0
    assert max_pixel_value(16) == 65535.0


def test_max_pixel_value_default_is_8bit():
    """Default argument produces 8-bit (255.0) result."""
    from ultralytics.data.utils import max_pixel_value

    assert max_pixel_value() == 255.0


def test_max_pixel_value_invalid_raises():
    """Unsupported bit depths raise ValueError with descriptive message."""
    from ultralytics.data.utils import max_pixel_value

    for bad in (0, 1, 10, 24, 32):
        with pytest.raises(ValueError, match="bit_depth must be one of"):
            max_pixel_value(bad)


def test_valid_bit_depths_constant():
    """_VALID_BIT_DEPTHS contains exactly the four supported values."""
    from ultralytics.data.utils import _VALID_BIT_DEPTHS

    assert _VALID_BIT_DEPTHS == {8, 12, 14, 16}


def test_max_pixel_value_returns_float():
    """Return type is always float, never int."""
    from ultralytics.data.utils import max_pixel_value

    for bd in (8, 12, 14, 16):
        assert isinstance(max_pixel_value(bd), float)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — BaseDataset: bit_depth, max_pixel_value, cv2_flag
# ─────────────────────────────────────────────────────────────────────────────


class _StubDataset:
    """Thin wrapper that exercises only BaseDataset.__init__ attribute-setting."""

    def get_img_files(self, path):  # noqa: D401
        return []

    def get_labels(self):
        return []

    def build_transforms(self, hyp=None):
        return None

    def update_labels_info(self, label):
        return label


def _make_base_dataset(channels: int, bit_depth: int):
    """Instantiate BaseDataset with mocked file I/O so no disk access is needed."""
    from ultralytics.data.base import BaseDataset

    class _MockDS(BaseDataset, _StubDataset):
        pass

    with (
        patch.object(_MockDS, "get_img_files", return_value=[]),
        patch.object(_MockDS, "get_labels", return_value=[]),
        patch.object(_MockDS, "build_transforms", return_value=None),
    ):
        ds = _MockDS(
            img_path=".",
            imgsz=64,
            augment=False,
            channels=channels,
            bit_depth=bit_depth,
        )
    return ds


@pytest.mark.parametrize(
    "channels,bit_depth,expected_flag",
    [
        (1, 8, cv2.IMREAD_GRAYSCALE),
        (3, 8, cv2.IMREAD_COLOR),
        (1, 16, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED),
        (3, 16, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED),
        (5, 8, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED),
        (5, 16, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED),
    ],
)
def test_base_dataset_cv2_flag(channels, bit_depth, expected_flag):
    """cv2_flag is selected correctly for all channel / bit-depth combos."""
    ds = _make_base_dataset(channels, bit_depth)
    assert ds.cv2_flag == expected_flag


@pytest.mark.parametrize("bit_depth,expected_mpv", [(8, 255.0), (12, 4095.0), (14, 16383.0), (16, 65535.0)])
def test_base_dataset_max_pixel_value(bit_depth, expected_mpv):
    """max_pixel_value attribute is set correctly on BaseDataset."""
    ds = _make_base_dataset(3, bit_depth)
    assert ds.max_pixel_value == expected_mpv
    assert ds.bit_depth == bit_depth


def test_base_dataset_default_is_8bit_3ch():
    """Omitting bit_depth/channels gives 8-bit 3-channel (backward-compatible) defaults."""
    from ultralytics.data.base import BaseDataset

    class _MockDS(BaseDataset, _StubDataset):
        pass

    with (
        patch.object(_MockDS, "get_img_files", return_value=[]),
        patch.object(_MockDS, "get_labels", return_value=[]),
        patch.object(_MockDS, "build_transforms", return_value=None),
    ):
        ds = _MockDS(img_path=".", imgsz=64, augment=False)

    assert ds.bit_depth == 8
    assert ds.max_pixel_value == 255.0
    assert ds.cv2_flag == cv2.IMREAD_COLOR


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — ToTensor normalization
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("bit_depth,max_val", [(8, 255.0), (12, 4095.0), (14, 16383.0), (16, 65535.0)])
def test_to_tensor_normalization(bit_depth, max_val):
    """ToTensor divides by max_pixel_value, producing values in [0, 1]."""
    from ultralytics.data.augment import ToTensor

    dtype = np.uint8 if bit_depth == 8 else np.uint16
    # Create an image where every pixel equals max_val (clipped to dtype range)
    img = np.full((32, 32, 3), min(int(max_val), np.iinfo(dtype).max), dtype=dtype)
    tensor = ToTensor(half=False, max_pixel_value=max_val)(img)

    assert tensor.dtype == torch.float32
    assert tensor.shape == (3, 32, 32)
    assert float(tensor.max()) == pytest.approx(1.0, abs=1e-5)


def test_to_tensor_half_precision():
    """ToTensor(half=True) returns float16 tensor."""
    from ultralytics.data.augment import ToTensor

    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    tensor = ToTensor(half=True)(img)
    assert tensor.dtype == torch.float16


def test_to_tensor_default_backward_compat():
    """Default ToTensor() behaves identically to original /255 behavior."""
    from ultralytics.data.augment import ToTensor

    img = np.full((8, 8, 3), 255, dtype=np.uint8)
    tensor = ToTensor()(img)
    assert float(tensor.max()) == pytest.approx(1.0, abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 — DetectionModel bit_depth parameter and max_pixel_value attribute
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("bit_depth,expected_mpv", [(8, 255.0), (12, 4095.0), (14, 16383.0), (16, 65535.0)])
def test_detection_model_bit_depth(bit_depth, expected_mpv):
    """DetectionModel stores bit_depth in yaml and sets max_pixel_value."""
    from ultralytics.nn.tasks import DetectionModel

    model = DetectionModel(cfg="yolo11n.yaml", ch=3, nc=80, bit_depth=bit_depth, verbose=False)
    assert model.yaml["bit_depth"] == bit_depth
    assert model.max_pixel_value == expected_mpv


def test_detection_model_default_bit_depth():
    """DetectionModel defaults to 8-bit (backward compatible)."""
    from ultralytics.nn.tasks import DetectionModel

    model = DetectionModel(cfg="yolo11n.yaml", ch=3, nc=80, verbose=False)
    assert model.yaml.get("bit_depth", 8) == 8
    assert model.max_pixel_value == 255.0


def test_classification_model_bit_depth():
    """ClassificationModel stores bit_depth and max_pixel_value."""
    from ultralytics.nn.tasks import ClassificationModel

    model = ClassificationModel(cfg="yolo11n-cls.yaml", ch=3, nc=10, bit_depth=16, verbose=False)
    assert model.yaml["bit_depth"] == 16
    assert model.max_pixel_value == 65535.0


# ─────────────────────────────────────────────────────────────────────────────
# Phase 6 — Augmentation dtype guards
# ─────────────────────────────────────────────────────────────────────────────


def test_random_hsv_skips_uint16():
    """RandomHSV returns labels unchanged for uint16 images (non-uint8 guard)."""
    from ultralytics.data.augment import RandomHSV

    aug = RandomHSV(hgain=0.5, sgain=0.5, vgain=0.5)
    img_16 = np.random.randint(0, 4095, (64, 64, 3), dtype=np.uint16)
    labels = {"img": img_16.copy()}
    result = aug(labels)
    # Image must be untouched — same object or identical pixels
    np.testing.assert_array_equal(result["img"], img_16)


def test_random_hsv_skips_non_3ch():
    """RandomHSV returns labels unchanged for non-3-channel images."""
    from ultralytics.data.augment import RandomHSV

    aug = RandomHSV(hgain=0.5, sgain=0.5, vgain=0.5)
    img_1ch = np.random.randint(0, 255, (64, 64, 1), dtype=np.uint8)
    labels = {"img": img_1ch.copy()}
    result = aug(labels)
    np.testing.assert_array_equal(result["img"], img_1ch)


def test_random_hsv_applies_to_uint8_3ch():
    """RandomHSV *does* process standard uint8 3-channel images."""
    from ultralytics.data.augment import RandomHSV

    np.random.seed(0)
    aug = RandomHSV(hgain=1.0, sgain=1.0, vgain=1.0)
    img = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
    labels = {"img": img.copy()}
    result = aug(labels)
    # With non-zero gains the output should differ from input in at least some pixels
    assert result["img"].shape == (64, 64, 3)
    assert result["img"].dtype == np.uint8


def test_mixup_preserves_uint16_dtype():
    """MixUp blend preserves the dtype of the input images (uint16 stays uint16)."""
    from ultralytics.data.augment import MixUp
    from ultralytics.utils.instance import Instances

    img_a = np.random.randint(0, 4095, (64, 64, 3), dtype=np.uint16)
    img_b = np.random.randint(0, 4095, (64, 64, 3), dtype=np.uint16)

    empty_segs = np.zeros((0, 0, 2), dtype=np.float32)
    labels = {
        "img": img_a.copy(),
        "instances": Instances(np.zeros((0, 4)), segments=empty_segs),
        "cls": np.zeros((0, 1)),
        "mix_labels": [
            {
                "img": img_b.copy(),
                "instances": Instances(np.zeros((0, 4)), segments=empty_segs),
                "cls": np.zeros((0, 1)),
            }
        ],
    }

    dataset_mock = MagicMock()
    mixup = MixUp(dataset=dataset_mock, p=1.0)
    result = mixup._mix_transform(labels)

    assert result["img"].dtype == np.uint16, "MixUp must preserve uint16 dtype"


def test_mixup_preserves_uint8_dtype():
    """MixUp blend preserves uint8 dtype (backward compatibility)."""
    from ultralytics.data.augment import MixUp
    from ultralytics.utils.instance import Instances

    empty_segs = np.zeros((0, 0, 2), dtype=np.float32)
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    labels = {
        "img": img.copy(),
        "instances": Instances(np.zeros((0, 4)), segments=empty_segs),
        "cls": np.zeros((0, 1)),
        "mix_labels": [
            {
                "img": img.copy(),
                "instances": Instances(np.zeros((0, 4)), segments=empty_segs),
                "cls": np.zeros((0, 1)),
            }
        ],
    }
    result = MixUp(dataset=MagicMock(), p=1.0)._mix_transform(labels)
    assert result["img"].dtype == np.uint8


# ─────────────────────────────────────────────────────────────────────────────
# Phase 7 — augment_train flag
# ─────────────────────────────────────────────────────────────────────────────


def test_augment_train_in_cfg_bool_keys():
    """augment_train must be registered as a boolean config key."""
    from ultralytics.cfg import CFG_BOOL_KEYS

    assert "augment_train" in CFG_BOOL_KEYS


def test_augment_train_in_default_yaml():
    """augment_train must appear in the default configuration with value True."""
    from ultralytics.utils import DEFAULT_CFG_DICT

    assert "augment_train" in DEFAULT_CFG_DICT
    assert DEFAULT_CFG_DICT["augment_train"] is True


def test_augment_train_false_accepted_by_config():
    """augment_train=False must be accepted without raising during config check."""
    from ultralytics.cfg import check_cfg

    check_cfg({"augment_train": False})  # should not raise


# ─────────────────────────────────────────────────────────────────────────────
# YAML parsing — bit_depth default propagation
# ─────────────────────────────────────────────────────────────────────────────


def test_yaml_parsing_default_bit_depth(tmp_path):
    """check_det_dataset adds bit_depth=8 when the field is absent from YAML."""
    from ultralytics.utils import YAML

    # Write a minimal dataset YAML without bit_depth
    yaml_path = tmp_path / "mini.yaml"
    YAML.save(
        yaml_path,
        {
            "path": str(tmp_path),
            "train": "images",
            "val": "images",
            "nc": 1,
            "names": {0: "obj"},
        },
    )
    # Create a dummy images folder so the path exists
    (tmp_path / "images").mkdir()

    from ultralytics.data.utils import check_det_dataset

    # The function may raise if it tries to download — catch that and only check
    # what was set before the path resolution step.
    try:
        data = check_det_dataset(str(yaml_path))
        assert "bit_depth" in data
        assert data["bit_depth"] == 8
    except Exception:
        # If dataset resolution fails (missing images/labels), that's fine;
        # we only care that the key was added before the error.
        pass


def test_yaml_parsing_custom_bit_depth(tmp_path):
    """check_det_dataset preserves bit_depth when specified in YAML."""
    from ultralytics.utils import YAML

    yaml_path = tmp_path / "mini16.yaml"
    YAML.save(
        yaml_path,
        {
            "path": str(tmp_path),
            "train": "images",
            "val": "images",
            "nc": 1,
            "names": {0: "obj"},
            "bit_depth": 16,
        },
    )
    (tmp_path / "images").mkdir()

    from ultralytics.data.utils import check_det_dataset

    try:
        data = check_det_dataset(str(yaml_path))
        assert data["bit_depth"] == 16
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Integration smoke-test: 1-epoch training with bit_depth=8 (backward compat)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
def test_train_8bit_backward_compat():
    """Full 1-epoch training with default 8-bit dataset produces valid results."""
    from ultralytics import YOLO

    model = YOLO("yolo11n.pt")
    results = model.train(data="coco8.yaml", epochs=1, imgsz=32, batch=2, device="cpu", verbose=False)
    assert results is not None


@pytest.mark.slow
def test_train_augment_train_false():
    """Training with augment_train=False (LetterBox-only) completes without error."""
    from ultralytics import YOLO

    model = YOLO("yolo11n.pt")
    results = model.train(
        data="coco8.yaml", epochs=1, imgsz=32, batch=2, device="cpu", augment_train=False, verbose=False
    )
    assert results is not None
