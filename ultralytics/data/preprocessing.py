"""Image preprocessing utilities for CLAHE, histogram equalization, and per-image minmax normalization."""

import cv2
import numpy as np
import torch

_SPATIAL_METHODS = {"clahe", "histogram_eq"}
_NORM_METHODS = {"minmax"}
VALID_METHODS = _SPATIAL_METHODS | _NORM_METHODS | {None}


def _clahe_channel_u16(ch: np.ndarray, clahe, bit_depth: int) -> np.ndarray:
    """Apply CLAHE to a single uint16 channel, handling sub-16-bit data.

    OpenCV CLAHE on uint16 maps output to the full 0–65535 range. For sub-16-bit
    data (e.g. 14-bit values in 0–16383) we must scale up to fill uint16 before
    CLAHE, then scale back down, so the result stays within the original bit range.

    Args:
        ch (np.ndarray): 2D uint16 array (H, W).
        clahe: cv2.CLAHE object.
        bit_depth (int): Actual bit depth of the data (8, 12, 14, or 16).

    Returns:
        (np.ndarray): CLAHE-enhanced uint16 channel in the original bit-depth range.
    """
    shift = 16 - bit_depth  # e.g. 2 for 14-bit, 4 for 12-bit, 0 for 16-bit
    if shift > 0:
        ch = ch << shift  # scale 0–16383 → 0–65535
    ch = clahe.apply(ch)  # CLAHE operates on full uint16 range
    if shift > 0:
        ch = ch >> shift  # scale back to 0–16383
    return ch


def apply_clahe(im: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8, bit_depth: int = 8) -> np.ndarray:
    """Apply CLAHE to a HWC numpy array (uint8 or uint16).

    For 1-channel images: applies CLAHE directly to the single channel.
      OpenCV CLAHE natively supports both uint8 and uint16.
    For 3-channel uint8 images: converts to LAB color space, applies CLAHE to
      the L (lightness) channel only, then converts back to preserve color balance.
    For 3-channel uint16 images: applies CLAHE per-channel (OpenCV LAB conversion
      does not support uint16).

    Note: For sub-16-bit data stored in uint16 containers (e.g. 14-bit in uint16),
    channels are scaled to the full uint16 range before CLAHE and scaled back after,
    so the output stays within the original bit-depth range.

    Args:
        im (np.ndarray): HWC image array, uint8 or uint16, shape (H, W, C).
        clip_limit (float): Threshold for contrast limiting.
        tile_size (int): Size of the grid tiles (NxN).
        bit_depth (int): Bit depth of the source image (8, 12, 14, or 16).

    Returns:
        (np.ndarray): CLAHE-enhanced image, same dtype and shape as input.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    c = im.shape[2]

    if c == 1:
        if im.dtype == np.uint16:
            im[:, :, 0] = _clahe_channel_u16(im[:, :, 0], clahe, bit_depth)
        else:
            im[:, :, 0] = clahe.apply(im[:, :, 0])
    elif im.dtype == np.uint8:
        # LAB-space CLAHE for color images preserves hue/saturation
        lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        im = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        # uint16 multi-channel: per-channel CLAHE (LAB not supported at uint16)
        for i in range(c):
            im[:, :, i] = _clahe_channel_u16(im[:, :, i], clahe, bit_depth)

    return im


def apply_histogram_eq(im: np.ndarray, bit_depth: int = 8) -> np.ndarray:
    """Apply global histogram equalization to a HWC numpy array.

    cv2.equalizeHist only supports uint8. For uint16 images the channel is
    right-shifted to uint8, equalized, then shifted back, preserving the
    original bit depth range.

    Args:
        im (np.ndarray): HWC image array, uint8 or uint16, shape (H, W, C).
        bit_depth (int): Bit depth of the source image (8, 12, 14, or 16).

    Returns:
        (np.ndarray): Histogram-equalized image, same dtype and shape as input.
    """
    shift = max(bit_depth - 8, 0)  # e.g. 6 for 14-bit, 8 for 16-bit
    for i in range(im.shape[2]):
        ch = im[:, :, i]
        if im.dtype == np.uint8:
            im[:, :, i] = cv2.equalizeHist(ch)
        else:
            ch8 = (ch >> shift).astype(np.uint8)
            im[:, :, i] = cv2.equalizeHist(ch8).astype(im.dtype) << shift
    return im


def apply_spatial_preprocessing(
    im: np.ndarray,
    method: str,
    clip_limit: float = 2.0,
    tile_size: int = 8,
    bit_depth: int = 8,
) -> np.ndarray:
    """Dispatch spatial image preprocessing (CLAHE or histogram equalization).

    Args:
        im (np.ndarray): HWC image array, uint8 or uint16.
        method (str): One of 'clahe' or 'histogram_eq'.
        clip_limit (float): CLAHE clip limit (used only for 'clahe').
        tile_size (int): CLAHE tile grid size NxN (used only for 'clahe').
        bit_depth (int): Bit depth of the source image.

    Returns:
        (np.ndarray): Preprocessed image, same dtype and shape as input.
    """
    if method == "clahe":
        return apply_clahe(im, clip_limit, tile_size, bit_depth)
    if method == "histogram_eq":
        return apply_histogram_eq(im, bit_depth)
    return im


def apply_minmax_normalization(imgs: torch.Tensor) -> torch.Tensor:
    """Per-image min-max normalization for a batch of float tensors.

    Each image is independently scaled to [0, 1] based on its own min/max
    pixel value, rather than dividing by a global bit-depth maximum.

    Args:
        imgs (torch.Tensor): Float tensor of shape (B, C, H, W).

    Returns:
        (torch.Tensor): Normalized float tensor in [0, 1] per image.
    """
    b = imgs.shape[0]
    flat = imgs.view(b, -1)
    mn = flat.min(dim=1).values[:, None, None, None]
    mx = flat.max(dim=1).values[:, None, None, None]
    return (imgs - mn) / (mx - mn + 1e-8)
