
import os
from pathlib import Path

import numpy as np
import ctypes
import cv2


import random
import torch as th
from badapt import _deform_cpp as deform_cpp

# ------------------------------------------------------------
#      Main Deformation Function
# ------------------------------------------------------------

def deform_mask(mask, regional_sample_rate=0.1, sample_rate=0.1, iou_target = 0.90, allow_perturb=True, seed=None):

    move_rate = 0.0
    device = 'cpu'
    if th.is_tensor(mask):
        device = mask.device
        mask = mask.cpu().numpy()
    if mask.max() == 1: mask = mask*255
    mask = mask.astype(np.uint8)

    # get boundaries
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if seed is None:
        seed = int(random.randint(0, 2**31 - 1))
    modified_contours = deform_contours(
        list(contours), float(regional_sample_rate), float(sample_rate), float(move_rate), seed
    )

    gt = np.copy(mask)
    mask = np.zeros((mask.shape[0], mask.shape[1], 3))
    modified_contours = [cont for cont in modified_contours if len(cont) > 0]
    if len(modified_contours) == 0: mask = gt.copy()
    else: mask = cv2.drawContours(mask, modified_contours, -1, 255, -1)[:,:,0]

    if allow_perturb:
        mask = deform_seg(mask, iou_target, seed)

    return th.from_numpy(mask>=0.5).int().to(device)

# ------------------------------------------------------------
#         Deform the Mask
# ------------------------------------------------------------


def deform_seg(gt, iou_target=0.8, seed=-1):
    """
    Perturb a 2D segmentation mask using the C++ deformation extension.

    Parameters
    ----------
    gt : np.ndarray or th.Tensor
        2D segmentation mask. Must be convertible to a CPU uint8 tensor.
    iou_target : float
        Target IoU threshold.
    seed : int
        Random seed; negative values use nondeterministic seeding.

    Returns
    -------
    np.ndarray
        Perturbed uint8 mask with shape (H, W).
    """
    # Convert input to a contiguous CPU uint8 tensor.
    if isinstance(gt, np.ndarray):
        gt = np.ascontiguousarray(gt, dtype=np.uint8)
        gt = th.from_numpy(gt)
    elif isinstance(gt, th.Tensor):
        gt = gt.to(device="cpu", dtype=th.uint8).contiguous()
    else:
        raise TypeError("Input must be a NumPy array or th.Tensor")

    if gt.ndim != 2:
        raise ValueError("Input must be a 2D array")

    h, w = gt.shape

    if h == 0 or w == 0:
        return np.empty((h, w), dtype=np.uint8)

    # Call the pybind11 extension.
    output = deform_cpp.deform_seg(
        gt,
        float(iou_target),
        int(seed),
    )

    # Return a NumPy array, preserving the original wrapper's behavior.
    return output.numpy()


# ------------------------------------------------------------
#        Deform the Boundary
# ------------------------------------------------------------

def deform_contours(
    contours,
    regional_sample_rate=0.1,
    sample_rate=0.1,
    move_rate=0.0,
    seed=-1,
):
    """
    Deform contours returned by cv2.findContours() using the C++ extension.

    Parameters
    ----------
    contours : list of np.ndarray
        Contours, typically with shape (N, 1, 2), as returned by
        cv2.findContours().
    regional_sample_rate : float
        Regional sampling rate.
    sample_rate : float
        Point sampling rate.
    move_rate : float
        Point movement rate.
    seed : int
        Random seed; negative values use nondeterministic seeding.

    Returns
    -------
    list of np.ndarray
        Deformed contours as int32 arrays with shape (N, 1, 2),
        suitable for cv2.drawContours().
    """
    cpp_contours = []

    for contour in contours:
        arr = np.asarray(contour)

        if arr.size == 0:
            continue

        arr = np.asarray(arr).reshape(-1, 2)

        # Convert each point to a pair of Python integers.
        cpp_contour = [
            [int(x), int(y)]
            for x, y in arr
        ]

        cpp_contours.append(cpp_contour)

    result = deform_cpp.deform_contours(
        cpp_contours,
        float(regional_sample_rate),
        float(sample_rate),
        float(move_rate),
        int(seed),
    )

    output = []

    for contour in result:
        points = np.asarray(contour, dtype=np.int32).reshape(-1, 1, 2)
        output.append(points)

    return output


