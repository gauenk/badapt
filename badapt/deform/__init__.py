
import os
from pathlib import Path

import cppyy
import numpy as np
import ctypes
import cv2


import random
import torch as th


# ------------------------------------------------------------
#     Load and compile C++ implementation
# ------------------------------------------------------------

HERE = Path(__file__).resolve().parent

cppyy.add_include_path(str(HERE))
cppyy.include("deform.hpp")

cppyy.cppdef(
    '#include "deform.cpp"'
)

# ------------------------------------------------------------
#      Main Deformation Function
# ------------------------------------------------------------

def deform_mask(mask, regional_sample_rate=0.1, sample_rate=0.1, iou_target = 0.90, allow_perturb=True, seed=None):

    
    move_rate = 0.0
    if th.is_tensor(mask):
        device = mask.device
        mask = mask.cpu().numpy()
    else:
        device = 'cpu'
    # original = mask.copy()
    if mask.max() == 1: mask = mask*255
    mask = mask.astype(np.uint8)

    # get boundaries
    if int(cv2.__version__[0]) >= 4:
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    else:
        _, contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

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
    Perturb a 2D segmentation mask.

    Parameters
    ----------
    gt : np.ndarray
        2D image or segmentation mask.
    iou_target : float
        Target IoU threshold.
    seed : int
        Random seed; negative values use nondeterministic seeding.

    Returns
    -------
    np.ndarray
        Perturbed uint8 mask with shape (H, W).
    """
    gt = np.ascontiguousarray(gt, dtype=np.uint8)

    if gt.ndim != 2:
        raise ValueError("Input must be a 2D array")

    h, w = gt.shape
    output = np.empty((h, w), dtype=np.uint8)

    if h == 0 or w == 0:
        return output
    
    # Pass NumPy buffers as typed C++ pointers.
    gt_ptr = ctypes.cast(
        gt.ctypes.data,
        ctypes.POINTER(ctypes.c_uint8),
    )
    
    out_ptr = ctypes.cast(
        output.ctypes.data,
        ctypes.POINTER(ctypes.c_uint8),
    )

    cppyy.gbl.deform_seg_core(
        gt_ptr,
        int(h),
        int(w),
        float(iou_target),
        int(seed),
        out_ptr,
    )

    return output


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
    Process contours returned by cv2.findContours().

    Returns a list of int32 arrays with shape (N, 1, 2),
    suitable for cv2.drawContours().
    """
    cpp_contours = cppyy.gbl.std.vector["Contour"]()

    for contour in contours:
        arr = np.asarray(contour)

        if arr.size == 0:
            continue

        arr = np.ascontiguousarray(arr.reshape(-1, 2), dtype=np.int32)

        c = cppyy.gbl.Contour()

        for x, y in arr:
            point = cppyy.gbl.Point()
            point[0] = int(x)
            point[1] = int(y)
            c.push_back(point)

        cpp_contours.push_back(c)

    result = cppyy.gbl.deform_contours_core(
        cpp_contours,
        float(regional_sample_rate),
        float(sample_rate),
        float(move_rate),
        int(seed),
    )

    output = []

    for contour in result:
        points = np.empty((len(contour), 1, 2), dtype=np.int32)

        for i, point in enumerate(contour):
            points[i, 0, 0] = point[0]
            points[i, 0, 1] = point[1]

        output.append(points)

    return output



