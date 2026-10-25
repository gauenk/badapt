"""

   Triton for CUDA-Kernels for Easier Sharing

"""


import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def get_boundary_kernel(
    new_labels_ptr,
    maps_ptr,
    H: tl.constexpr,
    W: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    NO_CORNERS: tl.constexpr,
    NO_BKG: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Flattened pixel indices
    pixels = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = pixels < N

    # Decode (batch, height, width)
    hw = H * W
    b = pixels // hw
    r = pixels % hw
    h = r // W
    w = r % W

    # Read the current pixel's label
    label = tl.load(maps_ptr + pixels, mask=valid, other=0)

    # Background pixels are left unchanged when NO_BKG is enabled.
    # The output is zero-initialized by the wrapper.
    is_bg = label == 0
    skip = valid & NO_BKG & is_bg

    any_neq = tl.full((BLOCK_SIZE,), False, dtype=tl.int1)

    # Inspect the square neighborhood [-K, K] x [-K, K]
    for kh in range(-K, K + 1):
        for kw in range(-K, K + 1):

            # Match the original corner exclusion:
            # skip only offsets where abs(kh) == 1 and abs(kw) == 1.
            if NO_CORNERS and abs(kh) == 1 and abs(kw) == 1:
                continue

            ph = h + kh
            pw = w + kw

            neighbor_valid = (
                valid
                & (ph >= 0)
                & (ph < H)
                & (pw >= 0)
                & (pw < W)
            )

            neighbor_idx = b * hw + ph * W + pw

            neighbor_label = tl.load(
                maps_ptr + neighbor_idx,
                mask=neighbor_valid,
                other=0,
            )

            is_neq = neighbor_label != label
            any_neq = any_neq | (neighbor_valid & is_neq)

    # Match the original behavior: background pixels skipped by NO_BKG
    # retain their initial False output value.
    result = any_neq & valid & ~skip

    tl.store(new_labels_ptr + pixels, result, mask=valid)


def get_boundary(
    maps: torch.Tensor,
    K: int = 1,
    no_corners: bool = False,
    no_bkg: bool = False,
) -> torch.Tensor:
    """
    Args:
        maps: CUDA tensor of shape (B, H, W), containing integer labels.
        K: Neighborhood radius.
        no_corners: Skip offsets with abs(kh) == 1 and abs(kw) == 1.
        no_bkg: Leave pixels with label 0 as False.

    Returns:
        Boolean CUDA tensor of shape (B, H, W).
    """
    ndim = maps.ndim
    if maps.ndim == 2: maps = maps[None,:]
    assert maps.is_cuda, "maps must be on CUDA"
    assert maps.ndim == 3, "maps must have shape (B, H, W)"
    assert maps.dtype == torch.int32, "maps must have dtype torch.int32"
    assert K >= 0, "K must be non-negative"

    B, H, W = maps.shape
    N = B * H * W

    labels = torch.zeros(
        (B, H, W),
        dtype=torch.bool,
        device=maps.device,
    )

    if N == 0:
        return labels

    BLOCK_SIZE = 128
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    get_boundary_kernel[grid](
        labels,
        maps,
        H,
        W,
        N,
        K,
        no_corners,
        no_bkg,
        BLOCK_SIZE,
    )

    if ndim == 2: labels = labels[0]
    return labels
