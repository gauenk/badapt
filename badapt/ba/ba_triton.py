"""

   Triton for CUDA-Kernels for Easier Sharing

"""


import torch as th
import triton
import cppyy
import triton.language as tl
DEVICE = triton.runtime.driver.active.get_active_torch_device()

# list of functions:
# get_boundary
# check_is_changeable
# log_potts_ratio
# remove_artifacts; find_strands, find_islands


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
    # for kh in range(-K, K + 1):
    #     for kw in range(-K, K + 1):
    for kh in tl.static_range(-K, K+1):
        for kw in tl.static_range(-K, K+1):

            # Match the original corner exclusion:
            # skip only offsets where abs(kh) == 1 and abs(kw) == 1.
            # if NO_CORNERS and tl.abs(kh) == 1 and tl.abs(kw) == 1:
            #     continue

            invalid_d = NO_CORNERS & (tl.abs(kh) == 1) & (tl.abs(kw) == 1)

            ph = h + kh
            pw = w + kw

            neighbor_valid = (
                valid
                & (~invalid_d)
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
    maps: th.Tensor,
    K: int = 1,
    no_corners: bool = True,
    no_bkg: bool = False,
) -> th.Tensor:
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
    assert maps.dtype == th.int32, "maps must have dtype th.int32"
    assert K >= 0, "K must be non-negative"

    B, H, W = maps.shape
    N = B * H * W

    labels = th.zeros(
        (B, H, W),
        dtype=th.bool,
        device=maps.device,
    )

    if N == 0:
        return labels

    BLOCK_SIZE = 512
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
        num_warps=BLOCK_SIZE//32,
    )

    if ndim == 2: labels = labels[0]
    return labels




# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#
#                   Check is Changable:
#      Preserve Topology via Indicator in Potts Term
#
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


BASS_LUT_VALUES = [
    0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0,
    1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1,
    0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0,
    0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0,
    0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0,
    0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0
]

@triton.jit
def is_changeable_by_neighbors(n0, n1, n2, n3, n4, n5, n6, n7, label):

    a0 = (n0 == label) | (n0 == -1)
    a1 = (n1 == label) | (n1 == -1)
    a2 = (n2 == label) | (n2 == -1)
    a3 = (n3 == label) | (n3 == -1)
    a4 = (n4 == label) | (n4 == -1)
    a5 = (n5 == label) | (n5 == -1)
    a6 = (n6 == label) | (n6 == -1)
    a7 = (n7 == label) | (n7 == -1)

    # a0 = (n0 == label)
    # a1 = (n1 == label)
    # a2 = (n2 == label)
    # a3 = (n3 == label)
    # a4 = (n4 == label)
    # a5 = (n5 == label)
    # a6 = (n6 == label)
    # a7 = (n7 == label)

    num = (
        a0.to(tl.int32) * 128
        + a1.to(tl.int32) * 64
        + a2.to(tl.int32) * 32
        + a3.to(tl.int32) * 16
        + a4.to(tl.int32) * 8
        + a5.to(tl.int32) * 4
        + a6.to(tl.int32) * 2
        + a7.to(tl.int32)
    )

    return num

@triton.jit
def check_is_changeable_kernel(
    ischng_ptr,
    labels_ptr,
    query_ptr,
    BND: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    bid = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = bid < BND

    # Boundary coordinates: (height, width).
    h = tl.load(query_ptr + 2 * bid, mask=valid, other=0)
    w = tl.load(query_ptr + 2 * bid + 1, mask=valid, other=0)
    center_idx = h * W + w
    label = tl.load(labels_ptr + center_idx, mask=valid, other=0)

    # Eight neighbors, in the original C++ order:
    # NW, N, NE, W, E, SW, S, SE.
    n0 = tl.load(
        labels_ptr + center_idx - W - 1,
        mask=valid & (h > 0) & (w > 0),
        other=-1,
    )
    n1 = tl.load(
        labels_ptr + center_idx - W,
        mask=valid & (h > 0),
        other=-1,
    )
    n2 = tl.load(
        labels_ptr + center_idx - W + 1,
        mask=valid & (h > 0) & (w < W - 1),
        other=-1,
    )
    n3 = tl.load(
        labels_ptr + center_idx - 1,
        mask=valid & (w > 0),
        other=-1,
    )
    n4 = tl.load(
        labels_ptr + center_idx + 1,
        mask=valid & (w < W - 1),
        other=-1,
    )
    n5 = tl.load(
        labels_ptr + center_idx + W - 1,
        mask=valid & (h < H - 1) & (w > 0),
        other=-1,
    )
    n6 = tl.load(
        labels_ptr + center_idx + W,
        mask=valid & (h < H - 1),
        other=-1,
    )
    n7 = tl.load(
        labels_ptr + center_idx + W + 1,
        mask=valid & (h < H - 1) & (w < W - 1),
        other=-1,
    )

    # Each valid candidate (0 or 1) must pass the LUT test.
    sum_0 = is_changeable_by_neighbors(
        n0, n1, n2, n3, n4, n5, n6, n7, 0
    )
    sum_1 = is_changeable_by_neighbors(
        n0, n1, n2, n3, n4, n5, n6, n7, 1
    )
6
    # only update if faces include candidate point
    any_zeros = (n1 == 0) | (n3 == 0) | (n4 == 0) | (n6 == 0)
    sum_0 = tl.where(any_zeros,sum_0,255)

    any_ones = (n1 == 1) | (n3 == 1) | (n4 == 1) | (n6 == 1)
    sum_1 = tl.where(any_ones,sum_1,255)

    # Skip candidate neighbors whose label is -1.
    # result = check_0 & check_1
    tl.store(ischng_ptr + 2*bid + 0, sum_0.to(tl.int32), mask=valid)
    tl.store(ischng_ptr + 2*bid + 1, sum_1.to(tl.int32), mask=valid)

def check_is_changeable(
    labels: th.Tensor,
    query: th.Tensor,
) -> th.Tensor:

    assert labels.is_cuda
    assert labels.dtype == th.int32
    assert labels.ndim == 2
    assert labels.is_contiguous()

    assert query.is_cuda
    assert query.dtype == th.int32
    assert query.ndim == 2
    assert query.shape[1] == 2
    assert query.is_contiguous()
    assert query.device == labels.device

    H, W = labels.shape
    BND = query.shape[0]
    if BND == 0: return th.zeros(BND,dtype=th.int32,device=labels.device)

    ischng = th.empty((BND,2),dtype=th.int32,device=labels.device)

    block_size = 512

    check_is_changeable_kernel[
        (triton.cdiv(BND, block_size),)
    ](
        ischng,
        labels,
        query,
        BND,
        H,
        W,
        block_size,
        num_warps=block_size//32,
    )

    lut = th.tensor(
        BASS_LUT_VALUES,
        dtype=th.bool,
        device=labels.device,
    )

    ql = labels[query[:,0],query[:,1]]
    iso_0 = (ischng[:,0]==255) & (ql == 1)
    iso_1 = (ischng[:,1]==255) & (ql == 0)
    any_isolated = th.any(ischng[:,0]==255,dim=-1)
    ischng = th.logical_and(lut[ischng[:,0]],lut[ischng[:,1]])
    ischng = th.logical_or(ischng,any_isolated)

    return ischng.bool()

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#
#                Difference of Neq Neighbors:
#       Regularize Boundaries via Exp. in Potts Term
#
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

@triton.jit
def log_potts_ratio_kernel(
    lpotts_ptr,
    spix_ptr,
    query_ptr,
    BND: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):

    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = idx < BND
    h = tl.load(query_ptr + 2 * idx , mask=valid, other=0)
    w = tl.load(query_ptr + 2 * idx + 1, mask=valid, other=0)
    center_idx = h * W + w
    center = tl.load(spix_ptr + center_idx, mask=valid, other=0)

    count = tl.full((BLOCK_SIZE,), 0, dtype=tl.int32)

    # K = 1: scan the 8 neighbors.
    for dh in tl.static_range(-1, 1+1):
        for dw in tl.static_range(-1, 1+1):
            not_center = dh != 0 or dw != 0

            hn = h + dh
            wn = w + dw

            in_bounds = (
                valid
                & (hn >= 0)
                & (hn < H)
                & (wn >= 0)
                & (wn < W)
            )

            neighbor_idx = hn * W + wn

            neighbor = tl.load(
                spix_ptr + neighbor_idx,
                mask=in_bounds & not_center,
                other=0,
            )

            # Count neighbors whose label differs from center.
            count += tl.where(in_bounds & not_center,1-2*neighbor,0)
            # if in_bounds & not_center:
            #     count += 1 - 2*neighbor

    tl.store(lpotts_ptr + idx, count, mask=valid)

def log_potts_ratio(
    spix: th.Tensor,
    query: th.Tensor,
    K: int = 1,
) -> th.Tensor:
    assert spix.is_cuda
    assert spix.dtype == th.int32
    assert spix.ndim == 2
    assert spix.is_contiguous()

    assert query.is_cuda
    assert query.dtype == th.int32
    assert query.ndim == 2
    assert query.shape[1] == 2
    assert query.is_contiguous()
    assert query.device == spix.device
    assert K == 1

    # -- also conv2d, but must be applied to entire image so not really in the "spirit" of sparse refinement --
    # H,W = spix.shape
    # ins = th.stack([spix,1-spix],dim=0)[None,:].float()
    # weight = th.ones((2,1,3,3)).to(spix.device)
    # weight[:,:,1,1] = 0
    # outs = th.nn.functional.conv2d(ins.float(),weight,padding='same',groups=2)[0]
    # lpotts = (outs[1] - outs[0]).ravel()
    # idx = query[:,0]*W + query[:,1]
    # return lpotts[idx]
    # exit()

    H, W = spix.shape
    BND = query.shape[0]

    lpotts = th.empty(
        (BND,),
        dtype=th.int32,
        device=spix.device,
    )

    if BND == 0:
        return lpotts

    block_size = 512

    log_potts_ratio_kernel[(triton.cdiv(BND, block_size),)](
        lpotts,
        spix,
        query,
        BND,
        H,
        W,
        block_size,
        num_warps=block_size//32,
    )

    return lpotts


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#
#          Manually Remove Artifacts During BA:
#      The Potts Term is "too good", and almost always
#      we want merging when separation is just one pixel.
#
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


@triton.jit
def find_islands_kernel(
    bmask_ptr,
    islands_ptr,
    H: tl.constexpr,
    W: tl.constexpr,
    SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid_idx = idx < SIZE

    # Unpack flattened B x H x W coordinates.
    h_i = idx // W
    w_i = idx % W

    curr = tl.load(bmask_ptr + idx, mask=valid_idx, other=0)

    is_island = tl.full((BLOCK_SIZE,), False, dtype=tl.int1)

    # Scan square perimeters at radii 1 through 5.
    for level in tl.static_range(1, 6):
        count = tl.full((BLOCK_SIZE,), 0, dtype=tl.int32)

        # for dh in tl.static_range(-5,5+1):
        #     for dw in tl.static_range(-5,5+1):
        for dh in tl.static_range(-level, level+1):
            for dw in tl.static_range(-level, level+1):

                # Only visit the current square's perimeter.
                on_perimeter = (
                    (tl.abs(dh) == level)
                    | (tl.abs(dw) == level)
                )

                # Ignore offsets outside this perimeter.
                if on_perimeter:

                    h_prop = h_i + dh
                    w_prop = w_i + dw
                    neigh_idx = h_prop * W + w_prop

                    in_bounds = (
                        valid_idx
                        & (h_prop >= 0)
                        & (h_prop < H)
                        & (w_prop >= 0)
                        & (w_prop < W)
                    )

                    neigh = tl.load(
                        bmask_ptr + neigh_idx,
                        mask=in_bounds,
                        other=-1,
                    )

                    same = in_bounds & on_perimeter & (neigh == curr)
                    count += same.to(tl.int32)

        # Check if the current level is an island; no one connect out 
        level_is_island = (
            (count <= 1)
            | ((count <= 2) & (level > 1))
        )

        is_island = is_island | level_is_island

    tl.store(
        islands_ptr + idx,
        is_island,
        mask=valid_idx
    )


def find_islands(bmask: th.Tensor) -> th.Tensor:

    assert bmask.is_cuda
    assert bmask.ndim == 2
    assert bmask.dtype == th.int32
    assert bmask.is_contiguous()

    H, W = bmask.shape
    size = H * W

    islands = th.zeros(
        (H, W),
        device=bmask.device,
        dtype=th.bool,
    )

    block_size = 512

    find_islands_kernel[(triton.cdiv(size, block_size),)](
        bmask,
        islands,
        H,
        W,
        size,
        block_size,
        num_warps=block_size//32,
    )

    return islands

@triton.jit
def find_strands_kernel(
    bmask_ptr,
    strands_ptr,
    H: tl.constexpr,
    W: tl.constexpr,
    SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid_idx = idx < SIZE

    # HW = H * W
    # b_i = idx // HW
    # spatial_idx = idx % HW
    h = idx // W
    w = idx % W

    curr = tl.load(bmask_ptr + idx, mask=valid_idx, other=0)

    # Neighbor order:
    # 0 1 2
    # 3   4
    # 5 6 7

    # Each out-of-bounds neighbor is represented by -1.
    h0 = h - 1
    h1 = h
    h2 = h + 1
    w0 = w - 1
    w1 = w
    w2 = w + 1

    # NW
    v0 = valid_idx & (h0 >= 0) & (w0 >= 0)
    n0 = tl.load(
        bmask_ptr + h0 * W + w0,
        mask=v0, other=-1,
    )

    # N
    v1 = valid_idx & (h0 >= 0)
    n1 = tl.load(
        bmask_ptr +  h0 * W + w1,
        mask=v1, other=-1,
    )

    # NE
    v2 = valid_idx & (h0 >= 0) & (w2 < W)
    n2 = tl.load(
        bmask_ptr +  h0 * W + w2,
        mask=v2, other=-1,
    )

    # W
    v3 = valid_idx & (w0 >= 0)
    n3 = tl.load(
        bmask_ptr +  h1 * W + w0,
        mask=v3, other=-1,
    )

    # E
    v4 = valid_idx & (w2 < W)
    n4 = tl.load(
        bmask_ptr +  h1 * W + w2,
        mask=v4, other=-1,
    )

    # SW
    v5 = valid_idx & (h2 < H) & (w0 >= 0)
    n5 = tl.load(
        bmask_ptr +  h2 * W + w0,
        mask=v5, other=-1,
    )

    # S
    v6 = valid_idx & (h2 < H)
    n6 = tl.load(
        bmask_ptr +  h2 * W + w1,
        mask=v6, other=-1,
    )

    # SE
    v7 = valid_idx & (h2 < H) & (w2 < W)
    n7 = tl.load(
        bmask_ptr +  h2 * W + w2,
        mask=v7, other=-1,
    )

    # set_mask_v1:
    # A neighbor is active if it has the current label
    # or if it lies outside the image.
    a0 = (n0 == curr) | (n0 == -1)
    a1 = (n1 == curr) | (n1 == -1)
    a2 = (n2 == curr) | (n2 == -1)
    a3 = (n3 == curr) | (n3 == -1)
    a4 = (n4 == curr) | (n4 == -1)
    a5 = (n5 == curr) | (n5 == -1)
    a6 = (n6 == curr) | (n6 == -1)
    a7 = (n7 == curr) | (n7 == -1)

    # Exact pair conditions from is_strand_b.
    pair_found = (
        (a0 & a2) | (a0 & a4) | (a0 & a5) |
        (a0 & a6) | (a0 & a7) |

        (a1 & a3) | (a1 & a4) | (a1 & a5) |
        (a1 & a6) | (a1 & a7) |

        (a2 & a3) | (a2 & a5) | (a2 & a6) |
        (a2 & a7) |

        (a3 & a4) | (a3 & a6) | (a3 & a7) |

        (a4 & a5) | (a4 & a6) |

        (a5 & a7)
    )

    # Count active neighbors, exactly as the C++ loop does.
    count = (
        a0.to(tl.int32) +
        a1.to(tl.int32) +
        a2.to(tl.int32) +
        a3.to(tl.int32) +
        a4.to(tl.int32) +
        a5.to(tl.int32) +
        a6.to(tl.int32) +
        a7.to(tl.int32)
    )

    result = pair_found & (count <= 2)

    tl.store(strands_ptr + idx, result, mask=valid_idx)


def find_strands(bmask: th.Tensor) -> th.Tensor:
    assert bmask.is_cuda
    assert bmask.ndim == 2
    assert bmask.dtype == th.int32
    assert bmask.is_contiguous()

    H, W = bmask.shape
    size = H * W

    strands = th.empty(
        (H, W),
        device=bmask.device,
        dtype=th.bool,
    )

    block_size = 512

    find_strands_kernel[(triton.cdiv(size, block_size),)](
        bmask,
        strands,
        H,
        W,
        size,
        block_size,
        num_warps=block_size//32,
    )

    return strands



