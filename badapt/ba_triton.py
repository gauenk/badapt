"""

   Triton for CUDA-Kernels for Easier Sharing

"""


import torch
import triton
import triton.language as tl
DEVICE = triton.runtime.driver.active.get_active_torch_device()

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




# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#
#                   Check is Changable:
#      Preserve Topology via Indicator in Potts Term
#
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


BASS_LUT_VALUES = [
    0,0,1,1,0,0,1,1,1,1,0,1,0,0,0,1,1,0,0,0,1,0,1,1,
    0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,1,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,1,0,1,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,0,1,1,0,1,0,0,0,1,1,0,0,0,
    1,0,1,1,1,1,0,0,1,1,0,0,
]

@triton.jit
def is_changeable_by_neighbors(
        n0, n1, n2, n3, n4, n5, n6, n7,
        label, lut_ptr):

    a0 = (n0 == label) | (n0 == -1)
    a1 = (n1 == label) | (n1 == -1)
    a2 = (n2 == label) | (n2 == -1)
    a3 = (n3 == label) | (n3 == -1)
    a4 = (n4 == label) | (n4 == -1)
    a5 = (n5 == label) | (n5 == -1)
    a6 = (n6 == label) | (n6 == -1)
    a7 = (n7 == label) | (n7 == -1)

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

    return tl.load(lut_ptr + num)


@triton.jit
def check_is_changeable_kernel(
    ischng_ptr,
    labels_ptr,
    bcoords_ptr,
    lut_ptr,
    BND: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    bid = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = bid < BND

    # Boundary coordinates: (batch, height, width).
    # b = tl.load(bcoords_ptr + 3 * bid, mask=valid, other=0)
    # h = tl.load(bcoords_ptr + 3 * bid + 1, mask=valid, other=0)
    # w = tl.load(bcoords_ptr + 3 * bid + 2, mask=valid, other=0)
    # center_idx = b * H*W + h * W + w

    h = tl.load(bcoords_ptr + 2 * bid, mask=valid, other=0)
    w = tl.load(bcoords_ptr + 2 * bid + 1, mask=valid, other=0)
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

    # Original my_set = {1, 3, 4, 6}: N, W, E, S.
    # Each valid candidate must pass the LUT test.
    check_n = is_changeable_by_neighbors(
        n0, n1, n2, n3, n4, n5, n6, n7, n1, lut_ptr
    )
    check_w = is_changeable_by_neighbors(
        n0, n1, n2, n3, n4, n5, n6, n7, n3, lut_ptr
    )
    check_e = is_changeable_by_neighbors(
        n0, n1, n2, n3, n4, n5, n6, n7, n4, lut_ptr
    )
    check_s = is_changeable_by_neighbors(
        n0, n1, n2, n3, n4, n5, n6, n7, n6, lut_ptr
    )

    # Skip candidate neighbors whose label is -1.
    result = (
        ((n1 == -1) | check_n)
        & ((n3 == -1) | check_w)
        & ((n4 == -1) | check_e)
        & ((n6 == -1) | check_s)
    )

    tl.store(ischng_ptr + bid, result.to(tl.int1), mask=valid)


def check_is_changeable(
    labels: torch.Tensor,
    bcoords: torch.Tensor,
) -> torch.Tensor:

    assert labels.is_cuda
    assert labels.dtype == torch.int32
    assert labels.ndim == 2
    assert labels.is_contiguous()

    assert bcoords.is_cuda
    assert bcoords.dtype == torch.int32
    assert bcoords.ndim == 2
    assert bcoords.shape[1] == 2
    assert bcoords.is_contiguous()
    assert bcoords.device == labels.device

    H, W = labels.shape
    BND = bcoords.shape[0]


    ischng = torch.empty(
        (BND,),
        dtype=torch.bool,
        device=labels.device,
    )

    if BND == 0: return ischng


    lut = torch.tensor(
        BASS_LUT_VALUES,
        dtype=torch.int32,
        device=labels.device,
    )

    block_size = 128

    check_is_changeable_kernel[
        (triton.cdiv(BND, block_size),)
    ](
        ischng,
        labels,
        bcoords,
        lut,
        BND,
        H,
        W,
        block_size,
        num_warps=4,
    )

    return ischng


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#
#                Difference of Neq Neighbors:
#       Regularize Boundaries via Exp. in Potts Term
#
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

@triton.jit
def potts_ratio_kernel(
    lpotts_ptr,
    spix_ptr,
    bcoords_ptr,
    BND: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = idx < BND

    # Boundary coordinates: (batch, height, width).
    # b = tl.load(bcoords_ptr + 3 * idx, mask=valid, other=0)
    # h = tl.load(bcoords_ptr + 3 * idx + 1, mask=valid, other=0)
    # w = tl.load(bcoords_ptr + 3 * idx + 2, mask=valid, other=0)
    # center_idx = b * H*W + h * W + w

    h = tl.load(bcoords_ptr + 2 * idx , mask=valid, other=0)
    w = tl.load(bcoords_ptr + 2 * idx + 1, mask=valid, other=0)
    center_idx = h * W + w
    center = tl.load(spix_ptr + center_idx, mask=valid, other=0)

    count = tl.full((BLOCK_SIZE,), 0, dtype=tl.int32)

    # K = 1: scan the 8 neighbors.
    for dh in tl.static_range(-1, 2):
        for dw in tl.static_range(-1, 2):
            if dh != 0 or dw != 0:
                hn = h + dh
                wn = w + dw

                in_bounds = (
                    valid
                    & (hn >= 0)
                    & (hn < H)
                    & (wn >= 0)
                    & (wn < W)
                )

                neighbor_idx = b * HW + hn * W + wn

                neighbor = tl.load(
                    spix_ptr + neighbor_idx,
                    mask=in_bounds,
                    other=0,
                )

                # Count neighbors whose label differs from center.
                count += (
                    in_bounds & (1 - 2*neighbor)
                ).to(tl.int32)

    tl.store(lpotts_ptr + idx, count, mask=valid)

def log_potts_ratio(
    spix: torch.Tensor,
    bcoords: torch.Tensor,
    K: int = 1,
) -> torch.Tensor:
    assert spix.is_cuda
    assert spix.dtype == torch.int32
    assert spix.ndim == 2
    assert spix.is_contiguous()

    assert bcoords.is_cuda
    assert bcoords.dtype == torch.int32
    assert bcoords.ndim == 2
    assert bcoords.shape[1] == 2
    assert bcoords.is_contiguous()
    assert bcoords.device == spix.device
    assert K == 1

    B, H, W = spix.shape
    BND = bcoords.shape[0]

    lpotts = torch.empty(
        (BND,),
        dtype=torch.int32,
        device=spix.device,
    )

    if BND == 0:
        return lpotts

    block_size = 128

    count_neq_kernel[(triton.cdiv(BND, block_size),)](
        lpotts,
        spix,
        bcoords,
        BND,
        H,
        W,
        block_size,
        num_warps=4,
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


def remove_artifacts(mask):

    strands = find_strands(mask)
    mask[strands] = 1 - mask[strands]
        
    islands = find_islands(mask)
    mask[islands] = 1 - mask[islands]

    return mask


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

        for dh in tl.static_range(-5, 6):
            for dw in tl.static_range(-5, 6):

                # Only visit the current square's perimeter.
                on_perimeter = (
                    (tl.abs(dh) == level)
                    | (tl.abs(dw) == level)
                )

                # Ignore offsets outside this perimeter.
                if on_perimeter:
                    h_prop = h_i + dh
                    w_prop = w_i + dw

                    in_bounds = (
                        valid_idx
                        & (h_prop >= 0)
                        & (h_prop < H)
                        & (w_prop >= 0)
                        & (w_prop < W)
                    )

                    neigh_idx = (
                         h_prop * W + w_prop
                    )

                    neigh = tl.load(
                        bmask_ptr + neigh_idx,
                        mask=in_bounds,
                        other=0,
                    )

                    same = (
                        in_bounds & (neigh == curr)
                    )

                    count += same.to(tl.int32)

        # Preserve the original C++ condition.
        level_is_island = (
            (count <= 1)
            | ((count <= 2) & (level > 1))
        )

        is_island = is_island | level_is_island

    tl.store(
        islands_ptr + idx,
        is_island,
        mask=valid_idx,
    )


def find_islands(bmask: torch.Tensor) -> torch.Tensor:
    assert bmask.is_cuda
    assert bmask.ndim == 2
    assert bmask.dtype == torch.int32
    assert bmask.is_contiguous()

    H, W = bmask.shape
    size = H * W

    islands = torch.empty(
        (H, W),
        device=bmask.device,
        dtype=torch.bool,
    )

    block_size = 128

    find_islands_kernel[(triton.cdiv(size, block_size),)](
        bmask,
        islands,
        H,
        W,
        size,
        block_size,
        num_warps=4,
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


def find_strands(bmask: torch.Tensor) -> torch.Tensor:
    assert bmask.is_cuda
    assert bmask.ndim == 2
    assert bmask.dtype == torch.int32
    assert bmask.is_contiguous()

    H, W = bmask.shape
    size = H * W

    strands = torch.empty(
        (H, W),
        device=bmask.device,
        dtype=torch.bool,
    )

    block_size = 128

    find_strands_kernel[(triton.cdiv(size, block_size),)](
        bmask,
        strands,
        H,
        W,
        size,
        block_size,
        num_warps=4,
    )

    return strands

















































# __device__ inline
# void set_mask_v1(int NW,int N,int NE,int W,
#               int E,int SW,int S,int SE,
#               int label, bool* nbrs){
#   nbrs[0] = (label == NW) || (NW==-1);
#   nbrs[1] = (label == N)  || (N ==-1);
#   nbrs[2] = (label == NE) || (NE==-1);
#   nbrs[3] = (label == W)  || (W ==-1);
#   nbrs[4] = (label == E)  || (E ==-1);
#   nbrs[5] = (label == SW) || (SW==-1);
#   nbrs[6] = (label == S)  || (S ==-1);
#   nbrs[7] = (label == SE) || (SE==-1);
#   return;
# }


# __device__ inline bool is_strand(bool* nbrs) {
#     // Opposite neighbor pairs (8-connected, center excluded):
#     // Layout:  0 1 2
#     //          3   4
#     //          5 6 7

#     // bool pair_diag_tl_br = nbrs[0] && nbrs[7]; // top-left  <-> bottom-right
#     // bool pair_vert       = nbrs[1] && nbrs[6]; // top       <-> bottom
#     // bool pair_diag_tr_bl = nbrs[2] && nbrs[5]; // top-right <-> bottom-left
#     // bool pair_horiz      = nbrs[3] && nbrs[4]; // left      <-> right
#     // bool pair_horiz_off0 = nbrs[3] && nbrs[2];
#     // bool pair_horiz_off1 = nbrs[3] && nbrs[7];
#     // bool pair_horiz_off2 = nbrs[4] && nbrs[0];
#     // bool pair_horiz_off3 = nbrs[4] && nbrs[5];

#     // "3" and any that are NOT {0,5}
#     // ANY pair that are not adjacent or on the same axis...
#     /********
#              { 0: {1,2,3,5},
#                1: {0,2},
#                2: {0,1,4,7},
#                3: {0,5},
#                4: {2,7},
#                5: {0,3,6,7}
#                6: {5,7}
#                7: {5,6,2,4},
#              }
#     ***************/

#     bool is_strand_b = false;

#     is_strand_b = is_strand_b or (nbrs[0] & nbrs[2]);
#     is_strand_b = is_strand_b or (nbrs[0] & nbrs[4]);
#     is_strand_b = is_strand_b or (nbrs[0] & nbrs[5]);
#     is_strand_b = is_strand_b or (nbrs[0] & nbrs[6]);
#     is_strand_b = is_strand_b or (nbrs[0] & nbrs[7]);

#     is_strand_b = is_strand_b or (nbrs[1] & nbrs[3]);
#     is_strand_b = is_strand_b or (nbrs[1] & nbrs[4]);
#     is_strand_b = is_strand_b or (nbrs[1] & nbrs[5]);
#     is_strand_b = is_strand_b or (nbrs[1] & nbrs[6]);
#     is_strand_b = is_strand_b or (nbrs[1] & nbrs[7]);

#     is_strand_b = is_strand_b or (nbrs[2] & nbrs[0]);
#     is_strand_b = is_strand_b or (nbrs[2] & nbrs[3]);
#     is_strand_b = is_strand_b or (nbrs[2] & nbrs[5]);
#     is_strand_b = is_strand_b or (nbrs[2] & nbrs[6]);
#     is_strand_b = is_strand_b or (nbrs[2] & nbrs[7]);

#     // Layout:  0 1 2
#     //          3   4
#     //          5 6 7

#     is_strand_b = is_strand_b or (nbrs[3] & nbrs[1]);
#     is_strand_b = is_strand_b or (nbrs[3] & nbrs[2]);
#     is_strand_b = is_strand_b or (nbrs[3] & nbrs[4]);
#     is_strand_b = is_strand_b or (nbrs[3] & nbrs[6]);
#     is_strand_b = is_strand_b or (nbrs[3] & nbrs[7]);

#     is_strand_b = is_strand_b or (nbrs[4] & nbrs[0]);
#     is_strand_b = is_strand_b or (nbrs[4] & nbrs[1]);
#     is_strand_b = is_strand_b or (nbrs[4] & nbrs[3]);
#     is_strand_b = is_strand_b or (nbrs[4] & nbrs[5]);
#     is_strand_b = is_strand_b or (nbrs[4] & nbrs[6]);

#     // Layout:  0 1 2
#     //          3   4
#     //          5 6 7

#     is_strand_b = is_strand_b or (nbrs[5] & nbrs[0]);
#     is_strand_b = is_strand_b or (nbrs[5] & nbrs[1]);
#     is_strand_b = is_strand_b or (nbrs[5] & nbrs[2]);
#     is_strand_b = is_strand_b or (nbrs[5] & nbrs[4]);
#     is_strand_b = is_strand_b or (nbrs[5] & nbrs[7]);

#     is_strand_b = is_strand_b or (nbrs[6] & nbrs[0]);
#     is_strand_b = is_strand_b or (nbrs[6] & nbrs[1]);
#     is_strand_b = is_strand_b or (nbrs[6] & nbrs[2]);
#     is_strand_b = is_strand_b or (nbrs[6] & nbrs[3]);
#     is_strand_b = is_strand_b or (nbrs[6] & nbrs[4]);

#     is_strand_b = is_strand_b or (nbrs[7] & nbrs[0]);
#     is_strand_b = is_strand_b or (nbrs[7] & nbrs[1]);
#     is_strand_b = is_strand_b or (nbrs[7] & nbrs[2]);
#     is_strand_b = is_strand_b or (nbrs[7] & nbrs[3]);
#     is_strand_b = is_strand_b or (nbrs[7] & nbrs[5]);
  
#     // bool is_strand;
#     // skips = {
#     //          {1,2,3,5},
#     //          {0,2},
#     //          {0,1,4,7},
#     //          {0,5},
#     //          {2,7},
#     //          {0,3,6,7},
#     //          {5,7},
#     //          {5,6,2,4}
#     // };
#     // for (int idx = 0; idx < 8; idx++){
#     //   for (int elem=0; elemskips[idx]
#     // }
    
#     // bool pairs0 = nbrs[0] & 

#     // // Count how many neighbors are active (to detect branching)
#     int count = 0;
#     for (int i = 0; i < 8; i++) count += (int)nbrs[i];

#     // // A strand: exactly one line passes through, no extra branches
#     // bool has_line = pair_diag_tl_br || pair_vert || pair_diag_tr_bl || pair_horiz;
#     // return (has_line && count == 2) || (count <= 1);
#     // return (is_strand && count == 2) || (count <= 1);
#     return (is_strand_b && (count <= 2));

# }



# static __global__
# void find_strands_kernel(const int* bmask, bool* strands,
#                          int B, int H, int W){

#   // -- get index --
#   int SIZE = B*H*W;
#   int HW = H*W;
#   int index = blockIdx.x * blockDim.x + threadIdx.x;
#   if (index >= SIZE) return;

#   // -- unpack position --
#   int b_i = index/HW;
#   int iindex = index%HW;
#   int h_i = iindex / W;
#   int w_i = iindex % W;
#   int neighs[8];
#   bool mask[8];
#   int curr_b = bmask[index];
  
#   // -- reading all neighboring spix values --
#   int K = 1;
#   int curr_ix = 0;
#   for(int dh=-K; dh<=K; dh++){
#     for(int dw=-K; dw<=K; dw++){

#       if ((dh==0)&&(dw==0)){ continue; }
#       int h_prop = h_i + dh;
#       int w_prop = w_i + dw;

#       bool invalid_h = (h_prop < 0) or (h_prop>=H);
#       bool invalid_w = (w_prop < 0) or (w_prop>=W);
#       bool invalid = invalid_h or invalid_w;

#       if (invalid){
#         neighs[curr_ix] = -1;
#       }else{
#         neighs[curr_ix] = bmask[b_i*H*W + h_prop*W + w_prop];
#       }
#       curr_ix++;
#     }
#   }

#   // -- mark true if its a member of a strand --
#   set_mask_v1(neighs[0], neighs[1], neighs[2], neighs[3],
#               neighs[4], neighs[5], neighs[6], neighs[7], curr_b, mask);
#   strands[index] = is_strand(mask);
#   // strands[index] = curr_b>0;

# }


# torch::Tensor
# find_strands(torch::Tensor bmask) {

#     // -- check inputs --
#     CHECK_INPUT(bmask);
#     int B = bmask.size(0);
#     int H = bmask.size(1); 
#     int W = bmask.size(2);
#     int SIZE = B*H*W;

#     // -- count edges per boundary point --
#     int nthreads = 512;
#     int nblocks = (SIZE - 1)/nthreads + 1;
#     auto opt = torch::dtype(torch::kBool).device(bmask.device());
#     auto strands = torch::zeros({B,H,W},opt);
#     find_strands_kernel<<<nblocks, nthreads>>>(bmask.data_ptr<int>(),
#                                                strands.data_ptr<bool>(),
#                                                B, H, W);
#     return strands;
# }


# static __global__
# void find_islands_kernel(const int* bmask, bool* islands,
#                          int B, int H, int W){

#   // -- get index --
#   int SIZE = B*H*W;
#   int HW = H*W;
#   int index = blockIdx.x * blockDim.x + threadIdx.x;
#   if (index >= SIZE) return;

#   // -- unpack position --
#   int b_i = index/HW;
#   int iindex = index%HW;
#   int h_i = iindex / W;
#   int w_i = iindex % W;
#   // int neighs[8];
#   // bool mask[8];
#   int curr = bmask[index];
  
#   // -- reading all neighboring spix values --
#   // bool exterior_diff_K = true;
#   // bool exterior_diff_1 = true;
#   // bool exterior_diff_2 = true;
#   // int nsame_inner = 0;
#   // int nsame = 0;
#   int K = 5;

#   // spiral out from center
#   bool is_island = false;
#   int count = 0;
#   for(int level=1; level<=K; level++){

#     count = 0;
#     for(int dh=-level; dh<=level; dh++){
#       for(int dw=-level; dw<=level; dw++){

#         if ((abs(dh)!=level) and (abs(dw)!=level)){ continue; }

#         int h_prop = h_i + dh;
#         int w_prop = w_i + dw;

#         bool invalid_h = (h_prop < 0) or (h_prop>=H);
#         bool invalid_w = (w_prop < 0) or (w_prop>=W);
#         bool invalid = invalid_h or invalid_w;
#         if (invalid){ continue; }
        
#         int neigh = bmask[b_i*H*W + h_prop*W + w_prop];
#         count += (neigh == curr);
#       }
#     }

#     is_island = is_island or ( (count <= 1) or ( (count <= 2) and (level>1) ) );
    
#   }

#   // islands[index] = (nsame < 4) or (nsame_inner <= 1) or exterior_diff; // some small-ish number
#   islands[index] = is_island;


#   // for(int dh=-K; dh<=K; dh++){
#   //   for(int dw=-K; dw<=K; dw++){

#   //     if ((dh==0)&&(dw==0)){ continue; }
#   //     int h_prop = h_i + dh;
#   //     int w_prop = w_i + dw;

#   //     bool invalid_h = (h_prop < 0) or (h_prop>=H);
#   //     bool invalid_w = (w_prop < 0) or (w_prop>=W);
#   //     bool invalid = invalid_h or invalid_w;
#   //     if (invalid){ continue; }

#   //     int neigh = bmask[b_i*H*W + h_prop*W + w_prop];
#   //     nsame += (neigh == curr);

#   //     if ((abs(dh) <= 1) and (abs(dw) <= 1)){
#   //       nsame_inner += (neigh == curr);
#   //     }

#   //     if ((abs(dh) == K) and (abs(dw) == K)){
#   //       exterior_diff_K = exterior_diff_K and (neigh != curr);
#   //     }

#   //     if ((abs(dh) == 1) and (abs(dw) == 1)){
#   //       exterior_diff_1 = exterior_diff_1 and (neigh != curr);
#   //     }

#   //     if ((abs(dh) == 2) and (abs(dw) == 2)){
#   //       exterior_diff_2 = exterior_diff_2 and (neigh != curr);
#   //     }

#   //   }
#   // }


#   // -- mark true if its a member of a strand --
#   // set_mask_v1(neighs[0], neighs[1], neighs[2], neighs[3],
#   //             neighs[4], neighs[5], neighs[6], neighs[7], curr_b, mask);
#   // bool exterior_diff = exterior_diff_K or exterior_diff_1 or exterior_diff_2;
#   // islands[index] = (nsame < 4) or (nsame_inner <= 1) or exterior_diff; // some small-ish number

# }



# torch::Tensor
# find_islands(torch::Tensor bmask) {

#     // -- check inputs --
#     CHECK_INPUT(bmask);
#     int B = bmask.size(0);
#     int H = bmask.size(1); 
#     int W = bmask.size(2);
#     int SIZE = B*H*W;

#     // -- count edges per boundary point --
#     int nthreads = 512;
#     int nblocks = (SIZE - 1)/nthreads + 1;
#     auto opt = torch::dtype(torch::kBool).device(bmask.device());
#     auto islands = torch::zeros({B,H,W},opt);
#     find_islands_kernel<<<nblocks, nthreads>>>(bmask.data_ptr<int>(),
#                                                islands.data_ptr<bool>(),
#                                                B, H, W);
#     return islands;
# }




