"""

  The B-SAM Architecture

  logit_i := log[p(z_i=1|...)/p(z_i=0|...)] \approx BSAM(...) + log-Potts-ratio(...)

"""

import torch as th
from torch import nn

class BSAM(nn.Module):

    def __init__(self):
        super().__init__()
        self.sam = sam


    def forward_at_scale(self, image, coarse, refined, boundary,
                         ms_scale, sem_enc, ftrs0_base=None):
        """
        Single forward pass at one scale/step of the multiscale loop.

        image      : (1, H, W, C)  native resolution
        coarse : (1, H_s, W_s) mask at scaled resolution (H_s = H//ms_scale)
        refined : (1, H_s, W_s) previous step's mask at scaled resolution
                    (pass coarse on the very first step)
        boundary  : (1, H_s, W_s) boundary mask at scaled resolution
        ms_scale  : int            current scale factor
        sem_enc   : dict           output of sem_encode()

        Returns logits at boundary points, shape (N,).
        """

        # -- unpack --
        B, H, W, C = image.shape
        assert B == 1
        device = coarse.device

        # -- unpack --
        encoded       = sem_enc["encoded"]
        early_encoded = sem_enc["early_encoded"]
        sam_image      = sem_enc["sam_image"]
        ftrs1 = sem_enc["ftrs1"]
        hyper = sem_enc["hyper"]

        # ── scale-conditioned low-level features ──────────────────────────
        if ftrs0_base is None:
            ftrs0_base = self.get_ftrs0_base(early_encoded)
        ftrs0 = self.get_ll_ftrs_ms(ftrs0_base, ms_scale)
        mftrs = self.get_mask_ftrs_ms(coarse, refined, ms_scale, H, W)

        # ── coordinate grids (cached in native-pixel space) ───────────────
        fH0, fW0 = ftrs0.shape[-2:]
        fH1, fW1 = ftrs1.shape[-2:]
        mH,  mW  = mftrs.shape[-2:]
        ftrs0_coord, ftrs1_coord, mftrs_coord = self._get_coords(H, W, fH0, fW0, fH1, fW1, mH, mW, device)

        # ── boundary coords in native pixel space ─────────────────────────
        # make_coord_scaled gives native-pixel centers for each scaled pixel
        if boundary.shape[-1] == 2:
            coord = boundary # since its actually points here
        else:
            coord_grid = make_coord_scaled(H, W, ms_scale, device=device)  # (H_s, W_s, 2)
            coord = coord_grid[boundary[0] > 0]                             # (N, 2)

        # ── decode logits ─────────────────────────────────────────────────
        logits = self.logit_decoder(
            ftrs0, ftrs0_coord, ftrs1, ftrs1_coord, mftrs, mftrs_coord, hyper, coord, H, W)

        return logits   # (N,)  — caller places these back into a (H_s, W_s) map

    # ── logit decoder (native-pixel coord space) ──────────────────────────────
    def logit_decoder(self, feat0, feat0_coord, feat1, feat1_coord,
                      featM, featM_coord, hyper, coord, H, W):
        """
        coord : (N, 2) in *native* pixel space.
        feat0 : (1, Fll, fH0, fW0)  low-level features
        feat1 : (1, C1,  fH1, fW1)  SAM decoded features
        All feature grids live in native pixel space; no ms_scale needed here.
        """
        coord_encoder = self.coord_encoder

        # ── mftrs lookup ─────────────────────────────────────────────────────
        mH, mW = featM.shape[-2:]
        rowM = (coord[:, 0] / self.stride0).long().clamp(0, mH - 1)
        colM = (coord[:, 1] / self.stride0).long().clamp(0, mW - 1)
        posM     = featM_coord[rowM, colM]       # (N, 2) native-pixel patch centers
        rel_posM = coord - posM                  # (N, 2) native-pixel offset
        rel_encM = coord_encoder(rel_posM)       # (N, coord_ndim)
        qM_feat  = featM[0, :, rowM, colM].T     # (N, Fll)

        # ── ftrs0 lookup ─────────────────────────────────────────────────────
        fH0, fW0 = feat0.shape[-2:]
        sam_scale = 1024 / max(H, W)
        s0        = 1024 / fH0
        row0 = ((coord[:, 0] * sam_scale) / s0).long().clamp(0, fH0 - 1)
        col0 = ((coord[:, 1] * sam_scale) / s0).long().clamp(0, fW0 - 1)
        pos0     = feat0_coord[row0, col0]       # (N, 2) native-pixel feat centers
        rel_pos0 = coord - pos0                  # (N, 2)
        rel_enc0 = coord_encoder(rel_pos0)       # (N, coord_ndim)
        q0_feat  = feat0[0, :, row0, col0].T     # (N, C1)

        # ── ftrs1 lookup ─────────────────────────────────────────────────────
        fH1, fW1 = feat1.shape[-2:]
        sam_scale = 1024 / max(H, W)
        s1        = 1024 / fH1
        row1 = ((coord[:, 0] * sam_scale) / s1).long().clamp(0, fH1 - 1)
        col1 = ((coord[:, 1] * sam_scale) / s1).long().clamp(0, fW1 - 1)
        pos1     = feat1_coord[row1, col1]       # (N, 2) native-pixel feat centers
        rel_pos1 = coord - pos1                  # (N, 2)
        rel_enc1 = coord_encoder(rel_pos1)       # (N, coord_ndim)
        q1_feat  = feat1[0, :, row1, col1].T     # (N, C1)

        # ── imnet ────────────────────────────────────────────────────────────
        inp0 = torch.cat([q0_feat, rel_enc0], dim=-1)   # (N, Fll+coord_ndim)
        inp1 = torch.cat([q1_feat, rel_enc1], dim=-1)   # (N, C1+coord_ndim)
        inpM = torch.cat([qM_feat, rel_encM], dim=-1)   # (N, Fll+coord_ndim)

        logits = self.imnet(inp0, inp1, inpM, hyper.view(1, -1))  # (N,)
        return logits

    def refine(self,image,coarse):
        get_boundary(coarse)
        pass

    def forward(image,coarse,i):
        logits = 0
        return logit

    def ba(self,image,coarse,potts_coeff=0.0,steps=None,scales=None):
        return boundary_adaptation(self,image,coarse,potts_coeff,steps,scales)


# Algorithm 1
def boundary_adaptation(bsam,image,coarse,potts_coeff=0.0,steps=None,scales=None):

    if steps is None: steps = [12,9,3]
    if scales is None: scales = [4,2,1]
    def interp(mask,_H,_W,mode='nearest'):
        mask = F.interpolate(mask.float()[:,None], size=(_H,_W), mode=mode).squeeze(1)
        return (mask > 0.5).int()
    def unwrap(bsam):
        return bsam.module if hasattr(bsam, 'module') else bsam

    H, W, C = image.shape
    bsam = bsam.eval()
    if hasattr(bsam,'reset'): bsam.reset()
    else: bsam.module.reset()

    bsam = unwrap(bsam)
    iftrs,sftrs = bsam.get_global_features(image,coarse)
    refined = coarse.clone()

    for scale, nsteps in zip(scales, steps):
        H_s, W_s = H // scale, W // scale

        coarse_s = interp(coarse,H_s,W_s,'bilinear')
        refined = interp(refined,H_s,W_s,'bilinear')

        for step in range(nsteps):

            # get boundary
            boundary = get_boundary(refined)
            if th.sum(boundary) == 0: continue

            # log[p(x|z_i=1,..)/p(x|z_i=0,...)] \approx BSAM(...)
            logits = bsam(image, coarse_s, refined, boundary, scale=scale, iftrs=iftrs, sftrs=sftrs)

            # log[p(z_i=1|...)/p(z_i=0|...)] \approx BSAM(...) + log-Potts-ratio(...)
            refined = ba_update(logits,refined,boundary,potts_coeff)

    # -- ensure native-resolution --
    refined = interp(refined,H,W)
    
    return refined

