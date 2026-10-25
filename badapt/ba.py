
import math
import torch as th
import torch.nn.functional as F

from badapt.ba_triton import get_boundary

# Algorithm 1
@th.no_grad()
def boundary_adaptation(bsam,image,coarse,potts_coeff=0.0,steps=None,scales=None):

    if steps is None: steps = [12,9,3]
    if scales is None: scales = [4,2,1]
    def interp(mask,_H,_W,mode='nearest',thresh=True):
        kw = {"align_corners":False} if mode == 'bilinear' else {}
        mask = F.interpolate(mask.float()[None,None], size=(_H,_W), mode=mode, **kw)[0,0]
        if thresh: return (mask > 0.5).int()
        else: return mask

    def unwrap(bsam):
        return bsam.module if hasattr(bsam, 'module') else bsam

    device = image.device
    H, W, C = image.shape
    HW = (H,W)
    bsam = bsam.eval()
    bsam.to(device)
    bsam.sam['sam'].to(device)
    bsam = unwrap(bsam)
    bsam.reset()
    iftrs,sftrs,context = bsam.get_global_features(image,coarse)

    # -- [dev only]; run hqsam-refiner from here --
    # refined = (th.sum(sftrs*context[:,:,None,None],dim=1)>0).int()[0]
    # refined = interp(refined,1024,1024,'bilinear',thresh=False)
    # s = 1024/max(H,W)    
    # sH,sW = int(H*s),int(W*s)
    # refined = refined[...,:sH,:sW]
    # refined = interp(refined,H,W,'bilinear')
    # return refined

    # -- run refinement loop --
    refined = coarse.clone()
    for scale, nsteps in zip(scales, steps):

        H_s, W_s = H // scale, W // scale
        coarse_s = interp(coarse,H_s,W_s,'bilinear')
        refined = interp(refined,H_s,W_s,'bilinear')
        iftrs_s = bsam.iftr_net_ms(iftrs,scale)

        for step in range(nsteps):

            # get boundary points
            bndy = get_boundary(refined)
            query  = scale*th.stack(th.where(bndy)).view(-1,2)+(scale-1)/2.0
            if query.shape[0] == 0: continue

            # log[p(x|z_i=1,..)/p(x|z_i=0,...)] \approx BSAM(...)
            mftrs  = bsam.mftr_net(coarse_s, refined, scale, (H,W) )
            logits = bsam(query, iftrs_s, sftrs, mftrs, context, (H,W) )

            # log[p(z_i=1|...)/p(z_i=0|...)] \approx BSAM(...) + log-Potts-ratio(...)
            refined = ba_step(logits,refined,bndy,potts_coeff)

    # -- ensure native-resolution --
    refined = interp(refined,H,W)
    
    return refined

def ba_step(logits, mask, boundary, potts_coeff):

    # updating new mask
    mask = mask.clone() # contiguous() implied?
    bcoords = th.stack(th.where(boundary),dim=-1).int()

    # Checkerboard coloring to avoid conflicting updates
    color = (bcoords[:, 0] % 2) + 2 * (bcoords[:, 1] % 2)
    assert th.all(th.isin(color, th.tensor([0, 1, 2, 3], device=color.device)))

    for i in range(4):

        # -- select color & skip empty --
        sel_i = color == i
        if th.sum(sel_i) == 0: continue

        bcoords_i = bcoords[sel_i]
        # is_chng = sclr.sclr_cuda.check_is_changable(mask, bcoords_i.contiguous(), 1)[0].bool()
        is_chng = th.ones_like(bcoords_i[:,0]).bool()
        bcoords_i = bcoords_i[is_chng]
        logits_i = logits[sel_i][is_chng]

        # Optional Potts Term
        if potts_coeff > 0:
            potts = sclr.sclr_cuda.count_neq(mask,bcoords_i.int().contiguous(),1)
            logits_i = logits_i - potts_coeff*(potts[:,1] - potts[:,0])

        # Compute new labels
        mask[bcoords_i[:, 0], bcoords_i[:, 1]] = (logits_i > 0.0).int()

    return mask


def oracle_ba_step():
    pass


