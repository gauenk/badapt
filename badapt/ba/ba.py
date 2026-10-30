
# -- base --
import math
import torch as th
import torch.nn.functional as F

# -- cuda --
from badapt import _ba_cuda as ba_cuda
get_boundary = ba_cuda.get_boundary
check_is_changeable = ba_cuda.check_is_changeable
log_potts_ratio = ba_cuda.log_potts_ratio
find_islands = ba_cuda.find_islands
find_strands = ba_cuda.find_strands

# import sclr
# def check_is_changeable(mask,query,oob=True,ensure_bndy=False):
#     query = th.cat([th.zeros_like(query[:,[0]]),query],dim=1)
#     is_chng = sclr.sclr_cuda.check_is_changable_v2(
#         mask[None,:], query.contiguous(), 1
#     )[0].bool()
#     # is_chng = sclr.sclr_cuda.check_is_changable(
#     #     mask[None,:], query.contiguous(), 1
#     # )[0].bool()
#     return is_chng

# -- triton [much slower] --
# from badapt.ba import ba_triton
# # get_boundary = ba_triton.get_boundary
# check_is_changeable = ba_triton.check_is_changeable


# Algorithm 1
@th.no_grad()
def boundary_adaptation(bsam,image,coarse,potts_coeff=0.0,steps=None,scales=None):

    # defaults and helpers and such...
    if steps is None: steps = [12,9,3]
    if scales is None: scales = [4,2,1]
    def interp(mask,_H,_W,mode='bilinear',thresh=True):
        if mask.shape == (_H,_W): return mask.contiguous()
        kw = {"align_corners":False} if mode == 'bilinear' else {}
        mask = F.interpolate(mask.float()[None,None], size=(_H,_W), mode=mode, **kw)[0,0]
        if thresh: return (mask > 0.5).int()
        else: return mask
    def unwrap(bsam):
        return bsam.module if hasattr(bsam, 'module') else bsam

    # -- be prepared! --
    device = image.device
    H, W, C = image.shape
    bsam.eval()
    bsam = unwrap(bsam)
    bsam.reset()

    # -- extract base features --
    iftrs_raw,sftrs,context = bsam.get_sam_features(image,coarse)
    iftrs = bsam.iftr_net(iftrs_raw)

    # -- [dev only]; run hqsam-refiner from here --
    # refined = (th.sum(sftrs*context[:,:,None,None],dim=1)>0).int()[0]
    # refined = interp(refined,1024,1024,'bilinear',thresh=False)
    # s = 1024/max(H,W)
    # sH,sW = int(H*s),int(W*s)
    # refined = refined[...,:sH,:sW]
    # refined = interp(refined,H,W,'bilinear')
    # return refined

    # -- run refinement loop --
    refined = coarse.int().clone()
    for scale, nsteps in zip(scales, steps):

        H_s, W_s = H // scale, W // scale
        coarse_s = interp(coarse,H_s,W_s,'nearest')
        refined = interp(refined,H_s,W_s,'nearest')
        iftrs_s = bsam.iftr_net_ms(iftrs,scale)

        for step in range(nsteps):

            # get boundary points
            bndy = get_boundary(refined)
            query  = scale*th.stack(th.where(bndy)).T+(scale-1)/2.0
            if query.shape[0] == 0: continue

            # log[p(x|z_i=1,..)/p(x|z_i=0,...)] \approx BSAM(...)
            mftrs  = bsam.mftr_net(coarse_s, refined, scale, (H,W) )
            logits = bsam(query, iftrs_s, sftrs, mftrs, context, (H,W) )

            # log[p(z_i=1|...)/p(z_i=0|...)] \approx BSAM(...) + log-Potts-ratio(...)
            refined = ba_step(logits,refined,bndy,potts_coeff)

            # manually remove strands and islands
            # remove_artifacts(refined)
            # if scale == 1:
            #     if (step % 5 == 0) or (step==nsteps-1): remove_artifacts(refined)

    # ensure native-resolution
    refined = interp(refined,H,W,'nearest')

    return refined

def ba_step(logits, mask, bndy, potts_coeff):

    # updating new mask
    mask = mask.clone() # contiguous() implied?
    query = th.stack(th.where(bndy),dim=-1).int()

    # Checkerboard coloring to avoid race condition
    color = (query[:, 0] % 2) + 2 * (query[:, 1] % 2)
    assert th.all(th.isin(color, th.tensor([0, 1, 2, 3], device=color.device)))

    for i in range(4):

        # -- select color & skip empty --
        sel_i = color == i
        if th.sum(sel_i) == 0: continue
        query_i = query[sel_i].contiguous()
        # print(i,len(query_i))

        # -- select only changable --
        # is_chng = check_is_changeable(mask, query_i, False, True, False)
        is_chng = check_is_changeable(mask, query_i, True, True, False)
        # is_chng = check_is_changeable(mask, query_i, True, False, True)
        # is_chng = th.ones(query_i.shape[0],dtype=th.bool,device=query_i.device) # dev only
        query_i = query_i[is_chng]
        logits_i = logits[sel_i][is_chng]

        # -- [optional] Boundary Regularization with Potts Term  --
        if potts_coeff > 0:
            lpotts = log_potts_ratio(mask,query_i.contiguous())
            logits_i = logits_i - potts_coeff*lpotts

        # -- write. --
        mask[query_i[:, 0], query_i[:, 1]] = (logits_i > 0.0).int()

    return mask

def remove_artifacts(mask):

    strands = find_strands(mask)
    mask[strands] = 1 - mask[strands]

    islands = find_islands(mask)
    mask[islands] = 1 - mask[islands]

    return mask


def oracle_boundary_adaptation(gt,coarse,potts_coeff=0.0,
                               steps=None,scales=None,return_seq=False):

    # -- prepare --
    seq = [coarse.clone()] if return_seq else None
    if steps is None: steps = [12,9,3]
    if scales is None: scales = [4,2,1]
    def interp(mask,_H,_W,mode='bilinear',thresh=True):
        kw = {"align_corners":False} if mode == 'bilinear' else {}
        mask = F.interpolate(mask.float()[None,None], size=(_H,_W), mode=mode, **kw)[0,0]
        if thresh: return (mask > 0.5).int()
        else: return mask
    device = gt.device
    H, W = gt.shape

    # -- run refinement loop --
    refined = coarse.clone()
    for scale, nsteps in zip(scales, steps):

        H_s, W_s = H // scale, W // scale
        coarse_s = interp(coarse,H_s,W_s,'bilinear')
        refined = interp(refined,H_s,W_s,'bilinear')
        gt_s = interp(gt,H_s,W_s,'bilinear')

        for step in range(nsteps):

            # get boundary points
            bndy = get_boundary(refined)
            query  = scale*th.stack(th.where(bndy)).T+(scale-1)/2.0
            if query.shape[0] == 0: continue

            # log[p(x|z_i=1,..)/p(x|z_i=0,...)] \approx BSAM(...)
            logits = th.logit(gt_s[bndy].clamp(0.01,0.99))

            # log[p(z_i=1|...)/p(z_i=0|...)] \approx BSAM(...) + log-Potts-ratio(...)
            refined = ba_step(logits,refined,bndy,potts_coeff)

            if scale == 1: remove_artifacts(refined)
            if return_seq: seq.append(interp(refined,H,W,'nearest'))

    # -- ensure native-resolution --
    refined = interp(refined,H,W,'nearest')

    # -- return --
    if return_seq: return th.stack(seq)
    else: return refined


def example():
    import os
    import cv2
    import numpy as np
    from badapt.deform import deform_mask

    print("[%d] Running refinement example."%os.getpid())

    gt = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(gt, (128, 128), 60, 255, -1)
    cv2.rectangle(gt, (50, 50), (100, 100), 255, -1)
    gt = th.from_numpy(gt>0).int().cuda()
    coarse = deform_mask(gt)
    seq = oracle_boundary_adaptation(gt,coarse,return_seq=True)

    print("Saved:")
    for ix,mask in enumerate(seq):
        cv2.imwrite("mask_%d.png"%ix, mask.cpu().numpy().astype(np.uint8)*255)
        print("  mask_%d.png"%ix)

if __name__ == "__main__":
    example()


