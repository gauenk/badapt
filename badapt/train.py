

import numpy as np
import torch as th
import pandas as pd

from badapt.deform import deform_mask
from badapt.ba import get_boundary,ba_step

# ---------------------------------------
#
#
#     -=-=-=- Train/Val Loop -=-=-=-=-
#
#
# ---------------------------------------

def launch_training(cfg,bsam,optim,tr_data,val_data):

    # -=-=- init -=-=-=-
    tr_log = {str(s): {"loss":0., "corr":0, "total":0, "count":0} for s in cfg.scales}
    cfg.device   = f"cuda:{cfg.local_rank}"

    # -=-=- training loop -=-=-
    for i in range(cfg.niters_train):

        print(i)
        train_step(cfg,bsam,optim,tr_data,tr_log)

        if i % cfg.val_every == 0:
            val_log = update_val_log([])
            val_step(cfg,bsam,val_data,val_log)
            if cfg.local_rank == 0: print_val_log(val_log)

        if (cfg.global_step % cfg.log_every == 0) and (cfg.local_rank == 0):
            print_train_log(tr_log,cfg.global_step)

        if (cfg.global_step % cfg.ckpt_every == 0) and (cfg.local_rank == 0):
            save_checkpoint(cfg,bsam,optim)

    return

def print_train_log(stats,global_step):

    # -=-=- build logging string -=-=-
    parts = []
    scales = stats.keys()
    for scale in scales:
        key = str(scale)
        a   = stats[key]
        avg_loss = a["loss"] / max(a["count"], 1)
        acc      = a["corr"] / max(a["total"], 1)
        parts.append(f"x{scale}[loss={avg_loss:.3f} acc={acc:.3f}]")

    # -=-=- actually print it -=-=-
    print(f"[step {global_step}]  " + "  ".join(parts))

    # -=-= reset -=-=-
    for s in scales:
        stats[s] = {"loss":0., "corr":0, "total":0, "count":0} 

def save_checkpoint(cfg,bsam,optim):
    ckpt_path = (
        cfg.save_root /
        f"checkpoint_{cfg.dname}_{cfg.model_name}_{cfg.global_step}.pt"
    )
    bsam_state = (
        bsam.module.state_dict() if cfg.use_dist else bsam.state_dict()
    )
    th.save(
        {"step": cfg.global_step, "epoch": cfg.epoch,"model_state": bsam_state},
        ckpt_path,
    )
    print(f"[checkpoint] saved to {ckpt_path}")


# ---------------------------------------
#
#
#       -=-=-=- Training -=-=-=-=-
#
#
# ---------------------------------------

def train_step(cfg,bsam,optim,data,log):

    # -=-=- get (image, mask) pair -=-=-
    batch = next(data, None)
    if batch is None: return True
    image  = batch['image'].cuda(non_blocking=True)
    assert(image.shape[0] == 1) # batch size one for this project
    image = image[0]
    gt = (batch['label'][0].round() > 128.0).int()
    if gt.shape[0] == 0: return False
    assert(gt.shape[0] == 1)
    gt = gt[0]
    H, W, C = image.shape
    device = image.device
 
    # -=-=- deform the mask -=-=-=-
    coarse = deform_mask(gt,cfg.deform_rsr,cfg.deform_sr,cfg.deform_iou)
 
    # -=-=- reset coorindate caches & extract frozen features -=-=-
    # bsam = unwrap(bsam)
    bsam.train()
    if hasattr(bsam, 'reset'): bsam.reset()
    else: bsam.module.reset()
    with th.no_grad():
        iftrs_raw,sftrs,context = bsam.get_sam_features(image,coarse)

    # -=-=- steps/scales -=-=-=-
    total_loss = 0.
    refined = coarse.clone()
    for scale, nsteps in zip(cfg.scales, cfg.steps):

        H_s, W_s = H // scale, W // scale
        coarse_s = interp(coarse,H_s,W_s,'bilinear')
        gt_s = interp(gt,H_s,W_s,'bilinear')
        refined = interp(refined,H_s,W_s,'bilinear')

        for _ in range(nsteps):

            # get pixels along boundary
            bndy,aux = get_training_boundary(refined,aux_points)

            # log[p(x|z_i=1,..)/p(x|z_i=0,...)] \approx BSAM(...)
            query  = scale*th.stack(th.where(aux)).T+(scale-1)/2.0
            iftrs  = bsam.iftr_net_ms(bsam.iftr_net(iftrs_raw),scale)
            mftrs  = bsam.mftr_net(coarse_s, refined, scale, (H,W) )
            logits = bsam(aux, iftrs, sftrs, mftrs, context, (H,W) )

            # update
            with th.no_grad():
                refined = ba_step_tr(logits_B.detach(),bndy,refined,gt_s,scale,cfg.global_step,cfg.potts)

            # loss function (Eq 2)
            logits_B,logits_R = logits[bndy[aux]],logits[~bndy[aux]]

            gt_B = gt_s[bndy].float()
            loss_B = weighted_focal_loss_with_logits(logits_B,gt_B)

            gt_R = gt_s[(aux) & ~(bndy)].float()
            loss_R = weighted_focal_loss_with_logits(logits_R,gt_R)

            lamb = 0.5
            loss = loss_B + lamb*loss_R

            # skip if nonsense...
            if th.isnan(loss):
                print("found a nan.... ",loss_B.item(),loss_R.item())
                break

            # backprop
            optim.zero_grad()
            loss.backward()        # frees this step's graph
            optim.step()
            total_loss += loss.detach()

            # loggin
            with th.no_grad():
                preds_BR      = (logits.detach() > 0).float()
                gt_BR     = gt_s[aux].float()
                key = str(scale)
                log[key]["loss"]  += loss.detach().item()
                log[key]["corr"]  += (preds_BR == gt_BR).sum().item()
                log[key]["total"] += gt_BR.numel()
                log[key]["count"] += 1

    cfg.global_step += 1

def ba_step_tr(logits, bndy, refined, gt_s, scale, global_step, potts):
    """

    Scheduled sampling: probability of using gt decays from 1 → 0 over training.
    Coarser scales decay more slowly.

    """
    warmup_steps = 1000
    scale_factor = {4: 2.0, 2: 1.5, 1: 1.0}.get(scale, 1.0)
    p_teacher = max(0.0, 1.0 - global_step / (warmup_steps * scale_factor))
    p_teacher = max(0.1,p_teacher) # set a floor
    use_gt = (th.rand(1).item() < p_teacher)
    if use_gt:
        new_mask = refined.clone()
        new_mask[0][bndy[0]] = gt_s[0][bndy[0]].float()
    else:
        new_mask = ba_step(logits, refined.int(), bndy, potts).float()
    return new_mask.detach()

def get_training_boundary(mask,aux_points='thicc'):

    def subsample_mask(imask, max_points):
        """Keep at most max_points True values in a bool mask, chosen randomly."""
        idx = th.where(imask[0])[0]  # flat indices of True values
        if len(idx) > max_points:
            perm = th.randperm(len(idx), device=mask.device)[:max_points]
            idx = idx[perm]
        out = th.zeros_like(imask)
        out[0].view(-1)[idx] = True
        return out

    def sample_points_on_mask(mask: th.Tensor, n_points: int, generator: th.Generator = None) -> th.Tensor:
        H, W = mask.shape
        total = H * W
    
        # Ensure we don’t ask for more unique points than available
        n_points = min(n_points, total)
    
        # Sample without replacement
        flat_indices = th.randperm(total, generator=generator, device=mask.device)[:n_points]
    
        rows = flat_indices // W
        cols = flat_indices % W
    
        new_mask = mask.clone().bool()
        new_mask[rows, cols] = True
        return new_mask


    # -- get normal boundary --
    boundary = get_boundary(mask,1)

    # -- include random points --
    aux_boundary = boundary
    assert(boundary.shape[0]==1)
    if aux_points == 'thicc':
        thicc = get_boundary(mask, 4)
        extra = thicc & ~boundary
        extra = subsample_mask(extra, 2000)
        aux_boundary = boundary | extra
        aux_boundary = sample_points_on_mask(aux_boundary[0],1000)[None,:]
    elif aux_points == 'rand':
        aux_boundary = sample_points_on_mask(boundary[0],3000)[None,:]

    return boundary,aux_boundary


# ---------------------------------------
#
#
#       -=-=-=- Validation -=-=-=-=-
#
#
# ---------------------------------------


@th.no_grad()
def val_step(cfg, bsam, data, log):

    # -=-=- get (image, mask) pair -=-=-
    batch = next(data, None)
    if batch is None: return True
    image  = batch['image'].cuda(non_blocking=True)
    gt = (batch['label'][0].round() > 128.0).int()
    if gt.shape[0] == 0: return False
    assert(gt.shape[0] == 1)
    H,W,C = image.shape
    HW = (H,W)
 
    # -=-=- deform the mask -=-=-=-
    coarse = deform_mask(gt,cfg.deform_rsr,cfg.deform_sr,cfg.deform_iou)

    # -=-=- reset coorindate caches & extract base, frozen features -=-=-
    bsam.eval()
    bsam = unwrap(bsam)
    bsam.reset()
    iftrs_raw,sftrs,context = bsam.get_sam_features(image,coarse)
    iftrs = bsam.iftr_net(iftrs_raw)
    refined = coarse.clone()

    # -=-=- record baseline error -=-=-
    d_iou = np.mean(compute_miou(coarse, gt)[0])
    update_val_log(log,d_iou=d_iou)

    # -=-=- Start dat inference baby! -=-=-
    refined = coarse.clone()
    for scale, nsteps in zip(scales, steps):

        # -- standard BA --
        H_s, W_s = H // scale, W // scale
        coarse_s = interp(coarse,H_s,W_s,'bilinear')
        refined = interp(refined,H_s,W_s,'bilinear')
        iftrs_s = bsam.iftr_net_ms(iftrs,scale)

        # -- logging info --
        gt_s = interp(gt,H_s,W_s,'bilinear')
        gt_s_up = interp(gt_s,H,W)
        oracle = refined.clone() # reset upper-bound improvement every scale; just a logging choice

        for step in range(nsteps):

            # -=-=-=- inference -=-=-=-

            # get pixels along boundary
            bndy = get_boundary(refined,1)
            if th.sum(bndy) == 0: continue

            # log[p(x|z_i=1,..)/p(x|z_i=0,...)] \approx BSAM(...)
            query  = scale*th.stack(th.where(bndy)).T+(scale-1)/2.0
            mftrs  = bsam.mask_features(coarse_s, refined, scale, (H,W) )
            logits = bsam(query, iftrs_s, sftrs, mftrs, context, (H,W) )
                
            # -=-=- compute accuracy for 0s and 1s -=-=-=-
            gt_B   = gt_s[aux].float()
            preds   = (logits > 0).float()
            n0      = (gt_B == 0).sum().item()
            n1      = (gt_B == 1).sum().item()
            corr0   = ((preds == gt_B)[gt_B == 0]).sum().item()
            corr1   = ((preds == gt_B)[gt_B == 1]).sum().item()
            acc0    = corr0 / max(n0, 1)
            acc1    = corr1 / max(n1, 1)

            # -=-=-=- update masks -=-=-=-
            refined = ba_step(logits,refined,bndy,potts)
            oracle_logits = th.logits(gt_s[bndy].clamp(0.01,0.99))
            oracle = ba_step(oracle_logits,oracle,bndy,potts)

            # -=-=-=- compute mask quality -=-=-=-
            refined_up  = interp(refined,H,W)
            oracle_up = interp(oracle,H,W)
            iou   = np.mean(compute_miou(refined_up, gt)[0])
            d_iou = np.mean(compute_miou(oracle_up,  gt)[0])
            r_iou = np.mean(compute_miou(oracle_up,  gt_s_up)[0])

            # -=-=-=- record -=-=-=-
            update_val_log(log,{"acc0":acc0,"acc1":acc1,
                                "iou":iou,"d_iou":d_iou,"r_iou":r_iou,
                                "img_id":img_id,"dn":dname,"scale":scale,"i":step})
            

    return False

def update_val_log(record,**kwargs):
    keys = ["acc0","acc1","iou","d_iou","r_iou","img_id","dn","scale","i"]
    if len(record) == 0: # reset
        for key in keys: record[key] = []
        return record
    for key in kwargs:
        if not(key in kwargs): record[key].append(0)
        else: record[key].append(kwargs[key])

def print_val_log(record):
    record = pd.DataFrame(record)
    nids = len(record['img_id'].unique())
    gkeys = ['img_id', 'dn', 'scale' ,'i'] if nids <= 10 else ['dn', 'scale', 'i']
    grouped = (
        record.groupby(gkeys)[["acc0", "acc1", "iou", "d_iou"]]
        .mean()
    )
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.width', None):
        print(grouped.round(4))



# ---------------------------------------
#
#
#       -=-=-=- Misc -=-=-=-=-
#
#
# ---------------------------------------

def unwrap(model):
    return model.module if hasattr(model, 'module') else model

def interp(mask,_H,_W,mode='nearest'):
    mask = F.interpolate(mask.float()[:,None], size=(_H,_W), mode=mode).squeeze(1)
    return (mask > 0.5).int()
