

import numpy as np
import torch as th
import pandas as pd

import torch.nn.functional as F
from badapt.deform import deform_mask
from badapt.ba import get_boundary,ba_step

import time
from datetime import datetime
from tqdm.auto import tqdm

# ---------------------------------------
#
#
#     -=-=-=- Train/Val Loop -=-=-=-=-
#
#
# ---------------------------------------


def launch_training(cfg, bsam, optim, tr_data, val_data):

    # -=-=- init -=-=-
    tr_log = {
        str(s): {"loss": 0., "corr": 0, "total": 0, "count": 0}
        for s in cfg.scales
    }
    cfg.device = f"cuda:{cfg.local_rank}"

    is_main = cfg.local_rank == 0
    start_time = time.perf_counter()
    last_val_time = None
    last_ckpt_time = None

    if is_main:
        log_msg("=" * 65)
        log_msg(f"Training started | iterations={cfg.niters_train}")
        log_msg(f"Scales={cfg.scales} | Steps={cfg.steps}")
        log_msg(
            f"Validation every {cfg.val_every} iterations | "
            f"Checkpoint every {cfg.ckpt_every} steps"
        )
        log_msg("=" * 65)

    # -=-=- training loop -=-=-
    progress = tqdm(
        range(cfg.niters_train),
        desc="Training",
        unit="iter",
        dynamic_ncols=True,
        disable=not is_main,
    )

    for i in progress:
        iter_start = time.perf_counter()

        train_step(cfg, bsam, optim, tr_data, tr_log)

        iter_time = time.perf_counter() - iter_start
        elapsed = time.perf_counter() - start_time

        # -- validation --
        if i % cfg.val_every == 0:
            if is_main:
                log_msg(
                    f"Validation starting | iter={i + 1}/{cfg.niters_train} "
                    f"| step={cfg.global_step}"
                )

            val_start = time.perf_counter()
            val_log = update_val_log({})
            val_iter = iter(val_data)

            val_progress = tqdm(
                range(len(val_data)),
                desc="Validation",
                unit="batch",
                leave=False,
                dynamic_ncols=True,
                disable=not is_main,
            )

            for _ in val_progress:
                val_step(cfg, bsam, val_iter, val_log)

            val_duration = time.perf_counter() - val_start
            last_val_time = time.perf_counter()

            if is_main:
                log_msg(
                    f"Validation complete | "
                    f"{format_time(val_duration)} | "
                    f"{len(val_data)} batches"
                )
                print_val_log(val_log)

        # -- training statistics --
        if (
            cfg.global_step % cfg.log_every == 0
            and is_main
        ):
            print_train_log(tr_log, cfg.global_step)

        # -- checkpoint --
        if (
            cfg.global_step % cfg.ckpt_every == 0
            and is_main
        ):
            ckpt_start = time.perf_counter()
            log_msg(f"Saving checkpoint | step={cfg.global_step}")

            save_checkpoint(cfg, bsam, optim)

            ckpt_duration = time.perf_counter() - ckpt_start
            last_ckpt_time = time.perf_counter()

            log_msg(
                f"Checkpoint complete | "
                f"{format_time(ckpt_duration)}"
            )

        # -- update training progress --
        if is_main:
            avg_iter_time = elapsed / max(i + 1, 1)
            remaining_iters = cfg.niters_train - (i + 1)
            eta = remaining_iters * avg_iter_time

            # Iterations until the next scheduled event.
            next_val = cfg.val_every - ((i + 1) % cfg.val_every)
            next_ckpt = cfg.ckpt_every - (
                cfg.global_step % cfg.ckpt_every
            )

            progress.set_postfix({
                "step": cfg.global_step,
                "iter_s": f"{iter_time:.2f}",
                "elapsed": format_time(elapsed),
                "ETA": format_time(eta),
                "val_in": next_val,
                "ckpt_in": next_ckpt,
            })

    if is_main:
        log_msg("=" * 65)
        log_msg(
            f"Training finished | steps={cfg.global_step} | "
            f"total time={format_time(time.perf_counter() - start_time)}"
        )
        log_msg("=" * 65)

    return


# def _launch_training(cfg,bsam,optim,tr_data,val_data):

#     # -=-=- init -=-=-=-
#     tr_log = {str(s): {"loss":0., "corr":0, "total":0, "count":0} for s in cfg.scales}
#     cfg.device   = f"cuda:{cfg.local_rank}"

#     # -=-=- training loop -=-=-
#     for i in range(cfg.niters_train):

#         train_step(cfg,bsam,optim,tr_data,tr_log)

#         if i % cfg.val_every == 0:
#             val_log = update_val_log({})
#             val_iter = iter(val_data)
#             for _ in range(len(val_data)):
#                 val_step(cfg,bsam,val_iter,val_log)
#             if cfg.local_rank == 0: print_val_log(val_log)

#         if (cfg.global_step % cfg.log_every == 0) and (cfg.local_rank == 0):
#             print_train_log(tr_log,cfg.global_step)

#         if (cfg.global_step % cfg.ckpt_every == 0) and (cfg.local_rank == 0):
#             save_checkpoint(cfg,bsam,optim)

#     return

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
    # print(f"[step {global_step}]  " + "  ".join(parts))
    log_msg(f"[step {global_step}] " + "  ".join(parts))

    # -=-= reset -=-=-
    for s in scales:
        stats[s] = {"loss":0., "corr":0, "total":0, "count":0}


def save_checkpoint(cfg, bsam, optim):
    ckpt_path = (
        cfg.save_root /
        f"checkpoint_{cfg.dname}_{cfg.model_name}_{cfg.global_step}.pt"
    )

    bsam_state = (
        bsam.module.state_dict()
        if cfg.use_dist
        else bsam.state_dict()
    )

    th.save(
        {
            "step": cfg.global_step,
            "epoch": cfg.epoch,
            "model_state": bsam_state,
        },
        ckpt_path,
    )

    log_msg(f"Checkpoint saved | {ckpt_path}")



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
    gt = (batch['label'][0].round() > 128.0).cuda(non_blocking=True).int()
    if gt.shape[0] == 0: return False
    assert(gt.shape[0] == 1)
    image,gt = image[0],gt[0]
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
            bndy,aux = get_training_boundary(refined,cfg.aux_points)

            # log[p(x|z_i=1,..)/p(x|z_i=0,...)] \approx BSAM(...)
            query  = scale*th.stack(th.where(aux)).T+(scale-1)/2.0
            iftrs  = bsam.iftr_net_ms(bsam.iftr_net(iftrs_raw),scale)
            mftrs  = bsam.mftr_net(coarse_s, refined, scale, (H,W) )
            logits = bsam(query, iftrs, sftrs, mftrs, context, (H,W) )
            logits_B,logits_R = logits[bndy[aux]],logits[~bndy[aux]] # split into bndy vs aux

            # update
            with th.no_grad():
                refined = ba_step_tr(logits_B.detach(),refined,bndy,cfg.potts, gt_s,scale,cfg.global_step)

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
                preds_BR = (logits.detach() > 0).float()
                gt_BR    = gt_s[aux].float()
                key      = str(scale)
                log[key]["loss"]  += loss.detach().item()
                log[key]["corr"]  += (preds_BR == gt_BR).sum().item()
                log[key]["total"] += gt_BR.numel()
                log[key]["count"] += 1

    cfg.global_step += 1

def ba_step_tr(logits, refined, bndy, potts, gt_s, scale, global_step):
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
        new_mask[bndy] = gt_s[bndy]
    else:
        new_mask = ba_step(logits, refined.int(), bndy, potts)
    return new_mask.detach()

def get_training_boundary(mask,aux_points='thicc'):

    def subsample_mask(imask, max_points):
        """Keep at most max_points True values in a bool mask, chosen randomly."""
        idx = th.where(imask.ravel())[0]  # flat indices of True values
        if len(idx) > max_points:
            perm = th.randperm(len(idx), device=mask.device)[:max_points]
            idx = idx[perm]
        out = th.zeros_like(imask)
        out.view(-1)[idx] = True
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
    if aux_points == 'thicc':
        thicc = get_boundary(mask, 4)
        extra = thicc & ~boundary
        extra = subsample_mask(extra, 2000)
        aux_boundary = boundary | extra
        aux_boundary = sample_points_on_mask(aux_boundary,1000)
    elif aux_points == 'rand':
        aux_boundary = sample_points_on_mask(boundary,3000)

    return boundary,aux_boundary

def weighted_focal_loss_with_logits(logits, targets, gamma=2.0, reduction="mean"):
    n_neg = (targets == 0).sum()
    n_pos = (targets == 1).sum()
    alpha = th.clamp(n_neg / (n_pos + n_neg), 0.25, 0.75)
    loss = focal_loss_with_logits(logits,targets * 1.0,alpha=alpha,gamma=2.0)
    return loss

def focal_loss_with_logits(logits, targets, alpha=None, gamma=2.0, reduction="mean"):
    """
    logits: (N, ...)
    targets: same shape, float {0,1}
    alpha: scalar or tensor for class balancing (like pos_weight analogue)
    gamma: focusing parameter
    """
    # BCE with logits, no reduction
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

    # Probabilities
    probs = th.sigmoid(logits)

    # p_t = p if y=1 else (1-p)
    p_t = probs * targets + (1 - probs) * (1 - targets)

    # Focal modulation
    focal_weight = (1 - p_t) ** gamma

    # Optional alpha weighting (like pos_weight idea)
    if alpha is not None:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        bce_loss = alpha_t * bce_loss

    loss = focal_weight * bce_loss

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


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
    img_id = int(batch['imidx'].cpu())
    dname  = str(batch['dname'][0])
    image  = batch['image'].cuda(non_blocking=True)
    gt = (batch['label'][0].round() > 128.0).cuda(non_blocking=True).int()
    if gt.shape[0] == 0: return False
    assert(gt.shape[0] == 1)
    image,gt = image[0],gt[0]
    H,W,C = image.shape

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
    d_iou = compute_iou(coarse, gt)
    update_val_log(log,d_iou=d_iou)

    # -=-=- Start dat inference baby! -=-=-
    refined = coarse.clone()
    for scale, nsteps in zip(cfg.scales, cfg.steps):

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
            mftrs  = bsam.mftr_net(coarse_s, refined, scale, (H,W) )
            logits = bsam(query, iftrs_s, sftrs, mftrs, context, (H,W) )

            # -=-=- compute accuracy for 0s and 1s -=-=-=-
            gt_B   = gt_s[bndy].float()
            preds   = (logits > 0).float()
            n0      = (gt_B == 0).sum().item()
            n1      = (gt_B == 1).sum().item()
            corr0   = ((preds == gt_B)[gt_B == 0]).sum().item()
            corr1   = ((preds == gt_B)[gt_B == 1]).sum().item()
            acc0    = corr0 / max(n0, 1) * 100
            acc1    = corr1 / max(n1, 1) * 100

            # -=-=-=- update masks -=-=-=-
            refined = ba_step(logits,refined,bndy,cfg.potts)
            oracle_logits = th.logit(gt_s[bndy].clamp(0.01,0.99))
            oracle = ba_step(oracle_logits,oracle,bndy,cfg.potts)

            # -=-=-=- compute mask quality -=-=-=-
            refined_up  = interp(refined,H,W)
            oracle_up = interp(oracle,H,W)
            iou   = compute_iou(refined_up, gt)
            oracle_iou = compute_iou(oracle_up,  gt)
            r_iou = compute_iou(oracle_up,  gt_s_up)

            # -=-=-=- record -=-=-=-
            update_val_log(log,**{"acc0":acc0,"acc1":acc1,
                                "iou":iou,"oracle":oracle_iou,"oracle_res":r_iou,
                                "img_id":img_id,"dn":dname,"scale":scale,"i":step})

    return False

def update_val_log(record,**kwargs):
    keys = ["acc0","acc1","iou","oracle","oracle_res","img_id","dn","scale","i"]
    if len(record.keys()) == 0: # reset
        for key in keys: record[key] = []
        return record
    for key in keys:
        if not(key in kwargs): record[key].append(-1)
        else: record[key].append(kwargs[key])

def print_val_log(record):
    record = pd.DataFrame(record)
    nids = len(record['img_id'].unique())
    gkeys = ['img_id', 'dn', 'scale' ,'i'] if nids <= 10 else ['dn', 'scale', 'i']
    grouped = (
        record.groupby(gkeys)[["acc0", "acc1", "iou", "oracle"]]
        .mean()
    )
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.width', None):
        tqdm.write(grouped.round(4).to_string())
        # print(grouped.round(4))



# ---------------------------------------
#
#
#       -=-=-=- Misc -=-=-=-=-
#
#
# ---------------------------------------

def unwrap(model):
    return model.module if hasattr(model, 'module') else model

def interp(mask,_H,_W,mode='nearest',thresh=True):
    kw = {"align_corners":False} if mode == 'bilinear' else {}
    mask = F.interpolate(mask.float()[None,None], size=(_H,_W), mode=mode, **kw)[0,0]
    if thresh: return (mask > 0.5).int()
    else: return mask

def log_msg(message):
    """Timestamped messages that don't break tqdm progress bars."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    tqdm.write(f"[{timestamp}] {message}")

def format_time(seconds):
    """Format seconds as a compact human-readable duration."""
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)

    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"

def compute_iou(pred, anno):
    """Compute IoU for each class"""

    if not th.is_tensor(pred):
        pred = th.from_numpy(pred)
    if not th.is_tensor(anno):
        anno = th.from_numpy(anno).to(pred.device)

    if pred.ndim == 3:
        pred = pred[0]
        anno = anno[0]
        assert(pred.shape[0]==1)
        assert(anno.shape[0]==1)
    assert(pred.ndim == 2)
    assert(anno.ndim == 2)

    pred_mask = (pred > 0)
    anno_mask = (anno > 0)

    intersection = th.logical_and(pred_mask, anno_mask).sum().item()
    union = th.logical_or(pred_mask, anno_mask).sum().item()*1.0

    any_b = th.any(anno_mask)
    iou = (intersection / union)*100 if any_b>0 else -1.0

    return iou

