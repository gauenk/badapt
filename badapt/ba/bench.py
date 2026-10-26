"""


    Triton is much slower the compiled CUDA kernels


"""

import os
import cv2
import torch as th
import numpy as np
import torch.nn.functional as F
from badapt.deform import deform_mask


# -- cuda imports --
import badapt._ba_cuda as ba_cuda
get_boundary_cu = ba_cuda.get_boundary
check_is_changeable_cu = ba_cuda.check_is_changeable
log_potts_ratio_cu = ba_cuda.log_potts_ratio
find_islands_cu = ba_cuda.find_islands
find_strands_cu = ba_cuda.find_strands

# -- triton imports --
from badapt.ba import ba_triton
get_boundary_tr = ba_triton.get_boundary
check_is_changeable_tr = ba_triton.check_is_changeable
log_potts_ratio_tr = ba_triton.log_potts_ratio
find_islands_tr = ba_triton.find_islands
find_strands_tr = ba_triton.find_strands


def eval_loop(eval_fxn,ntimings=20,nwarmups=5):
    for _ in range(nwarmups):
        eval_fxn()

    # -- timing loop --
    timings = []
    start_event = th.cuda.Event(enable_timing=True)
    end_event = th.cuda.Event(enable_timing=True)
    for _ in range(ntimings):
        start_event.record()
        eval_fxn()
        end_event.record()
        th.cuda.synchronize()
        timings.append(start_event.elapsed_time(end_event))
    return timings

def eval_get_boundary_kernel():

    print("eval_get_boundary_kernel.")

    # -- get data --
    gt = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(gt, (128, 128), 60, 255, -1)
    cv2.rectangle(gt, (50, 50), (100, 100), 255, -1)
    gt = th.from_numpy(gt>0).int().cuda()
    gt = F.interpolate(gt.float()[None,None], size=(4000,4000), mode='nearest')[0,0].int()
    coarse = deform_mask(gt)

    # -- compare outputs --
    bndy_tr = get_boundary_tr(gt)
    bndy_cu = get_boundary_cu(gt)
    delta = th.sum(bndy_tr != bndy_cu).item()
    assert(delta==0)

    # -- time em --
    def eval_tr(): get_boundary_tr(gt)
    timings_tr = eval_loop(eval_tr)
    def eval_cu(): get_boundary_cu(gt)
    timings_cu = eval_loop(eval_cu)
    print("Triton: ",np.mean(timings_tr),np.std(timings_tr))
    print("CUDA: ",np.mean(timings_cu),np.std(timings_cu))


def eval_check_is_changeable():

    print("eval_check_is_changeable.")

    # -- get data --
    gt = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(gt, (128, 128), 60, 255, -1)
    cv2.rectangle(gt, (50, 50), (100, 100), 255, -1)
    gt = th.from_numpy(gt>0).int().cuda()
    gt = F.interpolate(gt.float()[None,None], size=(4000,4000), mode='nearest')[0,0].int()
    coarse = deform_mask(gt)

    # -- get query points --
    bndy = get_boundary_cu(gt)
    query = th.stack(th.where(bndy),dim=-1).int()

    # -- compare outputs --
    ischng_tr = check_is_changeable_tr(coarse,query)
    ischng_cu = check_is_changeable_cu(coarse,query)
    delta = th.sum(ischng_tr != ischng_cu).item()
    assert(delta==0)

    # -- time em --
    def eval_tr(): check_is_changeable_tr(coarse,query)
    timings_tr = eval_loop(eval_tr)
    def eval_cu(): check_is_changeable_cu(coarse,query)
    timings_cu = eval_loop(eval_cu)
    print("Triton: ",np.mean(timings_tr),np.std(timings_tr))
    print("CUDA: ",np.mean(timings_cu),np.std(timings_cu))


def eval_log_potts_ratio():

    print("eval_log_potts_ratio.")

    # -- get data --
    gt = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(gt, (128, 128), 60, 255, -1)
    cv2.rectangle(gt, (50, 50), (100, 100), 255, -1)
    gt = th.from_numpy(gt>0).int().cuda()
    gt = F.interpolate(gt.float()[None,None], size=(4000,4000), mode='nearest')[0,0].int()
    coarse = deform_mask(gt)

    # -- get query points --
    bndy = get_boundary_cu(gt)
    query = th.stack(th.where(bndy),dim=-1).int()

    # -- compare outputs --
    lpotts_tr = log_potts_ratio_tr(coarse,query)
    lpotts_cu = log_potts_ratio_cu(coarse,query)
    delta = th.sum(lpotts_tr != lpotts_cu).item()
    assert(delta==0)

    # -- time em --
    def eval_tr(): log_potts_ratio_tr(coarse,query)
    timings_tr = eval_loop(eval_tr)
    def eval_cu(): log_potts_ratio_cu(coarse,query)
    timings_cu = eval_loop(eval_cu)
    print("Triton: ",np.mean(timings_tr),np.std(timings_tr))
    print("CUDA: ",np.mean(timings_cu),np.std(timings_cu))


def eval_find_islands():

    print("eval_find_islands.")

    # -- get data --
    gt = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(gt, (128, 128), 60, 255, -1)
    cv2.rectangle(gt, (50, 50), (100, 100), 255, -1)
    gt = th.from_numpy(gt>0).int().cuda()
    gt = F.interpolate(gt.float()[None,None], size=(4000,4000), mode='nearest')[0,0].int()
    coarse = deform_mask(gt)

    # -- compare outputs --
    islands_tr = find_islands_tr(coarse)
    islands_cu = find_islands_cu(coarse)
    delta = th.sum(islands_tr != islands_cu).item()
    assert(delta==0)

    # -- time em --
    def eval_tr(): find_islands_tr(coarse)
    timings_tr = eval_loop(eval_tr)
    def eval_cu(): find_islands_cu(coarse)
    timings_cu = eval_loop(eval_cu)
    print("Triton: ",np.mean(timings_tr),np.std(timings_tr))
    print("CUDA: ",np.mean(timings_cu),np.std(timings_cu))


def eval_find_strands():

    print("eval_find_strands.")

    # -- get data --
    gt = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(gt, (128, 128), 60, 255, -1)
    cv2.rectangle(gt, (50, 50), (100, 100), 255, -1)
    gt = th.from_numpy(gt>0).int().cuda()
    gt = F.interpolate(gt.float()[None,None], size=(4000,4000), mode='nearest')[0,0].int()
    coarse = deform_mask(gt)

    # -- compare outputs --
    strands_tr = find_strands_tr(coarse)
    strands_cu = find_strands_cu(coarse)
    delta = th.sum(strands_tr != strands_cu).item()
    assert(delta==0)

    # -- time em --
    def eval_tr(): find_strands_tr(coarse)
    timings_tr = eval_loop(eval_tr)
    def eval_cu(): find_strands_cu(coarse)
    timings_cu = eval_loop(eval_cu)
    print("Triton: ",np.mean(timings_tr),np.std(timings_tr))
    print("CUDA: ",np.mean(timings_cu),np.std(timings_cu))


if __name__ == "__main__":

    print("[%d] Comparing wall-clock runtimes between CUDA and Triton."%os.getpid())
    eval_get_boundary_kernel()
    eval_check_is_changeable()
    eval_log_potts_ratio()
    eval_find_islands()
    eval_find_strands()
