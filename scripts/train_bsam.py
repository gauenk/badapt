"""

   Train a BSAM Network

"""


import os
from easydict import EasyDict as edict

import torch as th
from badapt.bsam import BSAM
from badapt.data import get_dataset
from badapt.train import launch_training

def main():

    print("[PID: %s] Running Training Script." % (str(os.getpid())))

    cfg = edict()
    cfg.dname = "frhq"
    cfg.deform_rsr = 0.10
    cfg.deform_sr = 0.10
    cfg.deform_iou = 0.90
    cfg.potts = 0.0
    cfg.scales = [4,2,1]
    cfg.steps = [3,3,3]
    cfg.local_rank = 0
    cfg.lr = 1e-4
    cfg.niters_train = 100_000
    cfg.log_every = 50
    cfg.val_every = 100

    bsam = BSAM()
    checkpoint = None
    if checkpoint: bsam.load(checkpoint)

    optim = th.optim.AdamW(bsam.parameters(),lr=cfg.lr)
    tr_data  = get_dataset(cfg.dname,'tr',use_dist=False,nsamples=0)
    val_data = get_dataset(cfg.dname,'val',use_dist=False,nsamples=0)
    launch_training(cfg,bsam,optim,tr_data,val_data)

if __name__ == "__main__":
    main()
