"""

   Train a BSAM Network

"""


import os
from pathlib import Path
from pprint import pformat
from easydict import EasyDict as edict

import torch as th
from badapt.bsam import BSAM
from badapt.data import get_dataset
from badapt.train import launch_training


def print_config(cfg):
    print("\n" + "=" * 65)
    print("                    BSAM TRAINING")
    print("=" * 65)

    print("\n[DATA]")
    print(f"  Dataset             : {cfg.dname}")

    print("\n[MODEL]")
    print(f"  Scales              : {cfg.scales}")
    print(f"  Steps per scale     : {cfg.steps}")
    print(f"  Total refinement steps / image: {sum(cfg.steps)}")

    print("\n[OPTIMIZATION]")
    print(f"  Learning rate       : {cfg.lr:.2e}")
    print(f"  Training iterations : {cfg.niters_train:,}")
    print(f"  Initial global step : {cfg.global_step:,}")

    print("\n[VALIDATION & LOGGING]")
    print(f"  Log interval        : every {cfg.log_every:,} iterations")
    print(f"  Validation interval : every {cfg.val_every:,} iterations")
    print(f"  Checkpoint interval : every {cfg.ckpt_every:,} steps")

    print("\n[BOUNDARY / DEFORMATION]")
    print(f"  Auxiliary points    : {cfg.aux_points}")
    print(f"  Deform RSR          : {cfg.deform_rsr:.3f}")
    print(f"  Deform SR           : {cfg.deform_sr:.3f}")
    print(f"  Deform IoU          : {cfg.deform_iou:.3f}")
    print(f"  Potts weight        : {cfg.potts:.3f}")

    print("\n[ESTIMATES]")
    print(f"  Training log events : ~{cfg.niters_train // cfg.log_every:,}")
    print(f"  Validation events   : ~{cfg.niters_train // cfg.val_every:,}")
    print(f"  Checkpoint events   : ~{cfg.niters_train // cfg.ckpt_every:,}")

    print("\n" + "=" * 65 + "\n")

def main():

    print("[PID: %s] Running Training Script." % (str(os.getpid())))

    cfg = edict()
    cfg.dname = "frhq"
    cfg.deform_rsr = 0.10
    cfg.deform_sr  = 0.10
    cfg.deform_iou = 0.90
    cfg.aux_points = 'thicc'
    cfg.potts = 0.0
    cfg.scales = [4,2,1]
    cfg.steps = [3,3,3]
    cfg.local_rank = 0
    cfg.lr = 1e-4
    cfg.niters_train = 50_000
    cfg.log_every = 100
    cfg.val_every = 1000
    cfg.ckpt_every = 5000
    cfg.global_step = 0
    cfg.save_root = "training_checkpoints"

    if cfg.local_rank == 0:
        print_config(cfg)
    if not Path(cfg.save_root).exists():
        Path(cfg.save_root).mkdir()

    bsam = BSAM()
    checkpoint = None
    if checkpoint: bsam.load(checkpoint)
    device = 'cuda'
    bsam = bsam.to(device)
    bsam.sam['sam'].to(device)

    optim = th.optim.AdamW(bsam.parameters(),lr=cfg.lr)
    tr_data  = get_dataset(cfg.dname,'tr',use_dist=False,nsamples=0)
    val_data = get_dataset(cfg.dname,'val',use_dist=False,nsamples=3)
    launch_training(cfg,bsam,optim,tr_data,val_data)

if __name__ == "__main__":
    main()
