"""

   Train a BSAM Network

"""


from easydict import EasyDict as edict

from badapt.bsam import BSAM
from badapt.data import get_datasets
from badapt.train import launch_training

def main():

    cfg = edict()
    cfg.dname = "frhq"
    cfg.deform_rsr = 0.10
    cfg.deform_sr = 0.10
    cfg.deform_iou = 0.90
    cfg.potts = 0.0
    cfg.scales = [4,2,1]
    cfg.steps = [3,3,3]
    cfg.local_rank = 0
    cfg.img_mean = th.tensor([0.485, 0.456, 0.406])
    cfg.img_std  = th.tensor([0.229, 0.224, 0.225])
    cfg.lr = 1e-4
    cfg.niters_train = 100_000
    cfg.log_every = 50
    cfg.val_every = 100

    bsam = BSAM()
    checkpoint = None
    if checkpoint: bsam.load(checkpoint)

    optim = th.optim.AdamW(lr=cfg.lr)
    tr_data,val_data = get_datasets(cfg.dname)
    launch_training(cfg,bsam,optim,tr_data,val_data)

if __name__ == "__main__":
    main()
