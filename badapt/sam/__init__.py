

from .sam_refiner import run_sam_refiner

import torch as th
import numpy as np
from segment_anything import SamPredictor

def run_sam(sam,image,mask):

    # -- prepare --
    predictor = SamPredictor(sam)
    if len(image.shape) == 4:
        assert(image.shape[0] == 1)
        image = image[0]
    assert(len(image.shape) == 3)
    if th.is_tensor(image):
        image = image.cpu().numpy()
    if image.shape[2] != 3:
        image = np.transpose(image,(1,2,0))
    image = image.astype("uint8")

    # -- run sam --
    box = mask_to_bbox(mask)[0]
    predictor.set_image(image)
    masks_np, scores, logits = predictor.predict(
        box=box,multimask_output=True,return_logits=True)

    # -- keep only the highest-scoring mask --
    # print(scores)
    best_idx = int(np.argmax(scores))
    best_mask = masks_np[best_idx]
    best_score = float(scores[best_idx])

    # -- thresh and return --
    device = sam.device
    logits = th.from_numpy(best_mask).to(th.float).to(device)
    coarse = (logits > predictor.model.mask_threshold).to(th.uint8)

    return coarse[None,:],logits[None,:]


def mask_to_bbox(masks):
    """Compute tight bounding box [x1, y1, x2, y2] from a binary mask tensor."""
    if len(masks.shape) == 2: # missing batch dim
        masks = masks[None,:]
    boxes = []
    for mask in masks:
        mask_np = mask.squeeze().cpu().numpy().astype(bool)
        rows = np.any(mask_np, axis=1)
        cols = np.any(mask_np, axis=0)
        if not rows.any():
            return None
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]
        box = np.array([x1, y1, x2, y2])
        boxes.append(box)
    return boxes



