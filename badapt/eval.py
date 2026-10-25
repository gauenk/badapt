

import cv2

import numpy as np
import torch as th

# from badapt.sam_refiner import run_sam_refiner
from badapt.bsam import BSAM,boundary_adaptation
# def boundary_adaptation(): pass
# class BSAM(th.nn.Module):
#     def __init__(*args): pass
from badapt.data import get_eval_dataset
from segment_anything import sam_model_registry,SamPredictor

import json
import pycocotools.mask as mask_utils

from pathlib import Path
from collections import defaultdict
from scipy.ndimage import label as scipy_label


def eval_refiner(rname,dname,potts,steps,scales):

    # -- prepare refiner --
    device = 'cuda:0'
    refiner = None
    if rname == "bsam":
        refiner = BSAM("checkpoints/bsam.pt").to(device)
    else:
        hq_mode = 'hq' in rname
        refiner = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
        # predictor = SamPredictor(sam)
        # predictor.set_image(<your_image>)
        # masks, _, _ = predictor.predict(<input_prompts>)
        # refiner.mask_decoder = HQMaskDecoder()

    # -- misc [run sam during eval... slower] --
    # sam = sam_model_registry["vit_h"](checkpoint="checkpoints/sam_vit_h_4b8939.pth")
    # sam = sam.to(device)

    # -- load dataset --
    data = get_eval_dataset(dname,'val')

    # -- prepare coarse masks --
    coarse_fname = f"data/sam_masks/{dname}.json"
    dts = json.load(open(coarse_fname))
    dts_by_img_id = defaultdict(list)
    for dt in dts: dts_by_img_id[int(dt['image_id'])].append(dt)

    # -- loop over masks --
    img_mean = th.tensor([0.485, 0.456, 0.406]).to(device)
    img_std  = th.tensor([0.229, 0.224, 0.225]).to(device)
    record = defaultdict(list)
    for img_id,dets in dts_by_img_id.items():

        # -=-=- get (image, mask) pair -=-=-
        sample = data[img_id]
        image  = sample['image'].cuda(non_blocking=True)
        gt = (sample['label'].round() > 128.0).int().cuda(non_blocking=True)
        if gt.shape[0] == 0: return False
        assert(gt.shape[0] == 1)
        gt = gt[0]

        # -=-=-=- read coarse masks -=-=-=-
        coarse,_ = read_annos(dets)
        if coarse.shape[0] == 0: continue
        assert(coarse.shape[0] == 1)
        coarse = th.from_numpy(coarse[0]).to(device).to(th.uint8)

        # -- run sam [very optional... just read coarse from file...] --
        # coarse = run_sam(sam,image,gt)[0][0].to(th.uint8)
        # miou_coarse = float(compute_iou(coarse, gt))

        # -=-=-=- get refined mask -=-=--
        if rname == "bsam":
            refined = boundary_adaptation(refiner,image,coarse,potts,steps,scales)
        else:
            refined = run_sam_refiner(refiner,rname,image,coarse,steps)

        # -=-=-=- compute mIoU -=-=-=-
        miou_coarse = float(compute_iou(coarse, gt))
        miou_refined = float(compute_iou(refined, gt))
        # print(img_id,miou_coarse,miou_refined)

        # -=-=-=- compute bIoU -=-=-=-
        coarse_b = mask_to_boundary(coarse)
        refined_b = mask_to_boundary(refined)
        gt_b = mask_to_boundary(gt)
        biou_coarse = float(compute_iou(coarse_b, gt_b))
        biou_refined = float(compute_iou(refined_b, gt_b))

        # -=-=-=-=- Connected Comps -=-=-=-=-
        cc_gt = ndi_label(gt.cpu())[1]
        cc_coarse = ndi_label(coarse.cpu())[1]
        cc_refined = ndi_label(refined.cpu())[1]

        # -=-=-=- summary -=-=-=-=-
        print("[%03d] mIoU: [%2.3f,%2.3f] bIoU: [%2.3f,%2.3f] CC: [%d,%d,%d]" % (img_id,miou_coarse,miou_refined,
                                                                                 biou_coarse,biou_refined,
                                                                                 cc_gt,cc_coarse,cc_refined))

        # -- record --
        record['mIoU-C'].append(miou_coarse)
        record['mIoU-R'].append(miou_refined)
        record['bIoU-C'].append(biou_coarse)
        record['bIoU-R'].append(biou_refined)
        record['size'].append((gt>0).float().mean().item())
        record['img_id'].append(img_id)
    
    return record


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

def ndi_label(mask):
    s = np.array([[0,1,0],[1,1,1],[0,1,0]])
    labeled_array, num_features = scipy_label(mask, structure=s)
    return labeled_array, num_features

def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """

    if mask.ndim == 3:
        assert(mask.shape[0] == 1)
        mask = mask[0]
    if th.is_tensor(mask):
        mask = mask.cpu().numpy()
    mask = mask.astype(np.uint8)

    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode


def read_annos(annos):
    masks = []
    classes = []
    for ann in annos:
        seg = ann["segmentation"]
        H,W = ann["segmentation"]['size']
        rle = mask_utils.merge([seg])
        m = mask_utils.decode(rle).astype(np.uint8)  # H x W
        masks.append(m)
        classes.append(ann['category_id'])
    masks = np.stack(masks)
    classes = np.stack(classes)
    return masks,classes


# def mask_to_bbox(masks):
#     """Compute tight bounding box [x1, y1, x2, y2] from a binary mask tensor."""
#     if len(masks.shape) == 2: # missing batch dim
#         masks = masks[None,:]
#     boxes = []
#     for mask in masks:
#         mask_np = mask.squeeze().cpu().numpy().astype(bool)
#         rows = np.any(mask_np, axis=1)
#         cols = np.any(mask_np, axis=0)
#         if not rows.any():
#             return None
#         y1, y2 = np.where(rows)[0][[0, -1]]
#         x1, x2 = np.where(cols)[0][[0, -1]]
#         box = np.array([x1, y1, x2, y2])
#         boxes.append(box)
#     return boxes

# def run_sam(sam,image,mask):

#     # -- prepare --
#     predictor = SamPredictor(sam)
#     if len(image.shape) == 4:
#         assert(image.shape[0] == 1)
#         image = image[0]
#     assert(len(image.shape) == 3)
#     if th.is_tensor(image):
#         image = image.cpu().numpy()
#     if image.shape[2] != 3:
#         image = np.transpose(image,(1,2,0))
#     image = image.astype("uint8")

#     # -- run sam --
#     box = mask_to_bbox(mask)[0]
#     predictor.set_image(image)
#     masks_np, scores, logits = predictor.predict(
#         box=box,multimask_output=True,return_logits=True)

#     # -- keep only the highest-scoring mask --
#     # print(scores)
#     best_idx = int(np.argmax(scores))
#     best_mask = masks_np[best_idx]
#     best_score = float(scores[best_idx])

#     # -- thresh and return --
#     device = sam.device
#     logits = th.from_numpy(best_mask).to(th.float).to(device)
#     coarse = (logits > predictor.model.mask_threshold).int()

#     return coarse[None,:],logits[None,:]


