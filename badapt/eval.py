
from badapt.bsam import BSAM,boundary_adaptation
from badapt.data import get_eval_dataset

from badapt.sam_refiner import run_sam_refiner

import pycocotools.mask as mask_utils

from pathlib import Path
from collections import defaultdict

def eval_refiner(rname,dname,potts,steps,scales=None):


    # -- prepare refiner --
    device = 'cuda:0'
    refiner = None
    if rname == "bsam":
        refiner = BSAM("checkpoints/bsam.th")
        state = th.load(checkpoint_fname)
        bsam.load_state(state['model_state'])
        bsam = bsam.to(device)
    else:
        hq_mode = 'hq' in rname
        refiner = SAM(hq_mode)

    # -- load dataset --
    data = get_eval_dataset(dname,'val')

    # -- prepare coarse masks --
    coarse_fname = f"assets/sam_masks/{dname}.json"
    dts = json.load(open(coarse_fname))
    dts_by_img_id = defaultdict(list)
    for dt in dts:
        if dt['image_id'] in data.img_ids and dt['category_id'] in data.cats:
            dts_by_img_id[int(dt['image_id'])].append(dt)

    # -- loop over masks --
    record = defaultdict(list)
    for img_id,dets in dts_by_img_id.items():

            # -- read coarse masks --
            coarse,_ = read_annos(dets)
            if coarse.shape[0] == 0: continue

            # -=-=- get (image, mask) pair -=-=-
            image,gt = data[img_id]
            image  = (batch['image'] / 255.0).cuda(non_blocking=True)
            image = (image - cfg.img_mean[None, None, None, :]) / cfg.img_std[None, None, None, :]
            gt = (batch['label'][0].round() > 128.0).int()
            if gt.shape[0] == 0: return False
            assert(gt.shape[0] == 1)
            H, W, C = image.shape

            # -=-=-=- get refined mask -=-=--
            if rname == "bsam":
                refined = boundary_adaptation(refiner,image,coarse,potts,steps,scales)
            else:
                refined = run_sam_refiner(refiner,rname,image,coarse,steps)

            # -=-=-=- compute mIoU -=-=-=-
            miou_coarse = float(np.mean(compute_iou(coarse, gt)))
            miou_refined = float(np.mean(compute_iou(refined, gt)))

            # -=-=-=- compute bIoU -=-=-=-
            coarse_b = mask_to_boundary(coarse)
            refined_b = mask_to_boundary(refined)
            gt_b = mask_to_boundary(gt)
            biou_coarse = float(np.mean(compute_iou(coarse_b, gt_b)))
            biou_refined = float(np.mean(compute_iou(refined_b, gt_b)))

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

    ious = []
    NumObjs = pred.shape[0]
    for obj_ix in range(NumObjs):
        pred_mask = (pred[obj_ix] == 1)
        anno_mask = (anno[obj_ix] == 1)

        intersection = th.logical_and(pred_mask, anno_mask).sum().item()
        union = th.logical_or(pred_mask, anno_mask).sum().item()

        if anno_mask.sum() == 0:
            iou = -1.0
        elif union == 0:
            iou = float('nan')  # class not present
        else:
            iou = intersection / union
        ious.append(iou)
    ious = np.array(ious)
    return ious

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
