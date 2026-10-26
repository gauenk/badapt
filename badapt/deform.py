"""

   Code to Synthetically Corrupt a Mask


"""

import cv2
import torch as th
import numpy as np
from PIL import Image
from pathlib import Path

import random
from sclr import perturb_seg_cpp # temporary... triton soon?


def deform_mask(mask, regional_sample_rate=0.1, sample_rate=0.1, iou_target = 0.8, allow_perturb=True, seed=None):

    
    move_rate = 0.0
    if th.is_tensor(mask):
        device = mask.device
        mask = mask.cpu().numpy()
    else:
        device = 'cpu'
    # original = mask.copy()
    if mask.max() == 1: mask = mask*255
    mask = mask.astype(np.uint8)

    # get boundaries
    if int(cv2.__version__[0]) >= 4:
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    else:
        _, contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # print(contours[0].shape)
    # print(len(contours))
    # contours = [c[:,0].tolist() for c in contours]
    # contours = [c for c in contours]
    # contours[0] = contours[0][:,0]
    # exit()

    # contours = [c.tolist() for c in contours]
    # for c in contours:
    #     print(len(c),len(c[0]))

    # only modified contours is needed actually. 
    # sampled_contours = []   

    # modified_contours = [] 
    # for contour in contours:
    #     if contour.shape[0] < 10:
    #         continue
    #     M = cv2.moments(contour)

    #     #remove region of contour
    #     number_of_vertices = contour.shape[0]
    #     number_of_removes = int(number_of_vertices * regional_sample_rate)
        
    #     idx_dist = []
    #     for i in range(number_of_vertices-number_of_removes):
    #         idx_dist.append([i, np.sum((contour[i] - contour[i+number_of_removes])**2)])
            
    #     idx_dist = sorted(idx_dist, key=lambda x:x[1])
        
    #     remove_start = random.choice(idx_dist[:math.ceil(0.1*len(idx_dist))])[0]
        
    #     #remove_start = random.randrange(0, number_of_vertices-number_of_removes, 1)
    #     new_contour = np.concatenate([contour[:remove_start], contour[remove_start+number_of_removes:]], axis=0)
    #     contour = new_contour
        

    #     #sample contours
    #     number_of_vertices = contour.shape[0]
    #     indices = random.sample(range(number_of_vertices), int(number_of_vertices * sample_rate))
    #     indices.sort()
    #     sampled_contour = contour[indices]
    #     # sampled_contours.append(sampled_contour)

    #     modified_contour = np.copy(sampled_contour)
    #     if (M['m00'] != 0):
    #         center = round(M['m10'] / M['m00']), round(M['m01'] / M['m00'])

    #         #modify contours
    #         for idx, coor in enumerate(modified_contour):

    #             change = np.random.normal(0,move_rate) # 0.1 means change position of vertex to 10 percent farther from center
    #             x,y = coor[0]
    #             new_x = x + (x-center[0]) * change
    #             new_y = y + (y-center[1]) * change

    #             modified_contour[idx] = [new_x,new_y]
    #     modified_contours.append(modified_contour)

    # print(regional_sample_rate,sample_rate)

    # print(regional_sample_rate, sample_rate)
    if seed is None:
        seed = int(random.randint(0, 2**31 - 1))
    # print(regional_sample_rate, sample_rate, move_rate, seed)
    modified_contours = perturb_seg_cpp.process_contours(
        list(contours), float(regional_sample_rate), float(sample_rate), float(move_rate), seed
    )
    # modified_contours = list(contours)
        
    #draw boundary
    gt = np.copy(mask)
    mask = np.zeros((mask.shape[0], mask.shape[1], 3))

    modified_contours = [cont for cont in modified_contours if len(cont) > 0]
    if len(modified_contours) == 0:
        mask = gt.copy()
    else:
        mask = cv2.drawContours(mask, modified_contours, -1, (255, 0, 0), -1)

    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    if allow_perturb:
        mask = perturb_seg_cpp.perturb_seg(mask, iou_target, seed)

    # mask = mask / 255
    # mask = (mask >= 0.5).astype(np.uint8)
    mask = (mask >= 0.5).astype(np.int32)
    
    return th.from_numpy(mask).to(device)

def main():

    # visualize a deformed mask
    image = read_image()
    mask = read_mask()
    deform = deform_mask(mask,0.10,0.10,0.80)
    tosave = th.stack([overlay(image,mask),overlay(image,deform)])

    save_root = Path("output")
    if not save_root.exists(): save_root.mkdir()
    tv_utils.save_image(tosave,save_root/"example_deform.png")


if __name__ == "__main__":
    main()

