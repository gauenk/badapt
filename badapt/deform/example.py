"""

   This script is from ChatGpt

"""

from badapt.deform import deform_mask,deform_seg,deform_contours

import cv2
import numpy as np
import torch as th

# ------------------------------------------------------------
# Example usage
# ------------------------------------------------------------

def main():
    # Create a sample binary mask.
    gt = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(gt, (128, 128), 60, 255, -1)
    cv2.rectangle(gt, (50, 50), (100, 100), 255, -1)

    # Perturb the segmentation.
    perturbed = deform_seg(
        gt,
        iou_target=0.9,
        seed=42,
    )

    print("Input mask:", gt.shape, gt.dtype)
    print("Perturbed mask:", perturbed.shape, perturbed.dtype)

    cv2.imwrite("mask_original.png", gt)
    cv2.imwrite("mask_perturbed.png", perturbed)

    # Extract contours from the original mask.
    contours, hierarchy = cv2.findContours(
        gt.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    # Process the contours.
    processed = deform_contours(
        contours,
        regional_sample_rate=0.10,
        sample_rate=0.10,
        move_rate=0.0,
        seed=42,
    )
    print("Original contour count:", len(contours[0]))
    print("Processed contour count:", len(processed[0]))

    # Draw processed contours.
    canvas = np.zeros_like(gt)
    if processed:
        cv2.drawContours(canvas, processed, -1, 255, -1)
    cv2.imwrite("mask_processed_contours.png", canvas)
    
    deform = deform_mask(gt)
    cv2.imwrite("mask_deformed.png", deform.cpu().numpy().astype(np.uint8)*255)

    print("Saved:")
    print("  mask_original.png")
    print("  mask_perturbed.png")
    print("  mask_processed_contours.png")
    print("  mask_deformed.png")


if __name__ == "__main__":
    main()
