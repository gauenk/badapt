#pragma once

#include <cstdint>
#include <vector>
#include <array>

// Perturb a 2D uint8 segmentation mask.
// Input and output are row-major, contiguous arrays.
// The caller owns the input and output memory.
void deform_seg_core(
    const uint8_t* data,
    int h,
    int w,
    float iou_target,
    int seed,
    uint8_t* output
);

using Point = std::array<int, 2>;
using Contour = std::vector<Point>;

// Process contours represented as vectors of (x, y) points.
// Contours with fewer than 10 points are discarded.
std::vector<Contour> deform_contours_core(
    const std::vector<Contour>& contours,
    float regional_sample_rate,
    float sample_rate,
    float move_rate,
    int seed
);
