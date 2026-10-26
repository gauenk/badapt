


#include <torch/extension.h>
#include <cstdint>
#include <vector>
#include <array>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <limits>

using Point = std::array<int, 2>;
using Contour = std::vector<Point>;

void deform_seg_core(
    const uint8_t* data,
    int h,
    int w,
    float iou_target,
    int seed,
    uint8_t* output
);

std::vector<Contour> deform_contours_core(
    const std::vector<Contour>& contours,
    float regional_sample_rate,
    float sample_rate,
    float move_rate,
    int seed
);

// ------------------------------------------------------------
// Kernel generation
// ------------------------------------------------------------

static void make_rect(std::vector<uint8_t>& k, int size)
{
    std::fill(k.begin(), k.end(), 1);
}

static void make_ellipse(
    std::vector<uint8_t>& k,
    int kw,
    int kh)
{
    float cx = (kw - 1) / 2.0f;
    float cy = (kh - 1) / 2.0f;

    float rx = cx == 0 ? 1.0f : cx;
    float ry = cy == 0 ? 1.0f : cy;

    for (int y = 0; y < kh; ++y) {
        for (int x = 0; x < kw; ++x) {
            float dx = (x - cx) / rx;
            float dy = (y - cy) / ry;

            k[y * kw + x] =
                (dx * dx + dy * dy <= 1.0f) ? 1 : 0;
        }
    }
}

static std::vector<uint8_t> get_random_structure(
    std::mt19937& rng,
    int size,
    int& kw,
    int& kh)
{
    std::uniform_int_distribution<int> dist(1, 4);
    int choice = dist(rng);

    if (choice == 1) {
        kw = kh = size;

        std::vector<uint8_t> k(size * size);
        make_rect(k, size);
        return k;
    }

    if (choice == 2) {
        kw = kh = size;

        std::vector<uint8_t> k(size * size);
        make_ellipse(k, kw, kh);
        return k;
    }

    if (choice == 3) {
        kw = size;
        kh = std::max(1, size / 2);

        std::vector<uint8_t> k(kw * kh);
        make_ellipse(k, kw, kh);
        return k;
    }

    kw = std::max(1, size / 2);
    kh = size;

    std::vector<uint8_t> k(kw * kh);
    make_ellipse(k, kw, kh);
    return k;
}

// ------------------------------------------------------------
// Morphological operations
// ------------------------------------------------------------

static void dilate_region(
    uint8_t* data,
    int h,
    int w,
    const std::vector<uint8_t>& kernel,
    int kw,
    int kh)
{
    std::vector<uint8_t> tmp(
        static_cast<size_t>(h) * w, 0);

    int rx = kw / 2;
    int ry = kh / 2;

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (data[y * w + x]) {
                for (int ky = 0; ky < kh; ++ky) {
                    for (int kx = 0; kx < kw; ++kx) {
                        if (!kernel[ky * kw + kx])
                            continue;

                        int ny = y + ky - ry;
                        int nx = x + kx - rx;

                        if (ny >= 0 && ny < h &&
                            nx >= 0 && nx < w) {
                            tmp[ny * w + nx] = 255;
                        }
                    }
                }
            }
        }
    }

    std::copy(tmp.begin(), tmp.end(), data);
}

static void erode_region(
    uint8_t* data,
    int h,
    int w,
    const std::vector<uint8_t>& kernel,
    int kw,
    int kh)
{
    std::vector<uint8_t> tmp(
        static_cast<size_t>(h) * w, 255);

    int rx = kw / 2;
    int ry = kh / 2;

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (!data[y * w + x]) {
                for (int ky = 0; ky < kh; ++ky) {
                    for (int kx = 0; kx < kw; ++kx) {
                        if (!kernel[ky * kw + kx])
                            continue;

                        int ny = y + ky - ry;
                        int nx = x + kx - rx;

                        if (ny >= 0 && ny < h &&
                            nx >= 0 && nx < w) {
                            tmp[ny * w + nx] = 0;
                        }
                    }
                }
            }
        }
    }

    std::copy(tmp.begin(), tmp.end(), data);
}

// ------------------------------------------------------------
// IoU
// ------------------------------------------------------------

static float compute_iou(
    const std::vector<uint8_t>& a,
    const std::vector<uint8_t>& b,
    int n)
{
    int inter = 0;
    int uni = 0;

    for (int i = 0; i < n; ++i) {
        int ai = a[i] > 0;
        int bi = b[i] > 0;

        inter += ai & bi;
        uni += ai | bi;
    }

    return uni == 0
        ? 0.0f
        : static_cast<float>(inter) / uni;
}

// ------------------------------------------------------------
// Segmentation perturbation
// ------------------------------------------------------------

void deform_seg_core(
    const uint8_t* data,
    int h,
    int w,
    float iou_target,
    int seed,
    uint8_t* output)
{
    if (!data || !output)
        throw std::invalid_argument("Null input/output pointer");

    if (h < 0 || w < 0)
        throw std::invalid_argument("Negative dimensions");

    if (h == 0 || w == 0)
        return;

    const size_t n_size =
        static_cast<size_t>(h) * static_cast<size_t>(w);

    if (n_size > static_cast<size_t>(
            std::numeric_limits<int>::max())) {
        throw std::invalid_argument("Image is too large");
    }

    const int n = static_cast<int>(n_size);

    if (h <= 2 || w <= 2) {
        std::memcpy(output, data, n_size);
        return;
    }

    std::vector<uint8_t> gt(n);
    std::vector<uint8_t> seg(n);

    for (int i = 0; i < n; ++i)
        gt[i] = data[i] > 127 ? 255 : 0;

    seg = gt;

    std::mt19937 rng(
        seed >= 0
            ? static_cast<unsigned>(seed)
            : std::random_device{}());

    std::uniform_int_distribution<int> rw(0, w - 1);
    std::uniform_int_distribution<int> rh_dist(0, h - 1);
    std::uniform_int_distribution<int> ksize_dist(3, 9);
    std::uniform_real_distribution<float> prob(0.0f, 1.0f);

    std::vector<uint8_t> prev(n);

    for (int iter = 0; iter < 250; ++iter) {
        prev = seg;

        for (int k = 0; k < 4; ++k) {
            int lx = rw(rng);
            int ly = rh_dist(rng);

            std::uniform_int_distribution<int> lw_dist(lx + 1, w);
            std::uniform_int_distribution<int> lh_dist(ly + 1, h);

            int lw = lw_dist(rng);
            int lh = lh_dist(rng);

            int rw_ = lw - lx;
            int rh_ = lh - ly;

            if (prob(rng) < 0.25f) {
                int cx = (lx + lw) / 2;
                int cy = (ly + lh) / 2;

                seg[cy * w + cx] =
                    prob(rng) < 0.5f ? 255 : 0;
            }

            std::vector<uint8_t> roi(
                static_cast<size_t>(rw_) * rh_);

            for (int y = 0; y < rh_; ++y) {
                std::memcpy(
                    roi.data() + static_cast<size_t>(y) * rw_,
                    seg.data() + static_cast<size_t>(ly + y) * w + lx,
                    rw_);
            }

            int kw, kh;
            int size = ksize_dist(rng);

            auto kernel =
                get_random_structure(rng, size, kw, kh);

            if (prob(rng) < 0.5f) {
                dilate_region(
                    roi.data(), rh_, rw_, kernel, kw, kh);
            } else {
                erode_region(
                    roi.data(), rh_, rw_, kernel, kw, kh);
            }

            for (int y = 0; y < rh_; ++y) {
                std::memcpy(
                    seg.data() + static_cast<size_t>(ly + y) * w + lx,
                    roi.data() + static_cast<size_t>(y) * rw_,
                    rw_);
            }
        }

        float iou = compute_iou(seg, gt, n);

        if (iou < 0.01f) {
            std::memcpy(output, prev.data(), n_size);
            return;
        }

        if (iou < iou_target)
            break;
    }

    std::memcpy(output, seg.data(), n_size);
}

// ------------------------------------------------------------
// Contour processing
// ------------------------------------------------------------

static Contour deform_one_contour(
    const Contour& verts,
    float regional_sample_rate,
    float sample_rate,
    float move_rate,
    std::mt19937& rng)
{
    const int N = static_cast<int>(verts.size());

    if (N < 10)
        return {};

    if (!std::isfinite(regional_sample_rate) ||
        !std::isfinite(sample_rate) ||
        !std::isfinite(move_rate)) {
        throw std::invalid_argument("Rates must be finite");
    }

    if (regional_sample_rate < 0.0f ||
        regional_sample_rate >= 1.0f ||
        sample_rate <= 0.0f ||
        sample_rate > 1.0f ||
        move_rate < 0.0f) {
        throw std::invalid_argument("Invalid contour processing rate");
    }

    // Polygon centroid using signed area.
    double m00 = 0.0;
    double m10 = 0.0;
    double m01 = 0.0;

    for (int i = 0; i < N; ++i) {
        int x = verts[i][0];
        int y = verts[i][1];

        int j = (i + 1) % N;
        int xj = verts[j][0];
        int yj = verts[j][1];

        double cross =
            static_cast<double>(x) * yj -
            static_cast<double>(xj) * y;

        m00 += cross;
        m10 += (x + xj) * cross;
        m01 += (y + yj) * cross;
    }

    m00 *= 0.5;

    float cx = 0.0f;
    float cy = 0.0f;

    bool has_center = std::abs(m00) > 1e-5;

    if (has_center) {
        cx = static_cast<float>(m10 / (6.0 * m00));
        cy = static_cast<float>(m01 / (6.0 * m00));
    }

    // Regional removal.
    int n_remove =
        static_cast<int>(N * regional_sample_rate);

    int n_search = N - n_remove;

    std::vector<std::pair<float, int>> idx_dist(n_search);

    for (int i = 0; i < n_search; ++i) {
        float dx = static_cast<float>(
            verts[i][0] - verts[i + n_remove][0]);

        float dy = static_cast<float>(
            verts[i][1] - verts[i + n_remove][1]);

        idx_dist[i] = {dx * dx + dy * dy, i};
    }

    std::sort(idx_dist.begin(), idx_dist.end());

    int top10 = std::max(
        1, static_cast<int>(std::ceil(0.1f * n_search)));

    std::uniform_int_distribution<int> pick(0, top10 - 1);

    int remove_start = idx_dist[pick(rng)].second;

    Contour trimmed;
    trimmed.reserve(N - n_remove);

    for (int i = 0; i < remove_start; ++i)
        trimmed.push_back(verts[i]);

    for (int i = remove_start + n_remove; i < N; ++i)
        trimmed.push_back(verts[i]);

    // Sparse sampling.
    int N2 = static_cast<int>(trimmed.size());

    if (N2 == 0)
        return {};

    int n_keep = std::max(
        1, static_cast<int>(N2 * sample_rate));

    std::vector<int> indices(N2);
    std::iota(indices.begin(), indices.end(), 0);

    std::shuffle(indices.begin(), indices.end(), rng);
    indices.resize(n_keep);
    std::sort(indices.begin(), indices.end());

    // Vertex displacement.
    std::normal_distribution<float> gauss(0.0f, move_rate);

    Contour result;
    result.reserve(n_keep);

    for (int i = 0; i < n_keep; ++i) {
        float x = static_cast<float>(
            trimmed[indices[i]][0]);

        float y = static_cast<float>(
            trimmed[indices[i]][1]);

        if (has_center && move_rate > 0.0f) {
            float change = gauss(rng);

            x += (x - cx) * change;
            y += (y - cy) * change;
        }

        result.push_back({
            static_cast<int>(std::lround(x)),
            static_cast<int>(std::lround(y))
        });
    }

    return result;
}

// ------------------------------------------------------------
// Contour batch entry point
// ------------------------------------------------------------

std::vector<Contour> deform_contours_core(
    const std::vector<Contour>& contours,
    float regional_sample_rate,
    float sample_rate,
    float move_rate,
    int seed)
{
    std::mt19937 rng(
        seed >= 0
            ? static_cast<unsigned>(seed)
            : std::random_device{}());

    std::vector<Contour> output;
    output.reserve(contours.size());

    for (const auto& contour : contours) {
        Contour result = deform_one_contour(
            contour,
            regional_sample_rate,
            sample_rate,
            move_rate,
            rng);

        if (!result.empty())
            output.push_back(std::move(result));
    }

    return output;
}






/*********************************************************************************


                              PyBind11


**********************************************************************************/

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "C++ deformation operations";

    m.def(
        "deform_seg",
        [](torch::Tensor input,
           float iou_target,
           int seed) -> torch::Tensor {

            TORCH_CHECK(
                input.device().is_cpu(),
                "input must be a CPU tensor"
            );

            TORCH_CHECK(
                input.scalar_type() == torch::kUInt8,
                "input must have dtype torch.uint8"
            );

            TORCH_CHECK(
                input.dim() == 2,
                "input must have shape (H, W)"
            );

            input = input.contiguous();

            const int h = static_cast<int>(input.size(0));
            const int w = static_cast<int>(input.size(1));

            auto output = torch::empty_like(input);

            deform_seg_core(
                input.data_ptr<uint8_t>(),
                h,
                w,
                iou_target,
                seed,
                output.data_ptr<uint8_t>()
            );

            return output;
        },
        pybind11::arg("input"),
        pybind11::arg("iou_target"),
        pybind11::arg("seed") = -1
    );

    m.def(
        "deform_contours",
        [](const std::vector<std::vector<std::array<int, 2>>>& contours,
           float regional_sample_rate,
           float sample_rate,
           float move_rate,
           int seed) {

            return deform_contours_core(
                contours,
                regional_sample_rate,
                sample_rate,
                move_rate,
                seed
            );
        },
        pybind11::arg("contours"),
        pybind11::arg("regional_sample_rate"),
        pybind11::arg("sample_rate"),
        pybind11::arg("move_rate"),
        pybind11::arg("seed") = -1
    );
}
