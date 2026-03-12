/**
 * Sequential Image Processing
 * ─────────────────────────────
 * All operations run on a single CPU core without any parallelism.
 * Serves as the baseline for speedup comparisons.
 *
 * Operations implemented:
 *   1. Grayscale conversion
 *   2. Gaussian blur (5×5)
 *   3. Sobel edge detection
 *   4. Brightness / contrast adjustment
 *   5. Histogram equalization
 */

#include "image_processing.h"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

// ─── Timing ──────────────────────────────────────────────────────────────────
double get_time_seconds() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// ─── Helper: validate that src is non-empty ───────────────────────────────────
static void check_src(const cv::Mat& src, const char* fn) {
    if (src.empty())
        throw std::invalid_argument(std::string(fn) + ": source image is empty");
}

// ─── 1. Grayscale ────────────────────────────────────────────────────────────
void grayscale_sequential(const cv::Mat& src, cv::Mat& dst) {
    check_src(src, "grayscale_sequential");
    if (src.channels() == 1) {
        dst = src.clone();
        return;
    }
    dst.create(src.rows, src.cols, CV_8UC1);
    for (int r = 0; r < src.rows; ++r) {
        const cv::Vec3b* row_ptr = src.ptr<cv::Vec3b>(r);
        uchar* dst_ptr = dst.ptr<uchar>(r);
        for (int c = 0; c < src.cols; ++c) {
            // ITU-R BT.601 luma coefficients (OpenCV stores BGR)
            dst_ptr[c] = static_cast<uchar>(
                0.114 * row_ptr[c][0] +   // B
                0.587 * row_ptr[c][1] +   // G
                0.299 * row_ptr[c][2]);   // R
        }
    }
}

// ─── 2. Gaussian Blur (5×5, σ=1.4) ──────────────────────────────────────────
static const float GAUSS5[5][5] = {
    {1,  4,  7,  4, 1},
    {4, 16, 26, 16, 4},
    {7, 26, 41, 26, 7},
    {4, 16, 26, 16, 4},
    {1,  4,  7,  4, 1}
};
static const float GAUSS5_SUM = 273.0f;

void gaussian_blur_sequential(const cv::Mat& src, cv::Mat& dst) {
    check_src(src, "gaussian_blur_sequential");
    dst.create(src.rows, src.cols, src.type());
    const int kext = 2; // kernel extends 2 pixels in each direction

    for (int r = 0; r < src.rows; ++r) {
        for (int c = 0; c < src.cols; ++c) {
            if (src.channels() == 1) {
                float acc = 0;
                for (int kr = -kext; kr <= kext; ++kr) {
                    int rr = std::clamp(r + kr, 0, src.rows - 1);
                    for (int kc = -kext; kc <= kext; ++kc) {
                        int cc = std::clamp(c + kc, 0, src.cols - 1);
                        acc += src.at<uchar>(rr, cc) *
                               GAUSS5[kr + kext][kc + kext];
                    }
                }
                dst.at<uchar>(r, c) = static_cast<uchar>(acc / GAUSS5_SUM);
            } else {
                cv::Vec3f acc(0, 0, 0);
                for (int kr = -kext; kr <= kext; ++kr) {
                    int rr = std::clamp(r + kr, 0, src.rows - 1);
                    for (int kc = -kext; kc <= kext; ++kc) {
                        int cc = std::clamp(c + kc, 0, src.cols - 1);
                        float w = GAUSS5[kr + kext][kc + kext];
                        cv::Vec3b px = src.at<cv::Vec3b>(rr, cc);
                        acc[0] += px[0] * w;
                        acc[1] += px[1] * w;
                        acc[2] += px[2] * w;
                    }
                }
                dst.at<cv::Vec3b>(r, c) = cv::Vec3b(
                    static_cast<uchar>(acc[0] / GAUSS5_SUM),
                    static_cast<uchar>(acc[1] / GAUSS5_SUM),
                    static_cast<uchar>(acc[2] / GAUSS5_SUM));
            }
        }
    }
}

// ─── 3. Sobel Edge Detection ──────────────────────────────────────────────────
void sobel_edge_sequential(const cv::Mat& src, cv::Mat& dst) {
    check_src(src, "sobel_edge_sequential");

    // Work on grayscale
    cv::Mat gray;
    if (src.channels() != 1) grayscale_sequential(src, gray);
    else gray = src;

    dst.create(gray.rows, gray.cols, CV_8UC1);

    for (int r = 1; r < gray.rows - 1; ++r) {
        for (int c = 1; c < gray.cols - 1; ++c) {
            int gx =
                -1 * gray.at<uchar>(r-1, c-1) + 1 * gray.at<uchar>(r-1, c+1) +
                -2 * gray.at<uchar>(r,   c-1) + 2 * gray.at<uchar>(r,   c+1) +
                -1 * gray.at<uchar>(r+1, c-1) + 1 * gray.at<uchar>(r+1, c+1);

            int gy =
                -1 * gray.at<uchar>(r-1, c-1) + -2 * gray.at<uchar>(r-1, c) + -1 * gray.at<uchar>(r-1, c+1) +
                 1 * gray.at<uchar>(r+1, c-1) +  2 * gray.at<uchar>(r+1, c) +  1 * gray.at<uchar>(r+1, c+1);

            int mag = static_cast<int>(std::sqrt(gx * gx + gy * gy));
            dst.at<uchar>(r, c) = static_cast<uchar>(std::min(mag, 255));
        }
    }
    // Border pixels set to 0
    for (int c = 0; c < dst.cols; ++c) { dst.at<uchar>(0, c) = 0; dst.at<uchar>(dst.rows-1, c) = 0; }
    for (int r = 0; r < dst.rows; ++r) { dst.at<uchar>(r, 0) = 0; dst.at<uchar>(r, dst.cols-1) = 0; }
}

// ─── 4. Brightness / Contrast: dst = alpha*src + beta ────────────────────────
void brightness_contrast_sequential(const cv::Mat& src, cv::Mat& dst,
                                    double alpha, int beta) {
    check_src(src, "brightness_contrast_sequential");
    dst.create(src.rows, src.cols, src.type());
    for (int r = 0; r < src.rows; ++r) {
        for (int c = 0; c < src.cols; ++c) {
            if (src.channels() == 1) {
                int v = static_cast<int>(src.at<uchar>(r, c) * alpha + beta);
                dst.at<uchar>(r, c) = static_cast<uchar>(std::clamp(v, 0, 255));
            } else {
                cv::Vec3b px = src.at<cv::Vec3b>(r, c);
                dst.at<cv::Vec3b>(r, c) = cv::Vec3b(
                    static_cast<uchar>(std::clamp(static_cast<int>(px[0]*alpha+beta), 0, 255)),
                    static_cast<uchar>(std::clamp(static_cast<int>(px[1]*alpha+beta), 0, 255)),
                    static_cast<uchar>(std::clamp(static_cast<int>(px[2]*alpha+beta), 0, 255)));
            }
        }
    }
}

// ─── 5. Histogram Equalization (grayscale) ───────────────────────────────────
void histogram_equalize_sequential(const cv::Mat& src, cv::Mat& dst) {
    check_src(src, "histogram_equalize_sequential");
    cv::Mat gray;
    if (src.channels() != 1) grayscale_sequential(src, gray);
    else gray = src;

    // Build histogram
    int hist[256] = {};
    for (int r = 0; r < gray.rows; ++r)
        for (int c = 0; c < gray.cols; ++c)
            hist[gray.at<uchar>(r, c)]++;

    // Cumulative distribution
    int total = gray.rows * gray.cols;
    int cdf[256] = {};
    cdf[0] = hist[0];
    for (int i = 1; i < 256; ++i) cdf[i] = cdf[i-1] + hist[i];

    // Find first non-zero cdf
    int cdf_min = 0;
    for (int i = 0; i < 256; ++i) { if (cdf[i] > 0) { cdf_min = cdf[i]; break; } }

    // LUT
    uchar lut[256] = {};
    for (int i = 0; i < 256; ++i) {
        if (total == cdf_min) { lut[i] = 0; continue; }
        lut[i] = static_cast<uchar>(
            std::round((double)(cdf[i] - cdf_min) / (total - cdf_min) * 255.0));
    }

    // Apply LUT
    dst.create(gray.rows, gray.cols, CV_8UC1);
    for (int r = 0; r < gray.rows; ++r) {
        const uchar* sp = gray.ptr<uchar>(r);
        uchar* dp = dst.ptr<uchar>(r);
        for (int c = 0; c < gray.cols; ++c)
            dp[c] = lut[sp[c]];
    }
}

// ─── CSV Result Writer ───────────────────────────────────────────────────────
void save_results_csv(const std::vector<BenchmarkResult>& results,
                      const std::string& filepath) {
    std::ofstream f(filepath);
    if (!f) throw std::runtime_error("Cannot open " + filepath);
    f << "version,operation,width,height,threads,processes,elapsed_sec,speedup\n";
    for (const auto& r : results) {
        f << r.version << ',' << r.operation << ','
          << r.image_width << ',' << r.image_height << ','
          << r.threads << ',' << r.processes << ','
          << r.elapsed_sec << ',' << r.speedup << '\n';
    }
}

// ─── Main ─────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    std::string input_path  = "test_images/sample.jpg";
    std::string output_dir  = "results/images";
    std::string results_csv = "results/data/sequential_results.csv";

    if (argc > 1) input_path = argv[1];

    cv::Mat img = cv::imread(input_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Error: cannot load image '" << input_path << "'\n";
        std::cerr << "Usage: " << argv[0] << " [image_path]\n";
        return 1;
    }

    std::cout << "Sequential Image Processing\n";
    std::cout << "Image: " << input_path
              << " (" << img.cols << "x" << img.rows << ")\n\n";

    std::vector<BenchmarkResult> results;
    cv::Mat dst;

    // Helper lambda to benchmark one operation
    auto bench = [&](const std::string& op, auto fn) {
        double t0 = get_time_seconds();
        fn();
        double elapsed = get_time_seconds() - t0;

        BenchmarkResult r;
        r.version    = "sequential";
        r.operation  = op;
        r.image_width  = img.cols;
        r.image_height = img.rows;
        r.threads    = 1;
        r.processes  = 1;
        r.elapsed_sec = elapsed;
        r.speedup    = 1.0; // baseline

        std::printf("  %-25s %.4f s\n", op.c_str(), elapsed);
        cv::imwrite(output_dir + "/seq_" + op + ".png", dst);
        results.push_back(r);
    };

    bench("grayscale",   [&]{ grayscale_sequential(img, dst); });
    bench("gaussian_blur",[&]{ gaussian_blur_sequential(img, dst); });
    bench("sobel_edge",  [&]{ sobel_edge_sequential(img, dst); });
    bench("brightness",  [&]{ brightness_contrast_sequential(img, dst, 1.2, 30); });
    bench("histogram_eq",[&]{ histogram_equalize_sequential(img, dst); });

    save_results_csv(results, results_csv);
    std::cout << "\nResults saved to " << results_csv << "\n";
    return 0;
}
