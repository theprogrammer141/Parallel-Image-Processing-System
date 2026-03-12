/**
 * OpenMP Parallel Image Processing
 * ──────────────────────────────────
 * Each operation parallelises the outer pixel loop with OpenMP, distributing
 * rows across available CPU threads.  The number of threads is controlled by
 * the OMP_NUM_THREADS environment variable or the -t command-line argument.
 *
 * Operations:
 *   1. Grayscale conversion
 *   2. Gaussian blur (5×5)
 *   3. Sobel edge detection
 *   4. Brightness / contrast adjustment
 *   5. Histogram equalization (parallel histogram + CDF, serial LUT build)
 */

#include "image_processing.h"
#include <omp.h>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

// ─── Timing ──────────────────────────────────────────────────────────────────
// Use omp_get_wtime() for wall-clock timing under OpenMP
static double wtime() { return omp_get_wtime(); }

// ─── 1. Grayscale ────────────────────────────────────────────────────────────
void grayscale_omp(const cv::Mat& src, cv::Mat& dst, int nthreads) {
    if (src.empty()) throw std::invalid_argument("grayscale_omp: empty image");
    if (src.channels() == 1) { dst = src.clone(); return; }
    dst.create(src.rows, src.cols, CV_8UC1);

    #pragma omp parallel for schedule(static) num_threads(nthreads)
    for (int r = 0; r < src.rows; ++r) {
        const cv::Vec3b* rp = src.ptr<cv::Vec3b>(r);
        uchar* dp = dst.ptr<uchar>(r);
        for (int c = 0; c < src.cols; ++c)
            dp[c] = static_cast<uchar>(
                0.114 * rp[c][0] + 0.587 * rp[c][1] + 0.299 * rp[c][2]);
    }
}

// ─── Gaussian kernel (5×5) ───────────────────────────────────────────────────
static const float G5[5][5] = {
    {1,  4,  7,  4, 1},
    {4, 16, 26, 16, 4},
    {7, 26, 41, 26, 7},
    {4, 16, 26, 16, 4},
    {1,  4,  7,  4, 1}
};
static constexpr float G5_SUM = 273.0f;

// ─── 2. Gaussian Blur ────────────────────────────────────────────────────────
void gaussian_blur_omp(const cv::Mat& src, cv::Mat& dst, int nthreads) {
    if (src.empty()) throw std::invalid_argument("gaussian_blur_omp: empty image");
    dst.create(src.rows, src.cols, src.type());

    #pragma omp parallel for schedule(static) num_threads(nthreads)
    for (int r = 0; r < src.rows; ++r) {
        for (int c = 0; c < src.cols; ++c) {
            if (src.channels() == 1) {
                float acc = 0;
                for (int kr = -2; kr <= 2; ++kr) {
                    int rr = std::clamp(r + kr, 0, src.rows - 1);
                    for (int kc = -2; kc <= 2; ++kc) {
                        int cc = std::clamp(c + kc, 0, src.cols - 1);
                        acc += src.at<uchar>(rr, cc) * G5[kr+2][kc+2];
                    }
                }
                dst.at<uchar>(r, c) = static_cast<uchar>(acc / G5_SUM);
            } else {
                cv::Vec3f acc(0,0,0);
                for (int kr = -2; kr <= 2; ++kr) {
                    int rr = std::clamp(r + kr, 0, src.rows - 1);
                    for (int kc = -2; kc <= 2; ++kc) {
                        int cc = std::clamp(c + kc, 0, src.cols - 1);
                        float w = G5[kr+2][kc+2];
                        cv::Vec3b px = src.at<cv::Vec3b>(rr, cc);
                        acc[0] += px[0]*w; acc[1] += px[1]*w; acc[2] += px[2]*w;
                    }
                }
                dst.at<cv::Vec3b>(r, c) = cv::Vec3b(
                    static_cast<uchar>(acc[0]/G5_SUM),
                    static_cast<uchar>(acc[1]/G5_SUM),
                    static_cast<uchar>(acc[2]/G5_SUM));
            }
        }
    }
}

// ─── 3. Sobel Edge Detection ─────────────────────────────────────────────────
void sobel_edge_omp(const cv::Mat& src, cv::Mat& dst, int nthreads) {
    if (src.empty()) throw std::invalid_argument("sobel_edge_omp: empty image");

    cv::Mat gray;
    if (src.channels() != 1) grayscale_omp(src, gray, nthreads);
    else gray = src;

    dst.create(gray.rows, gray.cols, CV_8UC1);
    dst.setTo(0);

    #pragma omp parallel for schedule(static) num_threads(nthreads)
    for (int r = 1; r < gray.rows - 1; ++r) {
        for (int c = 1; c < gray.cols - 1; ++c) {
            int gx =
                -gray.at<uchar>(r-1,c-1) + gray.at<uchar>(r-1,c+1)
                -2*gray.at<uchar>(r,c-1) + 2*gray.at<uchar>(r,c+1)
                -gray.at<uchar>(r+1,c-1) + gray.at<uchar>(r+1,c+1);
            int gy =
                -gray.at<uchar>(r-1,c-1) - 2*gray.at<uchar>(r-1,c) - gray.at<uchar>(r-1,c+1)
                +gray.at<uchar>(r+1,c-1) + 2*gray.at<uchar>(r+1,c) + gray.at<uchar>(r+1,c+1);
            dst.at<uchar>(r, c) =
                static_cast<uchar>(std::min(static_cast<int>(std::sqrt(gx*gx + gy*gy)), 255));
        }
    }
}

// ─── 4. Brightness / Contrast ────────────────────────────────────────────────
void brightness_contrast_omp(const cv::Mat& src, cv::Mat& dst,
                              double alpha, int beta, int nthreads) {
    if (src.empty()) throw std::invalid_argument("brightness_contrast_omp: empty image");
    dst.create(src.rows, src.cols, src.type());

    #pragma omp parallel for schedule(static) num_threads(nthreads)
    for (int r = 0; r < src.rows; ++r) {
        for (int c = 0; c < src.cols; ++c) {
            if (src.channels() == 1) {
                int v = static_cast<int>(src.at<uchar>(r,c)*alpha + beta);
                dst.at<uchar>(r,c) = static_cast<uchar>(std::clamp(v,0,255));
            } else {
                cv::Vec3b px = src.at<cv::Vec3b>(r,c);
                dst.at<cv::Vec3b>(r,c) = cv::Vec3b(
                    static_cast<uchar>(std::clamp(static_cast<int>(px[0]*alpha+beta),0,255)),
                    static_cast<uchar>(std::clamp(static_cast<int>(px[1]*alpha+beta),0,255)),
                    static_cast<uchar>(std::clamp(static_cast<int>(px[2]*alpha+beta),0,255)));
            }
        }
    }
}

// ─── 5. Histogram Equalization ───────────────────────────────────────────────
void histogram_equalize_omp(const cv::Mat& src, cv::Mat& dst, int nthreads) {
    if (src.empty()) throw std::invalid_argument("histogram_equalize_omp: empty image");

    cv::Mat gray;
    if (src.channels() != 1) grayscale_omp(src, gray, nthreads);
    else gray = src;

    int total = gray.rows * gray.cols;

    // Parallel histogram using thread-local arrays, then reduction
    int hist[256] = {};
    #pragma omp parallel num_threads(nthreads)
    {
        int local_hist[256] = {};
        #pragma omp for schedule(static) nowait
        for (int r = 0; r < gray.rows; ++r) {
            const uchar* rp = gray.ptr<uchar>(r);
            for (int c = 0; c < gray.cols; ++c)
                local_hist[rp[c]]++;
        }
        #pragma omp critical
        for (int i = 0; i < 256; ++i) hist[i] += local_hist[i];
    }

    // CDF and LUT (serial — O(256), negligible)
    int cdf[256] = {};
    cdf[0] = hist[0];
    for (int i = 1; i < 256; ++i) cdf[i] = cdf[i-1] + hist[i];
    int cdf_min = 0;
    for (int i = 0; i < 256; ++i) { if (cdf[i]>0) { cdf_min=cdf[i]; break; } }

    uchar lut[256] = {};
    for (int i = 0; i < 256; ++i) {
        if (total == cdf_min) { lut[i] = 0; continue; }
        lut[i] = static_cast<uchar>(
            std::round((double)(cdf[i]-cdf_min)/(total-cdf_min)*255.0));
    }

    // Apply LUT in parallel
    dst.create(gray.rows, gray.cols, CV_8UC1);
    #pragma omp parallel for schedule(static) num_threads(nthreads)
    for (int r = 0; r < gray.rows; ++r) {
        const uchar* sp = gray.ptr<uchar>(r);
        uchar* dp = dst.ptr<uchar>(r);
        for (int c = 0; c < gray.cols; ++c)
            dp[c] = lut[sp[c]];
    }
}

// ─── CSV Writer (reuse from sequential) ──────────────────────────────────────
static void save_csv(const std::vector<BenchmarkResult>& results,
                     const std::string& path) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot open " + path);
    f << "version,operation,width,height,threads,processes,elapsed_sec,speedup\n";
    for (const auto& r : results)
        f << r.version<<','<<r.operation<<','
          <<r.image_width<<','<<r.image_height<<','
          <<r.threads<<','<<r.processes<<','
          <<r.elapsed_sec<<','<<r.speedup<<'\n';
}

// ─── Main ─────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    std::string input  = "test_images/sample.jpg";
    std::string outdir = "results/images";
    std::string csv    = "results/data/openmp_results.csv";
    int nthreads = omp_get_max_threads();

    // Simple arg parsing: ./openmp_proc [image] [threads]
    if (argc > 1) input    = argv[1];
    if (argc > 2) nthreads = std::stoi(argv[2]);

    cv::Mat img = cv::imread(input, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Error: cannot load '" << input << "'\n";
        std::cerr << "Usage: " << argv[0] << " [image] [threads]\n";
        return 1;
    }

    std::cout << "OpenMP Image Processing  |  threads: " << nthreads << '\n';
    std::cout << "Image: " << input << " (" << img.cols << "x" << img.rows << ")\n\n";

    std::vector<BenchmarkResult> res;
    cv::Mat dst;

    auto bench = [&](const std::string& op, auto fn) {
        double t0 = wtime();
        fn();
        double el = wtime() - t0;
        BenchmarkResult r;
        r.version    = "openmp";
        r.operation  = op;
        r.image_width  = img.cols;
        r.image_height = img.rows;
        r.threads    = nthreads;
        r.processes  = 1;
        r.elapsed_sec = el;
        r.speedup    = 0.0; // filled in by benchmark script
        std::printf("  %-25s %d threads  %.4f s\n", op.c_str(), nthreads, el);
        cv::imwrite(outdir + "/omp_" + op + "_t" + std::to_string(nthreads) + ".png", dst);
        res.push_back(r);
    };

    bench("grayscale",    [&]{ grayscale_omp(img, dst, nthreads); });
    bench("gaussian_blur",[&]{ gaussian_blur_omp(img, dst, nthreads); });
    bench("sobel_edge",   [&]{ sobel_edge_omp(img, dst, nthreads); });
    bench("brightness",   [&]{ brightness_contrast_omp(img, dst, 1.2, 30, nthreads); });
    bench("histogram_eq", [&]{ histogram_equalize_omp(img, dst, nthreads); });

    save_csv(res, csv);
    std::cout << "\nResults saved to " << csv << "\n";
    return 0;
}
