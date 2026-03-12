#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// ─── Image Processing Operations ────────────────────────────────────────────

// Grayscale conversion
void grayscale_sequential(const cv::Mat& src, cv::Mat& dst);

// Gaussian blur (5x5 kernel)
void gaussian_blur_sequential(const cv::Mat& src, cv::Mat& dst);

// Sobel edge detection
void sobel_edge_sequential(const cv::Mat& src, cv::Mat& dst);

// Brightness / contrast adjustment
void brightness_contrast_sequential(const cv::Mat& src, cv::Mat& dst,
                                    double alpha, int beta);

// Histogram equalization (grayscale)
void histogram_equalize_sequential(const cv::Mat& src, cv::Mat& dst);

// ─── Timing Utility ──────────────────────────────────────────────────────────
double get_time_seconds();

// ─── Result Record ───────────────────────────────────────────────────────────
struct BenchmarkResult {
    std::string version;   // sequential / openmp / mpi / hybrid
    std::string operation; // grayscale / blur / edge / brightness / histogram
    int image_width;
    int image_height;
    int threads;           // OpenMP threads (1 for sequential/mpi)
    int processes;         // MPI processes (1 for sequential/openmp)
    double elapsed_sec;
    double speedup;        // relative to sequential baseline
};

void save_results_csv(const std::vector<BenchmarkResult>& results,
                      const std::string& filepath);

#endif // IMAGE_PROCESSING_H
