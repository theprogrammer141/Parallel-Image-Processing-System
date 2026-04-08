/**
 * MPI Distributed Image Processing
 * ──────────────────────────────────
 * Strategy:
 *   - Rank 0 loads the image, broadcasts dimensions, then scatters row-strips
 *     to all ranks using MPI_Scatterv.
 *   - Each rank processes its local strip independently (no inter-rank
 *     communication needed for embarrassingly parallel ops).
 *   - Results are gathered back to rank 0 with MPI_Gatherv.
 *   - Rank 0 writes the output image and timing CSV.
 *
 * Sobel: border rows that are shared between strips are broadcast as halo rows
 * so each rank has the one extra row above/below it needs.
 *
 * Operations:
 *   1. Grayscale
 *   2. Gaussian blur (5×5)  – 2-row halo
 *   3. Sobel edge detection  – 1-row halo
 *   4. Brightness / contrast
 *   5. Histogram equalization (parallel local hist → MPI_Reduce → broadcast LUT)
 */

#include "image_processing.h"
#include <mpi.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

// ─── Helpers ─────────────────────────────────────────────────────────────────
static double wtime() { return MPI_Wtime(); }

static const uchar* root_contiguous_ptr(const cv::Mat& src, cv::Mat& storage) {
    if (src.isContinuous()) return src.data;
    storage = src.clone();
    return storage.data;
}

static cv::Mat vec_view(std::vector<uchar>& v, int rows, int cols, int type) {
    if (rows <= 0 || cols <= 0) return cv::Mat(rows, cols, type);
    return cv::Mat(rows, cols, type, v.data());
}

static void pack_color_halo_rows(const std::vector<uchar>& local_src,
                                 int local_rows, int W, int halo,
                                 bool top, std::vector<uchar>& out) {
    out.assign(halo * W * 3, 0);
    if (local_rows <= 0) return;

    for (int h = 0; h < halo; ++h) {
        int src_row = 0;
        if (top) {
            src_row = std::min(h, local_rows - 1);
        } else {
            src_row = std::max(local_rows - halo + h, 0);
        }
        const uchar* row_ptr = local_src.data() + src_row * W * 3;
        std::copy(row_ptr, row_ptr + W * 3, out.data() + h * W * 3);
    }
}

static void pack_gray_halo_rows(const std::vector<uchar>& local_gray,
                                int local_rows, int W, int halo,
                                bool top, std::vector<uchar>& out) {
    out.assign(halo * W, 0);
    if (local_rows <= 0) return;

    for (int h = 0; h < halo; ++h) {
        int src_row = 0;
        if (top) {
            src_row = std::min(h, local_rows - 1);
        } else {
            src_row = std::max(local_rows - halo + h, 0);
        }
        const uchar* row_ptr = local_gray.data() + src_row * W;
        std::copy(row_ptr, row_ptr + W, out.data() + h * W);
    }
}

// ─── Build scatter/gather layout ─────────────────────────────────────────────
struct ScatterLayout {
    std::vector<int> counts;  // bytes per rank
    std::vector<int> displs;  // byte offset per rank
    std::vector<int> rows_per_rank;
    std::vector<int> row_offsets;
    int elem_size;
};

static ScatterLayout make_layout(int total_rows, int nprocs, int cols, int elem_size) {
    ScatterLayout sl;
    sl.elem_size = elem_size;
    sl.rows_per_rank.resize(nprocs);
    sl.row_offsets.resize(nprocs);
    sl.counts.resize(nprocs);
    sl.displs.resize(nprocs);

    int base = total_rows / nprocs;
    int rem  = total_rows % nprocs;
    int off  = 0;
    for (int i = 0; i < nprocs; ++i) {
        sl.rows_per_rank[i] = base + (i < rem ? 1 : 0);
        sl.row_offsets[i]   = off;
        sl.counts[i]        = sl.rows_per_rank[i] * cols * elem_size;
        sl.displs[i]        = off * cols * elem_size;
        off += sl.rows_per_rank[i];
    }
    return sl;
}

// ─── 1. Grayscale ────────────────────────────────────────────────────────────
static void mpi_grayscale(const cv::Mat& img, cv::Mat& dst,
                           int rank, int nprocs,
                           MPI_Comm comm, double& elapsed) {
    int W = 0, H = 0;
    if (rank == 0) { W = img.cols; H = img.rows; }
    MPI_Bcast(&W, 1, MPI_INT, 0, comm);
    MPI_Bcast(&H, 1, MPI_INT, 0, comm);

    ScatterLayout sl = make_layout(H, nprocs, W, 3); // BGR

    // Scatter source
    std::vector<uchar> local_src(sl.rows_per_rank[rank] * W * 3);
    cv::Mat root_src_storage;
    const uchar* root_src = nullptr;
    if (rank == 0) root_src = root_contiguous_ptr(img, root_src_storage);

    MPI_Barrier(comm);
    double t0 = wtime();

    MPI_Scatterv(root_src, sl.counts.data(), sl.displs.data(), MPI_UNSIGNED_CHAR,
                 local_src.data(), sl.counts[rank], MPI_UNSIGNED_CHAR, 0, comm);

    // Local compute
    int local_rows = sl.rows_per_rank[rank];
    cv::Mat local_bgr = vec_view(local_src, local_rows, W, CV_8UC3);
    cv::Mat local_gray(local_rows, W, CV_8UC1);
    for (int r = 0; r < local_rows; ++r) {
        const cv::Vec3b* sp = local_bgr.ptr<cv::Vec3b>(r);
        uchar* dp = local_gray.ptr<uchar>(r);
        for (int c = 0; c < W; ++c)
            dp[c] = static_cast<uchar>(0.114*sp[c][0]+0.587*sp[c][1]+0.299*sp[c][2]);
    }

    // Gather
    ScatterLayout sl_gray = make_layout(H, nprocs, W, 1);
    if (rank == 0) dst.create(H, W, CV_8UC1);

    MPI_Gatherv(local_rows > 0 ? local_gray.data : nullptr, sl_gray.counts[rank], MPI_UNSIGNED_CHAR,
                rank == 0 ? dst.data : nullptr, sl_gray.counts.data(), sl_gray.displs.data(),
                MPI_UNSIGNED_CHAR, 0, comm);

    elapsed = wtime() - t0;
}

// ─── 2. Gaussian Blur (each rank gets 2-row halo above/below) ────────────────
static const float G5[5][5] = {
    {1,  4,  7,  4, 1},
    {4, 16, 26, 16, 4},
    {7, 26, 41, 26, 7},
    {4, 16, 26, 16, 4},
    {1,  4,  7,  4, 1}
};
static constexpr float G5_SUM = 273.0f;

static void mpi_gaussian_blur(const cv::Mat& img, cv::Mat& dst,
                               int rank, int nprocs,
                               MPI_Comm comm, double& elapsed) {
    int W = 0, H = 0;
    if (rank == 0) { W = img.cols; H = img.rows; }
    MPI_Bcast(&W, 1, MPI_INT, 0, comm);
    MPI_Bcast(&H, 1, MPI_INT, 0, comm);

    // Always treat as BGR color for blur
    ScatterLayout sl = make_layout(H, nprocs, W, 3);

    std::vector<uchar> local_src(sl.rows_per_rank[rank] * W * 3);
    cv::Mat root_src_storage;
    const uchar* root_src = nullptr;
    if (rank == 0) root_src = root_contiguous_ptr(img, root_src_storage);

    MPI_Barrier(comm);
    double t0 = wtime();

    MPI_Scatterv(root_src, sl.counts.data(), sl.displs.data(), MPI_UNSIGNED_CHAR,
                 local_src.data(), sl.counts[rank], MPI_UNSIGNED_CHAR, 0, comm);

    int local_rows = sl.rows_per_rank[rank];

    // Exchange 2-row halos with neighbours
    int halo = 2;
    int halo_count = halo * W * 3;
    int prev = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int next = (rank + 1 < nprocs) ? rank + 1 : MPI_PROC_NULL;

    std::vector<uchar> send_top;
    std::vector<uchar> send_bottom;
    pack_color_halo_rows(local_src, local_rows, W, halo, true, send_top);
    pack_color_halo_rows(local_src, local_rows, W, halo, false, send_bottom);

    std::vector<uchar> halo_above(halo * W * 3, 0);
    std::vector<uchar> halo_below(halo * W * 3, 0);

    MPI_Sendrecv(send_top.data(), halo_count, MPI_UNSIGNED_CHAR, prev, 10,
                 halo_below.data(), halo_count, MPI_UNSIGNED_CHAR, next, 10,
                 comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(send_bottom.data(), halo_count, MPI_UNSIGNED_CHAR, next, 11,
                 halo_above.data(), halo_count, MPI_UNSIGNED_CHAR, prev, 11,
                 comm, MPI_STATUS_IGNORE);

    // Build extended image (halo_above + local + halo_below)
    int ext_rows = halo + local_rows + halo;
    std::vector<uchar> ext(ext_rows * W * 3, 0);
    std::copy(halo_above.begin(), halo_above.end(), ext.begin());
    if (local_rows > 0) {
        std::copy(local_src.begin(), local_src.end(), ext.begin() + halo*W*3);
    }
    std::copy(halo_below.begin(), halo_below.end(), ext.begin() + (halo+local_rows)*W*3);

    // Fill edge halos with border replication
    if (prev == MPI_PROC_NULL && local_rows > 0)
        for (int h = 0; h < halo; ++h)
            std::copy(ext.begin() + halo*W*3, ext.begin() + (halo+1)*W*3,
                      ext.begin() + h*W*3);
    if (next == MPI_PROC_NULL && local_rows > 0)
        for (int h = 0; h < halo; ++h)
            std::copy(ext.begin() + (halo+local_rows-1)*W*3,
                      ext.begin() + (halo+local_rows)*W*3,
                      ext.begin() + (halo+local_rows+h)*W*3);

    cv::Mat ext_mat = vec_view(ext, ext_rows, W, CV_8UC3);
    cv::Mat local_out(local_rows, W, CV_8UC3);

    for (int r = 0; r < local_rows; ++r) {
        for (int c = 0; c < W; ++c) {
            cv::Vec3f acc(0,0,0);
            for (int kr = -halo; kr <= halo; ++kr) {
                int rr = r + halo + kr; // index in ext_mat
                for (int kc = -halo; kc <= halo; ++kc) {
                    int cc = std::clamp(c + kc, 0, W-1);
                    float w = G5[kr+halo][kc+halo];
                    cv::Vec3b px = ext_mat.at<cv::Vec3b>(rr, cc);
                    acc[0]+=px[0]*w; acc[1]+=px[1]*w; acc[2]+=px[2]*w;
                }
            }
            local_out.at<cv::Vec3b>(r,c) = cv::Vec3b(
                static_cast<uchar>(acc[0]/G5_SUM),
                static_cast<uchar>(acc[1]/G5_SUM),
                static_cast<uchar>(acc[2]/G5_SUM));
        }
    }

    ScatterLayout sl_out = make_layout(H, nprocs, W, 3);
    if (rank == 0) dst.create(H, W, CV_8UC3);
    MPI_Gatherv(local_rows > 0 ? local_out.data : nullptr, sl_out.counts[rank], MPI_UNSIGNED_CHAR,
                rank == 0 ? dst.data : nullptr, sl_out.counts.data(), sl_out.displs.data(),
                MPI_UNSIGNED_CHAR, 0, comm);

    elapsed = wtime() - t0;
}

// ─── 3. Sobel Edge Detection ─────────────────────────────────────────────────
static void local_grayscale(const cv::Mat& src, cv::Mat& gray) {
    if (src.channels() == 1) { gray = src; return; }
    gray.create(src.rows, src.cols, CV_8UC1);
    for (int r = 0; r < src.rows; ++r) {
        const cv::Vec3b* sp = src.ptr<cv::Vec3b>(r);
        uchar* dp = gray.ptr<uchar>(r);
        for (int c = 0; c < src.cols; ++c)
            dp[c] = static_cast<uchar>(0.114*sp[c][0]+0.587*sp[c][1]+0.299*sp[c][2]);
    }
}

static void mpi_sobel(const cv::Mat& img, cv::Mat& dst,
                      int rank, int nprocs,
                      MPI_Comm comm, double& elapsed) {
    int W = 0, H = 0;
    if (rank == 0) { W = img.cols; H = img.rows; }
    MPI_Bcast(&W, 1, MPI_INT, 0, comm);
    MPI_Bcast(&H, 1, MPI_INT, 0, comm);

    // Convert to grayscale on rank 0 before scatter
    cv::Mat root_gray;
    const uchar* root_gray_ptr = nullptr;
    if (rank == 0) {
        local_grayscale(img, root_gray);
        if (!root_gray.isContinuous()) root_gray = root_gray.clone();
        root_gray_ptr = root_gray.data;
    }

    ScatterLayout sl = make_layout(H, nprocs, W, 1);
    std::vector<uchar> local_g(sl.rows_per_rank[rank] * W);

    MPI_Barrier(comm);
    double t0 = wtime();

    MPI_Scatterv(root_gray_ptr, sl.counts.data(), sl.displs.data(), MPI_UNSIGNED_CHAR,
                 local_g.data(), sl.counts[rank], MPI_UNSIGNED_CHAR, 0, comm);

    int local_rows = sl.rows_per_rank[rank];
    int halo = 1;
    int halo_count = halo * W;

    // Exchange 1-row halos
    int prev = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int next = (rank + 1 < nprocs) ? rank + 1 : MPI_PROC_NULL;

    std::vector<uchar> send_top;
    std::vector<uchar> send_bottom;
    pack_gray_halo_rows(local_g, local_rows, W, halo, true, send_top);
    pack_gray_halo_rows(local_g, local_rows, W, halo, false, send_bottom);

    std::vector<uchar> halo_above(W, 0), halo_below(W, 0);
    MPI_Sendrecv(send_top.data(), halo_count, MPI_UNSIGNED_CHAR, prev, 20,
                 halo_below.data(), halo_count, MPI_UNSIGNED_CHAR, next, 20,
                 comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(send_bottom.data(), halo_count, MPI_UNSIGNED_CHAR, next, 21,
                 halo_above.data(), halo_count, MPI_UNSIGNED_CHAR, prev, 21,
                 comm, MPI_STATUS_IGNORE);

    // Build extended strip
    int ext_rows = halo + local_rows + halo;
    std::vector<uchar> ext(ext_rows * W, 0);
    std::copy(halo_above.begin(), halo_above.end(), ext.begin());
    if (local_rows > 0) {
        std::copy(local_g.begin(), local_g.end(), ext.begin() + halo*W);
    }
    std::copy(halo_below.begin(), halo_below.end(), ext.begin() + (halo+local_rows)*W);

    // Edge replication
    if (prev == MPI_PROC_NULL && local_rows > 0)
        std::copy(ext.begin()+halo*W, ext.begin()+(halo+1)*W, ext.begin());
    if (next == MPI_PROC_NULL && local_rows > 0)
        std::copy(ext.begin()+(halo+local_rows-1)*W,
                                  ext.begin()+(halo+local_rows)*W,
                                  ext.begin()+(halo+local_rows)*W);

    cv::Mat ext_mat = vec_view(ext, ext_rows, W, CV_8UC1);
    cv::Mat local_out(local_rows, W, CV_8UC1);
    local_out.setTo(0);

    for (int r = 0; r < local_rows; ++r) {
        int rr = r + halo;
        for (int c = 1; c < W - 1; ++c) {
            int gx =
                -ext_mat.at<uchar>(rr-1,c-1) + ext_mat.at<uchar>(rr-1,c+1)
                -2*ext_mat.at<uchar>(rr,c-1)  + 2*ext_mat.at<uchar>(rr,c+1)
                -ext_mat.at<uchar>(rr+1,c-1) + ext_mat.at<uchar>(rr+1,c+1);
            int gy =
                -ext_mat.at<uchar>(rr-1,c-1) - 2*ext_mat.at<uchar>(rr-1,c) - ext_mat.at<uchar>(rr-1,c+1)
                +ext_mat.at<uchar>(rr+1,c-1) + 2*ext_mat.at<uchar>(rr+1,c) + ext_mat.at<uchar>(rr+1,c+1);
            local_out.at<uchar>(r,c) =
                static_cast<uchar>(std::min(static_cast<int>(std::sqrt(gx*gx+gy*gy)),255));
        }
        local_out.at<uchar>(r,0)   = 0;
        local_out.at<uchar>(r,W-1) = 0;
    }
    // Top/bottom border
    if (local_rows > 0 && rank == 0)        local_out.row(0).setTo(0);
    if (local_rows > 0 && rank == nprocs-1) local_out.row(local_rows-1).setTo(0);

    ScatterLayout sl_out = make_layout(H, nprocs, W, 1);
    if (rank == 0) dst.create(H, W, CV_8UC1);
    MPI_Gatherv(local_rows > 0 ? local_out.data : nullptr, sl_out.counts[rank], MPI_UNSIGNED_CHAR,
                rank == 0 ? dst.data : nullptr, sl_out.counts.data(), sl_out.displs.data(),
                MPI_UNSIGNED_CHAR, 0, comm);

    elapsed = wtime() - t0;
}

// ─── 4. Brightness / Contrast ────────────────────────────────────────────────
static void mpi_brightness(const cv::Mat& img, cv::Mat& dst,
                            double alpha, int beta,
                            int rank, int nprocs,
                            MPI_Comm comm, double& elapsed) {
    int W = 0, H = 0;
    if (rank == 0) { W = img.cols; H = img.rows; }
    MPI_Bcast(&W, 1, MPI_INT, 0, comm);
    MPI_Bcast(&H, 1, MPI_INT, 0, comm);
    MPI_Bcast(&alpha, 1, MPI_DOUBLE, 0, comm);
    MPI_Bcast(&beta,  1, MPI_INT, 0, comm);

    ScatterLayout sl = make_layout(H, nprocs, W, 3);
    std::vector<uchar> local_src(sl.rows_per_rank[rank] * W * 3);
    cv::Mat root_src_storage;
    const uchar* root_src = nullptr;
    if (rank == 0) root_src = root_contiguous_ptr(img, root_src_storage);

    MPI_Barrier(comm);
    double t0 = wtime();

    MPI_Scatterv(root_src, sl.counts.data(), sl.displs.data(), MPI_UNSIGNED_CHAR,
                 local_src.data(), sl.counts[rank], MPI_UNSIGNED_CHAR, 0, comm);

    int local_rows = sl.rows_per_rank[rank];
    cv::Mat local_bgr = vec_view(local_src, local_rows, W, CV_8UC3);
    cv::Mat local_out(local_rows, W, CV_8UC3);

    for (int r = 0; r < local_rows; ++r) {
        const cv::Vec3b* sp = local_bgr.ptr<cv::Vec3b>(r);
        cv::Vec3b* dp = local_out.ptr<cv::Vec3b>(r);
        for (int c = 0; c < W; ++c)
            dp[c] = cv::Vec3b(
                static_cast<uchar>(std::clamp(static_cast<int>(sp[c][0]*alpha+beta),0,255)),
                static_cast<uchar>(std::clamp(static_cast<int>(sp[c][1]*alpha+beta),0,255)),
                static_cast<uchar>(std::clamp(static_cast<int>(sp[c][2]*alpha+beta),0,255)));
    }

    if (rank == 0) dst.create(H, W, CV_8UC3);
    MPI_Gatherv(local_rows > 0 ? local_out.data : nullptr, sl.counts[rank], MPI_UNSIGNED_CHAR,
                rank == 0 ? dst.data : nullptr, sl.counts.data(), sl.displs.data(),
                MPI_UNSIGNED_CHAR, 0, comm);

    elapsed = wtime() - t0;
}

// ─── 5. Histogram Equalization ───────────────────────────────────────────────
static void mpi_histeq(const cv::Mat& img, cv::Mat& dst,
                        int rank, int nprocs,
                        MPI_Comm comm, double& elapsed) {
    int W = 0, H = 0;
    if (rank == 0) { W = img.cols; H = img.rows; }
    MPI_Bcast(&W, 1, MPI_INT, 0, comm);
    MPI_Bcast(&H, 1, MPI_INT, 0, comm);

    // Convert to gray on rank 0
    cv::Mat root_gray;
    const uchar* root_gray_ptr = nullptr;
    if (rank == 0) {
        local_grayscale(img, root_gray);
        if (!root_gray.isContinuous()) root_gray = root_gray.clone();
        root_gray_ptr = root_gray.data;
    }

    ScatterLayout sl = make_layout(H, nprocs, W, 1);
    std::vector<uchar> local_g(sl.rows_per_rank[rank] * W);

    MPI_Barrier(comm);
    double t0 = wtime();

    MPI_Scatterv(root_gray_ptr, sl.counts.data(), sl.displs.data(), MPI_UNSIGNED_CHAR,
                 local_g.data(), sl.counts[rank], MPI_UNSIGNED_CHAR, 0, comm);

    // Local histogram
    int local_hist[256] = {};
    for (uchar v : local_g) local_hist[v]++;

    // Global histogram via MPI_Reduce
    int global_hist[256] = {};
    MPI_Reduce(local_hist, global_hist, 256, MPI_INT, MPI_SUM, 0, comm);

    // Build LUT on rank 0, broadcast
    uchar lut[256] = {};
    if (rank == 0) {
        int total = H * W;
        int cdf[256] = {}; cdf[0] = global_hist[0];
        for (int i = 1; i < 256; ++i) cdf[i] = cdf[i-1] + global_hist[i];
        int cdf_min = 0;
        for (int i = 0; i < 256; ++i) { if (cdf[i]>0){cdf_min=cdf[i]; break;} }
        for (int i = 0; i < 256; ++i) {
            if (total == cdf_min) { lut[i]=0; continue; }
            lut[i] = static_cast<uchar>(
                std::round((double)(cdf[i]-cdf_min)/(total-cdf_min)*255.0));
        }
    }
    MPI_Bcast(lut, 256, MPI_UNSIGNED_CHAR, 0, comm);

    // Apply LUT locally
    std::vector<uchar> local_out(local_g.size());
    for (size_t i = 0; i < local_g.size(); ++i)
        local_out[i] = lut[local_g[i]];

    if (rank == 0) dst.create(H, W, CV_8UC1);
    MPI_Gatherv(local_out.data(), sl.counts[rank], MPI_UNSIGNED_CHAR,
                rank == 0 ? dst.data : nullptr, sl.counts.data(), sl.displs.data(),
                MPI_UNSIGNED_CHAR, 0, comm);

    elapsed = wtime() - t0;
}

// ─── CSV Writer ───────────────────────────────────────────────────────────────
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
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    std::string input  = "test_images/sample.jpg";
    std::string outdir = "results/images";
    std::string csv    = "results/data/mpi_results.csv";
    if (argc > 1) input = argv[1];

    cv::Mat img;
    if (rank == 0) {
        img = cv::imread(input, cv::IMREAD_COLOR);
        if (img.empty()) {
            std::cerr << "Error: cannot load '" << input << "'\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        std::cout << "MPI Image Processing  |  processes: " << nprocs << '\n';
        std::cout << "Image: " << input << " (" << img.cols << "x" << img.rows << ")\n\n";
    }

    std::vector<BenchmarkResult> res;
    cv::Mat dst;
    double elapsed = 0;

    struct OpEntry { std::string name; };
    auto record = [&](const std::string& op) {
        if (rank == 0) {
            BenchmarkResult r;
            r.version    = "mpi";
            r.operation  = op;
            r.image_width  = img.cols;
            r.image_height = img.rows;
            r.threads    = 1;
            r.processes  = nprocs;
            r.elapsed_sec = elapsed;
            r.speedup    = 0.0;
            std::printf("  %-25s %d procs  %.4f s\n", op.c_str(), nprocs, elapsed);
            cv::imwrite(outdir + "/mpi_" + op + "_p" + std::to_string(nprocs) + ".png", dst);
            res.push_back(r);
        }
    };

    mpi_grayscale(img, dst, rank, nprocs, MPI_COMM_WORLD, elapsed);  record("grayscale");
    mpi_gaussian_blur(img, dst, rank, nprocs, MPI_COMM_WORLD, elapsed); record("gaussian_blur");
    mpi_sobel(img, dst, rank, nprocs, MPI_COMM_WORLD, elapsed);       record("sobel_edge");
    mpi_brightness(img, dst, 1.2, 30, rank, nprocs, MPI_COMM_WORLD, elapsed); record("brightness");
    mpi_histeq(img, dst, rank, nprocs, MPI_COMM_WORLD, elapsed);      record("histogram_eq");

    if (rank == 0) {
        save_csv(res, csv);
        std::cout << "\nResults saved to " << csv << "\n";
    }

    MPI_Finalize();
    return 0;
}
