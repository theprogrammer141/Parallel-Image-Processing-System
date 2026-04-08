/**
 * Hybrid MPI + OpenMP Image Processing
 * -------------------------------------
 * Combines two levels of parallelism:
 *   - MPI distributes row strips across processes.
 *   - OpenMP parallelizes loops inside each process.
 *
 * Usage:
 *   mpirun -np <P> ./hybrid_proc [image] [omp_threads]
 *   mpirun -np <P> ./hybrid_proc -i <image> -t <omp_threads>
 */

#include "image_processing.h"
#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

static double wtime() { return MPI_Wtime(); }

static int parse_positive_int(const std::string& s, const char* what) {
    size_t pos = 0;
    long long v = 0;
    try {
        v = std::stoll(s, &pos);
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string("Invalid ") + what + ": '" + s + "'");
    }
    if (pos != s.size() || v <= 0 || v > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(std::string("Invalid ") + what + ": '" + s + "'");
    }
    return static_cast<int>(v);
}

struct SL {
    std::vector<int> counts;
    std::vector<int> displs;
    std::vector<int> rows;
    std::vector<int> offsets;
};

static SL make_sl(int H, int nprocs, int W, int esz) {
    SL sl;
    sl.rows.resize(nprocs);
    sl.offsets.resize(nprocs);
    sl.counts.resize(nprocs);
    sl.displs.resize(nprocs);

    int base = H / nprocs;
    int rem = H % nprocs;
    int off = 0;
    for (int i = 0; i < nprocs; ++i) {
        sl.rows[i] = base + (i < rem ? 1 : 0);
        sl.offsets[i] = off;
        sl.counts[i] = sl.rows[i] * W * esz;
        sl.displs[i] = off * W * esz;
        off += sl.rows[i];
    }
    return sl;
}

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

static void local_gray(const cv::Mat& src, cv::Mat& g, int nt) {
    if (src.channels() == 1) {
        g = src;
        return;
    }
    g.create(src.rows, src.cols, CV_8UC1);
    #pragma omp parallel for schedule(static) num_threads(nt)
    for (int r = 0; r < src.rows; ++r) {
        const cv::Vec3b* sp = src.ptr<cv::Vec3b>(r);
        uchar* dp = g.ptr<uchar>(r);
        for (int c = 0; c < src.cols; ++c) {
            dp[c] = static_cast<uchar>(0.114 * sp[c][0] + 0.587 * sp[c][1] + 0.299 * sp[c][2]);
        }
    }
}

static void hybrid_grayscale(const cv::Mat& img, cv::Mat& dst,
                             int rank, int nprocs, int nt,
                             MPI_Comm comm, double& elapsed) {
    int W = 0, H = 0;
    if (rank == 0) {
        W = img.cols;
        H = img.rows;
    }
    MPI_Bcast(&W, 1, MPI_INT, 0, comm);
    MPI_Bcast(&H, 1, MPI_INT, 0, comm);

    SL sl = make_sl(H, nprocs, W, 3);
    std::vector<uchar> local_src(sl.rows[rank] * W * 3);

    cv::Mat root_src_storage;
    const uchar* root_src = nullptr;
    if (rank == 0) root_src = root_contiguous_ptr(img, root_src_storage);

    MPI_Barrier(comm);
    double t0 = wtime();

    MPI_Scatterv(root_src, sl.counts.data(), sl.displs.data(), MPI_UNSIGNED_CHAR,
                 local_src.data(), sl.counts[rank], MPI_UNSIGNED_CHAR, 0, comm);

    int lr = sl.rows[rank];
    cv::Mat local_bgr = vec_view(local_src, lr, W, CV_8UC3);
    cv::Mat local_gray_mat(lr, W, CV_8UC1);

    #pragma omp parallel for schedule(static) num_threads(nt)
    for (int r = 0; r < lr; ++r) {
        const cv::Vec3b* sp = local_bgr.ptr<cv::Vec3b>(r);
        uchar* dp = local_gray_mat.ptr<uchar>(r);
        for (int c = 0; c < W; ++c) {
            dp[c] = static_cast<uchar>(0.114 * sp[c][0] + 0.587 * sp[c][1] + 0.299 * sp[c][2]);
        }
    }

    SL slg = make_sl(H, nprocs, W, 1);
    if (rank == 0) dst.create(H, W, CV_8UC1);

    MPI_Gatherv(lr > 0 ? local_gray_mat.data : nullptr, slg.counts[rank], MPI_UNSIGNED_CHAR,
                rank == 0 ? dst.data : nullptr, slg.counts.data(), slg.displs.data(),
                MPI_UNSIGNED_CHAR, 0, comm);

    elapsed = wtime() - t0;
}

static const float G5[5][5] = {
    {1, 4, 7, 4, 1},
    {4, 16, 26, 16, 4},
    {7, 26, 41, 26, 7},
    {4, 16, 26, 16, 4},
    {1, 4, 7, 4, 1}
};
static constexpr float G5S = 273.f;

static void hybrid_gaussian(const cv::Mat& img, cv::Mat& dst,
                            int rank, int nprocs, int nt,
                            MPI_Comm comm, double& elapsed) {
    int W = 0, H = 0;
    if (rank == 0) {
        W = img.cols;
        H = img.rows;
    }
    MPI_Bcast(&W, 1, MPI_INT, 0, comm);
    MPI_Bcast(&H, 1, MPI_INT, 0, comm);

    SL sl = make_sl(H, nprocs, W, 3);
    std::vector<uchar> local_src(sl.rows[rank] * W * 3);

    cv::Mat root_src_storage;
    const uchar* root_src = nullptr;
    if (rank == 0) root_src = root_contiguous_ptr(img, root_src_storage);

    MPI_Barrier(comm);
    double t0 = wtime();

    MPI_Scatterv(root_src, sl.counts.data(), sl.displs.data(), MPI_UNSIGNED_CHAR,
                 local_src.data(), sl.counts[rank], MPI_UNSIGNED_CHAR, 0, comm);

    int lr = sl.rows[rank];
    int halo = 2;
    int halo_count = halo * W * 3;

    int prev = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int next = (rank + 1 < nprocs) ? rank + 1 : MPI_PROC_NULL;

    std::vector<uchar> send_top;
    std::vector<uchar> send_bottom;
    pack_color_halo_rows(local_src, lr, W, halo, true, send_top);
    pack_color_halo_rows(local_src, lr, W, halo, false, send_bottom);

    std::vector<uchar> halo_above(halo_count, 0);
    std::vector<uchar> halo_below(halo_count, 0);

    MPI_Sendrecv(send_top.data(), halo_count, MPI_UNSIGNED_CHAR, prev, 10,
                 halo_below.data(), halo_count, MPI_UNSIGNED_CHAR, next, 10,
                 comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(send_bottom.data(), halo_count, MPI_UNSIGNED_CHAR, next, 11,
                 halo_above.data(), halo_count, MPI_UNSIGNED_CHAR, prev, 11,
                 comm, MPI_STATUS_IGNORE);

    int ext_rows = halo + lr + halo;
    std::vector<uchar> ext(ext_rows * W * 3, 0);
    std::copy(halo_above.begin(), halo_above.end(), ext.begin());
    if (lr > 0) {
        std::copy(local_src.begin(), local_src.end(), ext.begin() + halo * W * 3);
    }
    std::copy(halo_below.begin(), halo_below.end(), ext.begin() + (halo + lr) * W * 3);

    if (prev == MPI_PROC_NULL && lr > 0) {
        for (int h = 0; h < halo; ++h) {
            std::copy(ext.begin() + halo * W * 3,
                      ext.begin() + (halo + 1) * W * 3,
                      ext.begin() + h * W * 3);
        }
    }
    if (next == MPI_PROC_NULL && lr > 0) {
        for (int h = 0; h < halo; ++h) {
            std::copy(ext.begin() + (halo + lr - 1) * W * 3,
                      ext.begin() + (halo + lr) * W * 3,
                      ext.begin() + (halo + lr + h) * W * 3);
        }
    }

    cv::Mat ext_mat = vec_view(ext, ext_rows, W, CV_8UC3);
    cv::Mat local_out(lr, W, CV_8UC3);

    #pragma omp parallel for schedule(static) num_threads(nt)
    for (int r = 0; r < lr; ++r) {
        for (int c = 0; c < W; ++c) {
            cv::Vec3f acc(0, 0, 0);
            for (int kr = -halo; kr <= halo; ++kr) {
                int rr = r + halo + kr;
                for (int kc = -halo; kc <= halo; ++kc) {
                    int cc = std::clamp(c + kc, 0, W - 1);
                    float w = G5[kr + halo][kc + halo];
                    cv::Vec3b px = ext_mat.at<cv::Vec3b>(rr, cc);
                    acc[0] += px[0] * w;
                    acc[1] += px[1] * w;
                    acc[2] += px[2] * w;
                }
            }
            local_out.at<cv::Vec3b>(r, c) = cv::Vec3b(
                static_cast<uchar>(acc[0] / G5S),
                static_cast<uchar>(acc[1] / G5S),
                static_cast<uchar>(acc[2] / G5S));
        }
    }

    if (rank == 0) dst.create(H, W, CV_8UC3);
    MPI_Gatherv(lr > 0 ? local_out.data : nullptr, sl.counts[rank], MPI_UNSIGNED_CHAR,
                rank == 0 ? dst.data : nullptr, sl.counts.data(), sl.displs.data(),
                MPI_UNSIGNED_CHAR, 0, comm);

    elapsed = wtime() - t0;
}

static void hybrid_sobel(const cv::Mat& img, cv::Mat& dst,
                         int rank, int nprocs, int nt,
                         MPI_Comm comm, double& elapsed) {
    int W = 0, H = 0;
    if (rank == 0) {
        W = img.cols;
        H = img.rows;
    }
    MPI_Bcast(&W, 1, MPI_INT, 0, comm);
    MPI_Bcast(&H, 1, MPI_INT, 0, comm);

    cv::Mat root_gray;
    const uchar* root_gray_ptr = nullptr;
    if (rank == 0) {
        local_gray(img, root_gray, nt);
        if (!root_gray.isContinuous()) root_gray = root_gray.clone();
        root_gray_ptr = root_gray.data;
    }

    SL sl = make_sl(H, nprocs, W, 1);
    std::vector<uchar> local_gray_vec(sl.rows[rank] * W);

    MPI_Barrier(comm);
    double t0 = wtime();

    MPI_Scatterv(root_gray_ptr, sl.counts.data(), sl.displs.data(), MPI_UNSIGNED_CHAR,
                 local_gray_vec.data(), sl.counts[rank], MPI_UNSIGNED_CHAR, 0, comm);

    int lr = sl.rows[rank];
    int halo = 1;
    int halo_count = halo * W;

    int prev = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int next = (rank + 1 < nprocs) ? rank + 1 : MPI_PROC_NULL;

    std::vector<uchar> send_top;
    std::vector<uchar> send_bottom;
    pack_gray_halo_rows(local_gray_vec, lr, W, halo, true, send_top);
    pack_gray_halo_rows(local_gray_vec, lr, W, halo, false, send_bottom);

    std::vector<uchar> halo_above(halo_count, 0);
    std::vector<uchar> halo_below(halo_count, 0);

    MPI_Sendrecv(send_top.data(), halo_count, MPI_UNSIGNED_CHAR, prev, 20,
                 halo_below.data(), halo_count, MPI_UNSIGNED_CHAR, next, 20,
                 comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(send_bottom.data(), halo_count, MPI_UNSIGNED_CHAR, next, 21,
                 halo_above.data(), halo_count, MPI_UNSIGNED_CHAR, prev, 21,
                 comm, MPI_STATUS_IGNORE);

    int ext_rows = halo + lr + halo;
    std::vector<uchar> ext(ext_rows * W, 0);
    std::copy(halo_above.begin(), halo_above.end(), ext.begin());
    if (lr > 0) {
        std::copy(local_gray_vec.begin(), local_gray_vec.end(), ext.begin() + halo * W);
    }
    std::copy(halo_below.begin(), halo_below.end(), ext.begin() + (halo + lr) * W);

    if (prev == MPI_PROC_NULL && lr > 0) {
        std::copy(ext.begin() + halo * W, ext.begin() + (halo + 1) * W, ext.begin());
    }
    if (next == MPI_PROC_NULL && lr > 0) {
        std::copy(ext.begin() + (halo + lr - 1) * W,
                  ext.begin() + (halo + lr) * W,
                  ext.begin() + (halo + lr) * W);
    }

    cv::Mat ext_mat = vec_view(ext, ext_rows, W, CV_8UC1);
    cv::Mat local_out(lr, W, CV_8UC1);
    local_out.setTo(0);

    #pragma omp parallel for schedule(static) num_threads(nt)
    for (int r = 0; r < lr; ++r) {
        int rr = r + halo;
        for (int c = 1; c < W - 1; ++c) {
            int gx =
                -ext_mat.at<uchar>(rr - 1, c - 1) + ext_mat.at<uchar>(rr - 1, c + 1)
                - 2 * ext_mat.at<uchar>(rr, c - 1) + 2 * ext_mat.at<uchar>(rr, c + 1)
                - ext_mat.at<uchar>(rr + 1, c - 1) + ext_mat.at<uchar>(rr + 1, c + 1);

            int gy =
                -ext_mat.at<uchar>(rr - 1, c - 1) - 2 * ext_mat.at<uchar>(rr - 1, c) - ext_mat.at<uchar>(rr - 1, c + 1)
                + ext_mat.at<uchar>(rr + 1, c - 1) + 2 * ext_mat.at<uchar>(rr + 1, c) + ext_mat.at<uchar>(rr + 1, c + 1);

            local_out.at<uchar>(r, c) =
                static_cast<uchar>(std::min(static_cast<int>(std::sqrt(gx * gx + gy * gy)), 255));
        }
    }

    if (lr > 0 && rank == 0) local_out.row(0).setTo(0);
    if (lr > 0 && rank == nprocs - 1) local_out.row(lr - 1).setTo(0);

    if (rank == 0) dst.create(H, W, CV_8UC1);
    MPI_Gatherv(lr > 0 ? local_out.data : nullptr, sl.counts[rank], MPI_UNSIGNED_CHAR,
                rank == 0 ? dst.data : nullptr, sl.counts.data(), sl.displs.data(),
                MPI_UNSIGNED_CHAR, 0, comm);

    elapsed = wtime() - t0;
}

static void hybrid_brightness(const cv::Mat& img, cv::Mat& dst,
                              double alpha, int beta,
                              int rank, int nprocs, int nt,
                              MPI_Comm comm, double& elapsed) {
    int W = 0, H = 0;
    if (rank == 0) {
        W = img.cols;
        H = img.rows;
    }
    MPI_Bcast(&W, 1, MPI_INT, 0, comm);
    MPI_Bcast(&H, 1, MPI_INT, 0, comm);
    MPI_Bcast(&alpha, 1, MPI_DOUBLE, 0, comm);
    MPI_Bcast(&beta, 1, MPI_INT, 0, comm);

    SL sl = make_sl(H, nprocs, W, 3);
    std::vector<uchar> local_src(sl.rows[rank] * W * 3);

    cv::Mat root_src_storage;
    const uchar* root_src = nullptr;
    if (rank == 0) root_src = root_contiguous_ptr(img, root_src_storage);

    MPI_Barrier(comm);
    double t0 = wtime();

    MPI_Scatterv(root_src, sl.counts.data(), sl.displs.data(), MPI_UNSIGNED_CHAR,
                 local_src.data(), sl.counts[rank], MPI_UNSIGNED_CHAR, 0, comm);

    int lr = sl.rows[rank];
    cv::Mat local_bgr = vec_view(local_src, lr, W, CV_8UC3);
    cv::Mat local_out(lr, W, CV_8UC3);

    #pragma omp parallel for schedule(static) num_threads(nt)
    for (int r = 0; r < lr; ++r) {
        const cv::Vec3b* sp = local_bgr.ptr<cv::Vec3b>(r);
        cv::Vec3b* dp = local_out.ptr<cv::Vec3b>(r);
        for (int c = 0; c < W; ++c) {
            dp[c] = cv::Vec3b(
                static_cast<uchar>(std::clamp(static_cast<int>(sp[c][0] * alpha + beta), 0, 255)),
                static_cast<uchar>(std::clamp(static_cast<int>(sp[c][1] * alpha + beta), 0, 255)),
                static_cast<uchar>(std::clamp(static_cast<int>(sp[c][2] * alpha + beta), 0, 255)));
        }
    }

    if (rank == 0) dst.create(H, W, CV_8UC3);
    MPI_Gatherv(lr > 0 ? local_out.data : nullptr, sl.counts[rank], MPI_UNSIGNED_CHAR,
                rank == 0 ? dst.data : nullptr, sl.counts.data(), sl.displs.data(),
                MPI_UNSIGNED_CHAR, 0, comm);

    elapsed = wtime() - t0;
}

static void hybrid_histeq(const cv::Mat& img, cv::Mat& dst,
                          int rank, int nprocs, int nt,
                          MPI_Comm comm, double& elapsed) {
    int W = 0, H = 0;
    if (rank == 0) {
        W = img.cols;
        H = img.rows;
    }
    MPI_Bcast(&W, 1, MPI_INT, 0, comm);
    MPI_Bcast(&H, 1, MPI_INT, 0, comm);

    cv::Mat root_gray;
    const uchar* root_gray_ptr = nullptr;
    if (rank == 0) {
        local_gray(img, root_gray, nt);
        if (!root_gray.isContinuous()) root_gray = root_gray.clone();
        root_gray_ptr = root_gray.data;
    }

    SL sl = make_sl(H, nprocs, W, 1);
    std::vector<uchar> local_gray_vec(sl.rows[rank] * W);

    MPI_Barrier(comm);
    double t0 = wtime();

    MPI_Scatterv(root_gray_ptr, sl.counts.data(), sl.displs.data(), MPI_UNSIGNED_CHAR,
                 local_gray_vec.data(), sl.counts[rank], MPI_UNSIGNED_CHAR, 0, comm);

    int local_hist[256] = {};
    #pragma omp parallel for schedule(static) num_threads(nt) reduction(+:local_hist[:256])
    for (int i = 0; i < static_cast<int>(local_gray_vec.size()); ++i) {
        local_hist[local_gray_vec[i]]++;
    }

    int global_hist[256] = {};
    MPI_Reduce(local_hist, global_hist, 256, MPI_INT, MPI_SUM, 0, comm);

    uchar lut[256] = {};
    if (rank == 0) {
        int total = H * W;
        int cdf[256] = {};
        cdf[0] = global_hist[0];
        for (int i = 1; i < 256; ++i) cdf[i] = cdf[i - 1] + global_hist[i];

        int cdf_min = 0;
        for (int i = 0; i < 256; ++i) {
            if (cdf[i] > 0) {
                cdf_min = cdf[i];
                break;
            }
        }

        for (int i = 0; i < 256; ++i) {
            if (total == cdf_min) {
                lut[i] = 0;
                continue;
            }
            lut[i] = static_cast<uchar>(
                std::round((double)(cdf[i] - cdf_min) / (total - cdf_min) * 255.0));
        }
    }
    MPI_Bcast(lut, 256, MPI_UNSIGNED_CHAR, 0, comm);

    std::vector<uchar> local_out(local_gray_vec.size());
    #pragma omp parallel for schedule(static) num_threads(nt)
    for (int i = 0; i < static_cast<int>(local_gray_vec.size()); ++i) {
        local_out[i] = lut[local_gray_vec[i]];
    }

    if (rank == 0) dst.create(H, W, CV_8UC1);
    MPI_Gatherv(local_out.empty() ? nullptr : local_out.data(), sl.counts[rank], MPI_UNSIGNED_CHAR,
                rank == 0 ? dst.data : nullptr, sl.counts.data(), sl.displs.data(),
                MPI_UNSIGNED_CHAR, 0, comm);

    elapsed = wtime() - t0;
}

static void save_csv(const std::vector<BenchmarkResult>& r, const std::string& p) {
    std::ofstream f(p);
    if (!f) throw std::runtime_error("Cannot open " + p);
    f << "version,operation,width,height,threads,processes,elapsed_sec,speedup\n";
    for (const auto& x : r) {
        f << x.version << ',' << x.operation << ',' << x.image_width << ',' << x.image_height << ','
          << x.threads << ',' << x.processes << ',' << x.elapsed_sec << ',' << x.speedup << '\n';
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    std::string input = "test_images/sample.jpg";
    std::string outdir = "results/images";
    std::string csv = "results/data/hybrid_results.csv";
    int nt = omp_get_max_threads();

    bool show_help = false;
    bool parse_ok = true;
    std::string parse_err;
    std::vector<std::string> positional;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            show_help = true;
            break;
        }
        if (arg == "-i" || arg == "--image") {
            if (i + 1 >= argc) {
                parse_ok = false;
                parse_err = "Missing value for " + arg;
                break;
            }
            input = argv[++i];
            continue;
        }
        if (arg == "-t" || arg == "--threads") {
            if (i + 1 >= argc) {
                parse_ok = false;
                parse_err = "Missing value for " + arg;
                break;
            }
            try {
                nt = parse_positive_int(argv[++i], "thread count");
            } catch (const std::exception& e) {
                parse_ok = false;
                parse_err = e.what();
            }
            continue;
        }
        if (!arg.empty() && arg[0] == '-') {
            parse_ok = false;
            parse_err = "Unknown option: " + arg;
            break;
        }
        positional.push_back(arg);
    }

    if (parse_ok && !show_help) {
        try {
            if (!positional.empty()) input = positional[0];
            if (positional.size() >= 2) nt = parse_positive_int(positional[1], "thread count");
            if (positional.size() > 2) {
                parse_ok = false;
                parse_err = "Too many positional arguments";
            }
        } catch (const std::exception& e) {
            parse_ok = false;
            parse_err = e.what();
        }
    }

    if (show_help || !parse_ok) {
        if (rank == 0) {
            if (!parse_ok) std::cerr << "Error: " << parse_err << "\n";
            std::cout << "Usage: " << argv[0] << " [image] [threads]\n"
                      << "       " << argv[0] << " -i <image> -t <threads>\n";
        }
        MPI_Finalize();
        return parse_ok ? 0 : 1;
    }

    cv::Mat img;
    if (rank == 0) {
        img = cv::imread(input, cv::IMREAD_COLOR);
        if (img.empty()) {
            std::cerr << "Error: cannot load '" << input << "'\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        std::cout << "Hybrid MPI+OpenMP  |  processes: " << nprocs
                  << "  threads/proc: " << nt << '\n';
        std::cout << "Total workers: " << nprocs * nt << '\n';
        std::cout << "Image: " << input << " (" << img.cols << "x" << img.rows << ")\n\n";
    }

    std::vector<BenchmarkResult> res;
    cv::Mat dst;
    double elapsed = 0.0;

    auto record = [&](const std::string& op) {
        if (rank == 0) {
            BenchmarkResult r;
            r.version = "hybrid";
            r.operation = op;
            r.image_width = img.cols;
            r.image_height = img.rows;
            r.threads = nt;
            r.processes = nprocs;
            r.elapsed_sec = elapsed;
            r.speedup = 0.0;
            std::printf("  %-25s %d procs x %d threads  %.4f s\n", op.c_str(), nprocs, nt, elapsed);
            cv::imwrite(outdir + "/hybrid_" + op + "_p" + std::to_string(nprocs) + "_t" + std::to_string(nt) + ".png", dst);
            res.push_back(r);
        }
    };

    hybrid_grayscale(img, dst, rank, nprocs, nt, MPI_COMM_WORLD, elapsed); record("grayscale");
    hybrid_gaussian(img, dst, rank, nprocs, nt, MPI_COMM_WORLD, elapsed);  record("gaussian_blur");
    hybrid_sobel(img, dst, rank, nprocs, nt, MPI_COMM_WORLD, elapsed);     record("sobel_edge");
    hybrid_brightness(img, dst, 1.2, 30, rank, nprocs, nt, MPI_COMM_WORLD, elapsed); record("brightness");
    hybrid_histeq(img, dst, rank, nprocs, nt, MPI_COMM_WORLD, elapsed);    record("histogram_eq");

    if (rank == 0) {
        save_csv(res, csv);
        std::cout << "\nResults saved to " << csv << "\n";
    }

    MPI_Finalize();
    return 0;
}
