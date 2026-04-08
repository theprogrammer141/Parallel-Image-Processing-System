/**
 * OpenCL GPU Image Processing
 * ─────────────────────────────
 * Offloads all five image-processing operations to an OpenCL device.
 * GPU is preferred; the implementation falls back to a CPU OpenCL device
 * (e.g. POCL, Intel CPU runtime) when no GPU is available.
 *
 * All five .cl kernel files are loaded from disk at runtime so you can
 * tweak kernels without recompiling this host binary.
 *
 * Operations (identical semantics to the CPU versions):
 *   1. Grayscale conversion          (grayscale.cl)
 *   2. Gaussian blur 5×5             (gaussian_blur.cl)
 *   3. Sobel edge detection          (sobel_edge.cl)   — internally grays first
 *   4. Brightness / contrast  α=1.2  (brightness.cl)
 *   5. Histogram equalisation        (histogram_eq.cl)
 *       GPU: atomic histogram accumulation
 *       CPU: CDF prefix-sum + LUT (O(256) — negligible)
 *       GPU: LUT application
 *
 * Timing: wall-clock elapsed (includes host↔device transfers).
 *
 * Usage:  ./build/opencl_proc [image_path]
 */

#define CL_TARGET_OPENCL_VERSION 120
/* Suppress the clCreateCommandQueue deprecation warning that some OpenCL
   headers (targeting ≥ 2.0) emit even when we're building for 1.2. */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <CL/cl.h>
#pragma GCC diagnostic pop

#include <opencv2/opencv.hpp>
#include "image_processing.h"

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// ─── Timing ───────────────────────────────────────────────────────────────────
static double wtime() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// ─── OpenCL error helper ───────────────────────────────────────────────────────
static void cl_check(cl_int err, const char* msg) {
    if (err != CL_SUCCESS)
        throw std::runtime_error(std::string(msg) +
                                 " (CL error " + std::to_string(err) + ")");
}

// ─── Load a .cl file as a source string ───────────────────────────────────────
static std::string load_source(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open kernel file: " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// ─── Compile a single .cl file into a cl_program ─────────────────────────────
static cl_program build_program(cl_context ctx, cl_device_id dev,
                                const std::string& path) {
    std::string src = load_source(path);
    const char* ptr = src.c_str();
    size_t      len = src.size();
    cl_int err;
    cl_program prog = clCreateProgramWithSource(ctx, 1, &ptr, &len, &err);
    cl_check(err, "clCreateProgramWithSource");

    err = clBuildProgram(prog, 1, &dev, "-cl-std=CL1.2", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_sz = 0;
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG,
                              0, nullptr, &log_sz);
        std::string log(log_sz, '\0');
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG,
                              log_sz, &log[0], nullptr);
        clReleaseProgram(prog);
        throw std::runtime_error("OpenCL build failed [" + path + "]:\n" + log);
    }
    return prog;
}

// ─── OpenCL environment ────────────────────────────────────────────────────────
struct CLEnv {
    cl_device_id     device  = nullptr;
    cl_context       context = nullptr;
    cl_command_queue queue   = nullptr;
    size_t           wg      = 16;   // work-group side: wg × wg per 2-D dispatch
    std::string      device_name;
};

static CLEnv init_cl() {
    CLEnv env;
    cl_int err;

    // Enumerate platforms
    cl_uint nplat = 0;
    err = clGetPlatformIDs(0, nullptr, &nplat);
    if (err != CL_SUCCESS || nplat == 0)
        throw std::runtime_error(
            "No OpenCL platforms found.\n"
            "Install a runtime: 'sudo apt install pocl-opencl-icd' provides "
            "a CPU runtime, GPU drivers provide GPU runtimes.");

    std::vector<cl_platform_id> plats(nplat);
    clGetPlatformIDs(nplat, plats.data(), nullptr);

    // Prefer GPU across all platforms; fall back to CPU
    bool found = false;
    for (auto p : plats) {
        err = clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 1, &env.device, nullptr);
        if (err == CL_SUCCESS) { found = true; break; }
    }
    if (!found) {
        for (auto p : plats) {
            err = clGetDeviceIDs(p, CL_DEVICE_TYPE_CPU, 1, &env.device, nullptr);
            if (err == CL_SUCCESS) { found = true; break; }
        }
    }
    if (!found)
        throw std::runtime_error("No OpenCL GPU or CPU device found.");

    // Device name (trim trailing nulls that some drivers add)
    size_t name_sz = 0;
    clGetDeviceInfo(env.device, CL_DEVICE_NAME, 0, nullptr, &name_sz);
    env.device_name.resize(name_sz);
    clGetDeviceInfo(env.device, CL_DEVICE_NAME,
                    name_sz, &env.device_name[0], nullptr);
    while (!env.device_name.empty() && env.device_name.back() == '\0')
        env.device_name.pop_back();

    // Pick work-group size: 16×16 (256 WI) if device supports it, else 8×8
    size_t max_wgs = 64;
    clGetDeviceInfo(env.device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                    sizeof(size_t), &max_wgs, nullptr);
    env.wg = (max_wgs >= 256) ? 16 : 8;

    env.context = clCreateContext(nullptr, 1, &env.device,
                                  nullptr, nullptr, &err);
    cl_check(err, "clCreateContext");

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    env.queue = clCreateCommandQueue(env.context, env.device, 0, &err);
#pragma GCC diagnostic pop
    cl_check(err, "clCreateCommandQueue");

    return env;
}

// ─── 2-D NDRange dispatch + synchronous finish ────────────────────────────────
static void dispatch2d(cl_command_queue q, cl_kernel k,
                       int img_w, int img_h, size_t wg) {
    size_t lc[2] = {wg, wg};
    size_t gc[2] = {((size_t)img_w + wg - 1) / wg * wg,
                    ((size_t)img_h + wg - 1) / wg * wg};
    cl_int err = clEnqueueNDRangeKernel(q, k, 2, nullptr, gc, lc,
                                        0, nullptr, nullptr);
    cl_check(err, "clEnqueueNDRangeKernel");
    cl_check(clFinish(q), "clFinish");
}

// ─── CSV writer ───────────────────────────────────────────────────────────────
static void save_csv(const std::vector<BenchmarkResult>& res,
                     const std::string& path) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot write CSV: " + path);
    f << "version,operation,width,height,threads,processes,elapsed_sec,speedup\n";
    for (const auto& r : res)
        f << r.version << ',' << r.operation << ','
          << r.image_width << ',' << r.image_height << ','
          << r.threads << ',' << r.processes << ','
          << r.elapsed_sec << ',' << r.speedup << '\n';
}

// ═══════════════════════════════════════════════════════════════════════════════
// Per-operation GPU helpers
// ═══════════════════════════════════════════════════════════════════════════════

// ─── 1. Grayscale (BGR → gray) ────────────────────────────────────────────────
static void grayscale_gpu(const cv::Mat& src, cv::Mat& dst,
                           const CLEnv& cl, cl_program prog) {
    cv::Mat csrc = src.isContinuous() ? src : src.clone();
    dst.create(src.rows, src.cols, CV_8UC1);

    cl_int err;
    size_t sb = (size_t)src.rows * src.cols * 3;
    size_t db = (size_t)src.rows * src.cols;

    cl_mem d_src = clCreateBuffer(cl.context,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  sb, csrc.data, &err);
    cl_check(err, "buf src (gray)");
    cl_mem d_dst = clCreateBuffer(cl.context, CL_MEM_WRITE_ONLY,
                                  db, nullptr, &err);
    cl_check(err, "buf dst (gray)");

    cl_kernel k = clCreateKernel(prog, "grayscale", &err);
    cl_check(err, "kernel grayscale");

    int w = src.cols, h = src.rows;
    clSetKernelArg(k, 0, sizeof(cl_mem), &d_src);
    clSetKernelArg(k, 1, sizeof(cl_mem), &d_dst);
    clSetKernelArg(k, 2, sizeof(int),    &w);
    clSetKernelArg(k, 3, sizeof(int),    &h);

    dispatch2d(cl.queue, k, w, h, cl.wg);
    cl_check(clEnqueueReadBuffer(cl.queue, d_dst, CL_TRUE,
                                 0, db, dst.data, 0, nullptr, nullptr),
             "readbuf (gray)");

    clReleaseKernel(k);
    clReleaseMemObject(d_src);
    clReleaseMemObject(d_dst);
}

// ─── 2. Gaussian Blur (5×5) ───────────────────────────────────────────────────
static void gaussian_blur_gpu(const cv::Mat& src, cv::Mat& dst,
                               const CLEnv& cl, cl_program prog) {
    cv::Mat csrc = src.isContinuous() ? src : src.clone();
    dst.create(src.rows, src.cols, src.type());

    cl_int err;
    int ch = src.channels();
    size_t bytes = (size_t)src.rows * src.cols * ch;

    cl_mem d_src = clCreateBuffer(cl.context,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  bytes, csrc.data, &err);
    cl_check(err, "buf src (gauss)");
    cl_mem d_dst = clCreateBuffer(cl.context, CL_MEM_WRITE_ONLY,
                                  bytes, nullptr, &err);
    cl_check(err, "buf dst (gauss)");

    const char* kname = (ch == 1) ? "gaussian_blur_gray" : "gaussian_blur_color";
    cl_kernel k = clCreateKernel(prog, kname, &err);
    cl_check(err, kname);

    int w = src.cols, h = src.rows;
    clSetKernelArg(k, 0, sizeof(cl_mem), &d_src);
    clSetKernelArg(k, 1, sizeof(cl_mem), &d_dst);
    clSetKernelArg(k, 2, sizeof(int),    &w);
    clSetKernelArg(k, 3, sizeof(int),    &h);

    dispatch2d(cl.queue, k, w, h, cl.wg);
    cl_check(clEnqueueReadBuffer(cl.queue, d_dst, CL_TRUE,
                                 0, bytes, dst.data, 0, nullptr, nullptr),
             "readbuf (gauss)");

    clReleaseKernel(k);
    clReleaseMemObject(d_src);
    clReleaseMemObject(d_dst);
}

// ─── 3. Sobel Edge Detection ──────────────────────────────────────────────────
// colour → gray on GPU, then Sobel on GPU
static void sobel_edge_gpu(const cv::Mat& src, cv::Mat& dst,
                            const CLEnv& cl,
                            cl_program prog_gray, cl_program prog_sobel) {
    // Step A: grayscale on GPU (or reuse if already 1-channel)
    cv::Mat gray;
    if (src.channels() != 1)
        grayscale_gpu(src, gray, cl, prog_gray);
    else
        gray = src.isContinuous() ? src : src.clone();

    dst.create(gray.rows, gray.cols, CV_8UC1);
    cl_int err;
    size_t bytes = (size_t)gray.rows * gray.cols;

    cl_mem d_src = clCreateBuffer(cl.context,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  bytes, gray.data, &err);
    cl_check(err, "buf src (sobel)");
    cl_mem d_dst = clCreateBuffer(cl.context, CL_MEM_WRITE_ONLY,
                                  bytes, nullptr, &err);
    cl_check(err, "buf dst (sobel)");

    cl_kernel k = clCreateKernel(prog_sobel, "sobel_edge", &err);
    cl_check(err, "kernel sobel_edge");

    int w = gray.cols, h = gray.rows;
    clSetKernelArg(k, 0, sizeof(cl_mem), &d_src);
    clSetKernelArg(k, 1, sizeof(cl_mem), &d_dst);
    clSetKernelArg(k, 2, sizeof(int),    &w);
    clSetKernelArg(k, 3, sizeof(int),    &h);

    dispatch2d(cl.queue, k, w, h, cl.wg);
    cl_check(clEnqueueReadBuffer(cl.queue, d_dst, CL_TRUE,
                                 0, bytes, dst.data, 0, nullptr, nullptr),
             "readbuf (sobel)");

    clReleaseKernel(k);
    clReleaseMemObject(d_src);
    clReleaseMemObject(d_dst);
}

// ─── 4. Brightness / Contrast ─────────────────────────────────────────────────
static void brightness_gpu(const cv::Mat& src, cv::Mat& dst,
                             float alpha, int beta,
                             const CLEnv& cl, cl_program prog) {
    cv::Mat csrc = src.isContinuous() ? src : src.clone();
    dst.create(src.rows, src.cols, src.type());

    cl_int err;
    int ch = src.channels();
    size_t bytes = (size_t)src.rows * src.cols * ch;

    cl_mem d_src = clCreateBuffer(cl.context,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  bytes, csrc.data, &err);
    cl_check(err, "buf src (bri)");
    cl_mem d_dst = clCreateBuffer(cl.context, CL_MEM_WRITE_ONLY,
                                  bytes, nullptr, &err);
    cl_check(err, "buf dst (bri)");

    const char* kname = (ch == 1) ? "brightness_gray" : "brightness_color";
    cl_kernel k = clCreateKernel(prog, kname, &err);
    cl_check(err, kname);

    int w = src.cols, h = src.rows;
    clSetKernelArg(k, 0, sizeof(cl_mem), &d_src);
    clSetKernelArg(k, 1, sizeof(cl_mem), &d_dst);
    clSetKernelArg(k, 2, sizeof(int),    &w);
    clSetKernelArg(k, 3, sizeof(int),    &h);
    clSetKernelArg(k, 4, sizeof(float),  &alpha);
    clSetKernelArg(k, 5, sizeof(int),    &beta);

    dispatch2d(cl.queue, k, w, h, cl.wg);
    cl_check(clEnqueueReadBuffer(cl.queue, d_dst, CL_TRUE,
                                 0, bytes, dst.data, 0, nullptr, nullptr),
             "readbuf (bri)");

    clReleaseKernel(k);
    clReleaseMemObject(d_src);
    clReleaseMemObject(d_dst);
}

// ─── 5. Histogram Equalisation ────────────────────────────────────────────────
// GPU: atomic histogram build | CPU: CDF+LUT (O(256)) | GPU: LUT apply
static void histogram_eq_gpu(const cv::Mat& src, cv::Mat& dst,
                              const CLEnv& cl,
                              cl_program prog_gray, cl_program prog_hist) {
    // Step A: grayscale on GPU
    cv::Mat gray;
    if (src.channels() != 1)
        grayscale_gpu(src, gray, cl, prog_gray);
    else
        gray = src.isContinuous() ? src : src.clone();

    dst.create(gray.rows, gray.cols, CV_8UC1);
    cl_int err;
    size_t bytes = (size_t)gray.rows * gray.cols;

    cl_mem d_src = clCreateBuffer(cl.context,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  bytes, gray.data, &err);
    cl_check(err, "buf src (hist)");

    // Step B: build histogram on GPU using atomic_add
    int zero_hist[256] = {};
    cl_mem d_hist = clCreateBuffer(cl.context,
                                   CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                   256 * sizeof(int), zero_hist, &err);
    cl_check(err, "buf hist");

    cl_kernel k_build = clCreateKernel(prog_hist, "build_histogram", &err);
    cl_check(err, "kernel build_histogram");

    int w = gray.cols, h = gray.rows;
    cl_check(clSetKernelArg(k_build, 0, sizeof(cl_mem), &d_src), "setarg build_histogram src");
    cl_check(clSetKernelArg(k_build, 1, sizeof(cl_mem), &d_hist), "setarg build_histogram hist");
    cl_check(clSetKernelArg(k_build, 2, sizeof(int),    &w), "setarg build_histogram width");
    cl_check(clSetKernelArg(k_build, 3, sizeof(int),    &h), "setarg build_histogram height");
    cl_check(clSetKernelArg(k_build, 4, 256 * sizeof(cl_int), nullptr),
             "setarg build_histogram local_hist");

    dispatch2d(cl.queue, k_build, w, h, cl.wg);

    // Step C: read histogram → compute CDF and LUT on CPU (O(256))
    int hist[256] = {};
    cl_check(clEnqueueReadBuffer(cl.queue, d_hist, CL_TRUE,
                                 0, 256 * sizeof(int), hist, 0, nullptr, nullptr),
             "readbuf hist");

    int total = gray.rows * gray.cols;
    int cdf[256] = {};
    cdf[0] = hist[0];
    for (int i = 1; i < 256; ++i) cdf[i] = cdf[i-1] + hist[i];
    int cdf_min = 0;
    for (int i = 0; i < 256; ++i) { if (cdf[i] > 0) { cdf_min = cdf[i]; break; } }

    uchar lut[256] = {};
    for (int i = 0; i < 256; ++i) {
        if (total == cdf_min) { lut[i] = 0; continue; }
        lut[i] = static_cast<uchar>(
            std::round((double)(cdf[i] - cdf_min) / (total - cdf_min) * 255.0));
    }

    // Step D: apply LUT on GPU
    cl_mem d_lut = clCreateBuffer(cl.context,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  256, lut, &err);
    cl_check(err, "buf lut");
    cl_mem d_dst = clCreateBuffer(cl.context, CL_MEM_WRITE_ONLY,
                                  bytes, nullptr, &err);
    cl_check(err, "buf dst (hist)");

    cl_kernel k_lut = clCreateKernel(prog_hist, "apply_lut", &err);
    cl_check(err, "kernel apply_lut");
    clSetKernelArg(k_lut, 0, sizeof(cl_mem), &d_src);
    clSetKernelArg(k_lut, 1, sizeof(cl_mem), &d_dst);
    clSetKernelArg(k_lut, 2, sizeof(cl_mem), &d_lut);
    clSetKernelArg(k_lut, 3, sizeof(int),    &w);
    clSetKernelArg(k_lut, 4, sizeof(int),    &h);

    dispatch2d(cl.queue, k_lut, w, h, cl.wg);
    cl_check(clEnqueueReadBuffer(cl.queue, d_dst, CL_TRUE,
                                 0, bytes, dst.data, 0, nullptr, nullptr),
             "readbuf (hist out)");

    clReleaseKernel(k_build); clReleaseKernel(k_lut);
    clReleaseMemObject(d_src); clReleaseMemObject(d_hist);
    clReleaseMemObject(d_lut); clReleaseMemObject(d_dst);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════════
int main(int argc, char* argv[]) {
    const std::string input  = (argc > 1) ? argv[1] : "test_images/sample.jpg";
    const std::string outdir = "results/images";
    const std::string csv    = "results/data/opencl_results.csv";

    cv::Mat img = cv::imread(input, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Error: cannot load '" << input << "'\n";
        std::cerr << "Usage: " << argv[0] << " [image_path]\n";
        return 1;
    }
    if (!img.isContinuous()) img = img.clone();

    // ── OpenCL initialisation ─────────────────────────────────────────────────
    CLEnv cl;
    try {
        cl = init_cl();
    } catch (const std::exception& e) {
        std::cerr << "OpenCL init failed: " << e.what() << "\n";
        return 1;
    }

    std::cout << "OpenCL GPU Image Processing\n";
    std::cout << "Device : " << cl.device_name
              << "  (wg=" << cl.wg << "\xc3\x97" << cl.wg << ")\n";
    std::cout << "Image  : " << input
              << " (" << img.cols << "\xc3\x97" << img.rows << ")\n\n";

    // ── Compile kernel programs (loaded from .cl files at runtime) ────────────
    const std::string kdir = "src/opencl/kernels/";
    cl_program prog_gray, prog_gauss, prog_sobel, prog_bri, prog_hist;
    try {
        prog_gray  = build_program(cl.context, cl.device, kdir + "grayscale.cl");
        prog_gauss = build_program(cl.context, cl.device, kdir + "gaussian_blur.cl");
        prog_sobel = build_program(cl.context, cl.device, kdir + "sobel_edge.cl");
        prog_bri   = build_program(cl.context, cl.device, kdir + "brightness.cl");
        prog_hist  = build_program(cl.context, cl.device, kdir + "histogram_eq.cl");
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }

    std::vector<BenchmarkResult> results;
    cv::Mat dst;

    // bench: time fn(), save output image, record result
    auto bench = [&](const std::string& op, auto fn) {
        double t0 = wtime();
        fn();
        double el = wtime() - t0;

        BenchmarkResult r;
        r.version      = "opencl";
        r.operation    = op;
        r.image_width  = img.cols;
        r.image_height = img.rows;
        r.threads      = 1;
        r.processes    = 1;
        r.elapsed_sec  = el;
        r.speedup      = 0.0; // computed by benchmark / plot scripts vs sequential

        std::printf("  %-25s %.4f s\n", op.c_str(), el);
        cv::imwrite(outdir + "/ocl_" + op + ".png", dst);
        results.push_back(r);
    };

    try {
        bench("grayscale",
              [&]{ grayscale_gpu(img, dst, cl, prog_gray); });
        bench("gaussian_blur",
              [&]{ gaussian_blur_gpu(img, dst, cl, prog_gauss); });
        bench("sobel_edge",
              [&]{ sobel_edge_gpu(img, dst, cl, prog_gray, prog_sobel); });
        bench("brightness",
              [&]{ brightness_gpu(img, dst, 1.2f, 30, cl, prog_bri); });
        bench("histogram_eq",
              [&]{ histogram_eq_gpu(img, dst, cl, prog_gray, prog_hist); });
    } catch (const std::exception& e) {
        std::cerr << "Processing error: " << e.what() << "\n";
        return 1;
    }

    save_csv(results, csv);
    std::cout << "\nResults saved to " << csv << "\n";

    // ── Cleanup ───────────────────────────────────────────────────────────────
    clReleaseProgram(prog_gray);
    clReleaseProgram(prog_gauss);
    clReleaseProgram(prog_sobel);
    clReleaseProgram(prog_bri);
    clReleaseProgram(prog_hist);
    clReleaseCommandQueue(cl.queue);
    clReleaseContext(cl.context);

    return 0;
}
