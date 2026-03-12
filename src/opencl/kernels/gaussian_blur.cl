/*
 * gaussian_blur.cl
 * ─────────────────
 * 5×5 Gaussian blur — same kernel weights as the CPU implementations.
 *
 *   kernel:  1  4  7  4  1        normaliser: 273
 *            4 16 26 16  4
 *            7 26 41 26  7
 *            4 16 26 16  4
 *            1  4  7  4  1
 *
 * Boundary pixels are handled with clamp-to-edge.
 * Two kernels are provided — one for 1-channel (grayscale) input, one for
 * 3-channel (BGR) input — selected at host side.
 */

/* Gaussian weights stored in row-major order */
__constant float GAUSS[25] = {
     1.f,  4.f,  7.f,  4.f,  1.f,
     4.f, 16.f, 26.f, 16.f,  4.f,
     7.f, 26.f, 41.f, 26.f,  7.f,
     4.f, 16.f, 26.f, 16.f,  4.f,
     1.f,  4.f,  7.f,  4.f,  1.f
};
#define GAUSS_SUM 273.0f

/* ── Grayscale (1-channel) version ──────────────────────────────────────── */
__kernel void gaussian_blur_gray(
    __global const uchar* src,
    __global       uchar* dst,
    int width,
    int height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    float acc = 0.0f;
    for (int ky = -2; ky <= 2; ++ky) {
        int ry = clamp(y + ky, 0, height - 1);
        for (int kx = -2; kx <= 2; ++kx) {
            int rx = clamp(x + kx, 0, width - 1);
            float w = GAUSS[(ky + 2) * 5 + (kx + 2)];
            acc += (float)src[ry * width + rx] * w;
        }
    }
    dst[y * width + x] = (uchar)(acc / GAUSS_SUM);
}

/* ── BGR (3-channel) version ─────────────────────────────────────────────── */
__kernel void gaussian_blur_color(
    __global const uchar* src,   /* BGR: (y*width+x)*3 + channel */
    __global       uchar* dst,
    int width,
    int height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f;
    for (int ky = -2; ky <= 2; ++ky) {
        int ry = clamp(y + ky, 0, height - 1);
        for (int kx = -2; kx <= 2; ++kx) {
            int rx  = clamp(x + kx, 0, width - 1);
            float w = GAUSS[(ky + 2) * 5 + (kx + 2)];
            int sidx = (ry * width + rx) * 3;
            acc0 += (float)src[sidx + 0] * w;
            acc1 += (float)src[sidx + 1] * w;
            acc2 += (float)src[sidx + 2] * w;
        }
    }
    int didx = (y * width + x) * 3;
    dst[didx + 0] = (uchar)(acc0 / GAUSS_SUM);
    dst[didx + 1] = (uchar)(acc1 / GAUSS_SUM);
    dst[didx + 2] = (uchar)(acc2 / GAUSS_SUM);
}
