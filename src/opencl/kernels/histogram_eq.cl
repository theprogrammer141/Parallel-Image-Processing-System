/*
 * histogram_eq.cl
 * ───────────────
 * Two-pass histogram equalisation for grayscale images.
 *
 *  Pass 1 — build_histogram
 *      Each work-item reads one pixel and atomically increments the
 *      corresponding bin in a 256-element histogram stored in global memory.
 *      atomic_add is available in OpenCL 1.1+ for __global int pointers.
 *
 *  Pass 2 — apply_lut
 *      The host computes the CDF and builds a 256-byte LUT (O(256) — trivial).
 *      Each work-item maps one grayscale pixel through that LUT.
 *
 * Both kernels expect grayscale (1-channel) input.
 * Convert colour images on the host or with grayscale.cl before calling these.
 */

/* ── Pass 1: histogram accumulation ─────────────────────────────────────── */
__kernel void build_histogram(
    __global const uchar* src,    /* grayscale input              */
    __global       int*   hist,   /* 256-element int array (init to 0) */
    int width,
    int height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    uchar val = src[y * width + x];
    atomic_add(&hist[(int)val], 1);
}

/* ── Pass 2: apply look-up table ─────────────────────────────────────────── */
__kernel void apply_lut(
    __global const uchar* src,   /* grayscale input  */
    __global       uchar* dst,   /* grayscale output */
    __global const uchar* lut,   /* 256-byte mapping */
    int width,
    int height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    dst[idx] = lut[src[idx]];
}
