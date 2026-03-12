/*
 * brightness.cl
 * ─────────────
 * Brightness / contrast adjustment: dst = clamp(alpha*src + beta, 0, 255)
 *
 * Parameters alpha and beta match the CPU versions (alpha=1.2, beta=30).
 * Two kernels: one for grayscale (1-channel), one for BGR (3-channel).
 */

/* ── Grayscale (1-channel) ─────────────────────────────────────────────────── */
__kernel void brightness_gray(
    __global const uchar* src,
    __global       uchar* dst,
    int   width,
    int   height,
    float alpha,
    int   beta)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    int v   = (int)((float)src[idx] * alpha + (float)beta);
    dst[idx] = (uchar)clamp(v, 0, 255);
}

/* ── BGR (3-channel) ─────────────────────────────────────────────────────── */
__kernel void brightness_color(
    __global const uchar* src,
    __global       uchar* dst,
    int   width,
    int   height,
    float alpha,
    int   beta)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    int base = (y * width + x) * 3;
    for (int c = 0; c < 3; ++c) {
        int v = (int)((float)src[base + c] * alpha + (float)beta);
        dst[base + c] = (uchar)clamp(v, 0, 255);
    }
}
