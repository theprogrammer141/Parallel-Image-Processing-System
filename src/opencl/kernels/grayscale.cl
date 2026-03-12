/*
 * grayscale.cl
 * ────────────
 * Converts a BGR (3-byte-per-pixel, interleaved) image to grayscale using
 * the ITU-R BT.601 luma coefficients — identical weights to the CPU versions.
 *
 * Each 2-D work-item processes one pixel.
 */

__kernel void grayscale(
    __global const uchar* src,   /* BGR input  — width*height*3 bytes */
    __global       uchar* dst,   /* gray output — width*height bytes  */
    int width,
    int height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    /* Guard: extra work-items beyond image bounds do nothing */
    if (x >= width || y >= height) return;

    int idx = (y * width + x) * 3;
    float b = (float)src[idx + 0];
    float g = (float)src[idx + 1];
    float r = (float)src[idx + 2];

    /* BT.601: Y = 0.114·B + 0.587·G + 0.299·R */
    dst[y * width + x] = (uchar)(0.114f * b + 0.587f * g + 0.299f * r);
}
