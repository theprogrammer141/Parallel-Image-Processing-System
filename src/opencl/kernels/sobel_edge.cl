/*
 * sobel_edge.cl
 * ─────────────
 * Sobel edge detection on a grayscale (1-channel) image.
 *
 *   Gx:  -1  0  +1      Gy:  -1  -2  -1
 *         -2  0  +2            0   0   0
 *         -1  0  +1           +1  +2  +1
 *
 * Gradient magnitude: |Gx|+|Gy| is a fast approximation;
 * we use the true Euclidean magnitude clamped to [0, 255].
 * Border pixels are set to 0 (no neighbour ring available).
 *
 * Input (src) must be grayscale — convert with grayscale.cl first
 * if the source image is colour.
 */

__kernel void sobel_edge(
    __global const uchar* src,   /* grayscale input  */
    __global       uchar* dst,   /* grayscale output */
    int width,
    int height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    /* Border pixels → 0 */
    if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
        dst[y * width + x] = 0;
        return;
    }

    /* 3×3 neighbourhood */
    int tl = (int)src[(y-1)*width + (x-1)];
    int tc = (int)src[(y-1)*width +  x   ];
    int tr = (int)src[(y-1)*width + (x+1)];
    int ml = (int)src[ y   *width + (x-1)];
    int mr = (int)src[ y   *width + (x+1)];
    int bl = (int)src[(y+1)*width + (x-1)];
    int bc = (int)src[(y+1)*width +  x   ];
    int br = (int)src[(y+1)*width + (x+1)];

    int gx = -tl + tr - 2*ml + 2*mr - bl + br;
    int gy = -tl - 2*tc - tr + bl + 2*bc + br;

    int mag = (int)sqrt((float)(gx*gx + gy*gy));
    dst[y * width + x] = (uchar)(mag > 255 ? 255 : mag);
}
