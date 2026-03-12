#!/usr/bin/env python3
"""Generate a synthetic 4096x4096 BGR test image.
Tries backends in order: cv2 → Pillow → PPM fallback (no external deps)."""
import os, sys

os.makedirs("test_images", exist_ok=True)
W, H = 4096, 4096

# ── Backend 1: OpenCV Python ──────────────────────────────────────────────────
try:
    import cv2
    import numpy as np

    img = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
    cv2.imwrite("test_images/sample.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"Created test_images/sample.jpg ({W}x{H}) [via cv2]")
    sys.exit(0)
except ImportError:
    pass

# ── Backend 2: Pillow ─────────────────────────────────────────────────────────
try:
    from PIL import Image
    import numpy as np

    arr = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
    Image.fromarray(arr, "RGB").save("test_images/sample.jpg", quality=95)
    print(f"Created test_images/sample.jpg ({W}x{H}) [via Pillow]")
    sys.exit(0)
except ImportError:
    pass

# ── Backend 3: Pure Python — write PPM (no dependencies) ─────────────────────
import random

print("cv2 and Pillow not found; writing PPM fallback (install Pillow for JPEG).")
with open("test_images/sample.ppm", "wb") as f:
    f.write(f"P6\n{W} {H}\n255\n".encode())
    rng = random.Random(42)
    chunk = 128  # rows per write to keep memory modest
    for row_start in range(0, H, chunk):
        rows = min(chunk, H - row_start)
        f.write(bytes(rng.randint(0, 255) for _ in range(rows * W * 3)))
print(f"Created test_images/sample.ppm ({W}x{H}) [pure Python]")
print("Tip: set IMAGE=test_images/sample.ppm in make run, or install Pillow:")
print("       pip3 install --user Pillow")
