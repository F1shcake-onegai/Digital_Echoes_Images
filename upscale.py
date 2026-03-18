"""
Image 4x upscaler using NVIDIA Video Effects SDK (SuperRes filter).
Reads all images from ./input, upscales 4x, saves to ./output.
"""

import ctypes
from ctypes import (
    c_int, c_uint, c_ubyte, c_float, c_char_p, c_void_p, c_ulonglong,
    POINTER, Structure, byref,
)
import numpy as np
import cv2
import os
from pathlib import Path

# ---- Constants ----

# Pixel formats
NVCV_BGR = 4
NVCV_RGBA = 5

# Component types
NVCV_U8 = 1
NVCV_F32 = 7

# Layout
NVCV_INTERLEAVED = 0
NVCV_PLANAR = 1

# Memory
NVCV_CPU = 0
NVCV_GPU = 1

# Effect / parameter selectors
NVVFX_FX_SUPER_RES = b"SuperRes"
NVVFX_INPUT_IMAGE = b"SrcImage0"
NVVFX_OUTPUT_IMAGE = b"DstImage0"
NVVFX_MODEL_DIRECTORY = b"ModelDir"
NVVFX_CUDA_STREAM = b"CudaStream"
NVVFX_MODE = b"Mode"

SCALE_FACTOR = 4
MIN_DIM = 90  # SDK minimum input dimension
MAX_OUT_W = 3840  # SDK maximum output width
MAX_OUT_H = 2160  # SDK maximum output height
MAX_IN_W = MAX_OUT_W // SCALE_FACTOR  # 960
MAX_IN_H = MAX_OUT_H // SCALE_FACTOR  # 540
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}

SDK_DIR = r"C:\Program Files\NVIDIA Corporation\NVIDIA Video Effects"
MODEL_DIR = os.environ.get("NVVFX_MODEL_DIR", os.path.join(SDK_DIR, "models"))


# ---- NvCVImage ctypes structure ----

class NvCVImage(Structure):
    _fields_ = [
        ("width",          c_uint),
        ("height",         c_uint),
        ("pitch",          c_int),
        ("pixelFormat",    c_int),
        ("componentType",  c_int),
        ("pixelBytes",     c_ubyte),
        ("componentBytes", c_ubyte),
        ("numComponents",  c_ubyte),
        ("planar",         c_ubyte),
        ("gpuMem",         c_ubyte),
        ("colorspace",     c_ubyte),
        ("reserved",       c_ubyte * 2),
        ("pixels",         c_void_p),
        ("deletePtr",      c_void_p),
        ("deleteProc",     c_void_p),
        ("bufferBytes",    c_ulonglong),
    ]


# ---- Load DLLs ----

if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(SDK_DIR)
os.environ["PATH"] = SDK_DIR + os.pathsep + os.environ.get("PATH", "")

nvvfx = ctypes.CDLL(os.path.join(SDK_DIR, "NVVideoEffects.dll"))
nvcv = ctypes.CDLL(os.path.join(SDK_DIR, "NVCVImage.dll"))


# ---- Function prototypes: NVVideoEffects.dll ----

nvvfx.NvVFX_CreateEffect.restype = c_int
nvvfx.NvVFX_CreateEffect.argtypes = [c_char_p, POINTER(c_void_p)]

nvvfx.NvVFX_DestroyEffect.restype = None
nvvfx.NvVFX_DestroyEffect.argtypes = [c_void_p]

nvvfx.NvVFX_SetString.restype = c_int
nvvfx.NvVFX_SetString.argtypes = [c_void_p, c_char_p, c_char_p]

nvvfx.NvVFX_SetU32.restype = c_int
nvvfx.NvVFX_SetU32.argtypes = [c_void_p, c_char_p, c_uint]

nvvfx.NvVFX_SetImage.restype = c_int
nvvfx.NvVFX_SetImage.argtypes = [c_void_p, c_char_p, POINTER(NvCVImage)]

nvvfx.NvVFX_SetCudaStream.restype = c_int
nvvfx.NvVFX_SetCudaStream.argtypes = [c_void_p, c_char_p, c_void_p]

nvvfx.NvVFX_Load.restype = c_int
nvvfx.NvVFX_Load.argtypes = [c_void_p]

nvvfx.NvVFX_Run.restype = c_int
nvvfx.NvVFX_Run.argtypes = [c_void_p, c_int]

nvvfx.NvVFX_CudaStreamCreate.restype = c_int
nvvfx.NvVFX_CudaStreamCreate.argtypes = [POINTER(c_void_p)]

nvvfx.NvVFX_CudaStreamDestroy.restype = None
nvvfx.NvVFX_CudaStreamDestroy.argtypes = [c_void_p]


# ---- Function prototypes: NVCVImage.dll ----

nvcv.NvCVImage_Alloc.restype = c_int
nvcv.NvCVImage_Alloc.argtypes = [
    POINTER(NvCVImage), c_uint, c_uint, c_int, c_int, c_uint, c_uint, c_uint
]

nvcv.NvCVImage_Init.restype = c_int
nvcv.NvCVImage_Init.argtypes = [
    POINTER(NvCVImage), c_uint, c_uint, c_int, c_void_p, c_int, c_int, c_uint, c_uint
]

nvcv.NvCVImage_Transfer.restype = c_int
nvcv.NvCVImage_Transfer.argtypes = [
    POINTER(NvCVImage), POINTER(NvCVImage), c_float, c_void_p, POINTER(NvCVImage)
]

nvcv.NvCVImage_Dealloc.restype = None
nvcv.NvCVImage_Dealloc.argtypes = [POINTER(NvCVImage)]

nvcv.NvCV_GetErrorStringFromCode.restype = c_char_p
nvcv.NvCV_GetErrorStringFromCode.argtypes = [c_int]


# ---- Helpers ----

def check(err, context=""):
    if err != 0:
        msg = nvcv.NvCV_GetErrorStringFromCode(err)
        msg = msg.decode() if msg else "unknown"
        raise RuntimeError(f"NvVFX error {err} ({msg}): {context}")


def make_cpu_image(arr, pixel_format=None):
    """Wrap a contiguous numpy array as a CPU NvCVImage."""
    arr = np.ascontiguousarray(arr)
    h, w = arr.shape[:2]
    if pixel_format is None:
        pixel_format = NVCV_RGBA if arr.shape[2] == 4 else NVCV_BGR
    img = NvCVImage()
    check(
        nvcv.NvCVImage_Init(
            byref(img), c_uint(w), c_uint(h),
            c_int(arr.strides[0]),
            c_void_p(arr.ctypes.data),
            c_int(pixel_format), c_int(NVCV_U8),
            c_uint(NVCV_INTERLEAVED), c_uint(NVCV_CPU),
        ),
        "NvCVImage_Init (CPU wrapper)",
    )
    return img


# ---- Main ----

def main():
    input_dir = Path("./input")
    output_dir = Path("./output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect image files
    image_files = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not image_files:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_files)} image(s) to upscale {SCALE_FACTOR}x")
    print(f"Model directory: {MODEL_DIR}")

    # Create effect
    handle = c_void_p()
    check(nvvfx.NvVFX_CreateEffect(NVVFX_FX_SUPER_RES, byref(handle)), "CreateEffect")
    check(nvvfx.NvVFX_SetString(handle, NVVFX_MODEL_DIRECTORY, MODEL_DIR.encode()), "SetModelDir")

    # Mode 1 = detail enhancement (best for high-quality source images)
    check(nvvfx.NvVFX_SetU32(handle, NVVFX_MODE, c_uint(1)), "SetMode")

    # Create CUDA stream
    stream = c_void_p()
    check(nvvfx.NvVFX_CudaStreamCreate(byref(stream)), "CudaStreamCreate")
    check(nvvfx.NvVFX_SetCudaStream(handle, NVVFX_CUDA_STREAM, stream), "SetCudaStream")

    # GPU buffers and separate staging buffers for upload/download
    src_gpu = NvCVImage()
    dst_gpu = NvCVImage()
    stg_in = NvCVImage()
    stg_out = NvCVImage()
    prev_w, prev_h = 0, 0

    try:
        for i, img_path in enumerate(image_files):
            print(f"[{i + 1}/{len(image_files)}] {img_path.name}")

            # Load image (BGR, uint8)
            bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if bgr is None:
                print(f"  SKIP: failed to load")
                continue

            orig_h, orig_w = bgr.shape[:2]

            if orig_w < MIN_DIM or orig_h < MIN_DIM:
                print(f"  SKIP: {orig_w}x{orig_h} is below minimum {MIN_DIM}x{MIN_DIM}")
                continue

            # Downscale input if 4x output would exceed SDK max (3840x2160)
            src_w, src_h = orig_w, orig_h
            if src_w > MAX_IN_W or src_h > MAX_IN_H:
                scale = min(MAX_IN_W / src_w, MAX_IN_H / src_h)
                src_w = int(src_w * scale)
                src_h = int(src_h * scale)

            # Align dimensions to multiple of 8 (required by the neural network)
            src_w = src_w - (src_w % 8) if src_w % 8 else src_w
            src_h = src_h - (src_h % 8) if src_h % 8 else src_h

            if src_w < MIN_DIM or src_h < MIN_DIM:
                print(f"  SKIP: aligned size {src_w}x{src_h} below minimum {MIN_DIM}x{MIN_DIM}")
                continue

            if src_w != orig_w or src_h != orig_h:
                bgr = cv2.resize(bgr, (src_w, src_h), interpolation=cv2.INTER_AREA)
                print(f"  Resized: {orig_w}x{orig_h} -> {src_w}x{src_h}")

            dst_w, dst_h = src_w * SCALE_FACTOR, src_h * SCALE_FACTOR
            print(f"  {src_w}x{src_h} -> {dst_w}x{dst_h}")

            bgr = np.ascontiguousarray(bgr)

            # Reallocate GPU buffers if resolution changed
            if src_w != prev_w or src_h != prev_h:
                if prev_w != 0:
                    nvcv.NvCVImage_Dealloc(byref(src_gpu))
                    nvcv.NvCVImage_Dealloc(byref(dst_gpu))
                    nvcv.NvCVImage_Dealloc(byref(stg_in))
                    nvcv.NvCVImage_Dealloc(byref(stg_out))
                    src_gpu = NvCVImage()
                    dst_gpu = NvCVImage()
                    stg_in = NvCVImage()
                    stg_out = NvCVImage()

                # GPU buffers: RGBA F32 planar (only format accepted by SetImage)
                check(
                    nvcv.NvCVImage_Alloc(
                        byref(src_gpu), c_uint(src_w), c_uint(src_h),
                        c_int(NVCV_RGBA), c_int(NVCV_F32),
                        c_uint(NVCV_PLANAR), c_uint(NVCV_GPU), c_uint(1),
                    ),
                    "Alloc src_gpu",
                )
                check(
                    nvcv.NvCVImage_Alloc(
                        byref(dst_gpu), c_uint(dst_w), c_uint(dst_h),
                        c_int(NVCV_RGBA), c_int(NVCV_F32),
                        c_uint(NVCV_PLANAR), c_uint(NVCV_GPU), c_uint(1),
                    ),
                    "Alloc dst_gpu",
                )

                # Rebind images and reload effect for new resolution
                check(nvvfx.NvVFX_SetImage(handle, NVVFX_INPUT_IMAGE, byref(src_gpu)), "SetInputImage")
                check(nvvfx.NvVFX_SetImage(handle, NVVFX_OUTPUT_IMAGE, byref(dst_gpu)), "SetOutputImage")
                print("  Loading model...")
                check(nvvfx.NvVFX_Load(handle), "Load")
                prev_w, prev_h = src_w, src_h

            # Wrap input BGR U8 as CPU NvCVImage (3-channel matches SDK's internal ncomp=3)
            src_cpu = make_cpu_image(bgr, NVCV_BGR)

            # Transfer CPU -> GPU (U8 [0,255] -> F32 [0,1])
            check(
                nvcv.NvCVImage_Transfer(
                    byref(src_cpu), byref(src_gpu),
                    c_float(1.0 / 255.0), stream, byref(stg_in),
                ),
                "Transfer src CPU->GPU",
            )

            # Run super resolution
            check(nvvfx.NvVFX_Run(handle, c_int(0)), "Run")

            # Prepare output buffer (BGR U8, 3 channels)
            out_bgr = np.empty((dst_h, dst_w, 3), dtype=np.uint8)
            out_bgr = np.ascontiguousarray(out_bgr)
            dst_cpu = make_cpu_image(out_bgr, NVCV_BGR)

            # Transfer GPU -> CPU (F32 [0,1] -> U8 [0,255])
            check(
                nvcv.NvCVImage_Transfer(
                    byref(dst_gpu), byref(dst_cpu),
                    c_float(255.0), stream, byref(stg_out),
                ),
                "Transfer dst GPU->CPU",
            )

            # Save
            out_path = output_dir / img_path.name
            if not cv2.imwrite(str(out_path), out_bgr):
                print(f"  ERROR: failed to save {out_path}")
                continue
            print(f"  Saved: {out_path}")

    finally:
        # Cleanup
        if prev_w != 0:
            nvcv.NvCVImage_Dealloc(byref(src_gpu))
            nvcv.NvCVImage_Dealloc(byref(dst_gpu))
            nvcv.NvCVImage_Dealloc(byref(stg_in))
            nvcv.NvCVImage_Dealloc(byref(stg_out))
        nvvfx.NvVFX_DestroyEffect(handle)
        nvvfx.NvVFX_CudaStreamDestroy(stream)

    print(f"Done. {len(image_files)} image(s) processed.")


if __name__ == "__main__":
    main()
