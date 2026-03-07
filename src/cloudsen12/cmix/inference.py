"""Full-scene sliding-window inference for Sentinel-2 L1C tiles.

The CMIX benchmark requires processing complete Sentinel-2 scenes
(typically 10980x10980 pixels at 10 m resolution). This module implements:

1. Multi-band scene loading with on-the-fly resampling to a common resolution.
2. Sliding-window patch extraction with configurable overlap.
3. Two patch grid strategies:
   - **Regular grid**: standard sliding window (default).
   - **Offset grid**: alternate rows shifted by half the patch width,
     replicating the CloudS2Mask strategy (Wright et al. 2024, Fig. 4).
4. Optional gradient (edge) weighting for smoother patch merging
   (Wright et al. 2024, Fig. 6).
5. Logit accumulation and merging across overlapping patches.
6. Final argmax to produce the full-scene prediction mask.

Band order expected by all models (13 bands):
    B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10, B11, B12

References
----------
Wright, N. et al. (2024). CloudS2Mask: A novel deep learning approach for
improved cloud and cloud shadow masking in Sentinel-2 imagery.
Remote Sensing of Environment, 306, 114122.
"""

from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import math

import numpy as np
import rasterio
import torch
from rasterio.enums import Resampling
from tqdm import tqdm


# Sentinel-2 band filename identifiers (SAFE format and plain GeoTIFF).
BAND_IDENTIFIERS = [
    "B01", "B02", "B03", "B04", "B05",
    "B06", "B07", "B08", "B8A", "B09",
    "B10", "B11", "B12",
]

# Native resolution (metres) for each band.
BAND_NATIVE_RES: dict = {
    "B01": 60, "B02": 10, "B03": 10, "B04": 10,
    "B05": 20, "B06": 20, "B07": 20, "B08": 10,
    "B8A": 20, "B09": 60, "B10": 60, "B11": 20,
    "B12": 20,
}


# ---------------------------------------------------------------------------
# Scene loading
# ---------------------------------------------------------------------------

def _find_band_file(scene_dir: Path, band_id: str) -> Path:
    """Locate the GeoTIFF file for a given band inside a scene directory.

    Searches recursively for files whose name contains the band identifier
    (e.g., 'B02', 'B8A'). Supports both .SAFE directory structures and
    flat GeoTIFF collections.

    Args:
        scene_dir: Root directory of the Sentinel-2 scene.
        band_id: Band identifier string, e.g. 'B02'.

    Returns:
        Path to the matching file.

    Raises:
        FileNotFoundError: If no matching file is found.
    """
    # Prioritise exact suffix match to avoid 'B8' matching 'B8A'.
    for ext in ("*.tif", "*.tiff", "*.jp2"):
        candidates = list(scene_dir.rglob(ext))
        # Filter: the stem or a component must contain band_id exactly.
        matches = [
            p for p in candidates
            if (f"_{band_id}_" in p.name
                or p.name.endswith(f"_{band_id}.tif")
                or p.name.endswith(f"_{band_id}.tiff")
                or p.name.endswith(f"_{band_id}.jp2")
                or p.stem.upper().endswith(band_id.upper()))
        ]
        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"Could not find band '{band_id}' in '{scene_dir}'."
    )


def load_scene_bands(
    scene_dir: str,
    target_resolution: int = 10,
    dtype: np.dtype = np.float16,
) -> Tuple[np.ndarray, dict]:
    """Load all 13 Sentinel-2 bands, resampling to a common resolution.

    Args:
        scene_dir: Path to the Sentinel-2 scene directory (SAFE or flat).
        target_resolution: Spatial resolution in metres. All bands will be
            resampled to this resolution. Default is 10 m.
        dtype: NumPy dtype for the output array. Default is float16 to
            reduce memory (~3.1 GB vs 6.3 GB for a full S2 scene).

    Returns:
        Tuple (image, meta):
            image: Array of shape (13, H, W) with raw DN values.
            meta: rasterio metadata dict from the first 10 m band.
    """
    root = Path(scene_dir)
    stacked: List[np.ndarray] = []
    reference_meta = None

    for band_id in BAND_IDENTIFIERS:
        band_path = _find_band_file(root, band_id)
        native_res = BAND_NATIVE_RES[band_id]
        scale_factor = native_res / target_resolution  # >1 means upsample

        with rasterio.open(band_path) as src:
            new_h = math.ceil(src.height * scale_factor)
            new_w = math.ceil(src.width * scale_factor)

            band_data = src.read(
                1,
                out_shape=(new_h, new_w),
                resampling=Resampling.bilinear,
            ).astype(dtype)

            if reference_meta is None and native_res == target_resolution:
                reference_meta = src.meta.copy()
                reference_meta.update(count=13, dtype=str(dtype))

        stacked.append(band_data)

    # If we never hit a 10 m band (unlikely), use the last opened meta.
    if reference_meta is None:
        reference_meta = {}

    # Ensure all bands have the same spatial shape by cropping/padding to the
    # shape of the first band (typically the 10 m bands define the reference).
    ref_shape = stacked[0].shape
    aligned: List[np.ndarray] = []
    for arr in stacked:
        if arr.shape != ref_shape:
            h = min(arr.shape[0], ref_shape[0])
            w = min(arr.shape[1], ref_shape[1])
            padded = np.zeros(ref_shape, dtype=dtype)
            padded[:h, :w] = arr[:h, :w]
            aligned.append(padded)
        else:
            aligned.append(arr)

    image = np.stack(aligned, axis=0)  # (13, H, W)
    return image, reference_meta


# ---------------------------------------------------------------------------
# Patch grid strategies
# ---------------------------------------------------------------------------

def _compute_patch_grid(
    height: int,
    width: int,
    patch_size: int,
    stride: int,
) -> List[Tuple[int, int, int, int]]:
    """Return a regular grid of (row_start, row_end, col_start, col_end).

    The last patch in each dimension is adjusted to stay within bounds,
    ensuring every pixel is covered at least once.

    Args:
        height: Scene height in pixels.
        width: Scene width in pixels.
        patch_size: Patch height/width in pixels.
        stride: Step size between patches.

    Returns:
        List of (r0, r1, c0, c1) tuples (row/col in scene coordinates).
    """
    patches = []

    row_starts = list(range(0, height - patch_size + 1, stride))
    if not row_starts or row_starts[-1] + patch_size < height:
        row_starts.append(max(0, height - patch_size))

    col_starts = list(range(0, width - patch_size + 1, stride))
    if not col_starts or col_starts[-1] + patch_size < width:
        col_starts.append(max(0, width - patch_size))

    for r0 in row_starts:
        r1 = r0 + patch_size
        for c0 in col_starts:
            c1 = c0 + patch_size
            patches.append((r0, r1, c0, c1))

    return patches


def _compute_offset_grid(
    height: int,
    width: int,
    patch_size: int,
    stride: int,
) -> List[Tuple[int, int, int, int]]:
    """Return an offset grid of patches (Wright et al. 2024, Fig. 4).

    Alternate rows are shifted horizontally by half the patch width.
    This ensures that corner regions of one row overlap with the
    mid-edge of patches in adjacent rows rather than with other corners,
    improving prediction quality at overlap boundaries.

    Patches that would extend beyond scene bounds are clamped inward.
    Duplicate coordinates (from clamping) are removed.

    Args:
        height: Scene height in pixels.
        width: Scene width in pixels.
        patch_size: Patch height/width in pixels.
        stride: Step size between adjacent patches.

    Returns:
        Sorted, deduplicated list of (r0, r1, c0, c1) tuples.
    """
    half_patch = patch_size // 2
    patches_set: set = set()

    row_starts = list(range(0, height - patch_size + 1, stride))
    if not row_starts or row_starts[-1] + patch_size < height:
        row_starts.append(max(0, height - patch_size))

    col_starts_even = list(range(0, width - patch_size + 1, stride))
    if not col_starts_even or col_starts_even[-1] + patch_size < width:
        col_starts_even.append(max(0, width - patch_size))

    col_starts_odd = list(range(half_patch, width - patch_size + 1, stride))
    if not col_starts_odd or col_starts_odd[-1] + patch_size < width:
        col_starts_odd.append(max(0, width - patch_size))
    # Also include col=0 for odd rows to ensure left edge coverage.
    if col_starts_odd and col_starts_odd[0] > 0:
        col_starts_odd.insert(0, 0)

    for row_idx, r0 in enumerate(row_starts):
        r1 = r0 + patch_size
        col_starts = col_starts_even if row_idx % 2 == 0 else col_starts_odd
        for c0 in col_starts:
            c1 = c0 + patch_size
            patches_set.add((r0, r1, c0, c1))

    return sorted(patches_set)


# ---------------------------------------------------------------------------
# Gradient (edge) weighting
# ---------------------------------------------------------------------------

def _build_gradient_weight(
    patch_size: int,
    border: int = 64,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Build a 2D gradient weight map for smooth patch merging.

    Pixels near the patch centre receive weight 1.0; pixels within
    ``border`` pixels of any edge are linearly ramped from a small
    minimum up to 1.0. This avoids hard seams between adjacent patches
    (Wright et al. 2024, Fig. 6).

    Args:
        patch_size: Spatial size of the square patch.
        border: Width (in pixels) of the linear ramp at each edge.
            Defaults to 64, which works well with 512-pixel patches
            and 128-pixel overlap.
        dtype: Output dtype. Use float32 for the weight map even when
            accumulation buffers are float16 to avoid precision loss
            during the ramp computation.

    Returns:
        Array of shape (patch_size, patch_size) with values in
        [min_weight, 1.0].
    """
    if border <= 0 or border >= patch_size // 2:
        return np.ones((patch_size, patch_size), dtype=dtype)

    # Minimum weight at the very edge of the patch. Using 0.01 (1%)
    # ensures edge pixels still contribute but are strongly down-weighted.
    min_weight = dtype(0.01)
    ramp = np.ones(patch_size, dtype=dtype)
    linear = np.linspace(min_weight, 1.0, border, endpoint=True, dtype=dtype)
    ramp[:border] = linear
    ramp[-border:] = linear[::-1]

    # 2D weight: minimum of row-ramp and col-ramp at each pixel.
    weight_2d = np.minimum(
        ramp[np.newaxis, :],  # horizontal ramp
        ramp[:, np.newaxis],  # vertical ramp
    )
    return weight_2d


# ---------------------------------------------------------------------------
# Sliding-window inference
# ---------------------------------------------------------------------------

def run_full_scene_inference(
    image: np.ndarray,
    models: List[torch.nn.Module],
    device: Union[str, torch.device],
    patch_size: int = 512,
    stride: int = 256,
    batch_size: int = 8,
    normalize_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    num_classes: int = 4,
    use_offset_grid: bool = False,
    use_gradient_weight: bool = False,
    gradient_border: int = 64,
) -> np.ndarray:
    """Run sliding-window inference on a full Sentinel-2 scene.

    Supports two grid strategies and optional gradient weighting for
    smoother patch merging, replicating CloudS2Mask's inference pipeline.

    Memory-optimised for Colab / low-RAM environments:
    - Accumulation buffers use float32 for numerical stability.
    - Argmax is computed in row chunks to avoid a full-scene temp array.

    Args:
        image: Array (C, H, W) of raw DN values (any float dtype).
        models: List of models in eval mode.
        device: PyTorch device.
        patch_size: Spatial size of each square patch.
        stride: Step size between adjacent patches. For 128-pixel
            overlap with 512 patches, use stride=384.
        batch_size: Number of patches per forward pass.
        normalize_fn: Optional callable for input normalisation.
            Receives (B, C, H, W) float32 tensor, returns same shape.
        num_classes: Number of output classes.
        use_offset_grid: If True, uses the offset grid strategy where
            alternate rows are shifted by half the patch width
            (Wright et al. 2024). Recommended for CloudS2Mask models.
        use_gradient_weight: If True, applies a linear gradient ramp
            at patch edges before accumulation, producing smoother
            transitions between patches (Wright et al. 2024, Fig. 6).
        gradient_border: Width of the edge gradient ramp in pixels.
            Only used when use_gradient_weight=True. Default 64 works
            well with 512-pixel patches and 128-pixel overlap.

    Returns:
        Integer prediction mask of shape (H, W).
    """
    _, H, W = image.shape
    device = torch.device(device)

    # Accumulation buffers — float32 for numerical stability with
    # gradient weighting; ~1.9 GB for 4 classes on a 10980x10980 scene.
    weight_sum = np.zeros((num_classes, H, W), dtype=np.float32)
    weight_total = np.zeros((H, W), dtype=np.float32)

    # Select grid strategy.
    if use_offset_grid:
        patches = _compute_offset_grid(H, W, patch_size, stride)
    else:
        patches = _compute_patch_grid(H, W, patch_size, stride)

    n_patches = len(patches)
    grid_name = "offset" if use_offset_grid else "regular"
    print(
        f"Full-scene inference: {H}x{W} scene | "
        f"patch={patch_size} stride={stride} ({grid_name} grid) | "
        f"{n_patches} patches | batch={batch_size}"
    )

    # Precompute gradient weight map (or uniform weights).
    if use_gradient_weight:
        patch_weight = _build_gradient_weight(
            patch_size, border=gradient_border, dtype=np.float32
        )
        print(
            f"  Gradient weighting: border={gradient_border}px, "
            f"min={patch_weight.min():.4f}, max={patch_weight.max():.4f}"
        )
    else:
        patch_weight = np.ones((patch_size, patch_size), dtype=np.float32)

    for m in models:
        m.to(device).eval()

    with torch.no_grad():
        for batch_start in tqdm(
            range(0, n_patches, batch_size),
            desc="Predicting patches",
            unit="batch",
        ):
            batch_coords = patches[batch_start: batch_start + batch_size]

            # Build batch tensor directly — avoids intermediate list.
            batch_tensor = torch.from_numpy(
                np.stack(
                    [image[:, r0:r1, c0:c1] for r0, r1, c0, c1 in batch_coords],
                    axis=0,
                )
            ).float().to(device)  # (B, C, P, P) — always float32 on GPU

            if normalize_fn is not None:
                batch_tensor = normalize_fn(batch_tensor)

            # Ensemble: average softmax probabilities across models.
            if len(models) > 1:
                probs = torch.stack(
                    [torch.softmax(m(batch_tensor), dim=1) for m in models]
                ).mean(dim=0)
            else:
                probs = torch.softmax(models[0](batch_tensor), dim=1)

            probs_np = probs.cpu().numpy()  # (B, C, P, P) float32

            del batch_tensor, probs
            torch.cuda.empty_cache()

            for i, (r0, r1, c0, c1) in enumerate(batch_coords):
                # Apply gradient weight: broadcast (P, P) over classes.
                weighted = probs_np[i] * patch_weight[np.newaxis, :, :]
                weight_sum[:, r0:r1, c0:c1] += weighted
                weight_total[r0:r1, c0:c1] += patch_weight

    # Argmax in row chunks to avoid allocating a full (C, H, W) temp.
    pred_mask = np.empty((H, W), dtype=np.int32)
    chunk_rows = 512
    for r0 in range(0, H, chunk_rows):
        r1 = min(r0 + chunk_rows, H)
        wt = weight_total[r0:r1, :]
        wt = np.maximum(wt, 1e-8)  # avoid division by zero
        avg = weight_sum[:, r0:r1, :] / wt[np.newaxis, :, :]
        pred_mask[r0:r1, :] = np.argmax(avg, axis=0)

    del weight_sum, weight_total
    return pred_mask