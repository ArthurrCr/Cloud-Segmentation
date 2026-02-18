"""Full-scene sliding-window inference for Sentinel-2 L1C tiles.

The CMIX benchmark requires processing complete Sentinel-2 scenes
(typically 10980x10980 pixels at 10 m resolution). This module implements:

1. Multi-band scene loading with on-the-fly resampling to a common resolution.
2. Sliding-window patch extraction with configurable overlap.
3. Logit accumulation and merging across overlapping patches.
4. Final argmax to produce the full-scene prediction mask.

Band order expected by all models (13 bands):
    B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10, B11, B12
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
    dtype: np.dtype = np.float32,
) -> Tuple[np.ndarray, dict]:
    """Load all 13 Sentinel-2 bands, resampling to a common resolution.

    Args:
        scene_dir: Path to the Sentinel-2 scene directory (SAFE or flat).
        target_resolution: Spatial resolution in metres. All bands will be
            resampled to this resolution. Default is 10 m.
        dtype: NumPy dtype for the output array.

    Returns:
        Tuple (image, meta):
            image: Float32 array of shape (13, H, W) with raw DN values.
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
# Sliding-window inference
# ---------------------------------------------------------------------------

def _compute_patch_grid(
    height: int,
    width: int,
    patch_size: int,
    stride: int,
) -> List[Tuple[int, int, int, int]]:
    """Return a list of (row_start, row_end, col_start, col_end) patches.

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


def run_full_scene_inference(
    image: np.ndarray,
    models: List[torch.nn.Module],
    device: Union[str, torch.device],
    patch_size: int = 512,
    stride: int = 256,
    batch_size: int = 8,
    normalize_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    num_classes: int = 4,
) -> np.ndarray:
    """Run sliding-window inference on a full Sentinel-2 scene.

    Logits from overlapping patches are accumulated in a float accumulation
    buffer and averaged before the final argmax, which gives smoother
    predictions at seam boundaries compared to majority voting.

    Args:
        image: Float32 array (13, H, W) of raw DN values.
        models: List of models in eval mode. Multiple models are ensembled
            by averaging their softmax probabilities.
        device: PyTorch device.
        patch_size: Spatial size of each square patch fed to the model.
        stride: Step size between adjacent patches. Use stride < patch_size
            for overlapping patches (recommended: stride = patch_size // 2).
        batch_size: Number of patches per forward pass.
        normalize_fn: Optional callable applied to each batch tensor before
            the forward pass. Should accept and return a (B, 13, H, W) tensor.
            Pass None to feed raw DN values to the model.
        num_classes: Number of output classes.

    Returns:
        Integer prediction mask of shape (H, W).
    """
    _, H, W = image.shape
    device = torch.device(device)

    # Accumulation buffers.
    logit_sum = np.zeros((num_classes, H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    patches = _compute_patch_grid(H, W, patch_size, stride)
    n_patches = len(patches)
    print(
        f"Full-scene inference: {H}x{W} scene | "
        f"patch={patch_size} stride={stride} | "
        f"{n_patches} patches | batch={batch_size}"
    )

    for m in models:
        m.to(device).eval()

    with torch.no_grad():
        for batch_start in tqdm(
            range(0, n_patches, batch_size),
            desc="Predicting patches",
            unit="batch",
        ):
            batch_coords = patches[batch_start: batch_start + batch_size]
            batch_arrays: List[np.ndarray] = []

            for r0, r1, c0, c1 in batch_coords:
                patch = image[:, r0:r1, c0:c1].copy()  # (13, P, P)
                batch_arrays.append(patch)

            batch_tensor = torch.from_numpy(
                np.stack(batch_arrays, axis=0)
            ).float().to(device)  # (B, 13, P, P)

            if normalize_fn is not None:
                batch_tensor = normalize_fn(batch_tensor)

            # Ensemble: average softmax probabilities across models.
            if len(models) > 1:
                probs = torch.stack(
                    [torch.softmax(m(batch_tensor), dim=1) for m in models]
                ).mean(dim=0)  # (B, C, P, P)
            else:
                probs = torch.softmax(models[0](batch_tensor), dim=1)

            probs_np = probs.cpu().numpy()  # (B, C, P, P)

            for i, (r0, r1, c0, c1) in enumerate(batch_coords):
                logit_sum[:, r0:r1, c0:c1] += probs_np[i]
                count_map[r0:r1, c0:c1] += 1.0

            torch.cuda.empty_cache()

    # Average accumulated probabilities and take argmax.
    count_map = np.maximum(count_map, 1.0)  # avoid division by zero
    avg_probs = logit_sum / count_map[np.newaxis, :, :]
    pred_mask = np.argmax(avg_probs, axis=0).astype(np.int32)
    return pred_mask