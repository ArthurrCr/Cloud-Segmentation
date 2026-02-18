"""PixBox / CMIX dataset loader.

Downloads the PixBox reference labels from Zenodo (doi: 10.5281/zenodo.5036991)
and parses them into a structured format ready for evaluation against full-scene
model predictions.

CMIX defines 17,351 expert-labeled pixels distributed across 29 full
Sentinel-2 L1C scenes. Labels use a four-class scheme that maps directly
to CloudSEN12 classes.

CloudSEN12 class mapping
------------------------
0 = Clear
1 = Thick Cloud
2 = Thin Cloud
3 = Cloud Shadow

CMIX experiment mapping
-----------------------
Experiment 1 (all clouds):  positive = {1, 2},  negative = {0, 3}
Experiment 2 (thick only):  positive = {1},      negative = {0, 3},
                            pixels with label=2 are excluded entirely.
"""

import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests


# ---------------------------------------------------------------------------
# Zenodo download
# ---------------------------------------------------------------------------

# The PixBox dataset ships as a single 1.5 MB ZIP (PixBox-S2-CMIX.zip)
# containing the CSV and description txt. The Sentinel-2 scenes are in a
# separate 22 GB ZIP that must be obtained independently via SCENES_DIR.
PIXBOX_ZIP_URL = (
    "https://zenodo.org/records/5036991/files/PixBox-S2-CMIX.zip?download=1"
)
PIXBOX_ZIP_FILENAME = "PixBox-S2-CMIX.zip"
PIXBOX_CSV_FILENAME = "pixbox_sentinel2_cmix_20180425.csv"
PIXBOX_DESC_FILENAME = "pixbox_sentinel2_cmix_20180425_description.txt"

# CMIX class labels as stored in the PixBox CSV mapped to CloudSEN12 integers.
# The CSV stores 1-based integer codes:
#   1 = Clear, 2 = Thick cloud, 3 = Thin cloud, 4 = Cloud shadow
# Remapped to CloudSEN12 0-based: 0=Clear, 1=Thick, 2=Thin, 3=Shadow.
PIXBOX_LABEL_MAP: Dict[str, int] = {
    # PixBox native 1-based codes.
    "1": 0,   # Clear       -> CloudSEN12 Clear
    "2": 1,   # Thick cloud -> CloudSEN12 Thick
    "3": 2,   # Thin cloud  -> CloudSEN12 Thin
    "4": 3,   # Shadow      -> CloudSEN12 Shadow
    # String variants (lower-cased), in case the CSV uses text labels.
    "clear": 0,
    "thick": 1,
    "thick cloud": 1,
    "thin": 2,
    "thin cloud": 2,
    "shadow": 3,
    "cloud shadow": 3,
}

CMIX_EXPERIMENTS: Dict[str, Dict] = {
    "all_clouds": {
        "description": "Thick + Thin cloud vs. Clear + Shadow",
        "pos": {1, 2},
        "neg": {0, 3},
        "exclude": set(),
    },
    "thick_only": {
        "description": "Thick cloud vs. Clear + Shadow (thin excluded)",
        "pos": {1},
        "neg": {0, 3},
        "exclude": {2},
    },
}


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _download_file(url: str, dest: Path) -> None:
    """Stream-download a file via requests."""
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=65536):
            f.write(chunk)


def download_pixbox(local_dir: str = "./data/pixbox") -> Path:
    """Download and extract the PixBox-S2-CMIX ZIP from Zenodo.

    Downloads ``PixBox-S2-CMIX.zip`` (1.5 MB) which contains the CSV of
    17,351 labeled pixels and the class-description txt file. Skips the
    download if the CSV already exists on disk.

    Note: The 29 Sentinel-2 L1C scenes (22 GB ZIP) must be downloaded
    separately and pointed to via the ``scenes_dir`` argument of
    ``CmixEvaluator``.

    Args:
        local_dir: Destination directory for the extracted files.

    Returns:
        Path to the dataset directory.
    """
    out_dir = Path(local_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_dest = out_dir / PIXBOX_CSV_FILENAME
    if csv_dest.exists():
        print(f"PixBox CSV already present: {csv_dest}")
        return out_dir

    zip_dest = out_dir / PIXBOX_ZIP_FILENAME
    print(f"Downloading PixBox-S2-CMIX.zip from Zenodo...")
    print(f"  URL: {PIXBOX_ZIP_URL}")
    _download_file(PIXBOX_ZIP_URL, zip_dest)
    print(f"  Saved: {zip_dest} ({zip_dest.stat().st_size / 1024:.0f} KB)")

    print("Extracting...")
    with zipfile.ZipFile(zip_dest, "r") as zf:
        zf.extractall(out_dir)
    zip_dest.unlink()

    # Verify expected files were extracted.
    for fname in (PIXBOX_CSV_FILENAME, PIXBOX_DESC_FILENAME):
        matches = list(out_dir.rglob(fname))
        if not matches:
            print(f"  WARNING: '{fname}' not found after extraction.")
        else:
            print(f"  Extracted: {matches[0]}")

    print(f"PixBox download complete: {out_dir}")
    return out_dir


# ---------------------------------------------------------------------------
# Label parsing
# ---------------------------------------------------------------------------

def _parse_label(raw) -> int:
    """Convert a raw CSV label to a CloudSEN12 integer class (0-3).

    Args:
        raw: Raw label value from the CSV (int or string).

    Returns:
        Integer class index (0=Clear, 1=Thick, 2=Thin, 3=Shadow).

    Raises:
        ValueError: If the label cannot be mapped.
    """
    key = str(raw).strip().lower()
    if key in PIXBOX_LABEL_MAP:
        return PIXBOX_LABEL_MAP[key]
    raise ValueError(
        f"Unknown PixBox label '{raw}'. "
        f"Expected one of: {list(PIXBOX_LABEL_MAP.keys())}"
    )


def load_pixbox_labels(pixbox_dir: str) -> Dict[str, pd.DataFrame]:
    """Load PixBox reference labels grouped by Sentinel-2 scene.

    Parses ``pixbox_sentinel2_cmix_20180425.csv`` from the downloaded
    PixBox-S2-CMIX ZIP. Column names are matched via aliases to handle
    minor schema variants across PixBox releases.

    Expected CSV schema (from the PixBox description file)::

        product_id, row, col, class_id, [other columns...]

    where ``class_id`` uses 1-based integer codes:
        1=Clear, 2=Thick cloud, 3=Thin cloud, 4=Cloud shadow.

    Args:
        pixbox_dir: Directory containing the extracted PixBox CSV file.

    Returns:
        Dictionary mapping scene/product_id -> DataFrame with columns:
            row, col, label (CloudSEN12 integer: 0=Clear, 1=Thick,
            2=Thin, 3=Shadow).

    Raises:
        FileNotFoundError: If no CSV files are found.
        KeyError: If expected columns are absent (available columns are
            printed to assist debugging).
    """
    root = Path(pixbox_dir)

    # Prefer the canonical filename; fall back to any CSV found.
    canonical = root / PIXBOX_CSV_FILENAME
    if canonical.exists():
        csv_files = [canonical]
    else:
        # The ZIP may extract into a subdirectory.
        csv_files = sorted(root.rglob(PIXBOX_CSV_FILENAME))
        if not csv_files:
            csv_files = sorted(root.rglob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in '{root}'. "
            f"Run download_pixbox('{pixbox_dir}') first."
        )

    print(f"Found {len(csv_files)} CSV file(s) in '{root}'.")

    # Column aliases — ordered by priority (first match wins).
    _row_aliases = {"row", "pixel_row", "line", "y_pixel", "y"}
    _col_aliases = {"col", "column", "pixel_col", "sample", "x_pixel", "x"}
    _label_aliases = {
        "class_id", "class", "label", "cloud_class",
        "reference", "ref_class", "cloud_mask",
    }
    _scene_aliases = {
        "product_id", "scene_id", "scene", "granule",
        "tile", "image", "s2_product",
    }

    def _find_col(columns: List[str], aliases: set, context: str) -> str:
        for c in columns:
            if c.lower() in aliases:
                return c
        raise KeyError(
            f"Cannot identify {context} column.\n"
            f"  Searched aliases : {sorted(aliases)}\n"
            f"  Available columns: {columns}\n"
            f"  Check the description .txt in '{root}' for the exact schema."
        )

    scenes: Dict[str, List[pd.DataFrame]] = {}

    for csv_path in csv_files:
        df_raw = pd.read_csv(csv_path)
        cols = list(df_raw.columns)
        print(f"  Columns in '{csv_path.name}': {cols}")

        row_col = _find_col(cols, _row_aliases, "row")
        col_col = _find_col(cols, _col_aliases, "col")
        label_col = _find_col(cols, _label_aliases, "label")

        try:
            scene_col = _find_col(cols, _scene_aliases, "scene_id")
            scene_ids = df_raw[scene_col].astype(str)
        except KeyError:
            scene_ids = pd.Series([csv_path.stem] * len(df_raw))

        df_raw["_scene"] = scene_ids
        df_raw["_row"] = df_raw[row_col].astype(int)
        df_raw["_col"] = df_raw[col_col].astype(int)
        df_raw["_label"] = df_raw[label_col].apply(_parse_label)

        for scene_id, group in df_raw.groupby("_scene"):
            entry = (
                group[["_row", "_col", "_label"]]
                .rename(columns={"_row": "row", "_col": "col", "_label": "label"})
                .reset_index(drop=True)
            )
            scenes.setdefault(str(scene_id), []).append(entry)

    result: Dict[str, pd.DataFrame] = {
        sid: pd.concat(frames, ignore_index=True)
        for sid, frames in scenes.items()
    }

    total_pixels = sum(len(df) for df in result.values())
    print(f"Loaded {total_pixels} labeled pixels across {len(result)} scenes.")
    return result


# ---------------------------------------------------------------------------
# Prediction extraction
# ---------------------------------------------------------------------------

def extract_predictions_at_pixels(
    pred_mask: np.ndarray,
    pixels_df: pd.DataFrame,
) -> np.ndarray:
    """Extract model predictions at labeled pixel coordinates.

    Args:
        pred_mask: Full-scene prediction array (H, W) with integer class
            indices (0-based CloudSEN12).
        pixels_df: DataFrame with columns 'row' and 'col' (zero-indexed
            pixel coordinates within the scene).

    Returns:
        1-D integer array of predicted classes, aligned with pixels_df rows.
        Coordinates outside scene bounds are returned as -1.
    """
    rows = pixels_df["row"].to_numpy(dtype=np.int64)
    cols = pixels_df["col"].to_numpy(dtype=np.int64)

    h, w = pred_mask.shape
    valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    if not valid.all():
        n_invalid = (~valid).sum()
        print(
            f"  WARNING: {n_invalid} pixel coordinate(s) outside scene bounds "
            f"({h}x{w}) — these will be excluded from metrics."
        )

    preds = np.full(len(rows), fill_value=-1, dtype=np.int64)
    preds[valid] = pred_mask[rows[valid], cols[valid]]
    return preds


# ---------------------------------------------------------------------------
# Experiment filtering
# ---------------------------------------------------------------------------

def apply_experiment_filter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    experiment: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert multi-class arrays to binary arrays for a CMIX experiment.

    Pixels belonging to the 'exclude' set (e.g., thin cloud in the
    thick-only experiment) and invalid pixels (value -1) are dropped.

    Args:
        y_true: Reference class labels (CloudSEN12 integers).
        y_pred: Predicted class labels (CloudSEN12 integers).
        experiment: Key in CMIX_EXPERIMENTS ('all_clouds' or 'thick_only').

    Returns:
        Tuple (y_true_bin, y_pred_bin, valid_mask):
            y_true_bin: Binary reference (1=positive, 0=negative).
            y_pred_bin: Binary prediction (1=positive, 0=negative).
            valid_mask: Boolean mask applied to input arrays.
    """
    cfg = CMIX_EXPERIMENTS[experiment]
    pos = cfg["pos"]
    neg = cfg["neg"]

    valid = (
        np.isin(y_true, list(pos | neg))
        & (y_true >= 0)
        & (y_pred >= 0)
    )

    y_true_bin = np.isin(y_true[valid], list(pos)).astype(np.int64)
    y_pred_bin = np.isin(y_pred[valid], list(pos)).astype(np.int64)

    return y_true_bin, y_pred_bin, valid