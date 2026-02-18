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

from pathlib import Path
from typing import Dict, List, Tuple
from urllib.request import urlretrieve
import zipfile

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Zenodo download
# ---------------------------------------------------------------------------

ZENODO_URL = "https://zenodo.org/record/5036991/files/PixBox.zip"

# CMIX class labels as stored in the PixBox CSV mapped to CloudSEN12 integers.
# Adjust if the actual CSV uses different string/integer conventions.
PIXBOX_LABEL_MAP: Dict[str, int] = {
    "clear": 0,
    "thick": 1,
    "thin": 2,
    "shadow": 3,
    # Numeric variants in case the CSV stores integers as strings.
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
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


def download_pixbox(local_dir: str = "./data/pixbox") -> Path:
    """Download and extract the PixBox dataset from Zenodo.

    Skips the download if the target directory already exists and is non-empty.

    Args:
        local_dir: Destination directory for the extracted dataset.

    Returns:
        Path to the extracted dataset directory.
    """
    out_dir = Path(local_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    zip_path = out_dir / "PixBox.zip"

    if any(out_dir.iterdir()):
        # Check for the actual data files, not just the zip.
        csv_files = list(out_dir.rglob("*.csv"))
        if csv_files:
            print(f"PixBox already downloaded ({len(csv_files)} CSV files found).")
            return out_dir

    print(f"Downloading PixBox from Zenodo...")
    print(f"  URL: {ZENODO_URL}")
    urlretrieve(ZENODO_URL, zip_path)
    print(f"  Saved to: {zip_path}")

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    zip_path.unlink()
    print(f"Extraction complete: {out_dir}")

    return out_dir


# ---------------------------------------------------------------------------
# Label parsing
# ---------------------------------------------------------------------------

def _parse_label(raw: str) -> int:
    """Convert a raw label string to a CloudSEN12 integer class.

    Args:
        raw: Raw label value from the CSV (string or numeric).

    Returns:
        Integer class index (0=Clear, 1=Thick, 2=Thin, 3=Shadow).

    Raises:
        ValueError: If the label cannot be mapped.
    """
    key = str(raw).strip().lower()
    if key in PIXBOX_LABEL_MAP:
        return PIXBOX_LABEL_MAP[key]
    raise ValueError(
        f"Unknown CMIX label '{raw}'. "
        f"Expected one of: {list(PIXBOX_LABEL_MAP.keys())}"
    )


def load_pixbox_labels(pixbox_dir: str) -> Dict[str, pd.DataFrame]:
    """Load PixBox reference labels grouped by Sentinel-2 scene.

    Searches for CSV files in the dataset directory. Each CSV is expected
    to contain columns for the scene identifier, pixel row/column (or
    lat/lon) and the reference class label.

    The exact column names are inferred automatically from a set of known
    variants used across PixBox releases.

    Args:
        pixbox_dir: Root directory of the extracted PixBox dataset.

    Returns:
        Dictionary mapping scene_id -> DataFrame with columns:
            row, col, label (CloudSEN12 integer class).

    Raises:
        FileNotFoundError: If no CSV files are found.
        KeyError: If expected columns are absent from the CSV.
    """
    root = Path(pixbox_dir)
    csv_files = sorted(root.rglob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in '{root}'. "
            "Verify that the PixBox dataset was extracted correctly."
        )

    print(f"Found {len(csv_files)} CSV file(s) in '{root}'.")

    # Column name aliases for different PixBox versions.
    _row_aliases = {"row", "pixel_row", "y", "line"}
    _col_aliases = {"col", "pixel_col", "column", "x", "sample"}
    _label_aliases = {"class", "label", "reference", "ref_class", "cloud_mask"}
    _scene_aliases = {"scene", "scene_id", "granule", "tile", "image"}

    def _find_col(columns: List[str], aliases: set) -> str:
        for c in columns:
            if c.lower() in aliases:
                return c
        raise KeyError(
            f"Could not find column. Searched aliases: {aliases}. "
            f"Available columns: {columns}"
        )

    scenes: Dict[str, List[pd.DataFrame]] = {}

    for csv_path in csv_files:
        df_raw = pd.read_csv(csv_path)
        cols = list(df_raw.columns)

        row_col = _find_col(cols, _row_aliases)
        col_col = _find_col(cols, _col_aliases)
        label_col = _find_col(cols, _label_aliases)

        # Scene identifier: use filename stem if no scene column exists.
        try:
            scene_col = _find_col(cols, _scene_aliases)
            scene_ids = df_raw[scene_col].astype(str)
        except KeyError:
            scene_ids = pd.Series([csv_path.stem] * len(df_raw))

        df_raw["_scene"] = scene_ids
        df_raw["_row"] = df_raw[row_col].astype(int)
        df_raw["_col"] = df_raw[col_col].astype(int)
        df_raw["_label"] = df_raw[label_col].apply(_parse_label)

        for scene_id, group in df_raw.groupby("_scene"):
            entry = group[["_row", "_col", "_label"]].rename(
                columns={"_row": "row", "_col": "col", "_label": "label"}
            ).reset_index(drop=True)
            scenes.setdefault(scene_id, []).append(entry)

    result: Dict[str, pd.DataFrame] = {
        scene_id: pd.concat(frames, ignore_index=True)
        for scene_id, frames in scenes.items()
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
        pred_mask: Full-scene prediction array of shape (H, W) with integer
            class indices.
        pixels_df: DataFrame with columns 'row' and 'col' (pixel coordinates
            within the scene, zero-indexed).

    Returns:
        1-D integer array of predicted classes, aligned with pixels_df rows.
    """
    rows = pixels_df["row"].to_numpy(dtype=np.int64)
    cols = pixels_df["col"].to_numpy(dtype=np.int64)

    h, w = pred_mask.shape
    valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    if not valid.all():
        n_invalid = (~valid).sum()
        print(
            f"  WARNING: {n_invalid} pixel coordinate(s) are outside the "
            f"scene bounds ({h}x{w}) and will be skipped."
        )

    preds = np.full(len(rows), fill_value=-1, dtype=np.int64)
    preds[valid] = pred_mask[rows[valid], cols[valid]]
    return preds


# ---------------------------------------------------------------------------
# Experiment filtering helpers
# ---------------------------------------------------------------------------

def apply_experiment_filter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    experiment: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert multi-class arrays to binary arrays for a CMIX experiment.

    Pixels belonging to the 'exclude' set of the experiment (e.g., thin
    cloud in the thick-only experiment) are dropped.

    Args:
        y_true: Reference class labels (CloudSEN12 integers).
        y_pred: Predicted class labels (CloudSEN12 integers).
        experiment: Key in CMIX_EXPERIMENTS ('all_clouds' or 'thick_only').

    Returns:
        Tuple (y_true_bin, y_pred_bin, valid_mask, original_indices):
            y_true_bin: Binary reference (1=positive, 0=negative).
            y_pred_bin: Binary prediction (1=positive, 0=negative).
            valid_mask: Boolean mask applied to the input arrays.
    """
    cfg = CMIX_EXPERIMENTS[experiment]
    pos = cfg["pos"]
    neg = cfg["neg"]
    exclude = cfg["exclude"]

    # Keep only pixels that are not in the excluded set and are valid (>=0).
    valid = np.isin(y_true, list(pos | neg)) & (y_true >= 0) & (y_pred >= 0)

    y_true_filtered = y_true[valid]
    y_pred_filtered = y_pred[valid]

    y_true_bin = np.isin(y_true_filtered, list(pos)).astype(np.int64)
    y_pred_bin = np.isin(y_pred_filtered, list(pos)).astype(np.int64)

    return y_true_bin, y_pred_bin, valid