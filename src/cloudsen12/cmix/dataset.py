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
Experiment 1 (all types of clouds"):  positive = {1, 2},  negative = {0, 3}
Experiment 2 (without thin clouds):  positive = {1},      negative = {0, 3},
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

# CLOUD_CHARACTERISTICS_ID mapping to CloudSEN12 integers.
# The PixBox CSV uses a detailed 0-12 scheme for cloud characteristics:
#   0 = None of the classes    -> exclude (-1)
#   1 = None (no cloud)        -> 0 (Clear)
#   2 = Opaque                 -> 1 (Thick Cloud)
#   3 = Semi-transparent       -> 2 (Thin Cloud)
#   4 = Thick semi-transparent -> 2 (Thin Cloud)
#   5 = Avg semi-transparent   -> 2 (Thin Cloud)
#   6 = Thin semi-transparent  -> 2 (Thin Cloud)
#   7 = Clear                  -> 0 (Clear)
#   8 = Spatially mixed cloud  -> exclude (-1)
#   9 = Condensation trail     -> 2 (Thin Cloud)
#  10 = Fog                    -> 2 (Thin Cloud)
#  11 = Haze                   -> 2 (Thin Cloud)
#  12 = Cloud border           -> exclude (-1)
#
# Cloud Shadow is derived from SHADOW_ID == 3, handled in load_pixbox_labels.
CLOUD_CHAR_MAP: Dict[int, int] = {
    0: -1,   # None of the classes -> exclude
    1:  0,   # None (no cloud)     -> Clear
    2:  1,   # Opaque              -> Thick Cloud
    3:  2,   # Semi-transparent    -> Thin Cloud
    4:  2,   # Thick semi-transp.  -> Thin Cloud
    5:  2,   # Avg semi-transp.    -> Thin Cloud
    6:  2,   # Thin semi-transp.   -> Thin Cloud
    7:  0,   # Clear               -> Clear
    8: -1,   # Spatially mixed     -> exclude
    9:  2,   # Condensation trail  -> Thin Cloud
    10: 2,   # Fog                 -> Thin Cloud
    11: 2,   # Haze                -> Thin Cloud
    12: -1,  # Cloud border        -> exclude
}

# SHADOW_ID values from the PixBox description.
# Only SHADOW_ID == 3 (Cloud shadow) maps to CloudSEN12 class 3.
SHADOW_CLOUD_ID = 3

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

# Hard-coded PRODUCT_ID -> SAFE name mapping from the PixBox description file.
# The CSV uses numeric PRODUCT_IDs, but the scene directories use SAFE names.
PRODUCT_ID_TO_SAFE: Dict[str, str] = {
    "5472572":    "S2A_MSIL1C_20171205T143751_N0206_R096_T19KEU_20171205T180316.SAFE",
    "183807605":  "S2A_MSIL1C_20170929T211511_N0205_R143_T06VWN_20170929T211510.SAFE",
    "273136892":  "S2A_MSIL1C_20180126T182631_N0206_R127_T11SPC_20180126T201415.SAFE",
    "433871481":  "S2A_MSIL1C_20180303T140051_N0206_R067_T19FDV_20180303T202004.SAFE",
    "522655885":  "S2A_MSIL1C_20171107T150421_N0206_R125_T22WES_20171107T165127.SAFE",
    "623596858":  "S2A_MSIL1C_20170629T103021_N0205_R108_T31TFJ_20170629T103020.SAFE",
    "701404038":  "S2A_MSIL1C_20170908T100031_N0205_R122_T32SPJ_20170908T100655.SAFE",
    "784585929":  "S2B_MSIL1C_20170731T102019_N0205_R065_T33VVE_20170731T102348.SAFE",
    "872013732":  "S2A_MSIL1C_20180223T092031_N0206_R093_T36VUM_20180223T113049.SAFE",
    "885047116":  "S2B_MSIL1C_20171210T060229_N0206_R091_T42RUN_20171210T071154.SAFE",
    "960993216":  "S2A_MSIL1C_20180106T032121_N0206_R118_T48NUG_20180106T083912.SAFE",
    "1041598282": "S2A_MSIL1C_20180222T012651_N0206_R074_T54TWN_20180222T031349.SAFE",
    "1126775487": "S2B_MSIL1C_20170725T183309_N0205_R127_T11SPC_20170725T183309.SAFE",
    "1211409580": "S2A_MSIL1C_20170209T154541_N0204_R111_T17PPK_20170209T154543.SAFE",
    "1248336059": "S2A_MSIL1C_20170725T142751_N0205_R053_T19GBQ_20170725T143854.SAFE",
    "1309065466": "S2B_MSIL1C_20170712T113319_N0205_R080_T28PCV_20170712T114542.SAFE",
    "1386821964": "S2A_MSIL1C_20170726T102021_N0205_R065_T33VVE_20170726T102259.SAFE",
    "1407300752": "S2A_MSIL1C_20170113T072241_N0204_R006_T40UEE_20170113T072238.SAFE",
    "1470797653": "S2A_MSIL1C_20180217T053911_N0206_R005_T44SKJ_20180217T082149.SAFE",
    "1559102360": "S2A_MSIL1C_20170917T052641_N0205_R105_T48XWG_20170917T052642.SAFE",
    "1650478641": "S2A_MSIL1C_20170916T143741_N0205_R096_T19KEU_20170916T143942.SAFE",
    "1727138395": "S2A_MSIL1C_20170620T181921_N0205_R127_T11SPC_20170620T182846.SAFE",
    "1727140056": "S2A_MSIL1C_20180302T142851_N0206_R053_T19GBQ_20180302T192732.SAFE",
    "1821333386": "S2B_MSIL1C_20180302T150259_N0206_R125_T22WES_20180302T183800.SAFE",
    "1832501017": "S2B_MSIL1C_20170728T101029_N0205_R022_T32TPS_20170728T101024.SAFE",
    "1892503998": "S2B_MSIL1C_20170916T101019_N0205_R022_T32SPJ_20170916T101354.SAFE",
    "1899752240": "S2A_MSIL1C_20170712T071621_N0205_R006_T40UEE_20170712T071617.SAFE",
    "2019111565": "S2A_MSIL1C_20170706T051651_N0205_R062_T48XWG_20170706T051649.SAFE",
    "2065997836": "S2A_MSIL1C_20180102T140051_N0206_R067_T21LXK_20180102T154324.SAFE",
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

def _map_cloud_char(value: int) -> int:
    """Map a CLOUD_CHARACTERISTICS_ID value to CloudSEN12 class.

    Args:
        value: Integer from the CLOUD_CHARACTERISTICS_ID column (0-12).

    Returns:
        CloudSEN12 class (0=Clear, 1=Thick, 2=Thin) or -1 for excluded pixels.
    """
    return CLOUD_CHAR_MAP.get(int(value), -1)


def load_pixbox_labels(pixbox_dir: str) -> Dict[str, pd.DataFrame]:
    """Load PixBox reference labels grouped by Sentinel-2 scene.

    Parses ``pixbox_sentinel2_cmix_20180425.csv`` from the downloaded
    PixBox-S2-CMIX ZIP. The CloudSEN12 4-class label is derived by
    combining two PixBox columns:

    - **CLOUD_CHARACTERISTICS_ID** → Clear (0), Thick Cloud (1),
      Thin Cloud (2), or excluded (-1).
    - **SHADOW_ID == 3** (Cloud shadow) → overrides to Cloud Shadow (3).

    Pixels with label -1 (ambiguous categories like "spatially mixed"
    or "cloud border") are kept in the DataFrame but will be excluded
    during experiment filtering (they have y_true < 0).

    Args:
        pixbox_dir: Directory containing the extracted PixBox CSV file.

    Returns:
        Dictionary mapping scene/product_id -> DataFrame with columns:
            row, col, label (CloudSEN12 integer: 0=Clear, 1=Thick,
            2=Thin, 3=Shadow, -1=excluded).
    """
    root = Path(pixbox_dir)

    # Prefer the canonical filename; fall back to any CSV found.
    canonical = root / PIXBOX_CSV_FILENAME
    if canonical.exists():
        csv_files = [canonical]
    else:
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
    _row_aliases = {"pixel_y", "row", "pixel_row", "line", "y_pixel", "y"}
    _col_aliases = {"pixel_x", "col", "column", "pixel_col", "sample", "x_pixel", "x"}
    _cloud_char_aliases = {
        "cloud_characteristics_id", "cloud_char_id",
    }
    _shadow_aliases = {"shadow_id"}
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
        cloud_char_col = _find_col(cols, _cloud_char_aliases, "CLOUD_CHARACTERISTICS_ID")
        shadow_col = _find_col(cols, _shadow_aliases, "SHADOW_ID")

        # Print unique values for key columns.
        print(f"  {cloud_char_col} unique: {sorted(df_raw[cloud_char_col].unique().tolist())}")
        print(f"  {shadow_col} unique: {sorted(df_raw[shadow_col].unique().tolist())}")

        try:
            scene_col = _find_col(cols, _scene_aliases, "scene_id")
            scene_ids = df_raw[scene_col].astype(str)
        except KeyError:
            scene_ids = pd.Series([csv_path.stem] * len(df_raw))

        # Remap numeric PRODUCT_IDs to SAFE directory names.
        scene_ids = scene_ids.map(
            lambda pid: PRODUCT_ID_TO_SAFE.get(pid, pid)
        )
        n_mapped = scene_ids.isin(PRODUCT_ID_TO_SAFE.values()).sum()
        print(f"  Remapped {n_mapped}/{len(scene_ids)} PRODUCT_IDs to SAFE names.")

        # Derive CloudSEN12 labels from CLOUD_CHARACTERISTICS_ID + SHADOW_ID.
        labels = df_raw[cloud_char_col].apply(_map_cloud_char)

        # Override: SHADOW_ID == 3 (Cloud shadow) → CloudSEN12 class 3,
        # but only for pixels that are Clear (not already cloud).
        is_cloud_shadow = df_raw[shadow_col] == SHADOW_CLOUD_ID
        is_clear = labels == 0
        labels = labels.where(~(is_cloud_shadow & is_clear), other=3)
        # Also set cloud shadow for pixels marked as "none of classes" / excluded
        # that have SHADOW_ID == 3.
        is_excluded = labels == -1
        labels = labels.where(~(is_cloud_shadow & is_excluded), other=3)

        n_excluded = (labels == -1).sum()
        class_counts = labels.value_counts().sort_index()
        print(f"  Label distribution (CloudSEN12): {class_counts.to_dict()}")
        if n_excluded > 0:
            print(f"  -> {n_excluded} pixels excluded (ambiguous categories).")

        df_raw["_scene"] = scene_ids
        df_raw["_row"] = df_raw[row_col].astype(int)
        df_raw["_col"] = df_raw[col_col].astype(int)
        df_raw["_label"] = labels.astype(int)

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