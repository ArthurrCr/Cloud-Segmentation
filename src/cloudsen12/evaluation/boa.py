"""Patch-level Balanced Overall Accuracy (BOA) evaluation."""

from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from cloudsen12.config.constants import CLASS_NAMES, EXPERIMENTS, NUM_CLASSES, SENTINEL_BANDS
from cloudsen12.inference.normalization import get_normalization_stats, normalize_images
from cloudsen12.inference.prediction import get_predictions


CONVENTIONS = ("original", "cloudsen12")

# Positive class of "valid/invalid" under the CloudSEN12 convention. The paper
# counts thin-cloud/shadow confusions as true positives there, so the positive
# class is INVALID (thick cloud, thin cloud, shadow), not valid (clear).
CLOUDSEN12_POSITIVES: Dict[str, List[int]] = {"valid/invalid": [1, 2, 3]}


def safe_divide(numerator: float, denominator: float) -> float:
    """Return NaN when denominator is zero, otherwise numerator/denominator."""
    return np.nan if denominator == 0 else numerator / denominator


def compute_bucket_percentages(
    arr: np.ndarray, inclusive_middle: bool = False
) -> Tuple[float, float, float]:
    """Compute percentage of values in ranges <0.1, 0.1-0.9, >0.9.

    Args:
        arr: Array of values (NaN values are ignored).
        inclusive_middle: If False (original behaviour) the bins are
            [0, 0.1), [0.1, 0.9), [0.9, 1.01], so a value of exactly 0.9 counts
            as high. If True the middle bucket is [0.1, 0.9] and only values
            above 0.9 are high, which is what the CloudSEN12 paper describes
            and what reproduces its Table 6.

    Returns:
        Tuple of (low_pct, mid_pct, high_pct) as percentages, all NaN when
        there is no defined value.
    """
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return (np.nan, np.nan, np.nan)
    if inclusive_middle:
        low = np.count_nonzero(arr < 0.1)
        mid = np.count_nonzero((arr >= 0.1) & (arr <= 0.9))
        high = np.count_nonzero(arr > 0.9)
        return (low / arr.size * 100, mid / arr.size * 100, high / arr.size * 100)
    hist, _ = np.histogram(arr, [0, 0.1, 0.9, 1.01])
    return tuple(hist / arr.size * 100)


def _prepare_models(
    models: Union[torch.nn.Module, List[torch.nn.Module]],
    device: str,
) -> List[torch.nn.Module]:
    """Ensure models is a list and move all to device in eval mode."""
    if not isinstance(models, list):
        models = [models]
    for m in models:
        m.to(device).eval()
    return models


def _compute_binary_stats(
    tp: int, fn: int, fp: int, tn: int, mask_ua: bool = False
) -> Dict[str, float]:
    """Compute PA, UA, BOA, OE, CE from binary confusion counts.

    PA is NaN when the class is absent from the reference (tp + fn == 0) and UA
    is NaN when nothing was predicted as the class (tp + fp == 0). BOA is NaN
    when either the class or its complement is absent from the reference.
    mask_ua forces UA to NaN; without it, UA is 0 whenever the model predicts
    the class in a patch that does not contain it (tp = 0, fp > 0).
    """
    pa = safe_divide(tp, tp + fn)
    ua = safe_divide(tp, tp + fp)

    if mask_ua:
        ua = np.nan

    boa = 0.5 * (safe_divide(tp, tp + fn) + safe_divide(tn, tn + fp))
    oe = np.nan if np.isnan(pa) else (1.0 - pa)
    ce = np.nan if np.isnan(ua) else (1.0 - ua)

    return {"PA": pa, "UA": ua, "BOA": boa, "OE": oe, "CE": ce}


def _build_summary_table(
    exps: Dict[str, Dict], inclusive_middle: bool = False
) -> pd.DataFrame:
    """Build summary DataFrame from accumulated experiment metrics."""
    rows = []
    for name, cfg in exps.items():
        pa = np.asarray(cfg["PA"], dtype=float)
        ua = np.asarray(cfg["UA"], dtype=float)
        pa_low, pa_mid, pa_high = compute_bucket_percentages(pa, inclusive_middle)
        ua_low, ua_mid, ua_high = compute_bucket_percentages(ua, inclusive_middle)

        boa = np.asarray(cfg["BOA"], dtype=float)
        rows.append({
            "Experiment": name,
            "Median BOA": f"{np.nanmedian(cfg['BOA']):.4f}",
            "PA low%": f"{pa_low:.2f}",
            "PA middle%": f"{pa_mid:.2f}",
            "PA high%": f"{pa_high:.2f}",
            "UA low%": f"{ua_low:.2f}",
            "UA middle%": f"{ua_mid:.2f}",
            "UA high%": f"{ua_high:.2f}",
            # Patches with a defined BOA: the ones the median is computed over.
            "N patches": int(np.sum(~np.isnan(boa))),
            # Every patch that was evaluated, defined or not.
            "N total": len(boa),
            # Denominators of the PA and UA percentages.
            "N PA": int(np.sum(~np.isnan(pa))),
            "N UA": int(np.sum(~np.isnan(ua))),
        })

    return pd.DataFrame(rows)


def _build_patch_dataframe(exps: Dict[str, Dict]) -> pd.DataFrame:
    """Build a DataFrame with per-patch metrics for each experiment.

    Returns:
        DataFrame with columns: patch_idx, experiment, BOA, PA, UA.
    """
    rows = []
    for name, cfg in exps.items():
        for i, (boa, pa, ua) in enumerate(
            zip(cfg["BOA"], cfg["PA"], cfg["UA"])
        ):
            rows.append({
                "patch_idx": i,
                "experiment": name,
                "BOA": boa,
                "PA": pa,
                "UA": ua,
            })
    return pd.DataFrame(rows)


def evaluate_test_dataset(
    test_loader: torch.utils.data.DataLoader,
    models: Union[torch.nn.Module, List[torch.nn.Module]],
    device: str = "cuda",
    use_ensemble: bool = True,
    normalize_imgs: bool = True,
    convention: str = "original",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute patch-level BOA using argmax predictions.

    Evaluates each binary experiment defined in EXPERIMENTS by computing
    per-patch PA, UA, BOA, OE, and CE, then summarizes with median BOA
    and PA/UA bucket distributions.

    Args:
        test_loader: DataLoader with test data.
        models: Single model or list of models.
        device: Device for execution.
        use_ensemble: If True, uses ensemble prediction.
        normalize_imgs: If True, normalizes images before inference.
        convention: "original" reproduces the numbers this code first produced:
            UA is NaN only for "cloud/no cloud" in patches with no cloud,
            "valid/invalid" takes clear as the positive class, and a
            value of exactly 0.9 counts as high. "cloudsen12" follows
            Aybar et al. (2022), Table 6: UA is NaN in every experiment for
            patches with no thick or thin cloud in the reference, "valid/invalid"
            takes invalid (cloud or shadow) as the positive class, and the
            middle bucket includes 0.9. Median BOA is the same under both,
            because BOA is symmetric in the two classes.

    Returns:
        Tuple of (summary_df, patch_df):
            summary_df: Median BOA and PA/UA bucket percentages.
            patch_df: Per-patch BOA, PA, UA for each experiment.
    """
    if convention not in CONVENTIONS:
        raise ValueError(f"unknown convention {convention!r}; expected {CONVENTIONS}")
    paper = convention == "cloudsen12"

    mean, std = get_normalization_stats(device, False, SENTINEL_BANDS)
    models = _prepare_models(models, device)

    exps: Dict[str, Dict] = {
        k: dict(v, PA=[], UA=[], BOA=[], OE=[], CE=[])
        for k, v in EXPERIMENTS.items()
    }

    with torch.no_grad():
        for imgs, gts in tqdm(test_loader, desc="Processing patches"):
            imgs = imgs.to(device).float()
            gts = gts.to(device)

            if normalize_imgs:
                imgs = normalize_images(imgs, mean, std)

            preds = get_predictions(models, imgs, use_ensemble=use_ensemble)

            for gt, pr in zip(gts.cpu().numpy(), preds.cpu().numpy()):
                cm = confusion_matrix(gt.ravel(), pr.ravel(), labels=[0, 1, 2, 3])
                no_cloud = cm[[1, 2], :].sum() == 0

                for name, cfg in exps.items():
                    pos = CLOUDSEN12_POSITIVES.get(name, cfg["pos"]) if paper else cfg["pos"]
                    neg = [c for c in range(4) if c not in pos]

                    tp = cm[np.ix_(pos, pos)].sum()
                    fn = cm[np.ix_(pos, neg)].sum()
                    fp = cm[np.ix_(neg, pos)].sum()
                    tn = cm[np.ix_(neg, neg)].sum()

                    mask_ua = no_cloud and (paper or name == "cloud/no cloud")
                    stats = _compute_binary_stats(tp, fn, fp, tn, mask_ua=mask_ua)
                    for key in ("PA", "UA", "BOA", "OE", "CE"):
                        cfg[key].append(stats[key])

    summary_df = _build_summary_table(exps, inclusive_middle=paper)
    patch_df = _build_patch_dataframe(exps)
    return summary_df, patch_df


# ------------------------------------------------------------------
# Ground-truth statistics for stratified error analysis
# ------------------------------------------------------------------


def compute_patch_gt_stats(
    test_loader: torch.utils.data.DataLoader,
) -> pd.DataFrame:
    """Compute per-patch class distribution from ground-truth labels.

    Iterates through the test loader once (no model needed) and returns
    the fraction of pixels belonging to each class for every patch.

    Args:
        test_loader: DataLoader yielding (images, labels) batches.

    Returns:
        DataFrame with columns:
            patch_idx, frac_clear, frac_thick_cloud, frac_thin_cloud,
            frac_shadow, cloud_cover, has_shadow, has_thin_cloud.
    """
    rows: List[Dict] = []
    patch_idx = 0

    for _, gts in tqdm(test_loader, desc="Computing GT stats"):
        for gt in gts.numpy():
            n_pixels = float(gt.size)
            counts = np.bincount(gt.ravel(), minlength=NUM_CLASSES)
            fracs = counts / n_pixels

            rows.append({
                "patch_idx": patch_idx,
                "frac_clear": fracs[0],
                "frac_thick_cloud": fracs[1],
                "frac_thin_cloud": fracs[2],
                "frac_shadow": fracs[3],
                # Combined cloud cover = thick + thin
                "cloud_cover": fracs[1] + fracs[2],
                "has_shadow": fracs[3] > 0.0,
                "has_thin_cloud": fracs[2] > 0.0,
            })
            patch_idx += 1

    return pd.DataFrame(rows)
