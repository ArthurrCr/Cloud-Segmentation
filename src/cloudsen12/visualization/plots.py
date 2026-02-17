"""Plotting functions for confusion matrices, training curves, error analysis,
metric comparisons, and efficiency visualizations.

All functions accept data as arguments (no class dependency) and follow a
consistent pastel palette with canonical model ordering defined in
``cloudsen12.visualization.style``.
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from cloudsen12.config.constants import CLASS_NAMES
from cloudsen12.visualization.style import (
    clean_spines,
    get_style,
    save_and_show,
    sort_models,
)


# ------------------------------------------------------------------
# Confusion matrix
# ------------------------------------------------------------------


def plot_confusion_matrix(
    conf_matrix: np.ndarray,
    normalize: bool = True,
    class_names: List[str] = CLASS_NAMES,
    title: Optional[str] = None,
    cmap: str = "Blues",
    figsize: Tuple[int, int] = (8, 6),
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot a single confusion matrix (absolute or row-normalized)."""
    if normalize:
        with np.errstate(invalid="ignore", divide="ignore"):
            cm_show = (
                conf_matrix.astype(float)
                / conf_matrix.sum(axis=1, keepdims=True)
            )
        cm_show = np.nan_to_num(cm_show) * 100
        fmt = ".1f"
        cbar_label = "Percentage (%)"
    else:
        cm_show = conf_matrix
        fmt = "d"
        cbar_label = "Count"

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        cm_show,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": cbar_label},
        ax=ax,
    )
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")

    if title is None:
        title = "Confusion Matrix"
        if normalize:
            title += " (normalized)"
    ax.set_title(title)
    return ax


def plot_confusion_matrices_comparison(
    confusion_matrices: Dict[str, np.ndarray],
    model_names: Optional[List[str]] = None,
    class_names: List[str] = CLASS_NAMES,
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None,
) -> None:
    """Plot normalized confusion matrices side by side for multiple models.

    Args:
        confusion_matrices: Mapping of model name to its confusion matrix.
        model_names: Subset/order to display. Defaults to canonical order.
    """
    if model_names is None:
        model_names = sort_models(list(confusion_matrices.keys()))

    n = len(model_names)
    if figsize is None:
        figsize = (7 * n, 6)

    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, model_names):
        style = get_style(name)
        plot_confusion_matrix(
            confusion_matrices[name],
            normalize=True,
            class_names=class_names,
            title=style["label"],
            ax=ax,
        )

    save_and_show(fig, save_path)


# ------------------------------------------------------------------
# Training curves
# ------------------------------------------------------------------


def plot_training_history(
    history: dict,
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[str] = None,
) -> None:
    """Plot training history: loss, accuracy, and IoU over epochs."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    epochs = range(1, len(history["train_loss"]) + 1)

    panels = [
        ("train_loss", "val_loss", "Loss", "Training and Validation Loss"),
        ("train_acc", "val_acc", "Accuracy", "Training and Validation Accuracy"),
        ("train_iou", "val_iou", "IoU", "Training and Validation IoU"),
    ]

    for ax, (train_key, val_key, ylabel, title) in zip(axes, panels):
        ax.plot(epochs, history[train_key], "b-", label="Train")
        ax.plot(epochs, history[val_key], "r-", label="Validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        clean_spines(ax)

    save_and_show(fig, save_path)


# ------------------------------------------------------------------
# BOA distribution and stratified analysis
# ------------------------------------------------------------------


def plot_boa_distribution(
    patch_data: Dict[str, pd.DataFrame],
    experiment: str = "cloud/no cloud",
    model_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """Plot BOA distribution (violin + box) for each model.

    Returns:
        DataFrame with Median, Q25, Q75, Mean, N per model.
    """
    if model_names is None:
        model_names = sort_models(list(patch_data.keys()))

    records: List[Dict] = []
    stats_rows: List[Dict] = []

    for order_idx, name in enumerate(model_names):
        df_exp = patch_data[name][patch_data[name]["experiment"] == experiment]
        style = get_style(name)
        boas = df_exp["BOA"].dropna().values

        stats_rows.append({
            "Model": style["label"],
            "Median": round(np.nanmedian(boas), 4) if len(boas) else np.nan,
            "Q25": round(np.nanpercentile(boas, 25), 4) if len(boas) else np.nan,
            "Q75": round(np.nanpercentile(boas, 75), 4) if len(boas) else np.nan,
            "Mean": round(np.nanmean(boas), 4) if len(boas) else np.nan,
            "N": len(boas),
        })

        for _, row in df_exp.iterrows():
            if not np.isnan(row["BOA"]):
                records.append({
                    "Model": style["label"],
                    "BOA": row["BOA"],
                    "_order": order_idx,
                })

    plot_df = pd.DataFrame(records).sort_values("_order")
    labels_order = [get_style(m)["label"] for m in model_names]
    palette = {get_style(m)["label"]: get_style(m)["color"] for m in model_names}

    fig, ax = plt.subplots(figsize=figsize)

    sns.violinplot(
        data=plot_df, x="Model", y="BOA", hue="Model",
        order=labels_order, hue_order=labels_order,
        palette=palette, inner=None, alpha=0.3, ax=ax, cut=0,
        legend=False,
    )
    sns.boxplot(
        data=plot_df, x="Model", y="BOA", hue="Model",
        order=labels_order, hue_order=labels_order,
        palette=palette, width=0.15, fliersize=1, ax=ax,
        boxprops=dict(alpha=0.8), legend=False,
    )

    ax.set_ylabel("BOA", fontsize=12)
    ax.set_xlabel("")
    ax.set_title(f"BOA Distribution -- {experiment}", fontsize=13)
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.3)
    clean_spines(ax)

    save_and_show(fig, save_path)
    return pd.DataFrame(stats_rows)


def plot_stratified_boa(
    stratified_df: pd.DataFrame,
    experiment: str = "cloud/no cloud",
    stratify_label: str = "Cloud Coverage",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """Plot median BOA per coverage bin for each model.

    Returns:
        Pivot DataFrame with models as rows and bins as columns.
    """
    models = sort_models(list(stratified_df["model"].unique()))
    bins = stratified_df["bin"].unique()
    n_models = len(models)
    n_bins = len(bins)
    bar_width = 0.8 / n_models
    x = np.arange(n_bins)

    fig, ax = plt.subplots(figsize=figsize)

    for i, model in enumerate(models):
        style = get_style(model)
        df_m = stratified_df[stratified_df["model"] == model]

        medians, yerr_low, yerr_high = [], [], []
        for b in bins:
            row = df_m[df_m["bin"] == b]
            if len(row) > 0:
                med = row.iloc[0]["median_BOA"]
                medians.append(med)
                yerr_low.append(med - row.iloc[0]["q25_BOA"])
                yerr_high.append(row.iloc[0]["q75_BOA"] - med)
            else:
                medians.append(np.nan)
                yerr_low.append(0)
                yerr_high.append(0)

        offset = (i - n_models / 2 + 0.5) * bar_width
        ax.bar(
            x + offset, medians, bar_width,
            yerr=[yerr_low, yerr_high], capsize=3,
            label=style["label"], color=style["color"],
            alpha=0.85, edgecolor="white", linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(bins, fontsize=10)
    ax.set_xlabel(stratify_label, fontsize=12)
    ax.set_ylabel("Median BOA", fontsize=12)
    ax.set_title(f"BOA by {stratify_label} -- {experiment}", fontsize=13)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0.5)
    clean_spines(ax)

    for b_idx, b in enumerate(bins):
        row = stratified_df[
            (stratified_df["bin"] == b)
            & (stratified_df["model"] == models[0])
        ]
        if len(row) > 0:
            n = row.iloc[0]["n_patches"]
            ax.text(
                b_idx, ax.get_ylim()[0] + 0.01, f"n={n}",
                ha="center", va="bottom", fontsize=8, color="gray",
            )

    save_and_show(fig, save_path)

    rows = []
    for model in models:
        style = get_style(model)
        row_dict = {"Model": style["label"]}
        df_m = stratified_df[stratified_df["model"] == model]
        for b in bins:
            r = df_m[df_m["bin"] == b]
            row_dict[str(b)] = (
                round(r.iloc[0]["median_BOA"], 4) if len(r) > 0 else np.nan
            )
        rows.append(row_dict)
    return pd.DataFrame(rows)


def plot_stratified_shadow(
    stratified_df: pd.DataFrame,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """Plot shadow BOA stratified by shadow coverage fraction."""
    return plot_stratified_boa(
        stratified_df,
        experiment="cloud shadow",
        stratify_label="Shadow Coverage Fraction",
        figsize=figsize,
        save_path=save_path,
    )


# ------------------------------------------------------------------
# Metric comparison
# ------------------------------------------------------------------


def plot_metric_comparison(
    metrics_by_model: Dict[str, Dict[str, Dict[str, float]]],
    metric: str,
    model_names: Optional[List[str]] = None,
    class_names: List[str] = CLASS_NAMES,
    figsize: Tuple[int, int] = (20, 12),
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """Bar chart comparing a single metric across models and classes.

    Args:
        metrics_by_model: ``{model_name: {class_name: {metric: value}}}``.
        metric: Name of the metric to compare (e.g. ``"F1-Score"``).
        model_names: Subset/order of models. Defaults to canonical order.

    Returns:
        DataFrame with models as rows and classes as columns.
    """
    if model_names is None:
        model_names = sort_models(list(metrics_by_model.keys()))

    values_mat = np.array([
        [metrics_by_model[m][c][metric] for c in class_names]
        for m in model_names
    ])

    if metric in ("Omission Error", "Commission Error"):
        best_idx = values_mat.argmin(axis=0)
    else:
        best_idx = values_mat.argmax(axis=0)

    n_models = len(model_names)
    width = 0.8 / n_models
    x = np.arange(len(class_names))

    fig, ax = plt.subplots(figsize=figsize)

    for i, model in enumerate(model_names):
        style = get_style(model)
        vals = values_mat[i]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, vals, width, alpha=0.85,
            color=style["color"], label=style["label"],
        )
        for j, bar in enumerate(bars):
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.005,
                f"{h:.3f}",
                ha="center", va="bottom", fontsize=9,
            )
            if i == best_idx[j]:
                bar.set_edgecolor("k")
                bar.set_linewidth(2)
                bar.set_linestyle("--")

    legend_patches = [
        Patch(
            color=get_style(m)["color"],
            label=get_style(m)["label"],
            alpha=0.85,
        )
        for m in model_names
    ]

    ax.set_xlabel("Classes", fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f"{metric} Comparison Across Models", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=10)
    ax.legend(
        handles=legend_patches, title="Models",
        bbox_to_anchor=(1.02, 1), loc="upper left",
    )
    ax.grid(axis="y", alpha=0.3)
    clean_spines(ax)

    save_and_show(fig, save_path)

    rows = []
    for i, m in enumerate(model_names):
        row = {"Model": get_style(m)["label"]}
        for j, c in enumerate(class_names):
            row[c] = round(values_mat[i, j], 4)
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Error analysis
# ------------------------------------------------------------------


def plot_errors_for_class(
    metrics_by_model: Dict[str, Dict[str, Dict[str, float]]],
    class_name: str,
    model_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """Plot Omission and Commission Error for a single class.

    Returns:
        DataFrame with OE and CE per model.
    """
    if model_names is None:
        model_names = sort_models(list(metrics_by_model.keys()))

    oe_vals, ce_vals, labels, colors = [], [], [], []
    for m in model_names:
        cls_metrics = metrics_by_model[m].get(class_name, {})
        oe_vals.append(cls_metrics.get("Omission Error", 0.0))
        ce_vals.append(cls_metrics.get("Commission Error", 0.0))
        style = get_style(m)
        labels.append(style["label"])
        colors.append(style["color"])

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)
    bars_oe = ax.bar(
        x - width / 2, oe_vals, width, label="Omission Error",
        color=colors, alpha=0.85, edgecolor="white",
    )
    bars_ce = ax.bar(
        x + width / 2, ce_vals, width, label="Commission Error",
        color=colors, alpha=0.55, edgecolor="white", hatch="//",
    )

    for bars in (bars_oe, bars_ce):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 0.003,
                f"{h:.3f}", ha="center", va="bottom", fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, rotation=15, ha="right")
    ax.set_ylabel("Error Rate", fontsize=12)
    ax.set_title(f"Omission & Commission Error -- {class_name}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    clean_spines(ax)

    save_and_show(fig, save_path)

    return pd.DataFrame({
        "Model": labels,
        "Omission Error": [round(v, 4) for v in oe_vals],
        "Commission Error": [round(v, 4) for v in ce_vals],
    })


# ------------------------------------------------------------------
# Threshold curve
# ------------------------------------------------------------------


def plot_threshold_curve(
    thresholds: List[float],
    median_boas: List[float],
    best_threshold: float,
    best_median_boa: float,
    model_name: str,
    experiment: str,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> None:
    """Plot median BOA vs threshold for an experiment."""
    style = get_style(model_name)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(thresholds, median_boas, linewidth=2, color=style["color"])
    ax.scatter(
        best_threshold, best_median_boa,
        s=100, zorder=5, color=style["color"],
        label=f"t* = {best_threshold:.2f}",
    )
    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel("Median BOA", fontsize=12)
    ax.set_title(
        f"Threshold Optimization - {experiment} - {style['label']}",
        fontsize=14,
    )
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=12)
    clean_spines(ax)

    save_and_show(fig, save_path)


# ------------------------------------------------------------------
# Parameter counts
# ------------------------------------------------------------------


def plot_model_parameter_counts(
    param_data: Dict[str, Dict[str, float]],
    model_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """Horizontal bar chart of total parameter counts.

    Args:
        param_data: ``{model_name: {"total": <M>, "trainable": <M>}}``.
            Values in millions.

    Returns:
        DataFrame with total and trainable params per model.
    """
    if model_names is None:
        model_names = sort_models(list(param_data.keys()))

    labels, totals, trainable, colors = [], [], [], []
    for m in model_names:
        style = get_style(m)
        labels.append(style["label"])
        totals.append(param_data[m]["total"])
        trainable.append(param_data[m].get("trainable", 0))
        colors.append(style["color"])

    y = np.arange(len(model_names))
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(y, totals, color=colors, alpha=0.85, edgecolor="white")

    for bar, val in zip(bars, totals):
        ax.text(
            bar.get_width() + max(totals) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}M", va="center", fontsize=10,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Parameters (millions)", fontsize=12)
    ax.set_title("Model Parameter Counts", fontsize=13)
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()
    clean_spines(ax)
    ax.spines["bottom"].set_visible(True)

    save_and_show(fig, save_path)

    return pd.DataFrame({
        "Model": labels,
        "Total Params (M)": [round(t, 2) for t in totals],
        "Trainable Params (M)": [round(t, 2) for t in trainable],
    })


# ------------------------------------------------------------------
# Inference cost
# ------------------------------------------------------------------


def plot_inference_cost(
    inference_data: Dict[str, Dict[str, float]],
    model_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """Horizontal bar chart of inference time per image.

    Args:
        inference_data: ``{model_name: {"ms_per_image": <float>,
            "std_ms": <float>}}``.

    Returns:
        DataFrame with ms/image and std per model.
    """
    if model_names is None:
        model_names = sort_models(list(inference_data.keys()))

    labels, times_ms, stds_ms, colors = [], [], [], []
    for m in model_names:
        style = get_style(m)
        labels.append(style["label"])
        times_ms.append(inference_data[m]["ms_per_image"])
        stds_ms.append(inference_data[m].get("std_ms", 0.0))
        colors.append(style["color"])

    y = np.arange(len(model_names))
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(
        y, times_ms, xerr=stds_ms, color=colors,
        alpha=0.85, edgecolor="white", capsize=3,
    )

    for bar, val in zip(bars, times_ms):
        ax.text(
            bar.get_width() + max(times_ms) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f} ms", va="center", fontsize=10,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Inference Time per Image (ms)", fontsize=12)
    ax.set_title("Inference Cost Comparison", fontsize=13)
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()
    clean_spines(ax)
    ax.spines["bottom"].set_visible(True)

    save_and_show(fig, save_path)

    return pd.DataFrame({
        "Model": labels,
        "ms/image": [round(t, 1) for t in times_ms],
        "std (ms)": [round(s, 1) for s in stds_ms],
    })


# ------------------------------------------------------------------
# Efficiency bubble chart
# ------------------------------------------------------------------


def plot_efficiency_bubble(
    bubble_data: Dict[str, Dict[str, float]],
    model_names: Optional[List[str]] = None,
    y_label: str = "Mean IoU",
    figsize: Tuple[int, int] = (10, 7),
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """Bubble chart: parameters vs performance, sized by inference cost.

    Args:
        bubble_data: ``{model_name: {"params_m": <float>,
            "y_value": <float>, "inference_ms": <float|None>}}``.

    Returns:
        DataFrame with params, metric value, and inference time per model.
    """
    if model_names is None:
        model_names = sort_models(list(bubble_data.keys()))

    fig, ax = plt.subplots(figsize=figsize)
    table_rows: List[Dict] = []

    for m in model_names:
        d = bubble_data[m]
        style = get_style(m)

        x_val = d["params_m"]
        y_val = d["y_value"]
        inf_ms = d.get("inference_ms")
        bubble_size = max(inf_ms * 10, 80) if inf_ms is not None else 200

        ax.scatter(
            x_val, y_val, s=bubble_size,
            color=style["color"], alpha=0.75,
            edgecolors="white", linewidth=1.5, zorder=5,
        )
        ax.annotate(
            style["label"], (x_val, y_val),
            textcoords="offset points", xytext=(8, 8),
            fontsize=9, color=style["color"],
        )

        row = {
            "Model": style["label"],
            "Params (M)": round(x_val, 2),
            y_label: round(y_val, 4),
        }
        if inf_ms is not None:
            row["Inference (ms)"] = round(inf_ms, 1)
        table_rows.append(row)

    ax.set_xlabel("Parameters (M)", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    title = f"Efficiency: Parameters vs {y_label}"
    if any(bubble_data[m].get("inference_ms") for m in model_names):
        title += " (bubble = inference time)"
    ax.set_title(title, fontsize=13)
    ax.grid(alpha=0.3)
    clean_spines(ax)

    save_and_show(fig, save_path)
    return pd.DataFrame(table_rows)


# ------------------------------------------------------------------
# Peak memory usage
# ------------------------------------------------------------------


def plot_peak_memory(
    memory_data: Dict[str, Dict[str, float]],
    model_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """Horizontal bar chart of peak GPU memory with error bars.

    Args:
        memory_data: ``{model_name: {"mean_mb": <float>,
            "std_mb": <float>}}``.  Values obtained from
            ``ResultsManager.get_memory_data()``.

    Returns:
        DataFrame with mean and std peak memory per model.
    """
    if model_names is None:
        model_names = sort_models(list(memory_data.keys()))

    labels, means, stds, colors = [], [], [], []
    for m in model_names:
        style = get_style(m)
        labels.append(style["label"])
        means.append(memory_data[m]["mean_mb"])
        stds.append(memory_data[m].get("std_mb", 0.0))
        colors.append(style["color"])

    y = np.arange(len(model_names))
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(
        y, means, xerr=stds, color=colors,
        alpha=0.85, edgecolor="white", capsize=4,
    )

    for bar, mean_val, std_val in zip(bars, means, stds):
        ax.text(
            bar.get_width() + max(means) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{mean_val:.1f} +/- {std_val:.1f} MB",
            va="center", fontsize=10,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Peak Memory (MB)", fontsize=12)
    ax.set_title("Peak GPU Memory Usage", fontsize=13)
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()
    clean_spines(ax)
    ax.spines["bottom"].set_visible(True)

    save_and_show(fig, save_path)

    return pd.DataFrame({
        "Model": labels,
        "Peak Memory (MB)": [round(m, 1) for m in means],
        "std (MB)": [round(s, 1) for s in stds],
    })


# ------------------------------------------------------------------
# Qualitative examples
# ------------------------------------------------------------------

_QUALITATIVE_CMAP = ListedColormap(["#2ecc71", "#e74c3c", "#f39c12", "#3498db"])


def plot_qualitative_examples(
    rgb_images: List[np.ndarray],
    ground_truths: List[np.ndarray],
    predictions: Dict[str, List[np.ndarray]],
    indices: Optional[List[int]] = None,
    class_names: List[str] = CLASS_NAMES,
    figsize_per_col: float = 3.0,
    save_path: Optional[str] = None,
) -> None:
    """Show RGB, GT, and predictions side by side for sample patches.

    Args:
        rgb_images: List of H x W x 3 arrays (0-1 range).
        ground_truths: List of H x W integer label arrays.
        predictions: ``{model_name: [H x W array, ...]}``.
        indices: Patch indices to display. Defaults to evenly spaced.
    """
    model_names = sort_models(list(predictions.keys()))
    n_total = len(rgb_images)

    if indices is None:
        n_samples = min(4, n_total)
        indices = np.linspace(0, n_total - 1, n_samples, dtype=int).tolist()

    n_cols = 2 + len(model_names)
    n_rows = len(indices)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_col * n_cols, figsize_per_col * n_rows),
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(indices):
        axes[row, 0].imshow(rgb_images[idx])
        axes[row, 0].set_title("RGB" if row == 0 else "", fontsize=10)
        axes[row, 0].set_ylabel(f"Patch {idx}", fontsize=9)
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])

        axes[row, 1].imshow(
            ground_truths[idx], cmap=_QUALITATIVE_CMAP, vmin=0, vmax=3,
        )
        axes[row, 1].set_title("Ground Truth" if row == 0 else "", fontsize=10)
        axes[row, 1].set_xticks([])
        axes[row, 1].set_yticks([])

        for col, m_name in enumerate(model_names):
            style = get_style(m_name)
            axes[row, col + 2].imshow(
                predictions[m_name][idx],
                cmap=_QUALITATIVE_CMAP, vmin=0, vmax=3,
            )
            axes[row, col + 2].set_title(
                style["label"] if row == 0 else "", fontsize=10,
            )
            axes[row, col + 2].set_xticks([])
            axes[row, col + 2].set_yticks([])

    legend_patches = [
        Patch(color=_QUALITATIVE_CMAP(i), label=c)
        for i, c in enumerate(class_names)
    ]
    fig.legend(
        handles=legend_patches, loc="lower center",
        ncol=len(class_names), fontsize=9, frameon=True,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_qualitative_from_loader(
    models_dict: Dict[str, tuple],
    test_loader: "torch.utils.data.DataLoader",
    n_samples: int = 4,
    indices: Optional[List[int]] = None,
    dataset_scale: float = 1.0,
    class_names: List[str] = CLASS_NAMES,
    figsize_per_col: float = 3.0,
    save_path: Optional[str] = None,
) -> None:
    """Convenience wrapper: run inference and plot qualitative examples.

    Args:
        models_dict: ``{model_name: (model_list, use_ensemble, normalize)}``.
        test_loader: DataLoader yielding ``(images, ground_truths)`` batches.
        n_samples: Number of sample patches to display.
        indices: Specific patch indices. Overrides *n_samples* if given.
        dataset_scale: Factor by which the dataset divided the raw DN
            values. When a model requires CloudS2Mask normalization
            (``norm=True``), the loader values are multiplied by this
            factor to restore raw DN before applying
            ``normalize_images`` (which divides by 32767). Set to 1.0
            if the loader already returns raw DN values.
    """
    import torch

    from cloudsen12.config.constants import SENTINEL_BANDS
    from cloudsen12.inference.normalization import (
        get_normalization_stats,
        normalize_images,
    )
    from cloudsen12.inference.prediction import get_predictions

    device = next(
        (
            str(next(ms[0][0].parameters()).device)
            for ms in models_dict.values()
        ),
        "cpu",
    )
    mean, std = get_normalization_stats(device, False, SENTINEL_BANDS)

    # Resolve indices without loading the full dataset.
    n_total = len(test_loader.dataset)
    if indices is None:
        indices = np.linspace(0, n_total - 1, n_samples, dtype=int).tolist()

    target_set = set(indices)

    # Collect only the needed samples in a single pass.
    collected: Dict[int, torch.Tensor] = {}
    collected_gts: Dict[int, torch.Tensor] = {}
    global_idx = 0

    for imgs, gts in test_loader:
        batch_size = imgs.size(0)
        for i in range(batch_size):
            if global_idx in target_set:
                collected[global_idx] = imgs[i]
                collected_gts[global_idx] = gts[i]
            global_idx += 1
            if len(collected) == len(target_set):
                break
        if len(collected) == len(target_set):
            break

    rgb_images = []
    ground_truths = []
    for idx in indices:
        img_t = collected[idx]
        rgb = img_t[[3, 2, 1]].numpy().transpose(1, 2, 0)
        rgb = np.clip(rgb / np.percentile(rgb, 98), 0, 1)
        rgb_images.append(rgb)
        gt = collected_gts[idx].numpy()
        if gt.ndim == 3:
            gt = gt.squeeze(0)
        ground_truths.append(gt)

    model_names = sort_models(list(models_dict.keys()))
    predictions: Dict[str, List[np.ndarray]] = {m: [] for m in model_names}

    for m_name in model_names:
        model_list, use_ens, norm = models_dict[m_name]
        for idx in indices:
            inp = collected[idx].unsqueeze(0).to(device).float()
            if norm:
                inp = inp * dataset_scale
                inp = normalize_images(inp, mean, std)
            with torch.no_grad():
                pred = get_predictions(
                    model_list, inp, use_ensemble=use_ens,
                )[0].cpu().numpy()
            if pred.ndim == 3:
                pred = pred.squeeze(0)
            del inp
            predictions[m_name].append(pred)

    del collected, collected_gts
    if "cuda" in device:
        torch.cuda.empty_cache()

    plot_qualitative_examples(
        rgb_images=rgb_images,
        ground_truths=ground_truths,
        predictions=predictions,
        indices=list(range(len(indices))),
        class_names=class_names,
        figsize_per_col=figsize_per_col,
        save_path=save_path,
    )