"""Plotting functions for confusion matrices, training curves, and error analysis."""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

from cloudsen12.config.constants import CLASS_NAMES


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _clean_spines(ax: plt.Axes) -> None:
    """Remove top, right, and left spines. Keep only bottom."""
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(left=False)


def _add_table_below(
    fig: plt.Figure,
    ax_table: plt.Axes,
    col_labels: List[str],
    row_labels: List[str],
    cell_text: List[List[str]],
    row_colors: Optional[List[str]] = None,
) -> None:
    """Render a matplotlib table in the given axes."""
    ax_table.axis("off")
    table = ax_table.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)

    # Style header row.
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor("#E8E8E8")
        table[(0, j)].set_text_props(weight="bold")

    # Style row labels with model colors.
    if row_colors:
        for i, color in enumerate(row_colors):
            table[(i + 1, -1)].set_facecolor(color)
            table[(i + 1, -1)].set_alpha(0.3)


def _make_fig_with_table(
    figsize: Tuple[int, int], table_height_ratio: float = 0.3,
) -> Tuple[plt.Figure, plt.Axes, plt.Axes]:
    """Create a figure with a chart axes on top and a table axes below."""
    fig = plt.figure(figsize=(figsize[0], figsize[1] * (1 + table_height_ratio)))
    gs = gridspec.GridSpec(
        2, 1, height_ratios=[1, table_height_ratio], hspace=0.05,
    )
    ax_chart = fig.add_subplot(gs[0])
    ax_table = fig.add_subplot(gs[1])
    return fig, ax_chart, ax_table


# ------------------------------------------------------------------
# Core plots
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
    """Plot a confusion matrix (absolute or row-normalized)."""
    if normalize:
        with np.errstate(invalid="ignore", divide="ignore"):
            cm_show = (
                conf_matrix.astype(float) / conf_matrix.sum(axis=1, keepdims=True)
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
        annot=True, fmt=fmt, cmap=cmap,
        xticklabels=class_names, yticklabels=class_names,
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

    plt.tight_layout()
    return ax


def plot_training_history(
    history: dict,
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[str] = None,
) -> None:
    """Plot training history: loss, accuracy, and IoU over epochs."""
    _, axes = plt.subplots(1, 3, figsize=figsize)
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
        _clean_spines(ax)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# ------------------------------------------------------------------
# Error analysis plots
# ------------------------------------------------------------------

# Model display names and colors for consistent styling across figures.
MODEL_STYLES: Dict[str, Dict] = {
    "Unet + regnetz d8": {"label": "Ours (UNet+RegNetZ-D8)", "color": "#2196F3", "marker": "o"},
    "CloudS2Mask ensemble": {"label": "CloudS2Mask (ensemble)", "color": "#FF9800", "marker": "s"},
    "CloudS2Mask Dice_1 (single)": {"label": "CloudS2Mask Dice\u2081", "color": "#FFC107", "marker": "^"},
    "CloudS2Mask Dice_2 (single)": {"label": "CloudS2Mask Dice\u2082", "color": "#FF5722", "marker": "v"},
    "Unet + MobilenetV2": {"label": "UNetMobV2 (baseline)", "color": "#9E9E9E", "marker": "D"},
}


def _get_style(model_name: str) -> Dict:
    """Return style dict for a model, with fallback defaults."""
    if model_name in MODEL_STYLES:
        return MODEL_STYLES[model_name]
    colors = ["#4CAF50", "#9C27B0", "#00BCD4", "#795548", "#607D8B"]
    idx = hash(model_name) % len(colors)
    return {"label": model_name, "color": colors[idx], "marker": "o"}


def plot_stratified_boa(
    stratified_df: pd.DataFrame,
    experiment: str = "cloud/no cloud",
    stratify_label: str = "Cloud Coverage",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> plt.Axes:
    """Plot median BOA per coverage bin for each model, with table below."""
    fig, ax, ax_tab = _make_fig_with_table(figsize)

    models = stratified_df["model"].unique()
    bins = stratified_df["bin"].unique()
    n_models = len(models)
    n_bins = len(bins)
    bar_width = 0.8 / n_models
    x = np.arange(n_bins)

    table_data: Dict[str, List[str]] = {}

    for i, model in enumerate(models):
        style = _get_style(model)
        df_m = stratified_df[stratified_df["model"] == model]

        medians, yerr_low, yerr_high = [], [], []
        row_vals: List[str] = []
        for b in bins:
            row = df_m[df_m["bin"] == b]
            if len(row) > 0:
                med = row.iloc[0]["median_BOA"]
                q25 = row.iloc[0]["q25_BOA"]
                q75 = row.iloc[0]["q75_BOA"]
                medians.append(med)
                yerr_low.append(med - q25)
                yerr_high.append(q75 - med)
                row_vals.append(f"{med:.4f}")
            else:
                medians.append(np.nan)
                yerr_low.append(0)
                yerr_high.append(0)
                row_vals.append("\u2014")

        table_data[model] = row_vals

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
    ax.set_title(f"BOA by {stratify_label} \u2014 {experiment}", fontsize=13)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0.5)
    _clean_spines(ax)

    # Patch counts.
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

    # Table.
    col_labels = [str(b) for b in bins]
    row_labels = [_get_style(m)["label"] for m in models]
    cell_text = [table_data[m] for m in models]
    row_colors = [_get_style(m)["color"] for m in models]
    _add_table_below(fig, ax_tab, col_labels, row_labels, cell_text, row_colors)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    return ax


def plot_boa_distribution(
    patch_data: Dict[str, pd.DataFrame],
    experiment: str = "cloud/no cloud",
    model_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> plt.Axes:
    """Plot BOA distribution (violin + box) for each model, with table below."""
    if model_names is None:
        model_names = list(patch_data.keys())

    records: List[Dict] = []
    stats: Dict[str, Dict[str, str]] = {}
    for model_name in model_names:
        df = patch_data[model_name]
        df_exp = df[df["experiment"] == experiment]
        style = _get_style(model_name)
        boas = df_exp["BOA"].dropna().values
        stats[model_name] = {
            "Median": f"{np.nanmedian(boas):.4f}" if len(boas) > 0 else "\u2014",
            "Q25": f"{np.nanpercentile(boas, 25):.4f}" if len(boas) > 0 else "\u2014",
            "Q75": f"{np.nanpercentile(boas, 75):.4f}" if len(boas) > 0 else "\u2014",
            "Mean": f"{np.nanmean(boas):.4f}" if len(boas) > 0 else "\u2014",
            "N": str(len(boas)),
        }
        for _, row in df_exp.iterrows():
            if not np.isnan(row["BOA"]):
                records.append({
                    "Model": style["label"],
                    "BOA": row["BOA"],
                    "_order": model_names.index(model_name),
                })

    plot_df = pd.DataFrame(records).sort_values("_order")

    fig, ax, ax_tab = _make_fig_with_table(figsize)

    labels_order = [_get_style(m)["label"] for m in model_names]
    palette = {_get_style(m)["label"]: _get_style(m)["color"] for m in model_names}

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
    ax.set_title(f"BOA Distribution \u2014 {experiment}", fontsize=13)
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.3)
    _clean_spines(ax)

    # Table.
    stat_keys = ["Median", "Q25", "Q75", "Mean", "N"]
    col_labels = stat_keys
    row_labels = [_get_style(m)["label"] for m in model_names]
    cell_text = [[stats[m][k] for k in stat_keys] for m in model_names]
    row_colors = [_get_style(m)["color"] for m in model_names]
    _add_table_below(fig, ax_tab, col_labels, row_labels, cell_text, row_colors)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    return ax


def plot_confusion_matrices_comparison(
    results: Dict,
    model_names: List[str],
    class_names: List[str] = CLASS_NAMES,
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None,
) -> None:
    """Plot normalized confusion matrices side by side for multiple models."""
    n = len(model_names)
    if figsize is None:
        figsize = (7 * n, 6)

    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for ax, model_name in zip(axes, model_names):
        cm = results[model_name].confusion_matrix
        style = _get_style(model_name)
        plot_confusion_matrix(
            cm, normalize=True, class_names=class_names,
            title=style["label"], ax=ax,
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_stratified_shadow(
    stratified_df: pd.DataFrame,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> plt.Axes:
    """Plot shadow BOA stratified by shadow coverage fraction."""
    return plot_stratified_boa(
        stratified_df,
        experiment="cloud shadow",
        stratify_label="Shadow Coverage Fraction",
        figsize=figsize,
        save_path=save_path,
    )