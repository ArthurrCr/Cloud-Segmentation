"""Results management and comparative visualization for model evaluation."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from cloudsen12.config.constants import CLASS_NAMES, METRIC_NAMES
from cloudsen12.visualization.plots import _get_style


@dataclass
class ModelResult:
    """Stores evaluation results for a single model.

    Attributes:
        metrics: Per-class metrics dictionary.
        confusion_matrix: Confusion matrix as numpy array.
        overall_accuracy: Overall accuracy score.
        timestamp: ISO format creation timestamp.
        boa_baseline: BOA results per experiment.
        optimal_thresholds: Threshold optimization results per experiment.
        additional_info: Extra metadata (parameters, GFLOPs, etc.).
    """

    metrics: Dict
    confusion_matrix: np.ndarray
    overall_accuracy: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    boa_baseline: Dict = field(default_factory=dict)
    optimal_thresholds: Dict = field(default_factory=dict)
    additional_info: Dict = field(default_factory=dict)


class ResultsManager:
    """Manages model results and creates comparative visualizations.

    Attributes:
        results: Mapping of model names to ModelResult objects.
    """

    def __init__(self) -> None:
        self.results: Dict[str, ModelResult] = {}

    def save_model_results(
        self,
        model_name: str,
        metrics: Dict,
        conf_matrix: np.ndarray,
        overall_accuracy: float,
        additional_info: Optional[Dict] = None,
    ) -> None:
        """Save evaluation results for a model."""
        self.results[model_name] = ModelResult(
            metrics=metrics,
            confusion_matrix=conf_matrix,
            overall_accuracy=overall_accuracy,
            additional_info=additional_info or {},
        )

    def parse_metrics_from_output(
        self,
        model_name: str,
        metrics_dict: Dict,
        conf_matrix: np.ndarray,
    ) -> None:
        """Convert raw evaluation output into internal structure."""
        parsed = {}
        for c in CLASS_NAMES:
            if c in metrics_dict:
                parsed[c] = {
                    k: metrics_dict[c][k]
                    for k in METRIC_NAMES + ["Support"]
                    if k in metrics_dict[c]
                }

        self.save_model_results(
            model_name,
            parsed,
            conf_matrix,
            metrics_dict["Overall"]["Accuracy"],
        )

    def save_boa_results(
        self,
        model_name: str,
        df_results: Optional[pd.DataFrame] = None,
        threshold_results: Optional[Dict] = None,
        experiment: Optional[str] = None,
    ) -> None:
        """Save BOA baseline and/or threshold optimization results."""
        if model_name not in self.results:
            self.results[model_name] = ModelResult(
                metrics={},
                confusion_matrix=np.array([]),
                overall_accuracy=0.0,
            )

        if df_results is not None:
            for _, row in df_results.iterrows():
                exp = row["Experiment"]
                self.results[model_name].boa_baseline[exp] = float(row["Median BOA"])

        if threshold_results is not None and experiment is not None:
            self.results[model_name].optimal_thresholds[experiment] = threshold_results

    def plot_metric_comparison(
        self,
        metric: str,
        models: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (20, 12),
        save_path: Optional[str] = None,
    ) -> None:
        """Plot a metric by class comparing multiple models.

        Highlights the best model per class with a dashed border.

        Args:
            metric: Metric name to plot.
            models: Model names to include. If None, uses all.
            figsize: Figure size.
            save_path: If provided, saves the figure.
        """
        if models is None:
            models = sorted(self.results.keys())

        for m in models:
            for cname in CLASS_NAMES:
                if (
                    cname not in self.results[m].metrics
                    or metric not in self.results[m].metrics[cname]
                ):
                    raise ValueError(
                        f"Metric '{metric}' missing for class '{cname}' "
                        f"in model '{m}'"
                    )

        values_mat = np.array([
            [self.results[m].metrics[c][metric] for c in CLASS_NAMES]
            for m in models
        ])

        if metric in ("Omission Error", "Commission Error"):
            best_idx = values_mat.argmin(axis=0)
        else:
            best_idx = values_mat.argmax(axis=0)

        n_models = len(models)
        width = 0.8 / n_models
        x = np.arange(len(CLASS_NAMES))

        fig, ax = plt.subplots(figsize=figsize)

        for i, model in enumerate(models):
            vals = values_mat[i]
            offset = (i - n_models / 2 + 0.5) * width
            style = _get_style(model)
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
            Patch(color=_get_style(m)["color"], label=_get_style(m)["label"], alpha=0.85)
            for m in models
        ]

        ax.set_xlabel("Classes", fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f"{metric} Comparison Across Models", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(CLASS_NAMES, fontsize=10)
        ax.legend(
            handles=legend_patches,
            title="Models",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
        )
        ax.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_threshold_curve(
        self,
        model_name: str,
        experiment: str,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None,
    ) -> None:
        """Plot median BOA vs threshold for an experiment."""
        if experiment not in self.results[model_name].optimal_thresholds:
            print(f"Experiment '{experiment}' not found for {model_name}")
            return

        data = self.results[model_name].optimal_thresholds[experiment]
        style = _get_style(model_name)

        plt.figure(figsize=figsize)
        plt.plot(data["thresholds"], data["median_boas"], linewidth=2, color=style["color"])
        plt.scatter(
            data["best_threshold"],
            data["best_median_boa"],
            s=100, zorder=5, color=style["color"],
            label=f"t* = {data['best_threshold']:.2f}",
        )
        plt.xlabel("Threshold", fontsize=12)
        plt.ylabel("Median BOA", fontsize=12)
        plt.title(
            f"Threshold Optimization - {experiment} - {style['label']}",
            fontsize=14,
        )
        plt.grid(alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_errors_for_class(
        self,
        class_name: str,
        models: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None,
    ) -> None:
        """Plot Omission and Commission Error for a single class across models.

        Args:
            class_name: Class to analyze (e.g. 'Thick Cloud').
            models: Model names to include. If None, uses all.
            figsize: Figure size.
            save_path: If provided, saves the figure.
        """
        if models is None:
            models = sorted(self.results.keys())

        oe_vals, ce_vals, labels, colors = [], [], [], []
        for m in models:
            metrics = self.results[m].metrics.get(class_name, {})
            oe_vals.append(metrics.get("Omission Error", 0.0))
            ce_vals.append(metrics.get("Commission Error", 0.0))
            style = _get_style(m)
            labels.append(style["label"])
            colors.append(style["color"])

        x = np.arange(len(models))
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
        ax.set_title(f"Omission & Commission Error — {class_name}", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def get_summary_dataframe(
        self, models: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Generate summary DataFrame with key metrics for all models."""
        if models is None:
            models = sorted(self.results.keys())

        rows = []
        for model in models:
            result = self.results[model]
            row = {"Model": model, "Overall Accuracy": result.overall_accuracy}

            for class_name in CLASS_NAMES:
                if class_name in result.metrics:
                    for metric in ("F1-Score", "Precision", "Recall"):
                        if metric in result.metrics[class_name]:
                            row[f"{class_name} {metric}"] = result.metrics[class_name][metric]

            rows.append(row)

        return pd.DataFrame(rows)