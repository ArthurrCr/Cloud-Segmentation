"""Results management and comparative visualization for model evaluation."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap

from cloudsen12.config.constants import CLASS_NAMES, METRIC_NAMES
from cloudsen12.visualization.plots import _get_style, _clean_spines


@dataclass
class ModelResult:
    """Stores evaluation results for a single model."""

    metrics: Dict
    confusion_matrix: np.ndarray
    overall_accuracy: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    boa_baseline: Dict = field(default_factory=dict)
    optimal_thresholds: Dict = field(default_factory=dict)
    additional_info: Dict = field(default_factory=dict)


class ResultsManager:
    """Manages model results and creates comparative visualizations."""

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

    def save_param_count(
        self,
        model_name: str,
        model: torch.nn.Module,
    ) -> None:
        """Count and store model parameters in additional_info."""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        if model_name not in self.results:
            raise ValueError(f"No results for '{model_name}'. Run full_evaluation first.")

        self.results[model_name].additional_info["total_params"] = total
        self.results[model_name].additional_info["trainable_params"] = trainable
        print(f"{model_name}: {total / 1e6:.2f}M total, {trainable / 1e6:.2f}M trainable")

    # ------------------------------------------------------------------
    # Plot methods (all return pd.DataFrame with values)
    # ------------------------------------------------------------------

    def plot_metric_comparison(
        self,
        metric: str,
        models: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (20, 12),
        save_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """Plot a metric by class comparing multiple models.

        Returns:
            DataFrame with models as rows and classes as columns.
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
            handles=legend_patches, title="Models",
            bbox_to_anchor=(1.02, 1), loc="upper left",
        )
        ax.grid(axis="y", alpha=0.3)
        _clean_spines(ax)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

        # Build DataFrame.
        rows = []
        for i, m in enumerate(models):
            row = {"Model": _get_style(m)["label"]}
            for j, c in enumerate(CLASS_NAMES):
                row[c] = round(values_mat[i, j], 4)
            rows.append(row)
        return pd.DataFrame(rows)

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

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(data["thresholds"], data["median_boas"], linewidth=2, color=style["color"])
        ax.scatter(
            data["best_threshold"],
            data["best_median_boa"],
            s=100, zorder=5, color=style["color"],
            label=f"t* = {data['best_threshold']:.2f}",
        )
        ax.set_xlabel("Threshold", fontsize=12)
        ax.set_ylabel("Median BOA", fontsize=12)
        ax.set_title(
            f"Threshold Optimization - {experiment} - {style['label']}",
            fontsize=14,
        )
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=12)
        _clean_spines(ax)
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
    ) -> pd.DataFrame:
        """Plot Omission and Commission Error for a single class.

        Returns:
            DataFrame with OE and CE per model.
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
        ax.set_title(f"Omission & Commission Error \u2014 {class_name}", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        _clean_spines(ax)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

        return pd.DataFrame({
            "Model": labels,
            "Omission Error": [round(v, 4) for v in oe_vals],
            "Commission Error": [round(v, 4) for v in ce_vals],
        })

    def plot_model_parameter_counts(
        self,
        models: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """Plot total parameter count as horizontal bar chart.

        Returns:
            DataFrame with total and trainable params per model.
        """
        if models is None:
            models = [
                m for m in self.results
                if "total_params" in self.results[m].additional_info
            ]

        if not models:
            print("No parameter counts saved. Call save_param_count() first.")
            return pd.DataFrame()

        labels, totals, trainable, colors = [], [], [], []
        for m in models:
            info = self.results[m].additional_info
            style = _get_style(m)
            labels.append(style["label"])
            totals.append(info["total_params"] / 1e6)
            trainable.append(info.get("trainable_params", 0) / 1e6)
            colors.append(style["color"])

        y = np.arange(len(models))
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
        _clean_spines(ax)
        ax.spines["bottom"].set_visible(True)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

        return pd.DataFrame({
            "Model": labels,
            "Total Params (M)": [round(t, 2) for t in totals],
            "Trainable Params (M)": [round(t, 2) for t in trainable],
        })

    def get_summary_dataframe(
        self, models: Optional[List[str]] = None,
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

    # ------------------------------------------------------------------
    # Qualitative examples
    # ------------------------------------------------------------------

    _CLASS_CMAP = ListedColormap(["#2ecc71", "#e74c3c", "#f39c12", "#3498db"])
    _CLASS_LABELS = CLASS_NAMES

    def plot_qualitative_examples(
        self,
        models_dict: Dict[str, Tuple],
        test_loader: torch.utils.data.DataLoader,
        n_samples: int = 4,
        indices: Optional[List[int]] = None,
        figsize_per_col: float = 3.0,
        save_path: Optional[str] = None,
    ) -> None:
        """Show RGB, GT, and predictions side by side for sample patches."""
        from cloudsen12.inference.prediction import get_predictions
        from cloudsen12.inference.normalization import (
            get_normalization_stats, normalize_images,
        )
        from cloudsen12.config.constants import SENTINEL_BANDS

        device = next(
            (
                str(next(ms[0][0].parameters()).device)
                for ms in models_dict.values()
            ),
            "cpu",
        )

        all_imgs, all_gts = [], []
        for imgs, gts in test_loader:
            for i in range(imgs.size(0)):
                all_imgs.append(imgs[i])
                all_gts.append(gts[i])

        n_total = len(all_imgs)
        if indices is None:
            indices = np.linspace(0, n_total - 1, n_samples, dtype=int).tolist()

        model_names = list(models_dict.keys())
        n_cols = 2 + len(model_names)
        n_rows = len(indices)
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(figsize_per_col * n_cols, figsize_per_col * n_rows),
        )
        if n_rows == 1:
            axes = axes[np.newaxis, :]

        mean, std = get_normalization_stats(device, False, SENTINEL_BANDS)

        for row, idx in enumerate(indices):
            img_t = all_imgs[idx]
            gt = all_gts[idx].numpy()

            rgb = img_t[[3, 2, 1]].numpy().transpose(1, 2, 0)
            rgb = np.clip(rgb / np.percentile(rgb, 98), 0, 1)

            axes[row, 0].imshow(rgb)
            axes[row, 0].set_title("RGB" if row == 0 else "", fontsize=10)
            axes[row, 0].set_ylabel(f"Patch {idx}", fontsize=9)
            axes[row, 0].set_xticks([])
            axes[row, 0].set_yticks([])

            axes[row, 1].imshow(gt, cmap=self._CLASS_CMAP, vmin=0, vmax=3)
            axes[row, 1].set_title("Ground Truth" if row == 0 else "", fontsize=10)
            axes[row, 1].set_xticks([])
            axes[row, 1].set_yticks([])

            for col, m_name in enumerate(model_names):
                models_list, use_ens, norm = models_dict[m_name]
                inp = img_t.unsqueeze(0).to(device).float()
                if norm:
                    inp = normalize_images(inp, mean, std)
                with torch.no_grad():
                    pred = get_predictions(
                        models_list, inp, use_ensemble=use_ens,
                    )[0].cpu().numpy()

                style = _get_style(m_name)
                axes[row, col + 2].imshow(
                    pred, cmap=self._CLASS_CMAP, vmin=0, vmax=3,
                )
                axes[row, col + 2].set_title(
                    style["label"] if row == 0 else "", fontsize=10,
                )
                axes[row, col + 2].set_xticks([])
                axes[row, col + 2].set_yticks([])

        legend_patches = [
            Patch(color=self._CLASS_CMAP(i), label=c)
            for i, c in enumerate(self._CLASS_LABELS)
        ]
        fig.legend(
            handles=legend_patches, loc="lower center",
            ncol=4, fontsize=9, frameon=True,
        )
        plt.tight_layout(rect=[0, 0.04, 1, 1])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    # ------------------------------------------------------------------
    # Inference benchmarking
    # ------------------------------------------------------------------

    def benchmark_inference(
        self,
        model_name: str,
        models: Union[torch.nn.Module, List[torch.nn.Module]],
        test_loader: torch.utils.data.DataLoader,
        use_ensemble: bool = False,
        normalize_imgs: bool = False,
        n_batches: int = 20,
    ) -> float:
        """Benchmark inference time and store in additional_info."""
        from cloudsen12.inference.prediction import get_predictions
        from cloudsen12.inference.normalization import (
            get_normalization_stats, normalize_images,
        )
        from cloudsen12.config.constants import SENTINEL_BANDS

        if not isinstance(models, list):
            models = [models]

        device = str(next(models[0].parameters()).device)
        mean, std = get_normalization_stats(device, False, SENTINEL_BANDS)

        for m in models:
            m.eval()

        # Warmup.
        imgs, _ = next(iter(test_loader))
        imgs = imgs.to(device).float()
        if normalize_imgs:
            imgs = normalize_images(imgs, mean, std)
        with torch.no_grad():
            get_predictions(models, imgs, use_ensemble=use_ensemble)
        if "cuda" in device:
            torch.cuda.synchronize()

        times = []
        batch_iter = iter(test_loader)
        for _ in range(n_batches):
            try:
                imgs, _ = next(batch_iter)
            except StopIteration:
                batch_iter = iter(test_loader)
                imgs, _ = next(batch_iter)

            imgs = imgs.to(device).float()
            if normalize_imgs:
                imgs = normalize_images(imgs, mean, std)

            if "cuda" in device:
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            with torch.no_grad():
                get_predictions(models, imgs, use_ensemble=use_ensemble)

            if "cuda" in device:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)

        avg = np.mean(times)
        std_t = np.std(times)

        if model_name not in self.results:
            raise ValueError(f"No results for '{model_name}'.")

        self.results[model_name].additional_info["avg_batch_time"] = avg
        self.results[model_name].additional_info["std_batch_time"] = std_t
        self.results[model_name].additional_info["batch_size"] = test_loader.batch_size

        avg_per_img = avg / test_loader.batch_size
        print(
            f"{model_name}: {avg:.4f}s/batch \u00b1 {std_t:.4f}s "
            f"({avg_per_img * 1000:.1f} ms/image)"
        )
        return avg

    def plot_inference_cost(
        self,
        models: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """Plot inference time per image.

        Returns:
            DataFrame with ms/image and std per model.
        """
        if models is None:
            models = [
                m for m in self.results
                if "avg_batch_time" in self.results[m].additional_info
            ]

        if not models:
            print("No inference timing data. Call benchmark_inference() first.")
            return pd.DataFrame()

        labels, times_ms, stds_ms, colors = [], [], [], []
        for m in models:
            info = self.results[m].additional_info
            bs = info.get("batch_size", 1)
            style = _get_style(m)
            labels.append(style["label"])
            times_ms.append(info["avg_batch_time"] / bs * 1000)
            stds_ms.append(info["std_batch_time"] / bs * 1000)
            colors.append(style["color"])

        y = np.arange(len(models))
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
        _clean_spines(ax)
        ax.spines["bottom"].set_visible(True)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

        return pd.DataFrame({
            "Model": labels,
            "ms/image": [round(t, 1) for t in times_ms],
            "\u00b1 std (ms)": [round(s, 1) for s in stds_ms],
        })

    # ------------------------------------------------------------------
    # Efficiency bubble chart
    # ------------------------------------------------------------------

    def _compute_mean_iou(self, model_name: str) -> float:
        """Compute mean IoU from the stored confusion matrix."""
        cm = self.results[model_name].confusion_matrix
        if cm.size == 0:
            return 0.0
        intersection = np.diag(cm)
        union = cm.sum(axis=1) + cm.sum(axis=0) - intersection
        with np.errstate(invalid="ignore", divide="ignore"):
            iou = intersection / union
        return float(np.nanmean(iou))

    def plot_efficiency_bubble(
        self,
        models: Optional[List[str]] = None,
        x_metric: str = "total_params",
        y_metric: str = "mean_iou",
        size_metric: Optional[str] = "avg_batch_time",
        figsize: Tuple[int, int] = (10, 7),
        save_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """Bubble chart: parameters vs performance, sized by inference cost.

        Returns:
            DataFrame with params, metric value, and inference time per model.
        """
        if models is None:
            models = [
                m for m in self.results
                if "total_params" in self.results[m].additional_info
            ]

        if not models:
            print("No parameter data. Call save_param_count() first.")
            return pd.DataFrame()

        fig, ax = plt.subplots(figsize=figsize)

        table_rows: List[Dict] = []

        for m in models:
            info = self.results[m].additional_info
            result = self.results[m]
            style = _get_style(m)

            x_val = info.get("total_params", 0) / 1e6

            if y_metric == "mean_iou":
                y_val = self._compute_mean_iou(m)
            elif y_metric == "overall_accuracy":
                y_val = result.overall_accuracy
            elif " " in y_metric:
                parts = y_metric.rsplit(" ", 1)
                cls_name, metric_name = parts[0], parts[1]
                y_val = result.metrics.get(cls_name, {}).get(metric_name, 0)
            else:
                y_val = self._compute_mean_iou(m)

            if size_metric and size_metric in info:
                s_val = info[size_metric] * 1000
                bubble_size = max(s_val * 10, 80)
            else:
                s_val = None
                bubble_size = 200

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

            row = {"Model": style["label"], "Params (M)": round(x_val, 2), y_metric: round(y_val, 4)}
            if s_val is not None:
                row["Inference (ms)"] = round(s_val, 1)
            table_rows.append(row)

        y_label = "Mean IoU" if y_metric == "mean_iou" else y_metric.replace("_", " ").title()
        ax.set_xlabel("Parameters (M)", fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        title = f"Efficiency: Parameters vs {y_label}"
        if size_metric:
            title += " (bubble = inference time)"
        ax.set_title(title, fontsize=13)
        ax.grid(alpha=0.3)
        _clean_spines(ax)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

        return pd.DataFrame(table_rows)