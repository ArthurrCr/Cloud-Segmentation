"""Results management and inference benchmarking for model evaluation.

This module stores, organizes, and exports evaluation data. All
visualization is delegated to ``cloudsen12.visualization.plots``.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from cloudsen12.config.constants import CLASS_NAMES, METRIC_NAMES
from cloudsen12.visualization.style import get_style, sort_models


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
    """Manages model results and exposes data accessors for plotting.

    This class is responsible for storing, querying, and exporting
    evaluation data. Plotting should be done by calling the standalone
    functions in ``cloudsen12.visualization.plots`` with data obtained
    from the accessor methods below.
    """

    def __init__(self) -> None:
        self.results: Dict[str, ModelResult] = {}

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

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
                self.results[model_name].boa_baseline[exp] = float(
                    row["Median BOA"]
                )

        if threshold_results is not None and experiment is not None:
            self.results[model_name].optimal_thresholds[experiment] = (
                threshold_results
            )

    def save_param_count(
        self,
        model_name: str,
        model: torch.nn.Module,
    ) -> None:
        """Count and store model parameters."""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

        if model_name not in self.results:
            raise ValueError(
                f"No results for '{model_name}'. Run evaluation first."
            )

        self.results[model_name].additional_info["total_params"] = total
        self.results[model_name].additional_info["trainable_params"] = trainable
        print(
            f"{model_name}: {total / 1e6:.2f}M total, "
            f"{trainable / 1e6:.2f}M trainable"
        )

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
        """Benchmark inference time and store in additional_info.

        Returns:
            Average time per batch in seconds.
        """
        from cloudsen12.config.constants import SENTINEL_BANDS
        from cloudsen12.inference.normalization import (
            get_normalization_stats,
            normalize_images,
        )
        from cloudsen12.inference.prediction import get_predictions

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

        avg = float(np.mean(times))
        std_t = float(np.std(times))

        if model_name not in self.results:
            raise ValueError(f"No results for '{model_name}'.")

        info = self.results[model_name].additional_info
        info["avg_batch_time"] = avg
        info["std_batch_time"] = std_t
        info["batch_size"] = test_loader.batch_size

        avg_per_img = avg / test_loader.batch_size
        print(
            f"{model_name}: {avg:.4f}s/batch +/- {std_t:.4f}s "
            f"({avg_per_img * 1000:.1f} ms/image)"
        )
        return avg

    # ------------------------------------------------------------------
    # Data accessors (prepare dicts consumed by plots.py functions)
    # ------------------------------------------------------------------

    def get_model_names(self) -> List[str]:
        """Return stored model names in canonical order."""
        return sort_models(list(self.results.keys()))

    def get_confusion_matrices(
        self, model_names: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """Return ``{model_name: confusion_matrix}``."""
        names = model_names or self.get_model_names()
        return {m: self.results[m].confusion_matrix for m in names}

    def get_metrics_by_model(
        self, model_names: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Return ``{model_name: {class_name: {metric: value}}}``."""
        names = model_names or self.get_model_names()
        return {m: self.results[m].metrics for m in names}

    def get_threshold_data(
        self, model_name: str, experiment: str,
    ) -> Optional[Dict]:
        """Return threshold optimization data or None."""
        return self.results[model_name].optimal_thresholds.get(experiment)

    def get_param_data(
        self, model_names: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Return ``{model_name: {"total": M, "trainable": M}}``."""
        names = model_names or [
            m for m in self.get_model_names()
            if "total_params" in self.results[m].additional_info
        ]
        out = {}
        for m in names:
            info = self.results[m].additional_info
            out[m] = {
                "total": info["total_params"] / 1e6,
                "trainable": info.get("trainable_params", 0) / 1e6,
            }
        return out

    def get_inference_data(
        self, model_names: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Return ``{model_name: {"ms_per_image": ..., "std_ms": ...}}``."""
        names = model_names or [
            m for m in self.get_model_names()
            if "avg_batch_time" in self.results[m].additional_info
        ]
        out = {}
        for m in names:
            info = self.results[m].additional_info
            bs = info.get("batch_size", 1)
            out[m] = {
                "ms_per_image": info["avg_batch_time"] / bs * 1000,
                "std_ms": info["std_batch_time"] / bs * 1000,
            }
        return out

    def get_bubble_data(
        self,
        model_names: Optional[List[str]] = None,
        y_metric: str = "mean_iou",
    ) -> Dict[str, Dict[str, float]]:
        """Return data for the efficiency bubble chart.

        Returns:
            ``{model_name: {"params_m": ..., "y_value": ...,
            "inference_ms": ...}}``.
        """
        names = model_names or [
            m for m in self.get_model_names()
            if "total_params" in self.results[m].additional_info
        ]
        out = {}
        for m in names:
            info = self.results[m].additional_info
            result = self.results[m]

            if y_metric == "mean_iou":
                y_val = self._compute_mean_iou(m)
            elif y_metric == "overall_accuracy":
                y_val = result.overall_accuracy
            else:
                y_val = self._compute_mean_iou(m)

            inf_ms = None
            if "avg_batch_time" in info:
                bs = info.get("batch_size", 1)
                inf_ms = info["avg_batch_time"] / bs * 1000

            out[m] = {
                "params_m": info.get("total_params", 0) / 1e6,
                "y_value": y_val,
                "inference_ms": inf_ms,
            }
        return out

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

    # ------------------------------------------------------------------
    # Summary export
    # ------------------------------------------------------------------

    def get_summary_dataframe(
        self, model_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Generate summary DataFrame with key metrics for all models."""
        names = model_names or self.get_model_names()
        rows = []
        for model in names:
            result = self.results[model]
            style = get_style(model)
            row = {
                "Model": style["label"],
                "Overall Accuracy": result.overall_accuracy,
            }
            for class_name in CLASS_NAMES:
                if class_name in result.metrics:
                    for metric in ("F1-Score", "Precision", "Recall"):
                        if metric in result.metrics[class_name]:
                            row[f"{class_name} {metric}"] = (
                                result.metrics[class_name][metric]
                            )
            rows.append(row)
        return pd.DataFrame(rows)