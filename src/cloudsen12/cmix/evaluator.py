"""CMIX evaluation orchestrator.

Ties together the PixBox dataset loader, full-scene sliding-window inference,
and CMIX metrics into a single high-level API.

Typical usage
-------------
::

    evaluator = CmixEvaluator(
        pixbox_dir="./data/pixbox",
        scenes_dir="./data/scenes",
        output_dir="./cmix_results",
        device="cuda",
    )

    evaluator.run(
        model_name="Unet + RegNetZ-D8",
        models=[model_reg_d8],
        normalize_fn=normalize_fn_standardize,
    )
    evaluator.run(
        model_name="CloudS2Mask ensemble",
        models=models_ensemble,
        normalize_fn=normalize_fn_clouds2mask,
    )

    table = evaluator.summary_table()
    print(table)
"""

import gc
import json
import time
import zipfile
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from cloudsen12.cmix.dataset import (
    CMIX_EXPERIMENTS,
    apply_experiment_filter,
    extract_predictions_at_pixels,
    load_pixbox_labels,
)
from cloudsen12.cmix.inference import load_scene_bands, run_full_scene_inference
from cloudsen12.cmix.metrics import (
    BinaryConfusion,
    accumulate_global_confusion,
    compute_binary_confusion,
    compute_cmix_metrics,
    format_cmix_results_table,
)


class CmixEvaluator:
    """Orchestrates full-scene CMIX evaluation for one or more models.

    For each model, the evaluator:
    1. Iterates over all 29 Sentinel-2 scenes defined by the PixBox labels.
    2. Loads and preprocesses each scene.
    3. Runs sliding-window inference to produce a full-scene prediction mask.
    4. Extracts predictions at the labeled pixel coordinates.
    5. Accumulates binary confusion counts per CMIX experiment.
    6. Computes global OA, BOA, PA, UA from the aggregated counts.

    Results are cached to disk in JSON format so scenes already evaluated
    are not re-processed.

    Args:
        pixbox_dir: Directory containing the extracted PixBox dataset.
        scenes_dir: Root directory where Sentinel-2 scene subdirectories live.
            Each scene directory should be named by its scene/granule ID as
            it appears in the PixBox labels.
        output_dir: Directory for caching per-scene predictions and results.
        device: PyTorch device string ('cuda' or 'cpu').
        patch_size: Patch size for sliding-window inference.
        stride: Stride for sliding-window inference.
        inference_batch_size: Number of patches per forward pass.
        use_offset_grid: If True, uses the CloudS2Mask offset grid
            strategy where alternate rows are shifted by half the patch
            width (Wright et al. 2024, Fig. 4).
        use_gradient_weight: If True, applies linear gradient weighting
            at patch edges for smoother merging (Wright et al. 2024,
            Fig. 6).
        gradient_border: Width of the edge gradient ramp in pixels.
    """

    def __init__(
        self,
        pixbox_dir: str,
        scenes_dir: str,
        output_dir: str = "./cmix_results",
        device: str = "cuda",
        patch_size: int = 512,
        stride: int = 256,
        inference_batch_size: int = 8,
        use_offset_grid: bool = False,
        use_gradient_weight: bool = False,
        gradient_border: int = 64,
    ) -> None:
        self.pixbox_dir = Path(pixbox_dir)
        self.scenes_dir = Path(scenes_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.patch_size = patch_size
        self.stride = stride
        self.inference_batch_size = inference_batch_size
        self.use_offset_grid = use_offset_grid
        self.use_gradient_weight = use_gradient_weight
        self.gradient_border = gradient_border

        # {model_name: {experiment: {metric: value}}}
        self._results: Dict[str, Dict[str, Dict[str, float]]] = {}

        print(f"CmixEvaluator initialized.")
        print(f"  Device:          {self.device}")
        print(f"  PixBox dir:      {self.pixbox_dir}")
        print(f"  Scenes dir:      {self.scenes_dir}")
        print(f"  Output dir:      {self.output_dir}")
        print(f"  Patch / stride:  {self.patch_size} / {self.stride}")
        print(f"  Offset grid:     {self.use_offset_grid}")
        print(f"  Gradient weight: {self.use_gradient_weight}"
              + (f" (border={self.gradient_border})" if self.use_gradient_weight else ""))

        print("\nLoading PixBox labels...")
        self._labels: Dict[str, pd.DataFrame] = load_pixbox_labels(
            str(self.pixbox_dir)
        )

        # Build a scene index: map SAFE names to paths (dirs or zips).
        self._scene_index: Dict[str, Path] = {}
        self._build_scene_index()

    def _build_scene_index(self) -> None:
        """Scan scenes_dir recursively for .SAFE dirs and .SAFE.zip files."""
        for p in self.scenes_dir.rglob("*.SAFE"):
            if p.is_dir():
                self._scene_index[p.name] = p
        for p in self.scenes_dir.rglob("*.SAFE.zip"):
            safe_name = p.name.replace(".zip", "")
            if safe_name not in self._scene_index:
                self._scene_index[safe_name] = p  # points to zip
        print(f"  Scene index: {len(self._scene_index)} scenes found "
              f"({sum(1 for v in self._scene_index.values() if v.suffix == '.zip')} zipped).")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scene_dir(self, scene_id: str) -> Path:
        """Resolve the filesystem path for a given scene ID.

        Uses the pre-built scene index. If the entry is a .zip file,
        extracts it in place and updates the index.

        Args:
            scene_id: SAFE directory name, e.g.
                'S2A_MSIL1C_20170113T072241_...SAFE'.

        Raises:
            FileNotFoundError: If no matching directory or ZIP is found.
        """
        path = self._scene_index.get(scene_id)
        if path is None:
            raise FileNotFoundError(
                f"No directory or ZIP found for scene '{scene_id}' "
                f"under '{self.scenes_dir}'."
            )

        # Already extracted.
        if path.is_dir():
            return path

        # It's a .zip — extract it.
        print(f"  Extracting {path.name}...")
        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(path.parent)

        extracted = path.parent / scene_id
        if extracted.is_dir():
            self._scene_index[scene_id] = extracted
            return extracted

        # Fallback: look for any .SAFE dir that appeared.
        for d in path.parent.iterdir():
            if d.is_dir() and scene_id.replace(".SAFE", "") in d.name:
                self._scene_index[scene_id] = d
                return d

        raise FileNotFoundError(
            f"Extraction of '{path.name}' did not produce expected "
            f"directory '{scene_id}'."
        )

    def _cache_path(self, model_name: str, scene_id: str) -> Path:
        """Return path for caching per-scene prediction coordinates."""
        safe_model = model_name.replace("/", "-").replace(" ", "_")
        safe_scene = scene_id.replace("/", "-")
        return self.output_dir / f"{safe_model}_{safe_scene}_preds.npz"

    def _results_path(self, model_name: str) -> Path:
        safe_model = model_name.replace("/", "-").replace(" ", "_")
        return self.output_dir / f"{safe_model}_cmix_results.json"

    def _load_cached_scene_preds(
        self, model_name: str, scene_id: str
    ) -> Optional[dict]:
        """Load cached per-scene predictions if available."""
        path = self._cache_path(model_name, scene_id)
        if path.exists():
            data = np.load(path)
            return {"y_true": data["y_true"], "y_pred": data["y_pred"]}
        return None

    def _save_scene_preds(
        self,
        model_name: str,
        scene_id: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> None:
        path = self._cache_path(model_name, scene_id)
        np.savez_compressed(path, y_true=y_true, y_pred=y_pred)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        model_name: str,
        models: List[torch.nn.Module],
        normalize_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        force: bool = False,
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate a model on the full CMIX benchmark.

        Args:
            model_name: Display name for the model (used in output tables).
            models: List of PyTorch models. Multiple models are ensembled
                by averaging their softmax probabilities.
            normalize_fn: Callable that accepts a (B, 13, H, W) float tensor
                of raw DN values and returns a normalized tensor. Pass None
                to use raw DN values (not recommended).
            force: If True, re-run inference even for scenes that are cached.

        Returns:
            Nested dict {experiment: {metric: value}}.
        """
        print(f"\n{'=' * 70}")
        print(f"CMIX Evaluation: {model_name}")
        print(f"{'=' * 70}")

        for m in models:
            m.to(self.device).eval()

        scene_ids = list(self._labels.keys())
        print(f"Scenes to process: {len(scene_ids)}")

        # Per-experiment confusion accumulators.
        confusions: Dict[str, List[BinaryConfusion]] = {
            exp: [] for exp in CMIX_EXPERIMENTS
        }

        for scene_idx, scene_id in enumerate(scene_ids, start=1):
            print(f"\n[{scene_idx}/{len(scene_ids)}] Scene: {scene_id}")
            pixels_df = self._labels[scene_id]

            # Try to load from cache first.
            cached = None if force else self._load_cached_scene_preds(
                model_name, scene_id
            )

            if cached is not None:
                print(f"  Using cached predictions ({len(cached['y_true'])} pixels).")
                y_true = cached["y_true"]
                y_pred = cached["y_pred"]
            else:
                # Locate and load the scene.
                try:
                    scene_dir = self._scene_dir(scene_id)
                except FileNotFoundError as exc:
                    print(f"  SKIPPED: {exc}")
                    continue

                print(f"  Loading bands from: {scene_dir.name}")
                t0 = time.time()
                image, _ = load_scene_bands(str(scene_dir), target_resolution=10)
                print(f"  Bands loaded in {time.time() - t0:.1f}s | shape: {image.shape} | "
                      f"dtype: {image.dtype} | "
                      f"{image.nbytes / 1024**3:.1f} GB")

                # Run full-scene inference.
                t0 = time.time()
                pred_mask = run_full_scene_inference(
                    image=image,
                    models=models,
                    device=self.device,
                    patch_size=self.patch_size,
                    stride=self.stride,
                    batch_size=self.inference_batch_size,
                    normalize_fn=normalize_fn,
                    use_offset_grid=self.use_offset_grid,
                    use_gradient_weight=self.use_gradient_weight,
                    gradient_border=self.gradient_border,
                )
                del image  # Free ~3 GB immediately.
                print(f"  Inference done in {time.time() - t0:.1f}s")

                # Extract predictions at labeled coordinates.
                y_true = pixels_df["label"].to_numpy(dtype=np.int64)
                y_pred = extract_predictions_at_pixels(pred_mask, pixels_df)
                del pred_mask  # Free ~0.5 GB.

                self._save_scene_preds(model_name, scene_id, y_true, y_pred)
                print(f"  Cached predictions for {len(y_true)} pixels.")

            # Accumulate per-experiment confusion counts.
            for exp_name in CMIX_EXPERIMENTS:
                y_true_bin, y_pred_bin, valid = apply_experiment_filter(
                    y_true, y_pred, exp_name
                )
                if valid.sum() == 0:
                    print(f"  WARNING: 0 valid pixels for experiment '{exp_name}'.")
                    continue
                confusion = compute_binary_confusion(y_true_bin, y_pred_bin)
                confusions[exp_name].append(confusion)

            # Force garbage collection between scenes.
            gc.collect()
            torch.cuda.empty_cache()

        # Aggregate and compute global metrics.
        print(f"\n{'=' * 70}")
        print("Global CMIX metrics:")
        exp_results: Dict[str, Dict[str, float]] = {}

        for exp_name, exp_confusions in confusions.items():
            if not exp_confusions:
                print(f"  No data for experiment '{exp_name}'.")
                exp_results[exp_name] = {"OA": float("nan"), "BOA": float("nan"),
                                          "PA": float("nan"), "UA": float("nan")}
                continue

            global_conf = accumulate_global_confusion(exp_confusions)
            metrics = compute_cmix_metrics(global_conf)
            exp_results[exp_name] = metrics

            cfg = CMIX_EXPERIMENTS[exp_name]
            print(f"\n  Experiment: {exp_name} ({cfg['description']})")
            print(f"    TP={global_conf.tp:,}  FN={global_conf.fn:,}  "
                  f"FP={global_conf.fp:,}  TN={global_conf.tn:,}")
            for metric, val in metrics.items():
                print(f"    {metric}: {val:.4f}")

        self._results[model_name] = exp_results

        # Persist results to JSON.
        with open(self._results_path(model_name), "w") as f:
            json.dump(exp_results, f, indent=2)
        print(f"\nResults saved to: {self._results_path(model_name)}")

        return exp_results

    def summary_table(
        self,
        model_names: Optional[List[str]] = None,
    ) -> str:
        """Return a formatted comparison table for all evaluated models.

        Args:
            model_names: Subset of models to include. Defaults to all.

        Returns:
            Formatted string ready for printing.

        Raises:
            RuntimeError: If no results are available.
        """
        if not self._results:
            raise RuntimeError(
                "No results available. Call run() for at least one model first."
            )

        subset = (
            {k: self._results[k] for k in model_names if k in self._results}
            if model_names
            else self._results
        )

        return format_cmix_results_table(subset)

    def summary_dataframe(
        self,
        model_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Return results as a tidy DataFrame for further analysis.

        Returns:
            DataFrame with columns: model, experiment, OA, BOA, PA, UA.
        """
        rows = []
        subset = model_names or list(self._results.keys())

        for model_name in subset:
            if model_name not in self._results:
                continue
            for exp_name, metrics in self._results[model_name].items():
                rows.append({
                    "model": model_name,
                    "experiment": exp_name,
                    **metrics,
                })

        return pd.DataFrame(rows)

    def load_results_from_disk(self, model_name: str) -> bool:
        """Load previously saved results from JSON cache.

        Args:
            model_name: Model name whose results file should be loaded.

        Returns:
            True if results were loaded successfully, False otherwise.
        """
        path = self._results_path(model_name)
        if not path.exists():
            return False
        with open(path) as f:
            self._results[model_name] = json.load(f)
        print(f"Loaded cached results for '{model_name}' from {path}.")
        return True