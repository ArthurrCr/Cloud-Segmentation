"""Core experiment runner: data loading, persistence, and single run."""

import os
import pickle
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import torch

from cloudsen12.data.dataset import create_dataloaders
from cloudsen12.evaluation.boa import evaluate_test_dataset
from cloudsen12.evaluation.metrics import compute_metrics, evaluate_model
from cloudsen12.experiment.config import ExperimentConfig, set_seed
from cloudsen12.losses.segmentation import get_loss
from cloudsen12.training.trainer import train_model


class ExperimentRunner:
    """Runs and manages ML experiments for cloud segmentation.

    Supports three-stage experimentation via methods in the stages module.
    Model selection uses VALIDATION metrics. Test metrics are computed
    for final reporting only.

    Attributes:
        parts: Dataset parts for tacoreader.
        base_dir: Base directory for checkpoints and results.
        device: PyTorch device string.
        seed: Global random seed.
        results: List of experiment result dictionaries.
    """

    def __init__(
        self,
        parts: List[str],
        base_dir: str = "/content/drive/MyDrive/pibic",
        device: str = "cuda",
        seed: int = 42,
    ) -> None:
        self.parts = parts
        self.base_dir = base_dir
        self.device = device
        self.seed = seed
        self.results: List[Dict] = []

        self.checkpoints_dir = os.path.join(base_dir, "checkpoints")
        self.results_dir = os.path.join(base_dir, "results")
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        self._train_loader = None
        self._val_loader = None
        self._test_loader = None
        self._current_batch_size = None

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def get_dataloaders(self, batch_size: int):
        """Get or create dataloaders, caching by batch size."""
        if self._train_loader is None or self._current_batch_size != batch_size:
            self._train_loader, self._val_loader, self._test_loader = (
                create_dataloaders(
                    self.parts,
                    batch_size=batch_size,
                    normalize=True,
                    seed=self.seed,
                )
            )
            self._current_batch_size = batch_size
        return self._train_loader, self._val_loader, self._test_loader

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load_results(self, path: Optional[str] = None) -> int:
        """Load previous results from a CSV file.

        Returns:
            Number of results loaded.
        """
        if path is None:
            path = os.path.join(self.results_dir, "experiments_summary.csv")

        if os.path.exists(path):
            df = pd.read_csv(path)
            self.results = df.to_dict("records")
            print(f"Loaded {len(self.results)} previous results from {path}")
            return len(self.results)

        print(f"No previous results found at {path}")
        return 0

    def save_results(self) -> None:
        """Persist results to CSV and pickle."""
        df = pd.DataFrame(self.results)
        df.to_csv(
            os.path.join(self.results_dir, "experiments_summary.csv"),
            index=False,
        )
        with open(
            os.path.join(self.results_dir, "experiments_full.pkl"), "wb"
        ) as f:
            pickle.dump(self.results, f)

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def get_completed_experiments(self) -> set:
        return {r["name"] for r in self.results}

    def is_completed(self, name: str) -> bool:
        return name in self.get_completed_experiments()

    # ------------------------------------------------------------------
    # BOA metric extraction
    # ------------------------------------------------------------------

    def extract_boa_metrics(
        self, df_boa: pd.DataFrame, prefix: str = ""
    ) -> Dict:
        """Extract BOA metrics from an evaluation DataFrame."""
        cloud_boa = float(
            df_boa[df_boa["Experiment"] == "cloud/no cloud"]["Median BOA"].values[0]
        )
        shadow_boa = float(
            df_boa[df_boa["Experiment"] == "cloud shadow"]["Median BOA"].values[0]
        )
        valid_boa = float(
            df_boa[df_boa["Experiment"] == "valid/invalid"]["Median BOA"].values[0]
        )
        mean_boa = (cloud_boa + shadow_boa) / 2

        return {
            f"{prefix}cloud_boa": cloud_boa,
            f"{prefix}shadow_boa": shadow_boa,
            f"{prefix}valid_boa": valid_boa,
            f"{prefix}mean_boa": mean_boa,
        }

    # ------------------------------------------------------------------
    # Single experiment
    # ------------------------------------------------------------------

    def run(
        self, config: ExperimentConfig, resume: bool = True
    ) -> Optional[Dict]:
        """Run a single experiment.

        Trains the model and evaluates on both validation and test sets.
        Validation metrics drive model selection; test metrics are for
        final reporting only.

        Returns:
            Result dictionary, or None if the experiment was skipped.
        """
        if self.is_completed(config.name):
            print(f"[SKIP] {config.name} already completed")
            return None

        set_seed(config.seed)

        separator = "=" * 70
        print(f"\n{separator}")
        print(f"RUNNING: {config.name}")
        print(separator)
        print(f"  Loss:               {config.loss_name}")
        print(f"  LR:                 {config.learning_rate}")
        print(f"  Batch size:         {config.batch_size}")
        print(f"  Scheduler factor:   {config.scheduler_factor}")
        print(f"  Scheduler patience: {config.scheduler_patience}")
        print(f"  Seed:               {config.seed}")
        print("-" * 70)

        checkpoint_dir = os.path.join(self.checkpoints_dir, config.name)
        os.makedirs(checkpoint_dir, exist_ok=True)

        train_loader, val_loader, test_loader = self.get_dataloaders(
            config.batch_size
        )

        model = config.model_class(
            encoder_name=config.encoder_name,
            encoder_weights=config.encoder_weights,
            in_channels=config.in_channels,
            num_classes=config.num_classes,
        )

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters:         {n_params:,}")

        loss_fn = get_loss(config.loss_name, **config.loss_kwargs)
        print(f"  Loss fn:            {loss_fn.__class__.__name__}")

        resume_path = os.path.join(checkpoint_dir, "checkpoint.pth")
        if not resume or not os.path.isfile(resume_path):
            resume_path = None
            print("  Resume:             Starting from scratch")
        else:
            print("  Resume:             From checkpoint")

        print("-" * 70)

        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config.num_epochs,
            lr=config.learning_rate,
            device=self.device,
            checkpoint_dir=checkpoint_dir,
            resume_checkpoint=resume_path,
            metric_to_monitor="val_loss",
            mode="min",
            patience=config.patience,
            loss_fn=loss_fn,
            scheduler_factor=config.scheduler_factor,
            scheduler_patience=config.scheduler_patience,
        )

        best_path = os.path.join(checkpoint_dir, "best_model.pth")
        checkpoint = torch.load(
            best_path, map_location=self.device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device).eval()

        # Validation evaluation (for model selection)
        print("Evaluating on VALIDATION set (for model selection)...")
        df_boa_val, _ = evaluate_test_dataset(
            val_loader, model, device=self.device, normalize_imgs=False
        )
        val_metrics = self.extract_boa_metrics(df_boa_val, prefix="val_")
        conf_matrix_val = evaluate_model(
            val_loader, model, device=self.device, normalize_imgs=False
        )
        val_metrics["val_accuracy"] = compute_metrics(conf_matrix_val)[
            "Overall"
        ]["Accuracy"]

        # Test evaluation (for reporting)
        print("Evaluating on TEST set (for final reporting)...")
        df_boa_test, _ = evaluate_test_dataset(
            test_loader, model, device=self.device, normalize_imgs=False
        )
        test_metrics = self.extract_boa_metrics(df_boa_test, prefix="test_")
        conf_matrix_test = evaluate_model(
            test_loader, model, device=self.device, normalize_imgs=False
        )
        test_metrics["test_accuracy"] = compute_metrics(conf_matrix_test)[
            "Overall"
        ]["Accuracy"]

        result = {
            "name": config.name,
            "loss": config.loss_name,
            "lr": config.learning_rate,
            "batch_size": config.batch_size,
            "sched_factor": config.scheduler_factor,
            "sched_patience": config.scheduler_patience,
            "seed": config.seed,
            "best_val_loss": checkpoint.get("best_metric", -1),
            "best_epoch": checkpoint.get("best_epoch", -1),
            **val_metrics,
            **test_metrics,
            "n_params": n_params,
            "timestamp": datetime.now().isoformat(),
        }

        self.results.append(result)
        self.save_results()

        self._print_run_results(config.name, result, val_metrics, test_metrics)
        return result

    def _print_run_results(
        self,
        name: str,
        result: Dict,
        val_metrics: Dict,
        test_metrics: Dict,
    ) -> None:
        separator = "=" * 70
        print(f"\n{separator}")
        print(f"RESULTS: {name}")
        print(separator)
        print(
            f"  Best val_loss: {result['best_val_loss']:.4f} "
            f"(epoch {result['best_epoch']})"
        )
        print("\n  [VALIDATION - for selection]")
        print(f"    Cloud BOA:  {val_metrics['val_cloud_boa']:.4f}")
        print(f"    Shadow BOA: {val_metrics['val_shadow_boa']:.4f}")
        print(f"    Valid BOA:  {val_metrics['val_valid_boa']:.4f}")
        print(f"    Mean BOA:   {val_metrics['val_mean_boa']:.4f}")
        print(f"    Accuracy:   {val_metrics['val_accuracy']:.4f}")
        print("\n  [TEST - for reporting]")
        print(f"    Cloud BOA:  {test_metrics['test_cloud_boa']:.4f}")
        print(f"    Shadow BOA: {test_metrics['test_shadow_boa']:.4f}")
        print(f"    Valid BOA:  {test_metrics['test_valid_boa']:.4f}")
        print(f"    Mean BOA:   {test_metrics['test_mean_boa']:.4f}")
        print(f"    Accuracy:   {test_metrics['test_accuracy']:.4f}")
        print(separator + "\n")