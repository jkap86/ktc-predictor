"""Model registry: auto-discovers, loads, and caches multiple model iterations.

Each iteration lives in backend/iterations/<name>/ with:
- manifest.json (metadata)
- predict.py (inference pipeline)
- io.py (bundle loader)
- models/ (trained artifacts)
"""

import importlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ITERATIONS_DIR = Path(__file__).resolve().parent.parent.parent / "iterations"


class ModelIteration:
    """A single loaded model iteration."""

    def __init__(self, iteration_id: str, iteration_dir: Path, manifest: dict):
        self.id = iteration_id
        self.dir = iteration_dir
        self.manifest = manifest
        self._bundle: dict | None = None
        self._predict_module = None

    @property
    def name(self) -> str:
        return self.manifest.get("name", self.id)

    @property
    def description(self) -> str:
        return self.manifest.get("description", "")

    @property
    def models_dir(self) -> Path:
        return self.dir / "models"

    def _load_predict_module(self):
        """Dynamically import this iteration's predict module."""
        if self._predict_module is None:
            module_path = f"iterations.{self.id}.predict"
            self._predict_module = importlib.import_module(module_path)
        return self._predict_module

    def _load_io_module(self):
        """Dynamically import this iteration's io module."""
        module_path = f"iterations.{self.id}.io"
        return importlib.import_module(module_path)

    def load_bundle(self) -> dict:
        """Load model bundle from this iteration's models/ directory."""
        if self._bundle is None:
            io_mod = self._load_io_module()
            self._bundle = io_mod.load_bundle(str(self.models_dir))
            logger.info("Loaded model bundle for iteration '%s'", self.id)

            # Validate feature contract if available
            predict_mod = self._load_predict_module()
            if hasattr(predict_mod, "validate_feature_contract"):
                predict_mod.validate_feature_contract(
                    self._bundle.get("feature_names")
                )
        return self._bundle

    @property
    def bundle(self) -> dict:
        return self.load_bundle()

    def predict_end_ktc(self, **kwargs) -> dict:
        """Run this iteration's predict_end_ktc with the loaded bundle."""
        b = self.bundle
        predict_mod = self._load_predict_module()
        return predict_mod.predict_end_ktc(
            models=b["models"],
            clip_bounds=b["clip_bounds"],
            calibrators=b["calibrators"],
            sentinel_impute=b.get("sentinel_impute"),
            residual_correction=b.get("residual_correction"),
            knn_adjuster=b.get("knn_adjuster"),
            target_type=b.get("target_type", "log_ratio"),
            quantile_models=b.get("quantile_models"),
            **kwargs,
        )

    def get_metrics(self) -> dict:
        """Return metrics from the loaded bundle."""
        return self.bundle.get("metrics", {})

    def get_info(self) -> dict:
        """Return manifest + metrics for API responses."""
        info = dict(self.manifest)
        # Only include metrics if bundle is already loaded (don't force-load just for info)
        if self._bundle is not None:
            info["metrics"] = self._bundle.get("metrics", {})
        else:
            # Try loading metrics.json directly without loading full bundle
            metrics_path = self.models_dir / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path) as f:
                    info["metrics"] = json.load(f)
        return info


class ModelRegistry:
    """Registry of all available model iterations."""

    def __init__(self):
        self._iterations: dict[str, ModelIteration] = {}
        self._default_id: str | None = None

    def discover(self, iterations_dir: Path | None = None) -> list[str]:
        """Scan iterations directory and register all found iterations.

        Returns list of discovered iteration IDs.
        """
        search_dir = iterations_dir or ITERATIONS_DIR
        if not search_dir.exists():
            logger.warning("Iterations directory not found: %s", search_dir)
            return []

        discovered = []
        for manifest_path in sorted(search_dir.glob("*/manifest.json")):
            iteration_dir = manifest_path.parent
            try:
                with open(manifest_path) as f:
                    manifest = json.load(f)

                iteration_id = manifest.get("id", iteration_dir.name)
                self._iterations[iteration_id] = ModelIteration(
                    iteration_id=iteration_id,
                    iteration_dir=iteration_dir,
                    manifest=manifest,
                )
                discovered.append(iteration_id)
                logger.info("Discovered iteration: %s (%s)", iteration_id, manifest.get("name"))
            except Exception:
                logger.exception("Failed to load manifest from %s", manifest_path)

        # Set default to first discovered if not already set
        if discovered and self._default_id is None:
            self._default_id = discovered[0]

        return discovered

    def set_default(self, iteration_id: str) -> None:
        """Set the default model iteration."""
        if iteration_id not in self._iterations:
            raise KeyError(f"Unknown iteration: {iteration_id}")
        self._default_id = iteration_id

    @property
    def default_id(self) -> str:
        if self._default_id is None:
            raise RuntimeError("No model iterations registered")
        return self._default_id

    def get(self, iteration_id: str | None = None) -> ModelIteration:
        """Get a model iteration by ID, or the default."""
        target_id = iteration_id or self.default_id
        if target_id not in self._iterations:
            raise KeyError(f"Unknown iteration: {target_id}. Available: {list(self._iterations.keys())}")
        return self._iterations[target_id]

    def list_iterations(self) -> list[dict]:
        """Return info for all registered iterations."""
        return [
            {**it.get_info(), "is_default": it.id == self._default_id}
            for it in self._iterations.values()
        ]

    def list_ids(self) -> list[str]:
        """Return all registered iteration IDs."""
        return list(self._iterations.keys())


# ── Singleton ──

_registry: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    """Get or create the global model registry singleton."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
        discovered = _registry.discover()
        logger.info("Model registry initialized with %d iteration(s): %s", len(discovered), discovered)
    return _registry
