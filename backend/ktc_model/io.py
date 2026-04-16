"""Pickle compatibility: re-export EnsembleModel from v1 iteration."""
from iterations.v1_hgb_baseline.io import EnsembleModel, load_bundle, save_bundle

__all__ = ["EnsembleModel", "load_bundle", "save_bundle"]
