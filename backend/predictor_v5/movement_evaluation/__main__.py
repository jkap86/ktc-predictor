"""Isolated command-line entrypoint for the historical movement evaluator."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .contract import (
    MODEL_SOURCE_SHA256,
    ContractRefusal,
    KtcMode,
    validate_ktc_mode,
)
from .data import load_pinned_bundle, prepare_evaluation_data
from .metrics import summarize_primary
from .models import (
    attach_start_observable_diagnostics,
    run_ktc_diagnostic,
    run_primary_folds,
)
from .report import (
    build_report,
    publish_report,
    runtime_identity,
    source_identities,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the evaluator's deliberately narrow public grammar."""
    parser = argparse.ArgumentParser(
        description="Run the pinned 2020-2025 historical market-movement evaluation."
    )
    parser.add_argument(
        "--ktc-diagnostic",
        choices=("off", "archive"),
        default="off",
        help="optional archive-only KTC diagnostic (default: off)",
    )
    return parser.parse_args(argv)


def run(*, ktc_mode: KtcMode, repo_root: Path) -> Path:
    """Run every fixed stage and publish only the complete canonical report."""
    mode = validate_ktc_mode(ktc_mode)
    root = Path(repo_root)
    runtime = runtime_identity()
    sources = source_identities(root / "backend" / "predictor_v5")
    if sources.get("model.py") != MODEL_SOURCE_SHA256:
        raise ContractRefusal("protected model.py source identity does not match the contract")
    bundle = load_pinned_bundle(root, ktc_mode=mode)
    data = prepare_evaluation_data(bundle)
    folds = run_primary_folds(data.primary_scored)
    fold_scores, pooled = summarize_primary(folds)
    start_diagnostics = attach_start_observable_diagnostics(data, folds)
    pooled_with_diagnostics = dict(pooled)
    pooled_with_diagnostics["start_observable_diagnostics"] = start_diagnostics
    optional_ktc = run_ktc_diagnostic(
        data,
        mode=mode,
        primary_folds=folds,
    )
    report = build_report(
        bundle=bundle,
        data=data,
        folds=folds,
        fold_scores=fold_scores,
        pooled=pooled_with_diagnostics,
        optional_ktc=optional_ktc,
        sources=sources,
        mode=mode,
        runtime=runtime,
    )
    return publish_report(report, root)


def main(argv: Sequence[str] | None = None) -> int:
    """Return 0 for publication, 2 for contract refusal, and 1 otherwise."""
    arguments = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[3]
    try:
        run(ktc_mode=arguments.ktc_diagnostic, repo_root=repo_root)
    except ContractRefusal as exc:
        print(f"movement evaluation refused: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - the CLI maps every unexpected stage failure to exit 1.
        print(f"movement evaluation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
