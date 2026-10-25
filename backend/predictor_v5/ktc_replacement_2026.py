"""One-shot, fixed-protocol 2026 KTC replacement evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from predictor_v5.model import WhatIfModel
from predictor_v5.replacement_cohort import (
    COHORT_COLUMNS,
    matched_cohort,
    verify_common_stats,
)
from predictor_v5.replacement_ktc import _canonical, _read, load_ktc_target
from predictor_v5.replacement_lifecycle import preflight, run_claimed_stages
from predictor_v5.replacement_metrics import diagnostics, gate, score
from predictor_v5.replacement_mfl import load_mfl_target, load_stats
from predictor_v5.replacement_report import _render_report


class ProtocolFailure(ValueError):
    """A deterministic protocol or registered-input refusal."""


def _defaults() -> dict:
    return {
        "preflight": preflight,
        "run_claimed_stages": run_claimed_stages,
        "load_stats": load_stats,
        "load_ktc_target": load_ktc_target,
        "load_mfl_target": load_mfl_target,
        "verify_common_stats": verify_common_stats,
        "read_identity": _read,
        "parse_canonical": _canonical,
        "model_factory": WhatIfModel,
        "matched_cohort": matched_cohort,
        "score": score,
        "diagnostics": diagnostics,
        "gate": gate,
        "validate_outcomes": _validate_outcomes,
        "fit_evaluation": _fit_evaluation,
    }


def _validate_outcomes(context: dict, dependencies: dict, root: Path) -> dict:
    registration = context["registration"]
    starts = context["starts"]
    stats = {
        source: dependencies["load_stats"](
            root / registration[source]["target_stats"]["path"]
        )
        for source in ("ktc", "mfl")
    }
    ktc = dependencies["load_ktc_target"](registration, starts, stats["ktc"], root)
    mfl = dependencies["load_mfl_target"](registration, starts, stats["mfl"], root)
    dependencies["verify_common_stats"](ktc, mfl)

    read_identity = dependencies["read_identity"]
    parse_canonical = dependencies["parse_canonical"]
    ktc_sidecar = parse_canonical(read_identity(root, registration["ktc"]["target_sidecar"]))
    mfl_sidecar = parse_canonical(read_identity(root, registration["mfl"]["target_sidecar"]))
    confirmation_bytes = {
        name: read_identity(root, identity)
        for name, identity in context["confirmation_registration"].items()
        if isinstance(identity, dict) and set(identity) == {"path", "sha256"}
    }
    if len(confirmation_bytes) != 3:
        raise ValueError("confirmation registration outcomes are incomplete")
    return {
        "ktc": ktc,
        "mfl": mfl,
        "actual_dates": {
            "ktc_observations": ktc_sidecar["observation_dates"],
            "mfl_start": mfl_sidecar["start_cutoff"],
            "mfl_end": mfl_sidecar["end_cutoff"],
            "timezone": mfl_sidecar["timezone"],
        },
        "coverage": {
            "ktc": dict(ktc.attrs.get("coverage", {})),
            "mfl": dict(mfl.attrs.get("coverage", {})),
        },
    }


def _prediction(model, target: pd.DataFrame) -> pd.DataFrame:
    ordered = target.sort_values(["season", "player_id"], kind="stable")
    predicted = model.predict(ordered) if len(ordered) else pd.Series(dtype=float)
    return pd.DataFrame(
        {
            "player_id": ordered["player_id"].to_numpy(),
            "season": ordered["season"].to_numpy(),
            "fitted_start": ordered["start_value"].to_numpy(dtype=float),
            "predicted_end": predicted,
        }
    )


def _fit_evaluation(context: dict, outcomes: dict, dependencies: dict) -> dict:
    manifest = context["manifest"]
    predictions = {}
    fits = {}
    for source, scale in (
        ("ktc", manifest["model"]["ktc_scale_max"]),
        ("mfl", manifest["model"]["mfl_scale_max"]),
    ):
        training = context["training_frames"][source]
        fitted = dependencies["model_factory"](
            label_source=source, scale_max=float(scale)
        ).fit(training)
        predictions[source] = _prediction(fitted, outcomes[source])
        fits[source] = {
            "label_source": source,
            "n_train": len(training),
            "train_seasons": sorted(int(value) for value in training["season"].unique()),
            "scale_max": scale,
        }
    try:
        cohort, coverage = dependencies["matched_cohort"](
            context["starts"], outcomes["ktc"], outcomes["mfl"], predictions
        )
    except ValueError as exc:
        raise ProtocolFailure(str(exc)) from exc
    metrics = dependencies["score"](cohort)
    diagnostic = dependencies["diagnostics"](cohort)
    evaluated_gate = dependencies["gate"](metrics)
    return {
        "fits": fits,
        "cohort_frame": cohort,
        "coverage": coverage,
        "metrics": metrics,
        "diagnostics": diagnostic,
        "gate": evaluated_gate,
    }


def _payload(context: dict, outcomes: dict, fitted: dict) -> dict:
    manifest = context["manifest"]
    diagnostic = dict(fitted["diagnostics"])
    diagnostic["bootstrap_statistic"] = "paired raw MFL-minus-KTC MAE difference"
    cohort = fitted["cohort_frame"]
    rows = cohort[COHORT_COLUMNS].to_dict("records") if not cohort.empty else []
    evaluated_gate = fitted["gate"]
    return {
        "protocol": {
            "name": manifest["protocol"],
            "schema_version": manifest["schema_version"],
            "season": 2026,
            "sealing_commit": manifest["sealing_commit"],
            "sealed_at": manifest["sealed_at"],
        },
        "input_identities": {
            "manifest_sha256": context["manifest_sha256"],
            "registry_sha256": context["registry_sha256"],
            "registration_sha256": context["registration_sha256"],
            "joint_review_sha256": context["joint_review_sha256"],
            "confirmation_manifest_sha256": context["confirmation_manifest_sha256"],
            "confirmation_registry_sha256": context["confirmation_registry_sha256"],
            "confirmation_registration_sha256": context["confirmation_registration_sha256"],
        },
        "registry": context["registry"],
        "registration": context["registration"],
        "confirmation_registry": context["confirmation_registry"],
        "confirmation_registration": context["confirmation_registration"],
        "joint_review": context["joint_review"],
        "source_identities": manifest["source_identities"],
        "archive_identities": manifest["training"],
        "start_identity": manifest["starts"],
        "recipe_identities": manifest["recipes"],
        "model": manifest["model"],
        "cutoffs": manifest["cutoffs"],
        "actual_dates": outcomes["actual_dates"],
        "filters": manifest["filters"],
        "fits": fitted["fits"],
        "cohort": {"columns": COHORT_COLUMNS, "rows": rows},
        "coverage": {"outcomes": outcomes["coverage"], "matched": fitted["coverage"]},
        "metrics": fitted["metrics"],
        "diagnostics": diagnostic,
        "gate": evaluated_gate,
        "decision": {
            "recommendation": evaluated_gate["decision"],
            "promotion_authorized": False,
        },
    }


def run(root: Path, dependencies: dict | None = None) -> dict:
    """Run the fixed one-shot protocol; dependency overrides are test-only."""
    configured = _defaults()
    if dependencies is not None:
        configured.update(dependencies)
    root = Path(root)
    try:
        context = configured["preflight"](root)
    except ValueError as exc:
        raise ProtocolFailure(str(exc)) from exc

    result = {}

    def validate(bound):
        try:
            return configured["validate_outcomes"](bound, configured, root)
        except ValueError as exc:
            raise ProtocolFailure(str(exc)) from exc

    def fit(bound, outcomes):
        evaluated = configured["fit_evaluation"](bound, outcomes, configured)
        result["outcomes"] = outcomes
        result["fitted"] = evaluated
        return _payload(bound, outcomes, evaluated)

    def render(payload):
        raw = _render_report(payload)
        result["wrapper"] = json.loads(raw)
        return raw

    configured["run_claimed_stages"](
        context, validate_outcomes=validate, fit=fit, write_report=render
    )
    return result["wrapper"]


def main() -> int:
    try:
        run(Path(__file__).resolve().parents[2])
    except ProtocolFailure as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - fixed CLI classification boundary
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="python -m predictor_v5.ktc_replacement_2026")
    parser.parse_args()
    raise SystemExit(main())
