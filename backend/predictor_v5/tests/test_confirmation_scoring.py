from pathlib import Path

import pandas as pd
import pytest

from predictor_v5 import confirmation


def _scored(errors, residuals=None):
    residuals = errors if residuals is None else residuals
    return pd.DataFrame(
        {
            "player_id": [str(i) for i in range(len(errors))],
            "season": [2026] * len(errors),
            "position": ["QB"] * len(errors),
            "start_value": [100.0] * len(errors),
            "end_value": [100.0] * len(errors),
            "pred_value": [100.0 + value for value in residuals],
            "abs_error": errors,
            "no_change_abs_error": [10.0] * len(errors),
            "residual": residuals,
        }
    )


def test_score_predictions_applies_relative_change_to_shared_target_scale():
    predictions = pd.DataFrame(
        [{"player_id": "7", "season": 2026, "position": "QB", "start_value": 200.0, "end_value": 1.0, "pred_value": 300.0}]
    )
    target = pd.DataFrame(
        [{"player_id": "7", "season": 2026, "position": "WR", "start_value": 1000.0, "end_value": 1400.0}]
    )

    got = confirmation.score_predictions(predictions, target, [("7", 2026)])

    assert list(got.columns) == [
        "player_id", "season", "position", "start_value", "end_value", "pred_value",
        "abs_error", "no_change_abs_error", "residual",
    ]
    assert got.to_dict("records") == [{
        "player_id": "7", "season": 2026, "position": "WR", "start_value": 1000.0,
        "end_value": 1400.0, "pred_value": 1500.0, "abs_error": 100.0,
        "no_change_abs_error": 400.0, "residual": 100.0,
    }]


def test_summarize_metrics_reports_distribution_and_non_promotional_tags():
    got = confirmation.summarize_metrics(_scored([1.0, 2.0, 9.0], [-1.0, 2.0, 9.0]))

    assert got["n"] == 3
    assert got["mae"] == 4.0
    assert got["median_absolute_error"] == 2.0
    assert got["p90_absolute_error"] == pytest.approx(7.6)
    assert got["signed_residual_mean"] == pytest.approx(10 / 3)
    assert set(got["residual_summary"]) == {"min", "p10", "median", "p90", "max"}
    assert got["used_for_configuration"] is False
    assert got["used_for_promotion"] is False


def test_coverage_and_bootstrap_have_stable_shapes_and_tags():
    raw = pd.DataFrame([
        {"season": 2021, "position": "QB"}, {"season": 2021, "position": "QB"},
        {"season": 2022, "position": "WR"},
    ])
    empty = raw.iloc[0:0]
    coverage = confirmation.coverage_report(raw, raw.iloc[:2], raw.iloc[:1], raw.iloc[:1], empty, raw.iloc[:1], empty)
    assert coverage["raw"] == {"total": 3, "by_season": {"2021": 2, "2022": 1}, "by_position": {"QB": 2, "WR": 1}}
    assert set(coverage) == {"raw", "filtered", "joined", "trainable", "one_sided", "matched", "missing", "used_for_configuration", "used_for_promotion"}
    assert coverage["used_for_configuration"] is coverage["used_for_promotion"] is False

    paired = _scored([1.0, 2.0, 3.0]).rename(columns={"abs_error": "baseline_abs_error"})
    paired["candidate_abs_error"] = [0.0, 2.0, 5.0]
    first = confirmation.bootstrap_delta(paired, seed=9, resamples=100, percentile=(5.0, 95.0))
    second = confirmation.bootstrap_delta(paired.sample(frac=1, random_state=3), seed=9, resamples=100, percentile=(5.0, 95.0))
    assert first == second
    assert first["mean_delta"] == pytest.approx(1 / 3)
    assert first["median_delta"] == 0.0
    assert (first["candidate_wins"], first["ties"], first["candidate_losses"]) == (1, 1, 1)
    assert set(first["interval"]) == {"lower", "upper", "percentile"}
    assert first["used_for_configuration"] is first["used_for_promotion"] is False


def _summary(skill, n=30, no_change_mae=10.0):
    return {"n": n, "model_mae": None if skill is None else no_change_mae * (1 - skill), "no_change_mae": no_change_mae, "skill": skill}


@pytest.mark.parametrize(
    "gain,drop,count,want",
    [(0.02, -0.05, 30, True), (0.019999999, -0.05, 30, False), (0.02, -0.050000001, 30, False), (0.02, -0.05, 29, False)],
)
def test_gate_uses_inclusive_full_precision_boundaries(gain, drop, count, want):
    baseline = {"overall": _summary(0.10), "by_position": {p: _summary(0.10, count) for p in ("QB", "RB", "TE", "WR")}}
    candidate = {"overall": _summary(0.10 + gain), "by_position": {p: _summary(0.10 + drop, count) for p in ("QB", "RB", "TE", "WR")}}

    assert confirmation.evaluate_gate(baseline, candidate, ("QB", "RB", "TE", "WR"))["passed"] is want


@pytest.mark.parametrize("case", ["empty", "zero_mae", "missing_skill", "missing_position"])
def test_gate_retains_when_evidence_is_insufficient(case):
    positions = ("QB", "RB", "TE", "WR")
    baseline = {"overall": _summary(0.10), "by_position": {p: _summary(0.10) for p in positions}}
    candidate = {"overall": _summary(0.13), "by_position": {p: _summary(0.13) for p in positions}}
    if case == "empty":
        baseline["overall"] = _summary(None, n=0, no_change_mae=0.0)
        candidate["overall"] = _summary(None, n=0, no_change_mae=0.0)
    elif case == "zero_mae":
        baseline["overall"] = _summary(None, no_change_mae=0.0)
        candidate["overall"] = _summary(None, no_change_mae=0.0)
    elif case == "missing_skill":
        candidate["overall"] = _summary(None)
    else:
        del candidate["by_position"]["TE"]

    gate = confirmation.evaluate_gate(baseline, candidate, positions)
    assert gate["passed"] is False
    assert gate["reasons"]


def _context(tmp_path: Path):
    rows = []
    for season in range(2021, 2026):
        for position in ("QB", "RB", "TE", "WR"):
            rows.append({"player_id": f"{season}-{position}", "season": season, "position": position, "age": 25.0, "points": 100.0, "start_value": 100.0, "end_value": 100.0, "start_trades_window": 5, "end_trades_window": 5})
    baseline = pd.DataFrame(rows)
    candidate = baseline.copy()
    baseline_path, candidate_path = tmp_path / "baseline.csv", tmp_path / "candidate.csv"
    baseline.to_csv(baseline_path, index=False)
    candidate.to_csv(candidate_path, index=False)
    target_rows = []
    for position in ("QB", "RB", "TE", "WR"):
        for i in range(30):
            target_rows.append({"player_id": f"target-{position}-{i:02d}", "season": 2026, "position": position, "age": 25.0, "points": 100.0, "start_value": 1000.0, "end_value": 900.0, "start_trades_window": 5, "end_trades_window": 5})
    target = pd.DataFrame(target_rows)
    context = confirmation.PreflightContext(
        manifest={"gate": {"minimum_gain": 0.02, "minimum_position_count": 30, "maximum_position_drop": 0.05}, "walk_forward": {"folds": {"2022": [2021], "2023": [2021, 2022], "2024": [2021, 2022, 2023], "2025": [2021, 2022, 2023, 2024]}}, "bootstrap": {"seed": 20261024, "resamples": 10000, "percentiles": [2.5, 97.5]}},
        manifest_sha256="a" * 64, input_registration={}, input_registration_sha256="b" * 64,
        holdout_registry={}, holdout_registry_sha256="c" * 64, baseline_csv=baseline_path,
        candidate_csv=candidate_path, training_archive=tmp_path / "stats.zip", target_csv=tmp_path / "target.csv",
        target_sidecar=tmp_path / "target.json", output_path=tmp_path / "out.json",
        train_seasons=(2021, 2022, 2023, 2024, 2025), positions=("QB", "RB", "TE", "WR"),
        label_filter={"positive_values": True, "min_trades": 5}, target_recipe={}, target_loader=lambda *_: target,
    )
    context.target_frame = target
    return context, baseline[["player_id", "season", "position", "age", "points"]]


def test_fit_and_score_uses_exact_windows_common_target_and_is_deterministic(tmp_path, monkeypatch):
    context, stats = _context(tmp_path)
    calls = []

    class FakeModel:
        def __init__(self, label_source, scale_max):
            self.label_source, self.scale_max = label_source, scale_max

        def fit(self, frame):
            calls.append((self.label_source, tuple(sorted(frame.season.unique())), len(frame)))
            return self

        def predict(self, frame):
            ratio = 1.0 if self.label_source == "baseline" else 0.9
            return frame["start_value"].to_numpy(float) * ratio

    monkeypatch.setattr(confirmation, "WhatIfModel", FakeModel)
    monkeypatch.setattr(confirmation, "read_season_stats", lambda _: stats)
    first = confirmation.fit_and_score(context)
    context.target_frame = context.target_frame.sample(frac=1, random_state=4)
    second = confirmation.fit_and_score(context)

    expected_windows = [(2021,), (2021, 2022), (2021, 2022, 2023), (2021, 2022, 2023, 2024), (2021, 2022, 2023, 2024, 2025)]
    assert [entry[1] for entry in calls[:5]] == expected_windows
    assert [entry[1] for entry in calls[5:10]] == expected_windows
    assert first == second
    assert first["fits"]["baseline"]["train_seasons"] == [2021, 2022, 2023, 2024, 2025]
    assert first["fits"]["candidate"]["train_seasons"] == [2021, 2022, 2023, 2024, 2025]
    assert first["cohort"]["n"] == 120
    assert first["cohort"]["baseline"]["overall"]["model_mae"] == 100.0
    assert first["cohort"]["candidate"]["overall"]["model_mae"] == 0.0
    assert first["gate"]["passed"] is True
    assert first["decision"]["status"] == "recommend_promotion"
    assert first["decision"]["promotion_authorized"] is False
    assert first["diagnostics"]["walk_forward"]["folds"] == {"2022": [2021], "2023": [2021, 2022], "2024": [2021, 2022, 2023], "2025": [2021, 2022, 2023, 2024]}
    for diagnostic in first["diagnostics"].values():
        assert diagnostic["used_for_configuration"] is False
        assert diagnostic["used_for_promotion"] is False
