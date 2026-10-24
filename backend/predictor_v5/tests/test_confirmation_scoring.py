from pathlib import Path

import pandas as pd
import pytest

from predictor_v5 import confirmation, model


def test_confirmation_preserves_the_serving_model_constants():
    # Break caught: confirmation changes the predictor that production requests use.
    assert model.FEATURES == ["log_start", "age", "points", "position_code"]
    assert model.MONOTONIC == [0, 0, 1, 0]
    assert model.HGB_PARAMS == {
        "loss": "absolute_error",
        "max_iter": 300,
        "learning_rate": 0.05,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0,
        "random_state": 0,
    }
    assert model.POINTS_GRID_STEP == 1.0


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
    training = {
        "baseline": {"raw": raw, "filtered": raw.iloc[:2], "joined": raw.iloc[:1], "trainable": raw.iloc[:1]},
        "candidate": {"raw": raw.iloc[:2], "filtered": raw.iloc[:1], "joined": empty, "trainable": empty},
    }
    coverage = confirmation.coverage_report(
        {name: stages["raw"] for name, stages in training.items()},
        {name: stages["filtered"] for name, stages in training.items()},
        {name: stages["joined"] for name, stages in training.items()},
        {name: stages["trainable"] for name, stages in training.items()},
        empty, raw.iloc[:1], raw.iloc[1:2],
    )
    assert coverage["training"]["baseline"] == {
        "raw": {"total": 3, "by_season": {"2021": 2, "2022": 1}, "by_position": {"QB": 2, "WR": 1}},
        "filtered": {"total": 2, "by_season": {"2021": 2}, "by_position": {"QB": 2}},
        "joined": {"total": 1, "by_season": {"2021": 1}, "by_position": {"QB": 1}},
        "trainable": {"total": 1, "by_season": {"2021": 1}, "by_position": {"QB": 1}},
    }
    assert coverage["training"]["candidate"]["raw"]["total"] == 2
    assert coverage["training"]["candidate"]["trainable"]["total"] == 0
    assert coverage["target"]["matched"]["total"] == 1
    assert coverage["target"]["missing"]["total"] == 1
    assert set(coverage) == {"training", "target", "used_for_configuration", "used_for_promotion"}
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


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("field", ["n", "no_change_mae", "skill"])
def test_gate_retains_for_non_finite_overall_metrics(field, bad):
    positions = ("QB", "RB", "TE", "WR")
    baseline = {"overall": _summary(0.10), "by_position": {p: _summary(0.10) for p in positions}}
    candidate = {"overall": _summary(0.13), "by_position": {p: _summary(0.13) for p in positions}}
    candidate["overall"][field] = bad

    gate = confirmation.evaluate_gate(baseline, candidate, positions)

    assert gate["passed"] is False
    assert any(reason.startswith(f"invalid_overall_{field}:candidate") for reason in gate["reasons"])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("field", ["n", "no_change_mae", "skill"])
def test_gate_retains_for_non_finite_position_metrics(field, bad):
    positions = ("QB", "RB", "TE", "WR")
    baseline = {"overall": _summary(0.10), "by_position": {p: _summary(0.10) for p in positions}}
    candidate = {"overall": _summary(0.13), "by_position": {p: _summary(0.13) for p in positions}}
    candidate["by_position"]["QB"][field] = bad

    gate = confirmation.evaluate_gate(baseline, candidate, positions)

    assert gate["passed"] is False
    assert any(reason.startswith(f"invalid_position_{field}:candidate:QB") for reason in gate["reasons"])


def test_gate_retains_when_derived_gain_or_position_delta_overflows():
    positions = ("QB", "RB", "TE", "WR")
    baseline = {"overall": _summary(-1e308), "by_position": {p: _summary(-1e308) for p in positions}}
    candidate = {"overall": _summary(1e308), "by_position": {p: _summary(1e308) for p in positions}}

    gate = confirmation.evaluate_gate(baseline, candidate, positions)

    assert gate["passed"] is False
    assert "invalid_gain" in gate["reasons"]
    assert "invalid_position_delta:QB" in gate["reasons"]


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
        target_sidecar=tmp_path / "target.json", target_stats=tmp_path / "target-stats.zip", output_path=tmp_path / "out.json",
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


def test_coverage_filtered_stage_uses_loader_dropna_semantics(tmp_path, monkeypatch):
    context, stats = _context(tmp_path)
    baseline = pd.read_csv(context.baseline_csv, dtype={"player_id": str})
    invalid = baseline.iloc[[0]].assign(player_id=None, season=None)
    pd.concat([baseline, invalid], ignore_index=True).to_csv(context.baseline_csv, index=False)

    class FakeModel:
        def __init__(self, label_source, scale_max):
            self.label_source = label_source

        def fit(self, frame):
            return self

        def predict(self, frame):
            return frame["start_value"].to_numpy(float)

    monkeypatch.setattr(confirmation, "WhatIfModel", FakeModel)
    monkeypatch.setattr(confirmation, "read_season_stats", lambda _: stats)

    coverage = confirmation.fit_and_score(context)["diagnostics"]["coverage"]["training"]

    assert coverage["baseline"]["raw"]["total"] == 21
    assert coverage["baseline"]["filtered"]["total"] == 20
    assert coverage["baseline"]["joined"]["total"] == 20
    assert coverage["baseline"]["trainable"]["total"] == 20
    assert coverage["candidate"]["raw"]["total"] == 20
    assert coverage["candidate"]["filtered"]["total"] == 20


def test_missing_walk_forward_history_refuses_before_any_fit(tmp_path, monkeypatch):
    context, _stats = _context(tmp_path)
    dataset = pd.read_csv(context.baseline_csv)
    dataset = dataset[dataset.season != 2023]
    fits = []
    monkeypatch.setattr(confirmation.WhatIfModel, "fit", lambda self, frame: fits.append(frame))
    with pytest.raises(confirmation.PreflightError, match="baseline.*coverage.*2023"):
        confirmation._fit_label_set("baseline", dataset, context.target_frame, context.train_seasons,
                                    context.manifest["walk_forward"]["folds"])
    assert fits == []
