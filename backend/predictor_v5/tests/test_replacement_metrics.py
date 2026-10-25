import copy
import json

import pandas as pd
import pytest

from predictor_v5.replacement_cohort import COHORT_COLUMNS
from predictor_v5.replacement_metrics import diagnostics, gate, score


def cohort(rows):
    return pd.DataFrame(rows, columns=COHORT_COLUMNS)


def row(player, position="QB"):
    return [player, 2026, position, 25, 17, 100, 100, 120, 110, 200, 240, 230]


def test_source_specific_baselines_and_targets():
    metrics = score(cohort([row("a")]))
    assert metrics["overall"]["ktc"] == {"mae": 10.0, "baseline": 20.0, "skill": 0.5}
    assert metrics["overall"]["mfl"] == {"mae": 10.0, "baseline": 40.0, "skill": 0.75}
    assert metrics["overall"]["delta"] == 0.25


def test_gate_exact_boundaries_and_counts():
    def group(count, skill):
        return {"count": count, "ktc": {"mae": 1 - skill, "baseline": 1, "skill": skill},
                "mfl": {"mae": 1, "baseline": 1, "skill": 0}, "delta": -skill}
    metrics = {"overall": group(120, 0.01), "positions": {
        p: group(30, 0.05) for p in ("QB", "RB", "TE", "WR")}}
    assert gate(metrics)["decision"] == "recommend_replacement"
    assert gate(metrics)["promotion_authorized"] is False
    metrics["overall"] = group(120, 0.010000001)
    assert gate(metrics)["decision"] == "retain_ktc"
    metrics["overall"] = group(120, 0.01)
    metrics["positions"]["QB"] = group(30, 0.050000001)
    assert gate(metrics)["decision"] == "retain_ktc"
    metrics["positions"]["QB"] = group(29, 0.05)
    assert gate(metrics)["decision"] == "retain_ktc"


def test_diagnostics_ties_and_non_gating():
    rows = [row("a"), row("b"), row("c")]
    for r, actual, pred in zip(rows, [3, 3, 1], [3, 2, 2]):
        r[7], r[8] = actual, pred
    data = cohort(rows)
    first = diagnostics(data, resamples=20)
    assert first == diagnostics(data, resamples=20)
    assert first["overall"]["ktc"]["rank"]["mean_absolute_rank_error"] == pytest.approx(2 / 3)
    assert first["overall"]["mfl"]["rank"]["spearman"] is None
    assert first["used_for_configuration"] is False
    assert first["used_for_promotion"] is False
    before = gate(score(data))
    first["overall"] = {"malicious": True}
    assert gate(score(data)) == before
    json.dumps(first, allow_nan=False)


def test_empty_and_zero_baseline_retain():
    assert gate(score(cohort([])))["decision"] == "retain_ktc"
    value = row("a")
    value[7] = value[6]
    metrics = score(cohort([value]))
    assert metrics["overall"]["ktc"]["skill"] is None
    assert gate(metrics)["decision"] == "retain_ktc"


@pytest.mark.parametrize("column,value", [("age", True), ("points", float("inf")),
                                            ("ktc_pred", float("nan")), ("season", 2025),
                                            ("position", "K"), ("player_id", "")])
def test_invalid_cohort_rejected(column, value):
    frame = cohort([row("a")]).astype(object)
    frame.loc[0, column] = value
    with pytest.raises(ValueError):
        score(frame)
    with pytest.raises(ValueError):
        diagnostics(frame, resamples=1)


def test_duplicate_and_extra_column_rejected():
    frame = cohort([row("a"), row("a")])
    with pytest.raises(ValueError):
        score(frame)
    with pytest.raises(ValueError):
        score(cohort([row("a")]).assign(extra=1))


@pytest.mark.parametrize("seed,resamples", [(True, 1), (-1, 1), (1.5, 1), (1, 0),
                                              (1, True), (1, 1.5)])
def test_invalid_bootstrap_settings_rejected(seed, resamples):
    with pytest.raises(ValueError):
        diagnostics(cohort([]), seed=seed, resamples=resamples)


def test_overflow_baseline_and_missing_or_invalid_counts_retain():
    value = row("a")
    value[6], value[7] = 1e308, -1e308
    metrics = score(cohort([value]))
    assert metrics["overall"]["ktc"]["baseline"] is None
    assert gate(metrics)["decision"] == "retain_ktc"
    json.dumps(diagnostics(cohort([value]), resamples=2), allow_nan=False)
    for count in [True, -1, 2.5, None]:
        metrics["overall"]["count"] = count
        assert "invalid_overall_count" in gate(metrics)["reasons"]
    assert "qb_missing" in gate({"overall": {"count": 1, "delta": 0}, "positions": {}})["reasons"]


def test_inconsistent_count_and_missing_skill_cannot_pass_gate():
    data = cohort([row(f"{p}{i}", p) for p in ("QB", "RB", "TE", "WR") for i in range(30)])
    metrics = score(data)
    assert gate(metrics)["decision"] == "recommend_replacement"
    metrics["positions"]["QB"]["ktc"]["skill"] = None
    assert gate(metrics)["decision"] == "retain_ktc"
    metrics = score(data)
    metrics["overall"]["count"] = 121
    assert gate(metrics)["decision"] == "retain_ktc"


@pytest.mark.parametrize("path,value", [
    (("overall", "ktc", "baseline"), 0), (("overall", "ktc", "baseline"), -1),
    (("overall", "ktc", "baseline"), float("inf")), (("overall", "ktc", "baseline"), True),
    (("overall", "ktc", "mae"), -1), (("overall", "ktc", "mae"), float("nan")),
    (("overall", "ktc", "mae"), True), (("overall", "ktc", "skill"), 0.7),
    (("overall", "delta"), 0.9), (("overall", "count"), True),
    (("positions", "QB", "mfl", "baseline"), 0),
    (("positions", "RB", "delta"), 1), (("positions", "TE", "count"), False),
])
def test_gate_rejects_contradictory_metrics(path, value):
    data = cohort([row(f"{p}{i}", p) for p in ("QB", "RB", "TE", "WR") for i in range(30)])
    metrics = score(data)
    target = metrics
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    result = gate(metrics)
    assert result["decision"] == "retain_ktc"
    assert result["reasons"]
    assert result == gate(copy.deepcopy(metrics))


@pytest.mark.parametrize("mutation", ["root_extra", "position_extra", "group_extra", "source_extra", "source_missing", "missing_position", "root_list"])
def test_gate_rejects_invalid_shapes(mutation):
    data = cohort([row(f"{p}{i}", p) for p in ("QB", "RB", "TE", "WR") for i in range(30)])
    metrics = score(data)
    if mutation == "root_extra": metrics["extra"] = 1
    elif mutation == "position_extra": metrics["positions"]["K"] = {}
    elif mutation == "group_extra": metrics["overall"]["extra"] = 1
    elif mutation == "source_extra": metrics["overall"]["ktc"]["extra"] = 1
    elif mutation == "source_missing": del metrics["overall"]["ktc"]["mae"]
    elif mutation == "missing_position": del metrics["positions"]["QB"]
    else: metrics = []
    assert gate(metrics)["decision"] == "retain_ktc"
