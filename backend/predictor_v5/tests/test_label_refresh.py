"""Tests for the reproducible Backchannel label refresh and model gate."""

import csv
import io
import json
import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import pytest

from predictor_v5 import compare_labels
from predictor_v5.compare_labels import (
    LabelSet,
    apply_gate,
    compare_datasets,
    coverage_by_position,
    coverage_counts,
    holdout_record,
    matched_keys,
    resolve_label_set,
    score_on_target,
    select_training_window,
)
from predictor_v5.label_export import (
    LABEL_EXPORT_COLUMNS,
    build_label_frame,
    build_labels_file,
    file_sha256,
    label_coverage,
    label_identity,
    provenance_path,
    validate_market_export,
)
from predictor_v5.tests.test_predictor_v5 import _synthetic

SQL_PATH = Path(__file__).parents[1] / "sql" / "backchannel_labels.sql"
GATE_POSITIONS = ("QB", "RB", "WR", "TE")


# --- label export -----------------------------------------------------------


def _market_values() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": ["thin", "star", "missing-stats"],
            "season": [2025, 2025, 2025],
            "start_value": [100, 500, 300],
            "end_value": [110, 650, 310],
            "start_evidence": [0.0, 9.5, 1.0],
            "end_evidence": [0.0, 12.5, 1.0],
            "start_trades_window": [0, 8, 1],
            "end_trades_window": [0, 11, 1],
            "end_value_next_board": [90, 610, None],
            "end_evidence_next_board": [0.0, 7.0, None],
        }
    )


def _stats() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": ["star", "thin", "college"],
            "season": [2025, 2025, 2025],
            "position": ["WR", "RB", "QB"],
            "age": [25.0, 23.0, 21.0],
            "games": [17, 0, 0],
            "points": [300.5, 0.0, 0.0],
            "years_exp": [4, 1, 0],
        }
    )


def test_build_label_frame_keeps_thin_rows_and_existing_contract():
    """Dropping zero-trade rows would hide the coverage comparison."""
    out = build_label_frame(_market_values(), _stats())

    assert list(out.columns) == LABEL_EXPORT_COLUMNS
    assert len(LABEL_EXPORT_COLUMNS) == 14
    assert list(out["player_id"]) == ["star", "thin"]
    assert out.loc[out["player_id"] == "thin", "start_trades_window"].item() == 0
    assert out.loc[out["player_id"] == "star", "points"].item() == 300.5


def test_build_label_frame_rejects_duplicate_player_seasons():
    """A duplicate market row would make model weighting depend on join multiplicity."""
    duplicate = pd.concat([_market_values(), _market_values().iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate player-season"):
        build_label_frame(duplicate, _stats())
    with pytest.raises(ValueError, match="duplicate player-season"):
        validate_market_export(duplicate)


def test_build_label_frame_applies_existing_pre_nfl_filter():
    values = _market_values().iloc[:2].copy()
    college_value = values.iloc[[0]].assign(player_id="college", start_value=250)
    values = pd.concat([values, college_value], ignore_index=True)

    out = build_label_frame(values, _stats())

    assert "college" not in set(out["player_id"])


def _with_provenance(values: pd.DataFrame, **overrides) -> pd.DataFrame:
    columns = {
        "model_version": "baseline-hl28-v1",
        "start_generation_id": 10,
        "start_revision": 2,
        "end_generation_id": 11,
        "end_revision": 1,
        "next_generation_id": None,
        "next_revision": None,
        "start_board_rows": 5,
        "end_board_rows": 4,
        "exported_at": "2026-10-22T14:00:00Z",
    }
    columns.update(overrides)
    return values.assign(**columns)


def test_validate_market_export_accepts_missing_optional_next_board():
    validate_market_export(_with_provenance(_market_values()), expected_seasons=(2025,), model_version="baseline-hl28-v1")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda v: v.assign(start_value=[100, None, 300]), "start_value"),
        (lambda v: v.assign(end_value=[110, -1, 310]), "end_value"),
        (lambda v: v.assign(end_trades_window=[0, "x", 1]), "end_trades_window"),
        (lambda v: v.assign(end_value_next_board=[90, -5, None]), "end_value_next_board"),
    ],
)
def test_validate_market_export_rejects_invalid_required_values(change, message):
    with pytest.raises(ValueError, match=message):
        validate_market_export(change(_market_values()))


def test_validate_market_export_fails_loudly_on_missing_season():
    with pytest.raises(ValueError, match=r"no rows for seasons \[2024\]"):
        validate_market_export(_market_values(), expected_seasons=(2024, 2025))


def test_validate_market_export_rejects_mixed_or_wrong_model_versions():
    mixed = _with_provenance(_market_values(), model_version=["baseline-hl28-v1", "baseline-hl28-v3", "baseline-hl28-v1"])
    with pytest.raises(ValueError, match="mixes model versions"):
        validate_market_export(mixed)
    with pytest.raises(ValueError, match="expected baseline-hl28-v3"):
        validate_market_export(_with_provenance(_market_values()), model_version="baseline-hl28-v3")


def test_validate_market_export_requires_one_start_and_end_generation_per_season():
    with pytest.raises(ValueError, match=r"start_generation_id differs within seasons \[2025\]"):
        validate_market_export(_with_provenance(_market_values(), start_generation_id=[10, 12, 10]))
    with pytest.raises(ValueError, match=r"empty provenance values in \['end_generation_id'\]"):
        validate_market_export(_with_provenance(_market_values(), end_generation_id=[11, None, 11]))


def test_label_coverage_counts_exclusions_by_reason():
    values = pd.concat(
        [_market_values(), _market_values().iloc[[1]].assign(player_id="college", start_value=250)],
        ignore_index=True,
    )
    labels = build_label_frame(values, _stats())

    report = label_coverage(values, _stats(), labels, expected_seasons=(2025,))

    season = report["by_season"]["2025"]
    assert season["market_rows"] == 4
    assert season["labelled_rows"] == 2
    assert season["pre_nfl_excluded"] == 1
    assert season["no_nfl_stats"] == 1
    assert season["next_board_missing"] == 1
    assert season["below_min_trades"] == 2  # thin and missing-stats
    assert season["labelled_below_min_trades"] == 1
    assert season["labelled_by_position"] == {"QB": 0, "RB": 1, "WR": 1, "TE": 0}
    assert report["status"] == "warnings"  # QB and TE are empty


def test_label_coverage_treats_season_without_archive_stats_as_expected_gap():
    """2019 has no stats in training-data.zip; that is an exclusion, not an error."""
    extra_ids = ["xQB", "xTE"]
    values = pd.concat(
        [
            _market_values(),
            _market_values().iloc[[1, 1]].assign(player_id=extra_ids),
            _market_values().assign(season=2019),
        ],
        ignore_index=True,
    )
    extra = [_stats().iloc[[0]].assign(player_id=f"x{p}", position=p) for p in ("QB", "TE")]
    stats = pd.concat([_stats(), *extra], ignore_index=True)
    labels = build_label_frame(values, stats)

    report = label_coverage(values, stats, labels, expected_seasons=(2019, 2025))

    assert report["by_season"]["2019"]["training_archive_has_season"] is False
    assert report["by_season"]["2019"]["no_nfl_stats"] == 3
    assert report["warnings"] == []


def test_label_coverage_warns_when_a_season_with_stats_joins_nothing():
    values = _market_values().assign(player_id=["a", "b", "c"])
    report = label_coverage(values, _stats(), build_label_frame(values, _stats()))

    assert report["status"] == "warnings"
    assert "none joined" in report["warnings"][0]


def test_build_labels_file_writes_contract_csv_and_provenance(tmp_path, monkeypatch):
    from predictor_v5 import label_export

    raw = tmp_path / "raw.csv"
    _with_provenance(_market_values()).to_csv(raw, index=False)
    zip_path = tmp_path / "training-data.zip"
    zip_path.write_bytes(b"stub")
    monkeypatch.setattr(label_export, "read_season_stats", lambda path: _stats())

    out = tmp_path / "labels.csv"
    build_labels_file(raw, out, zip_path, expected_seasons=(2025,), model_version="baseline-hl28-v1")

    assert list(pd.read_csv(out).columns) == LABEL_EXPORT_COLUMNS
    report = json.loads(provenance_path(out).read_text())
    assert report["model_version"] == "baseline-hl28-v1"
    assert report["generations"]["2025"] == {
        "start": {"id": 10, "revision": 2},
        "end": {"id": 11, "revision": 1},
        "next": None,
    }
    assert report["exported_at"] == "2026-10-22T14:00:00Z"
    assert len(report["label_file"]["sha256"]) == 64
    assert report["coverage"]["by_season"]["2025"]["start_board_rows"] == 5


# --- evaluation methodology -------------------------------------------------


def test_matched_keys_match_player_and_season():
    """The gate must not compare skill on different test-player populations."""
    baseline = pd.DataFrame({"player_id": ["a", "b", "old"], "season": [2025, 2025, 2024]})
    candidate = pd.DataFrame({"player_id": ["b", "c", "old"], "season": [2025, 2025, 2025]})

    assert matched_keys([baseline, candidate], 2025) == [("b", 2025)]


def test_coverage_counts_reports_surviving_rows_for_each_season():
    frame = pd.DataFrame({"season": [2022, 2022, 2024, 2025, 2025, 2025], "position": ["QB", "WR", "RB", "TE", "TE", "QB"]})

    assert coverage_counts(frame) == {"2022": 2, "2024": 1, "2025": 3}
    assert coverage_by_position(frame)["2025"] == {"QB": 1, "RB": 0, "WR": 0, "TE": 2}


def test_score_on_target_uses_one_ground_truth_for_every_predictor():
    """Two predictors on different scales are judged on the target's values only."""
    target = pd.DataFrame(
        {"player_id": ["a", "b"], "season": [2025, 2025], "position": ["QB", "WR"], "start_value": [100.0, 200.0], "end_value": [120.0, 180.0]}
    )
    same_scale = target.assign(pred_value=[110.0, 190.0])
    other_scale = target.assign(start_value=[1000.0, 2000.0], end_value=[0.0, 0.0], pred_value=[1100.0, 1900.0])
    keys = [("a", 2025), ("b", 2025)]

    first = score_on_target(same_scale, target, keys)
    second = score_on_target(other_scale, target, keys)

    assert first == second  # same relative changes, same target, same score
    assert first["overall"]["no_change_mae"] == 20.0
    assert first["overall"]["model_mae"] == 10.0


def test_score_on_target_refuses_keys_missing_from_either_side():
    target = pd.DataFrame({"player_id": ["a"], "season": [2025], "position": ["QB"], "start_value": [1.0], "end_value": [1.0]})
    with pytest.raises(KeyError):
        score_on_target(target.assign(pred_value=1.0), target, [("a", 2025), ("b", 2025)])


def _fake_holdout(skills: dict, calls: list | None = None, final_season: int = 2025):
    """holdout() stand-in: skill by (label set, window length); predictions echo the test rows."""

    def fake(df, label_source, scale_max, train_seasons, test_season):
        if calls is not None:
            calls.append({"max_season_seen": int(df["season"].max()), "train": list(train_seasons), "test": test_season})
        test = df[df["season"] == test_season].copy()
        skill = skills[(label_source, len(train_seasons), test_season == final_season)]
        # Leave (1 - skill) of the no-change error so summarize() returns `skill`.
        test["pred_value"] = test["start_value"] + skill * (test["end_value"] - test["start_value"])
        return {"predictions": test}

    return fake


def _frame(seasons=range(2020, 2026), n=120, scale=1.0) -> pd.DataFrame:
    rows = []
    for season in seasons:
        for i in range(n):
            rows.append(
                {
                    "player_id": f"p{i}",
                    "season": season,
                    "position": GATE_POSITIONS[i % 4],
                    "age": 25.0,
                    "points": 150.0 + i,
                    "start_value": scale * (1000.0 + 10 * i),
                    "end_value": scale * (1100.0 + 10 * i),
                }
            )
    return pd.DataFrame(rows)


SKILLS = {
    ("baseline", 3, True): 0.40,
    ("baseline", 4, True): 0.41,
    ("baseline", 5, True): 0.42,
    ("baseline", 3, False): 0.40,
    ("baseline", 4, False): 0.39,
    ("candidate", 3, False): 0.45,
    ("candidate", 4, False): 0.50,
    ("candidate", 3, True): 0.44,
    ("candidate", 4, True): 0.43,
    ("candidate", 5, True): 0.60,
    ("model_effect", 4, True): 0.70,
}


def test_selection_only_trains_and_validates_before_the_test_season(monkeypatch):
    calls = []
    monkeypatch.setattr(compare_labels, "holdout", _fake_holdout(SKILLS, calls))

    result = select_training_window(_ls(_frame()), train_years=(3, 4, 5), validation_seasons=(2024,), test_season=2025)

    assert calls and all(call["max_season_seen"] < 2025 for call in calls)
    assert all(max(call["train"]) < call["test"] < 2025 for call in calls)
    assert result["selected_train_years"] == 4
    assert result["candidates"][2] == {
        "train_years": 5,
        "windows": {"2024": [2019, 2020, 2021, 2022, 2023]},
        "eligible": False,
        "reason": "no training rows for [2019]",
    }


def test_selection_rejects_validation_on_or_after_the_test_season():
    with pytest.raises(ValueError, match="must precede"):
        select_training_window(_ls(_frame()), train_years=(3,), validation_seasons=(2025,), test_season=2025)


def test_selection_is_unchanged_when_holdout_outcomes_change():
    """Real model: rewriting every 2025 label leaves the selection byte-identical."""
    frame = _synthetic(n_per_season=60, seasons=range(2020, 2026))
    altered = frame.copy()
    altered.loc[altered["season"] == 2025, "end_value"] *= 7.0

    kwargs = {"train_years": (2, 3, 4), "validation_seasons": (2023, 2024), "test_season": 2025}
    first = select_training_window(_ls(frame), **kwargs)
    second = select_training_window(_ls(altered), **kwargs)

    assert first == second
    assert first["selected_train_years"] in (2, 3)  # 4 needs 2019 for the 2023 fold


def test_selection_prefers_shorter_window_within_one_point(monkeypatch):
    skills = {("candidate", 3, False): 0.441, ("candidate", 4, False): 0.45}
    monkeypatch.setattr(compare_labels, "holdout", _fake_holdout(skills))

    result = select_training_window(_ls(_frame()), train_years=(4, 3), validation_seasons=(2024,), test_season=2025)

    assert result["selected_train_years"] == 3


def _gate_summary(skill: float, per_position: dict[str, tuple[int, float]]) -> dict:
    return {
        "overall": {"n": sum(n for n, _ in per_position.values()), "skill": skill},
        "by_position": {p: {"n": n, "skill": pos_skill} for p, (n, pos_skill) in per_position.items()},
    }


EVEN = {"QB": (40, 0.30), "RB": (40, 0.30), "WR": (40, 0.30), "TE": (40, 0.30)}


@pytest.mark.parametrize(
    ("candidate", "failed"),
    [
        (_gate_summary(0.419, EVEN), "overall_guard"),
        (_gate_summary(0.45, {**EVEN, "TE": (29, 0.30)}), "sample_guard"),
        (_gate_summary(0.45, {**EVEN, "TE": (40, 0.249)}), "position_guard"),
    ],
)
def test_gate_guards_reject(candidate, failed):
    gate = apply_gate(_gate_summary(0.40, EVEN), candidate)

    assert gate[failed] is False
    assert gate["qualified"] is False


def test_gate_passes_at_exactly_two_points():
    gate = apply_gate(_gate_summary(0.40, EVEN), _gate_summary(0.42, {**EVEN, "TE": (30, 0.25)}))

    assert gate["qualified"] is True


def _ls(frame: pd.DataFrame, name: str = "candidate", version: str | None = "baseline-hl28-v1", verified=True):
    return LabelSet(name, frame, version, verified, f"{name}.csv")


def _sets(candidate_version="baseline-hl28-v1", verified=True, seasons=range(2020, 2026)):
    return (
        _ls(_frame(seasons), "baseline", "baseline-hl28-v1", verified),
        _ls(_frame(seasons), "candidate", candidate_version, verified),
    )


def test_compare_datasets_freezes_selection_before_final_evaluation(monkeypatch):
    calls = []
    monkeypatch.setattr(compare_labels, "holdout", _fake_holdout(SKILLS, calls))
    baseline, candidate = _sets()
    model_effect = _ls(_frame(scale=0.75), "model_effect", "baseline-hl28-v3")

    report = compare_datasets(baseline, candidate, model_effect=model_effect)

    # 2025 favours the 5-season window (0.60), but selection only read 2024.
    assert report["selection"]["selected_train_years"] == 4
    assert report["frozen_candidate"]["train_seasons"] == [2021, 2022, 2023, 2024]
    assert report["final_evaluation"]["candidate"]["train_seasons"] == [2021, 2022, 2023, 2024]
    assert report["final_evaluation"]["matched_test_rows"] == 120
    # Same target, same keys: control and candidate share the no-change error.
    for scores in report["final_evaluation"]["matched"].values():
        assert scores["control"]["overall"]["no_change_mae"] == scores["candidate"]["overall"]["no_change_mae"]
    assert report["gate"]["evaluated"] is True
    assert report["decision"]["action"] == "update"
    assert report["decision"]["gate_passed"] is True
    assert report["decision"]["promotion_eligible"] is True
    # 2025 is a reused holdout, so even a pass is not enough evidence, and nothing is ever deployed.
    assert report["decision"]["evidence_sufficient_for_promotion"] is False
    assert report["decision"]["promotion_authorized"] is False
    assert report["decision"]["requires_unseen_season_confirmation"] is True
    assert report["final_evaluation"]["holdout"]["status"] == "reused"
    assert all(max(fit["train_seasons"]) < fit["test_season"] for fit in report["fits"])
    assert report["experiments"]["model_version"]["gate_eligible"] is False
    assert report["experiments"]["model_version"]["missing_cells"]


def test_compare_datasets_retains_when_the_gate_fails(monkeypatch):
    skills = {**SKILLS, ("candidate", 4, True): 0.41}
    monkeypatch.setattr(compare_labels, "holdout", _fake_holdout(skills))

    report = compare_datasets(*_sets())

    assert report["gate"]["qualified"] is False
    assert report["decision"]["action"] == "retain"


def test_cross_version_comparison_can_never_approve(monkeypatch):
    skills = {**SKILLS, ("candidate", 4, True): 0.95}
    monkeypatch.setattr(compare_labels, "holdout", _fake_holdout(skills))

    report = compare_datasets(*_sets(candidate_version="baseline-hl28-v3"))

    assert report["gate"]["evaluated"] is False
    assert report["decision"]["action"] == "retain"
    assert "not on one known market model version" in report["decision"]["reason"]
    assert report["experiments"]["label_effect"]["status"] == "not_comparable"


def test_holdout_outcomes_cannot_change_the_selected_window(monkeypatch):
    results = []
    for final_skill in (0.10, 0.99):
        skills = {**SKILLS, ("candidate", 3, True): final_skill, ("candidate", 5, True): 1 - final_skill}
        monkeypatch.setattr(compare_labels, "holdout", _fake_holdout(skills))
        report = compare_datasets(*_sets())
        results.append((report["selection"], report["frozen_candidate"]))

    assert results[0] == results[1]


def test_label_effect_holds_window_players_and_target_fixed(monkeypatch):
    monkeypatch.setattr(compare_labels, "holdout", _fake_holdout(SKILLS))

    report = compare_datasets(*_sets())

    rows = {row["train_years"]: row for row in report["experiments"]["label_effect"]["windows"]}
    assert set(rows) == {3, 4, 5}
    for row in rows.values():
        assert row["matched_test_rows"] == 120
        for scores in row["scores"].values():
            assert scores["baseline"]["overall"]["no_change_mae"] == scores["candidate"]["overall"]["no_change_mae"]
    assert rows[5]["skill_gain"]["baseline"] == pytest.approx(0.18)


def test_report_is_deterministic(monkeypatch):
    monkeypatch.setattr(compare_labels, "holdout", _fake_holdout(SKILLS))
    first = json.dumps(compare_datasets(*_sets()), sort_keys=True)
    second = json.dumps(compare_datasets(*_sets()), sort_keys=True)

    assert first == second


def test_fit_predict_refuses_future_training_seasons():
    with pytest.raises(ValueError, match="must all precede"):
        compare_labels._fit_predict(_ls(_frame()), [2024, 2025], 2025, [])


# --- review fixes: provenance, populations, coverage, precision, publishing, holdout status ---

LEGACY = Path(__file__).parents[1] / "data"


def _label_csv(path: Path, scale: float = 1.0) -> Path:
    frame = _frame(range(2023, 2026), n=8, scale=scale).assign(start_trades_window=9, end_trades_window=9)
    frame.to_csv(path, index=False)
    return path


def _manifest(path: Path, entries: dict) -> Path:
    path.write_text(json.dumps({"files": entries}))
    return path


def test_committed_label_files_are_verified_by_hash_not_name(tmp_path):
    expected = {
        "backchannel_labels_2019_2025.csv": "baseline-hl28-v1",
        "backchannel_labels_2019_2025_mfl.csv": "baseline-hl28-v1",
        "backchannel_labels_2019_2025_mfl_v3.csv": "baseline-hl28-v3",
    }
    for name, version in expected.items():
        identity = label_identity(LEGACY / name)
        assert (identity["verified"], identity["model_version"], identity["source"]) == (True, version, "manifest")

    # A copy under another name is still the same file; a different file under the old name is not.
    renamed = tmp_path / "anything.csv"
    shutil.copy(LEGACY / "backchannel_labels_2019_2025_mfl_v3.csv", renamed)
    assert label_identity(renamed)["model_version"] == "baseline-hl28-v3"
    impostor = _label_csv(tmp_path / "backchannel_labels_2019_2025.csv")
    assert label_identity(impostor)["verified"] is False


def test_v3_file_declared_v1_is_rejected():
    frame = _frame()
    with pytest.raises(ValueError, match="is baseline-hl28-v3 by its manifest provenance, not baseline-hl28-v1"):
        resolve_label_set(
            "candidate", LEGACY / "backchannel_labels_2019_2025_mfl_v3.csv", frame, "baseline-hl28-v1"
        )


def test_unknown_label_file_keeps_assertion_but_stays_unverified(tmp_path):
    path = _label_csv(tmp_path / "labels.csv")
    manifest = _manifest(tmp_path / "manifest.json", {})

    label_set, info = resolve_label_set("candidate", path, _frame(), "baseline-hl28-v1", manifest)
    bare, _ = resolve_label_set("candidate", path, _frame(), None, manifest)

    assert (label_set.model_version, label_set.verified) == ("baseline-hl28-v1", False)
    assert info["provenance_problems"] == ["no provenance sidecar and not in the label manifest"]
    assert (bare.model_version, bare.verified) == (None, False)


def _sidecar(csv_path: Path, **overrides) -> Path:
    report = {
        "label_file": {"sha256": file_sha256(csv_path)},
        "model_version": "baseline-hl28-v1",
        "model_version_source": "export",
    }
    report.update(overrides)
    path = provenance_path(csv_path)
    path.write_text(json.dumps(report))
    return path


def test_sidecar_verifies_only_the_file_it_describes(tmp_path):
    path = _label_csv(tmp_path / "labels.csv")
    manifest = _manifest(tmp_path / "manifest.json", {})
    _sidecar(path)
    assert label_identity(path, manifest)["model_version"] == "baseline-hl28-v1"

    with path.open("a") as handle:
        handle.write("extra,2025,1,1,QB,25,1,1,1,1,9,9,,\n")
    stale = label_identity(path, manifest)
    assert stale["verified"] is False
    assert "stale provenance" in stale["problems"][0]


@pytest.mark.parametrize(
    ("sidecar", "manifest_version", "problem"),
    [
        ({"model_version_source": "argument"}, None, "came from 'argument'"),
        ({"model_version": None}, None, "came from 'export', not the export"),
        ({"label_file": {}}, None, "unreadable or incomplete"),
        ({}, "baseline-hl28-v3", "provenance sources disagree"),
        (None, "", "without a model version"),
    ],
)
def test_incomplete_or_conflicting_provenance_is_unverified(tmp_path, sidecar, manifest_version, problem):
    path = _label_csv(tmp_path / "labels.csv")
    if sidecar is not None:
        _sidecar(path, **sidecar)
    entries = {} if manifest_version is None else {file_sha256(path): {"model_version": manifest_version}}
    identity = label_identity(path, _manifest(tmp_path / "manifest.json", entries))

    assert identity["verified"] is False
    assert identity["model_version"] is None
    assert any(problem in p for p in identity["problems"])


def test_unverified_labels_cannot_become_promotion_eligible(monkeypatch):
    monkeypatch.setattr(compare_labels, "holdout", _fake_holdout(SKILLS))

    report = compare_datasets(*_sets(verified=False))

    assert report["decision"]["gate_passed"] is True
    assert report["decision"]["promotion_eligible"] is False
    assert report["decision"]["action"] == "retain"
    assert "unverified" in report["decision"]["reason"]


def test_label_effect_ignores_candidate_only_training_rows():
    """Real model: rows only the candidate has change the operational view, never the isolated one."""
    rng = np.random.default_rng(3)
    base = _synthetic(n_per_season=80, seasons=range(2021, 2026))
    candidate = base.assign(end_value=base["end_value"] * rng.uniform(0.9, 1.1, len(base)))
    extra = _synthetic(n_per_season=40, seasons=range(2021, 2025), seed=9)
    extra = extra.assign(player_id="extra-" + extra["player_id"], end_value=extra["start_value"] * 5)
    sets = [_ls(base, "baseline"), _ls(candidate), _ls(pd.concat([candidate, extra], ignore_index=True))]

    def run(candidate_set, effect=compare_labels.label_effect):
        return effect(sets[0], candidate_set, (3,), 2024, 2025, True, log := [])["windows"][0], log

    plain, _ = run(sets[1])
    padded, padded_log = run(sets[2])

    assert padded["scores"] == plain["scores"]
    assert padded["training_population"]["candidate_only_by_season"] == {"2022": 40, "2023": 40, "2024": 40}
    assert padded["training_population"]["shared_by_season"] == {"2022": 80, "2023": 80, "2024": 80}
    # Both models in the isolated run trained on the same rows per season.
    assert padded_log[0]["train_rows_by_season"] == padded_log[1]["train_rows_by_season"]
    assert {fit["training_population"] for fit in padded_log} == {"shared"}

    operational = run(sets[2], compare_labels._label_and_population_effect)[0]
    assert operational["scores"] != plain["scores"]


def test_missing_control_training_season_fails_instead_of_shortening(monkeypatch):
    monkeypatch.setattr(compare_labels, "holdout", _fake_holdout(SKILLS))
    baseline, candidate = _sets()
    baseline = _ls(baseline.frame[baseline.frame["season"] != 2023], "baseline")

    with pytest.raises(ValueError, match=r"baseline \(baseline.csv\): no training rows for seasons \[2023\]"):
        compare_datasets(baseline, candidate, control_train_seasons=[2022, 2023, 2024])


def test_missing_final_candidate_season_fails(monkeypatch):
    monkeypatch.setattr(compare_labels, "holdout", _fake_holdout({**SKILLS, ("candidate", 2, False): 0.5}))
    baseline, candidate = _sets()
    # Selection validates on 2023 (training 2021-22), then the frozen window 2023-24 lacks 2024.
    thin = candidate.frame[candidate.frame["season"] != 2024]
    with pytest.raises(ValueError, match=r"candidate \(candidate.csv\): no training rows for seasons \[2024\]"):
        compare_datasets(baseline, _ls(thin), train_years=(2,), validation_seasons=(2023,))


def test_fits_record_the_rows_each_model_trained_on(monkeypatch):
    monkeypatch.setattr(compare_labels, "holdout", _fake_holdout(SKILLS))

    report = compare_datasets(*_sets())

    control_fit = next(f for f in report["fits"] if f["label_set"] == "baseline" and f["test_season"] == 2025)
    assert control_fit["train_rows_by_season"] == {"2022": 120, "2023": 120, "2024": 120}
    assert control_fit["test_rows"] == 120


def test_gate_uses_full_precision_gain():
    """0.41951 - 0.40049 = 0.01902: rounds to 0.420 - 0.400 but must fail the 2-point bar."""
    gate = apply_gate(_gate_summary(0.40049, EVEN), _gate_summary(0.41951, EVEN))

    assert gate["overall_guard"] is False
    assert gate["gain"] == pytest.approx(0.01902)


def test_position_guard_uses_full_precision_drop():
    control = _gate_summary(0.40, {**EVEN, "TE": (40, 0.30049)})
    at_limit = apply_gate(_gate_summary(0.40, EVEN), _gate_summary(0.43, {**EVEN, "TE": (40, 0.25)}))
    just_over = apply_gate(control, _gate_summary(0.43, {**EVEN, "TE": (40, 0.25)}))

    assert at_limit["position_guard"] is True
    assert just_over["position_guard"] is False  # 0.05049 drop; rounded it would be 0.050


def test_gate_fails_safely_without_a_no_change_error():
    gate = apply_gate(_gate_summary(0.40, EVEN), _gate_summary(None, EVEN))

    assert gate["qualified"] is False
    assert gate["gain"] is None


@pytest.mark.parametrize(("shorter", "selected"), [(0.4404, 4), (0.4405, 3)])
def test_selection_tie_margin_uses_full_precision(monkeypatch, shorter, selected):
    """0.45049 - 0.4404 = 0.01009 > 1 point, though both round to 0.450 and 0.440."""
    skills = {("candidate", 3, False): shorter, ("candidate", 4, False): 0.45049}
    monkeypatch.setattr(compare_labels, "holdout", _fake_holdout(skills))

    result = select_training_window(_ls(_frame()), train_years=(3, 4), validation_seasons=(2024,), test_season=2025)

    assert result["selected_train_years"] == selected


def test_report_rounds_for_display_and_keeps_exact_skill(monkeypatch):
    skills = {**SKILLS, ("candidate", 4, True): 0.4412345}
    monkeypatch.setattr(compare_labels, "holdout", _fake_holdout(skills))

    report = compare_datasets(*_sets())

    overall = report["final_evaluation"]["matched"]["candidate"]["candidate"]["overall"]
    assert overall["skill"] == 0.441
    assert overall["skill_exact"] == pytest.approx(0.4412345)


def test_holdout_records():
    assert (holdout_record(2025)["status"], holdout_record(2025)["verified"]) == ("reused", True)
    assert holdout_record(2026) == {
        "season": 2026,
        "status": "unknown",
        "evidence": None,
        "source": "none",
        "verified": False,
    }
    declared = holdout_record(2026, {"status": "untouched", "evidence": "sealed before PR #4"})
    assert (declared["status"], declared["source"], declared["verified"]) == ("untouched", "declared", False)
    with pytest.raises(ValueError, match="on record as 'reused'"):
        holdout_record(2025, {"status": "untouched", "evidence": "x"})
    with pytest.raises(ValueError, match="needs evidence"):
        holdout_record(2026, {"status": "untouched"})


FUTURE_SKILLS = {
    ("baseline", 4, True): 0.40,
    ("candidate", 3, False): 0.45,
    ("candidate", 4, False): 0.50,
    ("candidate", 5, False): 0.40,
    ("candidate", 4, True): 0.45,
    ("candidate", 3, True): 0.40,
    ("candidate", 5, True): 0.40,
    ("baseline", 3, True): 0.40,
    ("baseline", 5, True): 0.40,
    ("baseline", 3, False): 0.40,
    ("baseline", 4, False): 0.40,
    ("baseline", 5, False): 0.40,
}


def _future_report(declared=None):
    return compare_datasets(
        *_sets(seasons=range(2020, 2027)),
        train_end=2025,
        test_season=2026,
        validation_seasons=(2025,),
        control_train_seasons=[2022, 2023, 2024, 2025],
        holdout_declaration=declared,
    )


@pytest.mark.parametrize(
    ("declared", "status"),
    [(None, "unknown"), ({"status": "untouched", "evidence": "sealed"}, "untouched")],
)
def test_unverified_future_season_never_suffices_for_promotion(monkeypatch, declared, status):
    """`--holdout-status untouched --holdout-evidence sealed` alone is a claim, not verified evidence."""
    monkeypatch.setattr(compare_labels, "holdout", _fake_holdout(FUTURE_SKILLS, final_season=2026))

    report = _future_report(declared)

    holdout = report["final_evaluation"]["holdout"]
    assert (holdout["status"], holdout["verified"]) == (status, False)
    assert "PR #3" not in json.dumps(holdout)
    assert report["decision"]["gate_passed"] is True
    assert report["decision"]["promotion_eligible"] is True
    assert report["decision"]["holdout_verified"] is False
    assert report["decision"]["evidence_sufficient_for_promotion"] is False
    assert report["decision"]["requires_unseen_season_confirmation"] is True
    assert report["decision"]["promotion_authorized"] is False


def test_registry_verified_untouched_season_suffices_but_never_authorizes(monkeypatch):
    monkeypatch.setattr(compare_labels, "holdout", _fake_holdout(FUTURE_SKILLS, final_season=2026))
    monkeypatch.setitem(
        compare_labels.KNOWN_HOLDOUTS, 2026, {"status": "untouched", "evidence": "sealed in commit abc123"}
    )

    report = _future_report()

    assert report["final_evaluation"]["holdout"]["source"] == "registry"
    assert report["decision"]["holdout_verified"] is True
    assert report["decision"]["evidence_sufficient_for_promotion"] is True
    assert report["decision"]["requires_unseen_season_confirmation"] is False
    assert report["decision"]["promotion_authorized"] is False


def _old_pair(tmp_path) -> tuple[Path, Path]:
    out = tmp_path / "labels.csv"
    out.write_text("old csv\n")
    provenance_path(out).write_text("old provenance\n")
    return out, provenance_path(out)


def _build(tmp_path, monkeypatch, values=None):
    from predictor_v5 import label_export

    raw = tmp_path / "raw.csv"
    (values if values is not None else _with_provenance(_market_values())).to_csv(raw, index=False)
    zip_path = tmp_path / "training-data.zip"
    zip_path.write_bytes(b"stub")
    monkeypatch.setattr(label_export, "read_season_stats", lambda path: _stats())
    return build_labels_file(raw, tmp_path / "labels.csv", zip_path, expected_seasons=(2025,))


def _assert_untouched(tmp_path, out, sidecar):
    assert out.read_text() == "old csv\n"
    assert sidecar.read_text() == "old provenance\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["labels.csv", "labels.provenance.json", "raw.csv", "training-data.zip"]


def test_failed_coverage_report_leaves_previous_pair(tmp_path, monkeypatch):
    from predictor_v5 import label_export

    out, sidecar = _old_pair(tmp_path)
    monkeypatch.setattr(label_export, "label_coverage", lambda *a, **k: 1 / 0)

    with pytest.raises(ZeroDivisionError):
        _build(tmp_path, monkeypatch)
    _assert_untouched(tmp_path, out, sidecar)


def test_failed_serialization_leaves_previous_pair(tmp_path, monkeypatch):
    from predictor_v5 import label_export

    out, sidecar = _old_pair(tmp_path)
    monkeypatch.setattr(label_export.json, "dumps", lambda *a, **k: (_ for _ in ()).throw(TypeError("boom")))

    with pytest.raises(TypeError, match="boom"):
        _build(tmp_path, monkeypatch)
    _assert_untouched(tmp_path, out, sidecar)


def test_failed_publication_restores_previous_pair(tmp_path, monkeypatch):
    from predictor_v5 import label_export

    out, sidecar = _old_pair(tmp_path)
    real_replace = os.replace
    calls = []

    def flaky_replace(src, dst):
        calls.append(Path(dst).name)
        # Backups (2 calls) and the CSV move succeed; moving the provenance in fails.
        if len(calls) == 4:
            raise OSError("disk full")
        real_replace(src, dst)

    monkeypatch.setattr(label_export.os, "replace", flaky_replace)
    with pytest.raises(OSError, match="disk full"):
        _build(tmp_path, monkeypatch)
    _assert_untouched(tmp_path, out, sidecar)


def test_successful_publication_writes_a_matching_pair(tmp_path, monkeypatch):
    out, sidecar = _old_pair(tmp_path)

    _build(tmp_path, monkeypatch)

    assert list(pd.read_csv(out).columns) == LABEL_EXPORT_COLUMNS
    assert json.loads(sidecar.read_text())["label_file"]["sha256"] == file_sha256(out)
    assert label_identity(out)["model_version"] == "baseline-hl28-v1"
    assert not any(p.name.startswith(".") for p in tmp_path.iterdir())


def test_new_format_export_needs_complete_provenance(tmp_path, monkeypatch):
    partial = _with_provenance(_market_values()).drop(columns=["exported_at"])
    with pytest.raises(ValueError, match=r"not \['exported_at'\]"):
        validate_market_export(partial)
    with pytest.raises(ValueError, match="empty provenance values"):
        validate_market_export(_with_provenance(_market_values(), start_revision=[2, None, 2]))
    with pytest.raises(ValueError, match="next-board generation id without its revision"):
        validate_market_export(_with_provenance(_market_values(), next_generation_id=[5, 5, 5]))


# --- SQL (local disposable Postgres only) -----------------------------------

SEASONS = range(2019, 2026)
CUTOFFS = {
    2019: ("2019-09-05", "2019-12-29"),
    2020: ("2020-09-10", "2021-01-03"),
    2021: ("2021-09-09", "2022-01-09"),
    2022: ("2022-09-08", "2023-01-08"),
    2023: ("2023-09-07", "2024-01-07"),
    2024: ("2024-09-05", "2025-01-05"),
    2025: ("2025-09-04", "2026-01-04"),
}


def _database_url() -> str:
    url = os.getenv("PREDICTOR_V5_TEST_DATABASE_URL")
    if not url:
        pytest.skip("set PREDICTOR_V5_TEST_DATABASE_URL to a disposable local Postgres to run the SQL tests")
    if shutil.which("psql") is None:
        pytest.skip("psql is not installed")
    assert urlparse(url).hostname in {"localhost", "127.0.0.1", "::1"}, "SQL tests require local Postgres"
    return url


def _fixture_sql(version: str = "baseline-hl28-v1", skip: set = frozenset(), extra: str = "", base: int = 100) -> str:
    """Generations for every cutoff; ``skip`` drops (season, 'start'|'end'|'next') generations."""
    gens, values = [], []
    gid = 0
    for season in SEASONS:
        start, end = CUTOFFS[season]
        for kind, gen_season, day in (("start", season, start), ("end", season, end), ("next", season + 1, end)):
            if (season, kind) in skip:
                continue
            gid += 1
            gens.append(f"({gid}, '{gen_season}', '{day}', '{version}', 1)")
            values.append(f"({gid}, 'dynasty', 'superflex', 'p{season}', {base + gid}, 0.5, {gid})")
    return (
        "INSERT INTO market_history_generations VALUES " + ", ".join(gens) + ";\n"
        "INSERT INTO market_history_values VALUES " + ", ".join(values) + ";\n" + extra
    )


def _run_sql(fixture: str, *psql_args: str) -> subprocess.CompletedProcess:
    url = _database_url()
    schema = f"labels_test_{uuid.uuid4().hex[:12]}"
    setup = f"""
        CREATE SCHEMA {schema};
        SET search_path = {schema};
        CREATE TABLE market_history_generations (
          id bigint PRIMARY KEY, season varchar(8), valuation_date date,
          model_version varchar(64), revision integer
        );
        CREATE TABLE market_history_values (
          generation_id bigint, format varchar(8), board varchar(10), asset varchar(64),
          value integer, effective_evidence double precision, trades_window integer
        );
        {fixture}
    """
    subprocess.run(["psql", url, "-v", "ON_ERROR_STOP=1", "-q", "-c", setup], check=True, capture_output=True)
    try:
        env = {**os.environ, "PGOPTIONS": f"-c search_path={schema}"}
        return subprocess.run(
            ["psql", url, "-v", "ON_ERROR_STOP=1", *psql_args, "-f", str(SQL_PATH), "--csv"],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
    finally:
        subprocess.run(["psql", url, "-q", "-c", f"DROP SCHEMA {schema} CASCADE"], check=True, capture_output=True)


def _rows(result: subprocess.CompletedProcess) -> list[dict]:
    assert result.returncode == 0, result.stderr
    return list(csv.DictReader(io.StringIO(result.stdout)))


def test_sql_defaults_to_v1_and_keeps_thin_rows():
    rows = _rows(_run_sql(_fixture_sql()))

    assert [r["season"] for r in rows] == [str(s) for s in SEASONS]
    assert {r["model_version"] for r in rows} == {"baseline-hl28-v1"}
    assert rows[0]["start_trades_window"] == "1"  # thin rows are not filtered
    assert rows[0]["end_value_next_board"] == "103"


def test_sql_selects_v3_explicitly_without_mixing_versions():
    # v1 generations on ids 1..21, v3 on 101..121 with different values.
    fixture = _fixture_sql() + _shift_ids(_fixture_sql("baseline-hl28-v3", base=5000), 100)

    default = _rows(_run_sql(fixture))
    explicit = _rows(_run_sql(fixture, "-v", "model_version=baseline-hl28-v3"))

    assert {r["model_version"] for r in default} == {"baseline-hl28-v1"}
    assert {r["model_version"] for r in explicit} == {"baseline-hl28-v3"}
    assert all(int(r["start_generation_id"]) > 100 and int(r["end_generation_id"]) > 100 for r in explicit)
    assert all(int(r["next_generation_id"]) > 100 for r in explicit)
    assert [r["start_value"] for r in explicit] != [r["start_value"] for r in default]


def _shift_ids(fixture: str, offset: int) -> str:
    """Renumber a ``_fixture_sql`` block's generation ids (and their values' ids) by ``offset``."""
    return re.sub(r"\((\d+), ", lambda m: f"({int(m.group(1)) + offset}, ", fixture)


def test_sql_uses_latest_revision():
    newer = (
        "INSERT INTO market_history_generations VALUES (500, '2019', '2019-09-05', 'baseline-hl28-v1', 2);\n"
        "INSERT INTO market_history_values VALUES (500, 'dynasty', 'superflex', 'p2019', 999, 0.5, 9);\n"
    )
    rows = _rows(_run_sql(_fixture_sql(extra=newer)))

    first = rows[0]
    assert (first["start_generation_id"], first["start_revision"], first["start_value"]) == ("500", "2", "999")


@pytest.mark.parametrize(
    ("skip", "message"),
    [
        ({(2021, "start")}, "no baseline-hl28-v1 generation for season 2021 on its start cutoff 2021-09-09"),
        ({(2025, "end")}, "no baseline-hl28-v1 generation for season 2025 on its end cutoff 2026-01-04"),
    ],
)
def test_sql_fails_loudly_on_missing_required_generation(skip, message):
    result = _run_sql(_fixture_sql(skip=skip))

    assert result.returncode != 0
    assert message in result.stderr


def test_sql_fails_when_every_generation_is_missing():
    """With no rows to join, the guard must still run."""
    result = _run_sql("", "-v", "model_version=baseline-hl28-v9")

    assert result.returncode != 0
    assert "no baseline-hl28-v9 generation for season 2019" in result.stderr


def test_sql_fails_on_tied_latest_revision_and_empty_board():
    tie = "INSERT INTO market_history_generations VALUES (500, '2020', '2020-09-10', 'baseline-hl28-v1', 1);\n"
    tied = _run_sql(_fixture_sql(extra=tie))
    assert tied.returncode != 0
    assert "share the newest revision for season 2020" in tied.stderr

    empty = "DELETE FROM market_history_values WHERE generation_id = 1;\n"
    emptied = _run_sql(_fixture_sql(extra=empty))
    assert emptied.returncode != 0
    assert "start generation 1 for season 2019 has no dynasty superflex values" in emptied.stderr


def test_sql_allows_missing_optional_next_board():
    rows = _rows(_run_sql(_fixture_sql(skip={(2025, "next")})))

    last = rows[-1]
    assert last["season"] == "2025"
    assert last["end_value_next_board"] == ""
    assert last["next_generation_id"] == ""


def test_sql_export_feeds_build_labels(tmp_path, monkeypatch):
    """The psql CSV round-trips through validation and the 14-column contract."""
    from predictor_v5 import label_export

    raw = tmp_path / "raw.csv"
    raw.write_text(_run_sql(_fixture_sql(skip={(2025, "next")})).stdout)
    stats = pd.DataFrame(
        {
            "player_id": [f"p{s}" for s in SEASONS],
            "season": list(SEASONS),
            "position": ["WR"] * 7,
            "age": [25.0] * 7,
            "games": [17] * 7,
            "points": [200.0] * 7,
            "years_exp": [3] * 7,
        }
    )
    monkeypatch.setattr(label_export, "read_season_stats", lambda path: stats)
    zip_path = tmp_path / "training-data.zip"
    zip_path.write_bytes(b"stub")

    labels = build_labels_file(raw, tmp_path / "labels.csv", zip_path, model_version="baseline-hl28-v1")

    assert list(labels.columns) == LABEL_EXPORT_COLUMNS
    assert len(labels) == 7
    assert labels.attrs["provenance"]["generations"]["2025"]["next"] is None


def test_build_labels_file_accepts_exports_without_provenance_columns(tmp_path, monkeypatch):
    """Exports made before the provenance columns still build; the gap is recorded."""
    from predictor_v5 import label_export

    raw = tmp_path / "raw.csv"
    _market_values().to_csv(raw, index=False)
    zip_path = tmp_path / "training-data.zip"
    zip_path.write_bytes(b"stub")
    monkeypatch.setattr(label_export, "read_season_stats", lambda path: _stats())

    labels = build_labels_file(raw, tmp_path / "labels.csv", zip_path, expected_seasons=(2025,))

    report = labels.attrs["provenance"]
    assert report["generations"] is None
    assert report["model_version_source"] == "unknown"
    assert "predates the provenance columns" in report["note"]


@pytest.mark.parametrize(
    ("name", "version"),
    [("backchannel_labels_2019_2025_mfl.csv", "baseline-hl28-v1"), ("backchannel_labels_2019_2025_mfl_v3.csv", "baseline-hl28-v3")],
)
def test_committed_labels_declared_with_their_own_version_are_accepted(name, version):
    label_set, info = resolve_label_set("candidate", LEGACY / name, _frame(), version)

    assert (label_set.model_version, label_set.verified) == (version, True)
    assert info["provenance_source"] == "manifest"
    assert info["sha256"] == file_sha256(LEGACY / name)


def test_unverifiable_file_declared_v1_cannot_be_promoted(monkeypatch, tmp_path):
    """Declaring matching version strings does not stand in for verified provenance."""
    monkeypatch.setattr(compare_labels, "holdout", _fake_holdout(SKILLS))
    path = _label_csv(tmp_path / "relabelled_v3.csv", scale=0.75)
    candidate, _ = resolve_label_set("candidate", path, _frame(), "baseline-hl28-v1", _manifest(tmp_path / "m.json", {}))
    baseline = _ls(_frame(), "baseline")

    report = compare_datasets(baseline, candidate)

    assert report["decision"]["gate_passed"] is True
    assert report["decision"]["action"] == "retain"
    assert report["label_sets"]["candidate"] == {"model_version": "baseline-hl28-v1", "model_version_verified": False}


def test_legacy_export_with_asserted_version_is_never_verified(tmp_path, monkeypatch):
    from predictor_v5 import label_export

    raw = tmp_path / "raw.csv"
    _market_values().to_csv(raw, index=False)
    zip_path = tmp_path / "training-data.zip"
    zip_path.write_bytes(b"stub")
    monkeypatch.setattr(label_export, "read_season_stats", lambda path: _stats())

    out = tmp_path / "labels.csv"
    labels = build_labels_file(raw, out, zip_path, expected_seasons=(2025,), model_version="baseline-hl28-v1")

    report = labels.attrs["provenance"]
    assert (report["model_version"], report["asserted_model_version"]) == (None, "baseline-hl28-v1")
    identity = label_identity(out, _manifest(tmp_path / "m.json", {}))
    assert identity["verified"] is False


def test_malformed_provenance_fails_before_replacing_outputs(tmp_path, monkeypatch):
    out, sidecar = _old_pair(tmp_path)
    partial = _with_provenance(_market_values()).drop(columns=["end_revision"])

    with pytest.raises(ValueError, match="end_revision"):
        _build(tmp_path, monkeypatch, partial)
    _assert_untouched(tmp_path, out, sidecar)


def test_controlled_label_effect_responds_to_shared_label_values():
    """The isolation must not hide real label differences on the shared rows."""
    base = _synthetic(n_per_season=80, seasons=range(2021, 2026))
    shifted = base.copy()
    train = shifted["season"] < 2025
    shifted.loc[train, "end_value"] = shifted.loc[train, "start_value"] * 1.5

    def run(candidate_frame):
        return compare_labels.label_effect(
            _ls(base, "baseline"), _ls(candidate_frame), (3,), 2024, 2025, True, []
        )["windows"][0]["scores"]

    same = run(base.copy())
    changed = run(shifted)

    assert same["baseline"]["baseline"] == same["baseline"]["candidate"]
    assert changed["baseline"]["candidate"] != changed["baseline"]["baseline"]


def test_gate_fails_just_below_two_points():
    gate = apply_gate(_gate_summary(0.40, EVEN), _gate_summary(0.4199999, EVEN))

    assert gate["overall_guard"] is False


def _disjoint(frame: pd.DataFrame, season: int = 2025) -> pd.DataFrame:
    """Rename test-season players so they share no key with the original frame."""
    test = frame["season"] == season
    return frame.assign(player_id=frame["player_id"].where(~test, "other-" + frame["player_id"]))


def test_disjoint_test_populations_cannot_pass_and_serialize_strictly(monkeypatch, tmp_path):
    monkeypatch.setattr(compare_labels, "holdout", _fake_holdout(SKILLS))
    baseline, candidate = _sets()

    report = compare_datasets(baseline, _ls(_disjoint(candidate.frame)))

    final = report["final_evaluation"]
    assert final["status"] == "insufficient_overlap"
    assert final["overlap"] == {"first_rows": 120, "second_rows": 120, "matched_rows": 0}
    assert report["gate"]["evaluated"] is False
    assert report["decision"]["gate_passed"] is False
    assert report["decision"]["action"] == "retain"
    for name in ("label_effect", "label_and_population_effect"):
        for row in report["experiments"][name]["windows"]:
            assert row["status"] == "insufficient_overlap"
            assert row["test_rows"] == {"baseline": 120, "candidate": 120}
    compare_labels.write_report(report, tmp_path / "report.json")  # allow_nan=False: raises on NaN
    assert json.loads((tmp_path / "report.json").read_text())["decision"]["action"] == "retain"


def test_partial_overlap_scores_only_shared_keys(monkeypatch):
    monkeypatch.setattr(compare_labels, "holdout", _fake_holdout(SKILLS))
    baseline, candidate = _sets()
    frame = candidate.frame
    keep = (frame["season"] != 2025) | frame["player_id"].isin([f"p{i}" for i in range(60)])

    report = compare_datasets(baseline, _ls(frame[keep]))

    assert "status" not in report["final_evaluation"]
    assert report["final_evaluation"]["matched_test_rows"] == 60
    assert report["final_evaluation"]["matched"]["baseline"]["control"]["overall"]["n"] == 60


def test_zero_overlap_model_version_experiment_is_informational(monkeypatch):
    monkeypatch.setattr(compare_labels, "holdout", _fake_holdout(SKILLS))
    other = _ls(_disjoint(_frame(scale=0.75)), "model_effect", "baseline-hl28-v3")

    report = compare_datasets(*_sets(), model_effect=other)

    experiment = report["experiments"]["model_version"]
    assert experiment["status"] == "insufficient_overlap"
    assert experiment["gate_eligible"] is False
    assert report["decision"]["gate_passed"] is True  # primary evaluation unaffected


def test_model_version_experiment_skips_missing_test_season(monkeypatch):
    monkeypatch.setattr(compare_labels, "holdout", _fake_holdout(SKILLS))
    no_2025 = _frame(scale=0.75)
    other = _ls(no_2025[no_2025["season"] != 2025], "model_effect", "baseline-hl28-v3")

    with_other = compare_datasets(*_sets(), model_effect=other)
    without = compare_datasets(*_sets())

    experiment = with_other["experiments"]["model_version"]
    assert experiment["status"] == "insufficient_data"
    assert "no rows for test season 2025" in experiment["reason"]
    assert all(fit["label_set"] != "model_effect" for fit in with_other["fits"])
    for key in ("final_evaluation", "gate", "decision", "selection"):
        assert with_other[key] == without[key]
    assert json.dumps(with_other, sort_keys=True) == json.dumps(
        compare_datasets(*_sets(), model_effect=other), sort_keys=True
    )


def test_empty_summary_has_no_nan():
    from predictor_v5.evaluate import summarize, summarize_exact

    empty = pd.DataFrame(columns=["end_value", "pred_value", "start_value"])
    assert summarize_exact(empty) == {"n": 0, "model_mae": None, "no_change_mae": None, "skill": None}
    assert summarize(empty) == {"n": 0, "model_mae": None, "no_change_mae": None, "skill": None}


def _two_seasons(**overrides) -> pd.DataFrame:
    values = pd.concat([_market_values(), _market_values().assign(season=2024)], ignore_index=True)
    return _with_provenance(values, **overrides)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"start_revision": [2, 3, 2, 2, 2, 2]}, r"start_revision differs within seasons \[2025\]"),
        ({"end_board_rows": [4, 4, 5, 4, 4, 4]}, r"end_board_rows differs within seasons \[2025\]"),
        ({"start_board_rows": [0] * 6}, "start_board_rows must be positive whole numbers"),
        ({"end_revision": [1.5] * 6}, "end_revision must be non-negative whole numbers"),
        (
            {"next_generation_id": [7, None, 7, 8, 8, 8], "next_revision": [1, None, 1, 1, 1, 1]},
            r"next-board generation is set for some rows but not others in seasons \[2025\]",
        ),
        ({"exported_at": ["2026-10-22T14:00:00Z"] * 3 + ["2026-10-23T09:00:00Z"] * 3}, "several export timestamps"),
        ({"exported_at": ["yesterday"] * 6}, "not an ISO 8601 timestamp"),
    ],
)
def test_inconsistent_export_provenance_is_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        validate_market_export(_two_seasons(**overrides))


def test_season_without_any_next_board_stays_valid():
    values = _two_seasons(next_generation_id=[None, None, None, 8, 8, 8], next_revision=[None, None, None, 1, 1, 1])

    validate_market_export(values, expected_seasons=(2024, 2025))


def test_inconsistent_provenance_fails_before_publishing(tmp_path, monkeypatch):
    out, sidecar = _old_pair(tmp_path)

    with pytest.raises(ValueError, match="start_revision differs"):
        _build(tmp_path, monkeypatch, _with_provenance(_market_values(), start_revision=[2, 3, 2]))
    _assert_untouched(tmp_path, out, sidecar)


# --- duplicate keys and empty label files -------------------------------------


def _no_fitting(*args, **kwargs):
    raise AssertionError("validation must fail before anything is loaded or fitted")


def _label_files(tmp_path, duplicated=()):
    paths = {}
    for name in ("baseline", "candidate", "model_effect"):
        path = _label_csv(tmp_path / f"{name}.csv")
        if name in duplicated:
            frame = pd.read_csv(path, dtype={"player_id": str})
            # A thin repeat: it must be caught even though --min-trades would drop it.
            repeat = frame.iloc[[0, 1]].assign(start_trades_window=0)
            pd.concat([frame, repeat], ignore_index=True).to_csv(path, index=False)
        paths[name] = path
    return paths


@pytest.mark.parametrize(
    "duplicated",
    [("baseline",), ("candidate",), ("model_effect",), ("baseline", "candidate", "model_effect")],
)
def test_compare_label_files_rejects_duplicate_keys_before_fitting(tmp_path, monkeypatch, duplicated):
    monkeypatch.setattr(compare_labels, "load_season_stats", _no_fitting)
    monkeypatch.setattr(compare_labels, "holdout", _no_fitting)
    paths = _label_files(tmp_path, duplicated)

    with pytest.raises(ValueError) as error:
        compare_labels.compare_label_files(
            paths["baseline"], paths["candidate"], model_effect_path=paths["model_effect"]
        )
    message = str(error.value)
    # The first bad input in baseline, candidate, model_effect order is named.
    first = duplicated[0]
    assert f"duplicate player-season in {first} labels {first}.csv" in message
    assert "2 keys repeated across 4 rows" in message
    assert "e.g. p0/2023, p1/2023" in message


def test_compare_label_files_accepts_unique_keys_unchanged(tmp_path, monkeypatch):
    """Valid inputs reach the comparison exactly as before the duplicate check."""
    monkeypatch.setattr(compare_labels, "holdout", _fake_holdout(SKILLS))
    monkeypatch.setattr(compare_labels, "load_season_stats", lambda path: None)
    monkeypatch.setattr(compare_labels, "build_dataset", lambda stats, labels: _frame())
    monkeypatch.setattr(compare_labels, "file_sha256", lambda path: "0" * 64)
    paths = _label_files(tmp_path)
    manifest = _manifest(tmp_path / "m.json", {})

    report = compare_labels.compare_label_files(
        paths["baseline"], paths["candidate"], model_effect_path=paths["model_effect"], manifest_path=manifest
    )
    direct = compare_datasets(
        *(LabelSet(n, _frame(), None, False, report["inputs"][n]["file"] + ", sha256 000000000000") for n in ("baseline", "candidate")),
        model_effect=LabelSet("model_effect", _frame(), None, False, "model_effect.csv, sha256 000000000000"),
        train_years=(3, 4, 5),
    )

    assert {k: v for k, v in report.items() if k != "inputs"} == direct
    assert set(report["inputs"]) == {"baseline", "candidate", "model_effect", "training_data"}


@pytest.mark.parametrize("name", ["baseline", "candidate", "model_effect"])
def test_compare_datasets_rejects_duplicate_keys_after_the_stats_join(monkeypatch, name):
    """A repeat introduced by the join (or by a direct caller) is caught before any fit."""
    monkeypatch.setattr(compare_labels, "holdout", _no_fitting)
    frames = {n: _frame() for n in ("baseline", "candidate", "model_effect")}
    frames[name] = pd.concat([frames[name], frames[name].iloc[[5]]], ignore_index=True)
    sets = {n: _ls(f, n) for n, f in frames.items()}

    with pytest.raises(ValueError, match=rf"duplicate player-season in {name} \({name}.csv\) dataset: 1 keys .* p5/2020"):
        compare_datasets(sets["baseline"], sets["candidate"], model_effect=sets["model_effect"])


def test_duplicate_error_caps_its_sample():
    from predictor_v5.label_export import _reject_duplicates

    frame = _frame(range(2025, 2026), n=8)
    with pytest.raises(ValueError, match=r"8 keys repeated across 16 rows, e\.g\. (p\d/2025, ){4}p4/2025 \(\+3 more\)$"):
        _reject_duplicates(pd.concat([frame, frame]), "test")


def test_build_labels_refuses_to_publish_an_empty_label_file(tmp_path, monkeypatch):
    out, sidecar = _old_pair(tmp_path)
    unmatched = _with_provenance(_market_values().assign(player_id=["a", "b", "c"]))

    with pytest.raises(ValueError) as error:
        _build(tmp_path, monkeypatch, unmatched)
    message = str(error.value)
    assert "3 market rows joined to 0 NFL stat rows" in message
    assert "3 have no NFL stats for that player-season, 0 are pre-NFL seasons" in message
    assert "e.g. a/2025, b/2025, c/2025" in message
    _assert_untouched(tmp_path, out, sidecar)


def test_build_labels_rejects_a_file_of_only_pre_nfl_rows(tmp_path, monkeypatch):
    out, sidecar = _old_pair(tmp_path)
    college_only = _with_provenance(_market_values().assign(player_id=["college", "x", "y"]))

    with pytest.raises(ValueError, match="2 have no NFL stats for that player-season, 1 are pre-NFL seasons"):
        _build(tmp_path, monkeypatch, college_only)
    _assert_untouched(tmp_path, out, sidecar)


def test_build_labels_keeps_partial_coverage_and_the_2019_gap(tmp_path, monkeypatch):
    """Only an entirely empty result is refused: a season with no archive stats is still fine."""
    from predictor_v5 import label_export

    gap = _market_values().assign(season=2019)
    values = _with_provenance(pd.concat([_market_values(), gap], ignore_index=True))
    raw = tmp_path / "raw.csv"
    values.to_csv(raw, index=False)
    zip_path = tmp_path / "training-data.zip"
    zip_path.write_bytes(b"stub")
    monkeypatch.setattr(label_export, "read_season_stats", lambda path: _stats())

    labels = build_labels_file(raw, tmp_path / "labels.csv", zip_path, expected_seasons=(2019, 2025))

    assert list(labels.columns) == LABEL_EXPORT_COLUMNS
    assert sorted(labels["player_id"]) == ["star", "thin"]
    coverage = labels.attrs["provenance"]["coverage"]["by_season"]
    assert coverage["2019"]["labelled_rows"] == 0
    assert coverage["2019"]["training_archive_has_season"] is False
    assert coverage["2025"]["no_nfl_stats"] == 1


def _joined_values(start, end):
    """Market rows for the two players that join NFL stats (star, thin) plus one that doesn't."""
    return _with_provenance(_market_values().assign(start_value=start, end_value=end))


@pytest.mark.parametrize(
    ("start", "end", "counts"),
    [
        ([0, 0, 300], [110, 650, 310], "2 zero start only, 0 zero end only, 0 both zero"),
        ([100, 500, 300], [0, 0, 310], "0 zero start only, 2 zero end only, 0 both zero"),
        ([0, 500, 300], [110, 0, 310], "1 zero start only, 1 zero end only, 0 both zero"),
        ([0, 0, 300], [0, 650, 310], "1 zero start only, 0 zero end only, 1 both zero"),
    ],
)
def test_build_labels_refuses_a_file_with_no_trainable_rows(tmp_path, monkeypatch, start, end, counts):
    out, sidecar = _old_pair(tmp_path)

    with pytest.raises(ValueError, match=rf"2 joined rows, 0 with start_value and end_value both > 0 \({counts}\)"):
        _build(tmp_path, monkeypatch, _joined_values(start, end))
    _assert_untouched(tmp_path, out, sidecar)


def test_build_labels_keeps_zero_valued_rows_when_one_row_is_trainable(tmp_path, monkeypatch):
    labels = _build(tmp_path, monkeypatch, _joined_values([0, 500, 300], [110, 650, 310]))

    written = pd.read_csv(tmp_path / "labels.csv", dtype={"player_id": str})
    assert list(written.columns) == LABEL_EXPORT_COLUMNS
    assert sorted(written["player_id"]) == ["star", "thin"]
    assert written.set_index("player_id").loc["thin", "start_value"] == 0
    assert len(labels) == 2


def test_committed_label_files_all_have_trainable_rows():
    from predictor_v5.data import has_positive_values

    for path in sorted(LEGACY.glob("backchannel_labels_2019_2025*.csv")):
        frame = pd.read_csv(path)
        assert has_positive_values(frame).sum() > 0, path.name
