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

import pandas as pd
import pytest

from predictor_v5 import compare_labels
from predictor_v5.compare_labels import (
    LabelSet,
    apply_gate,
    compare_datasets,
    coverage_by_position,
    coverage_counts,
    matched_keys,
    score_on_target,
    select_training_window,
)
from predictor_v5.label_export import (
    LABEL_EXPORT_COLUMNS,
    build_label_frame,
    build_labels_file,
    label_coverage,
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
    with pytest.raises(ValueError, match="start generation is not unique"):
        validate_market_export(_with_provenance(_market_values(), start_generation_id=[10, 12, 10]))
    with pytest.raises(ValueError, match="end generation is missing"):
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


def _fake_holdout(skills: dict, calls: list | None = None):
    """holdout() stand-in: skill by (label set, window length); predictions echo the test rows."""

    def fake(df, label_source, scale_max, train_seasons, test_season):
        if calls is not None:
            calls.append({"max_season_seen": int(df["season"].max()), "train": list(train_seasons), "test": test_season})
        test = df[df["season"] == test_season].copy()
        skill = skills[(label_source, len(train_seasons), test_season == 2025)]
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

    result = select_training_window(_frame(), train_years=(3, 4, 5), validation_seasons=(2024,), test_season=2025)

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
        select_training_window(_frame(), train_years=(3,), validation_seasons=(2025,), test_season=2025)


def test_selection_is_unchanged_when_holdout_outcomes_change():
    """Real model: rewriting every 2025 label leaves the selection byte-identical."""
    frame = _synthetic(n_per_season=60, seasons=range(2020, 2026))
    altered = frame.copy()
    altered.loc[altered["season"] == 2025, "end_value"] *= 7.0

    kwargs = {"train_years": (2, 3, 4), "validation_seasons": (2023, 2024), "test_season": 2025}
    first = select_training_window(frame, **kwargs)
    second = select_training_window(altered, **kwargs)

    assert first == second
    assert first["selected_train_years"] in (2, 3)  # 4 needs 2019 for the 2023 fold


def test_selection_prefers_shorter_window_within_one_point(monkeypatch):
    skills = {("candidate", 3, False): 0.441, ("candidate", 4, False): 0.45}
    monkeypatch.setattr(compare_labels, "holdout", _fake_holdout(skills))

    result = select_training_window(_frame(), train_years=(4, 3), validation_seasons=(2024,), test_season=2025)

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


def _sets(candidate_version="baseline-hl28-v1"):
    return (
        LabelSet("baseline", _frame(), "baseline-hl28-v1"),
        LabelSet("candidate", _frame(), candidate_version),
    )


def test_compare_datasets_freezes_selection_before_final_evaluation(monkeypatch):
    calls = []
    monkeypatch.setattr(compare_labels, "holdout", _fake_holdout(SKILLS, calls))
    baseline, candidate = _sets()
    model_effect = LabelSet("model_effect", _frame(scale=0.75), "baseline-hl28-v3")

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
    assert report["decision"]["requires_unseen_season_confirmation"] is True
    assert "not an untouched holdout" in report["final_evaluation"]["holdout_status"]
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
    assert "different market model versions" in report["decision"]["reason"]
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
        compare_labels._fit_predict(_frame(), [2024, 2025], 2025, "candidate", [])


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
