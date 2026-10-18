"""Tests for the reproducible Backchannel label refresh and model gate."""

import asyncio
import os
from pathlib import Path
from urllib.parse import urlparse

import asyncpg
import pandas as pd
import pytest

from predictor_v5 import compare_labels
from predictor_v5.compare_labels import (
    compare_datasets,
    coverage_counts,
    matched_test_ids,
    select_candidate,
)
from predictor_v5.label_export import LABEL_EXPORT_COLUMNS, build_label_frame


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
    assert list(out["player_id"]) == ["star", "thin"]
    assert out.loc[out["player_id"] == "thin", "start_trades_window"].item() == 0
    assert out.loc[out["player_id"] == "star", "points"].item() == 300.5


def test_build_label_frame_rejects_duplicate_player_seasons():
    """A duplicate market row would make model weighting depend on join multiplicity."""
    duplicate = pd.concat([_market_values(), _market_values().iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate player-season"):
        build_label_frame(duplicate, _stats())


def test_build_label_frame_applies_existing_pre_nfl_filter():
    values = _market_values().iloc[:2].copy()
    college_value = values.iloc[[0]].assign(player_id="college", start_value=250)
    values = pd.concat([values, college_value], ignore_index=True)

    out = build_label_frame(values, _stats())

    assert "college" not in set(out["player_id"])


def test_matched_test_ids_uses_only_players_in_both_2025_cohorts():
    """The gate must not compare skill on different test-player populations."""
    baseline = pd.DataFrame({"player_id": ["a", "b", "old"], "season": [2025, 2025, 2024]})
    candidate = pd.DataFrame({"player_id": ["b", "c", "new"], "season": [2025, 2025, 2024]})

    assert matched_test_ids(baseline, candidate, 2025) == {"b"}


def test_coverage_counts_reports_surviving_rows_for_each_season():
    frame = pd.DataFrame({"season": [2022, 2022, 2024, 2025, 2025, 2025]})

    assert coverage_counts(frame) == {"2022": 2, "2024": 1, "2025": 3}


def test_compare_datasets_reports_all_and_matched_cohorts(monkeypatch):
    old = pd.DataFrame(
        {
            "player_id": ["train-old", "a", "b"],
            "season": [2024, 2025, 2025],
            "position": ["QB", "QB", "RB"],
        }
    )
    new = pd.DataFrame(
        {
            "player_id": ["train-new", "b", "c"],
            "season": [2024, 2025, 2025],
            "position": ["WR", "RB", "TE"],
        }
    )

    def fake_holdout(df, label_source, scale_max, train_seasons, test_season):
        skill = 0.40 if label_source == "old" else {2022: 0.43, 2021: 0.445, 2020: 0.44}[min(train_seasons)]
        positions = {position: {"n": 30, "skill": skill} for position in POSITIONS}
        return {
            "label_source": label_source,
            "train_seasons": train_seasons,
            "n_train": len(df[df["season"].isin(train_seasons)]),
            "test_season": test_season,
            "overall": {"n": len(df[df["season"] == test_season]), "skill": skill},
            "by_position": positions,
            "predictions": pd.DataFrame(),
        }

    monkeypatch.setattr(compare_labels, "holdout", fake_holdout)

    report = compare_datasets(old, new)

    assert report["coverage"] == {
        "old": {"2024": 1, "2025": 2},
        "mfl": {"2024": 1, "2025": 2},
    }
    assert report["matched_test_rows"] == 1
    assert report["baseline"]["all"]["overall"]["n"] == 2
    assert report["baseline"]["matched"]["overall"]["n"] == 1
    assert all(candidate["matched"]["overall"]["n"] == 1 for candidate in report["candidates"])
    assert report["decision"] == {"action": "update", "selected_train_start": 2021}
    assert "predictions" not in report["baseline"]["all"]


def _summary(skill: float, per_position: dict[str, tuple[int, float]]) -> dict:
    return {
        "overall": {"n": sum(n for n, _ in per_position.values()), "skill": skill},
        "by_position": {p: {"n": n, "skill": pos_skill} for p, (n, pos_skill) in per_position.items()},
    }


POSITIONS = {"QB": (40, 0.30), "RB": (40, 0.30), "WR": (40, 0.30), "TE": (40, 0.30)}


def test_select_candidate_applies_gain_position_and_sample_gates():
    """A high overall score must not mask a position regression or tiny cohort."""
    baseline = _summary(0.40, POSITIONS)
    candidates = [
        {"train_start": 2022, "matched": _summary(0.43, {**POSITIONS, "TE": (40, 0.24)})},
        {"train_start": 2021, "matched": _summary(0.45, {**POSITIONS, "TE": (29, 0.30)})},
        {"train_start": 2020, "matched": _summary(0.425, POSITIONS)},
    ]

    decision, evaluated = select_candidate(baseline, candidates)

    assert decision == {"action": "update", "selected_train_start": 2020}
    assert [row["gate"]["qualified"] for row in evaluated] == [False, False, True]
    assert evaluated[0]["gate"]["position_guard"] is False
    assert evaluated[1]["gate"]["sample_guard"] is False


def test_select_candidate_prefers_shorter_window_within_one_skill_point():
    """A marginal holdout difference should not pull older seasons into production."""
    baseline = _summary(0.40, POSITIONS)
    candidates = [
        {"train_start": 2022, "matched": _summary(0.44, POSITIONS)},
        {"train_start": 2021, "matched": _summary(0.451, POSITIONS)},
        {"train_start": 2020, "matched": _summary(0.46, POSITIONS)},
    ]

    decision, _ = select_candidate(baseline, candidates)

    assert decision == {"action": "update", "selected_train_start": 2021}


def test_backchannel_sql_uses_latest_revision_and_keeps_thin_rows():
    """A stale generation or trade filter would change the exported training labels."""
    database_url = os.getenv("PREDICTOR_V5_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("set PREDICTOR_V5_TEST_DATABASE_URL to run the SQL integration test")
    host = urlparse(database_url).hostname
    assert host in {"localhost", "127.0.0.1", "::1"}, "integration test requires local Postgres"

    async def run() -> list[asyncpg.Record]:
        connection = await asyncpg.connect(database_url)
        try:
            await connection.execute(
                """
                CREATE TEMP TABLE market_history_generations (
                  id bigint PRIMARY KEY, season varchar(8), valuation_date date,
                  model_version varchar(64), revision integer
                );
                CREATE TEMP TABLE market_history_values (
                  generation_id bigint, format varchar(8), board varchar(10), asset varchar(64),
                  value integer, effective_evidence double precision, trades_window integer
                );
                INSERT INTO market_history_generations VALUES
                  (1, '2019', '2019-09-05', 'baseline-hl28-v1', 1),
                  (2, '2019', '2019-09-05', 'baseline-hl28-v1', 2),
                  (3, '2019', '2019-12-29', 'baseline-hl28-v1', 1),
                  (4, '2020', '2019-12-29', 'baseline-hl28-v1', 1);
                INSERT INTO market_history_values VALUES
                  (1, 'dynasty', 'superflex', 'thin', 100, 0, 0),
                  (2, 'dynasty', 'superflex', 'thin', 125, 0, 0),
                  (3, 'dynasty', 'superflex', 'thin', 150, 0, 0),
                  (4, 'dynasty', 'superflex', 'thin', 140, 0, 0);
                """
            )
            sql = (Path(__file__).parents[1] / "sql" / "backchannel_labels.sql").read_text()
            return await connection.fetch(sql)
        finally:
            await connection.close()

    rows = asyncio.run(run())

    assert [dict(row) for row in rows] == [
        {
            "player_id": "thin",
            "season": 2019,
            "start_value": 125,
            "end_value": 150,
            "start_evidence": 0.0,
            "end_evidence": 0.0,
            "start_trades_window": 0,
            "end_trades_window": 0,
            "end_value_next_board": 140,
            "end_evidence_next_board": 0.0,
        }
    ]
