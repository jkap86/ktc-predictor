"""Matched-cohort comparison and deployment gate for Backchannel labels."""

import json
from copy import deepcopy
from pathlib import Path

import pandas as pd

from predictor_v5.data import DEFAULT_ZIP, build_dataset, load_season_stats
from predictor_v5.evaluate import holdout
from predictor_v5.labels import CsvLabels

POSITIONS = ("QB", "RB", "WR", "TE")


def coverage_counts(frame: pd.DataFrame) -> dict[str, int]:
    """Deterministic player-season row counts for each represented season."""
    counts = frame.groupby("season", sort=True).size()
    return {str(int(season)): int(count) for season, count in counts.items()}


def matched_test_ids(baseline: pd.DataFrame, candidate: pd.DataFrame, test_season: int) -> set[str]:
    """Player ids represented by both label sources in the holdout season."""
    baseline_ids = set(baseline.loc[baseline["season"] == test_season, "player_id"].astype(str))
    candidate_ids = set(candidate.loc[candidate["season"] == test_season, "player_id"].astype(str))
    return baseline_ids & candidate_ids


def _without_predictions(result: dict) -> dict:
    summary = dict(result)
    summary.pop("predictions", None)
    return summary


def _matched_holdout_frame(frame: pd.DataFrame, ids: set[str], test_season: int) -> pd.DataFrame:
    keep = (frame["season"] != test_season) | frame["player_id"].astype(str).isin(ids)
    return frame.loc[keep].copy()


def compare_datasets(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    train_starts: tuple[int, ...] = (2022, 2021, 2020),
    train_end: int = 2024,
    test_season: int = 2025,
) -> dict:
    """Evaluate the old control and candidate windows on all and matched rows."""
    matched_ids = matched_test_ids(baseline, candidate, test_season)
    baseline_train = list(range(2022, train_end + 1))
    baseline_all = _without_predictions(holdout(baseline, "old", 10000.0, baseline_train, test_season))
    baseline_matched = _without_predictions(
        holdout(
            _matched_holdout_frame(baseline, matched_ids, test_season),
            "old",
            10000.0,
            baseline_train,
            test_season,
        )
    )

    candidate_rows = []
    for train_start in train_starts:
        train_seasons = list(range(train_start, train_end + 1))
        all_rows = _without_predictions(holdout(candidate, "mfl", 10000.0, train_seasons, test_season))
        matched = _without_predictions(
            holdout(
                _matched_holdout_frame(candidate, matched_ids, test_season),
                "mfl",
                10000.0,
                train_seasons,
                test_season,
            )
        )
        candidate_rows.append({"train_start": train_start, "all": all_rows, "matched": matched})

    decision, candidate_rows = select_candidate(baseline_matched, candidate_rows)
    return {
        "schema_version": 1,
        "config": {
            "baseline_train_seasons": baseline_train,
            "candidate_train_starts": list(train_starts),
            "train_end": train_end,
            "test_season": test_season,
            "min_trades": 5,
        },
        "coverage": {"old": coverage_counts(baseline), "mfl": coverage_counts(candidate)},
        "matched_test_rows": len(matched_ids),
        "baseline": {"all": baseline_all, "matched": baseline_matched},
        "candidates": candidate_rows,
        "decision": decision,
    }


def compare_label_files(
    baseline_path: Path | str,
    candidate_path: Path | str,
    *,
    training_data_path: Path | str = DEFAULT_ZIP,
    min_trades: int = 5,
    train_starts: tuple[int, ...] = (2022, 2021, 2020),
    train_end: int = 2024,
    test_season: int = 2025,
) -> dict:
    """Load both versioned CSVs and return a deterministic comparison report."""
    stats = load_season_stats(training_data_path)
    baseline_source = CsvLabels(baseline_path, name="old", min_trades=min_trades)
    candidate_source = CsvLabels(candidate_path, name="mfl", min_trades=min_trades)
    report = compare_datasets(
        build_dataset(stats, baseline_source),
        build_dataset(stats, candidate_source),
        train_starts=train_starts,
        train_end=train_end,
        test_season=test_season,
    )
    report["config"]["min_trades"] = min_trades
    report["inputs"] = {
        "baseline": Path(baseline_path).name,
        "candidate": Path(candidate_path).name,
        "training_data": Path(training_data_path).name,
    }
    return report


def write_report(report: dict, out_path: Path | str) -> None:
    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def select_candidate(
    baseline_matched: dict,
    candidates: list[dict],
    *,
    min_gain: float = 0.02,
    min_position_rows: int = 30,
    max_position_drop: float = 0.05,
    tie_margin: float = 0.01,
) -> tuple[dict, list[dict]]:
    """Apply the overall, position, and sample gates and choose a window."""
    baseline_skill = baseline_matched["overall"]["skill"]
    evaluated = deepcopy(candidates)

    for candidate in evaluated:
        matched = candidate["matched"]
        skill = matched["overall"]["skill"]
        gain = None if skill is None or baseline_skill is None else skill - baseline_skill
        overall_guard = gain is not None and gain + 1e-12 >= min_gain
        sample_guard = all(matched["by_position"].get(p, {}).get("n", 0) >= min_position_rows for p in POSITIONS)
        position_guard = all(
            matched["by_position"].get(p, {}).get("skill") is not None
            and baseline_matched["by_position"].get(p, {}).get("skill") is not None
            and matched["by_position"][p]["skill"] + max_position_drop + 1e-12
            >= baseline_matched["by_position"][p]["skill"]
            for p in POSITIONS
        )
        candidate["gate"] = {
            "gain": None if gain is None else round(gain, 6),
            "overall_guard": overall_guard,
            "sample_guard": sample_guard,
            "position_guard": position_guard,
            "qualified": overall_guard and sample_guard and position_guard,
        }
    ranked = [candidate for candidate in evaluated if candidate["matched"]["overall"]["skill"] is not None]
    if not ranked:
        return {"action": "retain", "selected_train_start": None}, evaluated

    best_skill = max(candidate["matched"]["overall"]["skill"] for candidate in ranked)
    near_best = [
        candidate
        for candidate in ranked
        if best_skill - candidate["matched"]["overall"]["skill"] <= tie_margin + 1e-12
    ]
    selected = max(near_best, key=lambda candidate: candidate["train_start"])
    action = "update" if selected["gate"]["qualified"] else "retain"
    return {"action": action, "selected_train_start": selected["train_start"]}, evaluated
