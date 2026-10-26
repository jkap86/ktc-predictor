import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from predictor_v5.movement_evaluation.contract import (
    POSITIONS,
    FoldSpec,
    canonical_bytes,
)
from predictor_v5.movement_evaluation.metrics import (
    _cluster_draw_indices,
    bootstrap_uncertainty,
    leave_one_season_out,
    score_contrast,
    summarize_primary,
)


def _prediction_frame(rows):
    records = []
    for season, player_id, position, actual, production, no_production in rows:
        start = 1000.0
        records.append(
            {
                "season": season,
                "player_id": player_id,
                "position": position,
                "age": 25.0,
                "games": 16.0,
                "points": 100.0,
                "start_trades_window": 10,
                "end_trades_window": 10,
                "value_tier": "mid",
                "S": start,
                "E": start * np.exp(actual),
                "r": actual,
                "d": start * np.expm1(actual),
                "production_E_hat": start * np.exp(production),
                "production_r_hat": production,
                "production_d_hat": start * np.expm1(production),
                "no_production_E_hat": start * np.exp(no_production),
                "no_production_r_hat": no_production,
                "no_production_d_hat": start * np.expm1(no_production),
                "no_change_E_hat": start,
                "no_change_r_hat": 0.0,
                "no_change_d_hat": 0.0,
            }
        )
    return pd.DataFrame(records).sort_values(["season", "player_id"], kind="mergesort").reset_index(drop=True)


def _bootstrap_frame():
    return _prediction_frame(
        [
            (2021, "a", "QB", -0.50, -0.40, -0.10),
            (2021, "b", "QB", 0.20, 0.10, -0.15),
            (2021, "c", "RB", 0.65, 0.45, 0.05),
            (2022, "a", "QB", 0.45, 0.35, 0.00),
            (2022, "d", "RB", -0.30, -0.25, 0.10),
            (2022, "e", "RB", 0.10, 0.30, -0.20),
        ]
    )


class _FixedRng:
    def __init__(self, draws):
        self._draws = np.asarray(draws, dtype=int)

    def integers(self, _low, _high, *, size):
        assert size == len(self._draws)
        return self._draws.copy()


def _intervals(result):
    for contrast in result["contrasts"].values():
        for name, cell in contrast.items():
            if name == "movement_spearman_difference":
                yield cell
                yield from cell["per_position"].values()
            else:
                yield cell


def _fold(season, rows):
    return SimpleNamespace(
        spec=FoldSpec(season, tuple(range(2020, season)), season == 2021),
        predictions=_prediction_frame(rows),
        training_majority="flat",
        _models=(object(), object()),
    )


def test_cluster_draws_carry_whole_players_and_whole_season_blocks():
    frame = _bootstrap_frame()

    player_indices = _cluster_draw_indices(frame, "player", _FixedRng([0, 0, 4, 4, 4]))
    sampled_players = frame.iloc[player_indices][["season", "player_id"]]
    assert sampled_players.value_counts().to_dict() == {
        (2021, "a"): 2,
        (2022, "a"): 2,
        (2022, "e"): 3,
    }

    fold = frame.loc[frame["season"] == 2021].reset_index(drop=True)
    fold_indices = _cluster_draw_indices(fold, "player", _FixedRng([1, 1, 2]))
    assert fold.iloc[fold_indices]["player_id"].tolist() == ["b", "b", "c"]

    season_indices = _cluster_draw_indices(frame, "season", _FixedRng([1, 1]))
    sampled_seasons = frame.iloc[season_indices]
    assert sampled_seasons["season"].unique().tolist() == [2022]
    assert sampled_seasons["player_id"].tolist() == ["a", "d", "e", "a", "d", "e"]


def test_bootstrap_is_repeatable_order_independent_paired_and_never_refits(monkeypatch):
    from predictor_v5 import model as protected_model

    def refuse_fit(*_args, **_kwargs):
        raise AssertionError("uncertainty must not refit a model")

    monkeypatch.setattr(protected_model.WhatIfModel, "fit", refuse_fit)
    frame = _bootstrap_frame()
    first = bootstrap_uncertainty(frame, seed=17, resamples=2000, cluster="player")
    shuffled = frame.sample(frac=1, random_state=9).reset_index(drop=True)
    repeated = bootstrap_uncertainty(shuffled, seed=17, resamples=2000, cluster="player")
    changed = bootstrap_uncertainty(frame, seed=18, resamples=2000, cluster="player")

    assert canonical_bytes(first) == canonical_bytes(repeated)
    assert canonical_bytes(first) != canonical_bytes(changed)
    first_interval = first["contrasts"]["production_vs_no_production"]["mae_log_difference"]
    changed_interval = changed["contrasts"]["production_vs_no_production"]["mae_log_difference"]
    assert (first_interval["lower"], first_interval["upper"]) != (
        changed_interval["lower"],
        changed_interval["upper"],
    )
    assert first["seed"] == 17
    assert first["requested_resamples"] == 2000
    assert first["percentiles"] == [2.5, 97.5]
    assert first["quantile_method"] == "linear"
    assert first["paired_draws"] is True
    assert all(cell["requested_draws"] == 2000 for cell in _intervals(first))


def test_bootstrap_records_all_contrast_families_and_undefined_draw_counts():
    frame = _prediction_frame(
        [
            (2021, "a", "QB", -0.5, -0.4, -0.1),
            (2021, "b", "QB", 0.5, 0.4, 0.1),
        ]
    )
    result = bootstrap_uncertainty(frame, seed=20261025, resamples=2000, cluster="player")

    expected = {
        "mae_log_difference",
        "rmse_log_difference",
        "mae_value_difference",
        "rmse_value_difference",
        "skill_log_gain",
        "skill_value_gain",
        "direction_accuracy_difference",
        "positive_mae_log_improvement",
        "movement_spearman_difference",
    }
    assert set(result["contrasts"]["production_vs_no_production"]) == expected
    point = score_contrast(frame, "production", "no_production")
    for metric in expected - {"movement_spearman_difference"}:
        assert result["contrasts"]["production_vs_no_production"][metric][
            "estimate"
        ] == pytest.approx(point[metric]["value"])
    point_rank = point["movement_spearman_difference"]
    bootstrap_rank = result["contrasts"]["production_vs_no_production"][
        "movement_spearman_difference"
    ]
    assert bootstrap_rank["estimate"] == pytest.approx(point_rank["value"])
    for position in POSITIONS:
        expected_position = point_rank["per_position"][position]["value"]
        actual_position = bootstrap_rank["per_position"][position]["estimate"]
        if expected_position is None:
            assert actual_position is None
        else:
            assert actual_position == pytest.approx(expected_position)
    for cell in _intervals(result):
        assert cell["effective_draws"] + cell["undefined_draws"] == 2000
        assert cell["requested_draws"] == 2000

    rank = result["contrasts"]["production_vs_no_production"]["movement_spearman_difference"]
    assert rank["effective_draws"] < 1800
    assert rank["lower"] is None and rank["upper"] is None
    assert rank["reason"] == "insufficient_valid_resamples"
    assert set(rank["per_position"]) == set(POSITIONS)
    no_change_rank = result["contrasts"]["production_vs_no_change"]["movement_spearman_difference"]
    assert no_change_rank["estimate"] is None
    assert no_change_rank["effective_draws"] == 0
    assert no_change_rank["reason"] == "insufficient_valid_resamples"
    for contrast in result["contrasts"].values():
        difference = contrast["mae_log_difference"]
        positive = contrast["positive_mae_log_improvement"]
        assert positive["estimate"] == pytest.approx(-difference["estimate"])
        assert positive["lower"] == pytest.approx(-difference["upper"])
        assert positive["upper"] == pytest.approx(-difference["lower"])
        assert positive["effective_draws"] == difference["effective_draws"]


def test_bootstrap_handles_empty_thin_and_one_cluster_with_finite_json():
    template = _bootstrap_frame().iloc[0:0].copy()
    empty = bootstrap_uncertainty(template, seed=4, resamples=20, cluster="player")
    assert empty["n_clusters"] == 0
    assert all(cell["reason"] == "insufficient_valid_resamples" for cell in _intervals(empty))

    thin = bootstrap_uncertainty(_bootstrap_frame().iloc[[0]], seed=4, resamples=20, cluster="player")
    assert thin["n_clusters"] == 1
    mae = thin["contrasts"]["production_vs_no_production"]["mae_log_difference"]
    assert mae["effective_draws"] == 20
    assert mae["lower"] == pytest.approx(mae["upper"])
    json.dumps(empty, allow_nan=False)
    json.dumps(thin, allow_nan=False)


def test_leave_one_season_out_omits_exactly_one_named_season():
    frame = _prediction_frame(
        [
            (2021, "a", "QB", 1.0, 0.0, 0.5),
            (2022, "b", "QB", 1.0, 1.0, 0.5),
            (2022, "c", "RB", 1.0, 1.0, 0.5),
            (2022, "d", "WR", 1.0, 1.0, 0.5),
        ]
    )
    results = leave_one_season_out(frame)

    assert [row["omitted_season"] for row in results] == [2021, 2022]
    assert [row["included_seasons"] for row in results] == [[2022], [2021]]
    assert [row["n"] for row in results] == [3, 1]
    assert results[0]["models"]["production"]["mae_log"]["value"] == 0.0
    assert results[1]["models"]["production"]["mae_log"]["value"] == 1.0


def test_summary_pins_five_fold_estimates_macro_pooling_and_exploratory_metadata():
    folds = (
        _fold(2021, [(2021, "a", "QB", 1.0, 0.0, 0.5)]),
        _fold(
            2022,
            [
                (2022, "b", "QB", 1.0, 1.0, 0.5),
                (2022, "c", "RB", 1.0, 1.0, 0.5),
                (2022, "d", "WR", 1.0, 1.0, 0.5),
            ],
        ),
        _fold(2023, [(2023, "e", "TE", 1.0, 0.8, 0.5)]),
        _fold(2024, [(2024, "f", "QB", 1.0, 0.6, 0.5)]),
        _fold(2025, [(2025, "g", "RB", 0.0, 0.6, 0.0)]),
    )

    fold_summaries, pooled = summarize_primary(folds)

    assert [row["test_season"] for row in fold_summaries] == [2021, 2022, 2023, 2024, 2025]
    assert [row["models"]["production"]["mae_log"]["value"] for row in fold_summaries] == pytest.approx(
        [1.0, 0.0, 0.2, 0.4, 0.6]
    )
    pooled_production = pooled["models"]["production"]
    macro_production = pooled["macro_by_season"]["production"]
    assert pooled_production["mae_log"]["value"] == pytest.approx(2.2 / 7)
    assert macro_production["mae_log"]["value"] == pytest.approx(0.44)
    assert pooled_production["mae_log"]["value"] != macro_production["mae_log"]["value"]
    assert pooled_production["skill_log"]["value"] == pytest.approx(1 - 2.2 / 6)
    assert macro_production["skill_log"]["value"] == pytest.approx(0.6)
    assert macro_production["skill_log"]["reason"] is None
    assert macro_production["skill_log"]["n_defined"] == 4
    assert macro_production["skill_log"]["n_undefined"] == 1
    assert macro_production["skill_log"]["n_total"] == 5

    uncertainty = pooled["uncertainty"]
    assert uncertainty["player_cluster"]["cluster"] == "player"
    assert list(uncertainty["by_position"]) == list(POSITIONS)
    exploratory = uncertainty["season_block_exploratory"]
    assert exploratory["cluster"] == "season"
    assert exploratory["n_clusters"] == 5
    assert exploratory["exploratory"] is True
    assert "five observed test-season blocks" in exploratory["caveat"]
    assert len(pooled["leave_one_season_out"]) == 5
    assert "simultaneous player/team/season/league uncertainty" in pooled["uncertainty_caveat"]
    assert pooled["multiplicity"] == "family-wide multiplicity is uncontrolled"
    json.dumps((fold_summaries, pooled), allow_nan=False)


@pytest.mark.parametrize("cluster", ["row", "team", "league", "PLAYER"])
def test_bootstrap_rejects_unknown_cluster(cluster):
    with pytest.raises(ValueError, match="cluster"):
        bootstrap_uncertainty(_bootstrap_frame(), seed=1, resamples=10, cluster=cluster)


@pytest.mark.parametrize("resamples", [0, -1, True, 1.5])
def test_bootstrap_rejects_invalid_resample_count(resamples):
    with pytest.raises(ValueError, match="resamples"):
        bootstrap_uncertainty(_bootstrap_frame(), seed=1, resamples=resamples, cluster="player")
