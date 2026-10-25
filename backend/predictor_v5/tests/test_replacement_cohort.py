import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from predictor_v5.replacement_cohort import matched_cohort, prepare_training
from predictor_v5.replacement_mfl import TARGET_COLUMNS

SEASONS = [2021, 2022, 2023, 2024, 2025]
POSITIONS = ["QB", "RB", "TE", "WR"]
START_COLUMNS = [
    "season", "player_id", "name", "position", "team", "birth_date",
    "bc_value", "bc_trades", "bc_date", "bc_model_version", "ktc_value", "ktc_date",
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def season(year, *, start=100, end=110, games=17, points=100, years_exp=2):
    return {"year": year, "age": 25, "years_exp": years_exp,
            "start_ktc": start, "end_ktc": end,
            "fantasy_points": points, "games_played": games}


def players():
    result = [{"player_id": str(10 + i), "name": position, "position": position,
               "seasons": [season(year, start=100 + i) for year in SEASONS]}
              for i, position in enumerate(POSITIONS)]
    result.append({"player_id": "college", "name": "College", "position": "QB",
                   "seasons": [season(2021, games=0, points=0, years_exp=0)]})
    return result


def write_archive(path, value, member="training-data.json", extra=False):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(member, json.dumps({"players": value}, allow_nan=False))
        if extra:
            archive.writestr("extra.json", "{}")


def mfl_rows(value):
    return [{"player_id": p["player_id"], "season": s["year"],
             "start_value": 200, "end_value": 220, "position": p["position"],
             "age": s["age"], "games": s["games_played"],
             "points": s["fantasy_points"], "start_evidence": 1, "end_evidence": 1,
             "start_trades_window": 5, "end_trades_window": 5,
             "end_value_next_board": "", "end_evidence_next_board": ""}
            for p in value for s in p["seasons"]]


def manifest(archive, labels, member="training-data.json"):
    return {
        "training": {"archive": {"path": archive.name, "sha256": sha(archive)},
                     "ktc_member": member,
                     "mfl_labels": {"path": labels.name, "sha256": sha(labels)},
                     "seasons": SEASONS, "positions": POSITIONS},
        "model": {"ktc_scale_max": 9999, "mfl_scale_max": 10000},
        "filters": {"ktc": {"minimum_exclusive": 0, "maximum_exclusive": 9999},
                    "mfl": {"minimum_exclusive": 0, "minimum_start_trades": 5,
                            "minimum_end_trades": 5}},
    }


@pytest.fixture
def training(tmp_path):
    ps = players()
    archive, labels = tmp_path / "training.zip", tmp_path / "mfl.csv"
    write_archive(archive, ps)
    pd.DataFrame(mfl_rows(ps), columns=TARGET_COLUMNS).to_csv(labels, index=False)
    return tmp_path, archive, labels, ps


def test_training_coverage_leakage_filters_and_pre_nfl_exclusion(training):
    root, archive, labels, _ = training
    result = prepare_training(root, manifest(archive, labels))
    assert set(result) == {"ktc", "mfl"}
    for frame in result.values():
        assert frame[["season", "player_id"]].values.tolist() == sorted(
            frame[["season", "player_id"]].values.tolist())
        assert set(frame.season) == set(SEASONS)
        assert set(frame.position) == set(POSITIONS)
        assert "college" not in set(frame.player_id)
        assert not frame.duplicated(["player_id", "season"]).any()
    assert (result["ktc"][["start_value", "end_value"]] > 0).all().all()
    assert (result["ktc"][["start_value", "end_value"]] < 9999).all().all()
    assert (result["mfl"][["start_value", "end_value"]] > 0).all().all()


def test_training_parses_captured_verified_bytes(monkeypatch, training):
    root, archive, labels, _ = training
    expected = manifest(archive, labels)
    real_read = Path.read_bytes

    def mutate_after_read(path):
        data = real_read(path)
        if path == archive:
            path.write_bytes(b"attacker")
        elif path == labels:
            path.write_text("attacker")
        return data

    monkeypatch.setattr(Path, "read_bytes", mutate_after_read)
    result = prepare_training(root, expected)
    assert {name: len(frame) for name, frame in result.items()} == {"ktc": 20, "mfl": 20}


def test_training_accepts_csv_float_round_trip_noise(training):
    root, archive, labels, ps = training
    ps[0]["seasons"][0]["fantasy_points"] = 217.70000000000002
    write_archive(archive, ps)
    rows = mfl_rows(ps)
    rows[0]["points"] = 217.7
    pd.DataFrame(rows, columns=TARGET_COLUMNS).to_csv(labels, index=False)
    result = prepare_training(root, manifest(archive, labels))
    assert len(result["mfl"]) == 20


@pytest.mark.parametrize("mutation,match", [
    ("archive_digest", "digest"), ("mfl_digest", "digest"),
    ("unsafe_member", "member"), ("extra_member", "member"),
    ("future_archive", "season"), ("future_mfl", "season"),
    ("duplicate_archive", "duplicate"), ("duplicate_mfl", "duplicate"),
    ("nonfinite_archive", "numeric"), ("nonfinite_mfl", "numeric"),
    ("missing_position", "coverage"), ("missing_season", "coverage"),
    ("scale", "scale"), ("trades", "filter"), ("integer_overflow", "integer"),
])
def test_training_rejects_invalid_complete_inputs(training, mutation, match):
    root, archive, labels, ps = training
    rows = mfl_rows(ps)
    spec = manifest(archive, labels)
    if mutation == "archive_digest":
        spec["training"]["archive"]["sha256"] = "0" * 64
    elif mutation == "mfl_digest":
        spec["training"]["mfl_labels"]["sha256"] = "0" * 64
    elif mutation == "unsafe_member":
        write_archive(archive, ps, "../training-data.json")
        spec = manifest(archive, labels, "../training-data.json")
    elif mutation == "extra_member":
        write_archive(archive, ps, extra=True)
        spec = manifest(archive, labels)
    elif mutation == "future_archive":
        ps[0]["seasons"].append(season(2026)); write_archive(archive, ps)
        spec = manifest(archive, labels)
    elif mutation == "future_mfl":
        rows[0]["season"] = 2026
    elif mutation == "duplicate_archive":
        ps[0]["seasons"].append(dict(ps[0]["seasons"][0])); write_archive(archive, ps)
        spec = manifest(archive, labels)
    elif mutation == "duplicate_mfl":
        rows.append(dict(rows[0]))
    elif mutation == "nonfinite_archive":
        ps[0]["seasons"][0]["age"] = float("inf")
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("training-data.json", json.dumps({"players": ps}))
        spec = manifest(archive, labels)
    elif mutation == "nonfinite_mfl":
        rows[0]["start_value"] = "inf"
    elif mutation == "missing_position":
        ps[:] = [p for p in ps if p["position"] != "TE"]
        write_archive(archive, ps); rows = mfl_rows(ps)
        pd.DataFrame(rows, columns=TARGET_COLUMNS).to_csv(labels, index=False)
        spec = manifest(archive, labels)
    elif mutation == "missing_season":
        for p in ps:
            p["seasons"] = [s for s in p["seasons"] if s["year"] != 2024]
        write_archive(archive, ps); rows = mfl_rows(ps)
        pd.DataFrame(rows, columns=TARGET_COLUMNS).to_csv(labels, index=False)
        spec = manifest(archive, labels)
    elif mutation == "scale":
        spec["model"]["mfl_scale_max"] = 9999
    elif mutation == "trades":
        spec["filters"]["mfl"]["minimum_start_trades"] = 4
    else:
        rows[0]["games"] = str(np.iinfo(np.int64).max + 1)
    if mutation in {"future_mfl", "duplicate_mfl", "nonfinite_mfl", "integer_overflow"}:
        pd.DataFrame(rows, columns=TARGET_COLUMNS).to_csv(labels, index=False)
        spec = manifest(archive, labels)
    with pytest.raises(ValueError, match=match):
        prepare_training(root, spec)


def starts(rows):
    defaults = {"name": "Player", "position": "QB", "team": "T",
                "birth_date": "2000-01-01", "bc_trades": 5,
                "bc_date": "2026-09-09", "bc_model_version": "baseline-hl28-v1",
                "ktc_date": "2026-09-09"}
    return pd.DataFrame([{**defaults, **row} for row in rows], columns=START_COLUMNS)


def ktc(rows):
    return pd.DataFrame(rows, columns=TARGET_COLUMNS[:8])


def mfl(rows):
    return pd.DataFrame([[*row, 1, 1, 5, 5, np.nan, np.nan] for row in rows],
                        columns=TARGET_COLUMNS)


def prediction(rows):
    return pd.DataFrame(rows, columns=["player_id", "season", "fitted_start", "predicted_end"])


def test_exact_intersection_relative_own_source_and_deterministic_coverage():
    frozen = starts([
        {"player_id": "b", "season": 2026, "ktc_value": 100, "bc_value": 200},
        {"player_id": "a", "season": 2026, "ktc_value": 80, "bc_value": 160},
        {"player_id": "start-only", "season": 2026, "ktc_value": 50, "bc_value": 60},
        {"player_id": "missing-ktc", "season": 2026, "ktc_value": np.nan,
         "ktc_date": np.nan, "bc_value": 70},
        {"player_id": "illiquid", "season": 2026, "ktc_value": 70,
         "bc_value": 90, "bc_trades": 4},
    ])
    ktc_target = ktc([["a", 2026, 80, 120, "QB", 25, 17, 100],
                      ["b", 2026, 100, 130, "RB", 26, 16, 90],
                      ["ktc-only", 2026, 10, 11, "WR", 24, 15, 80]])
    mfl_target = mfl([["b", 2026, 200, 250, "RB", 26, 16, 90],
                      ["a", 2026, 160, 180, "QB", 25, 17, 100],
                      ["mfl-only", 2026, 20, 21, "TE", 23, 14, 70]])
    predictions = {
        "ktc": prediction([["b", 2026, 50, 75], ["a", 2026, 40, 30],
                           ["ktc-pred-only", 2026, 10, 10]]),
        "mfl": prediction([["a", 2026, 80, 120], ["b", 2026, 100, 80],
                           ["mfl-pred-only", 2026, 10, 10]]),
    }
    cohort, coverage = matched_cohort(frozen, ktc_target, mfl_target, predictions)
    assert list(cohort.columns) == ["player_id", "season", "position", "age", "games",
                                    "points", "ktc_start", "ktc_actual", "ktc_pred",
                                    "mfl_start", "mfl_actual", "mfl_pred"]
    assert cohort.player_id.tolist() == ["a", "b"]
    assert cohort[["ktc_pred", "mfl_pred"]].values.tolist() == [[60, 240], [150, 160]]
    assert coverage["input_counts"] == {"starts": 3, "ktc": 3, "mfl": 3,
                                         "ktc_predictions": 3, "mfl_predictions": 3}
    assert coverage["matched_count"] == 2
    assert coverage["excluded_keys"]["starts"] == [
        [2026, "ktc-only"], [2026, "ktc-pred-only"],
        [2026, "mfl-only"], [2026, "mfl-pred-only"],
    ]


def test_empty_intersection_has_exact_safe_output():
    frozen = starts([{"player_id": "a", "season": 2026, "ktc_value": 10, "bc_value": 20}])
    cohort, coverage = matched_cohort(
        frozen, ktc([["b", 2026, 10, 11, "QB", 25, 1, 1]]),
        mfl([["c", 2026, 20, 21, "QB", 25, 1, 1]]),
        {"ktc": prediction([["d", 2026, 10, 10]]),
         "mfl": prediction([["e", 2026, 20, 20]])},
    )
    assert cohort.empty and len(cohort.columns) == 12
    assert coverage["matched_count"] == 0


def test_start_conflict_cannot_hide_behind_missing_prediction():
    frozen = starts([{"player_id": "a", "season": 2026, "ktc_value": 100, "bc_value": 200}])
    with pytest.raises(ValueError, match="target start"):
        matched_cohort(
            frozen,
            ktc([["a", 2026, 99, 110, "QB", 25, 1, 1]]),
            mfl([["a", 2026, 200, 220, "QB", 25, 1, 1]]),
            {"ktc": prediction([],), "mfl": prediction([["a", 2026, 200, 210]])},
        )


@pytest.mark.parametrize("mutation,match", [
    ("start_duplicate", "duplicate"), ("target_duplicate", "duplicate"),
    ("prediction_schema", "prediction schema"), ("target_schema", "target schema"),
    ("stats", "stats disagree"), ("ktc_scale", "KTC"), ("mfl_trade", "trades"),
    ("nonfinite_prediction", "predicted_end"), ("boolean_prediction", "fitted_start"),
    ("wrong_season", "season"), ("overflow", "relative prediction"),
])
def test_matched_cohort_rejects_schema_values_and_overflow(mutation, match):
    frozen = starts([{"player_id": "a", "season": 2026, "ktc_value": 100, "bc_value": 200}])
    ktc_target = ktc([["a", 2026, 100, 110, "QB", 25, 17, 100]])
    mfl_target = mfl([["a", 2026, 200, 220, "QB", 25, 17, 100]])
    predictions = {"ktc": prediction([["a", 2026, 100, 120]]),
                   "mfl": prediction([["a", 2026, 200, 210]])}
    if mutation == "start_duplicate": frozen = pd.concat([frozen, frozen])
    elif mutation == "target_duplicate": ktc_target = pd.concat([ktc_target, ktc_target])
    elif mutation == "prediction_schema": predictions["ktc"] = predictions["ktc"].rename(columns={"predicted_end": "prediction"})
    elif mutation == "target_schema": ktc_target = ktc_target.assign(extra=1)
    elif mutation == "stats": mfl_target.loc[0, "points"] = 99
    elif mutation == "ktc_scale": ktc_target.loc[0, "end_value"] = 9999
    elif mutation == "mfl_trade": mfl_target.loc[0, "start_trades_window"] = 4
    elif mutation == "nonfinite_prediction":
        predictions["mfl"] = prediction([["a", 2026, 200.0, np.inf]])
    elif mutation == "boolean_prediction":
        predictions["ktc"] = prediction([["a", 2026, True, 120.0]])
    elif mutation == "wrong_season": predictions["mfl"].loc[0, "season"] = 2025
    else:
        predictions["ktc"] = prediction([["a", 2026, 1e-300, 1e300]])
    with pytest.raises(ValueError, match=match):
        matched_cohort(frozen, ktc_target, mfl_target, predictions)
