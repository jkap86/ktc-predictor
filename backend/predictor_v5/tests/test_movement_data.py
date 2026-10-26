import io
import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from predictor_v5.movement_evaluation.contract import ContractRefusal
from predictor_v5.movement_evaluation.data import (
    HistoricalBundle,
    load_pinned_bundle,
    parse_archive_bytes,
    parse_label_bytes,
    prepare_evaluation_data,
)


def _archive_bytes(*rows):
    players = []
    for player_id, position, seasons in rows:
        players.append({"player_id": player_id, "position": position, "seasons": seasons})
    payload = json.dumps({"players": players}, separators=(",", ":")).encode()
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("training-data.json", payload)
    return output.getvalue()


def _season(year, *, age=25, points=100.0, games=10, years_exp=1, start_ktc=1000, end_ktc=1100):
    return {
        "year": year,
        "age": age,
        "fantasy_points": points,
        "games_played": games,
        "years_exp": years_exp,
        "start_ktc": start_ktc,
        "end_ktc": end_ktc,
    }


def _labels(rows):
    columns = [
        "player_id", "season", "start_value", "end_value", "position", "age", "games", "points",
        "start_evidence", "end_evidence", "start_trades_window", "end_trades_window",
        "end_value_next_board", "end_evidence_next_board",
    ]
    return pd.DataFrame(rows, columns=columns).to_csv(index=False).encode()


def _label_row(player_id, season, *, end=1200, end_trades=6, age=25.0, points=100.0, position="QB", games=10):
    return [player_id, season, 1000, end, position, age, games, points, 1.0, 1.0, 6, end_trades, 0.0, 0.0]


def test_parse_archive_extracts_core_without_touching_optional_ktc_fields():
    raw = _archive_bytes(("p1", "QB", [_season(2020, start_ktc=999, end_ktc=888)]))
    frame, member_hash = parse_archive_bytes(raw, include_ktc=False)
    assert list(frame.columns) == ["player_id", "season", "position", "age", "points", "games", "years_exp"]
    assert frame.iloc[0].to_dict() == {
        "player_id": "p1", "season": 2020, "position": "QB", "age": 25.0,
        "points": 100.0, "games": 10, "years_exp": 1,
    }
    assert len(member_hash) == 64


def test_parse_archive_rejects_duplicate_keys_unknown_position_and_2026():
    duplicate = _archive_bytes(("p1", "QB", [_season(2020), _season(2020)]))
    with pytest.raises(ContractRefusal):
        parse_archive_bytes(duplicate, include_ktc=False)
    unknown = _archive_bytes(("p1", "K", [_season(2020)]))
    with pytest.raises(ContractRefusal):
        parse_archive_bytes(unknown, include_ktc=False)
    future = _archive_bytes(("p1", "QB", [_season(2026)]))
    with pytest.raises(ContractRefusal):
        parse_archive_bytes(future, include_ktc=False)


def test_parse_label_rejects_extra_columns_and_negative_values():
    row = _label_row("p1", 2020)
    assert len(parse_label_bytes(_labels([row]))) == 1
    bad = pd.DataFrame([row], columns=[
        "player_id", "season", "start_value", "end_value", "position", "age", "games", "points",
        "start_evidence", "end_evidence", "start_trades_window", "end_trades_window",
        "end_value_next_board", "unexpected",
    ])
    with pytest.raises(ContractRefusal):
        parse_label_bytes(bad.to_csv(index=False).encode())
    negative = row.copy()
    negative[2] = -1
    with pytest.raises(ContractRefusal):
        parse_label_bytes(_labels([negative]))


def test_common_cohort_is_sorted_and_zero_end_is_diagnostic():
    archive_raw = _archive_bytes(
        ("p2", "RB", [_season(2021, age=24, points=0.0, games=0, years_exp=1)]),
        ("p1", "QB", [_season(2020)]),
    )
    archive, _ = parse_archive_bytes(archive_raw, include_ktc=False)
    labels = parse_label_bytes(_labels([
        _label_row("p2", 2021, end=0, end_trades=6, age=24.0, points=0.0, position="RB", games=0),
        _label_row("p1", 2020),
    ]))
    result = prepare_evaluation_data(HistoricalBundle(archive, labels, {}, {}))
    assert list(result.primary_scored[["season", "player_id"]].itertuples(index=False, name=None)) == [(2020, "p1")]
    assert list(result.observed_zero_end[["season", "player_id"]].itertuples(index=False, name=None)) == [(2021, "p2")]
    assert result.coverage["full_preseason_coverage"] == {"value": None, "reason": "not_recoverable_from_historical_export"}


def test_missing_nonfinite_duplicate_and_2026_rows_refuse():
    missing_age = _season(2020)
    del missing_age["age"]
    with pytest.raises(ContractRefusal):
        parse_archive_bytes(_archive_bytes(("p1", "QB", [missing_age])), include_ktc=False)
    nonfinite = _season(2020)
    nonfinite["fantasy_points"] = float("inf")
    with pytest.raises(ContractRefusal):
        parse_archive_bytes(_archive_bytes(("p1", "QB", [nonfinite])), include_ktc=False)
    with pytest.raises(ContractRefusal):
        parse_archive_bytes(_archive_bytes(("p1", "QB", [_season(2026)])), include_ktc=False)
    duplicate_json = b'{"players":[{"player_id":"p1","position":"QB","position":"RB","seasons":[]}]}'
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("training-data.json", duplicate_json)
    with pytest.raises(ContractRefusal):
        parse_archive_bytes(output.getvalue(), include_ktc=False)


def test_negative_next_board_and_archive_ktc_bounds_refuse():
    row = _label_row("p1", 2020)
    row[12] = -7
    with pytest.raises(ContractRefusal):
        parse_label_bytes(_labels([row]))
    bad_ktc = _season(2020, start_ktc=10000)
    with pytest.raises(ContractRefusal):
        parse_archive_bytes(_archive_bytes(("p1", "QB", [bad_ktc])), include_ktc=True)


def test_common_primary_cohort_is_sorted_and_one_to_one():
    archive_raw = _archive_bytes(
        ("p2", "RB", [_season(2021, age=24, points=0.0, games=0, years_exp=-1)]),
        ("p1", "QB", [_season(2020)]),
    )
    archive, _ = parse_archive_bytes(archive_raw, include_ktc=False)
    labels = parse_label_bytes(_labels([_label_row("p2", 2021, position="RB", age=24.0, points=0.0, games=0), _label_row("p1", 2020)]))
    result = prepare_evaluation_data(HistoricalBundle(archive, labels, {}, {}))
    keys = list(result.primary_scored[["season", "player_id"]].itertuples(index=False, name=None))
    assert keys == sorted(keys)
    assert len(keys) == len(set(keys))


def test_available_label_rates_exclude_archive_only_and_preserve_label_only_position():
    archive, _ = parse_archive_bytes(_archive_bytes(("archive-only", "RB", [_season(2020)])), include_ktc=False)
    labels = parse_label_bytes(_labels([_label_row("label-only", 2020, position="QB")]))
    result = prepare_evaluation_data(HistoricalBundle(archive, labels, {}, {}))
    available = result.coverage["available_label_population"]
    assert available["count"] == 1
    assert result.coverage["start_eligible_rate_within_available_labels"] == 0.0
    assert available["by_position"]["QB"] == 1
    assert result.coverage["archive_only_key_count"] == 1


def test_full_preseason_coverage_is_explicitly_unrecoverable():
    archive, _ = parse_archive_bytes(_archive_bytes(("p1", "QB", [_season(2020)])), include_ktc=False)
    labels = parse_label_bytes(_labels([_label_row("p1", 2020)]))
    result = prepare_evaluation_data(HistoricalBundle(archive, labels, {}, {}))
    assert result.coverage["full_preseason_population_count"] == {"value": None, "reason": "not_recoverable_from_historical_export"}
    assert result.coverage["full_preseason_coverage"] == {"value": None, "reason": "not_recoverable_from_historical_export"}


def test_load_pinned_bundle_off_checks_fixed_identity_and_schema():
    bundle = load_pinned_bundle(Path(__file__).parents[3], ktc_mode="off")
    assert bundle.identities["archive_member_sha256"]
    assert list(bundle.archive.columns) == ["player_id", "season", "position", "age", "points", "games", "years_exp"]
    assert list(bundle.labels.columns) == [
        "player_id", "season", "start_value", "end_value", "position", "age", "games", "points",
        "start_evidence", "end_evidence", "start_trades_window", "end_trades_window",
        "end_value_next_board", "end_evidence_next_board",
    ]


def test_wrong_member_envelope_and_missing_physical_csv_column_refuse():
    payload = json.dumps({"players": []}).encode()
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("other.json", payload)
    with pytest.raises(ContractRefusal):
        parse_archive_bytes(output.getvalue(), include_ktc=False)
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("training-data.json", json.dumps({"rows": []}).encode())
    with pytest.raises(ContractRefusal):
        parse_archive_bytes(output.getvalue(), include_ktc=False)
    row = _label_row("p1", 2020)
    frame = pd.read_csv(io.BytesIO(_labels([row]))).drop(columns=["points"])
    with pytest.raises(ContractRefusal):
        parse_label_bytes(frame.to_csv(index=False).encode())


@pytest.mark.parametrize("field,value", [("season", 2020.5), ("start_trades_window", 4.5), ("end_trades_window", 2.2), ("start_value", float("inf"))])
def test_malformed_nonintegral_and_nonfinite_label_values_refuse(field, value):
    row = _label_row("p1", 2020)
    if field == "season":
        row[1] = value
    elif field == "start_value":
        row[2] = value
    elif field == "start_trades_window":
        row[10] = value
    else:
        row[11] = value
    with pytest.raises(ContractRefusal):
        parse_label_bytes(_labels([row]))


def test_unknown_position_and_negative_trade_count_refuse():
    row = _label_row("p1", 2020, position="K")
    with pytest.raises(ContractRefusal):
        parse_label_bytes(_labels([row]))
    row = _label_row("p1", 2020)
    row[10] = -1
    with pytest.raises(ContractRefusal):
        parse_label_bytes(_labels([row]))


def test_negative_points_and_zero_point_zero_game_nfl_rows_survive():
    archive_raw = _archive_bytes(
        ("negative", "QB", [_season(2020, points=-2.0, games=1, years_exp=1)]),
        ("unknown", "RB", [_season(2020, points=0.0, games=0, years_exp=-1)]),
        ("pre", "WR", [_season(2020, points=0.0, games=0, years_exp=0)]),
    )
    archive, _ = parse_archive_bytes(archive_raw, include_ktc=False)
    labels = parse_label_bytes(_labels([
        _label_row("negative", 2020, points=-2.0, games=1),
        _label_row("unknown", 2020, points=0.0, position="RB", games=0),
        _label_row("pre", 2020, points=0.0, position="WR", games=0),
    ]))
    result = prepare_evaluation_data(HistoricalBundle(archive, labels, {}, {}))
    assert result.primary_scored["player_id"].tolist() == ["negative", "unknown"]
    assert result.joined.loc[result.joined["player_id"] == "pre", "pre_nfl"].item()
    assert result.coverage["negative_point_count"] == 1


@pytest.mark.parametrize("field", ["position", "games", "age", "points"])
def test_archive_csv_conflicts_refuse_before_eligibility_filter(field):
    archive, _ = parse_archive_bytes(_archive_bytes(("p1", "QB", [_season(2020)])), include_ktc=False)
    row = _label_row("p1", 2020)
    index = {"position": 4, "age": 5, "games": 6, "points": 7}[field]
    row[index] = {"position": "RB", "age": 99.0, "games": 9, "points": 99.0}[field]
    labels = parse_label_bytes(_labels([row]))
    with pytest.raises(ContractRefusal):
        prepare_evaluation_data(HistoricalBundle(archive, labels, {}, {}))


def test_thin_positive_endpoint_is_diagnostic_only():
    archive, _ = parse_archive_bytes(_archive_bytes(("p1", "QB", [_season(2020)])), include_ktc=False)
    labels = parse_label_bytes(_labels([_label_row("p1", 2020, end=1200, end_trades=4)]))
    result = prepare_evaluation_data(HistoricalBundle(archive, labels, {}, {}))
    assert result.primary_scored.empty
    assert result.thin_endpoint["player_id"].tolist() == ["p1"]


def test_six_season_fixture_is_sorted_and_key_hash_is_deterministic():
    seasons = [_season(year) for year in range(2020, 2026)]
    archive, _ = parse_archive_bytes(_archive_bytes(("p1", "QB", list(reversed(seasons)))), include_ktc=False)
    labels = parse_label_bytes(_labels([_label_row("p1", year) for year in reversed(range(2020, 2026))]))
    result = prepare_evaluation_data(HistoricalBundle(archive, labels, {}, {}))
    keys = list(result.primary_scored[["season", "player_id"]].itertuples(index=False, name=None))
    assert keys == [(year, "p1") for year in range(2020, 2026)]
    assert result.coverage["primary_scored_population"]["key_sha256"] == result.coverage["key_hashes"]["common"]["sha256"]
