"""Behavioral tests for independently verified replacement targets."""

import hashlib
import json
import zipfile

import pandas as pd
import pytest

from predictor_v5 import replacement_mfl
from predictor_v5.replacement_cohort import verify_common_stats
from predictor_v5.replacement_contract import canonical_bytes
from predictor_v5.replacement_mfl import load_mfl_target, load_stats


def test_stats_archive_requires_one_named_member(tmp_path):
    path = tmp_path / "stats.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("other.json", json.dumps({"players": []}))
    with pytest.raises(ValueError):
        load_stats(path)


def archive(path, players):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("training-data.json", json.dumps({"players": players}))


def player(**changes):
    result = {
        "player_id": "p",
        "position": "QB",
        "seasons": [
            {"year": 2026, "age": 25, "games_played": 10, "fantasy_points": 100}
        ],
    }
    result.update(changes)
    return result


@pytest.mark.parametrize(
    "bad",
    [
        player(player_id=" "),
        player(position="K"),
        player(seasons=[{"year": 2025, "age": 25}]),
        player(seasons=[{"year": 2026, "age": True}]),
        player(seasons=[{"year": 2026, "age": 25, "games_played": -1}]),
        player(seasons=[{"year": 2026, "age": 25, "games_played": 1.5}]),
        player(seasons=[{"year": 2026, "age": 25, "fantasy_points": float("nan")}]),
    ],
)
def test_stats_reject_invalid_rows_even_unmatched(tmp_path, bad):
    path = tmp_path / "stats.zip"
    if bad["player_id"] == "p":
        bad = {**bad, "player_id": "other"}
    archive(path, [player(), bad])
    with pytest.raises(ValueError):
        load_stats(path)


def test_stats_defaults_and_empty_frame(tmp_path):
    path = tmp_path / "stats.zip"
    archive(path, [player(seasons=[{"year": 2026, "age": 25}])])
    assert load_stats(path).iloc[0][["games", "points"]].tolist() == [0, 0]
    archive(path, [])
    assert list(load_stats(path).columns) == [
        "player_id",
        "season",
        "position",
        "age",
        "games",
        "points",
    ]
    assert str(load_stats(path).dtypes["season"]) == "int64"
    assert str(load_stats(path).dtypes["age"]) == "float64"


def test_duplicate_json_keys_refused(tmp_path):
    path = tmp_path / "stats.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("training-data.json", '{"players": [], "players": []}')
    with pytest.raises(ValueError):
        load_stats(path)


def test_common_stats_checks_all_common_rows():
    a = pd.DataFrame(
        [["p", 2026, "QB", 25.0, 10, 100.0]],
        columns=["player_id", "season", "position", "age", "games", "points"],
    )
    verify_common_stats(a, a.copy())
    b = a.copy()
    b.loc[0, "points"] = 101
    with pytest.raises(ValueError):
        verify_common_stats(a, b)
    b = pd.concat([a, a.assign(player_id="exclusive")], ignore_index=True)
    verify_common_stats(a, b)


@pytest.mark.parametrize(
    "field,value", [("position", "WR"), ("age", 26), ("games", 9), ("points", 101)]
)
def test_common_stats_each_fact(field, value):
    a = pd.DataFrame(
        [["p", 2026, "QB", 25.0, 10, 100.0]],
        columns=["player_id", "season", "position", "age", "games", "points"],
    )
    b = a.copy()
    b.loc[0, field] = value
    with pytest.raises(ValueError):
        verify_common_stats(a, b)


@pytest.mark.parametrize(
    "bad",
    [
        player(player_id=None),
        player(position=None),
        player(seasons=None),
        player(seasons=[{"year": True, "age": 25}]),
        player(seasons=[{"year": 2026, "age": float("inf")}]),
        player(seasons=[{"year": 2026, "age": 25, "games_played": True}]),
        player(seasons=[{"year": 2026, "age": 25, "fantasy_points": True}]),
        player(seasons=[{"year": 2026, "age": 25}, {"year": 2026, "age": 25}]),
    ],
)
def test_more_archive_invalid_facts(tmp_path, bad):
    path = tmp_path / "stats"
    archive(path, [bad])
    with pytest.raises(ValueError):
        load_stats(path)


def test_duplicate_archive_player(tmp_path):
    path = tmp_path / "stats"
    archive(path, [player(), player()])
    with pytest.raises(ValueError):
        load_stats(path)


def test_mfl_requires_registered_independent_sidecar(tmp_path):
    def identity(name, data):
        (tmp_path / name).write_bytes(data)
        return {"path": name, "sha256": hashlib.sha256(data).hexdigest()}

    ids = {k: identity(k, b"x") for k in ("target_csv", "target_stats", "sql", "build")}
    ids["target_sidecar"] = identity("sidecar", canonical_bytes({"schema_version": 1}))
    with pytest.raises(ValueError):
        load_mfl_target({"mfl": ids}, pd.DataFrame(), pd.DataFrame(), tmp_path)


def test_mfl_valid_target_and_frozen_trade_mismatch(tmp_path):
    def identity(name, data):
        (tmp_path / name).write_bytes(data)
        return {"path": name, "sha256": hashlib.sha256(data).hexdigest()}

    path = tmp_path / "stats"
    archive(path, [player()])
    stats = load_stats(path)
    csv = (
        b"player_id,season,start_value,end_value,position,age,games,points,start_evidence,end_evidence,start_trades_window,end_trades_window,end_value_next_board,end_evidence_next_board\n"
        b"p,2026,100,110,QB,25,10,100,1,2,5,6,,\n"
    )
    ids = {
        "target_csv": identity("target", csv),
        "target_stats": identity("stats", path.read_bytes()),
        "sql": identity("sql", b"sql"),
        "build": identity("build", b"build"),
    }
    side = {
        "schema_version": 1,
        "season": 2026,
        "model_version": "baseline-hl28-v1",
        "verified": True,
        "source": "export",
        "problems": [],
        "start_cutoff": "2026-09-09",
        "end_cutoff": "2027-01-10",
        "timezone": "America/New_York",
        **ids,
    }
    ids["target_sidecar"] = identity("sidecar", canonical_bytes(side))
    starts = pd.DataFrame(
        [
            {
                "player_id": "p",
                "season": 2026,
                "bc_value": 100,
                "bc_trades": 5,
                "bc_model_version": "baseline-hl28-v1",
            }
        ]
    )
    assert load_mfl_target(
        {"mfl": ids}, starts, stats, tmp_path
    ).player_id.tolist() == ["p"]
    starts.loc[0, "bc_trades"] = 4
    with pytest.raises(ValueError):
        load_mfl_target({"mfl": ids}, starts, stats, tmp_path)


@pytest.fixture
def mfl_case(tmp_path):
    def put(name, data):
        (tmp_path / name).write_bytes(data)
        return {"path": name, "sha256": hashlib.sha256(data).hexdigest()}

    path = tmp_path / "stats"
    archive(path, [player()])
    stats = load_stats(path)
    row = {
        "player_id": "p",
        "season": 2026,
        "start_value": 100,
        "end_value": 110,
        "position": "QB",
        "age": 25,
        "games": 10,
        "points": 100,
        "start_evidence": 1,
        "end_evidence": 2,
        "start_trades_window": 5,
        "end_trades_window": 6,
        "end_value_next_board": "",
        "end_evidence_next_board": "",
    }
    ids = {
        "target_stats": put("stats", path.read_bytes()),
        "sql": put("sql", b"sql"),
        "build": put("build", b"build"),
    }
    starts = pd.DataFrame(
        [
            {
                "player_id": "p",
                "season": 2026,
                "bc_value": 100,
                "bc_trades": 5,
                "bc_model_version": "baseline-hl28-v1",
            }
        ]
    )

    def run(frame=None, side_change=None):
        frame = pd.DataFrame([row]) if frame is None else frame
        ids["target_csv"] = put("target", frame.to_csv(index=False).encode())
        side = {
            "schema_version": 1,
            "season": 2026,
            "model_version": "baseline-hl28-v1",
            "verified": True,
            "source": "export",
            "problems": [],
            "start_cutoff": "2026-09-09",
            "end_cutoff": "2027-01-10",
            "timezone": "America/New_York",
            **{k: v for k, v in ids.items() if k != "target_sidecar"},
        }
        if side_change:
            side_change(side)
        ids["target_sidecar"] = put("sidecar", canonical_bytes(side))
        return load_mfl_target({"mfl": ids}, starts, stats, tmp_path)

    return row, run


@pytest.mark.parametrize(
    "column,value",
    [
        ("player_id", " "),
        ("season", 2025),
        ("position", "K"),
        ("start_value", 0),
        ("end_value", float("inf")),
        ("age", "bad"),
        ("games", -1),
        ("games", 1.5),
        ("points", "nan"),
        ("start_evidence", -1),
        ("end_evidence", "bad"),
        ("start_trades_window", 4),
        ("end_trades_window", 4),
        ("end_value_next_board", -1),
        ("end_evidence_next_board", "bad"),
    ],
)
def test_mfl_invalid_target_fields(mfl_case, column, value):
    row, run = mfl_case
    with pytest.raises(ValueError):
        run(pd.DataFrame([{**row, column: value}]))


def test_mfl_column_order_duplicate_key_and_stats_mismatch(mfl_case):
    row, run = mfl_case
    frame = pd.DataFrame([row])
    for bad in (
        frame[frame.columns[::-1]],
        pd.concat([frame, frame]),
        frame.assign(points=99),
        frame.assign(start_value=101),
        frame.assign(start_trades_window=6),
    ):
        with pytest.raises(ValueError):
            run(bad)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("season", 2025),
        ("verified", 1),
        ("source", "other"),
        ("problems", ["x"]),
        ("start_cutoff", "2026-09-10"),
        ("timezone", "UTC"),
        ("target_csv", {"path": "other", "sha256": "0" * 64}),
        ("sql", {"path": "sql", "sha256": "0" * 64}),
        ("build", {"path": "build", "sha256": "0" * 64}),
    ],
)
def test_mfl_sidecar_exact_values(mfl_case, field, value):
    _, run = mfl_case
    with pytest.raises(ValueError):
        run(side_change=lambda side: side.update({field: value}))


def test_mfl_sidecar_extra_and_missing_fields(mfl_case):
    _, run = mfl_case
    for mutation in (
        lambda side: side.update(extra=1),
        lambda side: side.pop("target_stats"),
    ):
        with pytest.raises(ValueError):
            run(side_change=mutation)


def test_registered_bytes_are_parsed_from_capture(mfl_case, monkeypatch):
    _, run = mfl_case
    original = replacement_mfl._read

    def capture_then_change(root, identity):
        data = original(root, identity)
        if identity["path"] == "target":
            (root / "target").write_bytes(b"changed after capture")
        return data

    monkeypatch.setattr(replacement_mfl, "_read", capture_then_change)
    assert run().player_id.tolist() == ["p"]


def test_missing_registered_sidecar_refused_even_if_adjacent_exists(tmp_path):
    def identity(name, data):
        (tmp_path / name).write_bytes(data)
        return {"path": name, "sha256": hashlib.sha256(data).hexdigest()}

    ids = {k: identity(k, b"x") for k in ("target_csv", "target_stats", "sql", "build")}
    identity("target_csv.json", b"adjacent")
    ids["target_sidecar"] = {"path": "missing", "sha256": "0" * 64}
    with pytest.raises(ValueError):
        load_mfl_target({"mfl": ids}, pd.DataFrame(), pd.DataFrame(), tmp_path)


def test_stats_rejects_extra_member_and_missing_envelope(tmp_path):
    path = tmp_path / "stats"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("training-data.json", '{"players": []}')
        z.writestr("extra", "x")
    with pytest.raises(ValueError):
        load_stats(path)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("training-data.json", "{}")
    with pytest.raises(ValueError):
        load_stats(path)


def test_stats_null_values_default_to_zero(tmp_path):
    path = tmp_path / "stats"
    archive(
        path,
        [
            player(
                seasons=[
                    {
                        "year": 2026,
                        "age": 25,
                        "games_played": None,
                        "fantasy_points": None,
                    }
                ]
            )
        ],
    )
    assert load_stats(path).iloc[0][["games", "points"]].tolist() == [0, 0]
