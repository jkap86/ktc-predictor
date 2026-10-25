import hashlib
import io
import json
import socket
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from predictor_v5.replacement_ktc import load_ktc_target, select_end_observations


def bundle(tmp_path):
    def put(name, data):
        (tmp_path / name).write_bytes(data)
        return {"path": name, "sha256": hashlib.sha256(data).hexdigest()}

    canonical = lambda obj: (
        json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    raw = put(
        "raw.csv",
        b"format,ktc_id,sleeper_id,date,sf_value\ndynasty,1,p,2027-01-10,20\n",
    )
    target = put(
        "target.csv",
        b"player_id,season,start_value,end_value,position,age,games,points\np,2026,10,20,QB,25,17,300\n",
    )
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr(
            "training-data.json",
            json.dumps(
                {
                    "players": [
                        {
                            "player_id": "p",
                            "position": "QB",
                            "seasons": [
                                {
                                    "year": 2026,
                                    "age": 25,
                                    "games_played": 17,
                                    "fantasy_points": 300,
                                }
                            ],
                        }
                    ]
                }
            ),
        )
    stats_id = put("stats.zip", archive.getvalue())
    mapping = put("mapping.csv", b"format,ktc_id,sleeper_id\ndynasty,1,p\ndynasty,2,q\n")
    sql = put("query.sql", b"query")
    extractor = put("extractor.py", b"extractor")
    code = put("mapping.py", b"mapping")
    ids = {
        "raw_end": raw,
        "target_csv": target,
        "target_stats": stats_id,
        "sql": sql,
        "extractor": extractor,
        "mapping_code": code,
        "mapping_data": mapping,
    }
    side = {
        "schema_version": 1,
        "provider": "local-thelab",
        "reviewer": "reviewer",
        "starts": {
            "path": "backend/predictor_v5/data/start_values_2026.csv",
            "sha256": "6366b85300c4f9532f0866c602c6d262baaccf6c3a84b9252a6d2e8d46c541bb",
        },
        "identities": ids,
        "observation_dates": [{"player_id": "p", "date": "2027-01-10"}],
        "coverage": {
            "raw_players": 1,
            "target_players": 1,
            "missing_start_keys": [],
            "excluded_raw_players": [],
        },
    }
    ids = dict(ids, target_sidecar=put("side.json", canonical(side)))
    starts = pd.DataFrame([{"season": 2026, "player_id": "p", "ktc_value": 10}])
    stats = pd.DataFrame(
        [
            {
                "season": 2026,
                "player_id": "p",
                "position": "QB",
                "age": 25.0,
                "games": 17,
                "points": 300.0,
            }
        ]
    )
    return {"ktc": ids}, starts, stats, side


def replace_side(tmp_path, registration, side):
    data = (json.dumps(side, sort_keys=True, separators=(",", ":")) + "\n").encode()
    (tmp_path / "side.json").write_bytes(data)
    registration["ktc"]["target_sidecar"]["sha256"] = hashlib.sha256(data).hexdigest()


def replace_input(tmp_path, registration, side, key, data):
    identity = registration["ktc"][key]
    (tmp_path / identity["path"]).write_bytes(data)
    identity["sha256"] = hashlib.sha256(data).hexdigest()
    side["identities"][key] = dict(identity)
    replace_side(tmp_path, registration, side)


@pytest.mark.parametrize("key", ["raw_end", "target_csv"])
@pytest.mark.parametrize("mutation", ["leading", "trailing", "short", "blank", "malformed"])
def test_physical_csv_records_refuse_before_pandas(tmp_path, key, mutation):
    registration, starts, stats, side = bundle(tmp_path)
    header, row = (tmp_path / registration["ktc"][key]["path"]).read_bytes().splitlines()
    bad = {
        "leading": b"IGNORED," + row,
        "trailing": row + b",IGNORED",
        "short": row.rsplit(b",", 1)[0],
        "blank": row + b"\n",
        "malformed": b'"' + row,
    }[mutation]
    replace_input(tmp_path, registration, side, key, header + b"\n" + bad + b"\n")
    with pytest.raises(ValueError, match="CSV"):
        load_ktc_target(registration, starts, stats, tmp_path)


@pytest.mark.parametrize("key", ["raw_end", "target_csv"])
def test_quoted_csv_fields_are_not_counted_as_extra_records(tmp_path, key):
    registration, starts, stats, side = bundle(tmp_path)
    path = tmp_path / registration["ktc"][key]["path"]
    header, row = path.read_bytes().splitlines()
    quoted = b",".join(b'"' + field + b'"' for field in row.split(b","))
    replace_input(tmp_path, registration, side, key, header + b"\n" + quoted + b"\n")
    assert load_ktc_target(registration, starts, stats, tmp_path).player_id.tolist() == ["p"]


@pytest.mark.parametrize("player_id", ['p,quoted', 'p\nquoted', 'p"quoted'])
def test_quoted_delimiters_in_identifiers_round_trip(tmp_path, player_id):
    registration, starts, stats, side = bundle(tmp_path)
    for key in ("raw_end", "target_csv", "mapping_data"):
        frame = pd.read_csv(tmp_path / registration["ktc"][key]["path"], dtype=str)
        column = "player_id" if key == "target_csv" else "sleeper_id"
        frame.loc[frame[column] == "p", column] = player_id
        replace_input(tmp_path, registration, side, key, frame.to_csv(index=False).encode())
    for frame in (starts, stats):
        frame.loc[0, "player_id"] = player_id
    side["observation_dates"][0]["player_id"] = player_id
    replace_archive(tmp_path, registration, side, {"players": [{
        "player_id": player_id, "position": "QB", "seasons": [{
            "year": 2026, "age": 25, "games_played": 17, "fantasy_points": 300,
        }],
    }]})
    assert load_ktc_target(registration, starts, stats, tmp_path).player_id.tolist() == [player_id]


def replace_archive(tmp_path, registration, side, envelope):
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("training-data.json", json.dumps(envelope))
    data = archive.getvalue()
    (tmp_path / "stats.zip").write_bytes(data)
    registration["ktc"]["target_stats"]["sha256"] = hashlib.sha256(data).hexdigest()
    side["identities"]["target_stats"] = registration["ktc"]["target_stats"]
    replace_side(tmp_path, registration, side)


@pytest.mark.parametrize(
    "field,bad",
    [
        ("player_id", 1),
        ("player_id", ""),
        ("position", "K"),
        ("age", True),
        ("age", float("inf")),
        ("age", "25"),
        ("games_played", True),
        ("games_played", -1),
        ("games_played", 1.5),
        ("fantasy_points", True),
        ("fantasy_points", float("nan")),
        ("fantasy_points", "1"),
        ("year", True),
        ("year", 2025),
    ],
)
def test_invalid_unmatched_archive_facts_rejected(tmp_path, field, bad):
    registration, starts, stats, side = bundle(tmp_path)
    player = {
        "player_id": "q",
        "position": "RB",
        "seasons": [{"year": 2026, "age": 25, "games_played": 1, "fantasy_points": 1}],
    }
    (player if field in ("player_id", "position") else player["seasons"][0])[field] = (
        bad
    )
    valid = {
        "player_id": "p",
        "position": "QB",
        "seasons": [
            {"year": 2026, "age": 25, "games_played": 17, "fantasy_points": 300}
        ],
    }
    replace_archive(tmp_path, registration, side, {"players": [valid, player]})
    with pytest.raises(ValueError):
        load_ktc_target(registration, starts, stats, tmp_path)


@pytest.mark.parametrize(
    "envelope",
    [
        None,
        {},
        {"players": {}},
        {"players": [None]},
        {"players": [{"player_id": "q", "position": "RB", "seasons": None}]},
    ],
)
def test_malformed_archive_envelope_rejected(tmp_path, envelope):
    registration, starts, stats, side = bundle(tmp_path)
    replace_archive(tmp_path, registration, side, envelope)
    with pytest.raises(ValueError):
        load_ktc_target(registration, starts, stats, tmp_path)


def test_duplicate_unmatched_archive_player_rejected(tmp_path):
    registration, starts, stats, side = bundle(tmp_path)
    q = {"player_id": "q", "position": "RB", "seasons": [{"year": 2026, "age": 25}]}
    replace_archive(tmp_path, registration, side, {"players": [q, q]})
    with pytest.raises(ValueError):
        load_ktc_target(registration, starts, stats, tmp_path)


@pytest.mark.parametrize("facts", [{}, {"games_played": None, "fantasy_points": None}])
def test_missing_or_null_games_and_points_default_to_zero(tmp_path, facts):
    registration, starts, stats, side = bundle(tmp_path)
    raw = (tmp_path / "target.csv").read_bytes().replace(b",17,300\n", b",0,0\n")
    (tmp_path / "target.csv").write_bytes(raw)
    registration["ktc"]["target_csv"]["sha256"] = hashlib.sha256(raw).hexdigest()
    side["identities"]["target_csv"] = registration["ktc"]["target_csv"]
    stats.loc[0, ["games", "points"]] = 0
    season = {"year": 2026, "age": 25, **facts}
    replace_archive(
        tmp_path,
        registration,
        side,
        {"players": [{"player_id": "p", "position": "QB", "seasons": [season]}]},
    )
    target = load_ktc_target(registration, starts, stats, tmp_path)
    assert target.loc[0, "games"] == 0 and target.loc[0, "points"] == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("raw_players", True),
        ("raw_players", 1.0),
        ("target_players", True),
        ("target_players", 1.0),
    ],
)
def test_coverage_counts_require_integers(tmp_path, field, value):
    registration, starts, stats, side = bundle(tmp_path)
    side["coverage"][field] = value
    replace_side(tmp_path, registration, side)
    with pytest.raises(ValueError):
        load_ktc_target(registration, starts, stats, tmp_path)


@pytest.mark.parametrize("value", [2026.0, True])
def test_missing_key_season_requires_integer(tmp_path, value):
    registration, starts, stats, side = bundle(tmp_path)
    starts = starts.iloc[0:0]
    side["coverage"]["missing_start_keys"] = [[value, "p"]]
    replace_side(tmp_path, registration, side)
    with pytest.raises(ValueError):
        load_ktc_target(registration, starts, stats, tmp_path)


def test_empty_bundle_loads(tmp_path):
    registration, starts, stats, side = bundle(tmp_path)
    for key, name, data in [
        ("raw_end", "raw.csv", b"format,ktc_id,sleeper_id,date,sf_value\n"),
        (
            "target_csv",
            "target.csv",
            b"player_id,season,start_value,end_value,position,age,games,points\n",
        ),
    ]:
        (tmp_path / name).write_bytes(data)
        registration["ktc"][key]["sha256"] = hashlib.sha256(data).hexdigest()
        side["identities"][key] = registration["ktc"][key]
    side["observation_dates"] = []
    side["coverage"] = {
        "raw_players": 0,
        "target_players": 0,
        "missing_start_keys": [],
        "excluded_raw_players": [],
    }
    replace_archive(tmp_path, registration, side, {"players": []})
    result = load_ktc_target(registration, starts.iloc[0:0], stats.iloc[0:0], tmp_path)
    assert result.empty
    assert list(result.columns) == [
        "player_id",
        "season",
        "start_value",
        "end_value",
        "position",
        "age",
        "games",
        "points",
    ]


def test_registered_bundle_matches_raw_and_stats(tmp_path):
    registration, starts, stats, _ = bundle(tmp_path)
    target = load_ktc_target(registration, starts, stats, tmp_path)
    assert target.to_dict("records") == [
        {
            "player_id": "p",
            "season": 2026,
            "start_value": 10.0,
            "end_value": 20.0,
            "position": "QB",
            "age": 25.0,
            "games": 17,
            "points": 300.0,
        }
    ]


def test_sidecar_records_excluded_raw_player_date(tmp_path):
    registration, starts, stats, side = bundle(tmp_path)
    raw = b"format,ktc_id,sleeper_id,date,sf_value\ndynasty,1,p,2027-01-10,20\ndynasty,2,q,2026-12-28,30\n"
    (tmp_path / "raw.csv").write_bytes(raw)
    identity = registration["ktc"]["raw_end"]
    identity["sha256"] = hashlib.sha256(raw).hexdigest()
    side["identities"]["raw_end"] = identity
    side["observation_dates"].append({"player_id": "q", "date": "2026-12-28"})
    side["coverage"] = {
        "raw_players": 2,
        "target_players": 1,
        "missing_start_keys": [],
        "excluded_raw_players": ["q"],
    }
    data = (json.dumps(side, sort_keys=True, separators=(",", ":")) + "\n").encode()
    (tmp_path / "side.json").write_bytes(data)
    registration["ktc"]["target_sidecar"]["sha256"] = hashlib.sha256(data).hexdigest()
    assert len(load_ktc_target(registration, starts, stats, tmp_path)) == 1


def test_offline_load_cannot_open_network(tmp_path, monkeypatch):
    registration, starts, stats, _ = bundle(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("network access")

    monkeypatch.setattr(socket, "socket", forbidden)
    assert len(load_ktc_target(registration, starts, stats, tmp_path)) == 1


def test_target_season_must_be_integer_literal(tmp_path):
    registration, starts, stats, side = bundle(tmp_path)
    data = (tmp_path / "target.csv").read_bytes().replace(b",2026,", b",2026.0,")
    (tmp_path / "target.csv").write_bytes(data)
    registration["ktc"]["target_csv"]["sha256"] = hashlib.sha256(data).hexdigest()
    side["identities"]["target_csv"] = registration["ktc"]["target_csv"]
    side_data = (
        json.dumps(side, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    (tmp_path / "side.json").write_bytes(side_data)
    registration["ktc"]["target_sidecar"]["sha256"] = hashlib.sha256(
        side_data
    ).hexdigest()
    with pytest.raises(ValueError):
        load_ktc_target(registration, starts, stats, tmp_path)


def test_frozen_sql_contract():
    base = Path(__file__).resolve().parents[1]
    sql = " ".join((base / "sql/ktc_replacement_end.sql").read_text().lower().split())
    assert "select h.format, h.ktc_id, k.sleeper_id, h.date, h.sf_value" in sql
    assert "join ktc_values k using (format, ktc_id)" in sql
    assert "h.format = 'dynasty'" in sql
    assert "k.sleeper_id is not null" in sql and "h.sf_value > 0" in sql
    assert "h.date between '2026-12-28'::date and '2027-01-10'::date" in sql
    assert "distinct" not in sql


def test_mapping_snapshot_has_exact_reviewed_rows_and_physical_schema():
    from predictor_v5 import replacement_ktc

    data = b"format,ktc_id,sleeper_id\ndynasty,1,p\ndynasty,10,q\n"
    assert replacement_ktc.mapping_snapshot(data) == {
        ("dynasty", "1"): "p", ("dynasty", "10"): "q",
    }


@pytest.mark.parametrize("data", [
    b'{}\n',
    b"format,sleeper_id,ktc_id\ndynasty,p,1\n",
    b"format,ktc_id,sleeper_id\nIGNORED,dynasty,1,p\n",
    b"format,ktc_id,sleeper_id\ndynasty,1,p,IGNORED\n",
    b"format,ktc_id,sleeper_id\ndynasty,1\n",
    b"format,ktc_id,sleeper_id\ndynasty,1,p\n\n",
    b'format,ktc_id,sleeper_id\n"dynasty,1,p\n',
    b"format,ktc_id,sleeper_id\ndynasty,1,p\ndynasty,1,p\n",
    b"format,ktc_id,sleeper_id\ndynasty,1,p\ndynasty,1,q\n",
    b"format,ktc_id,sleeper_id\ndynasty,1,p\ndynasty,2,p\n",
    b"format,ktc_id,sleeper_id\ndynasty,10,q\ndynasty,1,p\n",
    b"format,ktc_id,sleeper_id\ndynasty,01,p\n",
    b"format,ktc_id,sleeper_id\ndynasty,1,\n",
    b"format,ktc_id,sleeper_id\nredraft,1,p\n",
    b"format,ktc_id,sleeper_id\n",
    b'format,ktc_id,sleeper_id\n"dynasty",1,p\n',
    b"format,ktc_id,sleeper_id\r\ndynasty,1,p\r\n",
])
def test_mapping_snapshot_rejects_noncanonical_or_ambiguous_rows(data):
    from predictor_v5 import replacement_ktc

    with pytest.raises(ValueError):
        replacement_ktc.mapping_snapshot(data)


@pytest.mark.parametrize("mapping", [
    b"format,ktc_id,sleeper_id\ndynasty,2,q\n",  # absent source
    b"format,ktc_id,sleeper_id\ndynasty,1,q\n",  # changed Sleeper mapping
])
def test_raw_mapping_must_equal_frozen_snapshot(tmp_path, mapping):
    registration, starts, stats, side = bundle(tmp_path)
    replace_input(tmp_path, registration, side, "mapping_data", mapping)
    with pytest.raises(ValueError, match="raw mapping"):
        load_ktc_target(registration, starts, stats, tmp_path)


def test_mapping_bytes_cannot_drift_from_registered_hash(tmp_path):
    registration, starts, stats, _ = bundle(tmp_path)
    (tmp_path / "mapping.csv").write_bytes(b"format,ktc_id,sleeper_id\ndynasty,1,q\n")
    with pytest.raises(ValueError, match="digest differs"):
        load_ktc_target(registration, starts, stats, tmp_path)


def test_sidecar_cannot_substitute_mapping_identity(tmp_path):
    registration, starts, stats, side = bundle(tmp_path)
    side["identities"]["mapping_data"] = {"path": "other.csv", "sha256": "a" * 64}
    replace_side(tmp_path, registration, side)
    with pytest.raises(ValueError, match="sidecar identity mismatch"):
        load_ktc_target(registration, starts, stats, tmp_path)


@pytest.mark.parametrize("field", ["start", "stats", "raw", "sidecar"])
def test_bundle_disagreements_refused(tmp_path, field):
    registration, starts, stats, side = bundle(tmp_path)
    if field == "start":
        starts.loc[0, "ktc_value"] = 11
    elif field == "stats":
        stats.loc[0, "points"] = 301
    elif field == "raw":
        (tmp_path / "raw.csv").write_text("tampered")
    else:
        side["coverage"]["raw_players"] = 2
        data = (json.dumps(side, sort_keys=True, separators=(",", ":")) + "\n").encode()
        (tmp_path / "side.json").write_bytes(data)
        registration["ktc"]["target_sidecar"]["sha256"] = hashlib.sha256(
            data
        ).hexdigest()
    with pytest.raises(ValueError):
        load_ktc_target(registration, starts, stats, tmp_path)


def test_latest_inclusive_and_identical_duplicates():
    raw = pd.DataFrame(
        [
            ["dynasty", "1", "p", "2026-12-28", 10],
            ["dynasty", "1", "p", "2027-01-10", 20],
            ["dynasty", "1", "p", "2027-01-10", 20],
        ],
        columns=["format", "ktc_id", "sleeper_id", "date", "sf_value"],
    )
    result = select_end_observations(raw)
    assert result.to_dict("records") == [
        {"player_id": "p", "date": "2027-01-10", "end_value": 20.0}
    ]


def test_conflicting_older_values_are_rejected():
    raw = pd.DataFrame(
        [
            ["dynasty", "1", "p", "2026-12-28", 10],
            ["dynasty", "2", "p", "2026-12-28", 11],
            ["dynasty", "1", "p", "2027-01-10", 20],
        ],
        columns=["format", "ktc_id", "sleeper_id", "date", "sf_value"],
    )
    with pytest.raises(ValueError):
        select_end_observations(raw)


def test_source_identity_cannot_remap_across_dates():
    raw = pd.DataFrame(
        [
            ["dynasty", "1", "p", "2026-12-28", 10],
            ["dynasty", "1", "q", "2027-01-10", 20],
        ],
        columns=["format", "ktc_id", "sleeper_id", "date", "sf_value"],
    )
    with pytest.raises(ValueError):
        select_end_observations(raw)


def test_boolean_is_not_a_numeric_observation():
    raw = pd.DataFrame(
        [["dynasty", "1", "p", "2027-01-10", True]],
        columns=["format", "ktc_id", "sleeper_id", "date", "sf_value"],
    )
    with pytest.raises(ValueError):
        select_end_observations(raw)


@pytest.mark.parametrize("value", [0, 9999, -1, float("nan"), float("inf")])
def test_invalid_values_rejected(value):
    raw = pd.DataFrame(
        [["dynasty", "1", "p", "2027-01-10", value]],
        columns=["format", "ktc_id", "sleeper_id", "date", "sf_value"],
    )
    with pytest.raises(ValueError):
        select_end_observations(raw)
