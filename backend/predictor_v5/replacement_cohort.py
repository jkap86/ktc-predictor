"""Verified training inputs and the exact matched replacement cohort."""

import csv
import io
import json
import math
import numbers
import re
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

from predictor_v5.data import build_dataset, load_season_stats
from predictor_v5.labels import CsvLabels, KtcLabels
from predictor_v5.replacement_ktc import _read
from predictor_v5.replacement_mfl import STAT_COLUMNS, TARGET_COLUMNS

TRAIN_SEASONS = [2021, 2022, 2023, 2024, 2025]
POSITIONS = ["QB", "RB", "TE", "WR"]
START_COLUMNS = [
    "season",
    "player_id",
    "name",
    "position",
    "team",
    "birth_date",
    "bc_value",
    "bc_trades",
    "bc_date",
    "bc_model_version",
    "ktc_value",
    "ktc_date",
]
PREDICTION_COLUMNS = ["player_id", "season", "fitted_start", "predicted_end"]
COHORT_COLUMNS = [
    "player_id",
    "season",
    "position",
    "age",
    "games",
    "points",
    "ktc_start",
    "ktc_actual",
    "ktc_pred",
    "mfl_start",
    "mfl_actual",
    "mfl_pred",
]


def verify_common_stats(ktc, mfl):
    """Reject disagreement in any shared player-season fact."""
    keys = ["player_id", "season"]
    for frame in (ktc, mfl):
        if (
            not set(STAT_COLUMNS).issubset(frame.columns)
            or frame[keys].isna().any().any()
            or frame.duplicated(keys).any()
        ):
            raise ValueError("invalid stats keys")
    joined = ktc[STAT_COLUMNS].merge(
        mfl[STAT_COLUMNS], on=keys, suffixes=("_ktc", "_mfl"), validate="one_to_one"
    )
    if any(
        not joined[column + "_ktc"].eq(joined[column + "_mfl"]).all()
        for column in STAT_COLUMNS[2:]
    ):
        raise ValueError("common stats disagree")


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _safe_member(member):
    if not isinstance(member, str) or not member or "\\" in member:
        return False
    path = PurePosixPath(member)
    return not path.is_absolute() and all(part not in ("", ".", "..") for part in path.parts)


def _validate_archive(data, member):
    if not _safe_member(member):
        raise ValueError("invalid training archive member")
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            if archive.namelist() != [member]:
                raise ValueError("invalid training archive members")
            info = archive.getinfo(member)
            if info.is_dir() or info.flag_bits & 1 or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError("invalid training archive member")
            envelope = json.loads(
                archive.read(info),
                object_pairs_hook=_pairs,
                parse_constant=lambda _: (_ for _ in ()).throw(
                    ValueError("invalid numeric constant")
                ),
            )
    except (zipfile.BadZipFile, KeyError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid training archive") from exc
    if not isinstance(envelope, dict) or set(envelope) != {"players"} or not isinstance(envelope["players"], list):
        raise ValueError("invalid training players envelope")
    seen_players, seen_keys, rows = set(), set(), []
    for player in envelope["players"]:
        if not isinstance(player, dict):
            raise ValueError("invalid training player")  # noqa: TRY004
        player_id, position, seasons = player.get("player_id"), player.get("position"), player.get("seasons")
        if (
            not isinstance(player_id, str)
            or not player_id.strip()
            or player_id in seen_players
            or position not in POSITIONS
            or not isinstance(seasons, list)
        ):
            raise ValueError("invalid or duplicate training player")
        seen_players.add(player_id)
        for season in seasons:
            if not isinstance(season, dict) or type(season.get("year")) is not int:
                raise ValueError("invalid training season")
            year = season["year"]
            if year >= 2026:
                raise ValueError("future training season")
            key = (player_id, year)
            if key in seen_keys:
                raise ValueError("duplicate training player season")
            seen_keys.add(key)
            age, years_exp = season.get("age"), season.get("years_exp")
            start, end = season.get("start_ktc"), season.get("end_ktc")
            games, points = season.get("games_played"), season.get("fantasy_points")
            if not all(_number(value) for value in (age, years_exp, start, end, games, points)):
                raise ValueError("invalid training numeric value")
            if int(years_exp) != years_exp or int(games) != games or games < 0:
                raise ValueError("invalid training integer")
            if not (0 <= start <= 9999 and 0 <= end <= 9999):
                raise ValueError("invalid KTC training value")
            rows.append((player_id, year, position, float(age), int(games), float(points)))
    return pd.DataFrame(rows, columns=STAT_COLUMNS)


def _strict_integer(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]+", value):
        raise ValueError("invalid MFL integer")
    result = int(value)
    if result > np.iinfo(np.int64).max:
        raise ValueError("invalid MFL integer overflow")
    return result


def _validate_mfl_csv(data, archive_stats):
    try:
        text = data.decode("utf-8")
        records = list(csv.reader(io.StringIO(text, newline=""), strict=True))
        if not records or any(len(record) != len(TARGET_COLUMNS) for record in records):
            raise ValueError("invalid MFL CSV record width")
        frame = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False)
    except (UnicodeError, csv.Error, pd.errors.ParserError) as exc:
        raise ValueError("invalid MFL CSV") from exc
    if list(frame.columns) != TARGET_COLUMNS:
        raise ValueError("invalid MFL CSV schema")
    if frame["player_id"].map(lambda value: not value.strip()).any():
        raise ValueError("invalid MFL player ID")
    for column in ("season", "games", "start_trades_window", "end_trades_window"):
        frame[column] = frame[column].map(_strict_integer)
    if frame.duplicated(["player_id", "season"]).any():
        raise ValueError("duplicate MFL player season")
    if (frame["season"] >= 2026).any():
        raise ValueError("future MFL training season")
    if not frame["position"].isin(POSITIONS).all():
        raise ValueError("invalid MFL position")
    optional = {"end_value_next_board", "end_evidence_next_board"}
    numeric = [
        "start_value", "end_value", "age", "points", "start_evidence", "end_evidence",
        "end_value_next_board", "end_evidence_next_board",
    ]
    for column in numeric:
        original = frame[column]
        values = pd.to_numeric(original, errors="coerce")
        valid = np.isfinite(values) | ((original == "") if column in optional else False)
        if not valid.all():
            raise ValueError("invalid MFL numeric value")
        frame[column] = values
    if (
        (frame[["start_value", "end_value", "start_evidence", "end_evidence"]] < 0).any().any()
        or (frame[["start_trades_window", "end_trades_window", "games"]] < 0).any().any()
        or any((frame[column].dropna() < 0).any() for column in optional)
    ):
        raise ValueError("invalid MFL numeric value")
    joined = frame[STAT_COLUMNS].merge(
        archive_stats, on=["player_id", "season"], how="left",
        suffixes=("", "_archive"), validate="one_to_one",
    )
    exact = all(
        joined[column].eq(joined[column + "_archive"]).all()
        for column in ("position", "games")
    )
    close = all(
        np.isclose(
            joined[column].to_numpy(dtype=float),
            joined[column + "_archive"].to_numpy(dtype=float),
            rtol=0,
            atol=1e-12,
        ).all()
        for column in ("age", "points")
    )
    if not exact or not close:
        raise ValueError("MFL training stats disagree")
    return frame


def _validate_training_contract(manifest):
    if not isinstance(manifest, dict):
        raise ValueError("invalid training manifest")  # noqa: TRY004
    training = manifest.get("training")
    if not isinstance(training, dict) or set(training) != {"archive", "ktc_member", "mfl_labels", "seasons", "positions"}:
        raise ValueError("invalid training manifest")
    if training["seasons"] != TRAIN_SEASONS or training["positions"] != POSITIONS:
        raise ValueError("invalid training coverage contract")
    model, filters = manifest.get("model"), manifest.get("filters")
    if not isinstance(model, dict) or model.get("ktc_scale_max") != 9999 or model.get("mfl_scale_max") != 10000:
        raise ValueError("invalid training scale contract")
    if filters != {
        "ktc": {"minimum_exclusive": 0, "maximum_exclusive": 9999},
        "mfl": {"minimum_exclusive": 0, "minimum_start_trades": 5, "minimum_end_trades": 5},
    }:
        raise ValueError("invalid training filter contract")
    return training


def _validate_training_frame(frame, source):
    if frame.duplicated(["player_id", "season"]).any():
        raise ValueError(f"duplicate {source} training key")
    numeric = frame[["age", "games", "points", "start_value", "end_value"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"invalid {source} training numeric value")
    actual = set(zip(frame["season"], frame["position"]))
    missing = [(year, position) for year in TRAIN_SEASONS for position in POSITIONS if (year, position) not in actual]
    if missing:
        raise ValueError(f"{source} training coverage missing {missing[:5]}")


def prepare_training(root: Path, manifest: dict) -> dict[str, pd.DataFrame]:
    """Verify captured training bytes and return sorted, unfitted source frames."""
    training = _validate_training_contract(manifest)
    archive_bytes = _read(root, training["archive"])
    mfl_bytes = _read(root, training["mfl_labels"])
    archive_stats = _validate_archive(archive_bytes, training["ktc_member"])
    _validate_mfl_csv(mfl_bytes, archive_stats)
    with tempfile.TemporaryDirectory() as directory:
        temp = Path(directory)
        archive_path, mfl_path = temp / "training.zip", temp / "mfl.csv"
        archive_path.write_bytes(archive_bytes)
        mfl_path.write_bytes(mfl_bytes)
        stats = load_season_stats(archive_path)
        ktc = build_dataset(stats, KtcLabels(zip_path=archive_path))
        mfl = build_dataset(
            stats,
            CsvLabels(path=mfl_path, name="mfl", scale_max=10000.0, min_trades=5),
        )
    result = {
        "ktc": ktc[(ktc["season"].isin(TRAIN_SEASONS)) & (ktc["start_value"] < 9999) & (ktc["end_value"] < 9999)],
        "mfl": mfl[mfl["season"].isin(TRAIN_SEASONS)],
    }
    for source, frame in result.items():
        frame = frame.sort_values(["season", "player_id"], kind="stable").reset_index(drop=True)
        _validate_training_frame(frame, source)
        result[source] = frame
    return result


def _valid_key_frame(frame, *, schema=None, label="input"):
    if not isinstance(frame, pd.DataFrame) or (schema is not None and list(frame.columns) != schema):
        raise ValueError(f"invalid {label} schema")
    if not {"player_id", "season"}.issubset(frame.columns):
        raise ValueError(f"invalid {label} keys")
    if frame.duplicated(["player_id", "season"]).any():
        raise ValueError(f"duplicate {label} keys")
    if not frame["player_id"].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError(f"invalid {label} player ID")
    if not frame["season"].map(lambda value: isinstance(value, numbers.Integral) and not isinstance(value, bool) and value == 2026).all():
        raise ValueError(f"invalid {label} season")


def _finite(frame, column, *, nonnegative=False, positive=False, nullable=False):
    def valid(value):
        if nullable and pd.isna(value):
            return True
        if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
            return False
        return (value > 0 if positive else value >= 0) if (positive or nonnegative) else True
    if not frame[column].map(valid).all():
        raise ValueError(f"invalid {column}")


def _validate_target(frame, source):
    schema = TARGET_COLUMNS[:8] if source == "ktc" else TARGET_COLUMNS
    _valid_key_frame(frame, schema=schema, label=f"{source} target")
    if not frame["position"].map(lambda value: isinstance(value, str) and value in POSITIONS).all():
        raise ValueError("invalid target position")
    for column in ("age", "points"):
        _finite(frame, column)
    _finite(frame, "games", nonnegative=True)
    if not frame["games"].map(lambda value: not isinstance(value, bool) and isinstance(value, numbers.Integral)).all():
        raise ValueError("invalid target games")
    for column in ("start_value", "end_value"):
        _finite(frame, column, positive=True)
    if source == "ktc":
        if not (frame[["start_value", "end_value"]] < 9999).all().all():
            raise ValueError("invalid KTC scale")
    else:
        for column in ("start_evidence", "end_evidence"):
            _finite(frame, column, nonnegative=True)
        for column in ("start_trades_window", "end_trades_window"):
            _finite(frame, column, nonnegative=True)
            if not frame[column].map(lambda value: not isinstance(value, bool) and isinstance(value, numbers.Integral)).all():
                raise ValueError("invalid MFL trades")
            if (frame[column] < 5).any():
                raise ValueError("insufficient MFL trades")
        for column in ("end_value_next_board", "end_evidence_next_board"):
            _finite(frame, column, nonnegative=True, nullable=True)


def _key_set(frame):
    return set(frame[["season", "player_id"]].itertuples(index=False, name=None))


def matched_cohort(starts, ktc, mfl, predictions):
    """Return the sorted five-way intersection and deterministic exclusions."""
    if not isinstance(predictions, dict) or set(predictions) != {"ktc", "mfl"}:
        raise ValueError("invalid predictions")
    _valid_key_frame(starts, schema=START_COLUMNS, label="starts")
    for column in ("ktc_value", "bc_value"):
        _finite(starts, column, nullable=True)
    _finite(starts, "bc_trades", nonnegative=True, nullable=True)
    _validate_target(ktc, "ktc")
    _validate_target(mfl, "mfl")
    verify_common_stats(ktc, mfl)
    for source, frame in predictions.items():
        _valid_key_frame(frame, schema=PREDICTION_COLUMNS, label=f"{source} prediction")
        _finite(frame, "fitted_start", positive=True)
        _finite(frame, "predicted_end", nonnegative=True)

    start_mask = (
        starts["ktc_value"].notna()
        & starts["bc_value"].notna()
        & starts["bc_trades"].notna()
        & starts["ktc_value"].gt(0)
        & starts["ktc_value"].lt(9999)
        & starts["bc_value"].gt(0)
        & starts["bc_trades"].ge(5)
    )
    valid_starts = starts.loc[start_mask].copy()
    for source, target, value, frozen_value in (
        ("ktc", ktc, "start_value", "ktc_value"),
        ("mfl", mfl, "start_value", "bc_value"),
    ):
        shared = target.merge(
            valid_starts[["season", "player_id", frozen_value, "bc_trades"]],
            on=["season", "player_id"],
            how="inner",
            validate="one_to_one",
        )
        if not shared[value].eq(shared[frozen_value]).all():
            raise ValueError(f"{source} target start differs from frozen starts")
        if source == "mfl" and not shared["start_trades_window"].eq(shared["bc_trades"]).all():
            raise ValueError("mfl target start trades differ from frozen starts")
    inputs = {
        "starts": valid_starts,
        "ktc": ktc,
        "mfl": mfl,
        "ktc_predictions": predictions["ktc"],
        "mfl_predictions": predictions["mfl"],
    }
    sets = {name: _key_set(frame) for name, frame in inputs.items()}
    union = set().union(*sets.values())
    matched = set.intersection(*sets.values()) if sets else set()
    coverage = {
        "input_counts": {name: len(values) for name, values in sets.items()},
        "matched_count": len(matched),
        "excluded_keys": {
            name: [list(key) for key in sorted(union - values)]
            for name, values in sets.items()
        },
    }
    if not matched:
        return pd.DataFrame(columns=COHORT_COLUMNS), coverage

    keys = ["season", "player_id"]
    selected = pd.DataFrame(sorted(matched), columns=keys)
    result = selected.merge(
        valid_starts[keys + ["ktc_value", "bc_value"]], on=keys, validate="one_to_one"
    )
    result = result.merge(
        ktc[TARGET_COLUMNS[:8]].rename(
            columns={"start_value": "ktc_target_start", "end_value": "ktc_actual"}
        ),
        on=keys,
        validate="one_to_one",
    )
    result = result.merge(
        mfl[keys + ["start_value", "end_value"]].rename(
            columns={"start_value": "mfl_target_start", "end_value": "mfl_actual"}
        ),
        on=keys,
        validate="one_to_one",
    )
    if not result["ktc_value"].eq(result["ktc_target_start"]).all() or not result["bc_value"].eq(result["mfl_target_start"]).all():
        raise ValueError("target start differs from frozen starts")
    for source, start_column in (("ktc", "ktc_value"), ("mfl", "bc_value")):
        result = result.merge(
            predictions[source].rename(
                columns={"fitted_start": f"{source}_fitted", "predicted_end": f"{source}_own_predicted"}
            ),
            on=keys,
            validate="one_to_one",
        )
        result[f"{source}_start"] = result[start_column]
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            result[f"{source}_pred"] = (
                result[start_column].to_numpy(dtype=float)
                * result[f"{source}_own_predicted"].to_numpy(dtype=float)
                / result[f"{source}_fitted"].to_numpy(dtype=float)
            )
        if not np.isfinite(result[f"{source}_pred"]).all():
            raise ValueError("invalid relative prediction")
    return result[COHORT_COLUMNS].sort_values(keys, kind="stable").reset_index(drop=True), coverage
