"""Build the versioned Backchannel label CSV from a market-history export.

``build-labels`` checks the raw export before joining it to NFL stats:
required values must be present and non-negative, player-seasons unique,
every expected season present, and (when the export carries the provenance
columns from ``sql/backchannel_labels.sql``) one model version and one start
and end generation per season. Those are integrity errors and stop the build.
Ordinary exclusions (pre-NFL seasons, players with no NFL stats, seasons the
training archive has no stats for, a missing next-season board) are counted
in a coverage report written beside the CSV as ``<name>.provenance.json``.
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from predictor_v5.data import DEFAULT_ZIP, drop_pre_nfl, read_season_stats

DEFAULT_EXPECTED_SEASONS = tuple(range(2019, 2026))

LABEL_EXPORT_COLUMNS = [
    "player_id",
    "season",
    "start_value",
    "end_value",
    "position",
    "age",
    "games",
    "points",
    "start_evidence",
    "end_evidence",
    "start_trades_window",
    "end_trades_window",
    "end_value_next_board",
    "end_evidence_next_board",
]

MARKET_COLUMNS = [
    "player_id",
    "season",
    "start_value",
    "end_value",
    "start_evidence",
    "end_evidence",
    "start_trades_window",
    "end_trades_window",
    "end_value_next_board",
    "end_evidence_next_board",
]
REQUIRED_VALUE_COLUMNS = [
    "start_value",
    "end_value",
    "start_evidence",
    "end_evidence",
    "start_trades_window",
    "end_trades_window",
]
OPTIONAL_VALUE_COLUMNS = ["end_value_next_board", "end_evidence_next_board"]
PROVENANCE_COLUMNS = [
    "model_version",
    "start_generation_id",
    "start_revision",
    "end_generation_id",
    "end_revision",
    "next_generation_id",
    "next_revision",
    "start_board_rows",
    "end_board_rows",
    "exported_at",
]
STAT_COLUMNS = ["player_id", "season", "position", "age", "games", "points"]
STAT_INPUT_COLUMNS = [*STAT_COLUMNS, "years_exp"]


def _require_columns(frame: pd.DataFrame, columns: list[str], source: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} is missing columns: {', '.join(missing)}")


def _reject_duplicates(frame: pd.DataFrame, source: str) -> None:
    duplicates = frame.duplicated(["player_id", "season"], keep=False)
    if duplicates.any():
        keys = frame.loc[duplicates, ["player_id", "season"]].drop_duplicates()
        sample = ", ".join(f"{row.player_id}/{row.season}" for row in keys.itertuples(index=False))
        raise ValueError(f"duplicate player-season in {source}: {sample}")


def build_label_frame(values: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """Join market values to NFL stats without filtering low-evidence rows."""
    _require_columns(values, MARKET_COLUMNS, "market export")
    _require_columns(stats, STAT_INPUT_COLUMNS, "training stats")

    market = values[MARKET_COLUMNS].copy()
    season_stats = drop_pre_nfl(stats.copy())[STAT_COLUMNS].copy()
    for frame in (market, season_stats):
        frame["player_id"] = frame["player_id"].astype(str)
        frame["season"] = frame["season"].astype(int)

    _reject_duplicates(market, "market export")
    _reject_duplicates(season_stats, "training stats")

    labels = market.merge(season_stats, on=["player_id", "season"], how="inner", validate="one_to_one")
    labels = labels[LABEL_EXPORT_COLUMNS]
    return labels.sort_values(
        ["season", "start_value", "player_id"], ascending=[True, False, True], kind="stable"
    ).reset_index(drop=True)


def validate_market_export(
    values: pd.DataFrame,
    *,
    expected_seasons: tuple[int, ...] | None = None,
    model_version: str | None = None,
) -> None:
    """Raise on integrity errors in a raw export; ordinary gaps are left to the coverage report."""
    _require_columns(values, MARKET_COLUMNS, "market export")
    _reject_duplicates(values.assign(player_id=values["player_id"].astype(str)), "market export")
    if values["player_id"].isna().any() or values["season"].isna().any():
        raise ValueError("market export has rows without a player_id or season")

    problems = []
    for column in REQUIRED_VALUE_COLUMNS:
        numbers = pd.to_numeric(values[column], errors="coerce").astype(float)
        bad = ~np.isfinite(numbers) | (numbers < 0)
        if bad.any():
            problems.append(f"{column}: {int(bad.sum())} missing, non-numeric or negative")
    for column in OPTIONAL_VALUE_COLUMNS:
        raw = values[column]
        numbers = pd.to_numeric(raw, errors="coerce")
        bad = (raw.notna() & numbers.isna()) | (numbers < 0)
        if bad.any():
            problems.append(f"{column}: {int(bad.sum())} non-numeric or negative")
    if problems:
        raise ValueError("market export has invalid values: " + "; ".join(problems))

    seasons = {int(s) for s in values["season"]}
    if expected_seasons:
        missing = sorted(set(expected_seasons) - seasons)
        if missing:
            raise ValueError(
                f"market export has no rows for seasons {missing}: a required start or end generation is "
                "missing or empty; rerun sql/backchannel_labels.sql and check market_history_generations"
            )

    present = [c for c in PROVENANCE_COLUMNS if c in values.columns]
    if present and len(present) != len(PROVENANCE_COLUMNS):
        missing = [c for c in PROVENANCE_COLUMNS if c not in values.columns]
        raise ValueError(f"market export has some provenance columns but not {missing}")
    if present:
        required = [c for c in PROVENANCE_COLUMNS if not c.startswith("next_")]
        empty = [c for c in required if values[c].isna().any()]
        if empty:
            raise ValueError(f"market export has empty provenance values in {empty}")
        if (values["next_generation_id"].isna() != values["next_revision"].isna()).any():
            raise ValueError("market export has a next-board generation id without its revision, or the reverse")

    if "model_version" in values.columns:
        versions = sorted(values["model_version"].dropna().astype(str).unique())
        if len(versions) != 1 or values["model_version"].isna().any():
            raise ValueError(f"market export mixes model versions: {versions}")
        if model_version and versions[0] != model_version:
            raise ValueError(f"market export is {versions[0]}, expected {model_version}")
    for prefix, required in (("start", True), ("end", True), ("next", False)):
        column = f"{prefix}_generation_id"
        if column not in values.columns:
            continue
        per_season = values.groupby("season")[column].agg(lambda ids: ids.dropna().nunique())
        missing = values.groupby("season")[column].agg(lambda ids: ids.isna().any())
        if (per_season > 1).any():
            raise ValueError(f"{prefix} generation is not unique for seasons {sorted(per_season[per_season > 1].index)}")
        if required and missing.any():
            raise ValueError(f"{prefix} generation is missing for seasons {sorted(missing[missing].index)}")


def label_coverage(
    values: pd.DataFrame,
    stats: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    expected_seasons: tuple[int, ...] | None = None,
    min_trades: int = 5,
) -> dict:
    """Where each market row went: labelled, pre-NFL, or no NFL stats, per season and position."""
    market = values[["player_id", "season", *MARKET_COLUMNS[2:]]].copy()
    market["player_id"] = market["player_id"].astype(str)
    market["season"] = market["season"].astype(int)
    raw_stats = stats[STAT_INPUT_COLUMNS].copy()
    raw_stats["player_id"] = raw_stats["player_id"].astype(str)
    raw_stats["season"] = raw_stats["season"].astype(int)
    nfl = drop_pre_nfl(raw_stats)
    nfl_keys = set(zip(nfl["player_id"], nfl["season"]))
    all_keys = set(zip(raw_stats["player_id"], raw_stats["season"]))
    stat_seasons = {int(s) for s in raw_stats["season"].unique()}

    keys = list(zip(market["player_id"], market["season"]))
    market["outcome"] = [
        "labelled" if k in nfl_keys else "pre_nfl" if k in all_keys else "no_stats" for k in keys
    ]
    thin = (market["start_trades_window"] < min_trades) | (market["end_trades_window"] < min_trades)

    seasons = sorted(set(market["season"]) | set(expected_seasons or ()))
    by_season = {}
    warnings = []
    for season in seasons:
        rows = market[market["season"] == season]
        counts = rows["outcome"].value_counts()
        labelled = labels[labels["season"] == season]
        position_counts = labelled["position"].value_counts()
        entry = {
            "market_rows": len(rows),
            "labelled_rows": len(labelled),
            "pre_nfl_excluded": int(counts.get("pre_nfl", 0)),
            "no_nfl_stats": int(counts.get("no_stats", 0)),
            "training_archive_has_season": season in stat_seasons,
            "below_min_trades": int(thin[market["season"] == season].sum()),
            "labelled_below_min_trades": int(
                ((labelled["start_trades_window"] < min_trades) | (labelled["end_trades_window"] < min_trades)).sum()
            ),
            "next_board_missing": int(rows["end_value_next_board"].isna().sum()),
            "labelled_by_position": {p: int(position_counts.get(p, 0)) for p in ("QB", "RB", "WR", "TE")},
        }
        for column in ("start_board_rows", "end_board_rows"):
            if column in values.columns and len(rows):
                entry[column] = int(values.loc[values["season"] == season, column].iloc[0])
        by_season[str(season)] = entry
        if not len(rows):
            warnings.append(f"{season}: no market rows")
        elif season not in stat_seasons:
            pass  # e.g. 2019: the training archive has no stats for it
        elif not len(labelled):
            warnings.append(f"{season}: market rows exist but none joined to NFL stats")
        else:
            empty = [p for p, n in entry["labelled_by_position"].items() if n == 0]
            if empty:
                warnings.append(f"{season}: no labelled rows for {empty}")
            if entry["no_nfl_stats"] > entry["labelled_rows"]:
                warnings.append(f"{season}: more market rows lack NFL stats ({entry['no_nfl_stats']}) than joined")
    return {
        "min_trades_reported": min_trades,
        "market_rows": len(market),
        "labelled_rows": len(labels),
        "by_season": by_season,
        "warnings": warnings,
        "status": "warnings" if warnings else "ok",
    }


def _provenance(values: pd.DataFrame, model_version: str | None) -> dict:
    if "model_version" not in values.columns:
        return {
            "model_version": model_version,
            "model_version_source": "argument" if model_version else "unknown",
            "generations": None,
            "exported_at": None,
            "note": "export predates the provenance columns; generation ids are unknown",
        }
    generations = {}
    for season, rows in values.groupby("season", sort=True):
        first = rows.iloc[0]
        generations[str(int(season))] = {
            prefix: None
            if pd.isna(first[f"{prefix}_generation_id"])
            else {"id": int(first[f"{prefix}_generation_id"]), "revision": int(first[f"{prefix}_revision"])}
            for prefix in ("start", "end", "next")
        }
    exported = sorted(values["exported_at"].dropna().astype(str).unique())
    return {
        "model_version": str(values["model_version"].iloc[0]),
        "model_version_source": "export",
        "generations": generations,
        "exported_at": exported[0] if len(exported) == 1 else exported,
    }


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance_path(out_path: Path | str) -> Path:
    path = Path(out_path)
    return path.with_name(path.stem + ".provenance.json")


def build_labels_file(
    values_path: Path | str,
    out_path: Path | str,
    training_data_path: Path | str = DEFAULT_ZIP,
    *,
    expected_seasons: tuple[int, ...] | None = DEFAULT_EXPECTED_SEASONS,
    model_version: str | None = None,
) -> pd.DataFrame:
    """Read the SQL export and training zip, then publish the 14-column CSV and its provenance.

    Both files are built and checked in a staging directory first. Publishing
    moves the old pair aside, moves the new pair in, and puts the old pair
    back if either move fails, so a failure leaves the previous pair intact.
    The provenance also carries the CSV's SHA-256, so a reader can tell a
    mismatched pair from a matching one.
    """
    values = pd.read_csv(values_path, dtype={"player_id": str})
    validate_market_export(values, expected_seasons=expected_seasons, model_version=model_version)
    stats = read_season_stats(training_data_path)
    labels = build_label_frame(values, stats)
    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sidecar = provenance_path(destination)

    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".labels-staging-") as staging:
        staged_csv = Path(staging) / destination.name
        staged_sidecar = Path(staging) / sidecar.name
        labels.to_csv(staged_csv, index=False)
        report = {
            "schema_version": 1,
            "label_file": {"file": destination.name, "sha256": file_sha256(staged_csv), "rows": len(labels)},
            "market_export": {"file": Path(values_path).name, "sha256": file_sha256(values_path), "rows": len(values)},
            "training_data": {"file": Path(training_data_path).name, "sha256": file_sha256(training_data_path)},
            "expected_seasons": sorted(expected_seasons) if expected_seasons else None,
            "trade_threshold": "none applied; thin rows are kept and filtered at training time (--min-trades)",
            **_provenance(values, model_version),
            "coverage": label_coverage(values, stats, labels, expected_seasons=expected_seasons),
        }
        staged_sidecar.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        _check_staged_pair(staged_csv, staged_sidecar, len(labels))
        _publish_pair([(staged_csv, destination), (staged_sidecar, sidecar)])
    labels.attrs["provenance"] = report
    return labels


def _check_staged_pair(csv_path: Path, sidecar_path: Path, rows: int) -> None:
    written = pd.read_csv(csv_path, dtype={"player_id": str})
    if list(written.columns) != LABEL_EXPORT_COLUMNS or len(written) != rows:
        raise ValueError("staged label CSV does not match the 14-column contract")
    report = json.loads(sidecar_path.read_text())
    if report["label_file"]["sha256"] != file_sha256(csv_path):
        raise ValueError("staged provenance does not describe the staged label CSV")


def _publish_pair(moves: list[tuple[Path, Path]]) -> None:
    """Move staged files into place; on failure restore every original file."""
    backups: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        for _, final in moves:
            if final.exists():
                backup = final.with_name(f".{final.name}.previous")
                os.replace(final, backup)
                backups.append((backup, final))
        for staged, final in moves:
            os.replace(staged, final)
            published.append(final)
    except BaseException:
        for final in published:
            final.unlink(missing_ok=True)
        for backup, final in backups:
            os.replace(backup, final)
        raise
    for backup, _ in backups:
        backup.unlink(missing_ok=True)


LABEL_MANIFEST = Path(__file__).resolve().parent / "data" / "label_manifest.json"


def label_identity(path: Path | str, manifest_path: Path | str = LABEL_MANIFEST) -> dict:
    """The model version a label file was verifiably built from, keyed by its SHA-256.

    A ``.provenance.json`` sidecar written by ``build-labels`` from an export
    that carried its own model version counts if its recorded SHA-256 matches
    the file. Committed files that predate sidecars are listed by SHA-256 in
    ``data/label_manifest.json``. File names are never used. Anything else is
    unverified, with the reasons in ``problems``.
    """
    sha = file_sha256(path)
    problems: list[str] = []
    found: dict[str, str] = {}

    sidecar = provenance_path(path)
    if sidecar.exists():
        try:
            report = json.loads(sidecar.read_text())
            recorded = report["label_file"]["sha256"]
            version = report["model_version"]
            source = report["model_version_source"]
        except (ValueError, KeyError, TypeError) as error:
            problems.append(f"{sidecar.name} is unreadable or incomplete ({error!r})")
        else:
            if recorded != sha:
                problems.append(f"{sidecar.name} describes a different file (stale provenance)")
            elif source != "export" or not version:
                problems.append(f"{sidecar.name} model version came from {source!r}, not the export")
            else:
                found["sidecar"] = version

    manifest = Path(manifest_path)
    if manifest.exists():
        entry = json.loads(manifest.read_text()).get("files", {}).get(sha)
        if entry is not None:
            if entry.get("model_version"):
                found["manifest"] = entry["model_version"]
            else:
                problems.append(f"{manifest.name} lists this file without a model version")

    versions = set(found.values())
    if len(versions) > 1:
        problems.append(f"provenance sources disagree: {found}")
    if not found and not problems:
        problems.append("no provenance sidecar and not in the label manifest")
    verified = len(versions) == 1 and not problems
    return {
        "sha256": sha,
        "model_version": versions.pop() if verified else None,
        "verified": verified,
        "source": ",".join(sorted(found)) if verified else None,
        "problems": problems,
    }
