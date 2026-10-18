"""Build the versioned Backchannel label CSV from a market-history export."""

from pathlib import Path

import pandas as pd

from predictor_v5.data import DEFAULT_ZIP, drop_pre_nfl, load_season_stats

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


def build_labels_file(
    values_path: Path | str,
    out_path: Path | str,
    training_data_path: Path | str = DEFAULT_ZIP,
) -> pd.DataFrame:
    """Read the SQL export and training zip, then write the 14-column CSV."""
    values = pd.read_csv(values_path, dtype={"player_id": str})
    labels = build_label_frame(values, load_season_stats(training_data_path))
    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(destination, index=False)
    return labels
