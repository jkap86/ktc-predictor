"""Validate and compare the content-addressed 2026 start snapshot."""

import hashlib
import subprocess
from pathlib import Path

import pandas as pd

SHA256 = "6366b85300c4f9532f0866c602c6d262baaccf6c3a84b9252a6d2e8d46c541bb"
GIT_BLOB = "ead24b508102ca2a1071d07f3f665e9ea920aa96"
INTRODUCTION = "c415177"
RELATIVE_PATH = "backend/predictor_v5/data/start_values_2026.csv"


def load_starts(path: Path, root: Path) -> pd.DataFrame:
    path, root = Path(path).resolve(), Path(root).resolve()
    if (
        path != root / RELATIVE_PATH
        or hashlib.sha256(path.read_bytes()).hexdigest() != SHA256
    ):
        raise ValueError("frozen start content or path differs")
    try:
        blob = subprocess.check_output(
            ["git", "rev-parse", f"{INTRODUCTION}:{RELATIVE_PATH}"], cwd=root, text=True
        ).strip()
        introduced = subprocess.check_output(
            ["git", "log", "--diff-filter=A", "--format=%H", "--", RELATIVE_PATH],
            cwd=root,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, OSError) as exc:
        raise ValueError("cannot verify frozen start Git identity") from exc
    if blob != GIT_BLOB or not introduced.startswith(INTRODUCTION):
        raise ValueError("frozen start Git identity differs")
    starts = pd.read_csv(path, dtype={"player_id": str})
    if starts["player_id"].isna().any() or starts["player_id"].duplicated().any():
        raise ValueError("duplicate or missing start player ID")
    cutoff = pd.Timestamp("2026-09-09")
    for date_column, value_column, earliest in (
        ("bc_date", "bc_value", None),
        ("ktc_date", "ktc_value", pd.Timestamp("2026-08-26")),
    ):
        dates = pd.to_datetime(starts[date_column], errors="coerce")
        if (
            (dates.isna() != starts[value_column].isna()).any()
            or (dates > cutoff).any()
            or (earliest is not None and (dates < earliest).any())
        ):
            raise ValueError("invalid start observation date")
    if not starts["bc_model_version"].eq("baseline-hl28-v1").all():
        raise ValueError("invalid start model version")
    starts.insert(0, "season", 2026)
    starts = starts.sort_values(["season", "player_id"]).reset_index(drop=True)
    starts.attrs.update(
        sha256=SHA256,
        git_blob=GIT_BLOB,
        introduction=INTRODUCTION,
        provenance_limit="raw start provenance not independently verified",
    )
    return starts


def check_target_starts(
    starts: pd.DataFrame, target: pd.DataFrame, source: str
) -> dict:
    if source not in {"ktc", "mfl"}:
        raise ValueError("unknown start source")
    keys = ["season", "player_id"]
    for frame in (starts, target):
        if frame[keys].isna().any().any() or frame.duplicated(keys).any():
            raise ValueError("duplicate or missing start keys")
    left, right = starts.copy(), target.copy()
    left["player_id"] = left["player_id"].astype(str)
    right["player_id"] = right["player_id"].astype(str)
    merged = right.merge(
        left, on=keys, how="left", validate="one_to_one", indicator=True
    )
    value = "ktc_value" if source == "ktc" else "bc_value"
    present = merged["_merge"].eq("both") & merged[value].notna()
    if (merged.loc[present, "start_value"] != merged.loc[present, value]).any():
        raise ValueError("target start value conflicts with frozen starts")
    if (
        source == "mfl"
        and (
            merged.loc[present, "start_trades_window"]
            != merged.loc[present, "bc_trades"]
        ).any()
    ):
        raise ValueError("target start trades conflict with frozen starts")
    missing_keys = sorted(
        map(tuple, merged.loc[~present, keys].itertuples(index=False, name=None))
    )
    return {
        "source": source,
        "missing_keys": missing_keys,
        "missing_count": len(missing_keys),
        "matched_count": int(present.sum()),
        "missing": len(missing_keys),
        "matched": int(present.sum()),
    }
