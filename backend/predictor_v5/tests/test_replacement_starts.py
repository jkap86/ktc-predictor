import subprocess
from pathlib import Path

import pandas as pd
import pytest

from predictor_v5.replacement_starts import check_target_starts, load_starts

ROOT = Path(__file__).resolve().parents[3]
STARTS = ROOT / "backend/predictor_v5/data/start_values_2026.csv"


def test_frozen_identity_and_dates():
    starts = load_starts(STARTS, ROOT)
    assert (
        starts.attrs["sha256"]
        == "6366b85300c4f9532f0866c602c6d262baaccf6c3a84b9252a6d2e8d46c541bb"
    )
    assert starts.attrs["git_blob"] == "ead24b508102ca2a1071d07f3f665e9ea920aa96"
    assert starts.attrs["introduction"] == "c415177"
    assert (
        starts.attrs["provenance_limit"]
        == "raw start provenance not independently verified"
    )
    assert starts["bc_model_version"].dropna().eq("baseline-hl28-v1").all()
    assert (
        pd.to_datetime(starts["bc_date"].dropna()).le(pd.Timestamp("2026-09-09")).all()
    )
    ktc_dates = pd.to_datetime(starts["ktc_date"].dropna())
    assert ktc_dates.between("2026-08-26", "2026-09-09").all()


def test_parses_verified_bytes_if_file_changes_before_parse(monkeypatch, tmp_path):
    snapshot = tmp_path / "backend/predictor_v5/data/start_values_2026.csv"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(STARTS.read_bytes())
    real_read_bytes = Path.read_bytes

    def replace_after_read(path):
        original = real_read_bytes(path)
        path.write_text(
            "player_id,bc_value,bc_trades,bc_date,bc_model_version,ktc_value,ktc_date\nattacker,999,5,2026-09-09,baseline-hl28-v1,100,2026-09-09\n"
        )
        return original

    monkeypatch.setattr(Path, "read_bytes", replace_after_read)
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda command, **kwargs: (
            "ead24b508102ca2a1071d07f3f665e9ea920aa96\n"
            if command[1] == "rev-parse"
            else "c415177f9e7e0653879b2eed2a7a29205725e9e7\n"
        ),
    )
    starts = load_starts(snapshot, tmp_path)
    assert len(starts) == 528
    assert "attacker" not in starts.player_id.values


def test_rejects_altered_snapshot_bytes(monkeypatch):
    monkeypatch.setattr(Path, "read_bytes", lambda self: b"altered")
    with pytest.raises(ValueError, match="content or path"):
        load_starts(STARTS, ROOT)


def test_rejects_wrong_snapshot_path():
    with pytest.raises(ValueError, match="content or path"):
        load_starts(ROOT / "other.csv", ROOT)


@pytest.mark.parametrize("failure", ["blob", "introduction", "git failure"])
def test_rejects_invalid_git_identity(monkeypatch, failure):
    real_check_output = subprocess.check_output

    def altered_git(command, **kwargs):
        if failure == "git failure":
            raise subprocess.CalledProcessError(1, command)
        if failure == "blob" and command[1] == "rev-parse":
            return "0000000000000000000000000000000000000000\n"
        if failure == "introduction" and command[1] == "log":
            return "0000000000000000000000000000000000000000\n"
        return real_check_output(command, **kwargs)

    monkeypatch.setattr(subprocess, "check_output", altered_git)
    with pytest.raises(ValueError, match="Git identity"):
        load_starts(STARTS, ROOT)


def test_start_missing_is_coverage_but_conflict_refuses():
    starts = pd.DataFrame(
        [
            {
                "season": 2026,
                "player_id": "1",
                "ktc_value": 100,
                "bc_value": 80,
                "bc_trades": 5,
            }
        ]
    )
    target = pd.DataFrame(
        [
            {"season": 2026, "player_id": "1", "start_value": 100},
            {"season": 2026, "player_id": "2", "start_value": 10},
        ]
    )
    assert check_target_starts(starts, target, "ktc")["missing"] == 1
    with pytest.raises(ValueError):
        check_target_starts(starts, target.assign(start_value=[99, 10]), "ktc")
    mfl = target.assign(start_value=[80, 10], start_trades_window=[5, 1])
    assert check_target_starts(starts, mfl, "mfl")["missing"] == 1
    with pytest.raises(ValueError):
        check_target_starts(starts, mfl.assign(start_trades_window=[6, 1]), "mfl")
    with pytest.raises(ValueError):
        check_target_starts(starts, mfl.assign(start_value=[79, 10]), "mfl")


def test_missing_coverage_reports_sorted_identities():
    starts = pd.DataFrame([{"season": 2026, "player_id": "1", "ktc_value": 100}])
    target = pd.DataFrame(
        [
            {"season": 2026, "player_id": "3", "start_value": 3},
            {"season": 2026, "player_id": "1", "start_value": 100},
            {"season": 2026, "player_id": "2", "start_value": 2},
        ]
    )
    coverage = check_target_starts(starts, target, "ktc")
    assert coverage["source"] == "ktc"
    assert coverage["missing_keys"] == [(2026, "2"), (2026, "3")]
    assert coverage["missing_count"] == 2
    assert coverage["matched_count"] == 1


@pytest.mark.parametrize("duplicate_side", ["starts", "target"])
def test_duplicate_player_season_is_rejected(duplicate_side):
    starts = pd.DataFrame([{"season": 2026, "player_id": "1", "ktc_value": 100}])
    target = pd.DataFrame([{"season": 2026, "player_id": "1", "start_value": 100}])
    if duplicate_side == "starts":
        starts = pd.concat([starts, starts], ignore_index=True)
    else:
        target = pd.concat([target, target], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate or missing start keys"):
        check_target_starts(starts, target, "ktc")


def test_duplicate_frozen_player_is_rejected(monkeypatch):
    row = {
        "player_id": "1",
        "bc_value": 80,
        "bc_trades": 5,
        "bc_date": "2026-09-09",
        "bc_model_version": "baseline-hl28-v1",
        "ktc_value": 100,
        "ktc_date": "2026-08-26",
    }
    monkeypatch.setattr(
        pd, "read_csv", lambda *args, **kwargs: pd.DataFrame([row, row])
    )
    with pytest.raises(ValueError, match="duplicate or missing start player ID"):
        load_starts(STARTS, ROOT)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("bc_date", "2026-09-10"),
        ("ktc_date", "2026-09-10"),
        ("ktc_date", "2026-08-25"),
        ("bc_model_version", "baseline-hl28-v2"),
    ],
    ids=["bc-future", "ktc-future", "ktc-too-old", "wrong-model"],
)
def test_invalid_frozen_observation_is_rejected(monkeypatch, field, bad_value):
    # Preserve real identity checks; substitute only the already-parsed frame to
    # exercise policy rejection on independently specified invalid observations.
    row = {
        "player_id": "1",
        "bc_value": 80,
        "bc_trades": 5,
        "bc_date": "2026-09-09",
        "bc_model_version": "baseline-hl28-v1",
        "ktc_value": 100,
        "ktc_date": "2026-08-26",
    }
    row[field] = bad_value
    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: pd.DataFrame([row]))
    with pytest.raises(ValueError, match="invalid start"):
        load_starts(STARTS, ROOT)
