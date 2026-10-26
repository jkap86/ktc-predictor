"""Contract tests for deterministic movement-evaluation reports."""

from __future__ import annotations

import errno
import hashlib
import os
import stat
import threading
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest

from predictor_v5.movement_evaluation.contract import (
    BASE_COMMIT,
    FOLDS,
    ContractRefusal,
)
from predictor_v5.movement_evaluation.data import EvaluationData, HistoricalBundle
from predictor_v5.movement_evaluation.models import FoldResult, run_ktc_diagnostic
from predictor_v5.movement_evaluation.report import (
    build_report,
    build_run_identity,
    canonical_bytes,
    publish_report,
    report_path,
    runtime_identity,
    source_identities,
)


class _PrivateEstimator:
    secret = "PRIVATE_ESTIMATOR_MUST_NOT_SERIALIZE"


def _predictions(season: int, player_id: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season": season,
                "player_id": player_id,
                "position": "QB",
                "age": 25.0,
                "games": 16,
                "points": 210.0,
                "start_value": 1000.0,
                "end_value": 1100.0,
                "start_trades_window": 8,
                "end_trades_window": 9,
                "value_tier": "mid",
                "S": 1000.0,
                "E": 1100.0,
                "r": 0.09531017980432493,
                "d": 100.0,
                "production_E_hat": 1080.0,
                "production_r_hat": 0.0769610411361284,
                "production_d_hat": 80.0,
                "no_production_E_hat": 1020.0,
                "no_production_r_hat": 0.01980262729617973,
                "no_production_d_hat": 20.0,
                "no_change_E_hat": 1000.0,
                "no_change_r_hat": 0.0,
                "no_change_d_hat": 0.0,
                "player_name": "must not serialize",
                "future_outcome": 9999,
            }
        ]
    )


def _metric(value: float | None, reason: str | None = None) -> dict[str, object]:
    return {"n": 1 if value is not None else 0, "value": value, "reason": reason}


def _folds_and_scores() -> tuple[tuple[FoldResult, ...], list[dict[str, object]]]:
    folds: list[FoldResult] = []
    scores: list[dict[str, object]] = []
    for spec in FOLDS:
        player_id = f"p{spec.test_season}"
        train_keys = tuple((season, f"train-{season}") for season in spec.train_seasons)
        fold = FoldResult(
            spec=spec,
            train_keys=train_keys,
            test_keys=((spec.test_season, player_id),),
            predictions=_predictions(spec.test_season, player_id),
            tier_edges={"QB": (500.0, 1500.0), "RB": None, "TE": None, "WR": None},
            training_majority="flat",
            model_parameters={
                "production": {"do_early_stopping": False},
                "no_production": {"do_early_stopping": False},
            },
            clipping={"production_at_cap_count": 0, "no_production_at_cap_count": 0},
            _models=(_PrivateEstimator(), _PrivateEstimator()),
        )
        folds.append(fold)
        scores.append(
            {
                "test_season": spec.test_season,
                "train_seasons": list(spec.train_seasons),
                "short_history": spec.short_history,
                "n": 1,
                "models": {"production": {"mae_log": _metric(0.02)}},
                "contrasts": {
                    "production_vs_no_production": {
                        "positive_mae_log_improvement": _metric(0.04)
                    },
                    "production_vs_no_change": {
                        "positive_mae_log_improvement": _metric(0.06)
                    },
                },
                "direction_baselines": {"training_majority": "flat"},
                "slices": {"overall": {"n": 1}},
                "uncertainty": {"requested_resamples": 2000},
            }
        )
    return tuple(folds), scores


def _coverage() -> dict[str, object]:
    return {
        "upstream_provenance": {
            "model_version": "baseline-hl28-v1",
            "model_version_source": "export",
            "exported_at": "2026-10-24T11:39:31Z",
        },
        "ledger": {"start_eligible": 6, "primary_scored": 5},
        "reasons": {"zero_end": 1},
        "key_hashes": {"common": {"count": 6, "sha256": "c" * 64}},
        "full_preseason_population_count": {
            "value": None,
            "reason": "not_recoverable_from_historical_export",
        },
        "full_preseason_coverage": {
            "value": None,
            "reason": "not_recoverable_from_historical_export",
        },
    }


def _bundle_and_data() -> tuple[HistoricalBundle, EvaluationData]:
    identities = {
        "labels_sha256": "1" * 64,
        "provenance_sha256": "2" * 64,
        "archive_sha256": "3" * 64,
        "archive_member_sha256": "4" * 64,
        "labels_path": "backend/predictor_v5/data/backchannel_labels_2019_2025_mfl.csv",
        "provenance_path": "backend/predictor_v5/data/backchannel_labels_2019_2025_mfl.provenance.json",
        "archive_path": "backend/data/training-data.zip",
        "archive_member": "training-data.json",
    }
    empty = pd.DataFrame()
    bundle = HistoricalBundle(empty, empty, _coverage()["upstream_provenance"], identities)
    data = EvaluationData(empty, empty, empty, empty, _coverage())
    return bundle, data


def _pooled(incremental: float = 0.04, no_change: float = 0.06) -> dict[str, object]:
    return {
        "n": 5,
        "models": {
            "production": {"mae_log": _metric(0.02), "skill_log": _metric(0.5)},
            "no_production": {"mae_log": _metric(0.06)},
            "no_change": {"mae_log": _metric(0.08)},
        },
        "contrasts": {
            "production_vs_no_production": {
                "positive_mae_log_improvement": _metric(incremental)
            },
            "production_vs_no_change": {
                "positive_mae_log_improvement": _metric(no_change)
            },
        },
        "macro_by_season": {},
        "uncertainty": {"player_cluster": {"requested_resamples": 2000}},
        "leave_one_season_out": [],
        "slices": {"overall": {"n": 5}},
        "target_diagnostics": {"season": {}},
        "multiplicity": "family-wide multiplicity is uncontrolled",
    }


def _runtime() -> dict[str, object]:
    return {
        "python": "3.12.0",
        "numpy": "2.0.0",
        "pandas": "2.2.0",
        "scikit_learn": "1.5.0",
        "scipy": "1.14.0",
        "thread_settings": {
            "MKL_NUM_THREADS": "unset",
            "NUMEXPR_NUM_THREADS": "unset",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "unset",
        },
    }


def _sources() -> dict[str, str]:
    return {
        "model.py": "a" * 64,
        "movement_evaluation/__init__.py": "b" * 64,
        "movement_evaluation/__main__.py": "c" * 64,
        "movement_evaluation/contract.py": "d" * 64,
        "movement_evaluation/data.py": "e" * 64,
        "movement_evaluation/metrics.py": "f" * 64,
        "movement_evaluation/models.py": "0" * 64,
        "movement_evaluation/report.py": "9" * 64,
    }


def _report(*, mode: str = "off", optional_ktc: dict[str, object] | None = None):
    bundle, data = _bundle_and_data()
    folds, fold_scores = _folds_and_scores()
    if optional_ktc is None:
        optional_ktc = run_ktc_diagnostic(data, mode="off")
    return build_report(
        bundle=bundle,
        data=data,
        folds=folds,
        fold_scores=fold_scores,
        pooled=_pooled(),
        optional_ktc=optional_ktc,
        sources=_sources(),
        mode=mode,
        runtime=_runtime(),
    )


def test_run_identity_is_complete_stable_and_recomputable() -> None:
    bundle, _ = _bundle_and_data()
    identity, digest = build_run_identity(
        inputs=bundle.identities,
        sources=_sources(),
        mode="off",
        runtime=_runtime(),
    )

    assert set(identity) == {
        "protocol",
        "schema_version",
        "base_commit",
        "inputs",
        "sources",
        "folds",
        "models",
        "metrics",
        "bootstrap",
        "ktc_mode",
        "runtime",
    }
    assert identity["protocol"] == "market-movement-development"
    assert identity["schema_version"] == 1
    assert identity["base_commit"] == BASE_COMMIT
    assert identity["inputs"]["archive_member_sha256"] == "4" * 64
    assert identity["sources"] == _sources()
    assert len(identity["folds"]) == 5
    assert set(identity["models"]) == {"production", "no_production"}
    assert identity["bootstrap"] == {
        "seed": 20261025,
        "resamples": 2000,
        "percentiles": [2.5, 97.5],
    }
    assert identity["ktc_mode"] == "off"
    assert identity["runtime"] == _runtime()
    assert digest == hashlib.sha256(canonical_bytes(identity)).hexdigest()
    raw = canonical_bytes(identity)
    for forbidden in (b"current_head", b"branch", b"timestamp", b"hostname", b"tmp", b"scores"):
        assert forbidden not in raw


def test_runtime_and_all_eight_source_hashes_are_recorded(tmp_path: Path) -> None:
    package = tmp_path / "predictor_v5"
    movement = package / "movement_evaluation"
    movement.mkdir(parents=True)
    paths = [
        package / "model.py",
        *(movement / name for name in (
            "__init__.py",
            "__main__.py",
            "contract.py",
            "data.py",
            "metrics.py",
            "models.py",
            "report.py",
        )),
    ]
    for index, path in enumerate(paths):
        path.write_bytes(f"source-{index}\n".encode())

    identities = source_identities(package)

    assert set(identities) == set(_sources())
    assert identities["model.py"] == hashlib.sha256(b"source-0\n").hexdigest()
    assert identities["movement_evaluation/__main__.py"] == hashlib.sha256(
        b"source-2\n"
    ).hexdigest()
    runtime = runtime_identity()
    assert set(runtime) == {
        "python",
        "numpy",
        "pandas",
        "scikit_learn",
        "scipy",
        "thread_settings",
    }
    assert set(runtime["thread_settings"]) == set(_runtime()["thread_settings"])


def test_report_schema_ordering_private_exclusion_and_complete_off_payload() -> None:
    report = _report()

    assert set(report) == {
        "schema_version",
        "protocol",
        "run_identity",
        "run_sha256",
        "evidence",
        "target",
        "contract",
        "coverage",
        "folds",
        "pooled",
        "predictions",
        "optional_ktc",
        "interpretation",
    }
    assert report["evidence"] == {
        "class": "historical_development",
        "historical_exposure": "architecture_and_data_choices_informed_by_prior_historical_experiments",
        "2025_status": "reused_development",
        "conditional_points": True,
        "historical_asof_validity": "unverified",
        "measurement_limits": "generated_backchannel_market_measurement_not_objective_exchange_value",
        "promotion_authorized": False,
        "independent_confirmation_complete": False,
    }
    assert report["target"]["formula"] == "log(end_value / start_value)"
    assert report["target"]["columns"] == {"start": "start_value", "end": "end_value"}
    assert len(report["target"]["dates"]) == len(report["target"]["generations"]) == 6
    assert set(report["contract"]) == {
        "models",
        "folds",
        "eligibility",
        "metrics",
        "bins",
        "bootstrap",
        "ktc_mode",
    }
    assert len(report["folds"]) == 5
    assert report["coverage"] == _coverage()
    assert report["optional_ktc"] == {
        "status": "off",
        "provenance_limits": {"provenance_limited": True, "reason": "diagnostic_off"},
        "cohorts": [],
        "fits": [],
        "scores": {"value": None, "reason": "diagnostic_off"},
        "predictions": [],
    }
    keys = [(row["season"], row["player_id"]) for row in report["predictions"]]
    assert keys == sorted(keys) == list(dict.fromkeys(keys))
    assert set(report["predictions"][0]) == {
        "season",
        "player_id",
        "position",
        "S",
        "E",
        "r",
        "points",
        "age",
        "start_trades",
        "end_trades",
        "value_tier",
        "production_E_hat",
        "no_production_E_hat",
        "no_change_E_hat",
    }
    raw = canonical_bytes(report)
    assert b"PRIVATE_ESTIMATOR_MUST_NOT_SERIALIZE" not in raw
    assert b"player_name" not in raw
    assert b"future_outcome" not in raw
    assert b"_models" not in raw


def test_report_is_row_order_independent_and_ktc_never_changes_evidence_flags() -> None:
    first = _report()
    bundle, data = _bundle_and_data()
    folds, scores = _folds_and_scores()
    reversed_folds = tuple(reversed(folds))
    reversed_scores = list(reversed(scores))
    archive_payload = {
        "status": "diagnostic_only",
        "provenance_limits": {"provenance_limited": True},
        "cohorts": [{"test_season": 2025}],
        "fits": [{"test_season": 2025, "name": "ktc_production"}],
        "scores": [{"test_season": 2025}],
        "predictions": [{"season": 2025, "player_id": "z"}],
    }
    changed_optional = build_report(
        bundle=bundle,
        data=data,
        folds=reversed_folds,
        fold_scores=reversed_scores,
        pooled=_pooled(),
        optional_ktc=archive_payload,
        sources=_sources(),
        mode="archive",
        runtime=_runtime(),
    )

    assert [(fold["test_season"]) for fold in changed_optional["folds"]] == list(
        range(2021, 2026)
    )
    assert changed_optional["interpretation"]["evidence_flags"] == first[
        "interpretation"
    ]["evidence_flags"]
    repeated = _report()
    assert canonical_bytes(first) == canonical_bytes(repeated)


def test_interpretation_counts_defined_position_interval_estimates() -> None:
    bundle, data = _bundle_and_data()
    folds, scores = _folds_and_scores()
    pooled = _pooled()
    pooled["uncertainty"]["by_position"] = {
        "QB": {
            "contrasts": {
                "production_vs_no_production": {
                    "positive_mae_log_improvement": {"estimate": 0.03, "reason": None}
                }
            }
        },
        "RB": {
            "contrasts": {
                "production_vs_no_production": {
                    "positive_mae_log_improvement": {"estimate": -0.01, "reason": None}
                }
            }
        },
        "TE": {
            "contrasts": {
                "production_vs_no_production": {
                    "positive_mae_log_improvement": {
                        "estimate": None,
                        "reason": "insufficient_valid_resamples",
                    }
                }
            }
        },
        "WR": {},
    }

    report = build_report(
        bundle=bundle,
        data=data,
        folds=folds,
        fold_scores=scores,
        pooled=pooled,
        optional_ktc=run_ktc_diagnostic(data, mode="off"),
        sources=_sources(),
        mode="off",
        runtime=_runtime(),
    )

    assert report["interpretation"]["evidence_flags"]["improved_positions"] == {
        "count": 1,
        "defined": 2,
        "total": 4,
    }


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), {1, 2}])
def test_canonical_json_refuses_nonfinite_and_non_json_values(bad: object) -> None:
    with pytest.raises(ContractRefusal):
        canonical_bytes({"bad": bad})


def test_canonical_json_is_compact_sorted_full_precision_and_requires_string_keys() -> None:
    value = {"z": 1.2345678901234567, "a": [2, 1]}
    assert canonical_bytes(value) == b'{"a":[2,1],"z":1.2345678901234567}\n'
    assert not canonical_bytes(value).endswith(b"\n\n")
    with pytest.raises(ContractRefusal):
        canonical_bytes({1: "not allowed"})


def test_report_refuses_nullable_metric_without_reason_and_duplicate_predictions() -> None:
    bundle, data = _bundle_and_data()
    folds, scores = _folds_and_scores()
    bad_pooled = _pooled()
    bad_pooled["models"]["production"]["mae_log"] = {"n": 0, "value": None}
    with pytest.raises(ContractRefusal, match="reason"):
        build_report(
            bundle=bundle,
            data=data,
            folds=folds,
            fold_scores=scores,
            pooled=bad_pooled,
            optional_ktc=run_ktc_diagnostic(data, mode="off"),
            sources=_sources(),
            mode="off",
            runtime=_runtime(),
        )

    duplicate = list(folds)
    duplicate[1] = FoldResult(
        spec=duplicate[1].spec,
        train_keys=duplicate[1].train_keys,
        test_keys=((2022, "p2021"),),
        predictions=_predictions(2022, "p2021"),
        tier_edges=duplicate[1].tier_edges,
        training_majority="flat",
        model_parameters=duplicate[1].model_parameters,
        clipping=duplicate[1].clipping,
        _models=(_PrivateEstimator(), _PrivateEstimator()),
    )
    duplicate[1].predictions.loc[:, "season"] = 2021
    with pytest.raises(ContractRefusal):
        build_report(
            bundle=bundle,
            data=data,
            folds=duplicate,
            fold_scores=scores,
            pooled=_pooled(),
            optional_ktc=run_ktc_diagnostic(data, mode="off"),
            sources=_sources(),
            mode="off",
            runtime=_runtime(),
        )


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "backend" / "predictor_v5" / "artifacts").mkdir(parents=True)
    for name in ("confirmations", "replacements", "data", "runtime"):
        path = root / "backend" / "predictor_v5" / name
        path.mkdir()
        (path / "sentinel").write_text(name)
    (root / "backend" / "predictor_v5" / "model.py").write_text("protected")
    return root


def _sentinels(root: Path) -> dict[Path, bytes]:
    protected = [
        root / "backend" / "predictor_v5" / "model.py",
        *(root / "backend" / "predictor_v5" / name / "sentinel" for name in (
            "confirmations",
            "replacements",
            "data",
            "runtime",
        )),
    ]
    return {path: path.read_bytes() for path in protected}


def test_report_path_and_atomic_idempotent_no_replace_publication(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    report = _report()
    expected = (
        root
        / "backend"
        / "predictor_v5"
        / "artifacts"
        / "market_movement"
        / f"development-2020-2025-{report['run_sha256']}.json"
    )
    before = _sentinels(root)

    assert report_path(root, report["run_sha256"]) == expected
    assert publish_report(report, root) == expected
    first_stat = expected.stat()
    assert expected.read_bytes() == canonical_bytes(report)
    assert publish_report(report, root) == expected
    assert expected.stat().st_ino == first_stat.st_ino
    assert _sentinels(root) == before

    conflicting = deepcopy(report)
    conflicting["evidence"]["conditional_points"] = False
    with pytest.raises(ContractRefusal, match="different bytes"):
        publish_report(conflicting, root)
    assert expected.read_bytes() == canonical_bytes(report)


def test_publication_refuses_unsafe_paths_and_preserves_temp_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _report()
    root = _repo(tmp_path / "symlink")
    output = root / "backend" / "predictor_v5" / "artifacts" / "market_movement"
    outside = tmp_path / "outside"
    outside.mkdir()
    output.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ContractRefusal):
        publish_report(report, root)
    assert list(outside.iterdir()) == []

    root = _repo(tmp_path / "file")
    output = root / "backend" / "predictor_v5" / "artifacts" / "market_movement"
    output.write_text("not a directory")
    with pytest.raises(ContractRefusal):
        publish_report(report, root)

    root = _repo(tmp_path / "collision")
    output = root / "backend" / "predictor_v5" / "artifacts" / "market_movement"
    output.mkdir()
    collision = output / f".{report['run_sha256']}.tmp.fixed"
    collision.write_text("someone else's file")
    monkeypatch.setattr(
        "predictor_v5.movement_evaluation.report.secrets.token_hex", lambda _n: "fixed"
    )
    with pytest.raises(ContractRefusal, match="temporary"):
        publish_report(report, root)
    assert collision.read_text() == "someone else's file"
    assert not report_path(root, report["run_sha256"]).exists()


@pytest.mark.parametrize("component", ["backend", "predictor_v5", "artifacts"])
def test_publication_refuses_static_ancestor_symlinks(
    tmp_path: Path, component: str
) -> None:
    report = _report()
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "outside"
    if component == "backend":
        target = outside / "backend"
        (target / "predictor_v5" / "artifacts").mkdir(parents=True)
        (root / "backend").symlink_to(target, target_is_directory=True)
    elif component == "predictor_v5":
        (root / "backend").mkdir()
        target = outside / "predictor_v5"
        (target / "artifacts").mkdir(parents=True)
        (root / "backend" / "predictor_v5").symlink_to(
            target, target_is_directory=True
        )
    else:
        (root / "backend" / "predictor_v5").mkdir(parents=True)
        target = outside / "artifacts"
        target.mkdir(parents=True)
        (root / "backend" / "predictor_v5" / "artifacts").symlink_to(
            target, target_is_directory=True
        )

    with pytest.raises(ContractRefusal):
        publish_report(report, root)

    assert not list(outside.rglob("development-*.json"))


@pytest.mark.parametrize("stage", ["root", "child", "existing_report"])
def test_operational_open_errors_are_not_contract_refusals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    root = _repo(tmp_path)
    report = _report()
    destination = report_path(root, report["run_sha256"])
    if stage == "existing_report":
        publish_report(report, root)
    module = __import__("predictor_v5.movement_evaluation.report", fromlist=["report"])
    real_open = module.os.open

    def fail_open(path, flags, mode=0o777, *, dir_fd=None):
        is_target = (
            (stage == "root" and Path(path) == root)
            or (stage == "child" and path == "backend")
            or (stage == "existing_report" and path == destination.name)
        )
        if is_target:
            raise OSError(errno.EIO, "simulated device I/O error")
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(module.os, "open", fail_open)
    with pytest.raises(OSError) as raised:
        publish_report(report, root)
    assert raised.value.errno == errno.EIO
    assert not isinstance(raised.value, ContractRefusal)
    if stage != "existing_report":
        assert not destination.exists()


@pytest.mark.parametrize("failure", ["write", "file_fsync", "directory_fsync"])
def test_publication_failure_removes_only_writer_owned_inodes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    report = _report()
    root = _repo(tmp_path)
    before = _sentinels(root)
    output = root / "backend" / "predictor_v5" / "artifacts" / "market_movement"
    module = __import__("predictor_v5.movement_evaluation.report", fromlist=["report"])
    real_write = module.os.write
    real_fsync = module.os.fsync
    failed = False

    def write(descriptor: int, raw: bytes) -> int:
        if failure == "write":
            real_write(descriptor, raw[: max(1, len(raw) // 3)])
            raise OSError("partial write")
        return real_write(descriptor, raw)

    def fsync(descriptor: int) -> None:
        nonlocal failed
        info = os.fstat(descriptor)
        is_output = output.exists() and (info.st_dev, info.st_ino) == (
            output.stat().st_dev,
            output.stat().st_ino,
        )
        if not failed and failure == "file_fsync" and stat.S_ISREG(info.st_mode):
            failed = True
            raise OSError("file fsync")
        if not failed and failure == "directory_fsync" and is_output:
            failed = True
            raise OSError("directory fsync")
        real_fsync(descriptor)

    monkeypatch.setattr(module.os, "write", write)
    monkeypatch.setattr(module.os, "fsync", fsync)
    with pytest.raises(OSError):
        publish_report(report, root)

    assert not report_path(root, report["run_sha256"]).exists()
    assert not list(output.glob(".*.tmp.*"))
    assert _sentinels(root) == before


def test_concurrent_identical_publishers_converge_without_replacement(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    report = _report()
    barrier = threading.Barrier(3)
    results: list[Path] = []
    errors: list[BaseException] = []

    def publish() -> None:
        barrier.wait()
        try:
            results.append(publish_report(report, root))
        except (OSError, ContractRefusal) as exc:  # pragma: no cover - assertion detail
            errors.append(exc)

    threads = [threading.Thread(target=publish) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(results) == 2 and results[0] == results[1]
    assert results[0].read_bytes() == canonical_bytes(report)
    assert len(list(results[0].parent.glob("development-*.json"))) == 1
    assert not list(results[0].parent.glob(".*.tmp.*"))


def test_reentrant_peer_cannot_accept_an_uncommitted_final_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _repo(tmp_path)
    report = _report()
    output = root / "backend" / "predictor_v5" / "artifacts" / "market_movement"
    output.mkdir()
    module = __import__("predictor_v5.movement_evaluation.report", fromlist=["report"])
    real_fsync = module.os.fsync
    nested_results: list[Path] = []
    nested_errors: list[BaseException] = []
    triggered = False
    output_identity = (output.stat().st_dev, output.stat().st_ino)

    def failing_after_peer_probe(descriptor: int) -> None:
        nonlocal triggered
        info = os.fstat(descriptor)
        if (info.st_dev, info.st_ino) == output_identity and not triggered:
            triggered = True
            try:
                nested_results.append(publish_report(report, root))
            except (OSError, ContractRefusal) as exc:
                nested_errors.append(exc)
            raise OSError("original writer directory fsync failed")
        real_fsync(descriptor)

    monkeypatch.setattr(module.os, "fsync", failing_after_peer_probe)
    with pytest.raises(OSError, match="original writer"):
        publish_report(report, root)

    assert nested_results == []
    assert len(nested_errors) == 1
    assert isinstance(nested_errors[0], ContractRefusal)
    assert not report_path(root, report["run_sha256"]).exists()
    assert not list(output.glob(".*.tmp.*"))


def test_waiting_peer_publishes_only_after_failed_owner_rolls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _repo(tmp_path)
    report = _report()
    output = root / "backend" / "predictor_v5" / "artifacts" / "market_movement"
    output.mkdir()
    module = __import__("predictor_v5.movement_evaluation.report", fromlist=["report"])
    real_fsync = module.os.fsync
    peer_started = threading.Event()
    peer_results: list[Path] = []
    peer_errors: list[BaseException] = []
    peer: threading.Thread | None = None
    triggered = False
    output_identity = (output.stat().st_dev, output.stat().st_ino)

    def peer_publish() -> None:
        peer_started.set()
        try:
            peer_results.append(publish_report(report, root))
        except (OSError, ContractRefusal) as exc:  # pragma: no cover - assertion detail
            peer_errors.append(exc)

    def fail_owner_output_fsync(descriptor: int) -> None:
        nonlocal peer, triggered
        info = os.fstat(descriptor)
        if (info.st_dev, info.st_ino) == output_identity and not triggered:
            triggered = True
            peer = threading.Thread(target=peer_publish)
            peer.start()
            assert peer_started.wait(timeout=5)
            raise OSError("owner output fsync failed")
        real_fsync(descriptor)

    monkeypatch.setattr(module.os, "fsync", fail_owner_output_fsync)
    with pytest.raises(OSError, match="owner output"):
        publish_report(report, root)
    assert peer is not None
    peer.join(timeout=5)

    expected = report_path(root, report["run_sha256"])
    assert peer_errors == []
    assert peer_results == [expected]
    assert expected.read_bytes() == canonical_bytes(report)
    assert not list(output.glob(".*.tmp.*"))


def test_ancestor_swap_is_refused_without_writing_outside_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _repo(tmp_path / "inside")
    report = _report()
    backend = root / "backend"
    moved_backend = root / "backend-original"
    output = backend / "predictor_v5" / "artifacts" / "market_movement"
    output.mkdir()
    outside_backend = tmp_path / "outside" / "backend"
    outside_output = outside_backend / "predictor_v5" / "artifacts" / "market_movement"
    outside_output.mkdir(parents=True)
    module = __import__("predictor_v5.movement_evaluation.report", fromlist=["report"])
    real_open = module.os.open
    swapped = False

    def swapping_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if not swapped and (
            Path(path) == output or (path == "market_movement" and dir_fd is not None)
        ):
            swapped = True
            backend.rename(moved_backend)
            backend.symlink_to(outside_backend, target_is_directory=True)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(module.os, "open", swapping_open)
    with pytest.raises(ContractRefusal):
        publish_report(report, root)

    assert swapped is True
    assert not list(outside_output.glob("development-*.json"))
    assert not list(
        (moved_backend / "predictor_v5" / "artifacts" / "market_movement").glob(
            "development-*.json"
        )
    )


def test_first_publication_fsyncs_parent_then_file_then_output_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _repo(tmp_path)
    report = _report()
    artifacts = root / "backend" / "predictor_v5" / "artifacts"
    output = artifacts / "market_movement"
    artifacts_identity = (artifacts.stat().st_dev, artifacts.stat().st_ino)
    module = __import__("predictor_v5.movement_evaluation.report", fromlist=["report"])
    real_fsync = module.os.fsync
    events: list[str] = []

    def recording_fsync(descriptor: int) -> None:
        info = os.fstat(descriptor)
        identity = (info.st_dev, info.st_ino)
        if identity == artifacts_identity:
            events.append("artifacts")
        elif stat.S_ISREG(info.st_mode):
            events.append("file")
        elif output.exists() and identity == (output.stat().st_dev, output.stat().st_ino):
            events.append("output")
        else:  # pragma: no cover - assertion below reports unexpected descriptors
            events.append("other")
        real_fsync(descriptor)

    monkeypatch.setattr(module.os, "fsync", recording_fsync)
    publish_report(report, root)

    assert events == ["artifacts", "file", "output"]


def test_peer_fsyncs_parent_before_using_an_existing_output_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _repo(tmp_path)
    report = _report()
    artifacts = root / "backend" / "predictor_v5" / "artifacts"
    output = artifacts / "market_movement"
    output.mkdir()
    artifacts_identity = (artifacts.stat().st_dev, artifacts.stat().st_ino)
    module = __import__("predictor_v5.movement_evaluation.report", fromlist=["report"])
    real_fsync = module.os.fsync
    directory_events: list[tuple[int, int]] = []

    def recording_fsync(descriptor: int) -> None:
        info = os.fstat(descriptor)
        if stat.S_ISDIR(info.st_mode):
            directory_events.append((info.st_dev, info.st_ino))
        real_fsync(descriptor)

    monkeypatch.setattr(module.os, "fsync", recording_fsync)
    publish_report(report, root)

    assert directory_events[0] == artifacts_identity


def test_parent_directory_fsync_failure_removes_only_new_output_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _repo(tmp_path)
    report = _report()
    before = _sentinels(root)
    artifacts = root / "backend" / "predictor_v5" / "artifacts"
    output = artifacts / "market_movement"
    artifacts_identity = (artifacts.stat().st_dev, artifacts.stat().st_ino)
    module = __import__("predictor_v5.movement_evaluation.report", fromlist=["report"])
    real_fsync = module.os.fsync

    def fail_parent_fsync(descriptor: int) -> None:
        info = os.fstat(descriptor)
        if (info.st_dev, info.st_ino) == artifacts_identity:
            raise OSError("artifacts fsync failed")
        real_fsync(descriptor)

    monkeypatch.setattr(module.os, "fsync", fail_parent_fsync)
    with pytest.raises(OSError, match="artifacts fsync"):
        publish_report(report, root)

    assert not output.exists()
    assert _sentinels(root) == before
