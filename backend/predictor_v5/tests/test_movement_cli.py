from __future__ import annotations

import ast
import errno
import hashlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from predictor_v5.movement_evaluation import __main__ as cli_module
from predictor_v5.movement_evaluation import data as data_module
from predictor_v5.movement_evaluation import report as report_module
from predictor_v5.movement_evaluation.__main__ import main, parse_args, run
from predictor_v5.movement_evaluation.contract import (
    MODEL_SOURCE_SHA256,
    ContractRefusal,
    canonical_bytes,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = REPO_ROOT / "backend" / "predictor_v5"
SOURCE_NAMES = (
    "model.py",
    "movement_evaluation/__init__.py",
    "movement_evaluation/__main__.py",
    "movement_evaluation/contract.py",
    "movement_evaluation/data.py",
    "movement_evaluation/metrics.py",
    "movement_evaluation/models.py",
    "movement_evaluation/report.py",
)


def test_parse_args_defaults_to_off_and_accepts_only_archive() -> None:
    assert parse_args([]).ktc_diagnostic == "off"
    assert parse_args(["--ktc-diagnostic", "archive"]).ktc_diagnostic == "archive"


def test_help_exposes_only_the_isolated_diagnostic_option(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        parse_args(["--help"])
    assert raised.value.code == 0
    output = capsys.readouterr().out
    assert "--ktc-diagnostic {off,archive}" in output
    for forbidden in ("--season", "--data-path", "--output", "--model", "--threshold"):
        assert forbidden not in output


@pytest.mark.parametrize(
    "arguments",
    [
        ["--ktc", "archive"],
        ["--ktc-d", "off"],
        ["--ktc-di", "archive"],
        ["--h"],
        ["--hel"],
    ],
)
def test_abbreviated_options_exit_two_before_run(
    arguments: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_run(**_kwargs: object) -> Path:
        raise AssertionError("abbreviated options must refuse before evaluator stages")

    monkeypatch.setattr(
        "predictor_v5.movement_evaluation.__main__.run", forbidden_run
    )
    with pytest.raises(SystemExit) as raised:
        main(arguments)
    assert raised.value.code == 2


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        (["--ktc", "archive"], 2),
        (["--ktc-d", "off"], 2),
        (["--h"], 2),
        (["--ktc-diagnostic", "archive"], 0),
        (["--ktc-diagnostic=off"], 0),
        (["-h"], 0),
        (["--help"], 0),
    ],
)
def test_parser_subprocess_accepts_only_exact_documented_options(
    arguments: list[str], expected: int
) -> None:
    script = (
        "import sys; "
        "from predictor_v5.movement_evaluation.__main__ import parse_args; "
        "parse_args(sys.argv[1:])"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, *arguments],
        cwd=REPO_ROOT / "backend",
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == expected, completed.stderr


@pytest.mark.parametrize(
    "arguments",
    [
        ["--season", "2025"],
        ["--data-path", "/tmp/data"],
        ["--output", "/tmp/report.json"],
        ["--model", "production"],
        ["--threshold", "0.1"],
        ["--registry"],
        ["--promote"],
        ["--save"],
        ["positional"],
        ["--unknown"],
        ["--ktc-diagnostic", "2026"],
    ],
)
def test_invalid_arguments_exit_two_before_run(
    arguments: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_run(**_kwargs: object) -> Path:
        raise AssertionError("argument refusal must happen before any evaluator stage")

    monkeypatch.setattr(
        "predictor_v5.movement_evaluation.__main__.run", forbidden_run
    )
    with pytest.raises(SystemExit) as raised:
        main(arguments)
    assert raised.value.code == 2


def _install_stage_spies(
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
    published: list[bytes],
) -> None:
    bundle = SimpleNamespace(identities={"input": "identity"})
    evaluation = SimpleNamespace(primary_scored="primary", coverage={"ledger": "fixed"})
    folds = ("fold-2021", "fold-2022")
    fold_scores = ("score-2021", "score-2022")
    pooled = {"evidence_flags": {"positive": False}}
    optional = {"status": "off"}

    def validate(value: str) -> str:
        events.append("contract")
        return value

    def runtime_identity() -> dict[str, object]:
        events.append("runtime")
        return {"runtime": "fixed"}

    def source_identities(package_root: Path) -> dict[str, str]:
        events.append("sources")
        assert package_root.name == "predictor_v5"
        return {"model.py": MODEL_SOURCE_SHA256, "source": "hash"}

    def load_pinned_bundle(repo_root: Path, *, ktc_mode: str) -> object:
        events.append("load")
        assert isinstance(repo_root, Path)
        assert ktc_mode in {"off", "archive"}
        return bundle

    def prepare_evaluation_data(value: object) -> object:
        events.append("ledger")
        assert value is bundle
        return evaluation

    def run_primary_folds(rows: object) -> tuple[str, ...]:
        events.append("folds")
        assert rows == "primary"
        return folds

    def summarize_primary(values: object) -> tuple[tuple[str, ...], dict[str, object]]:
        events.append("summary")
        assert values is folds
        return fold_scores, pooled

    def attach_start_observable_diagnostics(value: object, values: object) -> dict[str, object]:
        events.append("start_diagnostics")
        assert value is evaluation
        assert values is folds
        return {"observed_zero_end": [], "thin_endpoint": []}

    def run_ktc_diagnostic(
        value: object, *, mode: str, primary_folds: object
    ) -> dict[str, object]:
        events.append("ktc")
        assert value is evaluation
        assert primary_folds is folds
        assert mode in {"off", "archive"}
        return optional

    def build_report(**kwargs: object) -> dict[str, object]:
        events.append("report")
        assert kwargs == {
            "bundle": bundle,
            "data": evaluation,
            "folds": folds,
            "fold_scores": fold_scores,
            "pooled": pooled
            | {
                "start_observable_diagnostics": {
                    "observed_zero_end": [],
                    "thin_endpoint": [],
                }
            },
            "optional_ktc": optional,
            "sources": {"model.py": MODEL_SOURCE_SHA256, "source": "hash"},
            "mode": "off",
            "runtime": {"runtime": "fixed"},
        }
        return {
            "run_sha256": "a" * 64,
            "interpretation": {"evidence_flags": {"positive": False}},
        }

    def publish_report(value: object, repo_root: Path) -> Path:
        events.append("publish")
        published.append(canonical_bytes(value))
        return repo_root / "backend/predictor_v5/artifacts/market_movement/report.json"

    module = "predictor_v5.movement_evaluation.__main__"
    monkeypatch.setattr(f"{module}.validate_ktc_mode", validate)
    monkeypatch.setattr(f"{module}.runtime_identity", runtime_identity)
    monkeypatch.setattr(f"{module}.source_identities", source_identities)
    monkeypatch.setattr(f"{module}.load_pinned_bundle", load_pinned_bundle)
    monkeypatch.setattr(f"{module}.prepare_evaluation_data", prepare_evaluation_data)
    monkeypatch.setattr(f"{module}.run_primary_folds", run_primary_folds)
    monkeypatch.setattr(f"{module}.summarize_primary", summarize_primary)
    monkeypatch.setattr(
        f"{module}.attach_start_observable_diagnostics",
        attach_start_observable_diagnostics,
    )
    monkeypatch.setattr(f"{module}.run_ktc_diagnostic", run_ktc_diagnostic)
    monkeypatch.setattr(f"{module}.build_report", build_report)
    monkeypatch.setattr(f"{module}.publish_report", publish_report)


def test_run_stages_are_ordered_and_report_assembly_is_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    published: list[bytes] = []
    _install_stage_spies(monkeypatch, events, published)

    first = run(ktc_mode="off", repo_root=tmp_path)
    second = run(ktc_mode="off", repo_root=tmp_path)

    expected_order = [
        "contract",
        "runtime",
        "sources",
        "load",
        "ledger",
        "folds",
        "summary",
        "start_diagnostics",
        "ktc",
        "report",
        "publish",
    ]
    assert events == expected_order * 2
    assert first == second
    assert published[0] == published[1]


def test_changed_protected_model_refuses_before_pinned_input_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "predictor_v5.movement_evaluation.__main__.runtime_identity", dict
    )
    monkeypatch.setattr(
        "predictor_v5.movement_evaluation.__main__.source_identities",
        lambda _root: {"model.py": "0" * 64},
    )

    def forbidden_load(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("model identity must refuse before reading inputs")

    monkeypatch.setattr(
        "predictor_v5.movement_evaluation.__main__.load_pinned_bundle",
        forbidden_load,
    )
    with pytest.raises(ContractRefusal, match="model.py"):
        run(ktc_mode="off", repo_root=tmp_path)


@pytest.mark.parametrize(
    ("failure", "exception"),
    [
        ("runtime_identity", ContractRefusal("contract")),
        ("source_identities", ContractRefusal("source")),
        ("load_pinned_bundle", ContractRefusal("input")),
        ("prepare_evaluation_data", ContractRefusal("cohort")),
        ("run_primary_folds", FloatingPointError("fit")),
        ("summarize_primary", ArithmeticError("metric")),
        ("attach_start_observable_diagnostics", RuntimeError("start diagnostic")),
        ("run_ktc_diagnostic", RuntimeError("diagnostic")),
        ("build_report", OSError("assembly")),
    ],
)
def test_every_prepublication_failure_skips_publication_and_leaves_no_report(
    failure: str,
    exception: Exception,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    published: list[bytes] = []
    _install_stage_spies(monkeypatch, events, published)

    def fail(*_args: object, **_kwargs: object) -> None:
        raise exception

    monkeypatch.setattr(
        f"predictor_v5.movement_evaluation.__main__.{failure}", fail
    )
    with pytest.raises(type(exception)):
        run(ktc_mode="off", repo_root=tmp_path)
    assert published == []
    assert not (tmp_path / "backend/predictor_v5/artifacts/market_movement").exists()


@pytest.mark.parametrize(
    ("exception", "expected"),
    [
        (ContractRefusal("invalid pinned input"), 2),
        (RuntimeError("fit failed"), 1),
        (FloatingPointError("numerical failure"), 1),
        (OSError("I/O failure"), 1),
    ],
)
def test_main_maps_only_contract_refusals_to_exit_two(
    exception: Exception,
    expected: int,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(**_kwargs: object) -> Path:
        raise exception

    monkeypatch.setattr("predictor_v5.movement_evaluation.__main__.run", fail)
    assert main([]) == expected
    assert str(exception) in capsys.readouterr().err


def _reader_boundary_path(boundary: str) -> Path:
    if boundary.startswith("source_"):
        return PACKAGE_ROOT / "movement_evaluation" / "contract.py"
    return REPO_ROOT / data_module.INPUT_IDENTITIES["labels"]["path"]


def _fail_reader_boundary(
    monkeypatch: pytest.MonkeyPatch,
    *,
    boundary: str,
    error_number: int,
) -> None:
    target = _reader_boundary_path(boundary)
    if boundary.endswith("_lstat"):
        real_lstat = Path.lstat

        def fail_lstat(path: Path):
            if path == target:
                raise OSError(error_number, "injected reader lstat failure")
            return real_lstat(path)

        monkeypatch.setattr(Path, "lstat", fail_lstat)
        return

    real_read_bytes = Path.read_bytes

    def fail_read_bytes(path: Path) -> bytes:
        if path == target:
            raise OSError(error_number, "injected reader read failure")
        return real_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", fail_read_bytes)


@pytest.mark.parametrize(
    "boundary",
    ["source_lstat", "source_read", "input_lstat", "input_read"],
)
@pytest.mark.parametrize(
    "error_number",
    [errno.EIO, errno.EMFILE, errno.EACCES, errno.EPERM, errno.ENFILE, errno.ENOSPC],
)
def test_actual_readers_map_operational_io_errors_to_exit_one(
    boundary: str,
    error_number: int,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _fail_reader_boundary(
        monkeypatch,
        boundary=boundary,
        error_number=error_number,
    )

    assert main([]) == 1
    error = capsys.readouterr().err
    assert "movement evaluation failed" in error
    assert "injected reader" in error


@pytest.mark.parametrize(
    "boundary",
    ["source_lstat", "source_read", "input_lstat", "input_read"],
)
@pytest.mark.parametrize("error_number", [errno.ENOENT, errno.ENOTDIR, errno.ELOOP])
def test_actual_readers_map_missing_or_raced_paths_to_exit_two(
    boundary: str,
    error_number: int,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _fail_reader_boundary(
        monkeypatch,
        boundary=boundary,
        error_number=error_number,
    )

    assert main([]) == 2
    error = capsys.readouterr().err
    assert "movement evaluation refused" in error


def test_main_returns_zero_for_published_report_regardless_of_evidence_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "predictor_v5.movement_evaluation.__main__.run",
        lambda **_kwargs: Path("report-with-negative-or-unavailable-evidence.json"),
    )
    assert main([]) == 0


def test_main_maps_real_publisher_enospc_to_exit_one_without_partial_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo_root = tmp_path / "repo"
    artifacts = repo_root / "backend/predictor_v5/artifacts"
    artifacts.mkdir(parents=True)
    synthetic_cli = (
        repo_root / "backend/predictor_v5/movement_evaluation/__main__.py"
    )
    bundle = SimpleNamespace()
    data = SimpleNamespace(primary_scored="primary")
    folds = ("folds",)
    identity = {"synthetic": "operational-io-boundary"}
    digest = hashlib.sha256(report_module.canonical_bytes(identity)).hexdigest()
    report = {"run_identity": identity, "run_sha256": digest}

    monkeypatch.setattr(cli_module, "__file__", str(synthetic_cli))
    monkeypatch.setattr(cli_module, "runtime_identity", dict)
    monkeypatch.setattr(
        cli_module,
        "source_identities",
        lambda _root: {"model.py": MODEL_SOURCE_SHA256},
    )
    monkeypatch.setattr(cli_module, "load_pinned_bundle", lambda *_args, **_kwargs: bundle)
    monkeypatch.setattr(cli_module, "prepare_evaluation_data", lambda value: data)
    monkeypatch.setattr(cli_module, "run_primary_folds", lambda _rows: folds)
    monkeypatch.setattr(cli_module, "summarize_primary", lambda _folds: ((), {}))
    monkeypatch.setattr(
        cli_module, "attach_start_observable_diagnostics", lambda *_args: {}
    )
    monkeypatch.setattr(
        cli_module,
        "run_ktc_diagnostic",
        lambda *_args, **_kwargs: {"status": "off"},
    )
    monkeypatch.setattr(cli_module, "build_report", lambda **_kwargs: report)
    real_mkdir = report_module.os.mkdir

    def disk_full(path: str, mode: int = 0o777, *, dir_fd: int | None = None) -> None:
        if path == "market_movement":
            raise OSError(errno.ENOSPC, "No space left on device")
        real_mkdir(path, mode, dir_fd=dir_fd)

    monkeypatch.setattr(report_module.os, "mkdir", disk_full)

    assert main([]) == 1
    assert "movement evaluation failed" in capsys.readouterr().err
    output = artifacts / "market_movement"
    assert not output.exists()
    assert not list(artifacts.rglob("development-*.json"))
    assert not list(artifacts.rglob(".*.tmp.*"))


def test_main_maps_removed_output_parent_race_to_exit_two_without_partial_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo_root = tmp_path / "repo"
    artifacts = repo_root / "backend/predictor_v5/artifacts"
    artifacts.mkdir(parents=True)
    synthetic_cli = (
        repo_root / "backend/predictor_v5/movement_evaluation/__main__.py"
    )
    bundle = SimpleNamespace()
    data = SimpleNamespace(primary_scored="primary")
    folds = ("folds",)
    identity = {"synthetic": "removed-output-parent-race"}
    digest = hashlib.sha256(report_module.canonical_bytes(identity)).hexdigest()
    report = {"run_identity": identity, "run_sha256": digest}

    monkeypatch.setattr(cli_module, "__file__", str(synthetic_cli))
    monkeypatch.setattr(cli_module, "runtime_identity", dict)
    monkeypatch.setattr(
        cli_module,
        "source_identities",
        lambda _root: {"model.py": MODEL_SOURCE_SHA256},
    )
    monkeypatch.setattr(cli_module, "load_pinned_bundle", lambda *_args, **_kwargs: bundle)
    monkeypatch.setattr(cli_module, "prepare_evaluation_data", lambda value: data)
    monkeypatch.setattr(cli_module, "run_primary_folds", lambda _rows: folds)
    monkeypatch.setattr(cli_module, "summarize_primary", lambda _folds: ((), {}))
    monkeypatch.setattr(
        cli_module, "attach_start_observable_diagnostics", lambda *_args: {}
    )
    monkeypatch.setattr(
        cli_module,
        "run_ktc_diagnostic",
        lambda *_args, **_kwargs: {"status": "off"},
    )
    monkeypatch.setattr(cli_module, "build_report", lambda **_kwargs: report)
    real_mkdir = report_module.os.mkdir

    def remove_parent_then_mkdir(
        path: str, mode: int = 0o777, *, dir_fd: int | None = None
    ) -> None:
        if path == "market_movement":
            artifacts.rmdir()
        real_mkdir(path, mode, dir_fd=dir_fd)

    monkeypatch.setattr(report_module.os, "mkdir", remove_parent_then_mkdir)

    assert main([]) == 2
    assert "movement evaluation refused" in capsys.readouterr().err
    assert not artifacts.exists()
    assert not list(repo_root.rglob("development-*.json"))
    assert not list(repo_root.rglob(".*.tmp.*"))


def test_real_runtime_reads_only_pinned_inputs_and_source_identity_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[Path] = []
    original_input_read = data_module._path_bytes
    original_source_read = report_module._regular_source
    original_prepare = data_module.prepare_evaluation_data

    def input_read(path: Path, expected_hash: str, label: str) -> bytes:
        observed.append(path.resolve())
        return original_input_read(path, expected_hash, label)

    def source_read(path: Path) -> bytes:
        observed.append(path.resolve())
        return original_source_read(path)

    def prepare_without_ktc(bundle: object) -> object:
        assert "start_ktc" not in bundle.archive
        assert "end_ktc" not in bundle.archive
        return original_prepare(bundle)

    def forbid_write(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("prepublication stages must not write")

    monkeypatch.setattr(data_module, "_path_bytes", input_read)
    monkeypatch.setattr(report_module, "_regular_source", source_read)
    monkeypatch.setattr(
        "predictor_v5.movement_evaluation.__main__.prepare_evaluation_data",
        prepare_without_ktc,
    )
    for method in ("write_bytes", "write_text", "mkdir", "unlink", "replace", "rename"):
        monkeypatch.setattr(Path, method, forbid_write)
    monkeypatch.setattr(
        "predictor_v5.movement_evaluation.__main__.run_primary_folds",
        lambda _rows: (),
    )
    monkeypatch.setattr(
        "predictor_v5.movement_evaluation.__main__.summarize_primary",
        lambda _folds: ((), {}),
    )
    monkeypatch.setattr(
        "predictor_v5.movement_evaluation.__main__.attach_start_observable_diagnostics",
        lambda _data, _folds: {},
    )
    monkeypatch.setattr(
        "predictor_v5.movement_evaluation.__main__.run_ktc_diagnostic",
        lambda _data, *, mode, primary_folds: {"status": mode},
    )
    monkeypatch.setattr(
        "predictor_v5.movement_evaluation.__main__.build_report",
        lambda **_kwargs: {"run_sha256": "b" * 64},
    )
    monkeypatch.setattr(
        "predictor_v5.movement_evaluation.__main__.publish_report",
        lambda _report, repo_root: repo_root / "allowed-report.json",
    )

    assert run(ktc_mode="off", repo_root=REPO_ROOT) == REPO_ROOT / "allowed-report.json"

    expected = {
        (REPO_ROOT / "backend/data/training-data.zip").resolve(),
        (PACKAGE_ROOT / "data/backchannel_labels_2019_2025_mfl.csv").resolve(),
        (
            PACKAGE_ROOT
            / "data/backchannel_labels_2019_2025_mfl.provenance.json"
        ).resolve(),
    } | {(PACKAGE_ROOT / name).resolve() for name in SOURCE_NAMES}
    assert set(observed) == expected
    assert len(observed) == len(expected)
    forbidden_fragments = (
        "start_values_2026",
        "current",
        "board",
        "registration",
        "claim",
        "holdout_registry",
        "outcome",
        "2026.csv",
    )
    assert not any(
        fragment in str(path).lower()
        for path in observed
        for fragment in forbidden_fragments
    )


def test_real_source_identity_hashes_all_seven_package_files_and_model() -> None:
    identities = report_module.source_identities(PACKAGE_ROOT)
    assert tuple(identities) == SOURCE_NAMES
    assert identities["model.py"] == MODEL_SOURCE_SHA256
    assert identities == {
        name: hashlib.sha256((PACKAGE_ROOT / name).read_bytes()).hexdigest()
        for name in SOURCE_NAMES
    }


def test_evaluator_sources_do_not_import_or_call_forbidden_runtime_boundaries() -> None:
    forbidden_import_parts = {
        "asyncpg",
        "aiohttp",
        "httpx",
        "requests",
        "socket",
        "urllib",
        "api",
        "router",
        "confirmation",
        "evaluate",
        "ktc_replacement_2026",
        "replacement_cohort",
        "replacement_contract",
        "replacement_ktc",
        "replacement_lifecycle",
        "replacement_metrics",
        "replacement_mfl",
        "replacement_report",
        "replacement_seal",
        "replacement_starts",
        "registry",
        "database",
        "db",
    }
    forbidden_calls = {"save", "connect", "create_pool", "request", "urlopen"}

    for source_name in SOURCE_NAMES[1:]:
        tree = ast.parse((PACKAGE_ROOT / source_name).read_text())
        imports: list[str] = []
        calls: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    calls.append(node.func.attr)
                elif isinstance(node.func, ast.Name):
                    calls.append(node.func.id)
        assert not {
            part
            for imported in imports
            for part in imported.split(".")
            if part in forbidden_import_parts
        }, source_name
        assert forbidden_calls.isdisjoint(calls), source_name
