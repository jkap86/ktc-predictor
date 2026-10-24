# Task 5 report — corrupt registered ZIP archives

Status: implemented and verified; ready for Task 3 resealing.

Commit: the task commit containing this report (see repository HEAD / final handoff for the exact SHA).

## Changes

- Added `_read_registered_stats_archive()` in `confirmation.py` and routed both the training archive and registered target-stats archive through it.
- Classified `BadZipFile`, `LargeZipFile`, missing JSON members, malformed JSON, invalid stats envelopes, and invalid stats rows as purpose-labelled `ValueError` failures that preflight exposes as `PreflightError`.
- Kept the exception boundary narrow; unexpected runtime failures still propagate and the CLI returns exit code 1.
- Added hash-rebound fixture tests for both registered archive purposes. Every corrupt-input case asserts preflight refusal, CLI exit 2, no report, and no fit.
- Updated scoring-test injection to target the new archive helper.

## TDD evidence

- RED: `pytest backend/predictor_v5/tests/test_confirmation_preflight.py -k corrupt_registered -q` — 8 intended failures before implementation (the matrix was subsequently expanded with invalid-row cases).
- GREEN: the final corruption matrix covers 10 cases (five corruptions across training and target-stats archives).

## Verification

- Focused preflight/CLI: `114 passed`.
- Scoring suite: `34 passed`.
- Ruff on the three changed Python files: clean.
- `git diff --check`: clean.
- Full predictor suite: `279 passed, 9 skipped, 6 failed`.
- The six full-suite failures are the expected stale-artifact failures in `test_confirmation_protocol.py`: the committed manifest has not yet been resealed and its `source_identities` do not match the current frozen source set. No manifest or registry artifacts were changed in this task.

## Self-review / concerns

- No blanket exception handler was introduced.
- Valid archive conversion preserves the existing field/default semantics used by `read_season_stats`.
- The helper continues the established behavior of selecting the first `.json` ZIP member.
- Remaining action is the separately assigned Task 3 reseal; until then the six committed-protocol tests intentionally remain red.
