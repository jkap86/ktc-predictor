"""Label-source experiments and the deployment gate for Backchannel labels.

Schema 2 runs the comparison in three separate stages so the final holdout
never picks the model it then judges:

A. Selection. The candidate's training-window length is chosen on seasons
   strictly before ``test_season`` (default: validate on 2024, training on
   seasons ending 2023). A length is eligible only when every season of its
   window has rows for every validation season, so no length is scored on a
   silently shorter window. The best pooled validation skill wins; lengths
   within ``tie_margin`` of it go to the shortest.
B. Freeze. Label file, market model version, window length, ``min_trades``,
   features, model parameters and evaluation target are recorded.
C. Final evaluation. The frozen candidate and the deployed control are scored
   on ``test_season`` and the deployment guards are applied.

Head-to-head scores use one target per comparison. The model predicts
log(end / start), so a prediction is a relative change; it is applied to the
target's own start value and scored against the target's end value on the
same player-season keys. Every predictor in a comparison then shares the
same no-change baseline, so MAE and skill compare directly.

Every threshold (selection tie margin, +2-point gain, 5-point position drop)
is applied to full-precision skill; the report rounds for display and keeps
``skill_exact`` beside each rounded skill. A model version counts only when
``label_export.label_identity`` verifies it from the file's SHA-256; a
caller's version is an assertion checked against that. Comparisons across
versions, or with an unverified label file, can never make a candidate
eligible for promotion, and nothing here deploys a model.
"""

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from predictor_v5.data import DEFAULT_ZIP, build_dataset, load_season_stats
from predictor_v5.evaluate import holdout, round_summary, summarize_exact
from predictor_v5.label_export import LABEL_MANIFEST, file_sha256, label_identity
from predictor_v5.labels import CsvLabels
from predictor_v5.model import FEATURES, HGB_PARAMS, MONOTONIC

SCHEMA_VERSION = 2
POSITIONS = ("QB", "RB", "WR", "TE")
SCALE_MAX = 10000.0

MIN_GAIN = 0.02
MIN_POSITION_ROWS = 30
MAX_POSITION_DROP = 0.05
TIE_MARGIN = 0.01
# Absorbs float error only (0.42 - 0.40 is 0.01999...98); far below any real difference.
EPS = 1e-12

TARGET_DEFINITION = (
    "end_value of the named label set at the season's end cutoff, on (player_id, season) keys present in "
    "every compared predictor's test rows after the min_trades filter; each prediction's relative change "
    "(pred_value / own start_value) is applied to the target's start_value, and no-change predicts the "
    "target's start_value"
)

HOLDOUT_STATUSES = ("reused", "untouched", "unknown")
# Seasons whose history as a test season is on record. This registry is the
# only verified source: an entry lands here through a reviewed commit, with
# evidence someone can check. A caller's --holdout-status is recorded as a
# declaration but stays unverified, so it can never make evidence sufficient.
# Anything else is "unknown".
KNOWN_HOLDOUTS = {
    2025: {
        "status": "reused",
        "evidence": "PR #3 (merged 2026-10-22) chose its training window from 2025 scores",
    },
}
PROMOTION_NOTE = (
    "compare-labels never deploys a model; promotion is a manual change to season.py after review, and needs "
    "a passing gate on verified, same-version labels and an untouched test season"
)


@dataclass(frozen=True)
class LabelSet:
    """A built training table plus what it was built from.

    ``model_version`` is None when unknown. ``verified`` is True only when the
    version was confirmed from the file's SHA-256, not from a caller's say-so.
    """

    name: str
    frame: pd.DataFrame
    model_version: str | None
    verified: bool = False
    source: str = ""

    @property
    def identity(self) -> str:
        return f"{self.name} ({self.source})" if self.source else self.name


def coverage_counts(frame: pd.DataFrame) -> dict[str, int]:
    """Deterministic player-season row counts for each represented season."""
    counts = frame.groupby("season", sort=True).size()
    return {str(int(season)): int(count) for season, count in counts.items()}


def coverage_by_position(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Row counts per season and position, every position listed."""
    out = {}
    for season, group in frame.groupby("season", sort=True):
        counts = group["position"].value_counts()
        out[str(int(season))] = {p: int(counts.get(p, 0)) for p in POSITIONS}
    return out


def window(years: int, last_season: int) -> list[int]:
    """The ``years`` seasons ending with ``last_season``."""
    return list(range(last_season - years + 1, last_season + 1))


def _keys(frame: pd.DataFrame) -> set[tuple[str, int]]:
    return set(zip(frame["player_id"].astype(str), frame["season"].astype(int)))


def matched_keys(frames: list[pd.DataFrame], season: int) -> list[tuple[str, int]]:
    """(player_id, season) keys present in every frame for ``season``, sorted."""
    sets = [_keys(f[f["season"] == season]) for f in frames]
    return sorted(set.intersection(*sets)) if sets else []


def _seasons(frame: pd.DataFrame) -> set[int]:
    return {int(s) for s in frame["season"].unique()}


def _fit_predict(
    label_set: LabelSet,
    train_seasons: list[int],
    test_season: int,
    log: list,
    train_keys: set[tuple[str, int]] | None = None,
) -> pd.DataFrame:
    """Fit on exactly ``train_seasons`` (optionally only ``train_keys``) and predict ``test_season``.

    Every requested season must have training rows: a window is never
    silently shortened.
    """
    if max(train_seasons) >= test_season:
        raise ValueError(f"{label_set.identity}: training seasons {train_seasons} must all precede {test_season}")
    frame = label_set.frame
    train = frame[frame["season"].isin(train_seasons)]
    if train_keys is not None:
        train = train[[key in train_keys for key in zip(train["player_id"].astype(str), train["season"].astype(int))]]
    missing = sorted(set(train_seasons) - _seasons(train))
    if missing:
        raise ValueError(
            f"{label_set.identity}: no training rows for seasons {missing} of the requested {sorted(train_seasons)}"
        )
    test = frame[frame["season"] == test_season]
    if test.empty:
        raise ValueError(f"{label_set.identity}: no rows for test season {test_season}")
    log.append(
        {
            "label_set": label_set.name,
            "train_seasons": sorted(train_seasons),
            "train_rows_by_season": coverage_counts(train),
            "training_population": "all" if train_keys is None else "shared",
            "test_season": test_season,
            "test_rows": len(test),
        }
    )
    result = holdout(pd.concat([train, test]), label_set.name, SCALE_MAX, sorted(train_seasons), test_season)
    return result["predictions"][["player_id", "season", "position", "start_value", "end_value", "pred_value"]]


def _summary(frame: pd.DataFrame) -> dict:
    """Full-precision summaries; ``present`` rounds them for the report."""
    return {
        "overall": summarize_exact(frame),
        "by_position": {p: summarize_exact(g) for p, g in frame.groupby("position", sort=True)},
    }


def score_on_target(predictions: pd.DataFrame, target: pd.DataFrame, keys: list[tuple[str, int]]) -> dict:
    """Score ``predictions`` against ``target`` on exactly ``keys``, at full precision.

    The predicted relative change is applied to the target's start value, so
    a model trained on another label set is judged on the target's scale and
    against the same no-change baseline as every other predictor.
    """
    index = pd.MultiIndex.from_tuples(keys, names=["player_id", "season"])
    own = predictions.assign(player_id=predictions["player_id"].astype(str)).set_index(["player_id", "season"])
    truth = target.assign(player_id=target["player_id"].astype(str)).set_index(["player_id", "season"])
    missing = [key for key in keys if key not in own.index or key not in truth.index]
    if missing:
        raise KeyError(f"keys missing from predictions or target: {missing[:5]}")
    own = own.loc[index]
    truth = truth.loc[index]
    scored = pd.DataFrame(
        {
            "position": truth["position"].to_numpy(),
            "start_value": truth["start_value"].to_numpy(float),
            "end_value": truth["end_value"].to_numpy(float),
            "pred_value": truth["start_value"].to_numpy(float)
            * own["pred_value"].to_numpy(float)
            / own["start_value"].to_numpy(float),
        }
    )
    return _summary(scored)


def select_training_window(
    label_set: LabelSet,
    *,
    train_years: tuple[int, ...],
    validation_seasons: tuple[int, ...],
    test_season: int,
    tie_margin: float = TIE_MARGIN,
    log: list | None = None,
) -> dict:
    """Stage A: choose a window length using only seasons before ``test_season``."""
    if not validation_seasons:
        raise ValueError("at least one validation season is needed")
    if max(validation_seasons) >= test_season:
        raise ValueError(f"validation seasons {list(validation_seasons)} must precede the final test season {test_season}")
    log = [] if log is None else log
    history = LabelSet(
        label_set.name,
        label_set.frame[label_set.frame["season"] < test_season],
        label_set.model_version,
        label_set.verified,
        label_set.source,
    )
    available = _seasons(history.frame)

    rows = []
    for years in sorted(set(train_years)):
        windows = {v: window(years, v - 1) for v in sorted(validation_seasons)}
        missing = sorted({s for w in windows.values() for s in w} - available)
        missing_validation = sorted(set(validation_seasons) - available)
        entry = {"train_years": years, "windows": {str(v): w for v, w in windows.items()}}
        if missing or missing_validation:
            reason = []
            if missing:
                reason.append(f"no training rows for {missing}")
            if missing_validation:
                reason.append(f"no validation rows for {missing_validation}")
            rows.append({**entry, "eligible": False, "reason": "; ".join(reason)})
            continue
        preds = pd.concat([_fit_predict(history, w, v, log) for v, w in windows.items()], ignore_index=True)
        rows.append(
            {
                **entry,
                "eligible": True,
                "by_validation_season": {str(int(v)): summarize_exact(g) for v, g in preds.groupby("season", sort=True)},
                "pooled": _summary(preds),
            }
        )

    ranked = [r for r in rows if r["eligible"] and r["pooled"]["overall"]["skill"] is not None]
    if not ranked:
        return {
            "validation_seasons": sorted(validation_seasons),
            "candidates": rows,
            "selected_train_years": None,
            "rationale": "no window length had complete training and validation data",
        }
    best = max(r["pooled"]["overall"]["skill"] for r in ranked)
    near = [r for r in ranked if best - r["pooled"]["overall"]["skill"] <= tie_margin + EPS]
    chosen = min(near, key=lambda r: r["train_years"])
    rationale = (
        f"best pooled validation skill {best:.4f}; {chosen['train_years']}-season window scores "
        f"{chosen['pooled']['overall']['skill']:.4f} and is the shortest within {tie_margin} of the best "
        "(compared at full precision)"
    )
    return {
        "validation_seasons": sorted(validation_seasons),
        "candidates": rows,
        "selected_train_years": chosen["train_years"],
        "rationale": rationale,
    }


def apply_gate(
    control: dict,
    candidate: dict,
    *,
    min_gain: float = MIN_GAIN,
    min_position_rows: int = MIN_POSITION_ROWS,
    max_position_drop: float = MAX_POSITION_DROP,
) -> dict:
    """Overall gain, per-position sample and per-position drop guards on one target, at full precision."""
    base_skill = control["overall"]["skill"]
    skill = candidate["overall"]["skill"]
    gain = None if skill is None or base_skill is None else skill - base_skill
    overall_guard = gain is not None and gain + EPS >= min_gain
    sample_guard = all(candidate["by_position"].get(p, {}).get("n", 0) >= min_position_rows for p in POSITIONS)
    position_guard = all(
        candidate["by_position"].get(p, {}).get("skill") is not None
        and control["by_position"].get(p, {}).get("skill") is not None
        and candidate["by_position"][p]["skill"] + max_position_drop + EPS >= control["by_position"][p]["skill"]
        for p in POSITIONS
    )
    return {
        "gain": gain,
        "overall_guard": overall_guard,
        "sample_guard": sample_guard,
        "position_guard": position_guard,
        "qualified": overall_guard and sample_guard and position_guard,
    }


def _head_to_head(predictors: dict[str, pd.DataFrame], targets: dict[str, pd.DataFrame], keys: list) -> dict:
    """Every predictor scored on every target, on the same keys."""
    return {
        target_name: {name: score_on_target(preds, target, keys) for name, preds in predictors.items()}
        for target_name, target in targets.items()
    }


def _complete(sets: list[LabelSet], seasons: list[int]) -> bool:
    return all(set(seasons) <= _seasons(s.frame) for s in sets)


def holdout_record(test_season: int, declared: dict | None = None) -> dict:
    """Test-season history from ``KNOWN_HOLDOUTS`` or an explicit declaration; otherwise unknown."""
    known = KNOWN_HOLDOUTS.get(test_season)
    if declared:
        status = declared.get("status")
        if status not in HOLDOUT_STATUSES:
            raise ValueError(f"holdout status must be one of {HOLDOUT_STATUSES}, not {status!r}")
        if known and status != known["status"]:
            raise ValueError(f"{test_season} is on record as {known['status']!r}: {known['evidence']}")
        if status == "untouched" and not declared.get("evidence"):
            raise ValueError("declaring an untouched test season needs evidence of how it was held out")
        if not known:
            return {
                "season": test_season,
                "status": status,
                "evidence": declared.get("evidence"),
                "source": "declared",
                "verified": False,
            }
    if known:
        return {"season": test_season, **known, "source": "registry", "verified": True}
    return {"season": test_season, "status": "unknown", "evidence": None, "source": "none", "verified": False}


def compare_datasets(
    baseline: LabelSet,
    candidate: LabelSet,
    *,
    model_effect: LabelSet | None = None,
    train_years: tuple[int, ...] = (3, 4, 5),
    control_train_seasons: list[int] | None = None,
    validation_seasons: tuple[int, ...] = (2024,),
    train_end: int = 2024,
    test_season: int = 2025,
    min_trades: int = 5,
    tie_margin: float = TIE_MARGIN,
    holdout_declaration: dict | None = None,
) -> dict:
    """Selection, freeze, final evaluation and the controlled experiments."""
    if train_end >= test_season:
        raise ValueError("train_end must precede the test season")
    control_train = sorted(control_train_seasons or range(2022, train_end + 1))
    record = holdout_record(test_season, holdout_declaration)
    log: list[dict] = []
    same_version = baseline.model_version is not None and baseline.model_version == candidate.model_version
    verified = baseline.verified and candidate.verified

    # Stage A: selection never sees test_season rows.
    selection = select_training_window(
        candidate,
        train_years=train_years,
        validation_seasons=validation_seasons,
        test_season=test_season,
        tie_margin=tie_margin,
        log=log,
    )
    selected = selection["selected_train_years"]

    # Stage B: freeze.
    frozen = None
    if selected is not None:
        frozen = {
            "label_set": candidate.name,
            "model_version": candidate.model_version,
            "model_version_verified": candidate.verified,
            "train_years": selected,
            "train_seasons": window(selected, train_end),
            "min_trades": min_trades,
            "features": list(FEATURES),
            "monotonic": list(MONOTONIC),
            "model_params": dict(HGB_PARAMS),
            "evaluation_targets": [baseline.name, candidate.name] if same_version else [candidate.name],
        }

    # Stage C: final evaluation of the frozen candidate against the control.
    # _fit_predict refuses a window with a missing season for either of them.
    control_preds = _fit_predict(baseline, control_train, test_season, log)
    final = {
        "test_season": test_season,
        "holdout": record,
        "control": {"label_set": baseline.name, "train_seasons": control_train, "all": _summary(control_preds)},
    }
    gate = {"evaluated": False}
    if frozen is None:
        reason = "no candidate window passed selection"
    else:
        candidate_preds = _fit_predict(candidate, frozen["train_seasons"], test_season, log)
        keys = matched_keys([control_preds, candidate_preds], test_season)
        final["candidate"] = {
            "label_set": candidate.name,
            "train_seasons": frozen["train_seasons"],
            "all": _summary(candidate_preds),
        }
        final["matched_test_rows"] = len(keys)
        if not keys:
            final["status"] = "insufficient_overlap"
            final["overlap"] = _overlap(control_preds, candidate_preds, keys)
            reason = "control and candidate share no test player-seasons, so nothing can be compared"
        elif same_version:
            targets = {baseline.name: baseline.frame, candidate.name: candidate.frame}
            final["matched"] = _head_to_head({"control": control_preds, "candidate": candidate_preds}, targets, keys)
            by_target = {t: apply_gate(s["control"], s["candidate"]) for t, s in final["matched"].items()}
            gate = {"evaluated": True, "by_target": by_target, "qualified": all(g["qualified"] for g in by_target.values())}
            if not gate["qualified"]:
                reason = "a guard failed on at least one target"
            elif not verified:
                reason = "every guard passed, but a label file's model version is unverified; informational only"
            else:
                reason = "every guard passed on every target with verified, same-version labels"
        else:
            reason = (
                f"control ({baseline.model_version or 'unknown'}) and candidate "
                f"({candidate.model_version or 'unknown'}) are not on one known market model version, so their "
                "targets are not comparable; informational only"
            )
    gate.update(
        {
            "min_gain": MIN_GAIN,
            "min_position_rows": MIN_POSITION_ROWS,
            "max_position_drop": MAX_POSITION_DROP,
            "requires_same_model_version": True,
            "requires_verified_model_versions": True,
            "requires_every_target": True,
            "full_precision": True,
        }
    )
    gate_passed = bool(gate.get("qualified"))
    promotion_eligible = gate_passed and same_version and verified
    verified_untouched = record["status"] == "untouched" and record["verified"]
    evidence_sufficient = promotion_eligible and verified_untouched
    decision = {
        # "update" is the experimental recommendation only; see promotion_authorized.
        "action": "update" if promotion_eligible else "retain",
        "gate_passed": gate_passed,
        "promotion_eligible": promotion_eligible,
        "evidence_sufficient_for_promotion": evidence_sufficient,
        "promotion_authorized": False,
        "promotion_note": PROMOTION_NOTE,
        "holdout_verified": record["verified"],
        "requires_unseen_season_confirmation": not verified_untouched,
        "selected_train_years": selected,
        "selected_train_start": frozen["train_seasons"][0] if frozen else None,
        "reason": reason,
    }

    comparable = same_version
    experiments = {
        "label_effect": label_effect(baseline, candidate, train_years, train_end, test_season, comparable, log),
        "label_and_population_effect": _label_and_population_effect(
            baseline, candidate, train_years, train_end, test_season, comparable, log
        ),
        "training_history": {
            s.name: _training_history(s, train_years, validation_seasons, train_end, test_season, tie_margin, log)
            for s in (baseline, candidate)
        },
        "model_version": _model_version_effect(candidate, model_effect, frozen, test_season, log),
    }

    for fit in log:
        if max(fit["train_seasons"]) >= fit["test_season"]:
            raise AssertionError(f"chronological leakage in {fit}")
    sets = [baseline, candidate] + ([model_effect] if model_effect is not None else [])
    limitations = [
        f"test season {test_season} holdout status: {record['status']}"
        + ("" if record["verified"] else " (unverified)")
        + _evidence(record)
    ]
    limitations += [f"{s.name}: model version unverified" for s in sets if not s.verified]
    if not same_version:
        limitations.append("control and candidate are not on one known model version; the gate was not evaluated")
    limitations.extend(experiments["model_version"].get("missing_cells", []))

    report = {
        "schema_version": SCHEMA_VERSION,
        "methodology": {
            "selection": (
                f"window length chosen on validation seasons {sorted(validation_seasons)} (each trained on the "
                f"preceding seasons only) using {candidate.name} labels; {test_season} rows are removed before "
                "selection runs"
            ),
            "final_evaluation": f"frozen candidate and deployed control trained through {train_end}, tested on {test_season}",
            "target": TARGET_DEFINITION,
            "precision": "thresholds use full-precision skill; skill is rounded to 0.001 for display, skill_exact is not",
            "holdout": record,
        },
        "config": {
            "control_train_seasons": control_train,
            "candidate_train_years": sorted(set(train_years)),
            "validation_seasons": sorted(validation_seasons),
            "train_end": train_end,
            "test_season": test_season,
            "min_trades": min_trades,
            "tie_margin": tie_margin,
        },
        "label_sets": {
            s.name: {"model_version": s.model_version, "model_version_verified": s.verified} for s in sets
        },
        "coverage": {
            s.name: {"by_season": coverage_counts(s.frame), "by_season_position": coverage_by_position(s.frame)}
            for s in sets
        },
        "selection": selection,
        "frozen_candidate": frozen,
        "final_evaluation": final,
        "gate": gate,
        "decision": decision,
        "experiments": experiments,
        "fits": log,
        "limitations": limitations,
    }
    return present(report)


def _evidence(record: dict) -> str:
    return f" ({record['evidence']})" if record.get("evidence") else ""


SUMMARY_KEYS = {"n", "model_mae", "no_change_mae", "skill"}


def present(value):
    """Round every summary for display, keeping full-precision skill as ``skill_exact``."""
    if isinstance(value, dict):
        if set(value) == SUMMARY_KEYS:
            return {**round_summary(value), "skill_exact": value["skill"]}
        return {k: present(v) for k, v in value.items()}
    if isinstance(value, list):
        return [present(v) for v in value]
    return value


def _training_population(baseline: LabelSet, candidate: LabelSet, seasons: list[int]) -> tuple[set, dict]:
    """Training keys both label sets have in ``seasons``, with counts of what each set alone adds."""
    b = baseline.frame[baseline.frame["season"].isin(seasons)]
    c = candidate.frame[candidate.frame["season"].isin(seasons)]
    b_keys, c_keys = _keys(b), _keys(c)
    shared = b_keys & c_keys
    shared_rows = b[[key in shared for key in zip(b["player_id"].astype(str), b["season"].astype(int))]]

    def by_season(keys: set) -> dict[str, int]:
        return {str(s): sum(1 for _, season in keys if season == s) for s in seasons}

    return shared, {
        "shared_rows": len(shared),
        "shared_by_season": by_season(shared),
        "shared_by_season_position": coverage_by_position(shared_rows),
        f"{baseline.name}_only_by_season": by_season(b_keys - c_keys),
        f"{candidate.name}_only_by_season": by_season(c_keys - b_keys),
    }


def label_effect(baseline, candidate, train_years, train_end, test_season, comparable, log) -> dict:
    """Experiment A: only the label values differ.

    Both models train on the same (player_id, season) keys (those both label
    sets keep after the min_trades filter), with the same window, features
    and parameters, and are scored on the same test keys against the same
    target. Rows only one label set has are left out of training.
    """
    if not comparable:
        return {
            "status": "not_comparable",
            "reason": "label sets are not on one known market model version, so a label effect cannot be isolated",
        }
    rows = []
    for years in sorted(set(train_years)):
        seasons = window(years, train_end)
        if not _complete([baseline, candidate], seasons):
            rows.append({"train_years": years, "train_seasons": seasons, "status": "insufficient_history"})
            continue
        shared, population = _training_population(baseline, candidate, seasons)
        if set(seasons) - {season for _, season in shared}:
            rows.append({"train_years": years, "train_seasons": seasons, "status": "no_shared_rows_in_a_season"})
            continue
        preds = {s.name: _fit_predict(s, seasons, test_season, log, train_keys=shared) for s in (baseline, candidate)}
        rows.append(
            {
                "train_years": years,
                "train_seasons": seasons,
                "training_population": population,
                **_scores(preds, baseline, candidate, test_season),
            }
        )
    return {
        "status": "informational",
        "used_for_selection": False,
        "training_population": "shared keys only",
        "test_season": test_season,
        "windows": rows,
    }


def _label_and_population_effect(baseline, candidate, train_years, train_end, test_season, comparable, log) -> dict:
    """Operational view: each label set trains on every row it has, so extra rows count too."""
    if not comparable:
        return {"status": "not_comparable"}
    rows = []
    for years in sorted(set(train_years)):
        seasons = window(years, train_end)
        if not _complete([baseline, candidate], seasons):
            rows.append({"train_years": years, "train_seasons": seasons, "status": "insufficient_history"})
            continue
        preds = {s.name: _fit_predict(s, seasons, test_season, log) for s in (baseline, candidate)}
        rows.append({"train_years": years, "train_seasons": seasons, **_scores(preds, baseline, candidate, test_season)})
    return {
        "status": "informational",
        "used_for_selection": False,
        "training_population": "all rows of each label set",
        "test_season": test_season,
        "windows": rows,
    }


def _overlap(first: pd.DataFrame, second: pd.DataFrame, keys: list) -> dict:
    return {"first_rows": len(first), "second_rows": len(second), "matched_rows": len(keys)}


def _scores(preds: dict, baseline: LabelSet, candidate: LabelSet, test_season: int) -> dict:
    keys = matched_keys(list(preds.values()), test_season)
    if not keys:
        return {
            "status": "insufficient_overlap",
            "matched_test_rows": 0,
            "test_rows": {name: len(frame) for name, frame in preds.items()},
        }
    scores = _head_to_head(preds, {s.name: s.frame for s in (baseline, candidate)}, keys)
    return {
        "matched_test_rows": len(keys),
        "scores": scores,
        "skill_gain": {t: _gain(s[candidate.name], s[baseline.name]) for t, s in scores.items()},
    }


def _training_history(label_set, train_years, validation_seasons, train_end, test_season, tie_margin, log) -> dict:
    """Experiment B: window lengths within one label set, validation and final season."""
    validation = select_training_window(
        label_set,
        train_years=train_years,
        validation_seasons=validation_seasons,
        test_season=test_season,
        tie_margin=tie_margin,
        log=log,
    )
    final = []
    for years in sorted(set(train_years)):
        seasons = window(years, train_end)
        if not _complete([label_set], seasons):
            final.append({"train_years": years, "train_seasons": seasons, "status": "insufficient_history"})
            continue
        preds = _fit_predict(label_set, seasons, test_season, log)
        final.append({"train_years": years, "train_seasons": seasons, **_summary(preds)})
    return {
        "validation": validation,
        "final_season_informational": {"used_for_selection": False, "windows": final},
    }


def _model_version_effect(candidate, other, frozen, test_season, log) -> dict:
    """Experiment C: the same trade corpus under two market model versions."""
    if other is None:
        return {"status": "not_run", "missing_cells": ["no second market model version was supplied"]}
    if frozen is None:
        return {"status": "not_run", "reason": "no frozen candidate window"}
    seasons = frozen["train_seasons"]
    if not _complete([other], seasons):
        return {"status": "insufficient_history", "train_seasons": seasons}
    if not (other.frame["season"] == test_season).any():
        return {
            "status": "insufficient_data",
            "reason": f"{other.identity} has no rows for test season {test_season}",
            "gate_eligible": False,
        }
    preds = {s.name: _fit_predict(s, seasons, test_season, log) for s in (candidate, other)}
    keys = matched_keys(list(preds.values()), test_season)
    if not keys:
        return {
            "status": "insufficient_overlap",
            "gate_eligible": False,
            "matched_test_rows": 0,
            "test_rows": {name: len(frame) for name, frame in preds.items()},
        }
    scores = _head_to_head(preds, {s.name: s.frame for s in (candidate, other)}, keys)
    return {
        "status": "informational",
        "gate_eligible": False,
        "versions": {s.name: {"model_version": s.model_version, "verified": s.verified} for s in (candidate, other)},
        "train_seasons": seasons,
        "matched_test_rows": len(keys),
        "scores": scores,
        "skill_gain": {t: _gain(s[other.name], s[candidate.name]) for t, s in scores.items()},
        "mapping": "relative change applied to each target's start value; the two scales differ, so read as informational",
        "missing_cells": [
            (
                f"{other.model_version or 'the other version'} labels without the candidate's added trades were not "
                "supplied, so the model-version effect without them is unresolved"
            )
        ],
    }


def _gain(candidate: dict, control: dict) -> float | None:
    a, b = candidate["overall"]["skill"], control["overall"]["skill"]
    return None if a is None or b is None else a - b


def resolve_label_set(
    name: str,
    path: Path | str,
    frame: pd.DataFrame,
    asserted_version: str | None,
    manifest_path: Path | str = LABEL_MANIFEST,
) -> tuple[LabelSet, dict]:
    """Verified identity for a label file; ``asserted_version`` must agree with it.

    An unverified file keeps the asserted version (if any) for informational
    comparisons, but it can never make a candidate eligible for promotion.
    """
    identity = label_identity(path, manifest_path)
    if identity["verified"] and asserted_version and asserted_version != identity["model_version"]:
        raise ValueError(
            f"{Path(path).name} is {identity['model_version']} by its {identity['source']} provenance, "
            f"not {asserted_version}"
        )
    version = identity["model_version"] if identity["verified"] else asserted_version
    raw = pd.read_csv(path, usecols=["season"])
    info = {
        "file": Path(path).name,
        "sha256": identity["sha256"],
        "rows": len(raw),
        "rows_by_season": coverage_counts(raw),
        "model_version": version,
        "model_version_verified": identity["verified"],
        "provenance_source": identity["source"],
        "asserted_model_version": asserted_version,
        "provenance_problems": identity["problems"],
    }
    source = f"{Path(path).name}, sha256 {identity['sha256'][:12]}"
    return LabelSet(name, frame, version, identity["verified"], source), info


def compare_label_files(
    baseline_path: Path | str,
    candidate_path: Path | str,
    *,
    training_data_path: Path | str = DEFAULT_ZIP,
    min_trades: int = 5,
    train_starts: tuple[int, ...] = (2022, 2021, 2020),
    train_end: int = 2024,
    test_season: int = 2025,
    validation_seasons: tuple[int, ...] = (2024,),
    model_effect_path: Path | str | None = None,
    baseline_model_version: str | None = None,
    candidate_model_version: str | None = None,
    model_effect_model_version: str | None = None,
    manifest_path: Path | str = LABEL_MANIFEST,
    holdout_declaration: dict | None = None,
) -> dict:
    """Load the versioned CSVs and return a deterministic schema-2 report.

    ``train_starts`` keeps the schema-1 CLI meaning: each start becomes a
    window length ending at ``train_end`` (2022 with train_end 2024 is 3).
    The ``*_model_version`` arguments are assertions checked against each
    file's verified provenance, never its identity.
    """
    stats = load_season_stats(training_data_path)
    specs = [("baseline", baseline_path, baseline_model_version), ("candidate", candidate_path, candidate_model_version)]
    if model_effect_path:
        specs.append(("model_effect", model_effect_path, model_effect_model_version))
    sets, inputs = {}, {}
    for name, path, asserted in specs:
        frame = build_dataset(stats, CsvLabels(path, name=name, min_trades=min_trades))
        sets[name], inputs[name] = resolve_label_set(name, path, frame, asserted, manifest_path)
    report = compare_datasets(
        sets["baseline"],
        sets["candidate"],
        model_effect=sets.get("model_effect"),
        train_years=tuple(train_end - start + 1 for start in train_starts),
        validation_seasons=validation_seasons,
        train_end=train_end,
        test_season=test_season,
        min_trades=min_trades,
        holdout_declaration=holdout_declaration,
    )
    report["inputs"] = inputs
    report["inputs"]["training_data"] = {"file": Path(training_data_path).name, "sha256": file_sha256(training_data_path)}
    return report


def write_report(report: dict, out_path: Path | str) -> None:
    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # allow_nan=False: a NaN or infinity anywhere is a bug, not something to write as non-standard JSON.
    destination.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")


def gate_summary(report: dict) -> dict:
    """The decision block without the bulky experiment tables."""
    keep = ("selection", "frozen_candidate", "gate", "decision", "limitations")
    return deepcopy({k: report[k] for k in keep})
