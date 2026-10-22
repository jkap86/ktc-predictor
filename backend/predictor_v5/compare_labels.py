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
same no-change baseline, so MAE and skill compare directly. Comparisons
across market model versions are reported for information only and can
never approve a candidate.
"""

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from predictor_v5.data import DEFAULT_ZIP, build_dataset, load_season_stats
from predictor_v5.evaluate import holdout, summarize
from predictor_v5.label_export import file_sha256
from predictor_v5.labels import CsvLabels
from predictor_v5.model import FEATURES, HGB_PARAMS, MONOTONIC

SCHEMA_VERSION = 2
POSITIONS = ("QB", "RB", "WR", "TE")
SCALE_MAX = 10000.0
DEFAULT_MODEL_VERSION = "baseline-hl28-v1"

MIN_GAIN = 0.02
MIN_POSITION_ROWS = 30
MAX_POSITION_DROP = 0.05
TIE_MARGIN = 0.01
EPS = 1e-12

TARGET_DEFINITION = (
    "end_value of the named label set at the season's end cutoff, on (player_id, season) keys present in "
    "every compared predictor's test rows after the min_trades filter; each prediction's relative change "
    "(pred_value / own start_value) is applied to the target's start_value, and no-change predicts the "
    "target's start_value"
)
HOLDOUT_STATUS = (
    "reused: PR #3 already chose a training window on this season's scores, so this is not an untouched "
    "holdout; an unseen season or separately held-out data is needed before any promotion"
)


@dataclass(frozen=True)
class LabelSet:
    """A built training table plus what it was built from."""

    name: str
    frame: pd.DataFrame
    model_version: str


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


def matched_keys(frames: list[pd.DataFrame], season: int) -> list[tuple[str, int]]:
    """(player_id, season) keys present in every frame for ``season``, sorted."""
    sets = [
        set(zip(f.loc[f["season"] == season, "player_id"].astype(str), f.loc[f["season"] == season, "season"].astype(int)))
        for f in frames
    ]
    return sorted(set.intersection(*sets)) if sets else []


def _seasons(frame: pd.DataFrame) -> set[int]:
    return {int(s) for s in frame["season"].unique()}


def _fit_predict(frame: pd.DataFrame, train_seasons: list[int], test_season: int, name: str, log: list) -> pd.DataFrame:
    if max(train_seasons) >= test_season:
        raise ValueError(f"{name}: training seasons {train_seasons} must all precede {test_season}")
    log.append({"label_set": name, "train_seasons": list(train_seasons), "test_season": test_season})
    result = holdout(frame, name, SCALE_MAX, list(train_seasons), test_season)
    return result["predictions"][["player_id", "season", "position", "start_value", "end_value", "pred_value"]]


def _summary(frame: pd.DataFrame) -> dict:
    return {
        "overall": summarize(frame),
        "by_position": {p: summarize(g) for p, g in frame.groupby("position", sort=True)},
    }


def score_on_target(predictions: pd.DataFrame, target: pd.DataFrame, keys: list[tuple[str, int]]) -> dict:
    """Score ``predictions`` against ``target`` on exactly ``keys``.

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
    frame: pd.DataFrame,
    *,
    train_years: tuple[int, ...],
    validation_seasons: tuple[int, ...],
    test_season: int,
    name: str = "candidate",
    tie_margin: float = TIE_MARGIN,
    log: list | None = None,
) -> dict:
    """Stage A: choose a window length using only seasons before ``test_season``."""
    if not validation_seasons:
        raise ValueError("at least one validation season is needed")
    if max(validation_seasons) >= test_season:
        raise ValueError(f"validation seasons {list(validation_seasons)} must precede the final test season {test_season}")
    log = [] if log is None else log
    history = frame[frame["season"] < test_season]
    available = _seasons(history)

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
        preds = pd.concat(
            [_fit_predict(history, w, v, name, log) for v, w in windows.items()], ignore_index=True
        )
        rows.append(
            {
                **entry,
                "eligible": True,
                "by_validation_season": {str(int(v)): summarize(g) for v, g in preds.groupby("season", sort=True)},
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
        f"best pooled validation skill {best:.3f}; {chosen['train_years']}-season window scores "
        f"{chosen['pooled']['overall']['skill']:.3f} and is the shortest within {tie_margin:.3f} of the best"
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
    """Overall gain, per-position sample and per-position drop guards on one target."""
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
        "gain": None if gain is None else round(gain, 6),
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
) -> dict:
    """Selection, freeze, final evaluation and the controlled experiments."""
    if train_end >= test_season:
        raise ValueError("train_end must precede the test season")
    control_train = sorted(control_train_seasons or range(2022, train_end + 1))
    log: list[dict] = []
    comparable = baseline.model_version == candidate.model_version

    # Stage A: selection never sees test_season rows.
    selection = select_training_window(
        candidate.frame,
        train_years=train_years,
        validation_seasons=validation_seasons,
        test_season=test_season,
        name=candidate.name,
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
            "train_years": selected,
            "train_seasons": window(selected, train_end),
            "min_trades": min_trades,
            "features": list(FEATURES),
            "monotonic": list(MONOTONIC),
            "model_params": dict(HGB_PARAMS),
            "evaluation_targets": [baseline.name, candidate.name] if comparable else [candidate.name],
        }
        if not _complete([candidate], frozen["train_seasons"]):
            frozen = None

    # Stage C: final evaluation of the frozen candidate against the control.
    control_preds = _fit_predict(baseline.frame, control_train, test_season, baseline.name, log)
    final = {
        "test_season": test_season,
        "holdout_status": HOLDOUT_STATUS,
        "control": {"label_set": baseline.name, "train_seasons": control_train, "all": _summary(control_preds)},
    }
    gate = {"evaluated": False}
    reason = None
    if frozen is None:
        reason = "no candidate window passed selection"
    else:
        candidate_preds = _fit_predict(candidate.frame, frozen["train_seasons"], test_season, candidate.name, log)
        keys = matched_keys([control_preds, candidate_preds], test_season)
        targets = {baseline.name: baseline.frame, candidate.name: candidate.frame} if comparable else {}
        final["candidate"] = {
            "label_set": candidate.name,
            "train_seasons": frozen["train_seasons"],
            "all": _summary(candidate_preds),
        }
        final["matched_test_rows"] = len(keys)
        if comparable:
            final["matched"] = _head_to_head(
                {"control": control_preds, "candidate": candidate_preds}, targets, keys
            )
            by_target = {
                t: apply_gate(scores["control"], scores["candidate"]) for t, scores in final["matched"].items()
            }
            qualified = all(g["qualified"] for g in by_target.values())
            gate = {"evaluated": True, "by_target": by_target, "qualified": qualified}
            reason = "every guard passed on every target" if qualified else "a guard failed on at least one target"
        else:
            gate = {"evaluated": False}
            reason = (
                f"control ({baseline.model_version}) and candidate ({candidate.model_version}) use different "
                "market model versions, so their targets are not comparable; informational only"
            )
    gate.update(
        {
            "min_gain": MIN_GAIN,
            "min_position_rows": MIN_POSITION_ROWS,
            "max_position_drop": MAX_POSITION_DROP,
            "requires_same_model_version": True,
            "requires_every_target": True,
        }
    )
    action = "update" if gate.get("qualified") else "retain"
    decision = {
        "action": action,
        "selected_train_years": selected if frozen else None,
        "selected_train_start": frozen["train_seasons"][0] if frozen else None,
        "reason": reason,
        "requires_unseen_season_confirmation": True,
    }

    experiments = {
        "label_effect": _label_effect(baseline, candidate, train_years, train_end, test_season, comparable, log),
        "training_history": {
            s.name: _training_history(s, train_years, validation_seasons, train_end, test_season, tie_margin, log)
            for s in (baseline, candidate)
        },
        "model_version": _model_version_effect(candidate, model_effect, frozen, test_season, log),
    }

    for fit in log:
        if max(fit["train_seasons"]) >= fit["test_season"]:
            raise AssertionError(f"chronological leakage in {fit}")
    limitations = [HOLDOUT_STATUS]
    if not comparable:
        limitations.append("control and candidate use different market model versions; the gate was not evaluated")
    if experiments["model_version"].get("missing_cells"):
        limitations.extend(experiments["model_version"]["missing_cells"])

    sets = [baseline, candidate] + ([model_effect] if model_effect is not None else [])
    return {
        "schema_version": SCHEMA_VERSION,
        "methodology": {
            "selection": (
                f"window length chosen on validation seasons {sorted(validation_seasons)} (each trained on the "
                f"preceding seasons only) using {candidate.name} labels; {test_season} rows are removed before "
                "selection runs"
            ),
            "final_evaluation": f"frozen candidate and deployed control trained through {train_end}, tested on {test_season}",
            "target": TARGET_DEFINITION,
            "holdout_status": HOLDOUT_STATUS,
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
        "label_sets": {s.name: {"model_version": s.model_version} for s in sets},
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


def _label_effect(baseline, candidate, train_years, train_end, test_season, comparable, log) -> dict:
    """Experiment A: same window, same players, same target; only the labels differ."""
    if not comparable:
        return {
            "status": "not_comparable",
            "reason": "label sets use different market model versions, so a label effect cannot be isolated",
        }
    rows = []
    for years in sorted(set(train_years)):
        seasons = window(years, train_end)
        if not _complete([baseline, candidate], seasons):
            rows.append({"train_years": years, "train_seasons": seasons, "status": "insufficient_history"})
            continue
        preds = {s.name: _fit_predict(s.frame, seasons, test_season, s.name, log) for s in (baseline, candidate)}
        keys = matched_keys(list(preds.values()), test_season)
        scores = _head_to_head(preds, {s.name: s.frame for s in (baseline, candidate)}, keys)
        rows.append(
            {
                "train_years": years,
                "train_seasons": seasons,
                "matched_test_rows": len(keys),
                "scores": scores,
                "skill_gain": {
                    t: _gain(s[candidate.name], s[baseline.name]) for t, s in scores.items()
                },
            }
        )
    return {"status": "informational", "used_for_selection": False, "test_season": test_season, "windows": rows}


def _training_history(label_set, train_years, validation_seasons, train_end, test_season, tie_margin, log) -> dict:
    """Experiment B: window lengths within one label set, validation and final season."""
    validation = select_training_window(
        label_set.frame,
        train_years=train_years,
        validation_seasons=validation_seasons,
        test_season=test_season,
        name=label_set.name,
        tie_margin=tie_margin,
        log=log,
    )
    final = []
    for years in sorted(set(train_years)):
        seasons = window(years, train_end)
        if not _complete([label_set], seasons):
            final.append({"train_years": years, "train_seasons": seasons, "status": "insufficient_history"})
            continue
        preds = _fit_predict(label_set.frame, seasons, test_season, label_set.name, log)
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
    preds = {s.name: _fit_predict(s.frame, seasons, test_season, s.name, log) for s in (candidate, other)}
    keys = matched_keys(list(preds.values()), test_season)
    scores = _head_to_head(preds, {s.name: s.frame for s in (candidate, other)}, keys)
    return {
        "status": "informational",
        "gate_eligible": False,
        "versions": {candidate.name: candidate.model_version, other.name: other.model_version},
        "train_seasons": seasons,
        "matched_test_rows": len(keys),
        "scores": scores,
        "skill_gain": {t: _gain(s[other.name], s[candidate.name]) for t, s in scores.items()},
        "mapping": "relative change applied to each target's start value; the two scales differ, so read as informational",
        "missing_cells": [
            (
                f"{other.model_version} labels without the candidate's added trades were not supplied, so the "
                "model-version effect without them is unresolved"
            )
        ],
    }


def _gain(candidate: dict, control: dict) -> float | None:
    a, b = candidate["overall"]["skill"], control["overall"]["skill"]
    return None if a is None or b is None else round(a - b, 6)


def _input(path: Path | str, model_version: str) -> dict:
    raw = pd.read_csv(path, usecols=["season"])
    return {
        "file": Path(path).name,
        "sha256": file_sha256(path),
        "rows": len(raw),
        "rows_by_season": coverage_counts(raw),
        "model_version": model_version,
    }


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
    baseline_model_version: str = DEFAULT_MODEL_VERSION,
    candidate_model_version: str = DEFAULT_MODEL_VERSION,
    model_effect_model_version: str = "baseline-hl28-v3",
) -> dict:
    """Load the versioned CSVs and return a deterministic schema-2 report.

    ``train_starts`` keeps the schema-1 CLI meaning: each start becomes a
    window length ending at ``train_end`` (2022 with train_end 2024 is 3).
    """
    stats = load_season_stats(training_data_path)
    specs = [("baseline", baseline_path, baseline_model_version), ("candidate", candidate_path, candidate_model_version)]
    if model_effect_path:
        specs.append(("model_effect", model_effect_path, model_effect_model_version))
    sets = {
        name: LabelSet(name, build_dataset(stats, CsvLabels(path, name=name, min_trades=min_trades)), version)
        for name, path, version in specs
    }
    report = compare_datasets(
        sets["baseline"],
        sets["candidate"],
        model_effect=sets.get("model_effect"),
        train_years=tuple(train_end - start + 1 for start in train_starts),
        validation_seasons=validation_seasons,
        train_end=train_end,
        test_season=test_season,
        min_trades=min_trades,
    )
    report["inputs"] = {name: _input(path, version) for name, path, version in specs}
    report["inputs"]["training_data"] = {"file": Path(training_data_path).name, "sha256": file_sha256(training_data_path)}
    return report


def write_report(report: dict, out_path: Path | str) -> None:
    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def gate_summary(report: dict) -> dict:
    """The decision block without the bulky experiment tables."""
    keep = ("selection", "frozen_candidate", "gate", "decision", "limitations")
    return deepcopy({k: report[k] for k in keep})
