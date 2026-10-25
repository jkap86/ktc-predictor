"""Source-specific replacement scores, exact gate, and non-gating diagnostics."""

import math
import numbers

import numpy as np
import pandas as pd

from predictor_v5.replacement_cohort import COHORT_COLUMNS, POSITIONS


def _validate(frame):
    if not isinstance(frame, pd.DataFrame) or list(frame.columns) != COHORT_COLUMNS:
        raise ValueError("invalid cohort schema")
    if frame.duplicated(["season", "player_id"]).any():
        raise ValueError("duplicate cohort key")
    if not frame.player_id.map(lambda v: isinstance(v, str) and bool(v.strip())).all():
        raise ValueError("invalid player_id")
    if not frame.season.map(lambda v: isinstance(v, numbers.Integral) and not isinstance(v, (bool, np.bool_)) and v == 2026).all():
        raise ValueError("invalid season")
    if not frame.position.isin(POSITIONS).all():
        raise ValueError("invalid position")
    for col in COHORT_COLUMNS[3:]:
        if not frame[col].map(lambda v: isinstance(v, numbers.Real) and not isinstance(v, (bool, np.bool_)) and math.isfinite(v)).all():
            raise ValueError(f"invalid {col}")


def _safe(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _source(frame, source):
    if frame.empty:
        return {"mae": None, "baseline": None, "skill": None}
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        actual = frame[f"{source}_actual"].to_numpy(dtype=float)
        mae = _safe(np.mean(np.abs(frame[f"{source}_pred"].to_numpy(dtype=float) - actual)))
        baseline = _safe(np.mean(np.abs(frame[f"{source}_start"].to_numpy(dtype=float) - actual)))
        skill = _safe(1 - mae / baseline) if mae is not None and baseline not in (None, 0) else None
    return {"mae": mae, "baseline": baseline, "skill": skill}


def _group(frame):
    ktc, mfl = _source(frame, "ktc"), _source(frame, "mfl")
    delta = None if ktc["skill"] is None or mfl["skill"] is None else _safe(mfl["skill"] - ktc["skill"])
    return {"count": len(frame), "ktc": ktc, "mfl": mfl, "delta": delta}


def score(cohort):
    _validate(cohort)
    return {"overall": _group(cohort), "positions": {p: _group(cohort[cohort.position == p]) for p in POSITIONS}}


def _number(value):
    if not isinstance(value, numbers.Real) or isinstance(value, (bool, np.bool_)):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError, ValueError):
        return False


def _consistent(left, right):
    """Allow at most 1e-15 absolute rounding noise, with no relative tolerance."""
    return _number(left) and _number(right) and math.isclose(left, right, rel_tol=0, abs_tol=1e-15)


def gate(metrics):
    reasons = []
    if not isinstance(metrics, dict) or set(metrics) != {"overall", "positions"}:
        reasons.append("invalid_metrics_shape")
    overall = metrics.get("overall", {}) if isinstance(metrics, dict) else {}
    positions = metrics.get("positions", {}) if isinstance(metrics, dict) else {}
    if not isinstance(overall, dict):
        overall = {}
    if not isinstance(positions, dict):
        positions = {}
    if set(positions) != set(POSITIONS):
        reasons.append("invalid_positions_shape")
    def validate_group(group, label):
        if set(group) != {"count", "ktc", "mfl", "delta"}:
            reasons.append(f"{label}_invalid_shape")
        skills = {}
        for source in ("ktc", "mfl"):
            item = group.get(source)
            if not isinstance(item, dict) or set(item) != {"mae", "baseline", "skill"}:
                reasons.append(f"{label}_{source}_invalid_shape")
                continue
            mae, baseline, skill = item["mae"], item["baseline"], item["skill"]
            if not _number(mae) or mae < 0:
                reasons.append(f"{label}_{source}_invalid_mae")
            elif not _number(baseline) or baseline <= 0:
                reasons.append(f"{label}_{source}_invalid_baseline")
            elif not _number(skill):
                reasons.append(f"{label}_{source}_missing_skill")
            else:
                try:
                    expected = 1 - mae / baseline
                except (OverflowError, ZeroDivisionError):
                    expected = float("nan")
                if not _consistent(skill, expected):
                    reasons.append(f"{label}_{source}_inconsistent_skill")
                else:
                    skills[source] = skill
        delta = group.get("delta")
        if not _number(delta):
            reasons.append(f"{label}_missing_delta")
            return None
        if len(skills) == 2:
            computed = skills["mfl"] - skills["ktc"]
            if not _consistent(delta, computed):
                reasons.append(f"{label}_inconsistent_delta")
                return None
            return computed
        return None
    count = overall.get("count")
    if not isinstance(count, numbers.Integral) or isinstance(count, (bool, np.bool_)) or count <= 0:
        reasons.append("invalid_overall_count")
    delta = validate_group(overall, "overall")
    if delta is not None and delta < -0.01:
        reasons.append("overall_delta_below_threshold")
    position_counts = []
    for position in POSITIONS:
        group = positions.get(position)
        if not isinstance(group, dict):
            reasons.append(f"{position.lower()}_missing")
            continue
        count = group.get("count")
        delta = validate_group(group, position.lower())
        if not isinstance(count, numbers.Integral) or isinstance(count, (bool, np.bool_)) or count < 30:
            reasons.append(f"{position.lower()}_insufficient_count")
        if isinstance(count, numbers.Integral) and not isinstance(count, (bool, np.bool_)) and count >= 0:
            position_counts.append(count)
        if delta is not None and delta < -0.05:
            reasons.append(f"{position.lower()}_delta_below_threshold")
    if len(position_counts) == len(POSITIONS) and isinstance(overall.get("count"), numbers.Integral) and sum(position_counts) != overall["count"]:
        reasons.append("count_mismatch")
    passed = not reasons
    return {"decision": "recommend_replacement" if passed else "retain_ktc", "reasons": reasons,
            "primary": "mfl_eligible_only" if passed else "ktc", "fallback": "ktc" if passed else None,
            "promotion_authorized": False}


def _summary(values):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        return {"mean": None, "median": None, "p90": None, "min": None, "max": None}
    with np.errstate(over="ignore", invalid="ignore"):
        return {"mean": _safe(np.mean(values)), "median": _safe(np.median(values)),
                "p90": _safe(np.percentile(values, 90)), "min": _safe(np.min(values)), "max": _safe(np.max(values))}


def _diagnostic_group(frame):
    result = {"count": len(frame)}
    errors = {}
    for source in ("ktc", "mfl"):
        with np.errstate(over="ignore", invalid="ignore"):
            residual = frame[f"{source}_pred"].to_numpy(dtype=float) - frame[f"{source}_actual"].to_numpy(dtype=float)
        absolute = np.abs(residual)
        errors[source] = absolute
        summary = _summary(absolute)
        residual_summary = _summary(residual)
        pred_rank = frame[f"{source}_pred"].rank(ascending=False, method="average")
        actual_rank = frame[f"{source}_actual"].rank(ascending=False, method="average")
        rank_error = _summary(np.abs(pred_rank.to_numpy() - actual_rank.to_numpy()))["mean"]
        correlation = None
        if len(frame) > 1 and pred_rank.nunique() > 1 and actual_rank.nunique() > 1:
            correlation = _safe(np.corrcoef(pred_rank, actual_rank)[0, 1])
        result[source] = {"mae": summary["mean"], "median_absolute_error": summary["median"],
                          "p90_absolute_error": summary["p90"], "bias": residual_summary["mean"],
                          "residual": residual_summary, "rank": {"spearman": correlation, "mean_absolute_rank_error": rank_error}}
    with np.errstate(over="ignore", invalid="ignore"):
        result["paired_absolute_error_difference_mfl_minus_ktc"] = _summary(errors["mfl"] - errors["ktc"])
    return result


def diagnostics(cohort, seed=20261024, resamples=10000):
    _validate(cohort)
    if not isinstance(seed, numbers.Integral) or isinstance(seed, (bool, np.bool_)) or seed < 0:
        raise ValueError("invalid seed")
    if not isinstance(resamples, numbers.Integral) or isinstance(resamples, (bool, np.bool_)) or resamples <= 0:
        raise ValueError("invalid resamples")
    frame = cohort.sort_values(["season", "player_id"], kind="stable")
    rng = np.random.default_rng(int(seed))
    intervals = {"mae_difference_mfl_minus_ktc": None}
    if len(frame):
        with np.errstate(over="ignore", invalid="ignore"):
            ktc = np.abs(frame.ktc_pred.to_numpy(dtype=float) - frame.ktc_actual.to_numpy(dtype=float))
            mfl = np.abs(frame.mfl_pred.to_numpy(dtype=float) - frame.mfl_actual.to_numpy(dtype=float))
            differences = mfl - ktc
            samples = np.array([np.mean(differences[rng.integers(0, len(frame), len(frame))]) for _ in range(resamples)])
            if np.isfinite(samples).all():
                intervals["mae_difference_mfl_minus_ktc"] = [_safe(v) for v in np.percentile(samples, [2.5, 97.5])]
    return {"used_for_configuration": False, "used_for_promotion": False,
            "coverage": {"matched_count": len(frame), "position_counts": {p: int((frame.position == p).sum()) for p in POSITIONS}},
            "overall": _diagnostic_group(frame), "positions": {p: _diagnostic_group(frame[frame.position == p]) for p in POSITIONS},
            "bootstrap": {"seed": int(seed), "resamples": int(resamples), "percentiles": [2.5, 97.5], "intervals": intervals}}
