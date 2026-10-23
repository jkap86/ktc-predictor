"""CLI.

    cd backend
    python -m predictor_v5 evaluate                 # season folds vs no-change
    python -m predictor_v5 train                    # fit on all seasons, save model
    python -m predictor_v5 predict --position WR --age 24 --start 5200 --points 260
    python -m predictor_v5 predict ... --sweep 100,150,200,250,300
    python -m predictor_v5 --labels backchannel:labels.csv --min-trades 5 holdout --train 2022-2024 --test 2025
    python -m predictor_v5 build-labels --values raw.csv --out labels.csv [--model-version baseline-hl28-v3]
    python -m predictor_v5 compare-labels           # schema-2 report: select on 2024, evaluate on 2025
    python -m predictor_v5 season --player "Ja'Marr Chase" --points 330
    python -m predictor_v5 season --player "Bijan Robinson" --sweep 200,275,350

``--labels csv:path/to/backchannel_reference_values.csv`` swaps the label
source (columns: player_id, season, start_value, end_value).
"""

import argparse
import json
import sys
from pathlib import Path

from predictor_v5.data import DEFAULT_ZIP, build_dataset, load_season_stats
from predictor_v5.evaluate import evaluate, holdout
from predictor_v5.labels import get_label_source
from predictor_v5.model import WhatIfModel

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
DATA = Path(__file__).resolve().parent / "data"


def _dataset(args):
    if args.labels == "ktc" and (args.min_trades or args.end_column != "end_value"):
        raise SystemExit("--min-trades and --end-column only apply to CSV labels, not --labels ktc")
    source = get_label_source(args.labels, min_trades=args.min_trades, end_column=args.end_column)
    return source, build_dataset(load_season_stats(), source)


def cmd_evaluate(args) -> None:
    source, df = _dataset(args)
    result = evaluate(df, source.name, source.scale_max)
    preds = result.pop("predictions")
    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / f"{source.name}_eval.json").write_text(json.dumps(result, indent=2) + "\n")
    if args.save_predictions:
        preds.to_csv(ARTIFACTS / f"{source.name}_eval_predictions.csv", index=False)
    print(json.dumps(result, indent=2))


def cmd_holdout(args) -> None:
    source, df = _dataset(args)
    first, _, last = args.train.partition("-")
    train_seasons = list(range(int(first), int(last or first) + 1))
    result = holdout(df, source.name, source.scale_max, train_seasons, args.test)
    result.pop("predictions")
    print(json.dumps(result, indent=2))


def cmd_train(args) -> None:
    source, df = _dataset(args)
    model = WhatIfModel(label_source=source.name, scale_max=source.scale_max).fit(df)
    path = Path(args.out or ARTIFACTS / f"{source.name}.joblib")
    model.save(path)
    print(f"Trained on {len(df)} player-seasons ({model.train_seasons}); saved {path}")


def cmd_predict(args) -> None:
    model = WhatIfModel.load(args.model or ARTIFACTS / f"{args.labels.split(':')[0]}.joblib")
    points = [float(p) for p in args.sweep.split(",")] if args.sweep else [args.points]
    for pts in points:
        value = model.predict_one(start_value=args.start, age=args.age, position=args.position, points=pts)
        print(
            f"{args.position} age {args.age:g}, start {args.start:g}, {pts:g} pts -> {value:.0f} ({value - args.start:+.0f})"
        )


def cmd_season(args) -> None:
    from predictor_v5.season import SOURCES, SeasonTool

    tool = SeasonTool(season=args.season, start_values_path=args.start_values)
    matches = tool.find(args.player)
    if len(matches) != 1:
        print(f"{len(matches)} players match {args.player!r}" + (":" if matches else "."))
        for m in matches:
            print(f"  {m['player_id']:>6}  {m['name']} ({m['position']}, {m['team'] or 'FA'})")
        return
    points = [float(p) for p in args.sweep.split(",")] if args.sweep else [args.points]
    out = tool.predict(matches[0]["player_id"], points, age=args.age)
    age = f"age {out['age']:g}" if out["age"] is not None else "age unknown"
    print(f"{out['name']} ({out['position']}, {out['team'] or 'FA'}, {age}), {out['season']} season")
    for name, cfg in SOURCES.items():
        src = out[name]
        if not src["predictions"]:
            print(f"  {cfg['label']}: {src['note']}")
            continue
        print(f"  {cfg['label']}: start {src['start_value']:.0f} on {src['start_date']}")
        for p in src["predictions"]:
            print(f"    {p['points']:g} pts -> {p['value']} ({p['change']:+d})")
        if src["note"]:
            print(f"    Note: {src['note']}")


def cmd_build_labels(args) -> None:
    from predictor_v5.label_export import build_labels_file, provenance_path

    expected = None
    if args.expected_seasons:
        first, _, last = args.expected_seasons.partition("-")
        expected = tuple(range(int(first), int(last or first) + 1))
    labels = build_labels_file(
        args.values, args.out, args.training_data, expected_seasons=expected, model_version=args.model_version
    )
    report = labels.attrs["provenance"]
    print(f"Wrote {len(labels)} player-seasons to {args.out} ({report['model_version'] or 'model version unverified'})")
    print(f"Coverage and provenance: {provenance_path(args.out)}")
    for warning in report["coverage"]["warnings"]:
        print(f"WARNING: {warning}", file=sys.stderr)


def cmd_compare_labels(args) -> None:
    from predictor_v5.compare_labels import (
        compare_label_files,
        gate_summary,
        write_report,
    )

    train_starts = tuple(int(value) for value in args.train_starts.split(","))
    validation_seasons = tuple(int(value) for value in args.validation_seasons.split(","))
    result = compare_label_files(
        args.baseline_values,
        args.candidate_values,
        training_data_path=args.training_data,
        min_trades=args.min_trades,
        train_starts=train_starts,
        train_end=args.train_end,
        test_season=args.test,
        validation_seasons=validation_seasons,
        model_effect_path=args.model_effect_values or None,
        baseline_model_version=args.baseline_model_version,
        candidate_model_version=args.candidate_model_version,
        model_effect_model_version=args.model_effect_model_version,
        holdout_declaration={"status": args.holdout_status, "evidence": args.holdout_evidence} if args.holdout_status else None,
    )
    write_report(result, args.out)
    print(json.dumps(gate_summary(result), indent=2, sort_keys=True))
    print(f"Wrote deterministic report to {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="predictor_v5")
    parser.add_argument("--labels", default="ktc", help="'ktc' or 'csv:<path>'")
    parser.add_argument("--min-trades", type=int, default=0, help="CSV labels: min trades at start and end")
    parser.add_argument("--end-column", default="end_value", help="CSV labels: column to use as end value")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ev = sub.add_parser("evaluate")
    ev.add_argument("--save-predictions", action="store_true")
    ev.set_defaults(func=cmd_evaluate)

    ho = sub.add_parser("holdout")
    ho.add_argument("--train", required=True, help="season or range, e.g. 2022-2024")
    ho.add_argument("--test", type=int, required=True)
    ho.set_defaults(func=cmd_holdout)

    tr = sub.add_parser("train")
    tr.add_argument("--out")
    tr.set_defaults(func=cmd_train)

    pr = sub.add_parser("predict")
    pr.add_argument("--model")
    pr.add_argument("--position", required=True, choices=["QB", "RB", "WR", "TE"])
    pr.add_argument("--age", type=float, required=True)
    pr.add_argument("--start", type=float, required=True, help="start-of-season value")
    pr.add_argument("--points", type=float, help="expected season fantasy points")
    pr.add_argument("--sweep", help="comma-separated points values")
    pr.set_defaults(func=cmd_predict)

    se = sub.add_parser("season", help="pre-season value + projected points -> end-of-season value")
    se.add_argument("--player", required=True, help="name, part of a name, or Sleeper id")
    se.add_argument("--points", type=float, help="projected season fantasy points (full PPR)")
    se.add_argument("--sweep", help="comma-separated points values")
    se.add_argument("--season", type=int, default=2026)
    se.add_argument("--age", type=float, help="override the age on file")
    se.add_argument("--start-values", help="start values CSV (default data/start_values_<season>.csv)")
    se.set_defaults(func=cmd_season)

    bl = sub.add_parser("build-labels", help="join a raw market-history export to training stats")
    bl.add_argument("--values", required=True, help="raw CSV produced by sql/backchannel_labels.sql")
    bl.add_argument("--out", required=True, help="destination for the 14-column label CSV")
    bl.add_argument("--training-data", default=str(DEFAULT_ZIP))
    bl.add_argument("--model-version", help="fail unless the export is this market model version")
    bl.add_argument(
        "--expected-seasons", default="2019-2025", help="fail if any of these seasons is absent; '' to skip"
    )
    bl.set_defaults(func=cmd_build_labels)

    cl = sub.add_parser(
        "compare-labels",
        help="select a candidate window on pre-test seasons, then compare it with the control on the test season",
    )
    cl.add_argument("--baseline-values", default=str(DATA / "backchannel_labels_2019_2025.csv"))
    cl.add_argument("--candidate-values", default=str(DATA / "backchannel_labels_2019_2025_mfl.csv"))
    cl.add_argument(
        "--model-effect-values",
        default=str(DATA / "backchannel_labels_2019_2025_mfl_v3.csv"),
        help="candidate's trade corpus under another market model version (informational); '' to skip",
    )
    # Assertions only: each file's version comes from its .provenance.json or data/label_manifest.json,
    # and a mismatch is an error.
    cl.add_argument("--baseline-model-version")
    cl.add_argument("--candidate-model-version")
    cl.add_argument("--model-effect-model-version")
    cl.add_argument(
        "--holdout-status",
        choices=["reused", "untouched", "unknown"],
        help="test-season history when it isn't on record (2025 is on record as reused)",
    )
    cl.add_argument("--holdout-evidence", help="how the test season was held out; required with --holdout-status untouched")
    # Schema-2 reports go to a new file so the schema-1 artifacts stay as committed.
    cl.add_argument("--out", default=str(ARTIFACTS / "backchannel_label_experiments_v2.json"))
    cl.add_argument("--training-data", default=str(DEFAULT_ZIP))
    cl.add_argument("--min-trades", type=int, default=5)
    cl.add_argument("--train-starts", default="2022,2021,2020", help="candidate window starts, read as lengths ending at --train-end")
    cl.add_argument("--train-end", type=int, default=2024)
    cl.add_argument("--validation-seasons", default="2024", help="seasons before --test used to choose the window")
    cl.add_argument("--test", type=int, default=2025)
    cl.set_defaults(func=cmd_compare_labels)

    args = parser.parse_args()
    if args.cmd in ("predict", "season") and args.points is None and not args.sweep:
        parser.error(f"{args.cmd} needs --points or --sweep")
    args.func(args)


if __name__ == "__main__":
    main()
