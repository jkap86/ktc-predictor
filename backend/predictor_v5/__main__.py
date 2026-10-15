"""CLI.

    cd backend
    python -m predictor_v5 evaluate                 # season folds vs no-change
    python -m predictor_v5 train                    # fit on all seasons, save model
    python -m predictor_v5 predict --position WR --age 24 --start 5200 --points 260
    python -m predictor_v5 predict ... --sweep 100,150,200,250,300
    python -m predictor_v5 --labels backchannel:labels.csv --min-trades 5 holdout --train 2022-2024 --test 2025

``--labels csv:path/to/backchannel_reference_values.csv`` swaps the label
source (columns: player_id, season, start_value, end_value).
"""

import argparse
import json
from pathlib import Path

from predictor_v5.data import build_dataset, load_season_stats
from predictor_v5.evaluate import evaluate, holdout
from predictor_v5.labels import get_label_source
from predictor_v5.model import WhatIfModel

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"


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

    args = parser.parse_args()
    if args.cmd == "predict" and args.points is None and not args.sweep:
        parser.error("predict needs --points or --sweep")
    args.func(args)


if __name__ == "__main__":
    main()
