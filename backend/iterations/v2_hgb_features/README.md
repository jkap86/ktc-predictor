# v2_hgb_features — research only

**Status:** Offline-only. Not served through the model registry.

v2 trained a HistGradientBoosting model with 30 features per position (v1's 20
plus: `boom_rate`, `bust_rate`, `weekly_fp_cv`, `start_position_rank`,
`last_4_vs_season`, `has_last4_data`, `snap_pct`, `has_snap_data`,
`offseason_percentile`, `trend_14d`).

## Why research-only

Most of the 10 new features are contextual/historical — derived from weekly
granularity (boom/bust rate, weekly fantasy-points CV, snap percentage) or
offseason KTC movement. They are **not user-tweakable** via the What-If slider
(which only adjusts `games_played` and `ppg`) and are not currently plumbed
through `backend/app/services/eos_model_service.py` at inference time.

Promoting v2 to a served model would require either:

1. Sourcing the 10 new features per player from weekly season data at predict
   time (real plumbing work through `data_loader` → `eos_model_service` →
   iteration predict path); or
2. NaN-imputing them at inference, which silently degrades accuracy compared
   to the training-time metrics and defeats the point of the enhanced feature
   set.

Neither was worth the complexity for the v2 experiment's modest gains
(see `models/metrics.json` vs. `../v1_hgb_baseline/models/metrics.json`).

## What lives here

- `train.py` — training script. Monkey-patches `v1_hgb_baseline.train`'s
  feature lists and monotonic-constraint builder, then invokes v1's
  `train_all`. Re-runnable offline if you want to update artifacts.
- `io.py` — shim over `v1_hgb_baseline.io` for bundle save/load.
- `models/` — saved artifacts and `metrics.json` from the last training run.
- `manifest.research-only.json` — renamed from `manifest.json` so
  `ModelRegistry.discover()` (glob `*/manifest.json`) skips this directory.
  Rename back to `manifest.json` only if you also do the inference plumbing.

## Takeaways for future iterations

- v2's best relative gains were on RB (MAE −24) and WR (MAE −16), marginal on
  QB and TE.
- Future iterations should restrict features to things derivable at inference
  from the API's current inputs: `gp`, `ppg`, `start_ktc`, `age`,
  `draft_pick`, `years_remaining`, contract info, and the player's stored
  season history via `data_loader`.
