# 2026 MFL-versus-KTC Replacement Evaluation Design

## Intent and why

KTC remains the incumbent season-value source. The refreshed MFL-inclusive
Backchannel v1 model is a replacement candidate, not a presumed upgrade. The
final decision asks, on the same 2026 players and season boundaries, whether
MFL predicts its own market's future values at least as well as KTC predicts
KTC's future values.

The MFL refresh's 2025 results are development evidence only. 2026 is the
untouched final decision. MFL's present advantage is coverage/provenance, not
an extra trainable year: usable stats begin in 2020 (the archive has no usable
2019 stats), so both models use the same fixed 2021--2025 train window.

## Goals, non-goals, and anti-goals

Goals:

- Keep KTC primary unless MFL wins or is similar on the liquid matched cohort
  under the precommitted 2026 gate.
- Compare each model on its own future values and own no-change baseline, but
  on one identical matched player-season cohort and identical cutoffs.
- Fail closed before target data is read when an identity, provenance, or
  untouched-evidence requirement is not met.
- Emit one deterministic, atomic, hash-addressed non-promotional report.

Non-goals:

- No runtime/API/UI, season.py, model artifact, start-value, or deployment
  change. Promotion is a later manual, reviewed change.
- No model/window/threshold selection from 2026 data.
- No conversion to a common value scale or use of one source's outcomes as the
  other's ground truth.

Anti-goals:

- Do not revise the sealed confirm-2026 protocol. Its manifest, registry,
  target recipe, command, old +2-point gate, and Backchannel-only output retain
  their existing semantics.
- Do not allow a favorable result to automatically replace KTC or authorize a
  deployment.
- Do not allow rank metrics, walk-forward, bootstrap, or coverage diagnostics
  to decide promotion.

## Constraints and fixed comparison

| Item | Decision |
| --- | --- |
| Incumbent/candidate | KTC is incumbent; verified backchannel_labels_2019_2025_mfl.csv on baseline-hl28-v1 is candidate. |
| Training | Both use the existing model contract and exactly 2021--2025. KTC is read from the frozen training archive; MFL uses the verified refreshed label CSV. |
| Cohort | Liquid matched players only: sorted intersection of valid frozen starts, KTC target, MFL target, both predictions and both source filters, with identical verified 2026 stats/position facts. Results do not establish performance for excluded players. |
| Dates | The committed 2026-09-09 start cutoff and 2027-01-10 end cutoff in America/New_York, matching the sealed MFL endpoint. Actual source observation dates are recorded. |
| MFL filter | Positive start/end values and at least five start and end trades. |
| KTC filter | Positive, finite, non-sentinel start/end values: 0 < value < 9999; no synthetic imputation. |
| Scale | KTC model scale_max is 9999; MFL/Backchannel model scale_max is 10000. |
| Outcome | On pass, recommend MFL primary only for eligible players, retaining KTC as fallback for missing/insufficient MFL evidence; otherwise retain KTC primary. promotion_authorized is always false. |

## Approaches considered

1. **Extend sealed confirm-2026 with KTC.** Rejected: its strict manifest
   source identities, one refreshed-Backchannel target, and semantics are
   already sealed. Altering them would invalidate or silently redefine it.
2. **Score both models against one Backchannel target.** Rejected: it is
   source-confounded and cannot justify replacing a KTC-facing model.
3. **Create an independent sealed replacement protocol with two targets and a
   matched cohort.** Chosen: it preserves the existing confirmation and
   measures each source on the market it would represent.

## Architecture and data flow

The implementation adds an independent ktc-replacement-2026 protocol as a
standalone module. It must not change the sealed confirmation's hashed
sources, including predictor_v5/__main__.py, treat its manifest as mutable
configuration, or write to its output path. Pure helper reuse is allowed only
when the new manifest freezes the helper identities.

~~~
frozen KTC 2021--25 labels + stats  -> KTC fit
frozen MFL 2021--25 labels + stats  -> MFL fit
                                     |
registered KTC 2026 target + stats --+--> KTC normalized score
registered MFL 2026 target + stats --+--> MFL normalized score
committed 2026 start values ---------+--> both source starts and liquidity
                                     |
valid-key intersection --------------+--> exact gate + diagnostics
                                     |
                       atomic replacement report (recommend/retain)
~~~

For each matched key and source s, apply the fitted model's relative change
only to that source's frozen start value:

~~~
pred_end_s = frozen_start_s * predicted_end_own_s / fitted_start_own_s
~~~

KTC predictions are scored against KTC end values and KTC no-change error;
MFL predictions are scored against MFL end values and MFL no-change error.
The frozen start CSV provides ktc_value and bc_value; bc_value is the
Backchannel v1 start value fed to the MFL-trained model, not a separate MFL
column. It also provides bc_trades, bc_date, bc_model_version, and ktc_date.
Players absent from the frozen start key set are excluded by the cohort
intersection and counted in coverage diagnostics. If a key exists in both the
frozen start file and a normalized target but its start values conflict, that
is an exit-2 integrity refusal, not an exclusion or substitution. The target
loader must one-to-one join each source to separately registered
2026 stats, verifying player id, position, age, games, and points. It rejects
duplicate keys, conflicted stats, non-finite values, future/training leakage,
and values outside each source contract. Inputs and rows are sorted by
(season, player_id) before fitting, matching, resampling, or serialization.

## Sealed-confirmation compatibility and lifecycle

The already sealed model-2026-confirmation remains a Backchannel-only
baseline-versus-refreshed-MFL confirmation. The new protocol has its own
manifest hash, registry key, registration, command, and output path. It may
reuse the MFL target bytes only by independently registering and validating
their CSV, stats, sidecar, recipe hashes, cutoff metadata, and attestation.
It must never infer target validity from an adjacent sidecar or old report.

Before either protocol opens 2026 outcomes, an independent joint review must
attest that both identified, already sealed manifest hashes and their fixed
recipes were committed before outcomes influenced either protocol, and that
the evidence remains untouched. A separate immutable joint review artifact
identifies both hashes, reviewer, evidence, and timestamp. The existing
confirmation's post-season registry and input registration refer to it via
their existing evidence_reference fields; the replacement registry and input
registration reference and hash the same artifact. No sealed artifact or
strict existing schema is edited. If either protocol already consumed or
exposed outcome evidence without that joint review, this comparison cannot
claim an untouched holdout. This is a human attestation, not an automatic
guarantee against observation outside the software.

The replacement registry's reviewed lifecycle is:

~~~
sealed -> untouched -> used
sealed -> reused
~~~

Untouched is review/Git evidence, not proof of non-observation. Missing,
duplicate, sealed, reused, used, noncanonical, or mismatched entries refuse
before either target is read. The new protocol's use status does not rewrite
the sealed command's registry or output. The one-shot claim below enforces
consumption independently of the reviewed registry update.

## Artifacts and registration

All protocol JSON is canonical UTF-8 JSON (sorted keys, compact separators,
trailing newline, no duplicate/unknown/missing fields). Paths are nonempty
repository-relative POSIX paths without traversal/backslashes; identities are
lowercase SHA-256.

Suggested independent paths:

~~~
backend/predictor_v5/data/ktc_replacement_2026_manifest.json
backend/predictor_v5/data/ktc_replacement_2026_holdout_registry.json
backend/predictor_v5/data/ktc_replacement_2026_inputs.json
backend/predictor_v5/artifacts/replacements/2026-<manifest-sha256>.json
~~~

### Pre-outcome manifest

Freeze and commit, before outcomes are reviewed:

- protocol/sealing identity and hashes for evaluator, label loaders,
  KTC/MFL target loaders, stats reader, model contract, and report writer;
- KTC archive member and MFL label identities, shared training archive, exact
  2021--2025 seasons, positions, features, parameters, source-specific scales,
  and filters;
- the frozen start CSV SHA-256 and Git identity described below;
- the 2026-09-09 start and 2027-01-10 America/New_York end cutoffs and the
  fixed source observation selection rules;
- source-specific target recipes; MFL's SQL/build/provenance contract; and
  the thelab KTC history/value-table join specified below, exact SQL/extractor
  hashes, Sleeper mapping code and data identities, raw fields, valid-value
  rules, and snapshot semantics;
- exact gate, rank-diagnostic definitions, deterministic bootstrap settings,
  and output template; and
- registry and input-registration paths.

It cannot contain future target hashes; post-season registration identifies
those bytes.

### Frozen 2026 starts and provenance limitation

Use the committed content-addressed
backend/predictor_v5/data/start_values_2026.csv: SHA-256
6366b85300c4f9532f0866c602c6d262baaccf6c3a84b9252a6d2e8d46c541bb,
Git blob ead24b508102ca2a1071d07f3f665e9ea920aa96, introduced by commit
c415177. Its documented cutoff is 2026-09-09. The existing start export
selects Backchannel v1 at or before cutoff and each KTC observation at or
before cutoff within 14 days. Record actual bc_date and ktc_date and enforce
that documented policy, not equality of observation dates. The file and Git
identities freeze its contents, but do not independently prove its underlying
raw export, KTC mapping, or historical source provenance. This known limit
must be reported, not upgraded into verified raw provenance.

### Required 2026 KTC outcome/target registration

There is no existing sealed KTC 2026 label artifact analogous to MFL's
Backchannel export. Register a dedicated KTC target bundle containing:

1. immutable raw **end** export of all candidate observations from the local
   thelab history/value-table join specified below, in the inclusive
   14-calendar-day window
   2026-12-28 through 2027-01-10, keyed by mapped Sleeper player_id, with
   source record identity, observation date, value, and independently
   reviewable SQL/extractor/mapping provenance;
2. immutable normalized target CSV with exact ordered columns and types:
   player_id (nonempty string), season (integer 2026), start_value and
   end_value (finite numeric), position (QB/RB/TE/WR string), age and points
   (finite numeric), and games (nonnegative integer). Values must meet the
   KTC filter; start_value must equal the frozen CSV's ktc_value where the
   frozen start key exists;
3. 2026-only stats archive in the established training-players ZIP format,
   with exactly one JSON member named training-data.json containing the
   players envelope and 2026 seasons only;
4. canonical provenance sidecar identifying raw end export, normalized CSV,
   frozen start CSV, stats,
   local source, SQL/extractor and mapping code hashes, observation dates,
   coverage/exclusions,
   reviewer, and all relevant hashes; and
5. path/SHA-256 identities for all registered files.

The local thelab extraction contract matches sql/start_values.sql:
ktc_value_history h JOIN ktc_values k USING (format, ktc_id), with
h.format = 'dynasty', k.sleeper_id IS NOT NULL, and h.sf_value > 0. Map
k.sleeper_id to player_id, use h.sf_value as value and h.date as observation
date, and export every candidate in the fixed inclusive window. The frozen
actual SQL/extractor and Sleeper mapping code hashes and mapping identities
are bound in the pre-outcome manifest; the post-season registration binds
those same identities and the raw export SHA-256. The exporter/evaluator
rejects conflicting values for any same-player, same-date observations; it
must not rely on nondeterministic DISTINCT ON tie behavior. For each player,
select the latest observation date at or before the 2027-01-10 cutoff within
that window and record that date. The KTC loader verifies normalized starts
for keys present in the frozen start file, each end and date against the raw
end bundle, and stat fields against the registered archive. The specified
local thelab tables are the only provider: no live network lookup or
production database access is allowed during evaluation.
The raw end export and its provenance are mandatory; the start-file exception
does not extend to the future outcome.

The same registration separately identifies the MFL target CSV, registered
sidecar, 2026 stats archive, and frozen SQL/build identities. If source stats
archives are not byte-identical, every matched common stat fact must be equal
or evaluation refuses. For shared start keys, MFL starts must equal frozen
bc_value with verified baseline-hl28-v1 and the frozen start-trade count;
registered MFL end values
and end-trade counts come from its reviewed target recipe. Registry and
registration must have matching reviewer,
evidence reference/commit, review timestamp, and transition source.

## Metrics, exact gate, and tie behavior

For each source s on the identical matched cohort:

~~~
MAE_s      = mean(abs(pred_end_s - actual_end_s))
baseline_s = mean(abs(start_s - actual_end_s))
skill_s    = 1 - MAE_s / baseline_s
delta      = skill_mfl - skill_ktc
delta_p    = skill_mfl,p - skill_ktc,p
~~~

Use full-precision finite values, never display rounding. A zero/non-finite
no-change MAE, missing skill/position, empty cohort, or invalid count is an
evaluated failure that writes retain_ktc after successful preflight.

The gate passes only when all conditions are true:

1. delta >= -0.01 overall. Exact equality passes: a one-point MFL loss is
   equivalent/similar.
2. Each QB/RB/TE/WR matched cohort has at least 30 rows.
3. Every delta_p >= -0.05. Exact equality passes.

Thus MFL wins pass subject to guards, and MFL results within the one-point band
pass. A -0.010000001 overall delta or -0.050000001 positional delta fails.
Pass emits recommend_replacement meaning MFL primary for eligible players
only, with KTC fallback for missing or insufficient MFL evidence. It is not a
recommendation to remove KTC or a claim about the excluded population.
Failure emits retain_ktc. Both carry stable machine-readable reasons and never
authorize promotion. A later reviewed manual promotion defines the runtime
eligibility implementation; removal of fallback is deferred until
comprehensive MFL coverage is demonstrated in separate work.

Non-gating diagnostics include MAE/median/p90, bias/residual summaries,
coverage funnels, paired errors where meaningful, deterministic bootstrap
intervals, and rank metrics. Rank diagnostics report Spearman predicted versus
actual end-value rank correlation and mean absolute rank error overall/by
position. Rank each source's predicted and actual values independently in the
same cohort, descending, using average rank for ties; report mean absolute
difference in raw rank positions (not percentiles). Undefined correlations
are null, never a gate failure. Every diagnostic is tagged
used_for_configuration: false and used_for_promotion: false; none may appear
in gate reasons.

## CLI, report, and preflight/failure semantics

Proposed command:

~~~
python -m predictor_v5.ktc_replacement_2026
~~~

It accepts no season, path, threshold, model, data, or output override. Tests
may inject dependencies; the CLI may not. Before any outcome bytes are opened,
preflight validates the manifest, source and training identities, training
coverage, registry, both registrations and joint attestation, validates target
path/identity metadata, and rejects an existing output or consumption claim.
Only then may it durably claim the one-shot. After the claim it hashes and
validates outcome bytes and their provenance, loads targets, and fits only
after all input checks pass. Hash verification necessarily reads bytes; it is
not performed before the untouched attestation and claim.

The canonical wrapper is {schema_version, payload, payload_sha256}, hashing
canonical payload bytes only. Payload includes protocol/input/registry hashes;
source, archive, model, target, raw, sidecar, mapping, and code identities;
cutoffs; filters; fits; exact cohort; metrics; diagnostics; gate; and decision.
It writes through temporary file, fsync, atomic no-replace creation, directory
fsync, and cleanup on error.

### One-shot consumption and recovery

Atomically create a unique, manifest-keyed claim file without replacement and
fsync it and its directory *before* opening outcome bytes. Concurrent runs
cannot both claim. The claim records manifest/registration/joint-review hashes
and remains durable even if target validation, fitting, reporting, or the
process crashes. It irrevocably consumes this protocol's opportunity: a
retry, including after an exit 1/2 or absent report, refuses. The reviewed
registry is subsequently transitioned to used, with claim and, when present,
report identity; the claim is authoritative while that manual review is
pending. A successful no-replace report also independently blocks reruns.
Do not delete a claim or regenerate a result to recover. If a claim exists
without a valid final report, preserve it and all evidence, independently review
and record the failure, and require a separately approved future holdout and
new protocol for a decision. A failure before the durable claim, while no
outcome bytes were opened, may be corrected and preflight retried after review.
The report and registry cannot be atomically committed together; claim-first
ordering closes the crash window without changing sealed confirmation state.

| Condition | Behavior |
| --- | --- |
| Before claim: manifest/registry/registration/identity metadata/attestation/one-shot failure, training leakage or missing required training season/position | stderr, exit 2; no outcome bytes read, fit, report, or claim |
| After claim: outcome hash/provenance/cutoff/schema failure, duplicate/misaligned keys, invalid snapshot, stat disagreement, or input integrity failure | stderr, exit 2; no fit or report; claim remains consumed |
| Valid inputs but insufficient/empty matched scoring cohort, a position below 30, missing skill, or zero no-change baseline | atomic retain_ktc report, exit 0; claim consumed |
| Final report already exists | stderr, exit 2; unchanged |
| Unexpected fit/IO/serialization/fsync failure after claim | stderr, exit 1; no partial report; claim remains consumed |

## Validation and tests

Implementation must test:

- strict JSON/schema/path/hash validation, all registry transitions, joint
  attestation cross-references, and proof outcome bytes/target loaders/fits
  cannot run before attestation and durable claim;
- frozen start SHA/Git identity, dates, Backchannel v1 fields, exclusion and
  coverage for absent start keys versus exit-2 refusal for conflicting values
  on shared keys, known provenance limitation, and KTC raw-end provenance;
- KTC CSV column order/types, exact ZIP member, inclusive window endpoints,
  latest-date selection, consistent and conflicting same-player same-date
  duplicates anywhere in the window, actual observation date, exact source
  join/filters/value field and frozen code/raw/mapping identities,
  sentinel/range, player mapping, and stat join checks; verify no network or
  database access during evaluation;
- MFL registered sidecar/recipe/stats checks, independent validation even when
  bytes are shared with confirmation, and divergent-common-stats refusal;
- duplicate/non-finite/filter/leakage/coverage/sorting failures;
- source-normalized scoring, exact common cohort, and prevention of accidental
  cross-source target/no-change use;
- full-precision -0.01/just-below, -0.05/just-below, 30/29, ties, zero
  baseline, and missing-skill boundaries;
- deterministic average-rank ties/raw rank-position error and proof diagnostics
  cannot affect decisions; bootstrap, payload digest, concurrent claims,
  crash boundaries, failure recovery, atomic no-overwrite/rollback, CLI exits;
  and
  regression assertions that sealed confirmation, season.py, API, frontend,
  and deployment state stay unchanged.

## Operational sequence

1. Review this design, implement the separate evaluator/tests, and leave sealed
   confirmation artifacts untouched.
2. Before viewing outcomes, commit the new manifest and sealed registry entry,
   including the frozen start identity and KTC end extraction/cutoff contract.
3. After season end, produce source target bundles and 2026 stats using frozen
   recipes; independently review provenance, cutoffs, coverage, and common
   stats.
4. Before either protocol reads outcome bytes, review a joint attestation and
   cross-reference it in both protocols' post-season review fields; register
   exact identities and set the replacement registry to untouched.
5. Run once; the durable claim consumes the opportunity before outcome reads.
   Archive any immutable report and review the registry transition to used,
   retaining claim and evidence even if the run failed.
6. Only if it recommends replacement, open a separate manual promotion change
   to make MFL primary for eligible players with KTC fallback after normal
   review. Fallback removal requires separate comprehensive-coverage evidence.

## Implementation risks

- The local raw export must include every candidate observation in the fixed
  window, not just a preselected latest row; omissions or unstable player
  mapping can bias the matched cohort and require independent review.
- SQL/extractor/mapping code and the raw export must match their bound hashes;
  drift or conflicting same-date observations fail closed, not silently repair.
- The frozen start CSV lacks independently verifiable raw-export provenance;
  its content-addressed identity is the explicitly accepted limitation.
