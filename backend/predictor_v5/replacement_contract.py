"""Strict, metadata-only contracts for the independent replacement protocol."""

import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def canonical_bytes(value: object) -> bytes:
    try:
        return (json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')
    except (TypeError, ValueError, UnicodeError) as error:
        raise ValueError('invalid canonical JSON') from error


def _fields(value, names):
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValueError('invalid object fields')
    return value


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def load_canonical(path: Path, required: set[str]) -> tuple[dict, str]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode('utf-8'), object_pairs_hook=_pairs)
        _fields(value, required)
        if canonical_bytes(value) != raw:
            raise ValueError('noncanonical JSON')
        return value, hashlib.sha256(raw).hexdigest()
    except (UnicodeError, json.JSONDecodeError, OSError) as error:
        raise ValueError('invalid JSON artifact') from error


def _string(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('expected nonblank string')


def _hex(value, length):
    if not isinstance(value, str) or re.fullmatch(f'[0-9a-f]{{{length}}}', value) is None:
        raise ValueError('invalid hexadecimal identity')


def _path(value, root=ROOT):
    _string(value)
    if value.startswith('/') or '\\' in value or any(part in ('', '.', '..') for part in value.split('/')):
        raise ValueError('invalid repository path')
    if not (root / value).resolve().is_relative_to(root.resolve()):
        raise ValueError('path escapes repository')


def _timestamp(value):
    _string(value)
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None or parsed.isoformat(timespec='seconds') != value:
            raise ValueError('invalid timestamp')
        return parsed
    except ValueError as error:
        raise ValueError('invalid timestamp') from error


def _equal(value, expected):
    # Equality alone conflates booleans and integers in Python.
    if type(value) is not type(expected):
        raise ValueError('invalid constant type')
    if isinstance(expected, dict):
        _fields(value, expected)
        for key in expected:
            _equal(value[key], expected[key])
    elif isinstance(expected, list):
        if len(value) != len(expected):
            raise ValueError('invalid list')
        for actual, wanted in zip(value, expected):
            _equal(actual, wanted)
    elif value != expected or isinstance(value, float) and not math.isfinite(value):
        raise ValueError('invalid constant')


def validate_identity(value: dict) -> None:
    _fields(value, {'path', 'sha256'})
    _path(value['path'])
    _hex(value['sha256'], 64)


TARGET = ['player_id', 'season', 'start_value', 'end_value', 'position', 'age', 'games', 'points']
FIXED = {
    'model': {'features': ['log_start', 'age', 'points', 'position_code'], 'monotonic': [0, 0, 1, 0], 'points_grid_step': 1, 'hgb_params': {'loss': 'absolute_error', 'learning_rate': 0.05, 'max_iter': 300, 'max_leaf_nodes': 15, 'min_samples_leaf': 20, 'l2_regularization': 1, 'random_state': 0}, 'ktc_scale_max': 9999, 'mfl_scale_max': 10000, 'mfl_model_version': 'baseline-hl28-v1'},
    'cutoffs': {'start': '2026-09-09', 'end': '2027-01-10', 'timezone': 'America/New_York', 'ktc_end_window_start': '2026-12-28', 'ktc_start_lookback_days': 14},
    'filters': {'ktc': {'minimum_exclusive': 0, 'maximum_exclusive': 9999}, 'mfl': {'minimum_exclusive': 0, 'minimum_start_trades': 5, 'minimum_end_trades': 5}},
    'gate': {'minimum_overall_delta': -0.01, 'minimum_position_delta': -0.05, 'minimum_position_count': 30},
    'diagnostics': {'bootstrap': {'seed': 20261024, 'resamples': 10000, 'percentiles': [2.5, 97.5]}, 'rank': {'order': 'descending', 'ties': 'average', 'correlation': 'spearman', 'error': 'mean_absolute_rank_error'}, 'used_for_configuration': False, 'used_for_promotion': False},
}


def validate_manifest(value: dict) -> None:
    _fields(value, {'schema_version', 'protocol', 'sealing_commit', 'sealed_at', 'registry_path', 'input_registration_path', 'output_template', 'claim_template', 'source_identities', 'training', 'starts', 'model', 'cutoffs', 'filters', 'recipes', 'gate', 'diagnostics'})
    _equal(value['schema_version'], 1)
    _equal(value['protocol'], 'ktc-replacement-2026')
    _hex(value['sealing_commit'], 40)
    _timestamp(value['sealed_at'])
    for key in ('registry_path', 'input_registration_path'):
        _path(value[key])
    for key, suffix in (('output_template', '.json'), ('claim_template', '.claim.json')):
        _equal(value[key], 'backend/predictor_v5/artifacts/replacements/2026-<manifest-sha256>' + suffix)
    for key, expected in FIXED.items():
        _equal(value[key], expected)
    sources = value['source_identities']
    if not isinstance(sources, list) or not sources:
        raise ValueError('source identities required')
    for item in sources:
        validate_identity(item)
    paths = [item['path'] for item in sources]
    if paths != sorted(set(paths)):
        raise ValueError('source identities must be unique and sorted')
    training = _fields(value['training'], {'archive', 'ktc_member', 'mfl_labels', 'seasons', 'positions'})
    for key in ('archive', 'mfl_labels'):
        validate_identity(training[key])
    _path(training['ktc_member'])
    _equal(training['seasons'], [2021, 2022, 2023, 2024, 2025])
    _equal(training['positions'], ['QB', 'RB', 'TE', 'WR'])
    _equal(value['starts'], {'identity': {'path': 'backend/predictor_v5/data/start_values_2026.csv', 'sha256': '6366b85300c4f9532f0866c602c6d262baaccf6c3a84b9252a6d2e8d46c541bb'}, 'git_blob': 'ead24b508102ca2a1071d07f3f665e9ea920aa96', 'introduced_commit': 'c415177', 'cutoff': '2026-09-09', 'provenance_limit': 'raw start provenance not independently verified'})
    recipes = _fields(value['recipes'], {'ktc', 'mfl'})
    ktc = _fields(recipes['ktc'], {'sql', 'extractor', 'mapping_code', 'mapping_data', 'raw_columns', 'target_columns', 'stats_format'})
    mfl = _fields(recipes['mfl'], {'sql', 'build', 'target_columns', 'stats_format', 'model_version', 'verified'})
    for recipe, keys in ((ktc, ('sql', 'extractor', 'mapping_code', 'mapping_data')), (mfl, ('sql', 'build'))):
        for key in keys:
            validate_identity(recipe[key])
            if key != 'mapping_data' and recipe[key] not in sources:
                raise ValueError('recipe source missing from sources')
    _equal(ktc['raw_columns'], ['format', 'ktc_id', 'sleeper_id', 'date', 'sf_value'])
    _equal(ktc['target_columns'], TARGET)
    _equal(mfl['target_columns'], TARGET + ['start_evidence', 'end_evidence', 'start_trades_window', 'end_trades_window', 'end_value_next_board', 'end_evidence_next_board'])
    for recipe in (ktc, mfl):
        _equal(recipe['stats_format'], 'training-players-zip')
    _equal(mfl['model_version'], 'baseline-hl28-v1')
    _equal(mfl['verified'], True)


def _review(value):
    _string(value['reviewer'])
    _hex(value['evidence_commit'], 40)
    _path(value['evidence_reference'])
    return _timestamp(value['reviewed_at'])


def validate_joint_attestation(value: dict) -> None:
    _fields(value, {'schema_version', 'season', 'replacement_manifest_sha256', 'confirmation_manifest_sha256', 'reviewer', 'evidence_commit', 'evidence_reference', 'reviewed_at', 'recipes_committed_before_outcomes', 'evidence_untouched'})
    _equal(value['schema_version'], 1)
    _equal(value['season'], 2026)
    for key in ('replacement_manifest_sha256', 'confirmation_manifest_sha256'):
        _hex(value[key], 64)
    _review(value)
    _equal(value['recipes_committed_before_outcomes'], True)
    _equal(value['evidence_untouched'], True)


REVIEW_FIELDS = {'reviewer', 'evidence_commit', 'evidence_reference', 'reviewed_at', 'transition_source', 'joint_review'}


def validate_registry(value: dict) -> None:
    _fields(value, {'schema_version', 'attestation', 'entries'})
    _equal(value['schema_version'], 1)
    _equal(value['attestation'], 'review and Git evidence; not proof of non-observation')
    if not isinstance(value['entries'], list):
        raise ValueError('entries must be a list')  # noqa: TRY004 -- protocol errors use ValueError
    seen = set()
    for entry in value['entries']:
        if not isinstance(entry, dict) or entry.get('status') not in ('sealed', 'untouched', 'reused', 'used'):
            raise ValueError('invalid status')
        status = entry['status']
        fields = {'season', 'manifest_sha256', 'status', 'sealing_commit', 'sealed_at'}
        if status != 'sealed':
            fields |= REVIEW_FIELDS
        if status == 'used':
            fields |= {'claim', 'report'}
        _fields(entry, fields)
        _equal(entry['season'], 2026)
        _hex(entry['manifest_sha256'], 64)
        _hex(entry['sealing_commit'], 40)
        sealed = _timestamp(entry['sealed_at'])
        if entry['manifest_sha256'] in seen:
            raise ValueError('duplicate registry key')
        seen.add(entry['manifest_sha256'])
        if status != 'sealed':
            if _review(entry) < sealed:
                raise ValueError('review predates seal')
            _equal(entry['transition_source'], 'untouched' if status == 'used' else 'sealed')
            validate_identity(entry['joint_review'])
        if status == 'used':
            validate_identity(entry['claim'])
            if entry['report'] is not None:
                validate_identity(entry['report'])


def validate_registration(value: dict) -> None:
    _fields(value, {'schema_version', 'season', 'manifest_sha256', 'ktc', 'mfl', 'holdout_attestation'})
    _equal(value['schema_version'], 1)
    _equal(value['season'], 2026)
    _hex(value['manifest_sha256'], 64)
    for name, keys in (('ktc', {'raw_end', 'target_csv', 'target_stats', 'target_sidecar', 'sql', 'extractor', 'mapping_code', 'mapping_data'}), ('mfl', {'target_csv', 'target_stats', 'target_sidecar', 'sql', 'build'})):
        _fields(value[name], keys)
        for identity in value[name].values():
            validate_identity(identity)
    review = _fields(value['holdout_attestation'], REVIEW_FIELDS | {'registry_key', 'status'})
    _equal(review['registry_key'], '2026:' + value['manifest_sha256'])
    _equal(review['status'], 'untouched')
    _equal(review['transition_source'], 'sealed')
    _review(review)
    validate_identity(review['joint_review'])


def validate_references(manifest: dict, registry: dict, registration: dict, attestation: dict, *, manifest_sha256: str, attestation_identity: dict, root: Path) -> dict:
    validate_manifest(manifest)
    validate_registry(registry)
    validate_registration(registration)
    validate_joint_attestation(attestation)
    validate_identity(attestation_identity)
    _hex(manifest_sha256, 64)
    if hashlib.sha256(canonical_bytes(manifest)).hexdigest() != manifest_sha256 or hashlib.sha256(canonical_bytes(attestation)).hexdigest() != attestation_identity['sha256']:
        raise ValueError('artifact hash mismatch')
    matches = [e for e in registry['entries'] if e['manifest_sha256'] == manifest_sha256]
    if len(matches) != 1 or matches[0]['status'] != 'untouched':
        raise ValueError('missing untouched registry entry')
    entry = matches[0]
    review = registration['holdout_attestation']
    if registration['manifest_sha256'] != manifest_sha256 or attestation['replacement_manifest_sha256'] != manifest_sha256 or entry['sealing_commit'] != manifest['sealing_commit'] or _timestamp(entry['sealed_at']) != _timestamp(manifest['sealed_at']):
        raise ValueError('manifest mismatch')
    if _timestamp(attestation['reviewed_at']) < _timestamp(manifest['sealed_at']):
        raise ValueError('review predates seal')
    if entry['joint_review'] != attestation_identity or review['joint_review'] != attestation_identity:
        raise ValueError('joint review mismatch')
    for key in ('reviewer', 'evidence_commit', 'evidence_reference'):
        if entry[key] != review[key] or entry[key] != attestation[key]:
            raise ValueError('review metadata mismatch')
    if entry['evidence_reference'] != attestation_identity['path'] or _timestamp(entry['reviewed_at']) != _timestamp(review['reviewed_at']) or _timestamp(entry['reviewed_at']) != _timestamp(attestation['reviewed_at']):
        raise ValueError('review path or time mismatch')
    for name, keys in (('ktc', ('sql', 'extractor', 'mapping_code', 'mapping_data')), ('mfl', ('sql', 'build'))):
        for key in keys:
            if registration[name][key] != manifest['recipes'][name][key]:
                raise ValueError('recipe mismatch')
    def paths(obj):
        if isinstance(obj, dict):
            for key, item in obj.items():
                if key in ('path', 'registry_path', 'input_registration_path', 'evidence_reference'):
                    _path(item, root)
                else:
                    paths(item)
        elif isinstance(obj, list):
            for item in obj:
                paths(item)
    for obj in (manifest, registry, registration, attestation, attestation_identity):
        paths(obj)
    return entry
