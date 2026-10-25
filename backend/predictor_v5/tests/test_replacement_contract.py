import copy
import hashlib

import pytest

from predictor_v5 import replacement_contract as c


def test_canonical_loader_rejects_ambiguous_json_and_keys(tmp_path):
    path = tmp_path / 'artifact.json'
    path.write_bytes(c.canonical_bytes({'é': 2, 'a': 1}))
    assert path.read_bytes() == b'{"a":1,"\xc3\xa9":2}\n'
    assert c.load_canonical(path, {'a', 'é'}) == ({'a': 1, 'é': 2}, hashlib.sha256(path.read_bytes()).hexdigest())
    for raw in (b'{"a":1,"a":2}\n', b'{"a": 1}\n', b'{"a":1}', b'{"a":NaN}\n'):
        path.write_bytes(raw)
        with pytest.raises(ValueError):
            c.load_canonical(path, {'a'})
    path.write_bytes(b'{"a":1}\n')
    for keys in (set(), {'a', 'b'}):
        with pytest.raises(ValueError):
            c.load_canonical(path, keys)


@pytest.mark.parametrize('path', ['/abs', 'a//b', 'a/./b', 'a/../b', 'a\\b', ''])
def test_identity_rejects_unsafe_path(path):
    with pytest.raises(ValueError):
        c.validate_identity({'path': path, 'sha256': 'a' * 64})


def test_identity_rejects_uppercase_digest_and_extra_fields():
    for value in ({'path': 'a', 'sha256': 'A' * 64}, {'path': 'a', 'sha256': 'a' * 64, 'x': 1}):
        with pytest.raises(ValueError):
            c.validate_identity(value)


def test_joint_review_requires_exact_schema_and_affirmative_attestations():
    value = {'schema_version': 1, 'season': 2026, 'replacement_manifest_sha256': 'a' * 64,
             'confirmation_manifest_sha256': 'b' * 64, 'reviewer': 'reviewer', 'evidence_commit': 'c' * 40,
             'evidence_reference': 'review.json', 'reviewed_at': '2026-10-25T12:00:00+00:00',
             'recipes_committed_before_outcomes': True, 'evidence_untouched': True}
    c.validate_joint_attestation(value)
    for bad in ({**value, 'extra': 1}, {k: v for k, v in value.items() if k != 'reviewer'},
                {**value, 'evidence_untouched': False}, {**value, 'reviewed_at': '2026-10-25'}):
        with pytest.raises(ValueError):
            c.validate_joint_attestation(bad)


def fixture():
    def identity(path):
        return {'path': path, 'sha256': 'a' * 64}
    sources = [identity(name) for name in ('build', 'extractor', 'ksql', 'mapping', 'msql')]
    ktc = {'sql': sources[2], 'extractor': sources[1], 'mapping_code': sources[3], 'mapping_data': identity('mapdata'), 'raw_columns': ['format', 'ktc_id', 'sleeper_id', 'date', 'sf_value'], 'target_columns': ['player_id', 'season', 'start_value', 'end_value', 'position', 'age', 'games', 'points'], 'stats_format': 'training-players-zip'}
    mfl = {'sql': sources[4], 'build': sources[0], 'target_columns': ktc['target_columns'] + ['start_evidence', 'end_evidence', 'start_trades_window', 'end_trades_window', 'end_value_next_board', 'end_evidence_next_board'], 'stats_format': 'training-players-zip', 'model_version': 'baseline-hl28-v1', 'verified': True}
    manifest = {'schema_version': 1, 'protocol': 'ktc-replacement-2026', 'sealing_commit': 'c' * 40, 'sealed_at': '2026-10-25T12:00:00+00:00', 'registry_path': 'registry.json', 'input_registration_path': 'inputs.json', 'output_template': 'backend/predictor_v5/artifacts/replacements/2026-<manifest-sha256>.json', 'claim_template': 'backend/predictor_v5/artifacts/replacements/2026-<manifest-sha256>.claim.json', 'source_identities': sources, 'training': {'archive': identity('archive'), 'ktc_member': 'ktc.csv', 'mfl_labels': identity('mfl'), 'seasons': [2021, 2022, 2023, 2024, 2025], 'positions': ['QB', 'RB', 'TE', 'WR']}, 'starts': {'identity': {'path': 'backend/predictor_v5/data/start_values_2026.csv', 'sha256': '6366b85300c4f9532f0866c602c6d262baaccf6c3a84b9252a6d2e8d46c541bb'}, 'git_blob': 'ead24b508102ca2a1071d07f3f665e9ea920aa96', 'introduced_commit': 'c415177', 'cutoff': '2026-09-09', 'provenance_limit': 'raw start provenance not independently verified'}, 'model': copy.deepcopy(c.FIXED['model']), 'cutoffs': copy.deepcopy(c.FIXED['cutoffs']), 'filters': copy.deepcopy(c.FIXED['filters']), 'gate': copy.deepcopy(c.FIXED['gate']), 'diagnostics': copy.deepcopy(c.FIXED['diagnostics']), 'recipes': {'ktc': ktc, 'mfl': mfl}}
    digest = hashlib.sha256(c.canonical_bytes(manifest)).hexdigest()
    meta = {'reviewer': 'reviewer', 'evidence_commit': 'd' * 40, 'evidence_reference': 'review.json', 'reviewed_at': '2026-10-25T13:00:00+00:00'}
    joint = {'schema_version': 1, 'season': 2026, 'replacement_manifest_sha256': digest, 'confirmation_manifest_sha256': 'b' * 64, **meta, 'recipes_committed_before_outcomes': True, 'evidence_untouched': True}
    joint_id = {'path': 'review.json', 'sha256': hashlib.sha256(c.canonical_bytes(joint)).hexdigest()}
    review = {**meta, 'transition_source': 'sealed', 'joint_review': joint_id}
    entry = {'season': 2026, 'manifest_sha256': digest, 'status': 'untouched', 'sealing_commit': 'c' * 40, 'sealed_at': manifest['sealed_at'], **review}
    registry = {'schema_version': 1, 'attestation': 'review and Git evidence; not proof of non-observation', 'entries': [entry]}
    registration = {'schema_version': 1, 'season': 2026, 'manifest_sha256': digest, 'ktc': {**{k: ktc[k] for k in ('sql', 'extractor', 'mapping_code', 'mapping_data')}, **{k: identity(k) for k in ('raw_end', 'target_csv', 'target_stats', 'target_sidecar')}}, 'mfl': {**{k: mfl[k] for k in ('sql', 'build')}, **{k: identity(k) for k in ('target_csv', 'target_stats', 'target_sidecar')}}, 'holdout_attestation': {'registry_key': '2026:' + digest, 'status': 'untouched', **review}}
    return manifest, registry, registration, joint, digest, joint_id


def test_manifest_and_joint_attestation_exact_schemas():
    manifest, _, _, joint, _, _ = fixture()
    for value, validate in ((manifest, c.validate_manifest), (joint, c.validate_joint_attestation)):
        validate(value)
        for key in value:
            bad = copy.deepcopy(value)
            del bad[key]
            with pytest.raises(ValueError):
                validate(bad)
        with pytest.raises(ValueError):
            validate({**value, 'unknown': 1})


def test_registry_and_registration_cross_references(tmp_path):
    manifest, registry, registration, joint, digest, joint_id = fixture()
    def check():
        return c.validate_references(manifest, registry, registration, joint, manifest_sha256=digest, attestation_identity=joint_id, root=tmp_path)
    assert check() == registry['entries'][0]
    for field, bad in (('reviewer', 'other'), ('evidence_commit', 'e' * 40), ('evidence_reference', 'other'), ('reviewed_at', '2026-10-25T14:00:00+00:00'), ('transition_source', 'untouched')):
        original = registration['holdout_attestation'][field]
        registration['holdout_attestation'][field] = bad
        with pytest.raises(ValueError):
            check()
        registration['holdout_attestation'][field] = original
    registry['entries'].append(copy.deepcopy(registry['entries'][0]))
    with pytest.raises(ValueError):
        check()
    registry['entries'].pop()
    for state in ('sealed', 'reused', 'used'):
        entry = registry['entries'][0]
        old = copy.deepcopy(entry)
        entry['status'] = state
        if state == 'sealed':
            for key in ('reviewer', 'evidence_commit', 'evidence_reference', 'reviewed_at', 'transition_source', 'joint_review'):
                del entry[key]
        if state == 'used':
            entry.update(transition_source='untouched', claim={'path': 'claim', 'sha256': 'a' * 64}, report=None)
        c.validate_registry(registry)
        with pytest.raises(ValueError):
            check()
        registry['entries'][0] = old
    joint_id['sha256'] = 'f' * 64
    with pytest.raises(ValueError):
        check()


def test_cross_references_reject_symlink_escape(tmp_path):
    manifest, registry, registration, joint, digest, joint_id = fixture()
    (tmp_path / 'raw_end').symlink_to(tmp_path.parent / 'outside')
    with pytest.raises(ValueError):
        c.validate_references(manifest, registry, registration, joint, manifest_sha256=digest, attestation_identity=joint_id, root=tmp_path)
