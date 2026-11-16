"""Frozen samples and strict output storage for the paper-completion ablations.

This module uses only the standard library and the local answer scorer. Planning
and resume validation never import torch, download weights, or run inference.
"""
import hashlib
import json
import os
import re
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

from .score import contains_answer, normalize, score

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = 'archive/Qwen3-8B-bfloat16-masking-2026-11-15.tar.gz'
ARCHIVE_SHA = '99f93d7dbc22c24768c9aec7c669b45e96a513722c314e0f941fd71a206bdd35'
PLAN = ROOT / 'experiments/paper_ablations_v1.json'
SEED = 20261116


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def frozen_json(path, value):
    """A changed sample, setting, runtime, or code version needs a new run name."""
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'{path}: protocol changed; use a new --run-name (do not overwrite results)')
    else:
        atomic_json(path, value)


def archived_rows(path):
    if file_sha(path) != ARCHIVE_SHA:
        raise ValueError('The input archive does not match the reviewed 8B masking archive')
    rows, labels = {}, None
    with tarfile.open(path) as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            if member.name.endswith('part0_requires_composition.json'):
                labels = json.load(tar.extractfile(member))['labels']
            elif member.name.endswith('.jsonl'):
                name = Path(member.name).stem
                if name in rows:
                    raise ValueError(f'duplicate archive member: {name}')
                records = [json.loads(line) for line in tar.extractfile(member).read().splitlines()]
                rows[name] = {r['id']: r for r in records}
                if len(rows[name]) != len(records):
                    raise ValueError(f'duplicate archived IDs: {name}')
    if labels is None:
        raise ValueError('composition labels missing from archive')
    return rows, labels


def benchmark_questions(folder, wanted):
    folder, wanted = Path(folder), set(wanted)
    expected = {name: sha for sha, name in (line.split() for line in
                (ROOT / 'benchmark/SHA256SUMS').read_text().splitlines())}
    questions, hashes = {}, {}
    for name in ['toy.jsonl', 'musique.jsonl']:
        path = folder / name
        if not path.exists():
            raise FileNotFoundError(f'{path}: build benchmarks with python -m latent_dispatch.data toy musique')
        actual = file_sha(path)
        if actual != expected[name]:
            raise ValueError(f'benchmark hash mismatch: {path}')
        hashes[name] = actual
        with path.open() as f:
            for line in f:
                q = json.loads(line)
                if q['id'] in wanted:
                    if q['id'] in questions:
                        raise ValueError(f'duplicate benchmark ID: {q["id"]}')
                    questions[q['id']] = q
    if wanted - questions.keys():
        raise ValueError(f'missing benchmark IDs: {sorted(wanted - questions.keys())[:3]}')
    return questions, hashes


def stable_order(values, salt):
    return sorted(values, key=lambda v: digest([SEED, salt, v]))


def balanced_sample(ids, questions, n, salt):
    """Round-robin dataset/order strata, deterministically; no new outputs select cases."""
    groups = defaultdict(list)
    for id in stable_order(ids, salt):
        q = questions[id]
        groups[q['dataset'], q['placement']].append(id)
    chosen = []
    while len(chosen) < n:
        advanced = False
        for key in sorted(groups):
            if groups[key] and len(chosen) < n:
                chosen.append(groups[key].pop(0))
                advanced = True
        if not advanced:
            raise ValueError(f'only {len(chosen)} eligible IDs for requested {n} in {salt}')
    return chosen


def toy_candidates(q):
    """Three mayors from the supporting cities paragraph, fixed before new inference."""
    ref = next(g for g in q['gold'] if g['hop'] == 1)
    paragraph = q['workers'][ref['worker']][ref['index']]
    names = re.findall(r'The mayor of [^.]+? is ([^.]+)\.', paragraph)
    gold = q['answers'][0]
    wrong = sorted({n for n in names if normalize(n) not in {normalize(a) for a in q['answers']}})
    if len(names) != 3 or len(wrong) != 2 or normalize(gold) not in {normalize(n) for n in names}:
        raise ValueError(f'expected the correct mayor and two incorrect mayors: {q["id"]}')
    return [gold, *wrong]


def build_plan(benchmark_dir, archive=ROOT / ARCHIVE):
    raw, labels = archived_rows(archive)
    ids = set(raw['mask_a_gold'])
    qs, hashes = benchmark_questions(benchmark_dir, ids)
    for id in ids:
        q = qs[id]
        if not labels[q['dataset'] + ':' + q['source_id']]:
            raise ValueError(f'noneligible question in masking archive: {id}')

    paired = ids & raw['b1'].keys() & raw['b1_mask'].keys()
    categories, provenance = defaultdict(list), {}
    for id in paired:
        q = qs[id]
        # The diagnostic sample covers split AB/BA cases. It is not an accuracy estimate.
        if q['placement'] not in ['AB', 'BA']:
            continue
        outcomes = {name: bool(score(raw[name][id]['text'], q['answers'])[1]) for name in
                    ['b1', 'b1_mask', 'mask_a_full', 'mask_a_blocked']}
        mismatch = outcomes['b1'] != outcomes['mask_a_full'] or outcomes['b1_mask'] != outcomes['mask_a_blocked']
        category = 'mismatch' if mismatch else 'agreement'
        categories[category].append(id)
        provenance[id] = {'archived_group': category, 'archived_em': outcomes}
    diagnostic = []
    for category in ['mismatch', 'agreement']:
        diagnostic.extend(balanced_sample(categories[category], qs, 12, category))

    thinking = []
    for dataset in ['toy_d2', 'musique']:
        sources = {q['source_id'] for q in qs.values() if q['dataset'] == dataset}
        selected = stable_order(sources, 'thinking:' + dataset)[:25]
        if len(selected) != 25:
            raise ValueError('not enough eligible sources')
        for source in selected:
            for placement in ['AB', 'BA']:
                matches = [id for id,q in qs.items() if q['dataset']==dataset and q['source_id']==source
                           and q['placement']==placement]
                if len(matches) != 1:
                    raise ValueError(f'missing or ambiguous mirrored layout: {source} {placement}')
                thinking.extend(matches)

    likelihood, candidates = [], {}
    for id, row in sorted(raw['mask_a_distractor'].items()):
        q = qs[id]
        if q['dataset'] != 'toy_d2' or q['placement'] not in ['AB','BA']:
            continue
        selected = row['mask']['a_paragraphs'][0]
        if selected >= min(row['mask']['a_gold_paragraphs']):
            continue
        paragraph = q['workers'][0][selected]
        if contains_answer(paragraph, q['answers']) or contains_answer(paragraph, q['bridges']):
            raise ValueError(f'pre-gold control contains bridge/answer: {id}')
        likelihood.append(id)
        candidates[id] = {'answers': toy_candidates(q), 'a_distractor_index': selected,
                          'archived': {c:{'gold_logprob':raw['mask_a_'+c][id]['gold_logprob'],
                                          'em':bool(score(raw['mask_a_'+c][id]['text'],q['answers'])[1]),
                                          'cut':raw['mask_a_'+c][id]['cut']}
                                       for c in ['blocked','distractor','gold']}}
    if Counter(qs[id]['placement'] for id in likelihood) != {'AB':42, 'BA':43}:
        raise ValueError('clean-control sample differs from reviewed 42 AB / 43 BA cohort')

    selected = set(diagnostic + thinking + likelihood)
    return {'protocol': 'paper_ablations_v1', 'selection_seed': SEED,
            'source_archive': ARCHIVE, 'source_archive_sha256': ARCHIVE_SHA,
            'benchmark_sha256': hashes, 'model': 'Qwen/Qwen3-8B', 'dtype': 'bfloat16',
            'question_sha256': {id:digest(qs[id]) for id in sorted(selected)},
            'validation': {'ids':diagnostic, 'selection':{id:provenance[id] for id in diagnostic},
                           'arms':['b1','masked_full','b1_mask','masked_blocked'], 'answer_tokens':256,
                           'expected_rows':96},
            'thinking': {'ids':thinking, 'conditions':['blocked','gold','full'], 'modes':['off','forced'],
                         'think_budget':1024, 'answer_tokens':1024, 'expected_rows':600},
            'likelihood': {'ids':likelihood, 'conditions':['blocked','distractor','gold'],
                           'candidates':candidates, 'expected_rows':255}}


def load_plan(benchmark_dir, plan_path=PLAN):
    plan = json.loads(Path(plan_path).read_text())
    if plan['protocol'] != 'paper_ablations_v1':
        raise ValueError('unknown protocol')
    qs, hashes = benchmark_questions(benchmark_dir, plan['question_sha256'])
    if hashes != plan['benchmark_sha256']:
        raise ValueError('frozen benchmark changed')
    for id,q in qs.items():
        if digest(q) != plan['question_sha256'][id]:
            raise ValueError(f'frozen question changed: {id}')
    return plan, qs


def specifications(job, plan):
    p = plan[job]
    if job == 'validation':
        return {a:{'variant':a, 'thinking':'off', 'answer_tokens':p['answer_tokens']} for a in p['arms']}
    if job == 'thinking':
        return {c+'_'+m:{'condition':c, 'thinking':m, 'answer_tokens':p['answer_tokens'],
                         'think_budget':p['think_budget']} for m in p['modes'] for c in p['conditions']}
    if job == 'likelihood':
        return {c:{'condition':c, 'thinking':'off'} for c in p['conditions']}
    raise ValueError(job)


class RowStore:
    """Append-only, fsynced rows. Preserve and remove only a torn final JSON line."""
    def __init__(self, path, fingerprint, allowed):
        self.path, self.fingerprint, self.allowed = Path(path), fingerprint, set(allowed)
        self.rows = {}
        if not self.path.exists():
            return
        content = self.path.read_bytes()
        lines = content.splitlines(keepends=True)
        offset = 0
        for n,line in enumerate(lines):
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                if n != len(lines)-1 or line.endswith(b'\n'):
                    raise ValueError(f'corrupt interior row in {path}, line {n+1}') from None
                backup = self.path.with_name(self.path.name + '.torn-' + hashlib.sha256(line).hexdigest()[:12])
                backup.write_bytes(line)
                with self.path.open('r+b') as f:
                    f.truncate(offset)
                break
            self._accept(row)
            offset += len(line)
        # A complete last row without newline is valid, but needs a separator before append.
        if self.path.stat().st_size and not self.path.read_bytes().endswith(b'\n'):
            with self.path.open('ab') as f:
                f.write(b'\n')

    def _validate(self, row):
        key = (row.get('id'), row.get('condition'))
        if row.get('fingerprint') != self.fingerprint or key not in self.allowed:
            raise ValueError(f'{self.path}: wrong fingerprint or unexpected question/condition {key}')
        if key in self.rows:
            raise ValueError(f'{self.path}: duplicate row {key}')
        return key

    def _accept(self, row):
        key = self._validate(row)
        self.rows[key] = row

    def append(self, row):
        row = {**row, 'fingerprint':self.fingerprint}
        payload = json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n'
        key = self._validate(row)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('a', encoding='utf-8') as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        self.rows[key] = row
