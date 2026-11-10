"""以冻结训练分数档比较训练集与测试集的分数分布。"""
from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from audit_data import sha256
from evaluate_models import near_equal_groups

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/psi'
PSEUDOCOUNT = 0.5


def checked_path(relative, expected):
    path = ROOT / relative
    if sha256(path) != expected:
        raise ValueError(f'冻结文件哈希不符 {relative}')
    return path


def read_scores(split, expected_hash, assignment):
    relative = f'results/module5/{split}_scores.csv'
    frame = pd.read_csv(checked_path(relative, expected_hash), float_precision='round_trip',
                        usecols=['row_id', 'score_unrounded'])
    expected_ids = assignment.loc[assignment.split.eq(split), 'row_id'].reset_index(drop=True)
    pd.testing.assert_series_equal(frame.row_id, expected_ids)
    values = frame.score_unrounded.to_numpy()
    if not len(values) or not np.isfinite(values).all() or not expected_ids.is_unique:
        raise ValueError(f'{split} 的冻结分数或索引无效')
    return values


def fit_cutpoints(train):
    """以训练样本目标十分位为准，只在完整同分块后切分。"""
    groups = near_equal_groups(train)
    return [float(train[groups == g].max()) for g in range(1, groups.max())]


def count_by_band(scores, cuts):
    # 左开右闭，等于切点的分数进入较低一档，两端覆盖全部有限分数。
    bands = np.searchsorted(cuts, scores, side='left')
    return np.bincount(bands, minlength=len(cuts)+1)


def psi_components(train_counts, test_counts):
    """如任一格为零，两组所有档位同加 0.5 后分别归一化。"""
    alpha = PSEUDOCOUNT if (train_counts == 0).any() or (test_counts == 0).any() else 0.0
    reference = (train_counts + alpha) / (train_counts.sum() + len(train_counts)*alpha)
    comparison = (test_counts + alpha) / (test_counts.sum() + len(test_counts)*alpha)
    contributions = (comparison - reference) * np.log(comparison/reference)
    return alpha, reference, comparison, contributions


def write_table(rows, total):
    lines = [r'\begin{tabular}{rlrrrrr}', r'\toprule',
             r'档 & 未取整分数范围 & 训练人数 & 训练比例 & 测试人数 & 测试比例 & PSI 分项 \\',
             r'\midrule']
    for row in rows.itertuples(index=False):
        train_pct = f'{row.train_share:.2%}'.replace('%', r'\%')
        test_pct = f'{row.test_share:.2%}'.replace('%', r'\%')
        lines.append(f'{row.band} & $' + f'{row.interval}$ & {row.train_n:,} & {train_pct} & '
                     f'{row.test_n:,} & {test_pct} & {row.psi:.6f}' + r' \\')
    lines.extend([r'\midrule',
                  f' & 合计 & {rows.train_n.sum():,} & 100.00\\% & {rows.test_n.sum():,} & 100.00\\% & '
                  f'{total:.6f}' + r' \\', r'\bottomrule', r'\end{tabular}'])
    (ROOT / 'latex/tables/psi_bands.tex').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    manifest_path = ROOT / 'results/module5/module5_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    assignment_meta = manifest['inputs']['module4_assignment']
    assignment = pd.read_csv(checked_path(assignment_meta['path'], assignment_meta['sha256']),
                             usecols=['row_id', 'split'])
    score_hashes = {Path(key).as_posix(): value for key, value in manifest['outputs'].items()}
    train_hash = score_hashes['results/module5/train_scores.csv']
    train = read_scores('train', train_hash, assignment)
    cuts = fit_cutpoints(train)
    OUT.mkdir(parents=True, exist_ok=True)
    frozen_path = OUT / 'psi_bins.json'
    frozen = dict(reference_split='train', score_field='score_unrounded', requested_bands=10,
                  rule='left open, right closed; unbounded first and last bands',
                  finite_cutpoints=cuts, train_scores_sha256=train_hash)
    # 先保存训练分数档，再打开测试分数。
    frozen_path.write_text(json.dumps(frozen, indent=2)+'\n', encoding='utf-8')

    test_hash = score_hashes['results/module5/test_scores.csv']
    test = read_scores('test', test_hash, assignment)
    train_n, test_n = (count_by_band(values, cuts) for values in (train, test))
    alpha, p, q, contributions = psi_components(train_n, test_n)
    edges = [r'-\infty', *(f'{cut:.3f}' for cut in cuts), r'+\infty']
    intervals = [f'({lo}, {hi}' + (')' if hi == r'+\infty' else ']')
                 for lo, hi in zip(edges, edges[1:])]
    rows = pd.DataFrame(dict(band=range(1, len(edges)), interval=intervals,
                             train_n=train_n, train_share=train_n/len(train),
                             test_n=test_n, test_share=test_n/len(test),
                             train_share_for_psi=p, test_share_for_psi=q, psi=contributions))
    total = float(contributions.sum())
    rows.to_csv(OUT / 'psi_by_band.csv', index=False, encoding='utf-8-sig')
    write_table(rows, total)
    summary = dict(psi=total, pseudocount_applied=alpha, train_n=len(train), test_n=len(test),
                   max_contribution_band=int(contributions.argmax())+1,
                   max_absolute_share_gap=float(np.max(np.abs(q-p))),
                   split_seed=manifest['seed_from_frozen_split'],
                   python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                   command='.venv/Scripts/python.exe scripts/score_psi.py',
                   inputs={'train_scores_sha256': train_hash, 'test_scores_sha256': test_hash,
                           'assignment_sha256': assignment_meta['sha256'],
                           'module5_manifest_sha256': sha256(manifest_path)},
                   code_sha256=sha256(Path(__file__)), frozen_bins_sha256=sha256(frozen_path))
    (OUT / 'psi_summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    print(f'PSI = {total:.9f}; applied pseudocount = {alpha:g}')
    print(rows[['band', 'train_n', 'test_n', 'train_share', 'test_share', 'psi']].to_string(index=False))


if __name__ == '__main__':
    main()
