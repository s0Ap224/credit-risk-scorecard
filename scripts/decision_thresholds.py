"""模块七。先运行 freeze，只用验证集选阈值；再运行 evaluate。"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from audit_data import TARGET, sha256
from feature_risk import wilson

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/module7'
FREEZE = OUT / 'frozen_thresholds.json'
FP_COST, FN_COST = 1, 5
THEORY = FP_COST / (FP_COST + FN_COST)
PROBABILITY_GRID = sorted({*(i / 100 for i in range(1, 100)), THEORY})
SCORE_GRID = list(range(300, 901))


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def load_split(split):
    """只打开指定子集的模型预测和分数，并追溯固定行号。"""
    inputs = {}
    manifests = {}
    for module in (4, 5):
        path = ROOT / f'results/module{module}/module{module}_manifest.json'
        manifests[module] = load_json(path)
        inputs[path.relative_to(ROOT).as_posix()] = sha256(path)

    def checked(relative, expected):
        path = ROOT / relative
        digest = sha256(path)
        if digest != expected:
            raise ValueError(f'冻结输入的哈希不一致 {relative}')
        inputs[path.relative_to(ROOT).as_posix()] = digest
        return path

    def output(module, name):
        path = Path(f'results/module{module}/{name}')
        hashes = {Path(k).as_posix(): v for k, v in manifests[module]['outputs'].items()}
        return checked(path, hashes[path.as_posix()])

    assignment_entry = manifests[4]['inputs']['assignment']
    assignment = pd.read_csv(checked(assignment_entry['path'], assignment_entry['sha256']),
                             usecols=['row_id', 'split'])
    expected_rows = assignment.loc[assignment.split.eq(split), 'row_id'].reset_index(drop=True)
    keys = ['row_id', TARGET]
    prediction = pd.read_csv(output(4, f'{split}_predictions.csv'), usecols=[*keys, 'woe_logistic'])
    scores = pd.read_csv(output(5, f'{split}_scores.csv'),
                         usecols=[*keys, 'score_unrounded', 'probability_model4'])
    scaling = load_json(output(5, 'scaling.json'))
    pd.testing.assert_series_equal(prediction.row_id, expected_rows)
    pd.testing.assert_frame_equal(prediction[keys], scores[keys])
    np.testing.assert_allclose(prediction.woe_logistic, scores.probability_model4, rtol=0, atol=1e-14)
    y = prediction[TARGET].to_numpy()
    p = prediction.woe_logistic.to_numpy()
    s = scores.score_unrounded.to_numpy()
    a, b = scaling['A'], scaling['B']
    if (len(y) != 30_000 or not expected_rows.is_unique or not np.isin(y, [0, 1]).all()
            or not ((0 < p) & (p < 1)).all() or not np.isfinite(s).all() or b <= 0):
        raise ValueError(f'{split} 行号、标签、概率或刻度无效')
    np.testing.assert_allclose(s, a + b * (np.log1p(-p) - np.log(p)), rtol=0, atol=1e-10)
    return y, p, s, scaling, inputs, manifests[4]['seed']


def flag_result(y, p, threshold):
    flagged = p >= threshold
    tp = int(y[flagged].sum())
    fp = int(flagged.sum()) - tp
    fn = int(y.sum()) - tp
    tn = len(y) - tp - fp - fn
    cost = FP_COST * fp + FN_COST * fn
    return dict(threshold=threshold, tn=tn, fp=fp, fn=fn, tp=tp,
                flagged=tp+fp, flagged_share=(tp+fp)/len(y),
                recall=tp/(tp+fn) if tp+fn else None,
                precision=tp/(tp+fp) if tp+fp else None,
                total_cost=cost, mean_cost=cost/len(y))


def approval_result(y, p, s, cutoff):
    accepted = s >= cutoff
    n = int(accepted.sum())
    bad = int(y[accepted].sum())
    # 5% 与 10% 用整数比较，恰好等于边界也满足要求。
    return dict(cutoff=cutoff, accepted=n, accepted_share=n/len(y), bad=bad,
                mean_probability=float(p[accepted].mean()) if n else None,
                bad_rate=bad/n if n else None,
                meets_bad_rate_limit=bool(n and 20*bad <= n),
                meets_min_share=10*n >= len(y))


def choose_thresholds(y, p, s, probability_grid, score_grid):
    flags = [flag_result(y, p, t) for t in probability_grid]
    approvals = [approval_result(y, p, s, c) for c in score_grid]
    # 同成本优先较高概率阈值；同接受人数优先较高分数门槛。
    best_flag = min(flags, key=lambda r: (r['total_cost'], -r['threshold']))
    best_approval = max((r for r in approvals if r['meets_bad_rate_limit'] and r['meets_min_share']),
                        key=lambda r: (r['accepted'], r['cutoff']), default=None)
    return flags, approvals, best_flag, best_approval


def score_cutoff(threshold, scaling):
    return scaling['A'] + scaling['B'] * (math.log1p(-threshold) - math.log(threshold))


def freeze():
    y, p, s, scaling, inputs, seed = load_split('validation')
    flags, approvals, flag, approval = choose_thresholds(y, p, s, PROBABILITY_GRID, SCORE_GRID)
    policy = {
        'model': 'woe_logistic', 'selected_on': 'validation', 'positive_label': 1,
        'seed': seed, 'new_random_sampling': False,
        'code_sha256': sha256(Path(__file__)), 'inputs': inputs,
        'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__},
        'commands': ['.venv/Scripts/python.exe scripts/decision_thresholds.py freeze',
                     '.venv/Scripts/python.exe scripts/decision_thresholds.py evaluate'],
        'score_scaling': scaling,
        'risk_marking': {'rule': 'p >= t, equivalently unrounded S <= score_cutoff(t)',
                         'fp_cost': FP_COST, 'fn_cost': FN_COST, 'grid': PROBABILITY_GRID,
                         'tie_break': 'higher threshold', 'theoretical_threshold': THEORY,
                         'selected': {**flag, 'score_cutoff': score_cutoff(flag['threshold'], scaling)}},
        'approval': {'rule': 'unrounded S >= c', 'max_bad_rate': .05, 'min_acceptance_share': .10,
                     'grid': SCORE_GRID, 'tie_break': 'higher score cutoff',
                     'status': 'feasible' if approval is not None else 'infeasible', 'selected': approval},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    if FREEZE.exists() and load_json(FREEZE) != policy:
        raise ValueError('既有冻结规则与本次验证结果不同，须核查差异')
    pd.DataFrame(flags).to_csv(OUT / 'validation_flag_grid.csv', index=False)
    pd.DataFrame(approvals).to_csv(OUT / 'validation_approval_grid.csv', index=False)
    # 写入完整文件后才发布冻结标记，测试阶段读取该文件。
    if not FREEZE.exists():
        with tempfile.NamedTemporaryFile(dir=OUT, delete=False, mode='w', encoding='utf-8') as pending:
            json.dump(policy, pending, ensure_ascii=False, indent=2)
        try:
            os.replace(pending.name, FREEZE)
        finally:
            Path(pending.name).unlink(missing_ok=True)
    print('Frozen from validation:', policy['risk_marking']['selected'], policy['approval']['selected'])


def write_tex(flags, approvals):
    directory = ROOT / 'latex/tables'

    def percent(value):
        return '--' if value is None else f'{value:.2%}'.replace('%', r'\%')

    flag_lines = [r'\begin{tabular}{lrrrrrrrrr}', r'\toprule',
                  r'规则 & 分数上限 & TN & FP & FN & TP & 标记率 & 召回率 & 精确率 & 均成本 \\',
                  r'\midrule']
    chosen = flags[-1]['threshold']
    chosen_text = r'$1/6$' if chosen == THEORY else f'{chosen:.2f}'
    names = {'baseline': '基准 0.5', 'theoretical': r'理论 $1/6$', 'validation': '验证 ' + chosen_text}
    for row in flags:
        cells = [names[row['rule']], f"{row['score_cutoff']:.3f}"]
        cells += [f'{row[key]:,}' for key in ('tn', 'fp', 'fn', 'tp')]
        cells += [percent(row[key]) for key in ('flagged_share', 'recall', 'precision')]
        cells.append(f"{row['mean_cost']:.5f}")
        flag_lines.append(' & '.join(cells) + r' \\')
    flag_lines.extend([r'\bottomrule', r'\end{tabular}'])
    (directory / 'decision_flag_results.tex').write_text('\n'.join(flag_lines)+'\n', encoding='utf-8')

    approval_lines = [r'\begin{tabular}{lrrrrrrr}', r'\toprule',
                      r'样本 & 门槛 & 接受人数 & 接受率 & 坏样本 & 平均预测 & 实际坏率 & 95\% 区间 \\',
                      r'\midrule']
    for row in approvals:
        interval = '--' if row['ci_low'] is None else f"［{percent(row['ci_low'])}, {percent(row['ci_high'])}］"
        name = '验证集' if row['split'] == 'validation' else '测试集'
        cells = [name, str(row['cutoff']), f"{row['accepted']:,}", percent(row['accepted_share']),
                 f"{row['bad']:,}", percent(row['mean_probability']), percent(row['bad_rate']), interval]
        approval_lines.append(' & '.join(cells) + r' \\')
    if not approvals:
        approval_lines.append(r'\multicolumn{8}{c}{验证集无可行门槛} \\')
    approval_lines.extend([r'\bottomrule', r'\end{tabular}'])
    (directory / 'decision_approval_results.tex').write_text('\n'.join(approval_lines)+'\n', encoding='utf-8')


def evaluate():
    policy = load_json(FREEZE)
    freeze_hash = sha256(FREEZE)
    if policy['selected_on'] != 'validation' or policy['code_sha256'] != sha256(Path(__file__)):
        raise ValueError('冻结规则的来源或代码版本不匹配')
    y, p, s, scaling, inputs, seed = load_split('test')
    common_inputs = set(inputs) & set(policy['inputs'])
    if (scaling != policy['score_scaling'] or seed != policy['seed']
            or any(inputs[key] != policy['inputs'][key] for key in common_inputs)):
        raise ValueError('测试输入与验证阶段冻结的模型或刻度不一致')
    selection = policy['risk_marking']
    flags = []
    for rule, t in [('baseline', .5), ('theoretical', selection['theoretical_threshold']),
                    ('validation', selection['selected']['threshold'])]:
        cutoff = score_cutoff(t, scaling)
        np.testing.assert_array_equal(p >= t, s <= cutoff)
        flags.append(dict(rule=rule, score_cutoff=cutoff, **flag_result(y, p, t)))
    pd.DataFrame(flags).to_csv(OUT / 'test_flag_comparison.csv', index=False)

    selected = policy['approval']['selected']
    approvals = []
    if selected is not None:
        test = approval_result(y, p, s, selected['cutoff'])
        low, high = wilson(test['bad'], test['accepted']) if test['accepted'] else (None, None)
        approvals = [dict(split='validation', **selected, ci_low=None, ci_high=None),
                     dict(split='test', **test, ci_low=low, ci_high=high)]
    pd.DataFrame(approvals, columns=['split', 'cutoff', 'accepted', 'accepted_share', 'bad',
                                    'mean_probability', 'bad_rate', 'meets_bad_rate_limit',
                                    'meets_min_share', 'ci_low', 'ci_high']).to_csv(
        OUT / 'approval_comparison.csv', index=False)
    write_tex(flags, approvals)
    if sha256(FREEZE) != freeze_hash:
        raise ValueError('测试期间冻结文件发生变化')
    generated = [*OUT.glob('*.csv'), ROOT / 'latex/tables/decision_flag_results.tex',
                 ROOT / 'latex/tables/decision_approval_results.tex']
    manifest = dict(frozen_thresholds_sha256=freeze_hash, test_inputs=inputs,
                    outputs={path.relative_to(ROOT).as_posix(): sha256(path) for path in sorted(generated)})
    (OUT / 'module7_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print(pd.DataFrame(flags).to_string(index=False))
    print(pd.DataFrame(approvals).to_string(index=False) if approvals else 'Approval infeasible on validation')


def smoke():
    y = np.r_[1, 1, np.zeros(38, dtype=int)]
    p = np.r_[.5, .5, np.full(38, .1)]
    s = np.r_[600, np.full(39, 620)]
    flags, approvals, flag, approval = choose_thresholds(y, p, s, [.4, .5, .9], [600, 619, 620])
    assert flags[1]['tp'] == 2 and flags[2]['precision'] is None and flag['threshold'] == .5
    assert approvals[0]['bad_rate'] == .05 and approval['cutoff'] == 600
    assert approvals[1]['accepted'] == approvals[2]['accepted'] == 39
    assert choose_thresholds(y, p, s, [.5], [621])[-1] is None
    assert choose_thresholds(y[2:], p[2:], s[2:], [.5], [619, 620])[-1]['cutoff'] == 620
    assert math.isclose(score_cutoff(.05, {'A': 515.0414497311283, 'B': 28.85390081777927}), 600)
    print('Module 7 smoke checks passed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'evaluate', 'smoke'])
    {'freeze': freeze, 'evaluate': evaluate, 'smoke': smoke}[parser.parse_args().stage]()
