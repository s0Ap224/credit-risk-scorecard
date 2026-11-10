"""模块六。运行 .venv/Scripts/python.exe scripts/evaluate_models.py。"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from audit_data import TARGET, sha256
from fit_logistic import metrics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/module6'
FIGURES = ROOT / 'figures/module6'
MODELS = ('woe_logistic', 'raw_logistic', 'constant')
METRICS = ('auc', 'ks', 'ap', 'brier', 'log_loss')


def read_frozen_test():
    """使用既有预测与未取整分数，并核对测试索引及标签。"""
    inputs = {}
    manifests = {}
    for module in (4, 5):
        path = ROOT / f'results/module{module}/module{module}_manifest.json'
        manifests[module] = json.loads(path.read_text(encoding='utf-8'))
        inputs[path.relative_to(ROOT).as_posix()] = sha256(path)
    for module, name in [(4, 'test_predictions.csv'), (4, 'model_comparison.csv'), (5, 'test_scores.csv')]:
        path = Path(f'results/module{module}/{name}')
        expected = {Path(k).as_posix(): v for k, v in manifests[module]['outputs'].items()}[path.as_posix()]
        if sha256(ROOT / path) != expected:
            raise ValueError(f'冻结结果的文件哈希不匹配 {path}')
        inputs[path.as_posix()] = expected
    entry = manifests[4]['inputs']['assignment']
    assignment_path = ROOT / entry['path']
    if sha256(assignment_path) != entry['sha256']:
        raise ValueError('固定样本索引的哈希不匹配')
    inputs[assignment_path.relative_to(ROOT).as_posix()] = entry['sha256']
    keys = ['row_id', TARGET]
    assignment = pd.read_csv(assignment_path, usecols=[*keys, 'split'])
    expected_rows = assignment.loc[assignment.split.eq('test'), keys].reset_index(drop=True)
    test = pd.read_csv(ROOT / 'results/module4/test_predictions.csv')
    scores = pd.read_csv(ROOT / 'results/module5/test_scores.csv',
                         usecols=[*keys, 'score_unrounded', 'probability_model4'])
    pd.testing.assert_frame_equal(test[keys], expected_rows)
    pd.testing.assert_frame_equal(scores[keys], expected_rows)
    if len(test) != 30_000 or test[TARGET].sum() != 2_005 or not test.row_id.is_unique:
        raise ValueError('测试集规模、标签或行号异常')
    np.testing.assert_allclose(test.woe_logistic, scores.probability_model4, rtol=0, atol=1e-14)
    test['score_unrounded'] = scores.score_unrounded
    return test, inputs


def near_equal_groups(values, groups=10):
    """按值升序分组。仅在同值块之间找距目标秩最近的边界。"""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or groups < 1:
        raise ValueError('分组需要非空的一维有限数值和正组数')
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    # 候选边界升序排列，距离相同时 argmin 取较小的累计人数。
    candidates = np.r_[0, ends]
    cuts = sorted({int(candidates[np.abs(candidates - len(values)*k/groups).argmin()])
                   for k in range(1, groups)} - {0, len(values)})
    return np.searchsorted(cuts, ends[inverse], side='left') + 1


def summarize_groups(y, probability, order_key):
    group = near_equal_groups(order_key)
    rows = pd.DataFrame({'group': group, 'target': y, 'probability': probability, 'key': order_key})
    summary = rows.groupby('group', as_index=False).agg(
        n=('target', 'size'), bad=('target', 'sum'), mean_probability=('probability', 'mean'),
        min_probability=('probability', 'min'), max_probability=('probability', 'max'),
        min_key=('key', 'min'), max_key=('key', 'max'))
    summary['bad_rate'] = summary.bad / summary.n
    summary['cumulative_population_share'] = summary.n.cumsum() / len(rows)
    summary['cumulative_bad_coverage'] = summary.bad.cumsum() / summary.bad.sum()
    return summary


def evaluate(test):
    y = test[TARGET].to_numpy()
    comparison, roc_parts, pr_parts, calibration_parts = [], [], [], []
    risk = summarize_groups(y, test.woe_logistic.to_numpy(), test.score_unrounded.to_numpy())
    risk = risk.rename(columns={'min_key': 'min_score', 'max_key': 'max_score'})
    for model in MODELS:
        probability = test[model].to_numpy()
        result = metrics(test[TARGET], probability)
        fpr, tpr, roc_threshold = roc_curve(y, probability, drop_intermediate=False)
        precision, recall, pr_threshold = precision_recall_curve(y, probability)
        roc_parts.append(pd.DataFrame({'model': model, 'fpr': fpr, 'tpr': tpr,
                                       'threshold': np.where(np.isfinite(roc_threshold), roc_threshold, np.nan)}))
        pr_parts.append(pd.DataFrame({'model': model, 'recall': recall, 'precision': precision,
                                      'threshold': np.r_[pr_threshold, np.nan]}))
        bins = (risk.drop(columns=['min_score', 'max_score']).copy() if model == 'woe_logistic'
                else summarize_groups(y, probability, -probability).drop(columns=['min_key', 'max_key']))
        bins.insert(0, 'model', model)
        calibration_parts.append(bins)
        comparison.append({'model': model, **result, 'mean_probability': probability.mean(),
                           'group_weighted_absolute_gap':
                               np.average(abs(bins.mean_probability-bins.bad_rate), weights=bins.n)})
    return (pd.DataFrame(comparison), pd.concat(roc_parts), pd.concat(pr_parts),
            pd.concat(calibration_parts), risk)


def plot_evaluation(roc, pr, calibration, risk, prevalence):
    """一张矢量图包含四个可独立阅读的面板。"""
    os.environ['SOURCE_DATE_EPOCH'] = '0'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter
    from PIL import Image
    import pymupdf

    sys.path.insert(0, str(ROOT / 'utils'))
    from export_figure import export_figure
    from setup_style import setup_style
    from visual_qa import audit_layout

    setup_style(journal='general', lang='zh', serif_for_zh=True, use_sciplots=False, constrained_layout=False)
    plt.rcParams.update({'font.serif': ['SimSun', 'Times New Roman'], 'font.size': 8,
                         'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                         'axes.linewidth': .55, 'svg.hashsalt': 'module6-evaluation'})
    size = (16 / 2.54, 10.8 / 2.54)
    fig, axes = plt.subplots(2, 2, figsize=size)
    fig.subplots_adjust(left=.105, right=.975, bottom=.105, top=.835, wspace=.32, hspace=.69)
    blue, orange, gray = '#0072B2', '#C54E10', '#626262'
    styles = [(blue, '-', 'o'), (orange, '--', '^')]
    ax_roc, ax_pr, ax_cal, ax_risk = axes.flat
    ax_roc.plot([0, 1], [0, 1], ':', color=gray, lw=1.0)
    ax_pr.axhline(prevalence, ls=':', color=gray, lw=1.0)
    for model, (color, ls, marker) in zip(MODELS[:2], styles):
        r, p = roc[roc.model.eq(model)], pr[pr.model.eq(model)]
        ax_roc.plot(r.fpr, r.tpr, ls, color=color, lw=1.35)
        ax_pr.step(p.recall.iloc[::-1], p.precision.iloc[::-1], where='pre', ls=ls, color=color, lw=1.35)
        c = calibration[calibration.model.eq(model)]
        ax_cal.scatter(c.mean_probability, c.bad_rate, c=color, marker=marker, s=21, zorder=3)
    c = calibration[calibration.model.eq('constant')]
    ax_cal.scatter(c.mean_probability, c.bad_rate, c=gray, marker='s', s=25, zorder=4)
    calibration_limit = .10
    calibration_ticks = np.linspace(0, calibration_limit, 6)
    ax_cal.plot([0, calibration_limit], [0, calibration_limit], ':', color='#888888', lw=.85)
    ax_cal.set(xlim=(0, calibration_limit), ylim=(0, calibration_limit),
               xticks=calibration_ticks, yticks=calibration_ticks)
    for ax in (ax_roc, ax_pr):
        ax.set(xlim=(0, 1), ylim=(0, 1.02))
        ax.set_xticks([0, .25, .5, .75, 1])
        ax.set_yticks([0, .25, .5, .75, 1])
    ax_risk.bar(risk.group, risk.bad_rate, width=.68, color='#B6D8E9', edgecolor=blue,
                linewidth=.55, label='实际坏率')
    ax_risk.scatter(risk.group, risk.mean_probability, c=orange, marker='D', s=13,
                    label='平均预测概率', zorder=3)
    risk_limit = np.ceil(calibration[['mean_probability', 'bad_rate']].max().max()*20) / 20
    ax_risk.set(ylim=(0, risk_limit), xlim=(.5, len(risk)+.5))
    ax_risk.set_xticks(risk.group)
    ax_risk.set_yticks(np.arange(0, risk_limit + .001, .1))
    ax_risk.legend(loc='upper right', fontsize=7, frameon=False, borderaxespad=.15)
    labels = [('(a) ROC', '假阳性率', '真正率'), ('(b) PR', '召回率', '精确率'),
              ('(c) 校准', '平均预测概率', '实际坏率'), ('(d) WOE 评分风险分层', '组号（高风险到低风险）', '坏样本比例')]
    for ax, (title, xlabel, ylabel) in zip(axes.flat, labels):
        ax.set_title(title, loc='left', fontsize=9, pad=4)
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.spines[['top', 'right']].set_visible(False)
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        if ax is not ax_risk:
            ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.tick_params(length=2, width=.55, pad=2)
    fig.legend([Line2D([], [], color=color, ls=ls, marker=marker, markersize=3.5)
                for color, ls, marker in [*styles, (gray, ':', 's')]],
               ['WOE Logistic', '原值 Logistic', '常数基准'], ncol=3, frameon=False,
               loc='upper center', bbox_to_anchor=(.54, .985), fontsize=8)
    issues = audit_layout(fig)
    if issues:
        raise RuntimeError(f'图形布局检查未通过 {issues}')
    FIGURES.mkdir(parents=True, exist_ok=True)
    base = FIGURES / 'test_evaluation'
    files = export_figure(fig, str(base), formats=['pdf', 'svg', 'png'], size_inches=size, dpi=300, tight=False)
    gray_path = base.with_name('test_evaluation_grayscale.png')
    with Image.open(base.with_suffix('.png')) as bitmap:
        bitmap.convert('L').save(gray_path, dpi=(300, 300))
    with pymupdf.open(base.with_suffix('.pdf')) as pdf:
        for font in pdf[0].get_fonts(full=True):
            if font[2] == 'Type3' or not pdf.extract_font(font[0])[3]:
                raise RuntimeError('矢量图字体未正确嵌入')
    plt.close(fig)
    shutil.copy2(base.with_suffix('.pdf'), ROOT / 'latex/figures/test_evaluation.pdf')
    return [Path(path) for path in files] + [gray_path]


def write_tex(risk):
    lines = [r'\begin{tabular}{rrrrrrr}', r'\toprule',
             r'组 & 观测分数范围 & 人数 & 坏样本 & 平均预测 & 实际坏率 & 累计覆盖 \\', r'\midrule']
    for row in risk.itertuples(index=False):
        lines.append((f'{row.group} & {row.min_score:.3f}--{row.max_score:.3f} & {row.n:,} & {row.bad:,} & '
                      f'{row.mean_probability:.2%} & {row.bad_rate:.2%} & {row.cumulative_bad_coverage:.2%}'
                      ).replace('%', r'\%') + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    path = ROOT / 'latex/tables/evaluation_risk_groups.tex'
    path.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--stats-only', action='store_true', help='生成评价表，跳过绘图')
    args = parser.parse_args()
    if args.smoke:
        values = np.array([3]*13 + [1]*5 + [4]*12)
        group = near_equal_groups(values)
        assert len(set(group)) == 3 and all(len(set(group[values == v])) == 1 for v in set(values))
        assert np.all(near_equal_groups(np.ones(30)) == 1)
        y = np.array([0, 1, 0, 0, 1, 0])
        toy = pd.DataFrame({TARGET: y, 'woe_logistic': [.2, .6, .2, .4, .6, .3],
                            'raw_logistic': [.2, .5, .2, .4, .5, .3], 'constant': np.repeat(1/3, len(y)),
                            'score_unrounded': [630, 530, 630, 570, 530, 590]})
        results, _, _, bins, risk = evaluate(toy)
        assert results.iloc[0].auc == 1 and results.iloc[-1].ap == 1/3
        assert risk.bad.sum() == 2 and risk.n.sum() == len(y) and risk.cumulative_bad_coverage.iloc[-1] == 1
        assert len(bins[bins.model.eq('constant')]) == 1
        print('Evaluation smoke checks passed')
        return
    test, inputs = read_frozen_test()
    comparison, roc, pr, calibration, risk = evaluate(test)
    previous = pd.read_csv(ROOT / 'results/module4/model_comparison.csv').query("split == 'test'").set_index('model')
    np.testing.assert_allclose(comparison.set_index('model')[list(METRICS)],
                               previous.loc[list(MODELS), list(METRICS)], atol=1e-14, rtol=0)
    if len(risk) != 10 or calibration.groupby('model').n.sum().ne(len(test)).any():
        raise ValueError('真实测试数据的分组规模不符')
    OUT.mkdir(parents=True, exist_ok=True)
    outputs = []
    for name, table in [('test_evaluation', comparison), ('roc_points', roc), ('pr_points', pr),
                        ('calibration_groups', calibration), ('risk_groups', risk)]:
        path = OUT / f'{name}.csv'
        table.to_csv(path, encoding='utf-8-sig', index=False)
        outputs.append(path)
    if not args.stats_only:
        outputs += plot_evaluation(roc, pr, calibration, risk, test[TARGET].mean())
        outputs.append(ROOT / 'latex/figures/test_evaluation.pdf')
    outputs.append(write_tex(risk))
    for path, expected in inputs.items():
        if sha256(ROOT / path) != expected:
            raise RuntimeError(f'评价期间输入发生变化 {path}')
    manifest = {'command': '.venv/Scripts/python.exe scripts/evaluate_models.py',
                'positive_class': 1, 'n': len(test), 'bad': int(test[TARGET].sum()),
                'grouping': 'nearest cumulative decile boundary among complete tie blocks, smaller boundary if equidistant',
                'risk_order': 'unrounded score ascending', 'refit': False, 'recalibration': False,
                'python': sys.version.split()[0],
                'packages': {n: importlib.metadata.version(n) for n in ['numpy','pandas','scikit-learn','matplotlib']},
                'inputs': inputs, 'code_sha256': sha256(Path(__file__)),
                'outputs': {p.relative_to(ROOT).as_posix(): sha256(p) for p in outputs}}
    (OUT / 'module6_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(comparison[['model', *METRICS, 'mean_probability', 'group_weighted_absolute_gap']].to_string(index=False))
    print(risk.to_string(index=False))


if __name__ == '__main__':
    main()
