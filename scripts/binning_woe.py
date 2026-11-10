"""模块三：业务分箱、训练集 WOE/IV、候选变量筛选与冻结查表。

完整复现：.venv/Scripts/python.exe scripts/binning_woe.py
仅计算、不绘图：python scripts/binning_woe.py --stats-only
最小门禁：python scripts/binning_woe.py --smoke
项目独立环境已固定依赖，避免公共 Python 环境变动影响复算。

分箱沿用已确定的 3.1 节；不搜索验证集切点，不读取测试集标签。
筛选诊断采用固定 L2 Logistic，正式模型与最终测试评价留待后续模块。
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from audit_data import LABELS, LATE_COLUMNS, TARGET, sha256
from preprocess import FEATURES, DEPENDENTS_FLAG

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/module3'
TABLES = ROOT / 'latex/tables'
FIGURES = ROOT / 'figures/module3'
ALPHA = 0.5
SEED = 20261103
TOLERANCES = {'auc_loss': .001, 'log_loss_increase': .0001, 'brier_increase': .0001}
NAMES = {k: v[0].replace('--', '–') for k, v in LABELS.items()}
NAMES[DEPENDENTS_FLAG] = '家属人数原缺失指示'

# side='left' 表示右闭区间；side='right' 表示左闭区间。合法域由 assign_bins 核验。
SPECS = {
    FEATURES[0]: {'cuts': [0, .1, .3, .6, 1], 'side': 'left',
                  'labels': ['0', '(0,0.1]', '(0.1,0.3]', '(0.3,0.6]', '(0.6,1]', '(1,+inf)']},
    'age': {'cuts': [30, 40, 50, 60, 70], 'side': 'right',
            'labels': ['(0,30)', '[30,40)', '[40,50)', '[50,60)', '[60,70)', '[70,+inf)']},
    LATE_COLUMNS[0]: {'cuts': [.5, 1.5, 2.5, 3.5], 'side': 'left',
                      'labels': ['0', '1', '2', '3', '>=4', '96/98'], 'special_indices': [5]},
    'DebtRatio': {'cuts': [0, .4, .6, 1, 10, 100], 'side': 'left',
                  'labels': ['0', '(0,0.4]', '(0.4,0.6]', '(0.6,1]', '(1,10]', '(10,100]', '(100,+inf)']},
    'MonthlyIncome': {'cuts': [4000, 6000, 10000], 'side': 'right',
                      'labels': ['MISSING', '[0,10)', '[10,4000)', '[4000,6000)', '[6000,10000)', '[10000,+inf)'],
                      'special_indices': [0, 1]},
    'NumberOfOpenCreditLinesAndLoans': {'cuts': [.5, 2.5, 5.5, 9.5, 14.5], 'side': 'left',
                                       'labels': ['0', '1-2', '3-5', '6-9', '10-14', '>=15']},
    'NumberOfTimes90DaysLate': {'cuts': [.5, 1.5, 2.5], 'side': 'left',
                               'labels': ['0', '1', '2', '>=3', '96/98'], 'special_indices': [4]},
    'NumberRealEstateLoansOrLines': {'cuts': [.5, 2.5, 3.5], 'side': 'left',
                                    'labels': ['0', '1-2', '3', '>=4']},
    LATE_COLUMNS[1]: {'cuts': [.5, 1.5], 'side': 'left',
                      'labels': ['0', '1', '>=2', '96/98'], 'special_indices': [3]},
    'NumberOfDependents': {'cuts': [.5, 1.5, 2.5, 3.5], 'side': 'left',
                           'labels': ['0', '1', '2', '3', '>=4']},
    DEPENDENTS_FLAG: {'cuts': [.5], 'side': 'left', 'labels': ['0', '1']},
}
SPECS = {name: SPECS[name] for name in [*FEATURES, DEPENDENTS_FLAG]}


def assign_bins(X: pd.DataFrame, specs: dict | None = None) -> pd.DataFrame:
    """处理后特征到箱编号；无需标签。未知结构、非法值及不一致标记均报错。"""
    specs = SPECS if specs is None else specs
    required = [*FEATURES, DEPENDENTS_FLAG]
    if not isinstance(X, pd.DataFrame) or X.columns.duplicated().any():
        raise ValueError('输入必须是无重复字段的处理后 DataFrame')
    if set(required) - set(X.columns):
        raise ValueError(f'缺少必需字段：{sorted(set(required)-set(X.columns))}')
    result = pd.DataFrame(index=X.index)
    for name in required:
        raw = X[name]
        x = pd.to_numeric(raw, errors='coerce').astype(float)
        if (raw.notna() & x.isna()).any() or (x.notna() & ~np.isfinite(x)).any():
            raise ValueError(f'{name}: 非法数值或无穷值')
        if name != 'MonthlyIncome' and x.isna().any():
            raise ValueError(f'{name}: 未约定缺失；必须先应用预处理')
        if x.lt(0).any() or (name == 'age' and x.le(0).any()):
            raise ValueError(f'{name}: 非法负值或零年龄')
        if name not in [FEATURES[0], 'DebtRatio', 'MonthlyIncome'] and (x.notna() & x.mod(1).ne(0)).any():
            raise ValueError(f'{name}: 必须为整数')
        if name == DEPENDENTS_FLAG and not x.isin([0, 1]).all():
            raise ValueError(f'{name}: 必须为0/1')
        spec = specs[name]
        ids = np.searchsorted(spec['cuts'], x.to_numpy(), side=spec['side'])
        if name in LATE_COLUMNS:
            mask = x.isin([96, 98])
            ids[mask] = len(spec['labels']) - 1
            flag = f'{name}_special'
            if flag in X and not X[flag].eq(mask.astype(int)).all():
                raise ValueError(f'{flag}: 与原始编码不一致')
        if name == 'MonthlyIncome':
            ids = np.searchsorted(spec['cuts'], x.fillna(0).to_numpy(), side='right') + 2
            ids[x.lt(10)] = 1
            ids[x.isna()] = 0
            expected = np.select([x.isna(), x.lt(10)], ['MISSING', 'LOW'], default='REGULAR')
            if 'MonthlyIncome_state' in X and not X.MonthlyIncome_state.eq(expected).all():
                raise ValueError('MonthlyIncome_state: 与原始收入不一致')
        result[name] = ids.astype('int16')
    if (X[DEPENDENTS_FLAG].eq(1) & X.NumberOfDependents.ne(0)).any():
        raise ValueError('NumberOfDependents_missing: 原缺失必须已填为0')
    return result


def fit_woe(X: pd.DataFrame, y: pd.Series, alpha: float = ALPHA) -> tuple[dict, pd.DataFrame]:
    """仅传训练特征和训练标签；将特殊档计入 K，不给空档凭空估计风险。"""
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError('alpha 必须为有限正数')
    if not isinstance(y, pd.Series) or not y.index.equals(X.index) or not y.isin([0, 1]).all():
        raise ValueError('训练标签必须为与特征索引一致的二分类 Series')
    n1 = int(y.sum())
    n0 = len(y) - n1
    if min(n0, n1) == 0:
        raise ValueError('训练集必须同时包含好坏样本')
    bins = assign_bins(X)
    rows = []
    mappings = {}
    for name, spec in SPECS.items():
        K = len(spec['labels'])
        mappings[name] = []
        for k, label in enumerate(spec['labels']):
            z = y[bins[name].eq(k)]
            n, bad = len(z), int(z.sum())
            good = n - bad
            if n == 0:
                raise ValueError(f'{name}: 空训练档 {label}；须核查或预先确定有依据的映射，不用平滑伪造风险')
            g, b = (good + alpha) / (n0 + alpha * K), (bad + alpha) / (n1 + alpha * K)
            woe = float(np.log(g / b))
            special = k in spec.get('special_indices', [])
            rows.append({'feature': name, 'feature_cn': NAMES[name], 'bin_id': k, 'bin': label,
                         'n': n, 'good': good, 'bad': bad, 'sample_share': n / len(y),
                         'bad_rate': bad / n, 'g': g, 'b': b, 'woe': woe, 'iv_contribution': (g-b)*woe,
                         'special': special, 'K': K, 'alpha': alpha,
                         'ordinary_minimum_pass': special or (n >= 450 and min(good, bad) >= 30)})
            mappings[name].append({'bin_id': k, 'bin': label, 'training_n': n, 'woe': woe})
    table = pd.DataFrame(rows)
    if not table.ordinary_minimum_pass.all():
        raise ValueError('普通箱未满足预定人数要求；请核查，不自动覆盖已批准的分箱')
    for _, t in table.groupby('feature'):
        np.testing.assert_allclose([t.g.sum(), t.b.sum()], [1, 1], atol=1e-12)
        assert int(t.n.sum()) == len(y) and int(t.bad.sum()) == n1
    rules = {'schema_version': 1, 'alpha': alpha, 'direction': 'ln(good_distribution/bad_distribution)',
             'training_n': len(y), 'training_good': n0, 'training_bad': n1, 'specs': SPECS,
             'mappings': mappings, 'bin_design': 'user_approved_business_bins_from_training_EDA',
             'unobserved_training_bin': 'raise_field_level_error_not_smoothed_estimate',
             'test_labels_used': False}
    return rules, table


def transform_woe(X: pd.DataFrame, rules: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """查训练 WOE，不接受标签；合法超训练范围值进入开口尾箱。"""
    if rules.get('schema_version') != 1:
        raise ValueError('WOE 规则版本不兼容')
    bins = assign_bins(X, rules['specs'])
    encoded = pd.DataFrame(index=X.index)
    for name in rules['specs']:
        mapping = {v['bin_id']: v['woe'] for v in rules['mappings'][name] if v['training_n'] > 0}
        encoded[name] = bins[name].map(mapping)
        if encoded[name].isna().any() or not np.isfinite(encoded[name]).all():
            raise ValueError(f'{name}: 未见训练档或非法 WOE 映射')
    return encoded, bins


def load_training_and_validation() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """核验既有原始输入与处理产物哈希；只读取 train/validation，不打开测试数据。"""
    manifest_path = ROOT / 'results/module2/preprocess_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    inputs = {'preprocess_manifest': {'path': str(manifest_path.relative_to(ROOT)), 'sha256': sha256(manifest_path)}}
    for key, entry in manifest['inputs'].items():
        path = ROOT / entry['path']
        if sha256(path) != entry['sha256']:
            raise ValueError(f'原始输入发生变化：{path}')
        inputs[key] = entry
    for key in ['rules', 'train', 'validation']:
        path = ROOT / ('results/module2/preprocess_rules.json' if key == 'rules' else f'results/module2/prepared/{key}.csv')
        expected = manifest['outputs'][str(path.relative_to(ROOT))]
        if sha256(path) != expected:
            raise ValueError(f'冻结处理结果发生变化：{path}')
        inputs[key] = {'path': str(path.relative_to(ROOT)), 'sha256': expected}
    train = pd.read_csv(ROOT / inputs['train']['path'])
    val = pd.read_csv(ROOT / inputs['validation']['path'])
    assignment = pd.read_csv(ROOT / inputs['assignment']['path'])
    for name, frame, n in [('train', train, 90000), ('validation', val, 30000)]:
        expected = assignment.loc[assignment.split.eq(name), ['row_id', TARGET]].reset_index(drop=True)
        pd.testing.assert_frame_equal(frame[['row_id', TARGET]], expected)
        if len(frame) != n:
            raise ValueError(f'{name}: 样本数不符')
    return train, val, inputs


def evaluate_candidates(train_w: pd.DataFrame, y: pd.Series, val_w: pd.DataFrame, val_y: pd.Series,
                        summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """固定诊断模型作有限删变量比较；所有系数只拟合训练集。"""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, roc_curve, log_loss, brier_score_loss

    cache = {}

    def fit(names: list[str]) -> dict:
        key = tuple(names)
        if key in cache:
            return cache[key].copy()
        model = LogisticRegression(C=1, penalty='l2', solver='lbfgs', max_iter=2000, tol=1e-8,
                                   class_weight=None, random_state=SEED)
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            model.fit(train_w[names], y)
        p = model.predict_proba(val_w[names])[:, 1]
        fpr, tpr, _ = roc_curve(val_y, p)
        metrics = {'n_features': len(names), 'features': json.dumps(names, ensure_ascii=False),
                   'auc': roc_auc_score(val_y, p), 'ks': float(np.max(tpr-fpr)),
                   'log_loss': log_loss(val_y, p), 'brier': brier_score_loss(val_y, p),
                   'n_iterations': int(model.n_iter_[0])}
        cache[key] = metrics
        return metrics.copy()

    all_names = list(SPECS)
    baseline = fit(all_names)
    comparisons = [{'scheme': 'full_11', **baseline}]
    for cutoff in [.02, .1]:
        names = [name for name in all_names if summary.set_index('feature').loc[name, 'iv'] >= cutoff]
        if names:
            comparisons.append({'scheme': f'iv_ge_{cutoff}', **fit(names)})
    best_late = summary.loc[summary.feature.isin(LATE_COLUMNS)].iloc[0].feature
    names = [name for name in all_names if name not in LATE_COLUMNS or name == best_late]
    comparisons.append({'scheme': 'only_highest_iv_delinquency', **fit(names)})
    # 一次排序、逐项尝试，不进行所有子集搜索；每步须相对完整模型满足预定容差。
    selected = all_names.copy()
    attempts = []
    for name in summary.sort_values(['iv', 'feature']).feature:
        names = [v for v in selected if v != name]
        if not names:
            continue
        metrics = fit(names)
        auc_loss = baseline['auc'] - metrics['auc']
        loss_inc = metrics['log_loss'] - baseline['log_loss']
        brier_inc = metrics['brier'] - baseline['brier']
        accept = auc_loss <= TOLERANCES['auc_loss'] and loss_inc <= TOLERANCES['log_loss_increase'] and brier_inc <= TOLERANCES['brier_increase']
        attempts.append({'removed_feature': name, 'accepted_removal': accept,
                         'auc_loss_vs_full': auc_loss, 'log_loss_increase_vs_full': loss_inc,
                         'brier_increase_vs_full': brier_inc, **metrics})
        if accept:
            selected = names
    comparisons.append({'scheme': 'selected_candidates', **fit(selected)})
    decision = {'selected_features': selected, 'excluded_features': [v for v in all_names if v not in selected],
                'selection_order': summary.sort_values(['iv', 'feature']).feature.tolist(),
                'criterion': 'one_pass_ascending_training_IV_backward_elimination_with_fixed_validation_tolerances_vs_full',
                'tolerances': TOLERANCES, 'diagnostic_model': {'C': 1, 'penalty': 'l2', 'solver': 'lbfgs',
                    'tol': 1e-8, 'max_iter': 2000, 'class_weight': None, 'seed': SEED},
                'status': 'candidates_for_module4_not_final_test_validated_model',
                'test_labels_used': False, 'validation_used_for_selection': True,
                'tolerances_are_not_significance_or_noninferiority_tests': True}
    return pd.DataFrame(comparisons), pd.DataFrame(attempts), decision


def tex_bin(label: str) -> str:
    if label == 'MISSING':
        return '缺失'
    if label == '96/98':
        return '96/98（特殊）'
    return '$' + label.replace('>=', r'\geq ').replace('+inf', r'+\infty') + '$'


def write_tex(table: pd.DataFrame, summary: pd.DataFrame, comparisons: pd.DataFrame, rules: dict) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    for feature, stem in [('DebtRatio', 'debt_binning'), ('NumberOfTimes90DaysLate', 'late90_binning')]:
        t = table[table.feature.eq(feature)]
        lines = [r'\begin{tabular}{lrrrrrrrr}', r'\toprule',
                 r'档位 & 人数 & 好 & 坏 & 坏率 & $g_k$ & $b_k$ & WOE & IV 贡献 \\', r'\midrule']
        for v in t.itertuples(index=False):
            lines.append(f'{tex_bin(v.bin)} & {v.n:,} & {v.good:,} & {v.bad:,} & {v.bad_rate:.2%}'.replace('%', r'\%') +
                         f' & {v.g:.5f} & {v.b:.5f} & {v.woe:.4f} & {v.iv_contribution:.5f} ' + r'\\')
        lines += [r'\midrule', f'合计 & {rules["training_n"]:,} & {rules["training_good"]:,} & {rules["training_bad"]:,} & ' +
                  f'{rules["training_bad"]/rules["training_n"]:.2%}'.replace('%', r'\%') +
                  f' & 1 & 1 & --- & {t.iv_contribution.sum():.5f} ' + r'\\', r'\bottomrule', r'\end{tabular}']
        (TABLES / f'{stem}.tex').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    lines = [r'\begin{tabular}{lrrrrl}', r'\toprule',
             r'候选变量 & 档数 & 最小档人数 & IV & 特殊档 IV 占比 & 初选 \\', r'\midrule']
    for v in summary.itertuples(index=False):
        share = f'{v.special_iv_share:.1%}'.replace('%', r'\%') if v.special_bins else '---'
        lines.append(f'{v.feature_cn} & {v.K} & {v.min_n:,} & {v.iv:.4f} & {share} & ' + ('保留' if v.selected else '暂不纳入') + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (TABLES / 'iv_summary.tex').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    display = {'full_11': '全部 11 项', 'iv_ge_0.02': r'仅 IV $\geq0.02$', 'iv_ge_0.1': r'仅 IV $\geq0.10$',
               'only_highest_iv_delinquency': '逾期仅留最高 IV 项', 'selected_candidates': '综合初选方案'}
    lines = [r'\begin{tabular}{lrrrrr}', r'\toprule',
             r'诊断方案 & 变量数 & AUC & KS & 对数损失 & Brier \\', r'\midrule']
    for v in comparisons.itertuples(index=False):
        lines.append(f'{display[v.scheme]} & {v.n_features} & {v.auc:.4f} & {v.ks:.4f} & {v.log_loss:.5f} & {v.brier:.5f} ' + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (TABLES / 'candidate_validation.tex').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def plot_woe(table: pd.DataFrame) -> dict:
    """两个独立点图展示冻结档位 WOE，特殊编码不与数值档连线。"""
    # 固定图形元数据时间与 SVG 标识，保证复算后资源绑定哈希稳定。
    os.environ['SOURCE_DATE_EPOCH'] = '0'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    import pymupdf
    sys.path.insert(0, str(ROOT / 'utils'))
    from setup_style import setup_style
    from export_figure import export_figure
    from visual_qa import audit_layout
    setup_style(journal='general', lang='zh', serif_for_zh=True, use_sciplots=False, constrained_layout=False)
    plt.rcParams.update({'font.serif': ['SimSun', 'Times New Roman'], 'font.size': 8, 'axes.labelsize': 8,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7, 'savefig.bbox': None,
                         'svg.hashsalt': 'module3-woe'})
    size = (16 / 2.54, 6.3 / 2.54)
    fig, axes = plt.subplots(1, 2, figsize=size)
    fig.subplots_adjust(left=.14, right=.98, bottom=.23, top=.84, wspace=.60)
    for ax, feature, title in zip(axes, ['DebtRatio', 'NumberOfTimes90DaysLate'], ['(a) 负债收入比', '(b) 90 天及以上逾期次数']):
        t = table[table.feature.eq(feature)]
        x = np.arange(len(t))
        normal = ~t.special.to_numpy()
        ax.scatter(t.woe[normal], x[normal], color='#0072B2', s=22, marker='o', zorder=3)
        ax.scatter(t.woe[~normal], x[~normal], color='#D55E00', s=25, marker='s', zorder=3)
        ax.axvline(0, color='#777777', linewidth=.65, linestyle='--')
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_title(title, loc='left', fontsize=9, pad=8)
        ax.set_xlabel('WOE（好 / 坏）')
        ax.set_yticks(x, t['bin'].str.replace('+inf', '∞', regex=False).str.replace('>=', '≥', regex=False))
        ax.set_xlim(-3.5, 1)
        ax.set_xticks([-3, -2, -1, 0, 1])
        ax.set_ylim(len(t)-.5, -.5)
        ax.tick_params(length=2, width=.5)
    issues = audit_layout(fig)
    if issues:
        raise RuntimeError(f'图形布局检查未通过：{issues}')
    FIGURES.mkdir(parents=True, exist_ok=True)
    base = FIGURES / 'woe_debt_late90'
    paths = export_figure(fig, str(base), formats=['pdf', 'svg', 'png'], size_inches=size, dpi=300, tight=False)
    with Image.open(base.with_suffix('.png')) as im:
        im.convert('L').save(FIGURES / 'woe_debt_late90_grayscale.png', dpi=(300, 300))
    with pymupdf.open(base.with_suffix('.pdf')) as pdf:
        for font in pdf[0].get_fonts(full=True):
            if not pdf.extract_font(font[0])[3] or font[2] == 'Type3':
                raise RuntimeError('图形字体未正确嵌入')
    plt.close(fig)
    shutil.copy2(base.with_suffix('.pdf'), ROOT / 'latex/figures/woe_debt_late90.pdf')
    return {'size_cm': [16, 6.3], 'dpi': 300, 'layout_issues': issues,
            'outputs': {str(Path(p).relative_to(ROOT)): sha256(Path(p)) for p in paths}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stats-only', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.smoke:
        import subprocess
        subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(ROOT/'tests'), '-p', 'test_binning_woe.py', '-v'], check=True, cwd=ROOT)
        return
    train, val, inputs = load_training_and_validation()
    rules, table = fit_woe(train.drop(columns=['row_id', TARGET]), train[TARGET])
    train_w, train_b = transform_woe(train.drop(columns=['row_id', TARGET]), rules)
    val_w, val_b = transform_woe(val.drop(columns=['row_id', TARGET]), rules)
    rows = []
    for feature, t in table.groupby('feature', sort=False):
        iv = float(t.iv_contribution.sum())
        special_iv = float(t.loc[t.special, 'iv_contribution'].sum())
        rows.append({'feature': feature, 'feature_cn': NAMES[feature], 'K': len(t), 'initial_K': len(t),
                     'min_n': int(t.n.min()), 'min_ordinary_n': int(t.loc[~t.special, 'n'].min()),
                     'min_good': int(t.good.min()), 'min_bad': int(t.bad.min()),
                     'iv': iv, 'special_bins': int(t.special.sum()), 'special_iv': special_iv,
                     'special_iv_share': special_iv/iv if iv else 0,
                     'iv_level': 'very_weak' if iv < .02 else 'weak' if iv < .1 else 'medium' if iv < .3 else 'strong',
                     'high_iv_review': iv >= .5})
    summary = pd.DataFrame(rows).sort_values(['iv', 'feature'], ascending=[False, True]).reset_index(drop=True)
    comparisons, attempts, decision = evaluate_candidates(train_w, train[TARGET], val_w, val[TARGET], summary)
    summary['selected'] = summary.feature.isin(decision['selected_features'])
    rules['candidate_selection'] = decision
    rules['inputs'] = inputs
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT/'binning_table.csv', index=False, encoding='utf-8-sig')
    summary.to_csv(OUT/'iv_summary.csv', index=False, encoding='utf-8-sig')
    comparisons.to_csv(OUT/'candidate_validation.csv', index=False, encoding='utf-8-sig')
    attempts.to_csv(OUT/'selection_trials.csv', index=False, encoding='utf-8-sig')
    train_w.corr().to_csv(OUT/'woe_correlation.csv', encoding='utf-8-sig')
    for split, frame, w, b in [('train', train, train_w, train_b), ('validation', val, val_w, val_b)]:
        pd.concat([frame[['row_id', TARGET]], w.add_suffix('_woe')], axis=1).to_csv(OUT/f'{split}_woe.csv', index=False, encoding='utf-8-sig')
        pd.concat([frame[['row_id']], b], axis=1).to_csv(OUT/f'{split}_bins.csv', index=False, encoding='utf-8-sig')
    (OUT/'woe_rules.json').write_text(json.dumps(rules, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    (OUT/'candidate_selection.json').write_text(json.dumps(decision, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    write_tex(table, summary, comparisons, rules)
    contract = {'claim': 'DebtRatio保留非单调区分信息；严重逾期普通档WOE随次数下降，96/98独立映射。',
                'source': 'training binning_table.csv only', 'figure': 'two aligned categorical point panels, no connecting lines',
                'quantity': 'smoothed WOE point estimates; no confidence intervals', 'special_marker': 'square',
                'size_cm': [16, 6.3], 'dpi': 300, 'scope_override': 'user requests module3 only, not a full-paper figure quota or Word document'}
    (OUT/'figure_contract.json').write_text(json.dumps(contract, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    manifest = {'command': '.venv/Scripts/python.exe scripts/binning_woe.py',
                'python_executable': sys.executable,
                'inputs': inputs, 'seed': SEED, 'alpha': ALPHA, 'python': platform.python_version(),
                'packages': {n: importlib.metadata.version(n) for n in ['numpy','pandas','scikit-learn','matplotlib','PyMuPDF','Pillow']},
                'code_sha256': sha256(Path(__file__)), 'new_random_sampling': False,
                'test_data_opened': False, 'test_labels_used': False,
                'training_n': len(train), 'validation_n': len(val), 'constraints': {'ordinary_min_n': 450, 'min_good_and_bad': 30},
                'selection': decision}
    if not args.stats_only:
        manifest['figure'] = plot_woe(table)
    for entry in inputs.values():
        if sha256(ROOT / entry['path']) != entry['sha256']:
            raise RuntimeError('运行期间输入发生变化')
    paths = [p for p in OUT.iterdir() if p.is_file() and p.name not in ['module3_manifest.json','training_profile.json']]
    paths += [TABLES / f'{s}.tex' for s in ['debt_binning','late90_binning','iv_summary','candidate_validation']]
    manifest['outputs'] = {str(p.relative_to(ROOT)): sha256(p) for p in paths}
    (OUT/'module3_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(summary.to_string(index=False))
    print(comparisons.drop(columns='features').to_string(index=False))
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
