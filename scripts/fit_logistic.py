"""模块四：八变量 WOE Logistic、原值 Logistic 与常数概率基准。

运行：.venv/Scripts/python.exe scripts/fit_logistic.py
最小检查：.venv/Scripts/python.exe scripts/fit_logistic.py --smoke
训练拟合 -> 固定参数及模型文件 -> 测试查表与三模型同样本评价。
本模块不搜索新变量、C、校准方法或决策阈值，不改前序数据与规则。
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             log_loss, roc_auc_score, roc_curve)

from audit_data import LATE_COLUMNS, TARGET, sha256
from binning_woe import NAMES, load_training_and_validation, transform_woe

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/module4'
TABLES = ROOT / 'latex/tables'
SEED = 20261103
FIT = {'C': 1.0, 'penalty': 'l2', 'solver': 'lbfgs', 'tol': 1e-8,
       'max_iter': 2000, 'class_weight': None, 'random_state': SEED}
LONG_TAIL = {'RevolvingUtilizationOfUnsecuredLines', 'DebtRatio',
             'NumberRealEstateLoansOrLines', 'NumberOfDependents', *LATE_COLUMNS}
MODEL_NAMES = {'woe_logistic': 'WOE Logistic', 'raw_logistic': '原值 Logistic', 'constant': '常数概率'}


def dump(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def freeze(path: Path, obj: dict) -> None:
    """已有冻结配置/参数改变时停止，避免在看过测试结果后静默改选。"""
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != obj:
        raise ValueError(f'冻结文件与本次结果不同：{path.name}。请核查，不能基于测试结果覆盖改选。')
    dump(path, obj)


def raw_design(X: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """同样八个原始字段的数值表示，缺失等须先经模块二；不使用标签。"""
    if not isinstance(X, pd.DataFrame) or X.columns.duplicated().any() or set(features)-set(X):
        raise ValueError('原值基准缺少所需字段或含重复字段')
    result = pd.DataFrame(index=X.index)
    special = pd.Series(False, index=X.index)
    for f in features:
        raw = X[f]
        x = pd.to_numeric(raw, errors='coerce').astype(float)
        if x.isna().any() or not np.isfinite(x).all() or x.lt(0).any():
            raise ValueError(f'{f}: 缺失、非法数值或负值')
        if f == 'age' and x.le(0).any():
            raise ValueError('age: 必须先应用训练中位数规则')
        if f not in ['RevolvingUtilizationOfUnsecuredLines', 'DebtRatio'] and x.mod(1).ne(0).any():
            raise ValueError(f'{f}: 非整数年龄或计数')
        if f in LATE_COLUMNS:
            mask = x.isin([96, 98])
            special |= mask
            x = x.mask(mask, 0)
        result[f] = np.log1p(x) if f in LONG_TAIL else x
    # 训练中三列特殊状态一致；仅用一个“任一逾期特殊”指示，避免三个完全重复的列。
    if any(f in LATE_COLUMNS for f in features):
        result['any_delinquency_special'] = special.astype(float)
    return result


def fit_raw_rules(train_X: pd.DataFrame, features: list[str]) -> dict:
    z = raw_design(train_X, features)
    means, scales = z.mean(), z.std(ddof=0)
    return {'input_features': features, 'design_columns': list(z),
            'mean': means.to_dict(), 'scale': scales.mask(scales.eq(0), 1).to_dict(),
            'log1p_features': [f for f in features if f in LONG_TAIL],
            'special_policy': 'per-field 96/98 -> zero in ordinary count; one shared any-special indicator',
            'constant_design_columns': scales.index[scales.eq(0)].tolist(),
            'statistics_scope': 'training_features_only', 'clip': False}


def transform_raw(X: pd.DataFrame, rules: dict) -> pd.DataFrame:
    z = raw_design(X, rules['input_features'])
    return ((z - pd.Series(rules['mean'])) / pd.Series(rules['scale']))[rules['design_columns']]


def fit_model(X: pd.DataFrame, y: pd.Series) -> tuple[dict, dict]:
    if not y.index.equals(X.index) or not y.isin([0, 1]).all() or y.nunique()!=2:
        raise ValueError('标签未对齐或缺少某类样本')
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model = LogisticRegression(**FIT).fit(X, y)
    model_json = {'columns': list(X), 'intercept': float(model.intercept_[0]),
                  'coefficients': dict(zip(X.columns, model.coef_[0].tolist()))}
    p = predict_matrix(X, model_json)
    np.testing.assert_allclose(p, model.predict_proba(X)[:, 1], atol=2e-14, rtol=1e-13)
    beta = model.coef_[0]
    grad = np.r_[np.mean(p-y), X.to_numpy().T@(p-y)/len(y) + beta/(FIT['C']*len(y))]
    eta = model.decision_function(X)
    objective = float(np.sum(np.logaddexp(0, eta)-y.to_numpy()*eta) + beta@beta/(2*FIT['C']))
    if np.abs(grad).max() > 1e-6:
        raise RuntimeError('优化驻点残差超过容许值')
    diagnostics = {'n_iterations': int(model.n_iter_[0]), 'converged': True,
                   'mean_objective_gradient_inf_norm': float(np.abs(grad).max()),
                   'sum_negative_log_likelihood_plus_l2': objective,
                   'probability_min': float(p.min()), 'probability_max': float(p.max()),
                   'near_0_or_1_training_predictions': int(((p<1e-12)|(p>1-1e-12)).sum())}
    return model_json, diagnostics


def predict_matrix(X: pd.DataFrame, model: dict) -> np.ndarray:
    beta = np.array([model['coefficients'][f] for f in model['columns']])
    return expit(model['intercept'] + X[model['columns']].to_numpy()@beta)


def diagnose_design(X: pd.DataFrame, y: pd.Series) -> tuple[dict, pd.DataFrame]:
    """线性规划检查完全/准分离；VIF/秩/条件数检查重复信息。"""
    v = X.to_numpy(dtype=float)
    means, scales = v.mean(axis=0), v.std(axis=0)
    z = np.column_stack([np.ones(len(v)), (v-means)/np.where(scales>0,scales,1)])
    # 重复的同类约束不改变分离是否存在，因此去重后求 LP。
    a = np.unique((2*y.to_numpy()-1)[:, None]*z, axis=0)
    p = z.shape[1]
    # beta=u-v, u,v>=0, ||beta||_1<=1。
    constraints = np.vstack([np.c_[-a, a], np.ones((1, 2*p))])
    b = np.r_[np.zeros(len(a)), 1.]
    common = {'method': 'highs', 'options': {'time_limit': 60, 'primal_feasibility_tolerance': 1e-9,
                                           'dual_feasibility_tolerance': 1e-9}}
    quasi = linprog(np.r_[-a.mean(axis=0), a.mean(axis=0)], A_ub=constraints, b_ub=b, **common)
    # 另最大化所有记录的共同最小边距，区分完全分离与准分离。
    full_constraints = np.column_stack([constraints, np.r_[np.ones(len(a)), 0.]])
    complete = linprog(np.r_[np.zeros(2*p), -1.], A_ub=full_constraints, b_ub=b, **common)
    if not quasi.success or not complete.success:
        raise RuntimeError(f'分离检查未完成：{quasi.message}; {complete.message}')
    threshold = 1e-7
    q, margin = max(0., float(-quasi.fun)), max(0., float(-complete.fun))
    separation = 'complete' if margin>threshold else 'quasi' if q>threshold else 'not_detected'
    corr = np.atleast_2d(np.corrcoef(v, rowvar=False))
    if np.any(scales==0):
        raise ValueError('设计矩阵含训练常量列，请核查')
    if np.linalg.matrix_rank(corr) < len(corr):
        raise ValueError('设计矩阵存在完全共线，不能直接解释独立系数')
    vif = np.diag(np.linalg.solve(corr, np.eye(len(corr))))
    out = pd.DataFrame({'feature': list(X), 'vif': vif})
    result = {'separation_check': separation, 'lp_threshold': threshold,
              'maximum_minimum_margin': margin, 'maximum_mean_nonnegative_margin': q,
              'lp_status': [int(quasi.status), int(complete.status)], 'unique_signed_constraints': len(a),
              'standardized_design_rank': int(np.linalg.matrix_rank(z)), 'design_columns_including_intercept': p,
              'standardized_design_condition_number': float(np.linalg.cond(z)), 'maximum_vif': float(vif.max())}
    return result, out


def metrics(y: pd.Series, p: np.ndarray) -> dict:
    if len(y)!=len(p) or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('概率非法或样本未对齐')
    fpr,tpr,_ = roc_curve(y,p)
    tn,fp,fn,tp = confusion_matrix(y,p>=.5,labels=[0,1]).ravel()
    return {'n': len(y), 'bad': int(y.sum()), 'bad_rate': float(y.mean()),
            'auc': roc_auc_score(y,p), 'ks': float((tpr-fpr).max()),
            'ap': average_precision_score(y,p), 'brier': brier_score_loss(y,p), 'log_loss': log_loss(y,p),
            'accuracy_at_0_5': float((tn+tp)/len(y)), 'recall_at_0_5': float(tp/(tp+fn)),
            'precision_at_0_5': float(tp/(tp+fp)) if tp+fp else 0.,
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}


def prepare_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict, dict, dict, dict]:
    train,val,inputs = load_training_and_validation()
    m3 = json.loads((ROOT/'results/module3/module3_manifest.json').read_text(encoding='utf-8'))
    if sha256(ROOT/'scripts/binning_woe.py') != m3['code_sha256']:
        raise ValueError('模块三代码变化，须先核查冻结编码')
    for name in ['woe_rules.json','candidate_selection.json']:
        path = ROOT/'results/module3'/name
        expected = m3['outputs'][str(path.relative_to(ROOT))]
        if sha256(path)!=expected:
            raise ValueError(f'冻结输入不匹配：{name}')
        inputs[name] = {'path': str(path.relative_to(ROOT)), 'sha256': expected}
    woe_rules = json.loads((ROOT/inputs['woe_rules.json']['path']).read_text(encoding='utf-8'))
    selection = json.loads((ROOT/inputs['candidate_selection.json']['path']).read_text(encoding='utf-8'))
    if len(selection['selected_features'])!=8 or selection['diagnostic_model']['C']!=FIT['C']:
        raise ValueError('初选变量或固定C与既定方案不一致')
    prep_manifest = json.loads((ROOT/'results/module2/preprocess_manifest.json').read_text(encoding='utf-8'))
    for entry in inputs.values():
        if sha256(ROOT/entry['path'])!=entry['sha256']:
            raise ValueError('上游输入发生变化')
    return train,val,woe_rules,selection,prep_manifest,inputs


def examples(train: pd.DataFrame, woe_rules: dict, bundle: dict) -> tuple[pd.DataFrame, dict]:
    """按特征条件选真实训练参照户，不用标签；两种改变仅为模型情景比较。"""
    mask = train[LATE_COLUMNS].eq(0).all(axis=1) & train.DebtRatio.gt(0) & train.DebtRatio.le(.4)
    mask &= train.RevolvingUtilizationOfUnsecuredLines.gt(.1) & train.RevolvingUtilizationOfUnsecuredLines.le(.3)
    mask &= train.age.ge(40) & train.age.lt(50)
    ref = train.loc[mask].sort_values('row_id').iloc[[0]].copy()
    w,_ = transform_woe(ref.drop(columns=['row_id',TARGET]),woe_rules)
    model = bundle['models']['woe_logistic']
    base = w[bundle['selected_features']].copy()
    rows = []
    for feature,bin_a,bin_b,display in [('DebtRatio','(0,0.4]','(1,10]','负债收入比'),
                                      ('NumberOfTimes90DaysLate','0','1','90 天及以上逾期次数')]:
        lookup = {r['bin']:r['woe'] for r in woe_rules['mappings'][feature]}
        wa,wb = lookup[bin_a],lookup[bin_b]
        first,second = base.copy(),base.copy()
        first[feature],second[feature] = wa,wb
        pa,pb = float(predict_matrix(first,model)[0]),float(predict_matrix(second,model)[0])
        coef = model['coefficients'][feature]
        ratio = float(np.exp(coef*(wb-wa)))
        np.testing.assert_allclose((pb/(1-pb))/(pa/(1-pa)),ratio,atol=1e-12)
        rows.append({'feature': feature, 'feature_cn': display, 'bin_a':bin_a,'bin_b':bin_b,
                     'woe_a':wa,'woe_b':wb,'coefficient':coef,'odds_ratio':ratio,
                     'probability_a':pa,'probability_b':pb,'reference_row_id':int(ref.row_id.iloc[0])})
    reference = {'row_id': int(ref.row_id.iloc[0]), 'prepared_features': ref[bundle['selected_features']].iloc[0].to_dict(),
                 'woe':base.iloc[0].to_dict(), 'target_used_to_choose_reference':False,
                 'interpretation':'other seven features held fixed; conditional model comparison, not a causal effect'}
    return pd.DataFrame(rows),reference


def write_tex(coefs: pd.DataFrame, comparison: pd.DataFrame, pairs: pd.DataFrame, diagnostics: dict,
              bundle: dict) -> None:
    TABLES.mkdir(parents=True,exist_ok=True)
    lines = [r'\begin{tabular}{lrr}',r'\toprule',r'项 & 系数 $\widehat\beta$ & $\exp(\widehat\beta)$ \\',r'\midrule']
    for r in coefs[coefs.model.eq('woe_logistic')].itertuples(index=False):
        name = '截距' if r.feature=='intercept' else NAMES[r.feature]
        lines.append(f'{name} & {r.coefficient:.5f} & {np.exp(r.coefficient):.4f} '+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (TABLES/'logistic_coefficients.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    lines = [r'\begin{tabular}{lrrrrr}',r'\toprule',r'模型 & AUC & KS & AP & Brier & 对数损失 \\',r'\midrule']
    for r in comparison[comparison.split.eq('test')].itertuples(index=False):
        lines.append(f'{MODEL_NAMES[r.model]} & {r.auc:.4f} & {r.ks:.4f} & {r.ap:.4f} & {r.brier:.5f} & {r.log_loss:.5f} '+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (TABLES/'logistic_comparison.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    lines = [r'\begin{tabular}{llrrr}',r'\toprule',r'变量 & 档位变化 $a\to b$ & 条件几率比 & $p_a$ & $p_b$ \\',r'\midrule']
    for r in pairs.itertuples(index=False):
        change = r'$(0,0.4]\to(1,10]$' if r.feature=='DebtRatio' else r'$0\to1$'
        lines.append(f'{r.feature_cn} & {change} & {r.odds_ratio:.3f} & {r.probability_a:.2%} & {r.probability_b:.2%}'.replace('%',r'\%')+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (TABLES/'logistic_odds_examples.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    w = diagnostics['woe_logistic']; r = diagnostics['raw_logistic']
    macros = {'LogitWoeIterations':str(w['n_iterations']),'LogitRawIterations':str(r['n_iterations']),
              'LogitMaxVIF':f'{w["maximum_vif"]:.2f}', 'LogitCondition':f'{w["standardized_design_condition_number"]:.2f}',
              'LogitReferenceRow':str(int(pairs.reference_row_id.iloc[0])),
              'LogitTrainingRate':f'{bundle["constant_probability"]:.4%}'.replace('%',r'\%')}
    (TABLES/'logistic_macros.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in macros.items())+'\n',encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args()
    if args.smoke:
        subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_logistic.py','-v'],cwd=ROOT,check=True)
        return
    train,val,woe_rules,selection,prep_manifest,inputs = prepare_inputs()
    selected = selection['selected_features']
    plan = {'schema_version':1,'selected_features':selected,'logistic':FIT,
            'woe_input_scaling':'none; same representation as module3 candidate selection',
            'raw_transform':'age identity; other selected numeric log1p; 96/98 -> ordinary zero plus shared any-special; train mean/sd',
            'test_policy':'freeze both representations and all fitted parameters before opening prepared test file',
            'test_used_for_selection':False,'reweight_or_resample':False,
            'input_woe_rules_sha256':inputs['woe_rules.json']['sha256'],
            'input_selection_sha256':inputs['candidate_selection.json']['sha256']}
    OUT.mkdir(parents=True,exist_ok=True)
    freeze(OUT/'model_config.json',plan)
    train_w,_ = transform_woe(train.drop(columns=['row_id',TARGET]),woe_rules)
    val_w,_ = transform_woe(val.drop(columns=['row_id',TARGET]),woe_rules)
    train_w,val_w = train_w[selected],val_w[selected]
    raw_rules = fit_raw_rules(train,selected)
    train_r,val_r = transform_raw(train,raw_rules),transform_raw(val,raw_rules)
    models,diagnostics,vifs = {},{},[]
    for name,X in [('woe_logistic',train_w),('raw_logistic',train_r)]:
        models[name],fitdiag = fit_model(X,train[TARGET])
        structural,vif = diagnose_design(X,train[TARGET])
        diagnostics[name] = {**fitdiag,**structural}
        vif['model']=name;vifs.append(vif)
    bundle = {'schema_version':1,'selected_features':selected,'models':models,'raw_rules':raw_rules,
              'constant_probability':float(train[TARGET].mean()), 'training_n':len(train),
              'training_good':int(train[TARGET].eq(0).sum()),'training_bad':int(train[TARGET].sum()),
              'model_config_sha256':sha256(OUT/'model_config.json'),
              'preprocess_rules_path':'results/module2/preprocess_rules.json',
              'preprocess_rules_sha256':inputs['rules']['sha256'],
              'woe_rules_path':'results/module3/woe_rules.json','woe_rules_sha256':inputs['woe_rules.json']['sha256']}
    freeze(OUT/'model_bundle.json',bundle)
    frozen_bundle_hash = sha256(OUT/'model_bundle.json')
    # 只有上述两模型及常数先保存后，才打开预处理后的测试文件及其标签。
    test_path = ROOT/'results/module2/prepared/test.csv'
    expected = prep_manifest['outputs'][str(test_path.relative_to(ROOT))]
    if sha256(test_path)!=expected:
        raise ValueError('测试文件与冻结预处理结果不匹配')
    test = pd.read_csv(test_path)
    if len(test)!=30000 or not test.row_id.is_unique:
        raise ValueError('测试样本数或行号不符')
    inputs['test']={'path':str(test_path.relative_to(ROOT)),'sha256':expected}
    test_w,test_bins = transform_woe(test.drop(columns=['row_id',TARGET]),woe_rules)
    test_w = test_w[selected]
    test_r = transform_raw(test,raw_rules)
    rows=[]
    for split,frame,w,r in [('train',train,train_w,train_r),('validation',val,val_w,val_r),('test',test,test_w,test_r)]:
        predictions = frame[['row_id',TARGET]].copy()
        probs = {'woe_logistic':predict_matrix(w,models['woe_logistic']),
                 'raw_logistic':predict_matrix(r,models['raw_logistic']),
                 'constant':np.full(len(frame),bundle['constant_probability'])}
        for name,p in probs.items():
            predictions[name]=p
            rows.append({'split':split,'model':name,**metrics(frame[TARGET],p)})
        predictions.to_csv(OUT/f'{split}_predictions.csv',index=False,encoding='utf-8-sig')
    comparison=pd.DataFrame(rows)
    comparison.to_csv(OUT/'model_comparison.csv',index=False,encoding='utf-8-sig')
    coeffs=[]
    for name,model in models.items():
        coeffs.append({'model':name,'feature':'intercept','coefficient':model['intercept']})
        coeffs += [{'model':name,'feature':f,'coefficient':v} for f,v in model['coefficients'].items()]
    coeffs=pd.DataFrame(coeffs)
    coeffs.to_csv(OUT/'coefficients.csv',index=False,encoding='utf-8-sig')
    pd.concat(vifs,ignore_index=True).to_csv(OUT/'vif.csv',index=False,encoding='utf-8-sig')
    dump(OUT/'diagnostics.json',diagnostics)
    pairs,reference=examples(train,woe_rules,bundle)
    pairs.to_csv(OUT/'odds_examples.csv',index=False,encoding='utf-8-sig')
    dump(OUT/'reference_profile.json',reference)
    pd.concat([test[['row_id',TARGET]],test_w.add_suffix('_woe')],axis=1).to_csv(OUT/'test_woe.csv',index=False,encoding='utf-8-sig')
    pd.concat([test[['row_id']],test_bins[selected]],axis=1).to_csv(OUT/'test_bins.csv',index=False,encoding='utf-8-sig')
    write_tex(coeffs,comparison,pairs,diagnostics,bundle)
    if sha256(OUT/'model_bundle.json')!=frozen_bundle_hash:
        raise RuntimeError('评价后模型发生变化')
    for entry in inputs.values():
        if sha256(ROOT/entry['path'])!=entry['sha256']:
            raise RuntimeError('运行期间输入发生变化')
    files=[p for p in OUT.iterdir() if p.suffix in ['.csv','.json'] and p.name!='module4_manifest.json']
    files += [TABLES/f'{f}.tex' for f in ['logistic_coefficients','logistic_comparison','logistic_odds_examples','logistic_macros']]
    manifest={'command':'.venv/Scripts/python.exe scripts/fit_logistic.py','inputs':inputs,'seed':SEED,
              'code_sha256':sha256(Path(__file__)),'python':platform.python_version(),
              'packages':{n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn']},
              'new_random_sampling':False,'freeze_before_test_read':True,'frozen_model_bundle_sha256':frozen_bundle_hash,
              'test_used_for_fitting_or_selection':False,'module_scope':'module4 only, three comparison tables; no module6 curve suite',
              'outputs':{str(p.relative_to(ROOT)):sha256(p) for p in files}}
    dump(OUT/'module4_manifest.json',manifest)
    print(coeffs.to_string(index=False))
    print(comparison.to_string(index=False))
    print(pairs.to_string(index=False))
    print(json.dumps(diagnostics,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
