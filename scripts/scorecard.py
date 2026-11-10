"""模块五：固定评分刻度、完整评分卡、三户核验与无标签新客户评分。

复现：.venv/Scripts/python.exe scripts/scorecard.py
小样本检查：.venv/Scripts/python.exe scripts/scorecard.py --smoke
新客户：.venv/Scripts/python.exe scripts/scorecard.py --customer-json results/module5/example_customer.json
程序只读取冻结模型及映射，不拟合、不调参、不改变前序数据。
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from audit_data import TARGET, sha256
from binning_woe import NAMES, tex_bin, transform_woe
from preprocess import FEATURES, transform as preprocess_transform

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/module5'
TABLES = ROOT / 'latex/tables'
PDO, BASE_SCORE, BASE_ODDS = 20., 600., 19.
B = PDO / np.log(2.)
A = BASE_SCORE - B*np.log(BASE_ODDS)
QUANTILES = [('低风险', .10), ('中风险', .50), ('高风险', .90)]


def dump(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def freeze(path: Path, value: dict) -> None:
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != value:
        raise ValueError(f'已有冻结评分规则变化：{path}，请先核查模型与刻度。')
    dump(path, value)


def probability_to_score(p):
    """坏样本概率到未取整分数；p须严格在(0,1)，采用稳定对数写法。"""
    p = np.asarray(p, dtype=float)
    if not np.isfinite(p).all() or np.any((p<=0)|(p>=1)):
        raise ValueError('概率必须为(0,1)内的有限数')
    return A + B*(np.log1p(-p)-np.log(p))


def score_to_probability(score):
    """分数反算坏样本概率；高分对应低风险。"""
    score = np.asarray(score, dtype=float)
    if not np.isfinite(score).all():
        raise ValueError('分数必须为有限数')
    return expit((A-score)/B)


def round_points(points):
    """最近整数，恰好半分时远离0；负数与正数对称处理，不用银行家舍入。"""
    x = np.asarray(points, dtype=float)
    if not np.isfinite(x).all() or np.any(np.abs(x)>np.iinfo(np.int64).max-2048):
        raise ValueError('分数非法或超过整数表示范围')
    result = (np.sign(x)*np.floor(np.abs(x)+.5)).astype(np.int64)
    return int(result) if result.ndim==0 else result


class CreditScorecard:
    """由冻结字典构造；不需要训练数据、客户标签或拟合方法。"""

    def __init__(self, model_bundle: dict, preprocess_rules: dict, woe_rules: dict):
        self.bundle, self.preprocess_rules, self.woe_rules = model_bundle, preprocess_rules, woe_rules
        self.features = list(model_bundle['selected_features'])
        self.model = model_bundle['models']['woe_logistic']
        if self.model['columns'] != self.features or len(self.features)!=8:
            raise ValueError('评分模型列顺序或候选变量数不一致')
        self.beta = np.array([self.model['coefficients'][f] for f in self.features])
        if not np.isfinite(self.beta).all() or not np.isfinite(self.model['intercept']):
            raise ValueError('系数必须为有限数')
        self.intercept_score = float(A-B*self.model['intercept'])
        self.intercept_score_rounded = round_points(self.intercept_score)
        rows = [{'kind':'intercept','feature':'intercept','feature_cn':'截距基准分','bin_id':-1,'bin':'BASE',
                 'woe':np.nan,'coefficient':self.model['intercept'],
                 'points_unrounded':self.intercept_score,'points_rounded':self.intercept_score_rounded,
                 'intercept_score':self.intercept_score}]
        self.point_maps, self.rounded_maps = {}, {}
        for f,beta in zip(self.features,self.beta):
            mapping = woe_rules['mappings'][f]
            if len({r['bin_id'] for r in mapping}) != len(mapping) or len(mapping)!=len(woe_rules['specs'][f]['labels']):
                raise ValueError(f'{f}: 档位映射不完整或重复')
            self.point_maps[f], self.rounded_maps[f] = {}, {}
            for r in mapping:
                if r['training_n']<=0 or not np.isfinite(r['woe']):
                    raise ValueError(f'{f}: 无训练样本或非法WOE，不能创建分值')
                point = float(-B*beta*r['woe'])
                rounded = round_points(point)
                self.point_maps[f][r['bin_id']] = point
                self.rounded_maps[f][r['bin_id']] = rounded
                rows.append({'kind':'feature_bin','feature':f,'feature_cn':NAMES[f],
                             'bin_id':r['bin_id'],'bin':r['bin'],'woe':r['woe'],'coefficient':float(beta),
                             'points_unrounded':point,'points_rounded':rounded,
                             'intercept_score':self.intercept_score})
        self.card = pd.DataFrame(rows)

    @classmethod
    def from_project(cls, root: Path = ROOT):
        """只需模型、预处理、WOE和小型清单，不读取训练或测试CSV。"""
        root = Path(root)
        manifest_path = root/'results/module4/module4_manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        bundle_path = root/'results/module4/model_bundle.json'
        expected = manifest['outputs'][str(bundle_path.relative_to(root))]
        if sha256(bundle_path)!=expected:
            raise ValueError('Logistic模型哈希不匹配')
        bundle = json.loads(bundle_path.read_text(encoding='utf-8'))
        inputs = {'module4_manifest':{'path':str(manifest_path.relative_to(root)),'sha256':sha256(manifest_path)},
                  'model_bundle':{'path':str(bundle_path.relative_to(root)),'sha256':expected}}
        for key,path,expected in [
            ('model_config',root/'results/module4/model_config.json',bundle['model_config_sha256']),
            ('preprocess_rules',root/bundle['preprocess_rules_path'],bundle['preprocess_rules_sha256']),
            ('woe_rules',root/bundle['woe_rules_path'],bundle['woe_rules_sha256'])]:
            if sha256(path)!=expected:
                raise ValueError(f'{key}: 冻结规则哈希不匹配')
            inputs[key]={'path':str(path.relative_to(root)),'sha256':expected}
        prep = json.loads((root/bundle['preprocess_rules_path']).read_text(encoding='utf-8'))
        woe = json.loads((root/bundle['woe_rules_path']).read_text(encoding='utf-8'))
        return cls(bundle,prep,woe),inputs

    def score_frame(self, raw_X: pd.DataFrame, details: bool = False) -> tuple[pd.DataFrame,pd.DataFrame | None]:
        """原始十特征 -> 冻结预处理 -> 冻结档位/WOE -> 查分值并返回概率与总分。"""
        processed,_ = preprocess_transform(raw_X,self.preprocess_rules)
        w,bins = transform_woe(processed,self.woe_rules)
        eta = self.model['intercept'] + w[self.features].to_numpy()@self.beta
        points = pd.DataFrame({f:bins[f].map(self.point_maps[f]) for f in self.features},index=raw_X.index)
        rounded = pd.DataFrame({f:bins[f].map(self.rounded_maps[f]) for f in self.features},index=raw_X.index)
        if points.isna().any().any() or rounded.isna().any().any():
            raise ValueError('评分卡没有对应档位')
        score_sum = self.intercept_score + points.sum(axis=1).to_numpy()
        # 以等价仿射公式作为决策分数的统一浮点口径，另保留逐项求和以检查数值误差。
        exact = A-B*eta
        np.testing.assert_allclose(score_sum,exact,atol=5e-12,rtol=0)
        p = expit(eta)
        from_score = score_to_probability(exact)
        np.testing.assert_allclose(from_score,p,atol=3e-15,rtol=1e-13)
        integer = self.intercept_score_rounded + rounded.sum(axis=1).to_numpy(dtype=np.int64)
        summary = pd.DataFrame({'probability':p,'score_unrounded':exact,'score_component_sum':score_sum,
                                'score_rounded_sum':integer,'score_rounded_once':round_points(exact),
                                'rounding_error':integer-exact,'probability_from_score':from_score,
                                'probability_from_rounded_score':score_to_probability(integer)},index=raw_X.index)
        if not details:
            return summary,None
        records=[]
        for pos in range(len(raw_X)):
            for f in self.features:
                raw = raw_X.iloc[pos][f]
                k = int(bins.iloc[pos][f])
                records.append({'row_position':pos,'feature':f,'feature_cn':NAMES[f],
                                'raw_value':None if pd.isna(raw) else float(raw),
                                'prepared_value':float(processed.iloc[pos][f]),
                                'bin_id':k,'bin':self.woe_rules['specs'][f]['labels'][k],
                                'woe':float(w.iloc[pos][f]),'coefficient':self.model['coefficients'][f],
                                'points_unrounded':float(points.iloc[pos][f]),
                                'points_rounded':int(rounded.iloc[pos][f])})
        return summary,pd.DataFrame(records)

    def score_customer(self, customer: dict) -> dict:
        if not isinstance(customer,dict):
            raise ValueError('客户输入必须为十个原始特征的字典')
        summary,detail = self.score_frame(pd.DataFrame([customer]),details=True)
        result = summary.iloc[0].to_dict()
        for k in ['score_rounded_sum','score_rounded_once']:
            result[k]=int(result[k])
        result['intercept_score']=self.intercept_score
        result['intercept_score_rounded']=self.intercept_score_rounded
        # pandas 对 None 数值列可能转成 NaN；用 JSON 规范的 null 返回原始缺失。
        result['breakdown']=detail.astype(object).where(pd.notna(detail),None).to_dict(orient='records')
        return result


def score_customer(customer: dict, project_root: Path = ROOT) -> dict:
    """公开便捷入口：输入一个无标签客户，返回坏样本概率、总分与逐项分解。"""
    scorer,_ = CreditScorecard.from_project(project_root)
    return scorer.score_customer(customer)


def format_raw(value) -> str:
    if pd.isna(value):
        return '缺失'
    return str(int(value)) if float(value).is_integer() else f'{value:.6g}'


def card_bin(label: str) -> str:
    # 整数范围用连接号，避免在数学模式中被读成减法。
    if '-' in label and label.replace('-','').isdigit():
        left,right=label.split('-')
        return f'${left}$--${right}$'
    return tex_bin(label)


def write_tex(scorer: CreditScorecard, cases: pd.DataFrame, details: pd.DataFrame, checks: dict) -> None:
    TABLES.mkdir(parents=True,exist_ok=True)
    # 全部43档和独立基准分按变量分组并排，组名在各组上方，系数仍逐档完整展示。
    panels=[['intercept',scorer.features[0],'age',scorer.features[2],'NumberRealEstateLoansOrLines'],
            ['DebtRatio','NumberOfTimes90DaysLate','NumberOfTime60-89DaysPastDueNotWorse','NumberOfDependents']]
    if set(sum(panels,[]))!=set(scorer.card.feature.unique()) or sum(map(len,panels))!=9:
        raise ValueError('完整评分卡分组未覆盖全部8变量及基准分')
    lines=[]
    for panel_index,features in enumerate(panels):
        if panel_index:
            lines.append(r'\hfill')
        lines += [r'\begin{minipage}[t]{0.49\linewidth}',r'\vspace{0pt}',r'\centering',r'\begin{tabular}{lrrrr}',r'\toprule',
                  r'档位 & WOE & 系数 & 分值 & 整数 \\',r'\midrule']
        for feature in features:
            part=scorer.card[scorer.card.feature.eq(feature)]
            lines.append(r'\multicolumn{5}{l}{\textbf{'+part.feature_cn.iloc[0]+r'}}\\')
            for r in part.itertuples(index=False):
                band='基准分' if r.kind=='intercept' else card_bin(r.bin)
                woe='---' if r.kind=='intercept' else f'{r.woe:.5f}'
                lines.append(f'{band} & {woe} & {r.coefficient:.5f} & {r.points_unrounded:.3f} & {r.points_rounded} '+r'\\')
            lines.append(r'\addlinespace[1pt]')
        lines += [r'\bottomrule',r'\end{tabular}',r'\end{minipage}%']
    (TABLES/'scorecard_complete.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    lines=[r'\begin{tabular}{llrlrr}',r'\toprule',
           r'客户 & 变量 & 原始值 & 档位 & WOE & 未取整分值 \\',r'\midrule']
    for label in cases.risk_group:
        part=details[details.risk_group.eq(label)]
        for i,r in enumerate(part.itertuples(index=False)):
            lines.append(f'{label if i==0 else ""} & {r.feature_cn} & {format_raw(r.raw_value)} & {card_bin(r.bin)} & {r.woe:.5f} & {r.points_unrounded:.3f} '+r'\\')
        if label!=cases.risk_group.iloc[-1]:
            lines.append(r'\midrule')
    lines += [r'\bottomrule',r'\end{tabular}']
    (TABLES/'scorecard_cases_detail.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    lines=[r'\begin{tabular}{lrrrrrr}',r'\toprule',
           r'客户（行号） & 原模型 $p$ & 分数反算 $p$ & 未取整总分 & 整数总分 & 分数误差 & 整数反算 $p$ \\',r'\midrule']
    for r in cases.itertuples(index=False):
        lines.append(f'{r.risk_group}（{r.row_id}） & {r.probability_model4:.4%} & {r.probability_from_score:.4%} & '
                     f'{r.score_unrounded:.3f} & {r.score_rounded_sum} & {r.rounding_error:+.3f} & {r.probability_from_rounded_score:.4%}'.replace('%',r'\%')+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (TABLES/'scorecard_cases_summary.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    macros={'ScoreA':f'{A:.6f}','ScoreB':f'{B:.6f}','ScoreIntercept':f'{scorer.intercept_score:.6f}',
            'ScoreMaxRoundError':f'{checks["test_max_abs_rounding_error"]:.3f}',
            'ScoreMaxProbabilityError':f'{checks["max_probability_error"]:.2e}',
            'ScoreCardBins':str(len(scorer.card)-1)}
    (TABLES/'scorecard_macros.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in macros.items())+'\n',encoding='utf-8')


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--customer-json',type=Path)
    args=parser.parse_args()
    if args.smoke:
        subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_scorecard.py','-v'],cwd=ROOT,check=True)
        return
    scorer,inputs=CreditScorecard.from_project()
    if args.customer_json:
        customer=json.loads(args.customer_json.read_text(encoding='utf-8-sig'))
        print(json.dumps(scorer.score_customer(customer),ensure_ascii=False,indent=2,allow_nan=False))
        return
    m4=json.loads((ROOT/'results/module4/module4_manifest.json').read_text(encoding='utf-8'))
    lineage_notes=[]
    for name,entry in m4['inputs'].items():
        actual=sha256(ROOT/entry['path'])
        if actual!=entry['sha256']:
            if name!='dictionary':
                raise ValueError(f'上游输入变化：{name}')
            # 字典不是数值评分输入。若仅文件字节变化，须与模块二保存的逐字段内容完全一致，
            # 同时保留原哈希和现哈希；不改旧清单，也不据此重新拟合任何参数。
            reference=ROOT/'results/module2/data_dictionary.csv'
            current=pd.read_excel(ROOT/entry['path'],header=1)
            current.columns=current.columns.str.strip()
            prior=pd.read_csv(reference)
            pd.testing.assert_frame_equal(current.reset_index(drop=True),prior.reset_index(drop=True))
            lineage_notes.append({'asset':entry['path'],'previous_sha256':entry['sha256'],
                                  'current_sha256':actual,'semantic_comparison':'all names, definitions, types identical',
                                  'numerical_scoring_input':False})
            inputs['dictionary_semantic_reference']={'path':str(reference.relative_to(ROOT)),'sha256':sha256(reference)}
            entry={**entry,'sha256':actual}
        inputs['module4_'+name]=entry
    raw_path=ROOT/inputs['module4_raw']['path']
    raw=pd.read_csv(raw_path,usecols=lambda c:c!=TARGET)
    raw=raw.rename(columns={raw.columns[0]:'row_id'}).set_index('row_id')
    if not raw.index.is_unique:
        raise ValueError('原始行号重复')
    OUT.mkdir(parents=True,exist_ok=True)
    dump(OUT/'input_lineage_check.json',{'notes':lineage_notes,'old_files_overwritten':False})
    scaling={'schema_version':1,'base_score':BASE_SCORE,'base_good_bad_odds':BASE_ODDS,'PDO':PDO,
             'A':float(A),'B':float(B),'direction':'higher_score_lower_bad_probability',
             'formula':'S=A-B*logit(p)=S0+sum(-B*beta_j*WOE_jk)',
             'rounding':'nearest integer; ties away from zero; rounded intercept plus rounded bin contributions',
             'decision_score':'score_unrounded only','score_clipping':False}
    freeze(OUT/'scaling.json',scaling)
    scorer.card.to_csv(OUT/'scorecard.csv',index=False,encoding='utf-8-sig')
    rules={'scaling':scaling,'intercept_score':scorer.intercept_score,
           'intercept_score_rounded':scorer.intercept_score_rounded,
           'points':{f:[{'bin_id':int(r.bin_id),'bin':r.bin,'woe':r.woe,'coefficient':r.coefficient,
                         'points_unrounded':r.points_unrounded,'points_rounded':int(r.points_rounded)}
                        for r in scorer.card[scorer.card.feature.eq(f)].itertuples(index=False)] for f in scorer.features},
           'frozen_model_inputs':{k:v for k,v in inputs.items() if k in ['model_bundle','model_config','preprocess_rules','woe_rules']}}
    freeze(OUT/'scorecard_rules.json',rules)
    all_checks=[]
    scored_test=None
    for split,n in [('train',90000),('validation',30000),('test',30000)]:
        path=ROOT/f'results/module4/{split}_predictions.csv'
        expected=m4['outputs'][str(path.relative_to(ROOT))]
        if sha256(path)!=expected:
            raise ValueError(f'{split}: 冻结概率文件变化')
        inputs[f'{split}_predictions']={'path':str(path.relative_to(ROOT)),'sha256':expected}
        previous=pd.read_csv(path)
        if len(previous)!=n or not previous.row_id.is_unique:
            raise ValueError(f'{split}: 记录数或行号不符')
        X=raw.loc[previous.row_id,FEATURES].reset_index(drop=True)
        scores,_=scorer.score_frame(X)
        np.testing.assert_allclose(scores.probability,previous.woe_logistic,atol=2e-14,rtol=1e-13)
        np.testing.assert_allclose(scores.probability_from_score,previous.woe_logistic,atol=2e-14,rtol=1e-13)
        result=pd.concat([previous[['row_id',TARGET]],scores],axis=1)
        result['probability_model4']=previous.woe_logistic
        result.to_csv(OUT/f'{split}_scores.csv',index=False,encoding='utf-8-sig')
        sorted_score=result.sort_values(['probability_model4','row_id']).score_unrounded.to_numpy()
        if np.any(np.diff(sorted_score)>1e-10):
            raise AssertionError('未取整分数的风险排序方向异常')
        all_checks.append({'split':split,'n':n,
                           'max_probability_error':float(np.abs(scores.probability_from_score-previous.woe_logistic).max()),
                           'max_component_sum_error':float(np.abs(scores.score_component_sum-scores.score_unrounded).max()),
                           'max_abs_rounding_error':float(np.abs(scores.rounding_error).max()),
                           'rank_order_check':'pass'})
        if split=='test':
            scored_test=result
    ordered=scored_test.sort_values(['probability_model4','row_id']).reset_index(drop=True)
    case_rows=[]
    for label,q in QUANTILES:
        pos=int(np.floor(q*(len(ordered)-1)))
        r=ordered.iloc[pos].to_dict()
        r.update({'risk_group':label,'target_quantile':q,'sorted_rank_1based':pos+1})
        case_rows.append(r)
    cases=pd.DataFrame(case_rows)
    for column in ['row_id',TARGET,'score_rounded_sum','score_rounded_once','sorted_rank_1based']:
        cases[column]=cases[column].astype('int64')
    X=raw.loc[cases.row_id,FEATURES].reset_index(drop=True)
    _,details=scorer.score_frame(X,details=True)
    details['risk_group']=details.row_position.map(dict(enumerate(cases.risk_group)))
    details['row_id']=details.row_position.map(dict(enumerate(cases.row_id)))
    cases.to_csv(OUT/'case_summary.csv',index=False,encoding='utf-8-sig')
    details.to_csv(OUT/'case_details.csv',index=False,encoding='utf-8-sig')
    example={f:None if pd.isna(v) else float(v) for f,v in X.iloc[1].to_dict().items()}
    dump(OUT/'example_customer.json',example)
    dump(OUT/'example_customer_score.json',scorer.score_customer(example))
    checks={'subsets':all_checks,'max_probability_error':max(c['max_probability_error'] for c in all_checks),
            'max_component_sum_error':max(c['max_component_sum_error'] for c in all_checks),
            'test_max_abs_rounding_error':all_checks[-1]['max_abs_rounding_error'],
            'theoretical_abs_component_rounding_error_bound':.5*(len(scorer.features)+1),
            'case_selection':'test predicted bad probability ranks 10%,50%,90%; floor(q*(n-1)); ties by row_id; no label selection',
            'anchor_probability':1/(1+BASE_ODDS),'anchor_score':float(probability_to_score(1/(1+BASE_ODDS))),
            'doubled_odds_score':float(probability_to_score(1/(1+2*BASE_ODDS)))}
    dump(OUT/'verification.json',checks)
    write_tex(scorer,cases,details,checks)
    for entry in inputs.values():
        if sha256(ROOT/entry['path'])!=entry['sha256']:
            raise RuntimeError('评分期间冻结输入发生变化')
    files=[p for p in OUT.iterdir() if p.suffix in ['.csv','.json'] and p.name!='module5_manifest.json']
    files += [TABLES/f'{n}.tex' for n in ['scorecard_complete','scorecard_cases_detail','scorecard_cases_summary','scorecard_macros']]
    dump(OUT/'module5_manifest.json',{'command':'.venv/Scripts/python.exe scripts/scorecard.py',
            'python':platform.python_version(),'seed_from_frozen_split':20261103,'new_random_sampling':False,
            'model_refit':False,'target_used_for_case_selection':False,'inputs':inputs,'code_sha256':sha256(Path(__file__)),
            'helper_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in [ROOT/'scripts/preprocess.py',ROOT/'scripts/binning_woe.py',ROOT/'scripts/audit_data.py']},
            'outputs':{str(p.relative_to(ROOT)):sha256(p) for p in files},'verification':checks})
    print(json.dumps(scaling,ensure_ascii=False,indent=2))
    print(cases[['risk_group','row_id','probability_model4','score_unrounded','score_rounded_sum','rounding_error','probability_from_score']].to_string(index=False))
    print(json.dumps(checks,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
