"""独立小样本核算、边界及无标签迁移检查。"""
import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from binning_woe import SPECS, assign_bins, fit_woe, transform_woe
from preprocess import FEATURES, DEPENDENTS_FLAG


def example():
    # 各普通档至少有100条好和100条坏，特殊档可稀疏。
    values = {
        FEATURES[0]: [0,.05,.2,.5,.9,2], 'age': [25,35,45,55,65,75],
        FEATURES[2]: [0,1,2,3,4,96], 'DebtRatio': [0,.2,.5,.8,2,20,200],
        'MonthlyIncome': [np.nan,0,100,5000,8000,20000],
        FEATURES[5]: [0,1,4,7,12,20], FEATURES[6]: [0,1,2,3,98],
        FEATURES[7]: [0,1,3,4], FEATURES[8]: [0,1,2,96],
        'NumberOfDependents': [0,0,1,2,3,4], DEPENDENTS_FLAG: [1,0,0,0,0,0],
    }
    X = pd.DataFrame({k: np.resize(v, 4200) for k,v in values.items()})
    y = pd.Series(np.arange(len(X)) // 7 % 2, index=X.index)
    return X,y


class BinningTests(unittest.TestCase):
    def test_hand_formula_and_normalization(self):
        X,y=example()
        rules,t=fit_woe(X,y)
        bins=assign_bins(X)
        for feature,part in t.groupby('feature'):
            self.assertAlmostEqual(part.g.sum(),1)
            self.assertAlmostEqual(part.b.sum(),1)
            self.assertEqual(part.n.sum(),len(X))
        row=t[t.feature.eq('DebtRatio')].iloc[0]
        mask=X.DebtRatio.eq(0)
        n1=int(y[mask].sum()); n0=int(mask.sum())-n1
        g=(n0+.5)/(int((y==0).sum())+.5*7)
        b=(n1+.5)/(int((y==1).sum())+.5*7)
        self.assertAlmostEqual(row.woe,math.log(g/b))
        self.assertAlmostEqual(row.iv_contribution,(g-b)*math.log(g/b))
        self.assertTrue((t.iv_contribution>=0).all())

    def test_income_boundaries_and_overlapping_specials(self):
        X,_=example(); X=X.iloc[:5].copy()
        X['MonthlyIncome']=[np.nan,0,9.99,10,1e20]
        X[FEATURES[6]]=[96,98,0,1,1e20]
        ids=assign_bins(X)
        self.assertEqual(ids.MonthlyIncome.tolist(),[0,1,1,2,5])
        self.assertEqual(ids[FEATURES[6]].tolist(),[4,4,0,1,3])

    def test_debt_closed_boundaries_and_tail(self):
        X,_=example(); X=X.iloc[:8].copy()
        X['DebtRatio']=[0,.4,.6,1,10,100,100.1,1e20]
        self.assertEqual(assign_bins(X).DebtRatio.tolist(),[0,1,2,3,4,5,6,6])

    def test_label_free_frozen_transform(self):
        X,y=example(); rules,_=fit_woe(X,y)
        restored=json.loads(json.dumps(rules))
        w,b=transform_woe(X,restored)
        self.assertEqual(w.shape,(len(X),11))
        self.assertTrue(np.isfinite(w.to_numpy()).all())
        altered=y.copy(); altered[:]=1-y
        w2,_=transform_woe(X,restored)
        pd.testing.assert_frame_equal(w,w2)
        for f in SPECS:
            self.assertEqual(w.loc[b[f].eq(0),f].nunique(),1)

    def test_empty_bin_and_invalid_inputs_rejected(self):
        X,y=example()
        with self.assertRaisesRegex(ValueError,'空训练档'):
            fit_woe(X.assign(DebtRatio=0),y)
        for column,value in [('DebtRatio',-1),('DebtRatio',np.inf),('age',0),
                             (FEATURES[6],2.5),('NumberOfDependents',np.nan),
                             ('MonthlyIncome','bad')]:
            bad=X.copy(); bad[column]=bad[column].astype(object); bad.loc[0,column]=value
            with self.subTest(column=column,value=value),self.assertRaises(ValueError):
                assign_bins(bad)
        with self.assertRaises(ValueError):
            fit_woe(X,y,alpha=0)
        with self.assertRaises(ValueError):
            fit_woe(X,y.rename(index=lambda i:i+1))

    def test_special_flags_and_income_state_consistency(self):
        X,_=example()
        X['MonthlyIncome_state']='REGULAR'
        with self.assertRaisesRegex(ValueError,'MonthlyIncome_state'):
            assign_bins(X)
        X=X.drop(columns='MonthlyIncome_state')
        X[f'{FEATURES[6]}_special']=0
        with self.assertRaisesRegex(ValueError,'special'):
            assign_bins(X)


if __name__=='__main__':
    unittest.main()
