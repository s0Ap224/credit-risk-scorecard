"""模块五刻度、逐项求和、无标签特殊值评分与整数规则检查。"""
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from scorecard import A,B,CreditScorecard,probability_to_score,score_to_probability,round_points
from preprocess import FEATURES,fit_rules,transform
from binning_woe import fit_woe
from test_binning_woe import example


class ScorecardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prepared,y=example()
        raw=prepared[FEATURES].copy()
        raw.loc[prepared.NumberOfDependents_missing.eq(1),'NumberOfDependents']=np.nan
        prep=fit_rules(raw)
        processed,_=transform(raw,prep)
        woe,_=fit_woe(processed,y)
        selected=[f for f in FEATURES if f not in ['MonthlyIncome','NumberOfOpenCreditLinesAndLoans']]
        bundle={'selected_features':selected,'models':{'woe_logistic':{'columns':selected,
                'intercept':-2.6,'coefficients':dict(zip(selected,[-.6,-.4,-.5,-.7,-.5,-.6,-.3,-.2]))}}}
        cls.scorer=CreditScorecard(bundle,prep,woe)
        cls.raw=raw

    def test_anchor_and_pdo(self):
        self.assertAlmostEqual(float(probability_to_score(.05)),600)
        self.assertAlmostEqual(float(probability_to_score(1/39)),620)
        self.assertGreater(B,0)
        self.assertAlmostEqual(A+B*np.log(19),600)

    def test_probability_roundtrip_and_direction(self):
        p=np.array([1e-12,.001,.05,.2,.9,1-1e-12])
        s=probability_to_score(p)
        np.testing.assert_allclose(score_to_probability(s),p,atol=1e-15)
        self.assertTrue((np.diff(s)<0).all())
        for bad in [0,1,-.1,np.nan,np.inf]:
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                probability_to_score(bad)

    def test_additive_card_and_frozen_json(self):
        X=self.raw.iloc[:12].copy()
        before=X.copy(deep=True)
        summary,detail=self.scorer.score_frame(X,details=True)
        pd.testing.assert_frame_equal(X,before)
        self.assertEqual(len(self.scorer.card),44)
        sums=detail.groupby('row_position').points_unrounded.sum().to_numpy()+self.scorer.intercept_score
        np.testing.assert_allclose(sums,summary.score_unrounded,atol=1e-12)
        self.assertTrue((np.abs(summary.rounding_error)<=4.5+1e-12).all())
        clone=CreditScorecard(json.loads(json.dumps(self.scorer.bundle)),
                             json.loads(json.dumps(self.scorer.preprocess_rules)),
                             json.loads(json.dumps(self.scorer.woe_rules)))
        pd.testing.assert_frame_equal(summary,clone.score_frame(X)[0])

    def test_half_away_rounding_and_component_sum(self):
        self.assertEqual(round_points(.5),1)
        self.assertEqual(round_points(-.5),-1)
        np.testing.assert_array_equal(round_points([1.49,1.5,-1.5,-1.49]),[1,2,-2,-1])
        summary,detail=self.scorer.score_frame(self.raw.iloc[:3],details=True)
        expected=detail.groupby('row_position').points_rounded.sum().to_numpy()+self.scorer.intercept_score_rounded
        np.testing.assert_array_equal(summary.score_rounded_sum,expected)

    def test_new_customer_special_missing_boundary_and_no_label(self):
        customer=self.raw.iloc[0].to_dict()
        customer.update({'NumberOfDependents':None,'MonthlyIncome':None,'age':0,
                         'NumberOfTimes90DaysLate':96,'DebtRatio':1e20})
        result=self.scorer.score_customer(customer)
        self.assertTrue(np.isfinite(result['score_unrounded']))
        rows={r['feature']:r for r in result['breakdown']}
        self.assertIsNone(rows['NumberOfDependents']['raw_value'])
        self.assertEqual(rows['NumberOfDependents']['prepared_value'],0)
        self.assertEqual(rows['NumberOfTimes90DaysLate']['bin'],'96/98')
        self.assertEqual(rows['DebtRatio']['bin'],'(100,+inf)')
        self.assertEqual(rows['age']['prepared_value'],self.scorer.preprocess_rules['age_median'])
        json.dumps(result,allow_nan=False)
        customer['SeriousDlqin2yrs']=0
        with self.assertRaises(ValueError):
            self.scorer.score_customer(customer)

    def test_unobserved_bin_and_illegal_raw_rejected(self):
        for column,value in [('DebtRatio',-1),('age',np.inf),('NumberOfTimes90DaysLate',1.5)]:
            customer=self.raw.iloc[1].to_dict();customer[column]=value
            with self.subTest(column=column),self.assertRaises(ValueError):
                self.scorer.score_customer(customer)
        woe=json.loads(json.dumps(self.scorer.woe_rules))
        woe['mappings']['DebtRatio'][0]['training_n']=0
        with self.assertRaises(ValueError):
            CreditScorecard(self.scorer.bundle,self.scorer.preprocess_rules,woe)


if __name__=='__main__':
    unittest.main()
