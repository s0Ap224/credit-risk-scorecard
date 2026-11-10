"""模块四：小样本完整链、分离诊断、冻结转换及几率比核算。"""
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fit_logistic import (diagnose_design,fit_model,fit_raw_rules,metrics,predict_matrix,raw_design,transform_raw)
from audit_data import LATE_COLUMNS


class LogisticTests(unittest.TestCase):
    def test_fit_serialize_and_probability(self):
        X=pd.DataFrame({'a':[-2,-1,0,1,2]*40,'b':np.tile([0,1,1,0,1],40)+np.arange(200)%3})
        y=pd.Series((np.arange(200)%7<2).astype(int))
        model,diag=fit_model(X,y)
        p=predict_matrix(X,json.loads(json.dumps(model)))
        self.assertTrue(diag['converged'])
        self.assertLess(diag['mean_objective_gradient_inf_norm'],1e-6)
        self.assertTrue(np.isfinite(p).all())
        self.assertTrue(((p>0)&(p<1)).all())
        self.assertAlmostEqual(metrics(y,np.full(len(y),y.mean()))['auc'],.5)

    def test_separation_none_complete_and_quasi(self):
        cases=[([-1,-1,0,0,1,1],[0,1,0,1,0,1],'not_detected'),
               ([-2,-1,1,2],[0,0,1,1],'complete'),
               ([-1,0,0,1],[0,0,1,1],'quasi')]
        for xs,ys,expected in cases:
            with self.subTest(expected=expected):
                diag,_=diagnose_design(pd.DataFrame({'x':xs}),pd.Series(ys))
                self.assertEqual(diag['separation_check'],expected)

    def test_raw_specials_and_no_label_fit(self):
        features=['age','DebtRatio',*LATE_COLUMNS]
        X=pd.DataFrame({'age':[30,40,50,60],'DebtRatio':[0,.5,10,1e10],
                        LATE_COLUMNS[0]:[0,1,96,98],LATE_COLUMNS[1]:[0,2,96,98],LATE_COLUMNS[2]:[0,3,96,98]})
        z=raw_design(X,features)
        self.assertEqual(z.any_delinquency_special.tolist(),[0,0,1,1])
        self.assertEqual(len(z.columns),6)
        self.assertEqual(z[LATE_COLUMNS[0]].tolist()[2:],[0,0])
        rules=fit_raw_rules(X,features)
        fitted=transform_raw(X,rules)
        self.assertTrue(np.isfinite(fitted.to_numpy()).all())
        novel=X.iloc[[0]].copy();novel['age']=100;novel['DebtRatio']=1e20
        before=json.dumps(rules,sort_keys=True)
        transform_raw(novel,rules)
        self.assertEqual(before,json.dumps(rules,sort_keys=True))
        self.assertGreater(float(transform_raw(novel,rules).DebtRatio.iloc[0]),float(fitted.DebtRatio.max()))

    def test_raw_invalid_inputs(self):
        X=pd.DataFrame({'age':[40],'DebtRatio':[.2]})
        for col,value in [('age',0),('age',40.5),('DebtRatio',-1),('DebtRatio',np.inf),('DebtRatio',np.nan)]:
            bad=X.copy();bad[col]=value
            with self.subTest(col=col,value=value),self.assertRaises(ValueError):
                raw_design(bad,['age','DebtRatio'])

    def test_odds_ratio_is_not_probability_ratio(self):
        X=pd.DataFrame({'w':[.2,-.7]})
        model={'columns':['w'],'intercept':-2.,'coefficients':{'w':-.8}}
        pa,pb=predict_matrix(X,model)
        ratio=np.exp(-.8*(-.7-.2))
        self.assertAlmostEqual((pb/(1-pb))/(pa/(1-pa)),ratio)
        self.assertNotAlmostEqual(pb/pa,ratio)


if __name__=='__main__':
    unittest.main()
