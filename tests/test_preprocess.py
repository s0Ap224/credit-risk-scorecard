"""验证迁移边界与原始缺失信息保留，不训练预测模型。"""
import copy
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from preprocess import (AGE_FLAG, DEPENDENTS_FLAG, FEATURES, INCOME_STATE, LATE_COLUMNS,
                        SHARED_FLAG, fit_rules, transform)


class PreprocessTests(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame({name: [0.0, 1.0, 2.0] for name in FEATURES})
        self.X["age"] = [20, 40, 60]
        self.X["MonthlyIncome"] = [2000, 4000, 6000]
        self.X["NumberOfDependents"] = [0, 1, np.nan]
        self.rules = fit_rules(self.X)

    def clients(self, n):
        return pd.concat([self.X.iloc[[0]]] * n, ignore_index=True)

    def test_fit_features_only_and_frozen_median(self):
        self.assertEqual(self.rules["age_median"], 40)
        X = self.clients(2)
        X["age"] = [0, 120]
        processed, _ = transform(X, self.rules)
        self.assertEqual(processed.age.tolist(), [40, 120])
        self.assertEqual(self.rules["constant_binary_markers"][AGE_FLAG], 0)
        self.assertEqual(self.rules["derived_column_roles"][AGE_FLAG], "audit_only_constant_in_training")
        with self.assertRaisesRegex(ValueError, "unknown_fields"):
            fit_rules(self.X.assign(SeriousDlqin2yrs=[0, 1, 0]))

    def test_original_zero_and_imputed_zero_stay_distinct(self):
        processed, _ = transform(self.X, self.rules)
        self.assertEqual(processed.NumberOfDependents.tolist(), [0, 1, 0])
        self.assertEqual(processed[DEPENDENTS_FLAG].tolist(), [0, 0, 1])

    def test_income_boundaries(self):
        X = self.clients(5)
        X["MonthlyIncome"] = [np.nan, 0, 9.99, 10, 1e9]
        processed, _ = transform(X, self.rules)
        self.assertEqual(processed[INCOME_STATE].tolist(), ["MISSING", "LOW", "LOW", "REGULAR", "REGULAR"])
        self.assertTrue(pd.isna(processed.MonthlyIncome.iloc[0]))
        self.assertEqual(processed.MonthlyIncome.iloc[-1], 1e9)

    def test_overlap_and_per_field_special_routes(self):
        X = self.clients(1)
        X.loc[0, ["NumberOfDependents", "MonthlyIncome", "age"]] = [np.nan, np.nan, 0]
        for name, value in zip(LATE_COLUMNS, [96, 98, 0]):
            X.loc[0, name] = value
        before = X.copy(deep=True)
        processed, audit = transform(X, self.rules)
        self.assertEqual(processed[DEPENDENTS_FLAG].iloc[0], 1)
        self.assertEqual(processed[SHARED_FLAG].iloc[0], 1)
        self.assertEqual(processed[INCOME_STATE].iloc[0], "MISSING")
        self.assertEqual([processed[name + "_special"].iloc[0] for name in LATE_COLUMNS], [1, 1, 0])
        self.assertEqual([processed[name].iloc[0] for name in LATE_COLUMNS], [96, 98, 0])
        self.assertEqual(audit["n_before"], audit["n_after"])
        pd.testing.assert_frame_equal(X, before)

    def test_no_clipping_and_no_label_requirement(self):
        X = self.clients(1)
        X.loc[0, ["DebtRatio", "RevolvingUtilizationOfUnsecuredLines", "MonthlyIncome"]] = [1e9, 1e12, 1e10]
        X.loc[0, LATE_COLUMNS[0]] = 1000
        X.loc[0, "NumberOfDependents"] = 1e20
        processed, audit = transform(X, self.rules)
        self.assertEqual(processed.DebtRatio.iloc[0], 1e9)
        self.assertEqual(processed.RevolvingUtilizationOfUnsecuredLines.iloc[0], 1e12)
        self.assertEqual(processed.NumberOfDependents.iloc[0], 1e20)
        self.assertEqual(processed[LATE_COLUMNS[0] + "_special"].iloc[0], 0)
        self.assertEqual(audit["dropped"], 0)

    def test_invalid_age_missing_and_negative(self):
        X = self.clients(3)
        X["age"] = [None, -1, 0]
        processed, audit = transform(X, self.rules)
        self.assertEqual(processed.age.tolist(), [40, 40, 40])
        self.assertEqual(audit["rule_hits"]["age_invalid_imputed"], 3)

    def test_bad_inputs_are_field_level_errors(self):
        cases = [("DebtRatio", "bad"), ("DebtRatio", np.inf), ("DebtRatio", -1),
                 ("DebtRatio", np.nan), ("MonthlyIncome", -1),
                 ("NumberOfDependents", -.5), ("NumberOfDependents", .5),
                 (LATE_COLUMNS[0], np.nan), (LATE_COLUMNS[1], .5), ("age", 20.5)]
        for name, value in cases:
            with self.subTest(field=name, value=value):
                X = self.clients(1).astype(object)
                X.loc[0, name] = value
                before = X.copy(deep=True)
                with self.assertRaisesRegex(ValueError, name):
                    transform(X, self.rules)
                pd.testing.assert_frame_equal(X, before)
        with self.assertRaisesRegex(ValueError, "DebtRatio"):
            transform(self.X.drop(columns="DebtRatio"), self.rules)

    def test_json_roundtrip_does_not_change_mapping(self):
        expected, _ = transform(self.X, self.rules)
        rules = json.loads(json.dumps(self.rules))
        snapshot = copy.deepcopy(rules)
        actual, _ = transform(self.X, rules)
        pd.testing.assert_frame_equal(actual, expected)
        self.assertEqual(rules, snapshot)

    def test_reprocessing_derived_columns_is_rejected(self):
        processed, _ = transform(self.X, self.rules)
        with self.assertRaisesRegex(ValueError, "unknown_fields"):
            transform(processed, self.rules)

    def test_fit_rejects_absent_valid_age(self):
        X = self.clients(2)
        X["age"] = [0, None]
        with self.assertRaisesRegex(ValueError, "年龄"):
            fit_rules(X)


if __name__ == "__main__":
    unittest.main()
