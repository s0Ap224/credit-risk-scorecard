"""模块二：训练集拟合、跨样本冻结的缺失和特殊值处理。

运行：python scripts/preprocess.py
API：fit_rules(train_X) -> dict；transform(raw_X, rules) -> (processed_X, audit)。
X 只包含十个原始特征，无需行号或标签；转换后的额外字段不接受重复处理。
MonthlyIncome 的 NA 原值仍保留，由 MISSING 状态路由至特殊档，而不是数值填补。
三类逾期原始编码保留，后续分箱必须先读取各字段的 _special 标记。
"""

from __future__ import annotations

import json
import hashlib
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from audit_data import LABELS, LATE_COLUMNS, TARGET, load_inputs, sha256

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "module2"
PREPARED = OUT / "prepared"
FEATURES = list(LABELS)
COUNT_FEATURES = [name for name, (_, kind) in LABELS.items() if kind == "计数"]
MISSING_ALLOWED = {"NumberOfDependents", "MonthlyIncome", "age"}
DEPENDENTS_FLAG = "NumberOfDependents_missing"
AGE_FLAG = "age_invalid"
SHARED_FLAG = "delinquency_special"
INCOME_STATE = "MonthlyIncome_state"
SPECIAL_FLAGS = [f"{name}_special" for name in LATE_COLUMNS]
BINARY_FLAGS = [DEPENDENTS_FLAG, *SPECIAL_FLAGS, SHARED_FLAG, AGE_FLAG]


def _validate_X(X: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X 必须是包含十个原始特征的 pandas DataFrame。")
    if X.columns.duplicated().any():
        raise ValueError("输入存在重复列名。")
    missing = sorted(set(FEATURES) - set(X.columns))
    extra = sorted(set(X.columns) - set(FEATURES))
    if missing or extra:
        raise ValueError(f"字段不符：missing_fields={missing}; unknown_fields={extra}。请只传原始特征，勿重复转换。")
    result = X[FEATURES].copy(deep=True)
    errors = []
    for name in FEATURES:
        raw = result[name]
        numeric = pd.to_numeric(raw, errors="coerce")
        if np.iscomplexobj(numeric.to_numpy()):
            errors.append(f"{name}: complex_value")
            continue
        invalid_text = raw.notna() & numeric.isna()
        if invalid_text.any():
            errors.append(f"{name}: nonnumeric rows={raw.index[invalid_text].tolist()[:3]}")
            continue
        try:
            numeric = numeric.astype("float64")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}: unsupported_numeric_format") from exc
        nonfinite = numeric.notna() & ~np.isfinite(numeric)
        if nonfinite.any():
            errors.append(f"{name}: infinite_value rows={raw.index[nonfinite].tolist()[:3]}")
        if name not in MISSING_ALLOWED and numeric.isna().any():
            errors.append(f"{name}: unexpected_missing")
        if name != "age" and numeric.lt(0).any():
            errors.append(f"{name}: negative_value")
        if name in COUNT_FEATURES and (numeric.notna() & numeric.mod(1).ne(0)).any():
            errors.append(f"{name}: noninteger_count")
        if name == "age" and (numeric.gt(0) & numeric.mod(1).ne(0)).any():
            errors.append("age: noninteger_positive_age")
        result[name] = numeric
    if errors:
        raise ValueError("; ".join(errors))
    return result


def fit_rules(train_X: pd.DataFrame) -> dict:
    """仅从训练特征学习年龄中位数，不使用标签、验证集或测试集。"""
    X = _validate_X(train_X)
    valid_age = X.loc[X.age.gt(0), "age"]
    if not len(X) or valid_age.empty:
        raise ValueError("训练集为空或不存在有效正年龄，无法估计年龄填补参数。")
    observed_income = X.MonthlyIncome.dropna()
    rules = {
        "schema_version": 1, "features": FEATURES,
        "age_median": float(valid_age.median()), "dependents_fill_value": 0,
        "income_low_upper_exclusive": 10, "income_states": ["MISSING", "LOW", "REGULAR"],
        "delinquency_special_values": [96, 98],
        "treatment_order": ["dependents", "delinquency", "income", "age"],
        "markers_may_overlap": True, "drop_rows": False, "clip_numeric_values": False,
        "income_raw_missing_retained": True,
        "comparison_only_income_median": float(observed_income.median()) if len(observed_income) else None,
        "binning_contract": {
            "income": "MISSING first; LOW iff 0 <= raw value < 10; REGULAR iff raw value >= 10",
            "delinquency": "Check each _special flag first: 1 -> pooled 96/98 special bin; 0 -> ordinary binning",
            "ordinary_cutpoints": "Learn in module three using training features and labels only",
            "legal_out_of_training_range": "Preserve; later ordinary bins must cover open-ended tails",
            "DebtRatio": "Preserve in every income state; conditional use remains a later validation question",
        },
        "derived_column_roles": {
            DEPENDENTS_FLAG: "candidate_predictor", INCOME_STATE: "special_bin_route",
            SHARED_FLAG: "audit_shared_state", AGE_FLAG: "audit_marker",
            **{name: "special_bin_route" for name in SPECIAL_FLAGS},
        },
        "training_rows": len(X),
        "training_features_sha256": hashlib.sha256(
            pd.util.hash_pandas_object(X, index=True).values.tobytes()
        ).hexdigest(),
        "training_domains": {},
    }
    for name in FEATURES:
        observed = X[name].dropna()
        if name == "age":
            observed = observed[observed.gt(0)]
        rules["training_domains"][name] = {
            "min": float(observed.min()) if len(observed) else None,
            "max": float(observed.max()) if len(observed) else None,
        }
    processed, _ = transform(train_X, rules)
    rules["constant_binary_markers"] = {
        name: int(processed[name].iloc[0]) for name in BINARY_FLAGS if processed[name].nunique() == 1
    }
    for name in rules["constant_binary_markers"]:
        rules["derived_column_roles"][name] = "audit_only_constant_in_training"
    return rules


def transform(X: pd.DataFrame, rules: dict) -> tuple[pd.DataFrame, dict]:
    """按冻结参数转换新数据；异常输入抛出字段级 ValueError，不静默删行。"""
    if rules.get("schema_version") != 1 or rules.get("features") != FEATURES:
        raise ValueError("冻结规则版本或特征定义不兼容。")
    median = rules.get("age_median")
    if median is None or not np.isfinite(median) or median <= 0:
        raise ValueError("冻结规则中 age_median 必须为有效正数。")
    raw = _validate_X(X)
    processed = raw.copy(deep=True)

    # 第一步：先识别原缺失，再填补，保留与原生零值的区别。
    processed[DEPENDENTS_FLAG] = raw.NumberOfDependents.isna().astype("int8")
    processed["NumberOfDependents"] = raw.NumberOfDependents.fillna(rules["dependents_fill_value"])

    # 第二步：每列独立识别特殊编码；共同标记只计同一记录一次。
    for name, flag in zip(LATE_COLUMNS, SPECIAL_FLAGS):
        processed[flag] = raw[name].isin(rules["delinquency_special_values"]).astype("int8")
    processed[SHARED_FLAG] = processed[SPECIAL_FLAGS].any(axis=1).astype("int8")

    # 第三步：所有人均判断收入状态，不能因逾期特殊编码而跳过收入缺失。
    income = raw.MonthlyIncome
    processed[INCOME_STATE] = np.select(
        [income.isna(), income.lt(rules["income_low_upper_exclusive"])],
        ["MISSING", "LOW"], default="REGULAR",
    )
    invalid_age = raw.age.isna() | raw.age.le(0)
    processed[AGE_FLAG] = invalid_age.astype("int8")
    processed["age"] = raw.age.mask(invalid_age, median)

    hits = {
        "dependents_missing_filled_0": int(processed[DEPENDENTS_FLAG].sum()),
        **{flag: int(processed[flag].sum()) for flag in SPECIAL_FLAGS},
        "any_delinquency_special": int(processed[SHARED_FLAG].sum()),
        "income_missing": int(processed[INCOME_STATE].eq("MISSING").sum()),
        "income_low": int(processed[INCOME_STATE].eq("LOW").sum()),
        "income_regular": int(processed[INCOME_STATE].eq("REGULAR").sum()),
        "age_invalid_imputed": int(invalid_age.sum()),
        "income_above_million_retained": int(raw.MonthlyIncome.gt(1_000_000).sum()),
        "DebtRatio_above_1_retained": int(raw.DebtRatio.gt(1).sum()),
        "utilization_above_1_retained": int(raw.RevolvingUtilizationOfUnsecuredLines.gt(1).sum()),
    }
    out_of_range = {}
    for name, domain in rules.get("training_domains", {}).items():
        s = raw[name]
        if name == "age":
            s = s.where(s.gt(0))
        out_of_range[name] = {
            "below_training_min": int(s.lt(domain["min"]).sum()) if domain["min"] is not None else 0,
            "above_training_max": int(s.gt(domain["max"]).sum()) if domain["max"] is not None else 0,
        }
    overlaps = processed.groupby([DEPENDENTS_FLAG, SHARED_FLAG, INCOME_STATE], dropna=False).size().reset_index(name="n")
    audit = {"n_before": len(raw), "n_after": len(processed), "dropped": 0, "rule_hits": hits,
             "out_of_training_range": out_of_range, "overlaps": overlaps.to_dict(orient="records")}
    for name, flag in zip(LATE_COLUMNS, SPECIAL_FLAGS):
        if (raw[name].isin([96, 98]) & processed[flag].eq(0)).any():
            raise AssertionError("特殊编码错误进入普通计数路由。")
    if not processed.index.equals(X.index):
        raise AssertionError("转换不得更改样本索引。")
    return processed, audit


def _write_comparison(raw_train: pd.DataFrame, rules: dict) -> None:
    """比较信息是否保留，风险数字只用于描述，不声称预测性能已经提升。"""
    rows = []
    for feature in ["NumberOfDependents", "MonthlyIncome"]:
        missing = raw_train[feature].isna()
        miss_risk = float(raw_train.loc[missing, TARGET].mean())
        obs_risk = float(raw_train.loc[~missing, TARGET].mean())
        if feature == "NumberOfDependents":
            alternatives = [("fill_0_only", False, False, 0), ("fill_0_plus_missing_flag", True, True, 0),
                            ("standalone_missing_bin", False, True, None)]
        else:
            median = rules["comparison_only_income_median"]
            alternatives = [("median_only", False, False, median), ("median_plus_missing_flag", False, True, median),
                            ("missing_and_low_standalone_states", True, True, None)]
        for scheme, selected, retained, value in alternatives:
            rows.append({"feature": feature, "scheme": scheme, "selected_for_preparation": selected,
                         "missing_information_retained": retained, "numeric_fill_value": value,
                         "training_missing_n": int(missing.sum()), "missing_bad_rate": miss_risk,
                         "observed_bad_rate": obs_risk,
                         "prediction_performance": "not evaluated; compare with validation data in later modeling"})
    pd.DataFrame(rows).to_csv(OUT / "missing_strategy_comparison.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    frame, assignment, _, split_meta, inputs = load_inputs()
    train = frame.loc[assignment.split.eq("train")]
    rules = fit_rules(train[FEATURES])
    rules.update({"inputs": inputs, "frozen_split_seed": split_meta["random_seed"]})
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = OUT / "preprocess_rules.json"
    if frozen.exists() and json.loads(frozen.read_text(encoding="utf-8")) != rules:
        raise ValueError("重新拟合与已冻结规则不同，停止覆盖；请调查规则或训练输入变化。")
    transformed, audits = {}, {}
    # 全部转换通过验证后再写结果，发现任何非法输入即停止。
    for split in ["train", "validation", "test"]:
        raw = frame.loc[assignment.split.eq(split)]
        processed, audit = transform(raw[FEATURES], rules)
        transformed[split] = pd.concat([raw[["row_id", TARGET]], processed], axis=1)
        audits[split] = audit
    PREPARED.mkdir(parents=True, exist_ok=True)
    frozen.write_text(json.dumps(rules, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for split, data in transformed.items():
        data.to_csv(PREPARED / f"{split}.csv", index=False, encoding="utf-8-sig")
    summary = pd.DataFrame([{"split": name, "n_before": value["n_before"], "n_after": value["n_after"],
                             "dropped": value["dropped"], **value["rule_hits"]} for name, value in audits.items()])
    summary.to_csv(OUT / "preprocess_summary.csv", index=False, encoding="utf-8-sig")
    hits = [{"split": split, "rule": key, "n_hit": value, "n_total": audit["n_before"]}
            for split, audit in audits.items() for key, value in audit["rule_hits"].items()]
    pd.DataFrame(hits).to_csv(OUT / "preprocess_rule_hits.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"split": split, **row} for split, audit in audits.items() for row in audit["overlaps"]]).to_csv(
        OUT / "preprocess_overlap.csv", index=False, encoding="utf-8-sig")
    (OUT / "preprocess_audit.json").write_text(json.dumps(audits, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    cleaning_rules = [
        {"object": "NumberOfDependents", "condition": "原缺失", "action": "先标记1再补0；原生0标记0", "estimated_from": "固定规则", "new_input": "同一映射；负数或非整数拒绝"},
        {"object": "三类逾期", "condition": "每列值为96或98", "action": "保留原值、逐列标记、共同标记；各列先路由特殊档", "estimated_from": "固定编码集合[96,98]", "new_input": "特殊标记不互斥；其他非负整数走普通开口档"},
        {"object": "MonthlyIncome", "condition": "NA / [0,10) / [10,+inf)", "action": "分别MISSING / LOW / REGULAR，原值保留", "estimated_from": "固定状态规则", "new_input": "10归REGULAR；负数拒绝；特殊档的WOE后续训练"},
        {"object": "age", "condition": "缺失或<=0", "action": f"填训练有效年龄中位数{rules['age_median']:g}并标记", "estimated_from": "仅训练特征", "new_input": "冻结中位数；正年龄非整数拒绝；训练恒定标记仅审计"},
        {"object": "比例及合法长尾", "condition": "有限且通过字段定义校验", "action": "保留，不截尾不删行，DebtRatio在所有收入状态下保留", "estimated_from": "不估计截尾参数", "new_input": "超训练范围也保留，后续普通箱使用开口尾部"},
        {"object": "输入校验", "condition": "缺字段/非法字符串/无穷/非法负值或计数/未约定缺失", "action": "ValueError指出字段，停止转换，不静默删行", "estimated_from": "固定schema", "new_input": "接收十原始特征，不需要标签，已处理额外列拒绝重复转换"},
    ]
    pd.DataFrame(cleaning_rules).to_csv(OUT / "cleaning_rules.csv", index=False, encoding="utf-8-sig")
    _write_comparison(train, rules)
    tables = ROOT / "latex" / "tables"
    macros = {"PreprocessAgeMedian": f"{rules['age_median']:g}",
              "PreprocessAgeChanged": str(int(summary.age_invalid_imputed.sum())),
              "PreprocessTrainN": f"{audits['train']['n_after']:,}",
              "PreprocessValidationN": f"{audits['validation']['n_after']:,}",
              "PreprocessTestN": f"{audits['test']['n_after']:,}"}
    (tables / "preprocess_macros.tex").write_text("\n".join(
        f"\\newcommand{{\\{key}}}{{{value}}}" for key, value in macros.items()) + "\n", encoding="utf-8")
    for item in inputs.values():
        if sha256(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError("原始数据或固定索引发生变化。")
    outputs = [frozen, *PREPARED.glob("*.csv"), *OUT.glob("preprocess_*.csv"), OUT / "preprocess_audit.json",
               OUT / "cleaning_rules.csv", OUT / "missing_strategy_comparison.csv", tables / "preprocess_macros.tex"]
    manifest = {"inputs": inputs, "command": "python scripts/preprocess.py", "python": platform.python_version(),
                "packages": {"numpy": np.__version__, "pandas": pd.__version__},
                "script_sha256": sha256(Path(__file__)), "audit_loader_sha256": sha256(ROOT / "scripts/audit_data.py"),
                "frozen_split_seed": split_meta["random_seed"], "new_random_sampling": False,
                "fit_uses_target": False, "outputs": {str(p.relative_to(ROOT)): sha256(p) for p in outputs}}
    (OUT / "preprocess_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"年龄中位数：{rules['age_median']:g}；恒定标记：{rules['constant_binary_markers']}")
    print(f"冻结规则：{frozen}\n处理后数据：{PREPARED}")


if __name__ == "__main__":
    main()
