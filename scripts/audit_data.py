"""模块二：全样本数据质量总览，以及固定训练集的独立描述统计。

复现：在项目根目录运行 python scripts/audit_data.py
依赖：numpy、pandas、xlrd。仅输出审计结果和 TeX 数字，不清洗、不重新划分。
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "module2"
TABLES = ROOT / "latex" / "tables"
TARGET = "SeriousDlqin2yrs"
QUANTILES = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
PLOT_FEATURES = ["RevolvingUtilizationOfUnsecuredLines", "DebtRatio", "NumberOfTimes90DaysLate"]
LABELS = {
    "RevolvingUtilizationOfUnsecuredLines": ("额度使用率", "比例"),
    "age": ("年龄", "年龄"),
    "NumberOfTime30-59DaysPastDueNotWorse": ("30--59 天逾期次数", "计数"),
    "DebtRatio": ("负债收入比", "比例"),
    "MonthlyIncome": ("月收入", "金额"),
    "NumberOfOpenCreditLinesAndLoans": ("在用信贷账户数", "计数"),
    "NumberOfTimes90DaysLate": ("90 天及以上逾期次数", "计数"),
    "NumberRealEstateLoansOrLines": ("房地产贷款数", "计数"),
    "NumberOfTime60-89DaysPastDueNotWorse": ("60--89 天逾期次数", "计数"),
    "NumberOfDependents": ("家属人数", "计数"),
}
LATE_COLUMNS = ["NumberOfTime30-59DaysPastDueNotWorse", "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfTimes90DaysLate"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict]:
    """在写文件之前核验原始数据及冻结索引，保证沿用模块一的同一批记录。"""
    paths = {
        "raw": ROOT / "cs-training.csv",
        "dictionary": ROOT / "Data Dictionary.xls",
        "assignment": ROOT / "results/module1/sample_assignment.csv",
        "split_metadata": ROOT / "results/module1/split_metadata.json",
        "previous_duplicate_audit": ROOT / "results/module1/duplicate_audit.json",
    }
    hashes = {name: sha256(path) for name, path in paths.items()}
    meta = json.loads(paths["split_metadata"].read_text(encoding="utf-8"))
    if hashes["raw"] != meta["input_sha256"] or hashes["assignment"] != meta["assignment_sha256"]:
        raise ValueError("原始数据或固定索引哈希不匹配，请先核查模块一产物。")
    if meta["random_seed"] != 20261103:
        raise ValueError("预期沿用种子 20261103 的固定划分。")
    frame = pd.read_csv(paths["raw"])
    frame = frame.rename(columns={frame.columns[0]: "row_id"})
    features = meta["features"]
    if frame.shape != (150_000, 12) or frame.columns.tolist() != ["row_id", TARGET, *features]:
        raise ValueError("原始数据规模或字段发生变化。")
    if set(features) != set(LABELS) or not frame[TARGET].isin([0, 1]).all():
        raise ValueError("特征或二分类标签不符合预期。")
    assignment = pd.read_csv(paths["assignment"])
    if len(assignment) != len(frame) or not assignment["row_id"].is_unique:
        raise ValueError("固定索引缺行或包含重复行标识。")
    pos = assignment["source_row_0based"].to_numpy()
    if not np.array_equal(pos, np.arange(len(frame))):
        raise ValueError("固定索引的原始行位置不完整或顺序发生变化。")
    if not frame["row_id"].equals(assignment["row_id"]) or not frame[TARGET].equals(assignment[TARGET]):
        raise ValueError("固定索引与原始行号或标签未对齐。")
    if assignment["split"].value_counts().to_dict() != {"train": 90_000, "validation": 30_000, "test": 30_000}:
        raise ValueError("固定划分的类别或样本数不符合 60/20/20。")
    if frame["row_id"].isna().any() or not frame["row_id"].is_unique:
        raise ValueError("原始行标识缺失或不唯一。")
    # 已核查该 XLS 的第 2 行为表头，不能用默认 header=0 吞掉字段定义。
    dictionary = pd.read_excel(paths["dictionary"], header=1)
    dictionary.columns = dictionary.columns.str.strip()
    if dictionary["Variable Name"].duplicated().any() or set(dictionary["Variable Name"]) != {TARGET, *features}:
        raise ValueError("数据字典缺字段或字段重复。")
    inputs = {name: {"path": str(path.relative_to(ROOT)), "sha256": hashes[name]} for name, path in paths.items()}
    return frame, assignment, dictionary, meta, inputs


def variable_audit(frame: pd.DataFrame, scope: str, dictionary: pd.DataFrame) -> pd.DataFrame:
    lookup = dictionary.set_index("Variable Name")
    rows = []
    for name in frame.columns:
        s = frame[name]
        role = "identifier" if name == "row_id" else "target" if name == TARGET else "feature"
        logical = "行标识" if role == "identifier" else "二分类" if role == "target" else LABELS[name][1]
        nonmissing = s.dropna()
        finite = nonmissing[np.isfinite(nonmissing)]
        row = {
            "scope": scope, "variable": name, "role": role,
            "storage_dtype": str(s.dtype), "business_type": logical,
            "dictionary_type": "identifier" if role == "identifier" else lookup.loc[name, "Type"],
            "description": "Original row identifier, excluded from predictors" if role == "identifier" else lookup.loc[name, "Description"],
            "n_total": len(s), "n_nonmissing": int(s.notna().sum()),
            "n_missing": int(s.isna().sum()), "missing_rate": float(s.isna().mean()),
            "n_unique_nonmissing": int(s.nunique(dropna=True)),
            "n_infinite": int(len(nonmissing) - len(finite)),
        }
        if role == "feature":
            q = finite.quantile(QUANTILES, interpolation="linear")
            row.update({"min": float(finite.min()), "max": float(finite.max())})
            row.update({f"p{round(prob * 100):02d}": float(value) for prob, value in q.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def extreme_audit(frame: pd.DataFrame, scope: str, features: list[str]) -> pd.DataFrame:
    """标记需核查的值，不将比例大于 1 或极端计数直接判定为错误。"""
    checks = [
        ("age", "equals_0", frame["age"].eq(0)),
        ("MonthlyIncome", "equals_0", frame["MonthlyIncome"].eq(0)),
        ("DebtRatio", "greater_than_1", frame["DebtRatio"].gt(1)),
        ("RevolvingUtilizationOfUnsecuredLines", "greater_than_1", frame["RevolvingUtilizationOfUnsecuredLines"].gt(1)),
    ]
    for name in LATE_COLUMNS:
        checks.extend([(name, "equals_96", frame[name].eq(96)), (name, "equals_98", frame[name].eq(98)),
                       (name, "in_96_or_98", frame[name].isin([96, 98]))])
    for name in features:
        checks.append((name, "negative_value", frame[name].lt(0)))
        if LABELS[name][1] in {"年龄", "计数"}:
            checks.append((name, "noninteger_nonmissing_value", frame[name].notna() & frame[name].mod(1).ne(0)))
    rows = [{"scope": scope, "variable": name, "check": check, "count": int(mask.sum()),
             "n_total": len(frame), "n_nonmissing": int(frame[name].notna().sum()),
             "rate_total": float(mask.mean())} for name, check, mask in checks]
    # 三个逾期字段分别计数后，再统计任一字段命中的记录数，避免把同一人重复相加。
    mask = frame[LATE_COLUMNS].isin([96, 98]).any(axis=1)
    rows.append({"scope": scope, "variable": "any_delinquency_field", "check": "in_96_or_98",
                 "count": int(mask.sum()), "n_total": len(frame), "n_nonmissing": len(frame),
                 "rate_total": float(mask.mean())})
    return pd.DataFrame(rows)


def duplicate_audit(frame: pd.DataFrame, features: list[str]) -> dict:
    groups = frame.groupby(features, dropna=False, sort=False).ngroup()
    sizes = groups.value_counts()
    labeled_sizes = frame.groupby([TARGET, *features], dropna=False, sort=False).size()
    return {
        "n_total": len(frame), "row_id_unique": bool(frame["row_id"].is_unique),
        "duplicate_labeled_excess_rows": int(frame.duplicated([TARGET, *features]).sum()),
        "duplicate_labeled_groups": int(labeled_sizes.gt(1).sum()),
        "rows_in_duplicate_labeled_groups": int(labeled_sizes[labeled_sizes.gt(1)].sum()),
        "duplicate_feature_excess_rows": int(frame.duplicated(features).sum()),
        "duplicate_feature_groups": int(sizes.gt(1).sum()),
        "rows_in_duplicate_feature_groups": int(sizes[sizes.gt(1)].sum()),
        "mixed_label_feature_groups": int(frame.groupby(groups)[TARGET].nunique().gt(1).sum()),
        "maximum_feature_group_size": int(sizes.max()),
    }


def tex_number(value: float) -> str:
    """计数不保留小数，长尾大值用科学记数法，其他值保留最多三位小数。"""
    if abs(value) >= 1_000_000:
        exponent = int(np.floor(np.log10(abs(value))))
        return f"${value / 10 ** exponent:.3f}\\times10^{{{exponent}}}$"
    if float(value).is_integer():
        return f"{value:,.0f}"
    return f"{value:,.3f}".rstrip("0").rstrip(".")


def write_tex(variables: pd.DataFrame, extremes: pd.DataFrame, balances: pd.DataFrame) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    full = variables.query("scope == 'all' and role == 'feature'").set_index("variable")
    table = [r"\begin{tabular}{llrrrrrrr}", r"\toprule",
             r"变量 & 类型 & 缺失率 & 最小值 & P25 & 中位数 & P75 & P99 & 最大值 \\", r"\midrule"]
    for name, row in full.iterrows():
        nums = " & ".join(tex_number(float(row[k])) for k in ["min", "p25", "p50", "p75", "p99", "max"])
        table.append(f"{LABELS[name][0]} & {row['business_type']} & {100 * row['missing_rate']:.3f}\\% & {nums} " + r"\\")
    table += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "audit_summary.tex").write_text("\n".join(table) + "\n", encoding="utf-8")

    def missing(scope: str, name: str) -> int:
        return int(variables.loc[(variables.scope == scope) & (variables.variable == name), "n_missing"].iloc[0])

    def flagged(scope: str, name: str, check: str) -> int:
        return int(extremes.loc[(extremes.scope == scope) & (extremes.variable == name) & (extremes["check"] == check), "count"].iloc[0])

    balance = balances.set_index("scope").loc["all"]
    values = {
        "AuditGoodN": f"{int(balance['good']):,}", "AuditBadN": f"{int(balance['bad']):,}",
        "AuditBadRate": f"{100 * balance['bad_rate']:.3f}\\%",
        "AuditGoodRate": f"{100 * balance['good_rate']:.3f}\\%",
        "AuditIncomeMissing": f"{missing('all', 'MonthlyIncome'):,}",
        "AuditDependentsMissing": f"{missing('all', 'NumberOfDependents'):,}",
        "AuditTrainIncomeMissing": f"{missing('train', 'MonthlyIncome'):,}",
        "AuditTrainDependentsMissing": f"{missing('train', 'NumberOfDependents'):,}",
        "AuditAgeZero": str(flagged('all', 'age', 'equals_0')),
        "AuditTrainAgeZero": str(flagged('train', 'age', 'equals_0')),
        "AuditZeroIncome": f"{flagged('all', 'MonthlyIncome', 'equals_0'):,}",
        "AuditSpecialLate": str(flagged('all', 'any_delinquency_field', 'in_96_or_98')),
        "AuditUtilizationAboveOne": f"{flagged('all', 'RevolvingUtilizationOfUnsecuredLines', 'greater_than_1'):,}",
        "AuditDebtAboveOne": f"{flagged('all', 'DebtRatio', 'greater_than_1'):,}",
    }
    (TABLES / "audit_macros.tex").write_text("\n".join(
        f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in values.items()
    ) + "\n", encoding="utf-8")


def main() -> None:
    frame, assignment, dictionary, split_meta, inputs = load_inputs()
    features = split_meta["features"]
    scopes = {"all": frame, "train": frame.loc[assignment["split"].eq("train")]}
    variables = pd.concat([variable_audit(df, name, dictionary) for name, df in scopes.items()], ignore_index=True)
    extremes = pd.concat([extreme_audit(df, name, features) for name, df in scopes.items()], ignore_index=True)
    balances = pd.DataFrame([{
        "scope": name, "n_total": len(df), "good": int(df[TARGET].eq(0).sum()),
        "bad": int(df[TARGET].eq(1).sum()), "good_rate": float(df[TARGET].eq(0).mean()),
        "bad_rate": float(df[TARGET].eq(1).mean()),
    } for name, df in scopes.items()])
    duplicates = {name: duplicate_audit(df, features) for name, df in scopes.items()}
    groups = frame.groupby(features, dropna=False, sort=False).ngroup()
    duplicates["cross_split_feature_groups"] = int(assignment.groupby(groups)["split"].nunique().gt(1).sum())
    previous = json.loads((ROOT / "results/module1/duplicate_audit.json").read_text(encoding="utf-8"))
    for key in ["duplicate_labeled_excess_rows", "duplicate_feature_excess_rows", "duplicate_feature_groups",
                "rows_in_duplicate_feature_groups", "mixed_label_feature_groups"]:
        if duplicates["all"][key] != previous[key]:
            raise ValueError(f"重复审计与模块一不一致：{key}")
    if duplicates["cross_split_feature_groups"] != previous["cross_split_feature_groups"]:
        raise ValueError("跨集重复组数与模块一不一致。")
    for name, df in scopes.items():
        b = balances.set_index("scope").loc[name]
        assert b["good"] + b["bad"] == len(df)
    OUT.mkdir(parents=True, exist_ok=True)
    variables.to_csv(OUT / "variable_audit.csv", index=False, encoding="utf-8-sig")
    extremes.to_csv(OUT / "extreme_checks.csv", index=False, encoding="utf-8-sig")
    balances.to_csv(OUT / "class_balance.csv", index=False, encoding="utf-8-sig")
    dictionary.to_csv(OUT / "data_dictionary.csv", index=False, encoding="utf-8-sig")
    (OUT / "duplicate_audit.json").write_text(json.dumps(duplicates, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "inputs": inputs, "scope_sizes": {name: len(df) for name, df in scopes.items()},
        "frozen_split_seed": split_meta["random_seed"], "new_random_sampling": False,
        "quantiles": QUANTILES, "quantile_interpolation": "linear",
        "quantile_population": "nonmissing finite feature values",
        "missing_and_flag_rate_denominator": "total rows of the corresponding scope",
        "future_plot_features": PLOT_FEATURES, "data_mutation": "none",
        "command": "python scripts/audit_data.py", "python": platform.python_version(),
        "packages": {pkg: importlib.metadata.version(pkg) for pkg in ["numpy", "pandas", "xlrd"]},
        "script_sha256": sha256(Path(__file__)),
    }
    write_tex(variables, extremes, balances)
    # 运行结束再核验一次，确认审计没有更改原始文件或冻结产物。
    for item in inputs.values():
        if sha256(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError(f"输入文件在审计过程中变化：{item['path']}")
    (OUT / "audit_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(balances.to_string(index=False))
    print(variables.loc[variables.n_missing.gt(0), ["scope", "variable", "storage_dtype", "business_type", "n_missing", "missing_rate"]].to_string(index=False))
    print(json.dumps(duplicates, ensure_ascii=False, indent=2))
    print(f"审计结果：{OUT}\n报告数字：{TABLES}")


if __name__ == "__main__":
    main()
