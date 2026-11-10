"""模块一：按标签分层，将原始样本固定划分为 60%/20%/20%。

运行方法（项目根目录）：python scripts/split_samples.py
依赖：numpy、pandas、scikit-learn；实际版本会保存到 split_metadata.json。
随机种子：20261103。划分前不填补、不分箱、不删行，不拟合任何模型。
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

PROJECT = Path(__file__).resolve().parents[1]
INPUT = PROJECT / "cs-training.csv"
OUTPUT = PROJECT / "results" / "module1"
TABLES = PROJECT / "latex" / "tables"
RANDOM_SEED = 20261103
TARGET = "SeriousDlqin2yrs"
NAMES = {"train": "训练集", "validation": "验证集", "test": "测试集"}


def main() -> None:
    input_sha256 = hashlib.sha256(INPUT.read_bytes()).hexdigest()
    frame = pd.read_csv(INPUT)
    if frame.shape != (150_000, 12):
        raise ValueError(f"输入规模与原始竞赛训练集不符：{frame.shape}")
    frame = frame.rename(columns={frame.columns[0]: "row_id"})
    if frame["row_id"].isna().any() or not frame["row_id"].is_unique:
        raise ValueError("原始行号缺失或不唯一，请核对来源。")
    if not frame[TARGET].isin([0, 1]).all():
        raise ValueError("目标标签必须为非缺失的 0 或 1。")
    features = frame.columns.drop(["row_id", TARGET]).tolist()
    if len(features) != 10:
        raise ValueError("候选解释变量应为十个；行号和目标不属于解释变量。")

    # 第一次分层：60% 训练，40% 留出；stratify 按目标变量保持两类比例。
    positions = np.arange(len(frame))
    train, holdout = train_test_split(
        positions, test_size=0.4, stratify=frame[TARGET].to_numpy(),
        random_state=RANDOM_SEED, shuffle=True,
    )
    # 第二次分层：留出样本各取一半，分别得到全样本的 20% 验证、20% 测试。
    validation, test = train_test_split(
        holdout, test_size=0.5, stratify=frame.iloc[holdout][TARGET].to_numpy(),
        random_state=RANDOM_SEED, shuffle=True,
    )
    indices = {"train": np.sort(train), "validation": np.sort(validation), "test": np.sort(test)}
    if [len(v) for v in indices.values()] != [90_000, 30_000, 30_000]:
        raise AssertionError("样本数未满足 60%/20%/20%。")
    union = np.concatenate(list(indices.values()))
    if len(np.unique(union)) != len(frame) or not np.array_equal(np.sort(union), positions):
        raise AssertionError("划分必须互斥并且完整覆盖原始数据。")

    assignment = pd.DataFrame({
        "row_id": frame["row_id"], "source_row_0based": positions,
        TARGET: frame[TARGET], "split": "",
    })
    summary_rows = []
    for name, idx in indices.items():
        assignment.loc[idx, "split"] = name
        bad = int(frame.iloc[idx][TARGET].sum())
        summary_rows.append({
            "split": name, "n": len(idx), "good": len(idx) - bad, "bad": bad,
            "bad_rate": bad / len(idx), "sample_share": len(idx) / len(frame),
        })
    summary = pd.DataFrame(summary_rows)

    # 重复审计排除 row_id；缺失位置也参与同特征记录的识别。
    # 特征一致并不能证明同一借款人，按本次指定比例保留全部原始记录。
    groups = frame.groupby(features, dropna=False, sort=False).ngroup()
    group_info = frame.assign(feature_group_id=groups, split=assignment["split"]).groupby(
        "feature_group_id", sort=False,
    ).agg(n=(TARGET, "size"), distinct_labels=(TARGET, "nunique"), n_splits=("split", "nunique"))
    repeated = group_info["n"].gt(1)
    audit = {
        "row_id_unique": True,
        "duplicate_labeled_excess_rows": int(frame.duplicated([TARGET, *features]).sum()),
        "duplicate_feature_excess_rows": int(frame.duplicated(features).sum()),
        "duplicate_feature_groups": int(repeated.sum()),
        "rows_in_duplicate_feature_groups": int(group_info.loc[repeated, "n"].sum()),
        "mixed_label_feature_groups": int(group_info["distinct_labels"].gt(1).sum()),
        "cross_split_feature_groups": int(group_info["n_splits"].gt(1).sum()),
        "removed_rows": 0,
        "same_customer_check": "No customer identifier or individual dates are available; row_id is only a row identifier.",
        "duplicate_policy": "Retain all original rows; stratified random row split. Identical-feature groups can cross subsets and are reported explicitly.",
    }

    # 防止数据或种子变动后静默更换已固定的样本。
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frozen = OUTPUT / "sample_assignment.csv"
    metadata_file = OUTPUT / "split_metadata.json"
    if metadata_file.exists():
        old = json.loads(metadata_file.read_text(encoding="utf-8"))
        if old["input_sha256"] != input_sha256 or old["random_seed"] != RANDOM_SEED:
            raise RuntimeError("输入或随机种子已变化，停止覆盖固定划分。")
    if frozen.exists() and not pd.read_csv(frozen).equals(assignment):
        raise RuntimeError("计算结果与已冻结索引不一致，须先核查环境或数据。")

    assignment.to_csv(frozen, index=False, encoding="utf-8")
    for name, idx in indices.items():
        # row_id 随文件保留便于追踪，后续建模必须排除 row_id 和 TARGET。
        frame.iloc[idx].to_csv(OUTPUT / f"{name}.csv", index=False, encoding="utf-8")
        assignment.loc[idx, ["row_id", "source_row_0based"]].to_csv(
            OUTPUT / f"{name}_indices.csv", index=False, encoding="utf-8",
        )
    summary.to_csv(OUTPUT / "split_summary.csv", index=False, encoding="utf-8")
    duplicate_rows = assignment.loc[groups.map(group_info["n"]).gt(1)].copy()
    duplicate_rows["feature_group_id"] = groups.loc[duplicate_rows.index]
    duplicate_rows.to_csv(OUTPUT / "duplicate_rows.csv", index=False, encoding="utf-8")
    group_info.loc[repeated].to_csv(OUTPUT / "duplicate_groups.csv", encoding="utf-8")
    (OUTPUT / "duplicate_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    metadata = {
        "input_file": "cs-training.csv", "input_sha256": input_sha256,
        "random_seed": RANDOM_SEED, "ratios": {"train": 0.6, "validation": 0.2, "test": 0.2},
        "target": TARGET, "positive_class": 1, "features": features,
        "split_method": "two-stage stratified train_test_split; 60/40 then 50/50 of holdout",
        "missing_or_extreme_value_processing_before_split": "none",
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__},
        "reproduce_command": "python scripts/split_samples.py",
        "assignment_sha256": hashlib.sha256(frozen.read_bytes()).hexdigest(),
    }
    metadata_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # 表格与核查数字由真实运行生成，避免正文与 CSV 手工抄写不一致。
    TABLES.mkdir(parents=True, exist_ok=True)
    table = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"子集 & 样本数 & 好样本数 & 坏样本数 & 坏样本率 & 样本占比 \\", r"\midrule"]
    for row in summary.itertuples(index=False):
        table.append(f"{NAMES[row.split]} & {row.n:,} & {row.good:,} & {row.bad:,} & {100 * row.bad_rate:.3f}\\% & {100 * row.sample_share:.0f}\\% " + r"\\")
    table.extend([r"\midrule", f"合计 & {len(frame):,} & {int(frame[TARGET].eq(0).sum()):,} & {int(frame[TARGET].sum()):,} & {100 * frame[TARGET].mean():.3f}\\% & 100\\% " + r"\\", r"\bottomrule", r"\end{tabular}"])
    (TABLES / "split_summary.tex").write_text("\n".join(table) + "\n", encoding="utf-8")
    macros = {
        "SplitSeed": str(RANDOM_SEED),
        "LabeledDuplicateExcess": str(audit["duplicate_labeled_excess_rows"]),
        "FeatureDuplicateGroups": str(audit["duplicate_feature_groups"]),
        "FeatureDuplicateRows": f"{audit['rows_in_duplicate_feature_groups']:,}",
        "CrossSplitFeatureGroups": str(audit["cross_split_feature_groups"]),
    }
    (TABLES / "split_macros.tex").write_text("\n".join(
        f"\\newcommand{{\\{key}}}{{{value}}}" for key, value in macros.items()
    ) + "\n", encoding="utf-8")
    print(f"随机种子：{RANDOM_SEED}")
    print(summary.to_string(index=False))
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(f"划分及索引已保存：{OUTPUT}")


if __name__ == "__main__":
    main()
