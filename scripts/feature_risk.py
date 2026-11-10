"""模块二：四个变量的训练集分组分布、坏样本率及缺失状态比较。

完整复现：python scripts/feature_risk.py
只计算统计：python scripts/feature_risk.py --stats-only
固定展示区间仅用于 EDA，不作为最终评分卡分箱；不修改原始数据或固定划分。
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from audit_data import load_inputs, sha256

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "module2" / "risk"
FIGURES = ROOT / "figures" / "risk"
TABLES = ROOT / "latex" / "tables"
TARGET = "SeriousDlqin2yrs"
FEATURES = ["DebtRatio", "MonthlyIncome", "NumberOfDependents", "NumberOfTime60-89DaysPastDueNotWorse"]
ORDERS = {
    FEATURES[0]: ["0", "(0,0.2]", "(0.2,0.4]", "(0.4,0.6]", "(0.6,1]", "(1,10]", "(10,100]", ">100"],
    FEATURES[1]: ["0", "(0,10)", "[10,2000)", "[2000,4000)", "[4000,6000)", "[6000,10000)", ">=10000", "missing"],
    FEATURES[2]: ["0", "1", "2", "3", ">=4", "missing"],
    FEATURES[3]: ["0", "1", "2", ">=3", "96/98"],
}
DISPLAY = {
    "DebtRatio": ORDERS[FEATURES[0]],
    "MonthlyIncome": ["0", "(0,10)", "[10,2千)", "[2千,4千)", "[4千,6千)", "[6千,1万)", "≥1万", "缺失"],
    "NumberOfDependents": ["0", "1", "2", "3", "≥4", "缺失"],
    "NumberOfTime60-89DaysPastDueNotWorse": ["0", "1", "2", "≥3", "96/98（特殊）"],
}
TITLES = ["(a) 负债收入比", "(b) 月收入（原始单位）", "(c) 家属人数", "(d) 60–89 天历史逾期次数"]
SIZE = (16 / 2.54, 9.5 / 2.54)


def wilson(bad: int, n: int) -> tuple[float, float]:
    """未调整的记录层面二项比例 Wilson 95% 区间，不进行显著性检验。"""
    if n <= 0:
        raise ValueError("不能对空组估计坏样本率。")
    p = bad / n
    z = NormalDist().inv_cdf(0.975)
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def group_labels(train: pd.DataFrame) -> dict[str, pd.Series]:
    """显式固定边界，不使用标签搜索切点；每个变量的所有训练记录均被纳入。"""
    debt, income, dependents, late = (train[name] for name in FEATURES)
    if train[FEATURES].lt(0).any().any() or debt.isna().any() or late.isna().any():
        raise ValueError("本次分组预期比例和计数非负，且 DebtRatio、逾期计数无缺失。")
    selections = {
        FEATURES[0]: np.select(
            [debt.eq(0), debt.le(.2), debt.le(.4), debt.le(.6), debt.le(1), debt.le(10), debt.le(100)],
            ORDERS[FEATURES[0]][:-1], default=">100"),
        FEATURES[1]: np.select(
            [income.isna(), income.eq(0), income.lt(10), income.lt(2000), income.lt(4000), income.lt(6000), income.lt(10000)],
            ["missing", "0", "(0,10)", "[10,2000)", "[2000,4000)", "[4000,6000)", "[6000,10000)"], default=">=10000"),
        FEATURES[2]: np.select(
            [dependents.isna(), dependents.eq(0), dependents.eq(1), dependents.eq(2), dependents.eq(3)],
            ["missing", "0", "1", "2", "3"], default=">=4"),
        FEATURES[3]: np.select(
            [late.isin([96, 98]), late.eq(0), late.eq(1), late.eq(2)],
            ["96/98", "0", "1", "2"], default=">=3"),
    }
    return {name: pd.Series(values, index=train.index) for name, values in selections.items()}


def group_statistics(train: pd.DataFrame, labels: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for feature, groups in labels.items():
        for order, label in enumerate(ORDERS[feature]):
            y = train.loc[groups.eq(label), TARGET]
            n, bad = len(y), int(y.sum())
            lower, upper = wilson(bad, n)
            rows.append({
                "feature": feature, "group_order": order, "group": label,
                "display_label": DISPLAY[feature][order], "n": n, "good": n - bad,
                "bad": bad, "sample_share": n / len(train), "bad_rate": bad / n,
                "wilson95_lower": lower, "wilson95_upper": upper,
                "special_group": label in {"missing", "96/98"},
            })
        subset = pd.DataFrame(rows).loc[lambda t: t.feature.eq(feature)]
        if subset.n.sum() != len(train) or subset.bad.sum() != train[TARGET].sum():
            raise AssertionError(f"分组未互斥完整覆盖训练集：{feature}")
    return pd.DataFrame(rows)


def write_statistics(train: pd.DataFrame, groups: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    OUT.mkdir(parents=True, exist_ok=True)
    groups.to_csv(OUT / "risk_groups.csv", index=False, encoding="utf-8-sig")
    missing_rows = []
    for name in ["MonthlyIncome", "NumberOfDependents"]:
        for status, mask in [("missing", train[name].isna()), ("observed", train[name].notna())]:
            y = train.loc[mask, TARGET]
            n, bad = len(y), int(y.sum())
            lower, upper = wilson(bad, n)
            missing_rows.append({"feature": name, "status": status, "n": n, "bad": bad,
                                 "bad_rate": bad / n, "wilson95_lower": lower, "wilson95_upper": upper})
    missing = pd.DataFrame(missing_rows)
    missing.to_csv(OUT / "missing_risk.csv", index=False, encoding="utf-8-sig")
    income = train.MonthlyIncome
    state = pd.Series(np.select([income.isna(), income.eq(0), income.lt(10)],
                               ["missing", "zero", "0<income<10"], default="income>=10"), index=train.index)
    cross_rows = []
    for label in ["missing", "zero", "0<income<10", "income>=10"]:
        for high in [False, True]:
            mask = state.eq(label) & train.DebtRatio.gt(100).eq(high)
            y = train.loc[mask, TARGET]
            n, bad = len(y), int(y.sum())
            lower, upper = wilson(bad, n)
            cross_rows.append({"income_state": label, "debt_above_100": high, "n": n, "bad": bad,
                               "bad_rate": bad / n, "wilson95_lower": lower, "wilson95_upper": upper})
    cross = pd.DataFrame(cross_rows)
    cross.to_csv(OUT / "debt_income_cross.csv", index=False, encoding="utf-8-sig")
    return missing, cross


def write_macros(groups: pd.DataFrame, missing: pd.DataFrame, cross: pd.DataFrame) -> None:
    indexed = groups.set_index(["feature", "group"])
    missing_index = missing.set_index(["feature", "status"])
    percent = lambda value: f"{100 * float(value):.2f}\\%"
    spec = {
        "DebtMiddleRisk": (FEATURES[0], "(0.2,0.4]"), "DebtHigherRisk": (FEATURES[0], "(0.6,1]"),
        "DebtTailRisk": (FEATURES[0], ">100"), "IncomeMiddleRisk": (FEATURES[1], "[2000,4000)"),
        "IncomeHighRisk": (FEATURES[1], ">=10000"), "IncomeZeroRisk": (FEATURES[1], "0"),
        "IncomeVeryLowRisk": (FEATURES[1], "(0,10)"), "DependentsZeroRisk": (FEATURES[2], "0"),
        "DependentsManyRisk": (FEATURES[2], ">=4"), "LateZeroRisk": (FEATURES[3], "0"),
        "LateOneRisk": (FEATURES[3], "1"), "LateTwoRisk": (FEATURES[3], "2"),
        "LateManyRisk": (FEATURES[3], ">=3"), "LateSpecialRisk": (FEATURES[3], "96/98"),
    }
    values = {key: percent(indexed.loc[pair, "bad_rate"]) for key, pair in spec.items()}
    for key, feature in [("Income", FEATURES[1]), ("Dependents", FEATURES[2])]:
        for status in ["missing", "observed"]:
            values[f"{key}{status.title()}Risk"] = percent(missing_index.loc[(feature, status), "bad_rate"])
    high = cross.loc[cross.debt_above_100]
    values["DebtTailSpecialShare"] = percent(high.loc[high.income_state.ne("income>=10"), "n"].sum() / high.n.sum())
    values["LateSpecialN"] = str(int(indexed.loc[(FEATURES[3], "96/98"), "n"]))
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "risk_macros.tex").write_text("\n".join(
        f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in values.items()
    ) + "\n", encoding="utf-8")


def plot_risk(groups: pd.DataFrame, baseline: float) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    from PIL import Image

    sys.path.insert(0, str(ROOT / "utils"))
    from setup_style import setup_style
    from export_figure import export_figure
    from visual_qa import audit_layout
    from plot_style import PALETTE

    style = setup_style(journal="general", lang="zh", serif_for_zh=True,
                        use_sciplots=False, constrained_layout=False)
    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5,
                         "ytick.labelsize": 7, "axes.linewidth": .5, "savefig.bbox": None,
                         "font.serif": ["SimSun", "Times New Roman"]})
    style.update({"cjk_font": "SimSun", "latin_font": "Times New Roman"})
    fig = plt.figure(figsize=SIZE)
    grid = fig.add_gridspec(2, 2, left=.015, right=.985, top=.90, bottom=.10, wspace=.17, hspace=.62)
    limits = {}
    for panel, feature in enumerate(FEATURES):
        sub = groups.loc[groups.feature.eq(feature)].sort_values("group_order")
        inner = grid[panel // 2, panel % 2].subgridspec(1, 3, width_ratios=[.85, 1.35, 1.05], wspace=.04)
        labels_ax, ax, count_ax = [fig.add_subplot(inner[0, j]) for j in range(3)]
        axes = [labels_ax, ax, count_ax]
        for current in axes:
            current.set_ylim(len(sub) - .5, -.5)
        labels_ax.axis("off")
        count_ax.axis("off")
        pos = grid[panel // 2, panel % 2].get_position(fig)
        fig.text(pos.x0, pos.y1 + .065, TITLES[panel], fontsize=8, fontweight="bold", va="bottom")
        labels_ax.text(.0, 1.035, "取值组", transform=labels_ax.transAxes, fontsize=6.5, color="#555555")
        count_ax.text(.5, 1.035, "人数（占比）", transform=count_ax.transAxes, ha="center", fontsize=6.5, color="#555555")
        top = max(.15, float(np.ceil(sub.wilson95_upper.max() * 100 / 5) * 5) / 100)
        limits[feature] = [0, top]
        ax.set_xlim(0, top * 100)
        ax.set_yticks([])
        ax.xaxis.set_major_locator(MultipleLocator(5 if top <= .20 else 20))
        ax.set_xlabel("坏样本率（%）", labelpad=2)
        ax.spines[["left", "right", "top"]].set_visible(False)
        ax.tick_params(axis="x", length=2, width=.5, pad=1.5)
        ax.axvline(baseline * 100, linestyle="--", linewidth=.65, color=PALETTE["neutral"], zorder=1)
        for row_position, row in enumerate(sub.itertuples(index=False)):
            color = PALETTE["secondary"] if row.special_group else PALETTE["primary"]
            ax.errorbar(row.bad_rate * 100, row_position,
                        xerr=[[100 * (row.bad_rate - row.wilson95_lower)], [100 * (row.wilson95_upper - row.bad_rate)]],
                        fmt="s" if row.special_group else "o", markersize=3.5,
                        color=color, ecolor=color, elinewidth=.85, capsize=1.8, zorder=3)
            labels_ax.text(.0, row_position, row.display_label, va="center", fontsize=7,
                           color="#222222")
            count_ax.text(.5, row_position, f"{row.n:,} ({row.sample_share * 100:.1f}%)",
                          va="center", ha="center", fontsize=6.5, color="#333333")
            if row.special_group:
                for current in axes:
                    current.axhline(row_position - .5, color="#999999", linewidth=.4, linestyle=":")
    issues = audit_layout(fig)
    if issues:
        plt.close(fig)
        raise RuntimeError(f"图形布局检查未通过：{issues}")
    FIGURES.mkdir(parents=True, exist_ok=True)
    base = FIGURES / "feature_risk"
    paths = export_figure(fig, str(base), formats=["pdf", "svg", "png"],
                          size_inches=SIZE, dpi=300, tight=False, grayscale_preview=False)
    # Type0 的字体描述符位于子字体，直接提取嵌入数据核验，避免只看父对象造成误报。
    import pymupdf
    with pymupdf.open(base.with_suffix(".pdf")) as pdf:
        fonts = []
        for font in pdf[0].get_fonts(full=True):
            embedded = pdf.extract_font(font[0])[3]
            if not embedded or font[2] == "Type3":
                raise RuntimeError(f"矢量图字体未正确嵌入：{font}")
            fonts.append({"name": font[3], "type": font[2], "embedded_bytes": len(embedded)})
    # 保持最终画布尺寸和 DPI，避免旧版导出器的灰度辅助函数重新 tight-crop 彩色图。
    gray_path = FIGURES / "feature_risk_grayscale.png"
    with Image.open(base.with_suffix(".png")) as image:
        image.convert("L").save(gray_path, dpi=(300, 300))
    plt.close(fig)
    latex_image = ROOT / "latex" / "figures" / "feature_risk.pdf"
    latex_image.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(base.with_suffix(".pdf"), latex_image)
    return {"style": style, "size_inches": SIZE, "dpi": 300, "layout_issues": issues, "embedded_fonts": fonts,
            "x_axis_limits_in_probability_units": limits,
            "outputs": {str(Path(path).relative_to(ROOT)): sha256(Path(path)) for path in [*paths, str(gray_path)]}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats-only", action="store_true", help="仅生成统计量，供正式绘图前核验")
    args = parser.parse_args()
    frame, assignment, _, split_meta, inputs = load_inputs()
    train = frame.loc[assignment.split.eq("train")].copy()
    labels = group_labels(train)
    groups = group_statistics(train, labels)
    missing, cross = write_statistics(train, groups)
    write_macros(groups, missing, cross)
    baseline = float(train[TARGET].mean())
    contract = {
        "core_claim": "四变量与训练集边际风险存在非线性及缺失状态差异，为后续缺失处理比较提供依据。",
        "panels": dict(zip(FEATURES, ["比例尾部风险与收入状态混合", "收入梯度与缺失组", "家属人数梯度与缺失组", "历史逾期梯度与特殊编码"])),
        "layout": "2x2 quantitative grid; each panel combines a horizontal risk interval plot with group counts and shares",
        "size_cm": [16, 9.5], "formats": ["pdf", "svg", "png"], "dpi": 300,
        "scope": "fixed training set only", "n": len(train), "group_rules": ORDERS,
        "interval": "record-level Wilson 95% binomial interval; descriptive, not an adjusted model effect",
        "baseline": baseline, "special_markers": "squares for missing and 96/98 groups",
        "binning_status": "fixed EDA display groups, not optimized or final scorecard bins",
    }
    (OUT / "figure_contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "inputs": inputs, "scope": "train", "n": len(train), "bad": int(train[TARGET].sum()),
        "baseline_bad_rate": baseline, "split_seed": split_meta["random_seed"], "new_random_sampling": False,
        "features": FEATURES, "data_cleaning": "none", "training_group_rules": ORDERS,
        "command": "python scripts/feature_risk.py", "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "matplotlib", "Pillow", "PyMuPDF"]},
        "code_hashes": {"scripts/feature_risk.py": sha256(Path(__file__)), "scripts/audit_data.py": sha256(ROOT / "scripts/audit_data.py")},
    }
    if not args.stats_only:
        manifest["figure"] = plot_risk(groups, baseline)
        manifest["helper_hashes"] = {str(path.relative_to(ROOT)): sha256(path) for path in (ROOT / "utils").glob("*.py")}
    for entry in inputs.values():
        if sha256(ROOT / entry["path"]) != entry["sha256"]:
            raise RuntimeError("原始数据或固定索引在运行过程中发生变化。")
    (OUT / "risk_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(groups[["feature", "group", "n", "bad", "bad_rate"]].to_string(index=False))
    print("\n缺失状态比较：\n" + missing.to_string(index=False))
    print(f"\n训练集基准坏样本率：{baseline:.6%}\n结果目录：{OUT}")


if __name__ == "__main__":
    main()
