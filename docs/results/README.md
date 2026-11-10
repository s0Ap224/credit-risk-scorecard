# 实验汇总表

这些文件来自原项目的固定样本划分和冻结模型。字段包含分组统计、模型指标及规则。完整计算步骤见[报告](../credit-risk-scorecard-report.pdf)。

| 内容 | 文件 |
|---|---|
| 分箱规则和训练集箱内样本统计 | [分箱统计](binning_table.csv) |
| IV 排名、特殊档贡献及入模标记 | [IV 汇总](iv_summary.csv) |
| 候选特征集的验证表现及删变量过程 | [候选验证](candidate_validation.csv) · [筛选记录](selection_trials.csv) |
| 模型区分度、校准和阈值评价 | [模型对比](model_comparison.csv) · [校准分组](calibration_groups.csv) |
| 风险分层 | [测试集风险分组](risk_groups.csv)。排序最靠前的一组约占测试集的 10%，覆盖 52.77% 的坏样本。 |
| 评分与决策 | [评分卡](scorecard.csv) · [风险标记](test_flag_comparison.csv) · [审批结果](approval_comparison.csv) |
| 分数分布对照 | [PSI 分档](psi_by_band.csv)。训练分数分档后应用到测试集，PSI 总值为 0.000418。 |
