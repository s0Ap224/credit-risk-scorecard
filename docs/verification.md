# 复现检查

2026-11-10 在 Windows / Python 3.12.14 环境完成以下检查，依赖版本见根目录 `requirements.txt`。

| 检查 | 结果 |
|---|---|
| 独立目录，从原始 CSV 与 XLS 开始运行 `run_pipeline.py` | 11 个阶段全部完成，退出码 0 |
| `python -m unittest discover -s tests -v` | 27 项测试全部通过 |
| `python scripts/evaluate_models.py --smoke` | 通过 |
| `python scripts/decision_thresholds.py smoke` | 通过 |
| 人工客户 JSON 经评分 CLI 推理 | 成功，输出见 `examples/customer_score.json` |
| 模型比较、评分卡、风险阈值、审批结果与发布快照比对 | 四张表全部一致，浮点容差 atol=1e-12、rtol=1e-10 |
| 三张展示图的视觉检查 | 文字、坐标、图例与布局可读 |

本轮完整运行使用默认数值模式。展示图沿用原项目已导出的文件，重新绘图选项及其他操作系统尚未在作品集副本中完整复跑。发布 ZIP 包含源代码、文档、人工示例及汇总快照。原始数据、逐户输出和运行日志保留在本地。
