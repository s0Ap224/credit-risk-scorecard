# 公开报告源稿

本目录包含[公开版报告](../credit-risk-scorecard-report.pdf)的 XeLaTeX 入口、章节、排版表格及三组矢量图。报告中的评分案例使用人工输入，通过已冻结模型重新计算。拟合、测试和阈值评估沿用原实验。

在本目录运行以下命令。报告使用 TeX Live 的 `ctexart` 与 Windows 字体配置，编译已在 Windows / TeX Live 2025 核验。PDF 图是报告插图，SVG 提供相同图表的可编辑版本。

```powershell
latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

编译生成 `build/main.pdf`。报告表格和插图已纳入源稿，可直接编译。需要复现统计量时，从仓库根目录按[项目说明](../../README.md)运行 `run_pipeline.py`。
