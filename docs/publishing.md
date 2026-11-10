# 发布到 s0Ap224 的 GitHub

建议仓库名 `credit-risk-scorecard`。仓库简介可填写以下文字。

> Interpretable credit risk scoring with WOE, logistic regression and validation-based decision thresholds. Based on Kaggle Give Me Some Credit.

建议 Topics 为 `credit-risk`、`scorecard`、`woe`、`logistic-regression`、`kaggle`、`python`。

## 上传方式

在 [GitHub 新建仓库页面](https://github.com/new) 选择所有者 `s0Ap224`，名称填写 `credit-risk-scorecard`，按作品集用途选择 Public，创建空仓库。

本地目录已初始化 Git，运行以下命令提交并推送。身份配置使用你自己的 GitHub 信息，认证交给 Git 的登录提示处理。

```powershell
cd 'D:\statistical modeling\credit-risk-scorecard'
git add .
git diff --cached --stat
git commit -m "Add reproducible credit risk scorecard portfolio"
git remote add origin https://github.com/s0Ap224/credit-risk-scorecard.git
git push -u origin main
```

若通过网页上传，解压提供的发布 ZIP，将项目内文件与目录上传到仓库根目录。ZIP 仅包含发布文件，数据、运行日志、环境和逐户预测均已排除。

发布后，在 GitHub 个人主页的自定义置顶区域选择此仓库。README 会展示项目说明、模型表现、评分解释和复现步骤。

## 文件范围

`docs/results/` 是已经核对的汇总结果快照。`results/` 是本地复现产物，包含样本索引和逐户预测，由 `.gitignore` 排除。人工客户样例位于 `examples/customer.json`。课程材料、原始数据及作业论文保留在原工作区。
