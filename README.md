# 凸优化课程大作业：梯度下降与直线搜索

完成两道指定作业：

1. 对 `f(x) = 0.5 * (x1^2 + gamma*x2^2)`，使用梯度下降与精确线搜索，验证课件的解析轨迹、条件数与收敛速度。
2. 对 `f(x) = exp(x1+3*x2-0.1) + exp(x1-3*x2-0.1) + exp(-x1-0.1)`，在同一初值下比较精确线搜索和不同 `alpha, beta` 的 Armijo 回溯线搜索。

这是可运行的实现、真实实验日志、完整分析报告和学习讲义。数值代码实现梯度下降外循环；SciPy 只用于精确搜索的一维导数求根。原课件不随仓库分发。

## 快速开始

推荐 Python 3.12。Linux/macOS：

```bash
git clone https://github.com/zhui711/convex-optim.git
cd convex-optim
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_experiments.py
python -m unittest discover -s tests -v
python build_report.py
```

Windows 激活命令为 `.venv\Scripts\activate`。项目不需要 GPU、训练数据或随机种子。

重跑会更新 `results/` 和 `figures/`，不会修改算法或学习讲义。也可把实验输出放入其他目录：

```bash
python run_experiments.py --output-dir /path/to/your/output
```

`build_report.py` 默认读取仓库内的 `results/`。中文图形优先使用 Noto Sans CJK、苹方、微软雅黑或黑体；缺少这些字体时需先安装一种中文字体。

## 先读什么

- **完整结果与代码附录**：[分析报告](docs/analysis_report.md)。
- **从零理解推导**：[逐步推导学习讲义](docs/derivation_guide.md)。
- **实现与日志核对**：[数值审阅记录](docs/numerical_review.md)。
- **数值核心**：[optimization.py](src/optimization.py)。
- **实验与全部图形**：[run_experiments.py](run_experiments.py)。

分析报告由实际 CSV 填入表格，并自动附入当前数值核心、完整实验绘图程序及验证程序。修改实验参数后，请同时重跑实验和 `build_report.py`，并检查解释文字是否仍匹配新的设定；正文针对本次固定网格撰写。

## 主要结果

二次问题初值为 `(gamma, 1)`，停止条件为 `f_k/f_0 <= 1e-8`。

| 条件数 | 1 | 2 | 3 | 10 | 100 | 1000 |
|---|---:|---:|---:|---:|---:|---:|
| 更新次数 | 1 | 9 | 14 | 46 | 461 | 4606 |

11 组 `gamma` 的实际次数均与解析公式一致，包括互为倒数的参数。

非二次问题所有方法的初值为 `(1, 1)`，停止条件为 `||grad f||_2 <= 1e-6`，解析最优解为 `(-log(2)/2, 0)`，最优值约为 `2.559266696658216`。精确搜索需 **21 次**更新。

| alpha \ beta | 0.3 | 0.5 | 0.7 | 0.9 |
|---|---:|---:|---:|---:|
| 0.01 | 18 | 33 | 202 | 186 |
| 0.10 | 18 | 32 | 36 | 51 |
| 0.30 | 19 | 33 | 32 | 38 |
| 0.49 | 22 | 32 | 24 | 18 |

精确搜索仅保证当前负梯度方向上的步长最优，不保证整条轨迹的更新次数最少。表格是指定函数、初值和阈值下的结果，不能推广成参数优劣的普遍排序。

![不同线搜索的实际轨迹](figures/fig03_nonquadratic_paths.svg)

![完整参数网格](figures/fig07_parameter_grid.svg)

## 文件结构

| 路径 | 内容 |
|---|---|
| `src/optimization.py` | 函数、导数、稳定差值、精确搜索、回溯搜索、梯度下降 |
| `run_experiments.py` | 全部实验与 8 组图形 |
| `tests/test_optimization.py` | 9 项数学与数值验证 |
| `results/*_summary.csv` | 二次与非二次实验汇总 |
| `results/*_history.csv` | 包含初始点在内的逐步日志 |
| `results/metadata.json` | Python/依赖版本、参数和记录口径 |
| `figures/*.svg` | 可编辑、可缩放的矢量图 |
| `docs/report_template.md` | 报告正文及结果表占位符 |
| `docs/analysis_report.md` | 已填入真实结果与源码的完整报告 |
| `docs/derivation_guide.md` | 逐步推导、首轮手算与自检问答 |
| `build_report.py` | 生成完整 Markdown 报告 |
| `render_docs.mjs` | 可选：将报告与学习讲义导出为离线 HTML/PDF |

PNG 预览图由程序重新生成；SVG 矢量图已存入仓库。

## 日志与成本定义

`k=0` 是初始点，第 `k` 行的 `step` 是从第 `k-1` 点更新到该点的步长。

- `f_calls`：目标值和稳定中心化值的联合评估次数，包含所有被拒绝的候选点。
- `g_calls`：初始点及接受迭代点上的完整梯度评估次数。
- `line_derivative_calls`：精确搜索内部的一维方向导数评估次数，其内部也包含指数和梯度相关运算。
- `ls_trials`：到达当前行的搜索次数；精确搜索为内部方向导数次数，回溯为候选目标值次数。两者成本不同。
- `backtracks`：到达当前行前的步长缩小次数；精确搜索为零。

不应把不同类型的调用简单相加后称为 FLOPs 或运行时间。精确搜索共 22 次目标评估、22 次接受点梯度评估、208 次方向导数评估；课件基准回溯 `(0.1,0.7)` 为 249、37、0。

## PDF/HTML 导出（可选）

数值实验不依赖文档导出工具。若要自行重新排版 PDF，请额外安装 Pandoc、Node.js、Playwright 和 Chromium：

```bash
npm install --no-save --package-lock=false playwright
CHROMIUM_PATH=/path/to/chromium node render_docs.mjs output
```

`CHROMIUM_PATH` 默认 `/usr/bin/chromium`。该步骤输出两份 PDF 和两份自包含 HTML；HTML 内嵌矢量图和 CSS，公式用 MathML，不依赖在线 MathJax。完整报告的代码附录可随 `build_report.py` 同步更新。

## 参考依据

用户提供的《第三章：无约束凸优化问题》PDF：第 19–20、25–27 页为算法；第 32–33 页为二次题；第 34 页为非二次例图；第 37 页为大作业。

补充教材：Boyd & Vandenberghe, *Convex Optimization*, Chapter 9, §9.2–9.3。[作者主页](https://web.stanford.edu/~boyd/cvxbook/)。
