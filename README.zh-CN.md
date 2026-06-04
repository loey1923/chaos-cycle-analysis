# 混沌置乱的循环阶分析

[![English](https://img.shields.io/badge/lang-English-blue.svg)](README.md)
[![中文](https://img.shields.io/badge/lang-中文-red.svg)](README.zh-CN.md)



利用混沌映射通过 `np.argsort` 生成置乱表（置换），系统分析其循环分解结构（循环长度分布、总阶 = LCM），并结合 Landau 上界参考线、有限精度退化效应、种子初值敏感性和排序偏差（argsort 主导性）评估其密码安全性。

---

## 项目背景

混沌映射因其初值敏感性和伪随机性被广泛用于图像加密中的像素置乱。本项目从**置换的循环结构**这一关键代数性质出发，系统评估混沌排序置乱的安全性：

- 混沌排序置乱的**阶**与 Fisher-Yates 均匀随机置换相比如何？是否接近 Landau 上界？
- 计算机**有限精度**（float32 vs float64 vs Decimal）是否导致安全性退化？
- `np.argsort` 是否主导循环结构，使混沌映射间的差异被抹除？
- 种子微扰能否引发置换的**初值敏感性变化**？

---

## 混沌映射（5 种）

| 映射 | 公式 | 参数 | 定义域 |
|------|------|------|--------|
| Logistic | $x_{n+1} = \mu x_n(1-x_n)$ | $\mu = 3.99$ | $(0,1)$ |
| Tent | $x_{n+1} = r\min(x_n, 1-x_n)$ | $r = 1.99$ | $(0,1)$ |
| Chebyshev (k=3) | $T_3(x) = 4x^3 - 3x$ | $k = 3$ | $(-1,1)$ |
| Sine | $x_{n+1} = r\sin(\pi x_n)$ | $r = 0.99$ | $(0,1)$ |
| Henon | $(x,y) \to (1-ax^2+y, bx)$ | $a=1.4, b=0.3$ | $x\in(-1.5,1.5)$ |

所有映射在收集 N 个样本前统一**预热 M=1000** 轮以消除暂态。

---

## 实验设计

| # | 实验 | 目的 |
|---|------|------|
| E1 | 平均阶 vs N 曲线 | 横向对比 5 种映射的循环特性 |
| E2 | Landau 上界参考 | $S_N$ 中最大可能阶，作为上界基准线 |
| E3 | 种子初值敏感性 | DiffRate + Spearman Footrule |
| E4 | 参数扫描 | Logistic $\mu$ 扫描，检测周期窗口 |
| E5 | 循环长度分布 | N=1024，各映射循环直方图 |
| E6 | 有限精度退化 | float32 vs float64 vs Decimal(50) |
| E7 | 排序偏差分析 | Chaos+argsort vs Random+argsort vs Fisher-Yates |

---

## 项目结构

```
chaos-cycle-analysis/
├── src/
│   ├── maps.py             # 5 种混沌映射实现
│   ├── permutation.py      # 置乱表生成 + 循环分解
│   ├── analysis.py         # 批量实验 + Landau + 初值敏感性 + 精度 + 偏差
│   └── visualize.py        # 6 张核心图
├── data/                   # 实验结果 CSV
├── figures/                # 输出图片 PNG
├── report/                 # LaTeX 报告源码
├── scripts/                # 辅助验证脚本
├── main.py                 # 实验入口
├── requirements.txt
└── .gitignore
```

---

## 输出图表

| 文件 | 内容 |
|------|------|
| `fig1_avg_ln_order.png` | 5 映射 $\mathbb{E}[\ln(\text{order})]$ vs N + Landau 参考线 + Fisher-Yates 基线 |
| `fig2_cycle_distribution.png` | N=1024 循环长度分布直方图 |
| `fig3_seed_avalanche.png` | 初值敏感性：各映射 DiffRate + Footrule 柱状图 |
| `fig4_precision_degradation.png` | float32/float64/Decimal(50) 精度对比 |
| `fig5_sorting_bias.png` | Chaos vs Random vs Fisher-Yates 箱线图 |
| `fig6_summary_metrics.png` | 多指标安全性热力图 |

---

## 快速开始

```bash
pip install -r requirements.txt
python main.py
```

CSV 结果写入 `data/`，图片写入 `figures/`。

---

## 依赖

- Python 3.9+（使用 `math.lcm`）
- numpy, pandas, matplotlib, seaborn

---

## 参考文献

1. Strogatz S H. *Nonlinear Dynamics and Chaos*. CRC Press, 2018.
2. Landau E. Über die Maximalordnung der Permutationen gegebenen Grades. *Archiv der Mathematik und Physik*, 1903.
3. Shepp L A, Lloyd S P. Ordered cycle lengths in a random permutation. *Transactions of the AMS*, 1966.
4. Arroyo D, et al. Chaos-based cryptography: A brief overview. *IEEE CAS Magazine*, 2021.
