# AutoBCI

**AI Agent system for autonomous scientific research.**
**让 AI Agent 自主进行科学研究的系统。**

Director thinks. Executor runs. Built for brain-computer interface, generalizable to any experimental research.

![SOTA Evolution](docs/assets/sota_evolution.png)

---

## What is this / 这是什么

AutoBCI is a research automation framework where two AI Agents collaborate to run scientific experiments autonomously:

- **Director** analyzes previous results, diagnoses why progress stalled, and decides the next research direction
- **Executor** sets up environments, modifies code, runs experiments, and writes back structured results
- They communicate through files — naturally persistent, observable, and crash-recoverable

AutoBCI 是一个研究自动化框架，由两个 AI Agent 协作完成科学实验：
Director 负责想：分析上一轮跑出来什么结果、为什么卡住了、下一步该往哪个方向走
Executor 负责做：配环境、改代码、跑实验、把结果写回来
两个 Agent 通过文件自动交接，晚上睡前启动，早上起来看结果
比如昨晚：睡前布置了步态脑电二分类任务，写好 program.md，先跑了一轮 plain 模型，所有参数组合都在 57% 左右，基本等于随机猜。Director 自己判断出这条路不通，自动切到 attention 机制，Executor 一夜之间把准确率拉到了 73.7%。提升 16 个百分点，全程没有人工干预。
系统已经跑了 800 多次自动实验，覆盖 12 个算法族。我们还在做一套叫 Carnese 的 benchmark，专门衡量研究harness框架自身的效率——多久能出一次突破、最长会卡多久、方向切换够不够多。

- 通过文件系统通信——天然持久、可观测、崩溃可恢复

## Recent result / 最近成果

On a gait-phase EEG binary classification task, Director detected that plain models were stuck near chance level (57.7%), autonomously switched to an attention-based approach, and Executor pushed accuracy to **73.7%** overnight — a 16-point improvement with zero human intervention.

在步态脑电二分类任务上，Director 发现 plain 模型全部接近随机水平（57.7%），自动切换到 attention 机制，Executor 一夜之间把准确率拉到 **73.7%**——提升 16 个百分点，全程无人工干预。

## Architecture / 架构

```
Director（想）                    Executor（做）
  读取实验结果                      读取指令
  诊断瓶颈                         配环境、改代码
  决定下一步方向          →          跑实验
  写出 program + tracks   文件交接    写回结果
                          ←              ↓
  再次分析...                       等待下一轮指令
```

## Framework benchmark / 框架基准

| Metric / 指标 | Value / 值 |
|--------|-------|
| Total iterations / 总迭代 | 800+ |
| Breakthrough rate / 突破率 | 1.3% |
| Cost per breakthrough / 每次突破成本 | 78.7 iterations |
| Direction diversity / 方向多样性 | 0.74 (12 families) |
| Direction switches / 方向切换 | 161 |

## Quick start / 快速开始

```bash
pip install -e .
python scripts/serve_dashboard.py --port 8878
autobci-agent direct
autobci-agent supervise --director-enabled --foreground
```

## Context / 背景

This project grew out of hands-on BCI work — craniotomy, electrode implantation, EEG acquisition, motion capture, and dataset creation for the China BCI Competition.


Core insight: EEG signals are lossy observations of a perpetually drifting biological system. We need dynamic systems to counter dynamic reality.


## License

Apache 2.0
