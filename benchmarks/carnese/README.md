# Benchmark Carnese

Carnese 是 AutoResearch 的框架基准台，不复用主仓当前研究记忆。

## V0：步态阶段划分

V0 的第一条任务包是：

- `benchmarks/carnese/tasks/gait_phase_v1/`

## 独立 seed 克隆的标准步骤

1. clone 到独立目录
2. 用 `scripts/materialize_carnese_seed.py` 把 benchmark-mode 默认文件铺到 clone
3. 软链接 `data/` 和 `.venv/`
4. 用 `scripts/run_carnese_gait_phase_campaign.py` 起固定 16 轮阶段 0 标签工程 benchmark

示例：

```bash
python scripts/materialize_carnese_seed.py --target-root ~/Code/AutoBci-Carnese-v0

cd ~/Code/AutoBci-Carnese-v0
ln -s ~/Code/AutoBci/data data
ln -s ~/Code/AutoBci/.venv .venv
mkdir -p artifacts/monitor

python scripts/run_carnese_gait_phase_campaign.py --campaign-id gait-phase-benchmark-v0

# 如需先做多方法人工 bootstrap：
python scripts/compare_gait_phase_rule_methods.py \
  --dataset-config configs/datasets/gait_phase_clean64_smoke.yaml \
  --output-json /tmp/gait_phase_rule_compare_smoke.json
```

## 关键文件

- benchmark-mode 总纲：`docs/CONSTITUTION_BENCHMARK_GAIT_PHASE.md`
- benchmark program：`tools/autoresearch/program.gait_phase.md`
- 当前附录：`tools/autoresearch/program.gait_phase.current.md`
- 最小 track：`tools/autoresearch/tracks.gait_phase.json`
- 阶段 0 pipeline：`scripts/run_gait_phase_label_engineering.py`
- 多方法对照：`scripts/compare_gait_phase_rule_methods.py`
- 脑电侧评分脚本（阶段 1 以后）：`scripts/eval_gait_phase.py`
- 参考标签：`scripts/build_gait_phase_reference_labels.py`
