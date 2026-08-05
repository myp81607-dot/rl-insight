# Prometheus 退化检测

这个实验只做一件事：直接读取已有 Prometheus 时序，训练健康基线，然后检测
`time` / `timing` 指标并给出关联 Top-5。

数据流如下：

```text
Prometheus /api/v1/series
  -> 获取全部具体时序的指标名和 labels
  -> 为每条时序生成精确 selector
  -> Prometheus /api/v1/query_range 获取历史 values
  -> 转为算法 TimeSeries
  -> global_step 对齐训练 step
  -> time/timing = target，其余 = candidate
  -> KDE 基线、异常检测、关联分析
```

同名指标只要 labels 不同，就会作为两条独立时序处理。脚本不会读取
`cpu_targets.yaml`、`npu_targets.yaml` 或 Grafana，也不会创建、聚合或修改指标。

## 1. 准备

进入仓库并激活环境：

```bash
cd /workspace/rl-insight-monitor
source .venv/bin/activate
pip install -e '.[degradation]'
```

先保证 Prometheus 已经有数据：

```bash
curl -sS http://127.0.0.1:9090/-/ready
curl -sS http://127.0.0.1:9090/api/v1/series \
  --data-urlencode 'match[]={__name__=~".+"}'
```

编辑 [config.yaml](./config.yaml)。最常改的是：

```yaml
prometheus_url: http://127.0.0.1:9090
baseline_steps: [5, 25]
poll_interval_seconds: 180
target_patterns: [time, timing]
top_k: 5
```

`127.0.0.1` 仅在脚本和 Prometheus 位于同一容器/网络命名空间时成立。如果脚本
在另一个容器或宿主机，把它改成可访问的 Prometheus 地址，例如
`http://prometheus:9090` 或 `http://服务器IP:9090`。

## 2. 查询全部时序

```bash
python -m experiment.degradation_perception query
```

结果写入 `output/query.json`，包含每条时序的指标名、labels、角色以及历史
timestamps/values。这个命令用于确认脚本实际读到了什么数据，不是训练前的必需步骤。

## 3. 训练

```bash
python -m experiment.degradation_perception train
```

脚本自动寻找名称含 `global_step` 的时序，用 step 5 到 25 对齐所有指标并训练。
结果写入 `output/training.json`，其中包含：

- 实际训练区间和观测到的 steps；
- 被选中的 global-step 时序；
- 成功训练的 target/candidate 及 labels；
- 因样本不足或无法拟合而跳过的时序；
- 完整 KDE 基线模型。

某一条时序失败只会被跳过，不会让其他时序一起失败；但至少要成功训练一个
`time` / `timing` target。

## 4. 检测一次

```bash
python -m experiment.degradation_perception detect
```

脚本读取 `output/training.json`，查询训练区间之后的数据，检测异常并默认执行
关联分析。精简结果写入 `output/anomalies.json`，重点看：

```text
status                 整体 normal / abnormal
lastGlobalStep         当前训练 step
targets                每个 time/timing target 的状态和异常区间
targets[].top5         最相关的五条 candidate 时序
abnormalContributionPercent  对该异常的贡献度百分比
anomalousMetrics       本轮所有异常指标汇总
```

## 5. 查看某个 target 的 Top-5

通用写法：

```bash
python -m experiment.degradation_perception analyze --target timing_s_step
```

也支持把唯一的指标片段直接写成参数：

```bash
python -m experiment.degradation_perception analyze --time_step
```

匹配结果必须唯一；如果同一指标有多组 labels，请把 `--target` 写得更具体。省略
`--target` 会显示全部 target。

## 6. 启动三分钟轮询

训练完成后运行：

```bash
python -m experiment.degradation_perception start
```

脚本每 180 秒重新查询并覆盖 `output/anomalies.json`。按 `Ctrl-C` 停止。关联分析
始终开启，不需要再启动第二个分析进程。

如果使用其他配置文件，所有命令都追加：

```bash
--config /path/to/config.yaml
```

`rl-insight server start` 的职责是启动 RL-Insight 服务和 Prometheus；本目录的
`start` 是异常检测轮询。前者已在运行且 9090 可访问时，不需要重复启动项目。

## 参数

连接、训练区间、轮询和角色识别在 `config.yaml` 修改。KDE 阈值、异常区间和关联
分析参数在 `algorithm_config.yaml` 修改；改动基线相关参数后重新运行 `train`。

核心算法仍保留在 `algorithm.py`、`baseline_model.py`、`kde_utils.py`、
`stable_segment_detector.py` 和 `association_analysis.py`，CLI 只负责 Prometheus
JSON 与算法 `TimeSeries` 之间的转换。
