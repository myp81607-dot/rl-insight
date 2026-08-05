# RL-Insight 劣化感知：VeRL 日志运行指南

本目录保留现有 KDE、平稳段、多模态、异常点、异常区间和关联分析算法，并将
真实 VeRL step-metrics 日志运行明确拆成三步：

- `fit`：读取正常 `healthy.log`，生成可复用的 `baseline_model.json`；
- `detect`：加载基线和本地 inference 日志，生成一次性的 `result.json`；
- `monitor`：启动时加载同一基线，通过 SSH 定时轮询远程 inference 日志。

持续检测是 SSH 轮询，实际延迟取决于配置的轮询间隔，不是真正的流式推送。

## 环境

在仓库根目录安装算法依赖：

```bash
python -m pip install -e ".[degradation]"
```

只有 SSH 轮询需要额外安装 Paramiko：

```bash
python -m pip install -r experiment/degradation_perception/requirements-remote.txt
```

## 日志格式

每条有效记录的第一段必须是 `step:<N>`：

```text
step:0 - global_seqlen/min:32117 - actor/entropy:4.9654 - timing_s/step:38.46
step:1 - global_seqlen/min:32410 - actor/entropy:4.9121 - timing_s/step:39.02
```

`step` 作为时间戳。指标名保留原始 `/`、`_` 和 `.`。非法行、非法字段、
NaN 和 Inf 会被跳过；一个指标缺点不会给它补零，也不会影响同一行其他指标。
各指标分别排序并去重，重复 step 复用现有规则：保留日志中最后一个有效值。

统一解析入口位于 `training_log.py`：

- `parse_verl_training_log_line`：解析一行；
- `parse_verl_training_log_with_metadata`：同时返回时序和有效 step；
- `parse_verl_training_log`：返回 `dict[str, TimeSeries]`；
- `load_verl_training_log`：读取本地 UTF-8 日志。

## 1. fit：学习并保存正常基线

先查看健康日志中成功解析到的原始指标名：

```bash
python -m experiment.degradation_perception.main \
  --standard-log healthy.log \
  --list-metrics
```

输出每行只有一个指标名，按第一次出现顺序去重，例如：

```text
global_seqlen/min
actor/entropy
timing_s/step
```

执行拟合：

```bash
python -m experiment.degradation_perception.main fit \
  --standard-log healthy.log \
  --metrics timing_s/step actor/entropy \
  --policy-config algorithm_config.yaml \
  --output-model models/baseline_model.json
```

`baseline_model.json` 保存每个指标的完整策略和所有独立正常模式。多模式不会被
平均或合并。写入使用同目录临时文件和原子替换；已有模型会先备份到同级
`history/`。新模型生成或校验失败时，不会损坏当前模型。

## 2. detect：加载基线进行离线检测

复制并修改 [real_test.example.json](./real_test.example.json)。其中所有相对路径均
相对于配置文件所在目录解析；`output` 的父目录不存在时会自动创建。

```json
{
  "baseline_model": "./models/baseline_model.json",
  "policy_config": "./algorithm_config.yaml",
  "inference_log": "./data/inference.log",
  "metrics": ["timing_s/step", "actor/entropy"],
  "task_id": "real-test-001",
  "debug_kde": true,
  "output": "./results/result.json"
}
```

```bash
python -m experiment.degradation_perception.main detect \
  --config real_test.json
```

此阶段只读取 `baseline_model.json` 和 `inference_log`，不读取 `healthy.log`，也不
重新拟合 KDE。inference 只用于本次检测，不能更新基线。配置中的 `output` 会保存
标准 JSON `result.json`；终端阈值可以按 `debug_kde` 展示两位小数，文件保留完整
浮点精度。

## 3. monitor：SSH 轮询检测

复制脱敏配置并填写测试环境信息，不要提交真实地址、用户名、密码、私钥内容或
内部日志路径：

```bash
cp experiment/degradation_perception/monitor_config.example.yaml monitor_config.yaml
cp experiment/degradation_perception/algorithm_config.yaml algorithm_config.yaml
```

持续运行：

```bash
python -m experiment.degradation_perception.remote_monitor \
  --config monitor_config.yaml
```

只执行一轮连接和配置检查：

```bash
python -m experiment.degradation_perception.remote_monitor \
  --config monitor_config.yaml \
  --once
```

轮询过程如下：

```text
启动时读取一次本地 baseline_model.json
→ 通过 Paramiko SFTP 只读获取远程 .log 快照
→ 丢弃尚未换行的活动日志尾行
→ 只接受 step > last_step 的选中指标
→ 累积到固定长度的 inference 滚动窗口
→ 每个指标都达到算法最小点数后执行检测
→ 原子保存本轮结果 JSON
→ 将 lastStep 和当前缓冲区一起原子保存到 state_file
→ 等待下一轮
```

数据不足时会保留在缓冲区，且不会生成伪异常结果。无新增 step 时不会重复检测。
滚动窗口由 `inference_window_steps` 限制，不会无限增长；重启后从 `state_file`
恢复 `lastStep` 和未完成缓冲。SSH、检测或写盘失败会记录错误并等待下一轮；失败
轮次不会提交候选状态。在线模式不需要也不会读取 `healthy.log`，远端文件只读。

## 4. Prometheus 轮询与自动基线

Prometheus 模式是已有 Prometheus HTTP API 的只读客户端，不会启动新端口。复制并
修改示例配置，确保每条 PromQL 通过 `project`、`experiment_name`、`job`、
`instance` 等标签唯一定位一条时序；如果确实需要合并副本，请在 PromQL 中显式
使用 `sum`、`avg` 等聚合：

```bash
cp experiment/degradation_perception/prometheus_monitor_config.example.yaml \
  prometheus_monitor_config.yaml
cp experiment/degradation_perception/algorithm_config.yaml algorithm_config.yaml
```

先执行一轮真实连接和数据检查：

```bash
python -m experiment.degradation_perception.prometheus_monitor \
  --config prometheus_monitor_config.yaml \
  --once
```

持续轮询：

```bash
python -m experiment.degradation_perception.prometheus_monitor \
  --config prometheus_monitor_config.yaml
```

`poll_interval_seconds` 默认是 180 秒，只控制多久发起一次请求；
`prometheus.query_step_seconds` 默认是 15 秒，控制每次 `query_range` 返回的采样
网格。每轮会从上次游标向前重叠两个采样间隔并按时间戳去重，不能把 query step
设置成三分钟。

如果 `baseline_model` 不存在，监控器会在 `lookback_seconds` 范围内查询
`global_step_query`，默认将 global step 5 至 25 各取一个对齐样本，完成覆盖率、
样本数和平稳段检查后调用现有 `fit_baseline_model`。生成的基线随即冻结，后续
inference 不会更新它。基线已存在时直接加载，不会重新拟合。

指标名称不区分大小写地包含 `time` 或 `timing` 时默认作为 target，其余指标默认
作为 association candidate。配置中的 `role` 或 `target_selection.overrides` 可以
显式覆盖。每次检测都会原子覆盖 `<task_id>_latest.json`；只有 target 在 NORMAL、
ABNORMAL、RECOVERED 之间发生状态变化时，才会额外生成带稳定 event ID 的
`event_*.json`，因此相同故障不会每三分钟重复上报。

状态文件同时绑定 Prometheus 查询配置指纹和 baseline SHA-256。修改指标查询、
target 规则、基线 step 范围或替换基线后，需要移动或删除旧状态文件；修改轮询
周期或输出目录不要求重新训练基线。认证信息只能通过配置中指定的环境变量读取，
不要把 token 或密码写入 YAML。

## 算法与输出

```text
fit：读取正常日志
→ 解析 step metrics
→ 找到平稳片段
→ KDE 学习正常分布
→ 计算每个指标的标准区间
→ 保存 baseline_model.json
detect / monitor：加载 baseline_model.json
→ 读取待检测日志
→ 标记异常点
→ 合并异常区间
→ 输出 JSON
```

核心返回字段保持不变：

- `states`：指标检测流程状态；`states[metric] == 0` 只表示流程完成，不表示一定异常；
- `abnormalTimeRange`：经现有规则确认的正式异常区间；
- `results`：当前核心实现的检测详情、逐点诊断和阈值；
- `metricErrors`：可选的指标级错误；
- `associationAnalysis`：仅在现有配置明确开启时出现。

适配层增加 `debugDetails.metrics`。每个成功建模的指标分别保存：

- `alpha`、实际用于上下分位阈值的 `lowerProbability` 和 `upperProbability`；
- standard/inference 有效点数；
- 异常点数和异常区间数；
- 每个正常模型各自的原始 KDE 分位区间、最终判定区间、ratio 和 bandwidth；
- 平稳 segment 的 step 范围和峰影响区间。

多正常模式保存在 `normalModels` 的多个独立元素中，不取平均、不合并阈值。
在线 JSON 还包含 `metadata`：任务轮次、本轮新增 step 范围、检测前 inference
缓冲范围、检测时间、各指标点数以及 `lastStepBefore/lastStepAfter`。

## 注意事项

- `healthy.log` 必须是已确认正常的数据，只在 `fit` 阶段读取；
- inference 数据不会进入正常基线，且不保证一定出现异常；
- 在线和离线检测使用同一个 `baseline_model.json`；
- 更换设备、模型、卡数、代码版本或训练配置后，通常需要重新 `fit`；
- 日志中的时间表示 step，SSH 数据仍以 `source_type=training_log` 调用算法；
- 指标名始终保留原始形式；
- 内部 KDE、分位数和阈值判断不做提前四舍五入；
- SSH 默认拒绝未知 host key，可使用系统 known_hosts 或配置 `known_hosts`；
- 密码应通过 `password_env` 指定的环境变量提供，不写进配置文件。
- 在线监控是按 `poll_interval_seconds` 轮询，不是流式推送。

## 测试

```bash
python -m pytest experiment/degradation_perception/tests -q
```
