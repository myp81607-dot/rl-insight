# RL-Insight 劣化感知运行指南

本目录保留现有 KDE、平稳段、多模态、异常点、异常区间和关联分析算法，并提供
三种数据入口：

- `fit` / `detect`：读取本地 VeRL step-metrics 日志，完成离线拟合和检测；
- `remote_monitor`：通过 SSH 定时轮询远程 inference 日志；
- `prometheus_monitor`：通过 Prometheus HTTP API 自动拟合基线并持续检测。

两种在线模式都是轮询，不是流式推送；实际延迟由各自的轮询间隔决定。离线检测
仍然保留，适合回放历史日志和验证算法参数。

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

## 4. Prometheus 在线检测：最短完整流程

Prometheus 模式只通过 HTTP API 读取已有 Prometheus 数据，算法本身不开放新端口。
下面的命令都从仓库根目录执行。

### 4.1 启动最小监控服务

算法只依赖 Prometheus；保留 RL-Insight server 用于训练进程发现服务和自动注册
`/metrics` 抓取地址，Tempo 和 Grafana 可以关闭：

```bash
cat > rl_insight/config/prometheus-only.yaml <<'YAML'
server:
  enable: true
prometheus:
  enable: true
tempo:
  enable: false
grafana:
  enable: false
YAML

rl-insight server install --config rl_insight/config/prometheus-only.yaml
rl-insight server start --config rl_insight/config/prometheus-only.yaml --detach

curl -sS http://127.0.0.1:18080/healthz
curl -sS http://127.0.0.1:9090/-/ready
```

### 4.2 名称从训练代码产生

`project`、`experiment_name` 和原始指标名都由训练代码决定，不是算法猜出来的：

```python
import rl_insight as insight

insight.init(project="degradation-demo", experiment_name="smoke")
insight.metric_gauge("training_global_step", step)
insight.metric_gauge("timing_s_step", timing)
insight.metric_gauge("actor_entropy", entropy)
```

启动训练前，在同一个容器或可访问 RL-Insight server 的训练环境中设置：

```bash
export RL_INSIGHT_SERVER_URL=http://127.0.0.1:18080
python <training-entry.py> <training-arguments>
```

默认指标 namespace 是 `rl_insight_monitor`，所以上面的原始指标会暴露为：

```text
rl_insight_monitor_training_global_step
rl_insight_monitor_timing_s_step
rl_insight_monitor_actor_entropy
```

如果真实训练已经集成 RL-Insight，可以从代码和 Prometheus 两边确认名称：

```bash
rg -n 'insight\.init|metric_gauge|metric_count|metric_histogram' <training-repository>

curl -sS http://127.0.0.1:9090/api/v1/targets | python -m json.tool
curl -sS http://127.0.0.1:9090/api/v1/label/__name__/values | python -m json.tool
curl -sS http://127.0.0.1:9090/api/v1/label/project/values | python -m json.tool
curl -sS http://127.0.0.1:9090/api/v1/label/experiment_name/values | python -m json.tool
```

### 4.3 数据调用链和实际读取代码

```text
训练代码调用 insight.metric_gauge(...)
→ RL-Insight Ray MonitorHub 在 :9092 暴露 /metrics
→ RL-Insight server 将 :9092 注册为 Prometheus target
→ Prometheus 按 scrape_interval 抓取并保存时序
→ prometheus_monitor.py 读取 YAML 中的 PromQL
→ PrometheusHttpClient 请求 /api/v1/query_range
→ 响应被解析成 TimeSeries，进入基线拟合、检测和关联分析
```

关键代码位置：

- `rl_insight/api.py::_emit`：把训练指标名、值和
  `project` / `experiment_name` 标签发送给 MonitorHub；
- `rl_insight/collector/ray_monitor_hub.py::MonitorHubActor`：开放 `:9092/metrics`
  并向 RL-Insight server 注册抓取地址；
- `prometheus_monitor.py::_query_window`：第 926 行读取 global step，第 932 行读取
  YAML 中配置的全部 target/candidate 指标；
- `prometheus_source.py::PrometheusHttpClient.query_range`：第 213 行的
  `self.session.get(...)` 真正请求 Prometheus `/api/v1/query_range`；
- `prometheus_source.py::_parse_matrix_result`：把 Prometheus matrix JSON 转成算法的
  `TimeSeries`。

直接查看负责读取的源码：

```bash
nl -ba experiment/degradation_perception/prometheus_monitor.py | sed -n '920,938p'
nl -ba experiment/degradation_perception/prometheus_source.py | sed -n '197,252p'
```

### 4.4 配置 PromQL 和指标角色

```bash
cp experiment/degradation_perception/prometheus_monitor_config.example.yaml \
  /workspace/prometheus_monitor_config.yaml
cp experiment/degradation_perception/algorithm_config.yaml \
  /workspace/algorithm_config.yaml
```

配置左侧是算法内部名称，右侧 `promql` 是 Prometheus 中的真实指标和训练标签：

```yaml
metrics:
  timing_s/step:
    promql: 'rl_insight_monitor_timing_s_step{project="degradation-demo", experiment_name="smoke"}'
    role: auto
    abnormal_type: UP
  actor/entropy:
    promql: 'rl_insight_monitor_actor_entropy{project="degradation-demo", experiment_name="smoke"}'
    role: auto
    abnormal_type: BOTH
```

名称包含 `time` 或 `timing` 的算法内部名称默认作为 target，其余指标默认作为
association candidate；也可以用 `role` 或 `target_selection.overrides` 显式覆盖。
每条 PromQL 必须通过 `project`、`experiment_name`、`job` 或 `instance` 等标签唯一
定位一条时序；需要合并副本时，在 PromQL 中显式使用 `sum`、`avg` 等聚合。

### 4.5 自动训练基线并启动轮询

训练超过 step 25 后，先执行一轮。没有基线时会自动查询并使用 step 5 至 25 的
已观测、已对齐样本拟合基线；生成后基线冻结，step > 25 才进入检测：

```bash
python -m experiment.degradation_perception.prometheus_monitor \
  --config /workspace/prometheus_monitor_config.yaml \
  --once
```

启动后台轮询：

```bash
nohup .venv/bin/python -u \
  -m experiment.degradation_perception.prometheus_monitor \
  --config /workspace/prometheus_monitor_config.yaml \
  > /workspace/prometheus_monitor.log 2>&1 &

echo $! > /workspace/prometheus_monitor.pid
tail -f /workspace/prometheus_monitor.log
```

`poll_interval_seconds` 默认 180 秒，表示多久查询一次；
`prometheus.query_step_seconds` 默认 15 秒，表示 `query_range` 的采样网格，不能把它
误设成三分钟。每轮会重叠查询一小段历史并按时间戳去重。

### 4.6 查看故障和关联结果

```bash
ls -lht /workspace/prometheus_output
python -m json.tool \
  /workspace/prometheus_output/degradation-prometheus-demo_latest.json
```

- `<task_id>_latest.json`：最近一次完整检测结果，每次完成检测时原子覆盖；
- `event_*.json`：只有 target 在 `NORMAL`、`ABNORMAL`、`RECOVERED` 之间变化时生成；
- `monitor_state.json`：轮询游标、推理缓冲和活动事件状态，不是对外故障报告；
- `states[metric] == 0`：只表示该指标检测流程成功完成，不表示一定正常；是否异常应
  查看 `events`、`abnormalTimeRange` 和 `results[metric].message`。

Prometheus 在线监控会把自动选中的 target 传给关联分析，不需要再单独运行一个
关联命令。要得到 Top-5，必须先把候选指标全部加入 `metrics`，并确保它们从基线
step 5 至 25 就有数据。当前关联模式只排名 target 故障窗口内自身也异常的候选；
正常、常量、点数不足或时间覆盖不足的候选会记录在 `excludedMetrics` 中，而不会
强行凑满五个：

```yaml
metrics:
  gpu/utilization:
    promql: 'rl_insight_monitor_gpu_utilization{project="degradation-demo", experiment_name="smoke"}'
    role: auto
    abnormal_type: BOTH
```

关联权重和数量在 `algorithm_config.yaml` 中调整：

```yaml
association:
  enabled: true
  target_metrics:
    - timing_s/step
  weights:
    correlation: 0.5
    random_forest: 0.5
  top_k: 5
  min_aligned_points: 10
  min_rf_samples: 30
  min_coverage_ratio: 0.6
```

结果位于：

```text
associationAnalysis.targets.<target>.events[].topAssociations
```

### 4.7 参数变更规则

- 只修改 `poll_interval_seconds`、输出目录或 association 排名参数：重启监控即可；
- 修改指标/PromQL、target 规则或基线 step 范围：换新的 `baseline_model`、
  `state_file` 和 `task_id`，重新训练基线；
- 修改 `alpha`、阈值 ratio、KDE 或平稳段参数：换新基线并重新拟合；
- 认证信息只能通过配置指定的环境变量读取，不要把 token 或密码写入 YAML。

## 算法与输出

```text
fit：读取正常日志
→ 解析 step metrics
→ 找到平稳片段
→ KDE 学习正常分布
→ 计算每个指标的标准区间
→ 保存 baseline_model.json
detect / monitor：加载 baseline_model.json
→ 离线/SSH 模式读取日志，Prometheus 模式查询时序
→ 标记异常点
→ 合并异常区间
→ 输出 JSON
```

核心返回字段保持不变：

- `states`：指标检测流程状态；`states[metric] == 0` 只表示流程完成，既不等于正常
  也不等于异常；
- `abnormalTimeRange`：经现有规则确认的正式异常区间；
- `results`：当前核心实现的检测详情、逐点诊断和阈值；
- `metricErrors`：可选的指标级错误；
- `associationAnalysis`：在策略开启关联分析，或调用方显式传入 association target
  时出现；Prometheus 在线监控会自动传入选中的 target。

适配层增加 `debugDetails.metrics`。每个成功建模的指标分别保存：

- `alpha`、实际用于上下分位阈值的 `lowerProbability` 和 `upperProbability`；
- standard/inference 有效点数；
- 异常点数和异常区间数；
- 每个正常模型各自的原始 KDE 分位区间、最终判定区间、ratio 和 bandwidth；
- 平稳 segment 的 step 范围和峰影响区间。

多正常模式保存在 `normalModels` 的多个独立元素中，不取平均、不合并阈值。
在线 JSON 还包含 `metadata`。SSH 模式记录新增 step、检测前后游标和 inference
缓冲范围；Prometheus 模式记录查询时间范围、查询网格、轮询周期、
`lastGlobalStep`、选中指标和各指标点数。

## 注意事项

- `healthy.log` 必须是已确认正常的数据，只在 `fit` 阶段读取；
- inference 数据不会进入正常基线，且不保证一定出现异常；
- 在线和离线检测使用相同的基线模型格式，但不同数据源、指标或查询配置不应复用
  同一个基线文件；
- 更换设备、模型、卡数、代码版本或训练配置后，通常需要重新 `fit`；
- 日志中的时间表示 step，SSH 数据仍以 `source_type=training_log` 调用算法；
- 日志模式保留日志中的原始指标名；Prometheus 模式由 YAML 左侧定义算法内部名称，
  并由右侧 PromQL 映射真实 Prometheus 指标；
- 内部 KDE、分位数和阈值判断不做提前四舍五入；
- SSH 默认拒绝未知 host key，可使用系统 known_hosts 或配置 `known_hosts`；
- 密码应通过 `password_env` 指定的环境变量提供，不写进配置文件；
- 在线监控是按 `poll_interval_seconds` 轮询，不是流式推送。

## 测试

```bash
python -m pytest experiment/degradation_perception/tests -q
```
