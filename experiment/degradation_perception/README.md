# RL-Insight 劣化感知

`experiment/degradation_perception` 是 RL-Insight 主监控链路的读侧分析能力：
训练任务继续使用 RL-Insight API 上报指标，RL-Insight Server 继续管理
Prometheus、Grafana 和 scrape targets，本模块只负责查询已存储的数据并运行现有
KDE、平稳段、异常区间和关联分析。

核心算法没有改写，也不依赖 Grafana。

## 正式流程

```text
安装并启动 RL-Insight Server
-> Server 管理 Prometheus 和 Grafana
-> 训练任务通过 RL-Insight API 上报指标
-> Prometheus 自动采集并存储
-> degradation CLI 通过 RL-Insight service discovery 找到 Prometheus
-> 根据 CLI/API 参数、任务元数据或默认时长构建 standard/inference 窗口
-> 查询 Prometheus query_range
-> 转换为 standard/inference 算法输入
-> 运行 DegradationPerception
-> 输出完整 DetectionResponse JSON
-> monitor 模式发布劣化派生指标
-> Grafana Dashboard 展示并支持告警
```

普通用户不再创建 workflow YAML，不再填写 `prometheus.base_url`，也不需要手工
查找 Prometheus 地址。

## Endpoint 自动发现

生产模式要求设置主项目已有的 Server 地址：

```bash
export RL_INSIGHT_SERVER_URL="http://rl-insight-server:18080"
```

劣化模块调用主项目公共的 `get_server_services()`，读取
`GET /api/v1/services` 返回的 `prometheus_port`，再用主项目公共的
`service_url_from_server_url()` 将 Server host 与 Prometheus port 组合为查询
endpoint。

这是 Prometheus endpoint discovery，即“分析程序向哪个 Prometheus 查询数据”。
它与 scrape target discovery 不同：后者是 Prometheus 去哪里抓取训练任务或
degradation monitor 的 `/metrics`。

`--prometheus-url` 只用于开发和测试 override，不是生产必填项。模块不会扫描
进程、猜固定端口或读取 Server 私有运行文件。

## CLI

本目录提供可注册到根 CLI 的 `add_degradation_parser()`，命令语义为：

```bash
rl-insight degradation analyze
rl-insight degradation monitor
```

当前根 `rl_insight/cli.py` 没有插件或 entry-point 扩展机制。由于本次修改严格
限制在 `experiment/degradation_perception`，根解析器尚未调用该注册函数。
仓库现有 console script 已可直接使用相同能力：

```bash
rl-insight-degradation analyze
rl-insight-degradation monitor
```

要真正启用上面的根命令，主项目只需在根 parser 创建后调用一次
`experiment.degradation_perception.cli.add_degradation_parser(subparsers)`。
这是唯一缺少的 CLI 接线，不涉及算法或服务生命周期。

### 单次分析

以下命令自动发现 Prometheus，使用最近 30 分钟作为 inference，并使用其前
30 分钟作为 standard：

```bash
rl-insight-degradation analyze \
  --project verl \
  --experiment-name ppo-run-042 \
  --metric timing_s/step=rl_insight_monitor_timing_s_step \
  --worker trainer_0 \
  --association-target timing_s/step \
  --output-dir .run-output/degradation
```

stdout 是完整、严格可 JSON 序列化的 `DetectionResponse`。同时写入：

```text
.run-output/degradation/detection_response.json
.run-output/degradation/query_diagnostics.json
```

`DetectionResponse` 保留 `states`、`results`、`abnormalTimeRange`、
`metricErrors` 和可选的 `associationAnalysis`。完整异常区间和复杂诊断只放在
JSON 中，不编码为 Prometheus labels。

### 持续监控

```bash
rl-insight-degradation monitor \
  --project verl \
  --experiment-name ppo-run-042 \
  --metric timing_s/step=rl_insight_monitor_timing_s_step \
  --worker trainer_0 \
  --association-target timing_s/step \
  --interval-seconds 60 \
  --metrics-port 9093
```

monitor 每轮都输出一行完整 JSON，并通过主项目已有的
`start_metrics_http_server()` 暴露派生指标，再通过
`update_prometheus_config()` 注册为 `rl_insight_degradation` scrape job。
进程必须保持运行，Prometheus 才能持续抓取这些指标。

如果自动检测到的本机地址不是 Prometheus 可达地址，可显式传
`--metrics-advertise-host`。该参数指定的是派生指标 scrape target 的主机，不是
Prometheus 查询 endpoint。

## 指标与 PromQL

`--metric` 可重复。格式是：

```text
逻辑指标名=Prometheus指标名
```

逻辑指标名用于算法配置和 JSON；Prometheus 指标名用于查询。两者相同时可只写
一个值。

模块自动将以下统一 labels 放入 standard 和 inference 查询：

- `project`
- `experiment_name`
- 可选 `worker`
- 可选 `replica`
- `--label KEY=VALUE` 指定的其他主项目 labels

如果健康基线来自另一个实验，使用：

```bash
--standard-experiment-name ppo-run-healthy
```

Counter、Histogram 或需要聚合的指标通过 `--query` 提供 PromQL 模板。模板中的
`{{metric}}` 和 `{{labels}}` 会分别替换为 Prometheus 指标名和当前 phase 的
label selector：

```bash
--metric latency=request_latency_seconds \
--query 'latency=rate({{metric}}_sum{{labels}}[1m]) / rate({{metric}}_count{{labels}}[1m])'
```

每条最终查询必须只返回一个标量时间序列。算法不会隐式选择 `result[0]`。

## 窗口解析

窗口优先级固定为：

```text
显式 CLI/API 参数
> task_metadata.start_time 与当前检测时间
> degradation 默认窗口时长
```

显式参数支持 epoch seconds 或 ISO-8601：

```bash
--standard-start 2026-07-30T02:00:00Z \
--standard-end 2026-07-30T02:30:00Z \
--inference-start 2026-07-30T04:00:00Z \
--inference-end 2026-07-30T04:45:00Z
```

CLI 可用 `--task-start` 提供已有任务起始元数据。程序调用
`PrometheusDegradationRunner` 时可直接传
`task_metadata={"start_time": ...}`。当前主项目没有公共任务元数据查询 API，
所以模块不会读取 Server 私有文件或伪造任务元数据。

默认值：

```text
baseline_window_seconds = 1800
inference_window_seconds = 1800
query_step_seconds = 10
```

## 算法配置

删除的是重复的数据获取 workflow YAML，不是算法配置。以下参数继续由
`default_config.yaml`、每指标配置和 `--config-dir` 管理：

- `alpha`
- `upper_ratio` / `lower_ratio`
- KDE bandwidth、grid 和 tail 参数
- 平稳段参数
- minimum standard/inference points
- 异常区间持续时间、异常点数和异常率规则
- 关联分析权重、Top-K 和随机森林参数

这些参数不会进入 Prometheus service config 或 scrape config。

未传 `--config-dir` 时，每指标算法配置保存在
`~/.rl-insight/degradation/config`，避免向安装包目录写运行时配置。

## Prometheus 派生指标

monitor 发布以下 gauges，统一前缀为 `rl_insight_degradation_`：

| 指标 | 含义 |
| --- | --- |
| `abnormal` | 是否存在已确认劣化区间 |
| `detection_state` | `0` 完成，`1/2` 数据不足，`-1` 指标执行错误 |
| `upper_threshold` | 当前 KDE 模型上阈值包络 |
| `lower_threshold` | 当前 KDE 模型下阈值包络 |
| `abnormal_rate` | 最新 inference 窗口异常率 |
| `abnormal_point_count` | 最新 inference 窗口异常点数 |
| `last_detected_timestamp_seconds` | 最近完成检测的 Unix 时间 |
| `interval_active` | 最新一轮是否存在活动异常区间 |
| `run_success` | 本轮是否无 metric error |
| `association_score_percent` | 现有关联分析的 Top-K 分数 |

固定 labels 只有 `project`、`experiment_name`、`worker`、`replica` 和受控的
metric/Top-K 维度。异常起止时间、完整区间、查询文本、诊断详情不会进入 labels。

## Grafana

Dashboard 文件位于：

```text
experiment/degradation_perception/grafana/degradation_dashboard.json
```

它展示：

- 原始性能指标；
- 上下检测阈值；
- 当前是否劣化；
- 检测执行状态；
- 异常率和异常点数量；
- 异常活动时间段；
- 最近检测时间；
- 关联分析 Top-K。

Dashboard 只查询 Prometheus。KDE、平稳段、异常区间和关联计算全部在
`DegradationPerception` 中完成。

当前 Server 只自动复制 `rl_insight/config/services/grafana/dashboards`，也没有
公共 Dashboard 注册 API。受“只修改 experiment”约束，该 Dashboard 资源尚不能
由 Server 自动 provision。要完成正式接线，需要主项目将此目录加入 dashboard
staging，或增加一个公共 Dashboard 注册接口；本模块不会写 Server 私有运行目录。

## 开发兼容入口

旧入口仅保留用于已有测试和开发排障：

```bash
python -m experiment.degradation_perception.prometheus_workflow
```

它不再是推荐生产入口，也不再提供 workflow YAML 模板。新的生产参数边界是
CLI/API 业务参数加独立算法配置。

离线 JSON 与模拟 Prometheus 测试入口继续可用，不会发起真实 HTTP 请求。

## 测试

```bash
python -m pytest experiment/degradation_perception/tests -q
```

服务发现、Prometheus HTTP、target 注册和指标 endpoint 测试均使用 mock；不会
连接或修改真实 Server、Prometheus 或 Grafana。
