# RL-Insight 劣化感知：VeRL 日志运行指南

本目录保留现有 KDE、平稳段、多模态、异常点和异常区间算法，并增加两种真实
VeRL step-metrics 日志入口：

- 离线检测：本地 `healthy.log` + 本地 inference 日志，执行一次后保存 JSON；
- 持续检测：本地固定 `healthy.log` + SSH 远程运行日志，按间隔轮询新增 step。

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

## 离线检测

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

执行检测：

```bash
python -m experiment.degradation_perception.main \
  --standard-log healthy.log \
  --inference-log inference.log \
  --metrics timing_s/step actor/entropy \
  --source-type training_log \
  --task-id degradation-demo \
  --debug-kde \
  --output result.json
```

`healthy.log` 只构造 `standard`，`inference.log` 只构造 `inference`；两份日志的
step 可以分别从 0 开始。`--debug-kde` 只控制终端详细展示，不改变算法计算。
终端阈值显示两位小数，JSON 保存完整浮点精度。

## SSH 轮询检测

复制脱敏配置并填写测试环境信息，不要提交真实地址、用户名、密码、私钥内容或
内部日志路径：

```bash
cp experiment/degradation_perception/monitor_config.example.yaml monitor_config.yaml
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
启动时读取一次本地 healthy.log，固定为 standard
→ 通过 Paramiko SFTP 只读获取远程 .log 快照
→ 丢弃尚未换行的活动日志尾行
→ 只接受 step > last_step 的选中指标
→ 累积到 inference 缓冲区
→ 每个指标都达到算法最小点数后执行检测
→ 原子保存本轮 JSON
→ 更新 last_step，并清空本轮已消费缓冲区
→ 等待下一轮
```

数据不足时会保留在缓冲区，且不会生成伪异常结果。无新增 step 时不会重复检测。
SSH、检测或写盘失败会记录错误并等待下一轮；失败轮次不会提交候选数据。

## 算法与输出

```text
读取正常日志
→ 解析 step metrics
→ 找到平稳片段
→ KDE 学习正常分布
→ 计算每个指标的标准区间
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

- `healthy.log` 必须是已确认正常的数据，运行期间不会自动更新；
- inference 数据不会进入正常基线，且不保证一定出现异常；
- 两类日志应尽量来自相同模型、代码版本和训练配置；
- 日志中的时间表示 step，SSH 数据仍以 `source_type=training_log` 调用算法；
- 指标名始终保留原始形式；
- 内部 KDE、分位数和阈值判断不做提前四舍五入；
- SSH 默认拒绝未知 host key，可使用系统 known_hosts 或配置 `known_hosts`；
- 密码应通过 `password_env` 指定的环境变量提供，不写进配置文件。

## 测试

```bash
python -m pytest experiment/degradation_perception/tests -q
```
