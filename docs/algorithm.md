# Degradation perception algorithm

## Module layout

```text
experiment/degradation/
├── __init__.py
├── metrics.py
├── config.py
├── prometheus.py
├── window.py
├── baseline.py
├── detector.py
├── association.py
├── storage.py
├── runtime.py
└── cli.py
```

公共数据类型放在拥有其语义的模块中，不单独增加 `types.py`。`metrics.py` 保存 metric catalog；
`config.py` 只构造类型化参数；`storage.py` 负责两个 JSON 的 schema 和文件操作；`runtime.py`
负责编排；`cli.py` 只解析用户意图并调用 runtime。

## v0.1 baseline collection

```text
启动程序
  ↓
轮询 rl_insight_monitor_training_global_step
  ↓
首次得到有效值
  ↓
记录 Prometheus 样本中的 baseline_start_time 和 baseline_start_step
  ↓
继续轮询，直到 current_step >= baseline_start_step + 30
  ↓
使用第31个 step 的时间戳作为30个基线 step 的结束边界
  ↓
调用 /series 发现该时间段内的 series
  ↓
调用 query_range 读取整个基线区间
  ↓
转换为 TimeSeries，交给 window.py 按 global step 对齐
```

例如，基线使用 step 101～130，则需要观察到 step 131，并查询
`[timestamp(101), timestamp(131))`。Step 131 只用作结束边界，不进入基线。

## v0.1 window contract

1. Step 由 `rl_insight_monitor_training_global_step` 的变化时间划分：
   `[timestamp(step k), timestamp(step k+1))`。`timestamp(step k)` 是首次观察到值
   `k` 的时间。
2. 一个 step 内有多个指标样本时，使用最后一个有限值 `last`。恰好位于右边界的样本
   属于下一个 step。
3. 某个 `SeriesId` 在某个 step 内没有值时，在 `StepFrame` 中保留 `None`，不删除整个
   step。
4. Window 始终输出目标数量的 `StepFrame`。KDE 的最少有效样本数由 `baseline.py`
   判断，不属于 window 职责。v0.1 的目标是30个 step，每个 series 至少需20个非空值。
5. 同 metric name、不同 labels 的时序是不同 `SeriesId`，分别保留和对齐。
6. Baseline 采集使用普通 `list[StepFrame]`，不使用 queue。
7. Global step 必须单调、按1递增并覆盖所有必需边界。缺少任意中间 step 或结束边界时直接
   报错，不推断或插值。

`window.py` 不判断指标类型；它只对传入的数值序列统一取 `last`。

## Metric value boundary

`metrics.py` 是 v0.1 默认 metric catalog 的唯一来源：

- `GLOBAL_STEP_METRIC` 是 `rl_insight_monitor_training_global_step`，只作为 step 标尺。
- `TARGET_METRICS` 包含15个可直接输入 KDE 的标量 latency target，统一使用 `Policy.UP`。
- `CANDIDATE_METRICS` 包含95个可直接输入 KDE 的标量 candidate，统一使用 `Policy.BOTH`。
  `CANDIDATE_METRICS_BY_CATEGORY` 仅提供 `latency`、`training_quality`、
  `rollout_quality`、`data_characteristics`、`hardware_resources`、`vllm_engine`
  和 `transfer_queue` 七类静态展示元数据，不参与检测、打分或排序。
- 7个 raw Histogram target 和16个 raw Counter candidate 继续保留在 catalog，但当前不输入 KDE。

runtime 将 `/series` 的实际发现结果与 global step、`TARGET_METRICS`、`CANDIDATE_METRICS` 取交集。
某个 target 或 candidate 在当前训练任务中不存在时自然跳过，不报错；global step 仍必须存在且唯一。
当前路径尚未查询 Prometheus metadata，也尚未把 Counter 自动改写为 `rate()`。

Counter 的既定后续方向仍是由 Prometheus 执行 `rate(counter[window])`，而不是在 Python 中对
raw Counter 手工差分。正式接入前还需确定显式 Counter 清单、metadata 兜底、rate window 和
histogram 排除规则；这些不能只靠 metric 名称猜测。

## v0.1 baseline contract

`baseline.py` 的输入是 `list[StepFrame]`，它不知道 Prometheus、target/candidate 分类或报警方向。

对每个 `SeriesId`：

1. 从30个基线 frame 中保留非 `None` 的有限值，少于20个有效值时不训练，并显式记录跳过原因。
2. 对完整基线序列拟合 Gaussian KDE，寻找密度峰，并用密度谷划分候选正常模式。
3. 将每个密度模式映射回按 step 连续的片段；过大的 step 间隔会切断片段。
4. 将候选片段严格分为前1/3、中1/3和剩余部分，执行六个有方向的均值稳定性比较；至少
   四项通过才接受该片段。
5. 对每个稳定片段再次独立拟合 Gaussian KDE，使用 CDF 的 `alpha` 和 `1 - alpha` 分位数
   得到原始上下界。
6. ratio 按符号向外扩张上下界，保证正数、负数区间都不会被向内压缩。
7. 一个指标可以保留多个独立正常区间。基线结果同时记录原始 KDE 阈值、最终阈值、带宽、
   step 范围和样本数。

v0.1 的默认拟合参数为 `minimum_samples=20`、`alpha=0.025`、
`lower_ratio=upper_ratio=1.05`。这些值集中在类型化参数对象中，并作为函数入参传入；后续
`config.py` 只负责读取和构造参数，不在算法内部重复写死。

## v0.1 detector contract

`detector.py` 接收冻结的 `SeriesBaseline` 和新的 `StepFrame`。它不读取 Prometheus、不训练
KDE，也不执行 association。

### Point-level classification

每个有效值相对全部正常区间只能处于以下一种位置：

```text
NORMAL:        落入任一正常区间
UP:            高于所有正常区间
DOWN:          低于所有正常区间
BETWEEN_MODES: 位于整体上下界内，但不属于任何正常区间
```

阈值边界属于正常区间。`policy` 决定哪些位置算异常：target 使用 `UP`，所以只有 `UP` 为
异常；candidate 使用 `BOTH`，所以 `UP`、`DOWN` 和 `BETWEEN_MODES` 都为异常。`None`
没有方向且不算异常。

### Target event evidence

每个 target 的 `EventTracker` 只长期保存：

```text
recent_points: deque(maxlen=5)  # 包含 None，且 step 严格连续
current_event: None 或已确认事件
```

`recent_points` 保存连续五个 global step，包括值为 `None` 的点。只要窗口中存在 `None`，
当前证据就是未知：既不确认新事件，也不关闭已有事件。只有该 `None` 滑出窗口、重新获得五个
连续有效点后，tracker 才再次判断事件状态。证据未知只暂停生命周期判断；其中已经明确观测到
的 UP 点仍会更新活动事件的最后异常 step、时间和异常点数。

EventTracker 只接受严格按1递增的 step。重复、倒序或出现 step gap 时直接报错，而且不改变
`recent_points`、`current_event` 或已接收的最后一个 step；runtime 负责补齐缺少的数据后重试。

事件使用对称证据判据：

```text
连续5个target step全部有效，且UP abnormal数量 >= 3
→ 有足够证据

连续5个target step全部有效，且UP abnormal数量 < 3
→ 没有足够证据

连续5个target step中存在None
→ 证据未知，不confirm也不close
```

首次获得足够证据时创建 confirmed event；已有事件失去足够证据时关闭。事件的 `end_step`
和 `end_time` 指向最后一个实际 UP 点，`closed_at_step` 和 `closed_at_time` 表示滑窗首次失去
证据的判断位置。v0.1 不使用 `duration`、动态 median gap、`n_keep` 或固定的时间边界扩张。

### Association boundary

EventTracker 只发出 confirmed/closed 生命周期通知。runtime 在 target 首次 confirmed
时调用一次 association，并在同一事件 closed 后再调用一次。candidate 只需要 point-level
异常结果，不要求形成正式事件；`detector.py` 不包含相关系数、随机森林或 Top-K 逻辑。

## v0.1 association contract

`association.py` 是无状态的事件后诊断算法。它接收 runtime 已经截取好的 target
`PointResult` 窗口、按 `SeriesId` 分组的 candidate `PointResult` 窗口和
`event_start_step`；它不查询 Prometheus、不选择分析窗口、不重新分类指标、不维护事件或
context buffer，也不持久化结果。

### Analysis window

runtime 按 global step 选择分析窗口：

```text
confirmed: [event.start_step - pre_context_steps, event.confirmed_at_step]
closed:    [event.start_step - pre_context_steps, event.closed_at_step]
```

runtime 的 `pre_context_steps` 默认30，历史不足时传入实际已有的连续 step，不伪造数据。
少于 `min_rf_samples` 的窗口仍可计算 correlation，但不训练 RF。runtime 负责在事件确认前
保留前置 context，并在确认后持续保存到事件关闭；`pre_context_steps` 不是 association 参数。

### Correlation evidence

所有覆盖率和对齐点数合格、且原始值不是常量的 candidate 都参与 correlation，不要求其越过
baseline。每个 candidate 与 target 使用完全相同的 global step 进行配对，缺失值按 pairwise
方式删除。算法同时计算 Pearson 和 Spearman，选择绝对值较大的结果作为强度，绝对值相同时
优先 Pearson；选中相关系数本身的正负号表达方向，不再保存重复的方向字段。

### Random-forest evidence

只有在 `[event_start_step, 分析窗口最后一步]` 中至少出现一个 point-level abnormal 的 candidate
才进入 RF。每个对齐 step 是一行样本：

```text
y    = target在该step是否abnormal
X[j] = candidate j在该step是否abnormal
```

RF 按 step 时间顺序使用前70%训练、后30%验证，并在验证集上计算 balanced accuracy 和
permutation importance。训练集和验证集都必须同时包含正常与异常 target 标签；任一侧只有一个
类别时，RF 明确返回 `time_split_lacks_both_classes`，不退回全量训练或 impurity importance。
符合资格的 RF candidate 集合一旦确定就保持固定。算法取所有候选的共同有效 step；不足30个
公共样本时返回 `insufficient_common_samples`，不通过动态删除低覆盖 candidate 强行训练。

### Scoring and output

correlation 强度和非负 RF importance 分别在全部有效候选中归一化：

```text
association_score[j]
    = correlation_weight × correlation_share[j]
    + random_forest_weight × random_forest_share[j]
```

默认权重为 `0.85 / 0.15`，必须通过类型化参数传入且两者之和为1。没有参加 RF 的 candidate
其 `random_forest_share` 为0。只有一路证据有效时，有效的一路自动获得权重1；两路都无效时
结果为 `insufficient_data`。

`association_score` 的范围为 `0.0～1.0`，表示当前候选集合中的相对关联证据，不表示因果关系
或性能退化幅度。算法只返回按分数排序的完整 `associations`。runtime 或 UI 使用
`associations[:top_k]` 生成 Top-K，`top_k` 默认25且不属于 association 参数。

公开结果保留 Pearson、Spearman、选中的 correlation、用于计分的非负 RF importance、两路
归一化分量、实际生效权重、样本数和覆盖率。RF 方法固定为 permutation，因此不在每个
candidate 中重复保存；clip 前的负 importance、correlation reason 等只属于内部诊断。
`skipped_candidates` 仅记录 coverage 不足、对齐点不足或常量序列等未进入排名的原因。

v0.1 association 默认参数为：`correlation_weight=0.85`、`random_forest_weight=0.15`、
`min_aligned_points=10`、`min_rf_samples=30`、`min_coverage_ratio=0.6`、
`n_estimators=200`、`class_weight="balanced"`、`random_state=42`、
`train_fraction=0.7`、`permutation_repeats=10` 和 `n_jobs=1`。runtime 后续单独提供
`pre_context_steps=30` 和 `top_k=25`。

## v0.1 storage contract

`JsonStorage` 只拥有 `standard_data.json` 和 `abnormal_data.json` 的路径、schema、JSON
序列化、反序列化和文件删除。它不知道何时训练、检测、confirm 或 close；runtime 决定这些时机
后调用 `load_baseline()`、`save_baseline()`、`save_event()` 或 `reset()`。

Top-K 截取和 candidate 主要方向属于产品语义，仍由 runtime 计算；storage 只把 runtime 传入的
`top_associations` 和 `directions` 写入 JSON。`BaselineSnapshot` 是重启检测所需的冻结模型，
不包含30-step原始训练数据。文件读写、删除和 schema 损坏统一报告为 `StorageError`。v0.1
不增加通用 backend、repository 或数据库抽象。

## v0.1 runtime contract

### Metric roles and state

`RuntimeParameters.target_metrics` 和 `candidate_metrics` 默认分别读取 `metrics.py` 的
`TARGET_METRICS` 与 `CANDIDATE_METRICS`。实际发现且成功建立 baseline 的 target 使用
`Policy.UP`，candidate 使用 `Policy.BOTH`；未暴露的 catalog 项不进入运行时状态。同名但 labels
不同的 series 分别保存和检测。global step 不建立 baseline，也不进入 association ranking。

runtime 只长期维护以下内存状态：

```text
frozen baselines and policies
next_step + next_start_time
one EventTracker per target
recent_results(maxlen=pre_context_steps + evidence_window)
active association context per confirmed target event
```

默认即保留 `30 + 5 = 35` 个最近 frame，确保3-of-5在确认事件时仍能取得事件开始前最多30个
step。这里没有生产者—消费者队列、线程或异步 scheduler。

### Initialization and polling

```text
standard_data.json 不存在
  → 记录首次观察到的 global step N
  → 每300秒轮询，直到至少观察到 N + 30
  → 读取范围数据并构造 N ... N + 29 共30个完整 StepFrame
  → 拟合并保存 baseline
  → cursor = N + 30

standard_data.json 已存在
  → 加载冻结 baseline，不重新拟合 KDE
  → 用启动时观察到的当前 global step 建立新 cursor
```

每轮默认300秒。轮询周期只决定多久查看一次，不决定窗口中有几个 step。若 cursor 为141，而
本轮看到 current step 145，则145区间尚未结束，只处理141～144，再将 cursor 移到145。
Prometheus 查询或 step 对齐抛错时不会推进 cursor。

`query_step_seconds` 默认沿用 RL-Insight Prometheus 的10秒 scrape interval，并可通过 CLI
覆盖。`window.py` 要求看到每一个整数 step 边界；如果训练 step 的可观察变化快于 scrape/query
分辨率，缺失的 step 不会被插值，而是直接报错。

### Event persistence

target confirmed 和同一事件 closed 时各执行一次 association。confirmed 分析创建事件记录；
closed 分析更新同一条记录。candidate 方向定义为 target 实际事件区间内 point-level abnormal
方向的众数；平票为 `MIXED`，没有 candidate abnormal 点为 `NORMAL`。

`standard_data.json` 只保存：

- baseline 的起止 step 和拟合参数；
- global-step `SeriesId`；
- 每个 metric `SeriesId` 的冻结 `NormalRange`、样本数和 KDE 诊断字段。

它不保存原始30-step训练值。`abnormal_data.json` 长期保存事件，以及 confirmed/closed 两阶段
各自的 Top-K、`association_percent`、Pearson、Spearman、correlation share、RF importance
和 RF share。`association_percent = association_score × 100`，仍然只是相对关联证据，不是因果
贡献率。

### Restart and reset boundary

重启会恢复 frozen baseline，但 v0.1 不恢复 detector 的 recent window、活动事件 context 或运行
cursor，也不回放停机期间的数据。启动时位于某个 step 中间时，该 step 只会收集启动后的剩余
部分；已有但尚未 closed 的旧事件记录不会自动续接。这是初版范围限制，不是生产级 checkpoint
语义。

`--reset` 由 CLI 表达用户意图；`runtime.reset()` 清空内存状态并调用 `storage.reset()`，后者只
删除配置指定的 `standard_data.json` 和 `abnormal_data.json`。随后 runtime 重新收集30个 step。
影响 baseline fitting 的显式参数与已有 JSON 不一致时，必须搭配 `--reset`，不会静默忽略新参数。

## v0.1 CLI

当前能力保持在 experiment 内，不修改根 `rl-insight` 命令：

```bash
# 自动训练或加载 baseline，然后每5分钟持续检测
python -m experiment.degradation.cli

# 初始化并只执行一轮检测
python -m experiment.degradation.cli --once

# 清空两个 JSON，并用新 ratio 重新训练
python -m experiment.degradation.cli --reset --baseline-ratio 1.6

# 查看异常中保存的 correlation/RF 分量
python -m experiment.degradation.cli --show-components
```

该命令当前从仓库根目录运行。上游 setuptools package discovery 尚未包含 `experiment`，因此
构建后的 wheel 不会安装这条模块命令；是否把它加入正式 package/根 CLI 留到社区集成范围确定后
再决定。

可重复使用 `--target-metric` 添加 target；`--state-dir` 控制两个固定文件的位置；
`--poll-interval`、`--query-step`、`--baseline-steps`、`--pre-context-steps` 和 `--top-k` 均为显式
入参。v0.1 的 instant global-step 查询仍使用裸 metric name，因此仍依赖“当前只有一个训练任务、
`rl_insight_monitor_training_global_step` 只有一条有效 series”的冻结假设；`--series-selector`
只约束 series 发现和 range query。
