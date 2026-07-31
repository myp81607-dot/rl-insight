# RL-Insight 劣化感知算法模块

`experiment/degradation_perception` 是可被 RL-Insight 主项目直接调用的算法模块。
它接收已经标准化的时序数据和内存策略，完成劣化感知与关联分析，返回结构化结果。

## 模块边界

本模块负责：

- 从健康历史数据构建 KDE 基线；
- 识别单模态或多模态稳定片段；
- 判断推理窗口中的异常点；
- 合并并确认异常区间；
- 对目标指标的异常事件执行关联分析；
- 提供稳定的 `TimeSeries`、`DetectionInput` 和 `DetectionResult` 接口。

本模块不采集指标，不管理外部服务，也不决定结果如何存储或展示。
调用方负责准备数据、解析业务配置并处理检测结果。

## 输入

每条序列使用 `TimeSeries`：

```python
TimeSeries(
    timestamps=[1.0, 2.0, 3.0],
    values=[10.0, 10.1, 9.9],
)
```

`DetectionInput` 明确区分两个阶段：

- `standard`：健康历史窗口，用于构建基线；
- `inference`：待分析窗口，用于异常判断。

同一指标的时间戳和值必须一一对应，详细规则见[输入输出契约](docs/INPUT_OUTPUT.md)。

算法策略由调用方以内存映射传入：

- `metric_policies`：每个指标的 KDE、稳定片段、异常方向和区间规则；
- `history_policy`：跨检测批次的确认规则；
- `association_targets`：需要执行关联分析的目标指标。

模块不会自行查找或读取运行时配置文件。

## 最小调用

```python
from experiment.degradation_perception import (
    DegradationPerception,
    DetectionInput,
    TimeSeries,
)

data: DetectionInput = {
    'standard': {
        'timing_s/step': TimeSeries(
            timestamps=healthy_timestamps,
            values=healthy_values,
        ),
    },
    'inference': {
        'timing_s/step': TimeSeries(
            timestamps=current_timestamps,
            values=current_values,
        ),
    },
}

detector = DegradationPerception(
    dataset=data,
    metrics=['timing_s/step'],
    metric_policies={'timing_s/step': metric_policy},
    history_policy=history_policy,
    association_targets=['timing_s/step'],
    task_id='run-001',
    source_type='prometheus',
)
result = detector.detect()
```

`metric_policy` 和 `history_policy` 由 RL-Insight 主项目创建并注入。
重复分析已获取的数据时，也可以调用 `detect_dataset(data)`。

## 核心流程

```text
DetectionInput
  -> 对齐、过滤、排序和去重
  -> 稳定片段与多模态识别
  -> 每个稳定模态建立独立 KDE 阈值
  -> 异常点判断
  -> 异常区间合并与确认
  -> 可选关联分析
  -> DetectionResult
```

算法细节见 [核心算法](docs/ALGORITHM.md) 和
[关联分析](docs/ASSOCIATION_ANALYSIS.md)。

## 输出

`DetectionResult` 是可序列化的结构化结果，主要包含：

- `states`：每个指标的检测状态；
- `results`：阈值、逐点诊断和指标级异常区间；
- `abnormalTimeRange`：已确认异常区间；
- `metricErrors`：可选的指标级错误，不影响其他指标；
- `associationAnalysis`：可选的关联分析排名和诊断。

关联分数表达统计关联强度，不代表因果关系。
随机森林证据需要 `scikit-learn`；未安装时会明确返回
`dependency_unavailable`，相关性分析仍可继续。

## 测试

核心测试覆盖 KDE、稳定片段、多模态、异常区间、时间对齐和关联分析：

```bash
python -m pytest experiment/degradation_perception/tests -q
```

测试只验证算法和内存接口，不要求任何外部服务。
