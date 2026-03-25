# UI-Distiller

基于大模型的 Browser Agent 系统，实现从用户自然语言指令到 Web 任务的自动执行。

## 项目初衷

本项目并不是为了“造一个完美无缺的 Browser Agent”，而是为了验证一个科研猜想：
在 Browser Agent 这个方向上，是否可以通过多Agent协作与输入压缩策略，在效率等指标上达到甚至超越当前开源 SOTA 方案 [Browser Use](https://github.com/browser-use/browser-use)。

目前的结论是：这条路线是可行的，部分实验指标已经体现出潜力。
但是 Browser Use 在模型训练、工程化成熟度上都仍有显著优势。 

## 项目特点

- **自然语言驱动自动化**：输入任务描述，Agent 自动探索网页，完成任务
- **多 Agent 协作调度**：设计了基于任务调度的多 Agent 协作机制，基于不同规模的模型并行/串行协作，在执行效率与决策准确率之间取得平衡
- **分层压缩机制**：针对 Web GUI 输入 Token 冗余问题，构建分层压缩与裁剪流程，Token 消耗降低 1 个数量级
- **可观测与可追溯**：自动保存日志、过程截图、结构化结果（JSON）与执行报告（Markdown + Gantt 图）

## Benchmark 评估

在 [Online-Mind2Web2](https://huggingface.co/datasets/osunlp/Online-Mind2Web) 基准测试中，本项目取得了以下结果：

| 难度     |         成功率 |  评估进度 |
| ------ | -------------------: | -------------------: |
| Easy   |   62/78 = 79.49% |   78/80 = 97.50% |
| Medium | 105/142 = 73.94% | 142/143 = 99.30% |
| Hard   |   53/77 = 68.83% |  77/77 = 100.00% |
| Total  | 220/297 = 74.07% | 297/300 = 99.00% |

> 说明：评测中少量未完成样本主要由环境因素导致，包括网页失效以及 Playwright 运行异常，并非任务内容本身不可完成。

### Token开销与时延分析

完成每个任务（包含成功、失败）的 Token 消耗如下：

| 难度     | 主模型 Token（输入 / 输出） | 次模型 Token（输入 / 输出）  |
| ------ | ------------------ | ------------------- |
| Easy   | 32806.75 / 870.71  | 53607.44 / 1599.94  |
| Medium | 61208.80 / 1624.87 | 99963.57 / 2883.44  |
| Hard   | 82921.47 / 2123.79 | 136125.65 / 4176.77 |
| 总体     | 59296.13 / 1554.09 | 97028.27 / 2877.38  |

每轮交互平均时延为 **11.17s**，其中 LLM 响应耗时约 **7.78s**，其余时间主要消耗在操作执行后的页面变化与网络加载等过程。

## 系统概览

6 个职责不同的 Agent，分别承担以下任务：

1. **Planning**：理解当前状态并生成下一步目标
2. **Act**：执行浏览器动作（点击、输入、滚动、导航等）
3. **Observation**：基于执行前后页面变化进行结果判断
4. **Extraction**：从网页中提取任务相关关键信息
5. **Feedback**：反馈任务完成度
6. **Refinement**：压缩任务历史上下文

核心为：`Planning -> Act -> Observation` 的 ReAct 循环

## 核心挑战与应对方案

### 1) GUI DOM Tree 输入冗余，充满噪音

在真实网页里，DOM Tree 通常非常庞大，且包含大量与当前任务无关的元素。
如果把这些信息一次性全部喂给模型，常见后果有两个：

- **TTFT（首 Token 延迟）变高**：输入更长，模型起步更慢
- **推理效果受干扰**：噪声信息过多，模型更容易偏离当前目标

#### 应对方案

1. **基于规则的 GUI 过滤与压缩**
   - 基于可见性、可交互性等规则过滤 GUI 元素
   - 在尽量不丢失关键信息的前提下压缩 GUI 表示

2. **基于 2B 小模型的任务相关性过滤**
   - 在每次交互前，用 2B 小模型过滤与当前任务无关的 GUI 元素
   - 额外耗时仅约 **0.6s**
   - 平均每次交互可为大参数模型节省约 **4K 输入 Token**，并显著降低噪声干扰

过滤效果示意：

<table align="center">
  <tr>
    <td><img src="screenshot/dom-before.png" width="100%" /></td>
    <td><img src="screenshot/dom-after.png" width="100%" /></td>
  </tr>
</table>

### 2) 单次响应承载多步推理：复杂度高、响应慢、稳定性差

Browser User 一次交互里，要同时完成多件事：

- 评估上一步操作是否有效
- 思考当前任务状态
- 规划下一步
- 生成操作指令

把这些目标压在同一次响应中，过于复杂，可能导致模型推理错误。

#### 应对方案

1. **按复杂度拆分为更轻量的子任务**
   - 大参数模型（A17B）处理关键逻辑推理子任务（如 Planning）
   - 小参数模型（2B）处理总结归纳型子任务（如 Observation）

2. **每个子任务仅保留必要上下文**
   - 虽然子任务之间存在一定上下文重叠
   - 但大参数模型的输入/输出 Token 明显下降，同时保证有效性

3. **对子任务进行并行化调度**
   - 可并行任务并行执行
   - 在某些场景下，子任务仅输出关键字段后即可触发下游调度
   - 进一步缩短交互时延

并行效果可在甘特图中直观看到：

![并行调度甘特图](screenshot/gantt.png)

## 快速开始

### 1) 安装依赖

使用 `uv` 安装 python 依赖：

```bash
uv sync
```

激活虚拟环境：

```bash
# linux/macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

### 2) 安装 Playwright 浏览器

```bash
playwright install
```

### 3) 配置模型与运行参数

复制并编辑配置文件：

```bash
cp env.example.json env.json
```

在 `env.json` 中填写你的模型服务配置：

- `vlm_primary_service` / `vlm_secondary_service`：视觉模态模型（VLM）配置名
- `llm_primary_service` / `llm_secondary_service`：文本模态模型（LLM）配置名
- 其中，`primary` 表示承担关键逻辑推理与决策的大参数模型，`secondary` 表示承担总结归纳与辅助处理的小参数模型
- 对应服务的 `api_key`、`base_url`、`model`、`temperature`

## 运行方式

### 方式 A：命令行运行（main.py）

```bash
python main.py \
  --out-dir output/test \
  --task "在B站找到一个关于如何在笔记本电脑部署Qwen大模型的教程视频" \
  --start-url "https://www.bilibili.com/"
```

### 方式 B：示例脚本（demo.py）

`demo.py` 提供了一个可直接运行的示例，方便自行修改与调试。

## 输出说明

每次任务执行会在 `--out-dir` 指定的目录下生成结果文件，包括：

- `log.log`：运行日志
- `result.json`：结构化执行结果
- `report.md`：可读执行报告
- `gantt.png`：时序甘特图
- 各阶段截图：动作前后、观察、规划等

## 项目结构

```text
.
├── agent/
│   ├── agent.py        # 主执行循环：planning/act/observation/extraction/feedback
│   ├── action.py       # 浏览器动作定义与执行
│   ├── dom.py          # DOM 解析、聚类与压缩相关逻辑
│   ├── llm.py          # 多模型调用封装与 token 统计
│   ├── config.py       # 配置初始化
│   └── record.py       # 执行记录结构
├── main.py             # CLI 入口
├── demo.py             # 可运行示例
├── env.example.json    # 配置模板
└── pyproject.toml      # 依赖配置
```
