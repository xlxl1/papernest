# 多 Agent 论文写作改造计划

## 验收目标

- 每个角色拥有独立输入、输出和失败状态。
- Agent 通过结构化消息传递证据、主张和评审意见。
- 证据不足、引用错配和全文冲突会触发动态重规划。
- 每条事实性主张都能回溯到已核验的页码或块。
- 评测能比较单次生成、串行流水线和多 Agent 协作的收益与成本。

## 阶段

1. **P0 数据契约**：`Evidence`、`Claim`、`EvidenceBundle`、`AgentMessage`、`ConsistencyIssue`；补充 schema 校验和序列化测试。
2. **P1 证据 Agent**：按章节目标检索正文、表格、公式和结论，生成可核验 evidence bundle。
3. **P2 主张驱动写作**：先生成 claims，再绑定证据，写作只消费已绑定主张。
4. **P3 写作协调器**：管理 Agent 状态、消息、重试、并行证据任务和断点恢复。
5. **P4 审计与一致性**：主张—引用匹配、数字核验、术语一致性、冲突报告和局部修复。
6. **P5 评测**：建立固定主题集，比较单 Agent、串行流水线、多 Agent 的证据覆盖率、引用准确率、数字保真度、重写次数、延迟和成本。

## 当前进度

- P0：完成。数据模型已接入选题证据与大纲要点，并有序列化/形状漂移回归测试。
- P1：完成。新增章节级 `EvidenceAgent`：检索章节块、页文本、表格、公式和图注；逐条做原文回取核验；输出 `EvidenceBundle`，并持久化到 `sections.evidence_json`。写作 Agent 会在生成前消费该 bundle。
- P2：完成。写作提示词只接收「已绑定到 Claim 且逐字核验通过」的证据；引用白名单直接由这些证据生成，未绑定或未核验的检索结果不会进入写作上下文。`evidence.py:_contains_in_source` 在收紧 provenance 后返回 `(verified, source_kind)`，白名单只接受 `native_page` / `imported_fulltext` 的 verified 证据（修复后 `pipeline.py:_bound_verified_evidence`）。
- P3：基础完成。持久化 `Coordinator` 已记录每个 Agent 的状态/尝试次数、结构化 hand-off 消息和下一步计划；章节证据并行构建、单节失败隔离；结合现有 jobs 的重试、取消和服务重启恢复能力，写作状态可回放。`stage_polish` 在审计仍有 major 时按 `Coordinator.plan()` 触发有界 `max_coordinator_rounds` 自动返工（默认 1 轮），并把每一轮重写与新 major 写入 `report.coordinator_rounds`；尚缺与 jobs 取消/重试耗尽的联动、跨进程恢复后的 lease 校验。
- P4：基础完成。离线机械审计 Agent 已就位：按 Claim 绑定关系检查引用缺口与主张错配，按引用对应论文的已核验证据逐句核验数字，按术语表 canonical/aliases 检查术语变体；输出结构化 `ConsistencyIssue`，并接入终稿终检、润色导出和全文报告。审计结果可区分引用缺口、主张冲突、数字冲突和术语冲突，提供位置、严重级别与局部修复建议。`stage_polish` 现在把未消解的 major 转为 `needs_review`，并拒绝以 `final` 模式导出。审计会把 `source_kind != native_page` 的引用标记为 minor `provenance_weak`，让协调器在返工时优先替换它们。
- P5：基础完成。固定 `writing_eval_set.json`（4 个主题、8 个章节、16 条金标准主张、8 个稳定来源）和隔离评测运行器 `papernest/writing_benchmark.py`；每个主题/模式使用临时 SQLite，比较 `single`、`serial`、`multi` 三种写作条件，逐项记录证据覆盖率、引用精确度、数字保真度、无支撑率、不足证据保留率、重写次数、端到端延迟、token、成本和失败状态。`single` 模式现在与 `serial`/`multi` 接收相同的主题、章节要点与来源摘要（gold claim 仍只给独立 evaluator），`_claim_metrics` 区分 `citation_precision`（基于 verified+claim 绑定证据）与 `citation_string_match`（字面对照），并报告 `usage_known` / `verified_source_count`。新增 `--max-cost-usd` / `--max-llm-calls` / `--max-tokens` / `--max-wallclock-s` 预算护栏，命中后立即停跑并记录 stop_reason；缺少 `LLM_PRICES_JSON` 时拒绝 cost 闸门。测试覆盖数据集逐字校验、空稿/错引/改数字惩罚、临时库隔离、失败项隔离、报告持久化和 budget 闸门预检。仍缺：真实 LLM 下的全主题均值/方差报告，以及对单模式在线 smk 的护栏进一步细化。

### 验收口径提醒

- `citation_precision == 0` 不再代表 multi 写作失败，而是新的指标口径（verified + claim 绑定）。要拿 string-match 指标请看 `citation_string_match`。
- `unsupported_retention` 现在返回 `None` 而不是 `1 - unsupported_rate`；新指标正在积累可信样本，旧的退化定义已删除。
- 完整 4 主题 × 3 模式 × ≥2 次的在线 benchmark 仍未跑通：当前扩展 smoke 显示单主题三模式约 19 次 LLM 调用、500–600 秒，需要把单次上限提高到 `--max-llm-calls 200` 或更高才能跑完整套。
