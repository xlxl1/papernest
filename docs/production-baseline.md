# PaperNest 生产加固基线

日期：2026-10-04（Asia/Shanghai）
代码基线：`v0.1-audit-baseline`（`cf4a0a9`）

## 数据集与检索

- 本地库：503 篇论文，4816 条向量，503/503 篇有向量覆盖。
- 自建评测 Recall@5：0.9479（32 条关键词查询）。
- 自然语言查询 Recall@5：1.0000（18 条）。
- 意图路由准确率：1.0000（34 条）。
- QASPER paper hit@5：0.7500（88 题）。
- QASPER evidence recall@sec2：0.4091（88 题）。
- 引用推荐 precision@5：0.2250（8 条）。

## 可复现命令

```powershell
python -m compileall -q papernest cli.py verify_sources.py
python cli.py eval --limit 50
python cli.py qasper eval
python cli.py health --offline
python -m unittest discover -s tests -p 'test_eval.py' -q
```

## 环境说明

- `compileall`、CLI help/demo/health 离线检查通过。
- `pip check` 通过。
- 完整 unittest 在当前 Windows 沙箱中受临时目录权限限制，不能把失败结果当作业务回归结论；阶段 1/2 的边界和任务回归已通过工作区内临时目录冒烟验证。
- 本文件是阶段 0 基线，后续指标必须同时记录旧值、新值和评测命令。

## 阶段 0/1/2 已实施的加固

- 阶段 0：保留 `v0.1-audit-baseline` 标签与本文件；增加 `python cli.py backup [path]`，使用 SQLite backup API 和 `PRAGMA integrity_check` 生成一致性快照。
- 阶段 1：关闭默认 `/openapi.json`；普通 API 只接受 `X-API-Key` 或鉴权后下发的 HttpOnly cookie，仅 SSE 允许 query key；统一 request ID 响应头和基础安全头；为消息、论文 ID、文本、分页、矩阵、上传批次设置上下界；页面接口支持分页；Markdown 重建复用本地 PDF 归属校验；外部/生成失败改用明确的 4xx/5xx。
- 阶段 2：笔记创建/更新校验关联论文；文档更新支持 `expected_version`/`version` 乐观锁并在冲突时返回 409；任务表新增幂等键、尝试次数、取消标记，提供取消接口、数据库级跨进程并发上限和可配置重试/重启恢复；预算数据库不可读时默认 fail-closed，可显式设置 `PAPERNEST_BUDGET_FAIL_OPEN=1` 作为开发环境逃生口。
- 容器运行面：Dockerfile 增加非 root 用户、只读静态入口健康检查和 Python 非缓冲输出配置。

## 阶段 1/2 验证

```powershell
python -m compileall -q papernest cli.py verify_sources.py
python -m unittest discover -s tests -p 'test_eval.py' -q
python cli.py health --offline
python cli.py backup .test-tmp/papernest-backup.db --force
```

Windows 沙箱仍禁止测试创建系统临时目录；依赖临时数据库的完整 API/任务测试会报告 `PermissionError`，这不是业务断言失败。已用工作区内 `.test-tmp` 做迁移、幂等、取消、乐观锁、预算故障闭合和 API 边界冒烟验证。
