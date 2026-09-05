"""检索健康探针：回答「此刻检索到底在不在工作」。

这不是上下文工程问题，是**一阶问题**——项目历史上量级最大的一次质量波动就出在这里：
`EMBED_MODEL` 填成了控制台上的中文显示名而不是 API 的真实 ID，chat 的中转端点又没有
`/embeddings` 路由（恒 404），于是系统**长期静默退化成纯 FTS**，没有任何地方报错，
评测数字看起来也「正常」，只是一直偏低。修好配置后自建集 Recall@5 0.6771 → 0.9427，
这 +0.2656 全部来自修配置，不是算法改进。

`search_hybrid` 里已经有降级说明，但那是**单次调用**的即时反馈，散落在各处的 degraded
字段里，没人会去看。探针把它变成一条可以在启动时、CI 里、或 `cli.py health` 手动跑的断言。

四项检查，每项独立给 ok / 原因，任何一项失败都不抛异常——探针自己不能成为新的故障点：
  1. 配置：EMBED_MODEL 填了吗？
  2. 索引：库里有多少条向量、维度是多少、模型名与配置对得上吗？
  3. 活体：现在打一次 embedding 调用能通吗？维度和库里一致吗？（`live=False` 可跳过）
  4. 检索：一次真实检索走的是混合路，还是降级成了纯 FTS？

`live=False` 只跑前两项——检索检查内部要 embed 查询，同样是网络调用，不能算「离线」。
而**前两项本身就足以照出历史上那次静默退化**：配置里的模型名与库里存的对不上，
这是纯 DB + 配置比对，0 次网络调用，正适合当 CI 门禁。

用法：
    python cli.py health           # 全部四项（会打真实 embedding 调用）
    python cli.py health --offline # 只跑配置与索引比对，0 次网络调用（CI 门禁）
"""
from __future__ import annotations

from . import config, db, embeddings

#: 探针用的固定查询。选它是因为它同时含中英文，且是项目 README 里那个招牌失败案例
#: （中文问、英文论文）——探针要盯的正是这条链路。
PROBE_QUERY = "近场信道估计 near-field channel estimation"


def _check(name: str, ok: bool | None, detail: str) -> dict:
    return {"name": name, "ok": ok, "detail": detail}


def _host_only(url: str) -> str:
    """把 URL 压成 host（含端口）。探针结果可能被匿名读取，不该泄露完整端点地址。"""
    from urllib.parse import urlparse
    try:
        p = urlparse(url or "")
        return p.netloc or (url or "")[:60]
    except Exception:
        return "（无法解析）"


def _check_config() -> dict:
    if not config.EMBED_MODEL:
        # 没配向量不是故障，是明确的降级选择——但要说清后果，别让人以为混合检索开着。
        return _check("配置", None, "未配置 EMBED_MODEL：检索为纯 FTS 词面匹配（非故障，但语义检索是哑的）")
    if not config.EMBED_API_BASE:
        return _check("配置", False, "配了 EMBED_MODEL 却没有 EMBED_API_BASE / LLM_API_BASE")
    # 只回显 host，不回显完整 URL。`/api/health/retrieval` 在未设口令时是可匿名
    # 访问的，把私有中转站的完整地址（含路径、有时还带 query）交出去没有必要。
    return _check("配置", True,
                  f"EMBED_MODEL={config.EMBED_MODEL!r} @ {_host_only(config.EMBED_API_BASE)}")


#: 向量覆盖率低于这个比例就判失败。取 0.95 而不是 1.0：刚采集进来还没跑 embed
#: 的几篇不该让探针变红，但**大面积缺失必须红**——审计时真实库是 104/503（20.7%），
#: 而当时四项检查全绿，因为没有任何一项在看覆盖率。
MIN_VECTOR_COVERAGE = 0.95


def _index_stats() -> tuple[dict[str, int], str | None, int | None]:
    """返回 (按 kind 的条数, 库里存的模型名, 向量维度)。空库时后两项为 None。"""
    counts: dict[str, int] = {}
    stored_model, dim = None, None
    with db.conn() as c:
        for r in c.execute("SELECT model, kind, COUNT(*) n FROM vectors "
                           "GROUP BY model, kind ORDER BY n DESC"):
            counts[r["kind"]] = counts.get(r["kind"], 0) + r["n"]
            stored_model = stored_model or r["model"]
        row = c.execute("SELECT vec FROM vectors LIMIT 1").fetchone()
    if row:
        dim = len(embeddings._from_blob(row["vec"]))
    return counts, stored_model, dim


def _coverage() -> tuple[int, int]:
    """返回 (能被向量检索到的论文数, 论文总数)。

    分子的口径必须与 `embeddings._paper_matrix` 完全一致（kind IN ('paper','chunk')
    且 model 匹配）——探针要回答的是「这篇论文能不能被语义检索命中」，
    不是「库里有多少条向量」。条数正常而覆盖率极低，恰恰是最难发现的那种退化。
    """
    with db.conn() as c:
        total = c.execute("SELECT COUNT(*) n FROM papers").fetchone()["n"]
        covered = c.execute(
            """SELECT COUNT(DISTINCT paper_id) n FROM vectors
               WHERE kind IN ('paper','chunk') AND model=?""",
            (config.EMBED_MODEL,)).fetchone()["n"]
    return covered, total


def _check_index(counts, stored_model, dim) -> dict:
    total = sum(counts.values())
    if not total:
        if not config.EMBED_MODEL:
            # 明确不用向量时，空索引是**一致的状态**，不是故障
            return _check("索引", None, "未配置向量，库内也没有索引（一致）")
        return _check("索引", False, "库内一条向量都没有（先跑 cli.py embed）")
    shape = "，".join(f"{k}={v}" for k, v in sorted(counts.items()))
    if config.EMBED_MODEL and stored_model != config.EMBED_MODEL:
        # 这正是那次静默退化的形态：配置里的模型名与建索引时用的对不上，
        # 于是每次检索都匹配不到任何向量，悄悄退回纯 FTS。
        return _check("索引", False,
                      f"库内向量的模型是 {stored_model!r}，与配置的 "
                      f"{config.EMBED_MODEL!r} **不一致** → 检索会匹配不到任何向量、"
                      f"静默退化成纯 FTS（核对真实 ID：cli.py models）")
    base = f"{total} 条向量（{shape}），维度 {dim}，模型 {stored_model!r}"
    if not config.EMBED_MODEL:
        return _check("索引", True, base)
    covered, papers = _coverage()
    if papers and covered / papers < MIN_VECTOR_COVERAGE:
        # 条数与模型名都对得上、却有大批论文压根没进向量索引：语义检索对它们
        # **恒不可达**，而 RRF 里向量路权重 1.0，进得去的那批被系统性抬高。
        # 这不是「覆盖不全」，是排序偏置，且完全静默——原来这里直接返回 ok。
        return _check("索引", False,
                      f"{base}；但只有 {covered}/{papers} 篇论文进得了向量索引"
                      f"（{covered / papers:.1%}，阈值 {MIN_VECTOR_COVERAGE:.0%}）"
                      f"——其余 {papers - covered} 篇对语义检索恒不可见，"
                      f"混合检索实际只对一部分库生效（补齐：cli.py embed）")
    return _check("索引", True, f"{base}；向量覆盖 {covered}/{papers}")


def _check_live(dim: int | None) -> dict:
    if not embeddings.available():
        return _check("活体", None, "未配置向量，跳过")
    try:
        vec = embeddings.embed_texts([PROBE_QUERY], purpose="health")[0]
    except Exception as exc:
        return _check("活体", False,
                      f"embedding 调用失败：{type(exc).__name__}: {str(exc)[:160]}")
    if not vec:
        return _check("活体", False, "embedding 返回空向量")
    if dim is not None and len(vec) != dim:
        return _check("活体", False,
                      f"在线维度 {len(vec)} 与库内 {dim} 不一致——换过模型就必须重建索引")
    return _check("活体", True, f"embedding 调用正常，维度 {len(vec)}")


def _check_retrieval() -> dict:
    try:
        _r = embeddings.search_hybrid(PROBE_QUERY, 5)
        ids, mode, notes = _r.ids, _r.mode, _r.degraded
    except Exception as exc:
        return _check("检索", False, f"检索抛异常：{type(exc).__name__}: {str(exc)[:160]}")
    if not ids:
        return _check("检索", False, f"探针查询一篇都没召回（mode={mode}）")
    # 降级现在由 search_hybrid 明确给出，不再从 mode 反推。mode 仍然要看：
    # 它覆盖「没有任何降级记录、但向量路就是没进融合」这种口径不一致的情况。
    if notes:
        return _check("检索", False, "；".join(n.message for n in notes))
    if embeddings.available() and not mode.startswith("hybrid"):
        return _check("检索", False,
                      f"配了向量却走的是 {mode!r}——语义检索这一路没生效")
    return _check("检索", True, f"召回 {len(ids)} 篇，口径 {mode!r}")


def probe(live: bool = True) -> dict:
    """跑一遍检索健康检查。返回 {"ok", "checks", "summary"}，不抛异常。

    ok 只在**有明确失败项**时为 False；口径为 None 的项（未配置向量）不算失败——
    「明确选择不用向量」和「以为在用其实没用上」必须区分开，这正是探针存在的理由。
    """
    db.init_db()
    counts, stored_model, dim = _index_stats()
    checks = [_check_config(), _check_index(counts, stored_model, dim)]
    if live:
        checks.append(_check_live(dim))
        checks.append(_check_retrieval())
    else:
        # 检索检查内部会 embed 查询，和活体一样是网络调用——离线就一起跳过，
        # 不能把「跳过了」说成「0 次网络调用还全跑了」。
        checks.append(_check("活体", None, "离线模式跳过"))
        checks.append(_check("检索", None, "离线模式跳过（需要 embedding 调用）"))
    failed = [c for c in checks if c["ok"] is False]
    skipped = [c for c in checks if c["ok"] is None]
    summary = (f"{len(checks) - len(failed) - len(skipped)} 项通过"
               f"，{len(failed)} 项失败，{len(skipped)} 项未检查")
    return {"ok": not failed, "checks": checks, "summary": summary}
