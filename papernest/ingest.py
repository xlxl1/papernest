"""采集流水线：关键词 → 查询缓存 → 检索（多源 + 去重）→ 增量 diff → L0/L1 入库。

验收证据链在 search_runs 表：同关键词二次调用 results>0 而 new_papers=0、
llm_calls=0 —— 「论文只精读一次」由数据说话。查询结果集本身也缓存：
同课题二次查询连外网请求都不出，离线也能跑。
"""
import json

from . import cards, db, llm
from .sources import arxiv, openalex, semantic_scholar


def _fetch_sources(query: str, limit: int) -> tuple[list[dict], str]:
    """查询路由三层：① 查询结果集缓存（同查询零外网请求）
    → ② S2 主源（arXiv 合并去重）→ ③ OpenAlex 兜底。
    全部失败才抛 RuntimeError（带每步死因与可行动建议）。
    """
    with db.conn() as c:
        hit = db.get_query_cache(c, query)
        if hit:
            keys = json.loads(hit["keys_json"])
            rows = [db.get_by_norm_key(c, k) for k in keys]
            found = [r for r in rows if r]
            if found:
                out = [{"norm_key": r["norm_key"], "title": r["title"],
                        "abstract": r["abstract"], "year": r["year"]} for r in found]
                return out, f"query-cache[{hit['source']}]"

    merged: dict[str, dict] = {}
    used = []
    errors = []
    s2_papers = None
    try:
        s2_papers = semantic_scholar.search(query, limit)
    except Exception as e:
        errors.append(f"S2: {e}")
    if s2_papers:
        used.append("s2")
        for p in s2_papers:
            if p["norm_key"] and p["norm_key"] not in merged:
                merged[p["norm_key"]] = p
    else:
        try:
            oa_papers = openalex.search(query, limit)
            used.append("openalex(S2兜底)")
            for p in oa_papers:
                if p["norm_key"] and p["norm_key"] not in merged:
                    merged[p["norm_key"]] = p
        except Exception as e:
            errors.append(f"OpenAlex兜底: {e}")
    try:
        ax_papers = arxiv.search(query, limit)
        used.append("arxiv")
        for p in ax_papers:
            if p["norm_key"] and p["norm_key"] not in merged:
                merged[p["norm_key"]] = p
        arxiv.polite_sleep()
    except Exception as e:
        errors.append(f"arXiv: {e}")

    if not merged:
        raise RuntimeError(
            "三个数据源都不可用（" + "；".join(errors)[:300] + "）。"
            "可行动作：① 稍等几分钟重试（S2 共享池限流按 5 分钟窗口恢复）；"
            "② 在 .env 填 S2_API_KEY（免费申请）；"
            "③ 本机网络屏蔽 OpenAlex/arXiv 时，开启代理并在 .env 填 "
            "PAPERNEST_PROXY=http://127.0.0.1:7897")
    with db.conn() as c:
        db.put_query_cache(c, query, "+".join(used),
                           [p["norm_key"] for p in merged.values() if p.get("norm_key")])
        c.commit()
    return list(merged.values()), "+".join(used)


def ingest(query: str, limit: int = 20, topic: str | None = None,
           year_from: int | None = None, year_to: int | None = None,
           progress=None, make_cards: bool = True) -> dict:
    """query 为已合并的关键词串；年份筛选在检索结果上做（查询缓存存的原始结果集）。

    progress(frac, stage, message)：异步任务的进度回调，CLI/同步调用不传即可。
    make_cards=False 时不生成 L1 卡片（规模实验用：纯元数据入库，0 次 LLM）。
    """
    db.init_db()
    papers, source_tag = _fetch_sources(query, limit)
    year_filtered = 0
    if year_from or year_to:
        before = len(papers)
        papers = [p for p in papers if p.get("year") and
                  (not year_from or p["year"] >= year_from) and
                  (not year_to or p["year"] <= year_to)]
        year_filtered = before - len(papers)
    new_papers = 0
    llm_calls = 0
    cache_hits = 0
    skipped = 0
    with db.conn() as c:
        run_id = db.new_search_run(c, query, source_tag)
        c.commit()
    total = max(len(papers), 1)
    for i, p in enumerate(papers):
        if progress:
            progress(i / total, "ingest", f"处理 {i + 1}/{total}：{p.get('title', '')[:40]}")
        if not p.get("norm_key"):
            skipped += 1
            continue
        with db.conn() as c:
            if db.get_by_norm_key(c, p["norm_key"]):
                cache_hits += 1
                continue  # 增量 diff：库内论文直接命中，0 LLM 调用
            pid = db.insert_l0(c, p)
            c.commit()
        p["_db_id"] = pid
        if make_cards:
            card, model = cards.make_card(p, topic)
            with db.conn() as c:
                db.save_card(c, pid, card, model)
                c.commit()
            if model != "mock-extractive":
                llm_calls += 1
        new_papers += 1
    with db.conn() as c:
        db.finish_search_run(c, run_id, len(papers), new_papers, llm_calls)
        c.commit()
    return {
        "query": query, "source": source_tag, "results": len(papers),
        "new_papers": new_papers, "cache_hits": cache_hits,
        "llm_calls": llm_calls, "skipped_no_key": skipped,
        "year_filtered": year_filtered,
    }


def dedupe_preview(query: str, limit: int = 20) -> list[str]:
    papers, _ = _fetch_sources(query, limit)
    keys = [p["norm_key"] for p in papers if p.get("norm_key")]
    return [k for k in dict.fromkeys(keys)]
