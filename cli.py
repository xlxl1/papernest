"""PaperNest CLI。

  python cli.py verify  [--query Q] [--limit N]   第 0 步数据源验证
  python cli.py ingest  "关键词" [--limit N] [--topic T]
  python cli.py stats                             成本与缓存证据
  python cli.py eval    [--k 5] [--qa] [--limit N] 评测三指标（--qa 需 LLM key）
  python cli.py demo                              离线演示（不连外网不调模型）
  python cli.py search  "关键词"                  库内检索（FTS）
  python cli.py models                            列出 API 实际可用的模型 ID（配 key 后跑）
  python cli.py embed                             为全库建/补向量索引
  python cli.py cite    "段落" [--topk N]         引用推荐（带支撑证据句）
  python cli.py export  1,2 --fmt bibtex|ris|gbt7714|ieee
  python cli.py read    <paper_id>                L2：下载 OA PDF + 按页抽取 + 精读卡片
  python cli.py ppt     1,2,3 [--topic T]         生成汇报 PPT（单篇=论文汇报，多篇=文献汇报）
  python cli.py qasper  download|import|eval      QASPER 公开基准：下载/导入/证据评测
  python cli.py scale   "q1;q2" [--limit N]       规模实验：批量采集 + 吞吐/去重/延迟报告
  python cli.py write   "课题描述" [--words N]     二期写作流水线：创建任务 + 选题 Agent
  python cli.py write-pick <run_id> "题目"        检查点①定题 → 大纲 Agent
  python cli.py write-outline <run_id>            查看当前大纲
  python cli.py write-go <run_id>                 检查点②确认开写（逐节撰写⇄文献白名单）
  python cli.py write-status <run_id>             流水线状态与各节评审分
  python cli.py write-finish <run_id>             润色 + 机械终检 + 导出 md/docx
  python cli.py write-export <run_id> [--fmt md|docx]  查看终稿
  python cli.py recard [--limit N]                mock 卡片重刷为真卡片
  python cli.py serve   [--port 8765]             启动 Web
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Windows 控制台默认 GBK，放不下 ✓/✗/→ 这类字符——命令跑到最后一行 print 才炸，
# 而且炸在"已经把事情做完了"之后，最有迷惑性。入口处统一切 UTF-8，编码不了的降级替换。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from papernest import cards, cite, config, db, embeddings, fulltext, ingest, llm  # noqa: E402


def cmd_verify(args):
    import verify_sources
    rep = verify_sources.run(args.query, args.limit)
    ok = verify_sources.print_report(rep)
    save = config.DATA_DIR / "verify_report.json"
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    save.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告已存：{save}")
    sys.exit(0 if ok else 1)


def cmd_ingest(args):
    try:
        r = ingest.ingest(args.query, args.limit, args.topic, make_cards=not args.no_cards)
    except RuntimeError as e:
        print(f"[ingest 失败] {e}")
        sys.exit(1)
    print(f"[ingest] query={r['query']!r}")
    print(f"  来源          : {r['source']}")
    print(f"  检索结果      : {r['results']} 篇（无法取主键跳过 {r['skipped_no_key']}）")
    print(f"  新入库        : {r['new_papers']} 篇（L0+L1）")
    if r.get("vectors_indexed"):
        print(f"  向量索引      : {r['vectors_indexed']} 条")
    print(f"  缓存命中      : {r['cache_hits']} 篇（0 次 LLM 调用）")
    print(f"  本轮 LLM 调用 : {r['llm_calls']} 次")
    if r["cache_hits"] and r["llm_calls"] == 0:
        print("  >>> 二次查询零 LLM 调用：验收点成立")


def cmd_health(args):
    """检索健康探针：一条命令回答「此刻检索到底在不在工作」。

    退出码非 0 便于直接挂进 CI —— 这是项目里唯一一个「静默失效会让所有数字变差、
    却不会让任何测试变红」的地方（EMBED_MODEL 填错导致长期退化成纯 FTS 的那次，
    Recall@5 从 0.6771 到 0.9427 全部来自修配置）。
    """
    from papernest import health
    r = health.probe(live=not args.offline)
    icon = {True: "✔", False: "✘", None: "–"}
    for c in r["checks"]:
        print(f"  {icon[c['ok']]} {c['name']}：{c['detail']}")
    print(f"\n{r['summary']}")
    if not r["ok"]:
        sys.exit(1)


def cmd_stats(args):
    db.init_db()
    with db.conn() as c:
        s = db.stats(c)
        s["vectors"] = c.execute("SELECT COUNT(*) n FROM vectors").fetchone()["n"]
        s["pages"] = c.execute("SELECT COUNT(*) n FROM pages").fetchone()["n"]
    print(f"论文总数 {s['papers_total']} | L0 {s['level_counts']['L0']} / "
          f"L1 {s['level_counts']['L1']} / L2 {s['level_counts']['L2']}"
          f" | mock 卡 {s['mock_cards']} | 向量 {s['vectors']} | 全文页 {s['pages']}")
    cost = f" | cost ${s['cost_usd']}" if s.get("cost_usd") else ""
    lat = f" | 平均延迟 {s['avg_latency_ms']}ms" if s.get("avg_latency_ms") else ""
    print(f"LLM 调用 {s['llm_calls']} 次 | prompt {s['prompt_tokens']} tok | "
          f"completion {s['completion_tokens']} tok{cost}{lat}")
    print("最近检索（增量缓存证据）:")
    for r in s["recent_search_runs"]:
        print(f"  #{r['id']} [{r['ts']}] {r['query']!r} via {r['source']}: "
              f"结果 {r['results']} / 新 {r['new_papers']} / LLM {r['llm_calls']}")


def cmd_search(args):
    db.init_db()
    with db.conn() as c:
        rows = db.search_fts(c, args.query, args.limit)
    print(f"库内命中 {len(rows)} 篇：")
    for r in rows:
        print(f"  [{r['id']}] (L{r['level']}) {r['title'][:70]}  {r['venue'] or ''} {r['year'] or ''}")


def cmd_models(args):
    """列出 key 实际可用的模型 ID（.env 里填错 ID 时用它核对）。"""
    if not config.LLM_API_KEY:
        print("先在 .env 填 LLM_API_KEY。")
        sys.exit(1)
    import httpx  # noqa
    from papernest import http
    with http.client(timeout=30) as client:
        r = client.get(config.LLM_API_BASE.rstrip("/") + "/models",
                       headers={"Authorization": f"Bearer {config.LLM_API_KEY}"})
        r.raise_for_status()
        data = r.json().get("data") or []
    ids = sorted(m.get("id", "") for m in data)
    print(f"共 {len(ids)} 个模型 ID（.env 里填这些准确值）：")
    for i in ids:
        print(" ", i)


def cmd_embed(args):
    db.init_db()
    if not embeddings.available():
        print("未配置向量模型（.env 的 EMBED_MODEL + LLM_API_KEY）。")
        sys.exit(1)
    with db.conn() as c:
        rows = c.execute("""SELECT id, title, abstract FROM papers
                            WHERE id NOT IN (SELECT DISTINCT paper_id FROM vectors
                                             WHERE kind='paper' AND model=?)""",
                         (config.EMBED_MODEL,)).fetchall()
        old_models = [r["model"] for r in
                      c.execute("SELECT DISTINCT model FROM vectors WHERE kind='paper'").fetchall()
                      if r["model"] and r["model"] != config.EMBED_MODEL]
        n_vecs = c.execute("SELECT COUNT(*) n FROM vectors WHERE kind='paper' AND model=?",
                           (config.EMBED_MODEL,)).fetchone()["n"]

    # 换了提供方/模型名但库里有旧向量：先验证新旧空间是否兼容，
    # 兼容则原地改名（0 token 完成迁移），不兼容才全量重建。
    if rows and old_models and n_vecs == 0:
        compat = embeddings.check_space_compatible(rows[0]["id"], rows[0]["title"],
                                                   rows[0]["abstract"] or "")
        if compat is True:
            with db.conn() as c:
                c.execute("UPDATE vectors SET model=? WHERE model=?",
                          (config.EMBED_MODEL, old_models[0]))
                c.commit()
            print(f"√ 新旧向量空间一致（余弦 ≥ 0.98），已原地迁移 {old_models[0]} → {config.EMBED_MODEL}，0 次重嵌入。")
            return
        print(f"× 新旧向量空间不一致（余弦 {compat if isinstance(compat, float) else '未知'}），全量重建。")

    total = 0
    t0 = __import__("time").perf_counter()
    for i, r in enumerate(rows, 1):
        n = embeddings.index_paper(r["id"], r["title"], r["abstract"] or "")
        total += n
        print(f"  [{i}/{len(rows)}] #{r['id']} {r['title'][:50]} -> {n} 条向量")
    dt = __import__("time").perf_counter() - t0
    print(f"完成：新增 {total} 条向量，耗时 {dt:.1f}s（{total / max(dt, 0.001):.0f} 条/s）。")


def cmd_cite(args):
    r = cite.recommend(args.text, args.topk)
    if r.get("degraded"):
        print(f"(降级：{r['degraded']})")
    for c in r["candidates"]:
        flag = "✓强" if c["strong"] else ("△弱" if c["verified"] else "×未核")
        print(f"  [{c['paper_id']}] {flag} score={c['score']} {c['title'][:60]}")
        if c["evidence_sentence"]:
            print(f"      证据句（{'已通过机械回取校验' if c['verified'] else '未通过校验'}）："
                  f"{c['evidence_sentence'][:100]}")
        else:
            print("      （摘要无句级数据，未配置向量模型时可能无证据句）")


def cmd_export(args):
    ids = [int(x) for x in args.ids.split(",") if x.strip()]
    print(cite.export(ids, args.fmt))


def cmd_read(args):
    r = fulltext.read_paper(args.paper_id, args.topic)
    if "error" in r:
        print(f"[read 失败] {r['error']}")
        sys.exit(1)
    print(f"[read] 论文 {r['paper_id']}：全文抽取 {r['pages_stored']} 页入库")
    card = r.get("card")
    if card:
        print(f"  精读卡片：{card.get('_verified_summary', '')}")
        for f in card.get("key_findings") or []:
            mark = "✓" if f.get("verified") else "?"
            print(f"  {mark} p{f.get('page','?')} {f.get('claim','')[:80]}")
    else:
        print(f"  {r.get('note','')}")


def cmd_export_obsidian(args):
    """导出 Obsidian 文献笔记：citekey 命名 + YAML frontmatter + 库内互引双链。"""
    from papernest import obsidian
    ids = ([int(x) for x in args.ids.split(",") if x.strip()] if args.ids else None)
    r = obsidian.export_vault(args.out, ids)
    print(f"[obsidian] 写出 {r['written']} 篇 → {r['out_dir']}")
    for f in r["files"][:8]:
        print(f"  · {f}")
    if r["written"] > 8:
        print(f"  …… 共 {r['written']} 个文件")
    for e in r["errors"][:5]:
        print(f"  ⚠ {e}")


def cmd_ask(args):
    """命令行问答。--deep 走迭代闭环补检索；--rcs 走重排 + 逐篇定向摘要。"""
    from papernest import deepsearch, rag, rcs
    if args.rcs:
        r = rcs.answer(args.question, top_k=args.top_k)
        print(r["answer"])
        t = r["trace"]
        rr, sm = t.get("rerank") or {}, t.get("summary") or {}
        print(f"\n—— RCS · 检索口径 {r['retrieval_mode']} ——")
        print(f"  重排：候选池 {rr.get('pool', 0)} 篇"
              + ("（模型重排生效）" if rr.get("used") else f"（未生效：{rr.get('error', '无 LLM')}）"))
        for s in (rr.get("scores") or [])[:5]:
            print(f"    {s['score']:>2}/10  #{s['paper_id']}  {s['why']}")
        print(f"  定向摘要：{sm.get('calls', 0)} 次调用 → 保留 {sm.get('kept', 0)} 篇"
              f"，判为不相关剔除 {sm.get('dropped_irrelevant', 0)} 篇")
        if sm.get("quotes_total"):
            print(f"  依据句机械回取：{sm['quotes_verified']}/{sm['quotes_total']} 通过"
                  f"（{r.get('verified_rate')}）")
        for e in (sm.get("errors") or [])[:3]:
            print(f"    ⚠ {e}")
    elif args.deep:
        # 与 api.DeepBody 同口径夹紧：CLI 也不该能把上下文顶到无上界
        rounds = max(1, min(args.rounds, 5))
        if rounds != args.rounds:
            print(f"  ⚠ --rounds {args.rounds} 超出范围，已夹到 {rounds}")
        r = deepsearch.deep_answer(args.question, top_k=args.top_k,
                                   max_rounds=rounds)
        print(r["answer"])
        print(f"\n—— 共 {r['rounds']} 轮 · 检索口径 {r['retrieval_mode']} ——")
        if r.get("unsupported"):
            print(f"  ⚠ 交付时仍有 {len(r['unsupported'])} 条论断无库内支撑（已标注在答案末尾）")
        for it in r["trace"]["iterations"]:
            line = (f"  第 {it['round']} 轮：上下文 {it['papers_in_context']} 篇，"
                    f"无证据论断 {it['unsupported_claims']} 条")
            if it.get("queries"):
                line += f"，补充查询 {it['queries']} → 新增 {it['new_papers']} 篇"
            if it.get("gated_out"):
                line += f"（闸门挡下 {it['gated_out']} 篇）"
            if it.get("dropped_for_budget"):
                line += f"（超篇数预算淘汰 {it['dropped_for_budget']} 篇）"
            if it.get("queries_from"):
                line += f"［{it['queries_from']}］"
            if it.get("stop"):
                line += f"（停止：{it['stop']}）"
            print(line)
    else:
        r = rag.answer([{"role": "user", "content": args.question}], top_k=args.top_k)
        print(r["answer"])
    for s in r.get("sources") or []:
        print(f"  [{s['idx']}] {s['title'][:70]}（{s.get('venue') or ''} {s.get('year') or ''}）")
    if r.get("degraded"):
        print(f"  ⚠ {r['degraded']}")


def cmd_digest(args):
    """新论文摘报：按库内画像给 arXiv 新文打分，只报没推过的。"""
    from papernest import subscribe
    if args.profile:
        p = subscribe.build_profile()
        print(f"[画像] 取自库内 {p['n_papers']} 篇 · 猜测分类 {p['arxiv_categories'] or '（猜不出）'}")
        for t in p["terms"][:15]:
            print(f"  {t['weight']:.3f}  {t['term']}")
        return
    cats = [c.strip() for c in (args.categories or "").split(",") if c.strip()]
    r = subscribe.digest(days=args.days, top_k=args.top_k, categories=cats or None,
                         use_llm=args.llm, repeat_after_days=args.repeat_after)
    for d in r.get("degraded") or []:
        print(f"  ⚠ {d}")
    print(f"[digest #{r.get('digest_id')}] 拉取 {r.get('n_fetched', 0)} 篇 → "
          f"去重后 {r.get('n_new', 0)} 篇新论文")
    for it in r.get("items") or []:
        print(f"\n  {it['score']:.3f}  {it['title'][:70]}")
        print(f"        arXiv:{it['arxiv_id']}  {it.get('published', '')[:10]}")
        for rs in (it.get("reasons") or [])[:3]:
            print(f"        · 命中「{rs['term']}」（{rs['where']}，贡献 {rs['contribution']:.3f}）")
        if it.get("llm_note"):
            print(f"        · 模型判断：{it['llm_note'][:80]}")


def cmd_matrix(args):
    """跨论文对比矩阵：抽不到的格子留空，表尾给覆盖率与核验率。"""
    from papernest import matrix
    ids = [int(x) for x in args.ids.split(",") if x.strip()]
    use_llm = None if args.llm is None else args.llm
    m = matrix.build(ids, use_llm=use_llm)
    print(matrix.export(m, args.fmt))
    print(f"\n覆盖率 {m['coverage']:.0%} · 机械核验率 "
          f"{'—' if m.get('verified_rate') is None else format(m['verified_rate'], '.0%')}"
          f" · {len(m['rows'])} 篇")
    for d in (m.get("degraded") or []) if isinstance(m.get("degraded"), list) else [m["degraded"]] if m.get("degraded") else []:
        print(f"  ⚠ {d}")
    for e in (m.get("errors") or [])[:5]:
        print(f"  ⚠ {e}")


def cmd_structure(args):
    """看某篇本地 PDF 的版面结构：章节树 + 参考文献抽取结果。"""
    from papernest import db, structure
    with db.conn() as c:
        row = c.execute("SELECT title, pdf_path FROM papers WHERE id=?",
                        (args.paper_id,)).fetchone()
    if not row or not row["pdf_path"]:
        print("[structure] 这篇论文没有本地 PDF（先 read 或 import-pdf）")
        sys.exit(1)
    s = structure.summarize(row["pdf_path"])
    print(f"[structure] {row['title'][:60]}")
    print(f"  正文字号 {s.get('body_font_size')} · {len(s.get('sections') or [])} 个章节 · "
          f"{s.get('n_chunks')} 个 chunk")
    if s.get("degraded"):
        print(f"  ⚠ {s['degraded']}")
    for sec in (s.get("sections") or [])[:20]:
        print(f"    {'  ' * (sec.get('level', 1) - 1)}p{sec.get('page_no')} "
              f"{sec.get('title', '')[:60]}  [{'/'.join(sec.get('matched_by') or [])}]")
    refs = s.get("references") or {}
    print(f"  参考文献 {refs.get('n_entries', 0)} 条"
          f"（带 DOI {refs.get('with_doi', 0)} · 带 arXiv {refs.get('with_arxiv', 0)}）")
    for e in (refs.get("entries") or [])[:5]:
        print(f"    · {(e.get('title_guess') or e.get('raw', ''))[:60]}"
              f"{'  doi:' + e['doi'] if e.get('doi') else ''}")


def cmd_reindex(args):
    """重建页级全文索引（pages_fts）。迁移会自动回填一次，这里是手动兜底/排障用。"""
    from papernest import db
    db.init_db()
    with db.conn() as c:
        pids = [r["paper_id"] for r in
                c.execute("SELECT DISTINCT paper_id FROM pages").fetchall()]
        c.execute("DELETE FROM pages_fts")
        for pid in pids:
            db.reindex_pages(c, pid)
        n = c.execute("SELECT COUNT(1) n FROM pages_fts").fetchone()["n"]
    print(f"[reindex] {len(pids)} 篇有全文的论文，pages_fts 共 {n} 条")


def cmd_import_doc(args):
    """多格式文档入库（Word/PPT/HTML/Markdown/纯文本）。PDF 请用 import-pdf。"""
    from papernest import docimport
    paths = docimport.collect_paths(args.paths)
    if not paths:
        print("[import-doc] 没有找到支持的文档"
              "（.docx/.pptx/.html/.md/.txt；PDF 用 import-pdf）")
        sys.exit(1)
    r = docimport.import_batch(paths, topic=args.topic, make_card=not args.no_cards)
    print(f"[import-doc] 共 {r['total']} 份："
          f"入库 {r['ok']} · 跳过 {r['skipped']} · 失败 {r['failed']}")
    for fmt, c in sorted(r["by_format"].items()):
        print(f"  {fmt:12s} ok {c['ok']} / skip {c['skipped']} / fail {c['failed']}")
    for it in r["items"]:
        if it["status"] == "ok":
            print(f"  ✓ #{it['paper_id']} [{it['format']}] {it['title'][:56]}"
                  f"（{it['n_chars']} 字 → {it['chunks']} 块）")
            for w in it.get("warnings") or []:
                print(f"      ⚠ {w}")
        else:
            print(f"  {'—' if it['status'] == 'skipped' else '✗'} "
                  f"{Path(it['path']).name}：{it.get('reason')}")


def cmd_sectiontree(args):
    """章节树索引：建索引 / 看规模 / 两级检索（先选范围再检索）。"""
    from papernest import sectiontree
    if args.action == "build":
        r = sectiontree.build_all()
        print(f"[sectiontree] 建索引 {r['indexed']} 篇 · 跳过 {r['skipped']} · "
              f"失败 {r['failed']}")
        for e in (r.get("errors") or [])[:5]:
            print(f"  ⚠ {e}")
    elif args.action == "stats":
        s = sectiontree.stats()
        print(f"[sectiontree] 已索引 {s['papers_indexed']} 篇 · 节点 {s['n_nodes']} 个 · "
              f"词项 {s['n_terms']} 条 · 平均每篇 {s['avg_nodes_per_paper']} 节")
    else:
        r = sectiontree.two_stage_search(args.query, args.top_k)
        t = r["trace"]
        print(f"[sectiontree] 口径 {r['mode']}：第一级筛出 {t['stage1_papers']} 篇 / "
              f"{t['stage1_nodes']} 节 → 第二级命中 {t['stage2_hits']} 条")
        for p in r["passages"][:args.top_k]:
            print(f"  #{p['paper_id']} {p.get('section_path', '')[:50]}")
            print(f"     {(p.get('text') or '')[:110]}")


def cmd_vec(args):
    """向量后端管理：状态 / 重建索引 / 为章节块补向量。"""
    from papernest import chunkembed, vectorstore
    if args.action == "status":
        store, degraded = vectorstore.get_store_or_degrade()
        cov = chunkembed.coverage()
        print(f"[vec] 后端 {store.name}" + (f"（降级：{degraded}）" if degraded else ""))
        print(f"  SQLite vectors（真相来源）: {store.count()} 条")
        idx = getattr(store, "index_count", None)
        if idx is not None:
            n = idx()
            flag = "" if n == store.count() else "  ← 与真相不一致，请跑 vec rebuild"
            print(f"  后端索引条数              : {n}{flag}")
        print(f"  章节块向量覆盖            : {cov['chunks_embedded']}/{cov['chunks_total']}"
              f"（{cov['coverage']:.1%}）")
    elif args.action == "rebuild":
        store = vectorstore.get_store()
        r = store.rebuild()
        print(f"[vec] 从 SQLite 重建 {r['backend']} 索引：{r['indexed']} 条，"
              f"耗时 {r['elapsed_s']:.1f}s")
        for e in (r.get("errors") or [])[:5]:
            print(f"  ⚠ {e}")
    else:                      # embed-chunks
        r = chunkembed.embed_chunks(limit=args.limit, dry_run=args.dry_run)
        if args.dry_run:
            print(f"[vec] 试算：{r['n_chunks']} 块 / {r['total_chars']} 字 / "
                  f"约 {r['est_tokens']} tokens（{r.get('note', '')}）")
            return
        print(f"[vec] 嵌入 {r['embedded']} 块（跳过 {r['skipped']} · "
              f"截断 {r['truncated']} · 批次 {r['batches']}），耗时 {r['elapsed_s']:.1f}s")
        if r.get("aborted"):
            print(f"  ⚠ 已中止：{r['aborted']}")
        for e in (r.get("errors") or [])[:5]:
            print(f"  ⚠ {e}")
        cov = chunkembed.coverage()
        print(f"  覆盖率 {cov['chunks_embedded']}/{cov['chunks_total']}（{cov['coverage']:.1%}）")


def cmd_rechunk(args):
    """把已有全文重切成章节级检索单元（有 PDF 的走章节切，没有的退化按页）。

    实测收益（25 篇真实 arXiv PDF、69 道 QASPER 带 gold 证据的题）：
    等上下文预算下证据召回是按页切的 2.2~3.2 倍。存量库跑一次即可。
    """
    from papernest import db, fulltext
    db.init_db()
    with db.conn() as c:
        rows = c.execute("""SELECT DISTINCT p.id, p.pdf_path FROM papers p
                            JOIN pages g ON g.paper_id=p.id ORDER BY p.id""").fetchall()
    sect = paged = 0
    with db.conn() as c:
        for r in rows:
            path = r["pdf_path"]
            chunks = (fulltext.build_chunks(path, c, r["id"])
                      if path and Path(path).exists()
                      else db.chunks_from_pages(c, r["id"]))
            # build_chunks 失败会退化成按页，这里据此分类计数（如实反映降级比例）
            got_sections = any(ch.get("section_path") for ch in chunks)
            db.replace_chunks(c, r["id"], chunks)
            if got_sections:
                sect += 1
            else:
                paged += 1
        n = c.execute("SELECT COUNT(1) n FROM chunks").fetchone()["n"]
    print(f"[rechunk] {len(rows)} 篇有全文：按章节切 {sect} 篇 · 退化按页 {paged} 篇")
    print(f"[rechunk] chunks 共 {n} 条")


def cmd_import_pdf(args):
    """本地 PDF 入库：逐份独立成败，抽错的元数据当场提示可用 fix-meta 修。"""
    from pathlib import Path as _P

    from papernest import pdfimport
    paths: list[_P] = []
    for pattern in args.paths:
        p = _P(pattern)
        paths.extend(sorted(p.glob("*.pdf")) if p.is_dir() else [p])
    if not paths:
        print("[import-pdf] 没有找到 PDF")
        sys.exit(1)
    ok = fail = dup = 0
    for p in paths:
        try:
            r = pdfimport.ingest_pdf(p.read_bytes(), p.name, topic=args.topic,
                                     make_card=not args.no_cards)
        except Exception as e:
            fail += 1
            print(f"  ✗ {p.name}：{e}")
            continue
        ok += 1
        dup += 1 if r["duplicate"] else 0
        flag = f"（已存在，{r['duplicate_by']} 去重）" if r["duplicate"] else ""
        print(f"  ✓ #{r['paper_id']} {r['title'][:60]}{flag}  {r['pages_stored']} 页")
        for w in r["warnings"]:
            print(f"      ⚠ {w}")
    print(f"[import-pdf] 成功 {ok}（其中重复 {dup}）· 失败 {fail} · 共 {len(paths)} 份")
    if fail:
        sys.exit(1)


def cmd_fix_meta(args):
    from papernest import pdfimport
    fields = {k: v for k, v in
              (("title", args.title), ("year", args.year), ("doi", args.doi),
               ("venue", args.venue), ("arxiv_id", args.arxiv_id))
              if v is not None}
    if args.authors:
        fields["authors"] = [a.strip() for a in args.authors.split(";") if a.strip()]
    try:
        r = pdfimport.update_metadata(args.paper_id, **fields)
    except Exception as e:
        print(f"[fix-meta 失败] {e}")
        sys.exit(1)
    print(f"[fix-meta] #{r['paper_id']} 已更新：{list(r['changed'])}")
    if r["norm_key_changed"]:
        print(f"  归一化主键已重算：{r['norm_key']}")
    for w in r["warnings"]:
        print(f"  ⚠ {w}")


def cmd_import_bib(args):
    """导入 BibTeX / RIS / Zotero CSV（从 Zotero、EndNote 搬家用）。"""
    from pathlib import Path as _P

    from papernest import bibimport
    text = _P(args.path).read_text(encoding="utf-8", errors="replace")
    r = bibimport.import_text(text, args.fmt, enrich=args.enrich,
                              source_name=_P(args.path).name)
    print(f"[import-bib] 格式 {r['format']} · 解析 {r['parsed']} 条 → "
          f"新增 {r['imported']} · 已存在跳过 {r['skipped']} · 失败 {r['failed']}")
    for e in r["errors"][:10]:
        print(f"  ⚠ {e}")
    if len(r["errors"]) > 10:
        print(f"  …… 另有 {len(r['errors']) - 10} 条，完整流水见 import_runs 表")


def cmd_graph(args):
    from papernest import graph
    if args.action == "fetch":
        ids = ([int(x) for x in args.ids.split(",") if x.strip()] if args.ids
               else [r["id"] for r in _all_paper_ids(args.limit)])
        for pid in ids:
            try:
                r = graph.fetch_edges(pid, args.direction, args.per_paper)
            except Exception as e:
                print(f"  ✗ #{pid}：{e}")
                continue
            print(f"  ✓ #{pid} 引用边 +{r.get('edges_added', 0)}"
                  f"（跳过 {r.get('skipped', '')}）" if r.get("skipped")
                  else f"  ✓ #{pid} 引用边 +{r.get('edges_added', 0)}")
        print(f"[graph] {graph.stats()}")
        return
    if args.action == "local-refs":
        ids = ([int(x) for x in args.ids.split(",") if x.strip()] if args.ids
               else [r["id"] for r in _papers_with_pdf(args.limit)])
        if not ids:
            print("[graph local-refs] 库里没有带本地 PDF 的论文（先 read 或 import-pdf）")
            sys.exit(1)
        total = 0
        for pid in ids:
            try:
                r = graph.ingest_local_references(pid)
            except Exception as e:
                print(f"  ✗ #{pid}：{e}")
                continue
            total += r["edges_added"]
            print(f"  ✓ #{pid} 参考文献 {r['n_entries']} 条 → 带标识 {r['with_id']} → "
                  f"新增边 {r['edges_added']}（无标识跳过 {r['skipped_no_id']}）")
        print(f"[graph] 共新增 {total} 条边 · {graph.stats()}")
        return
    if args.action == "gaps":
        rows = graph.gap_papers(args.top_k)
        print(f"[graph] 阅读缺口（被库内引用最多、但自己不在库里的 {len(rows)} 篇）：")
        for r in rows:
            print(f"  {r['cited_by_count']:>3} 次  {r['title'][:70]}  [{r['norm_key']}]")
            print(f"        引它的：{'；'.join(t[:40] for t in r['cited_by_titles'][:2])}")
        return
    if args.action == "related":
        ids = [int(x) for x in (args.ids or "").split(",") if x.strip()]
        if not ids:
            print("[graph related] 需要 --ids")
            sys.exit(1)
        for r in graph.related(ids, args.top_k):
            mark = "库内" if r["in_library"] else "库外"
            print(f"  {r['score']:.3f} [{mark}] {r['title'][:60]}")
            print(f"        {r['reason']}")
        return
    print(graph.stats())


def _all_paper_ids(limit: int):
    from papernest import db as _db
    with _db.conn() as c:
        return c.execute("SELECT id FROM papers ORDER BY citation_count DESC NULLS LAST, "
                         "id DESC LIMIT ?", (limit,)).fetchall()


def _papers_with_pdf(limit: int):
    from papernest import db as _db
    with _db.conn() as c:
        return c.execute("SELECT id FROM papers WHERE pdf_path IS NOT NULL "
                         "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()


def cmd_ppt(args):
    from papernest import pptgen
    ids = [int(x) for x in args.ids.split(",") if x.strip()]
    try:
        r = pptgen.deck(ids, args.topic)
    except ValueError as e:
        print(f"[ppt 失败] {e}")
        sys.exit(1)
    print(f"[ppt] 已生成 {r['slides']} 页（{r['papers']} 篇文献）：{r['path']}")


def cmd_qasper(args):
    from papernest import qasper
    if args.action == "download":
        r = qasper.download()
        print(r)
        return
    if args.action == "import":
        r = qasper.import_papers(args.n_papers)
        print(r)
        return
    if args.action == "ctxab":
        r = qasper.context_ab(max_papers=args.n_papers, max_questions=args.limit)
        print(f"上下文组装 A/B（{r['n_questions']} 道带 gold 证据的题 · 等预算 · 0 token）")
        print(f"{'预算':>8} | {'按页(删空白)':>12} {'按块(删空白)':>12} "
              f"| {'按页(留空白)':>12} {'按块(留空白)':>12}")
        for b, d in r["budgets"].items():
            print(f"{b:>8} | {d['page']['strip']:>12} {d['chunk']['strip']:>12} "
                  f"| {d['page']['ws']:>12} {d['chunk']['ws']:>12}")
        print("\n留空白口径衡量的是「上下文里的文字是否与原文逐字一致」——"
              "机械回取校验依赖它。")
        out = config.DATA_DIR / "qasper_ctxab.json"
        out.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"报告已存：{out}")
        return
    # eval
    r = qasper.run_eval(k=args.k, sec_k=args.sec_k, max_papers=args.n_papers,
                        max_questions=args.limit)
    print(f"QASPER 证据评测（{r['n_questions']} 题 · 论文级 {r['retrieval_mode']} 检索）")
    print(f"  ① paper_hit@{r['k']}          : {r['paper_hit_at_k']}   gold 论文进全库检索 top-{r['k']}")
    print(f"  ② evidence_recall@sec{r['sec_k']} : {r['evidence_recall_at_sec_k']}   gold 论文内 top-{r['sec_k']} 节命中证据片段")
    out = config.DATA_DIR / "qasper_eval.json"
    out.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  报告已存：{out}")


def cmd_scale(args):
    """规模实验：批量采集（--no-cards，0 LLM）→ 吞吐/去重/查询延迟报告。"""
    import time as _time
    queries = [q.strip() for q in args.queries.split(";") if q.strip()]
    print(f"[scale] 计划 {len(queries)} 组查询 × limit {args.limit}（--no-cards）")
    t0 = _time.perf_counter()
    for q in queries:
        tq = _time.perf_counter()
        try:
            r = ingest.ingest(q, args.limit, make_cards=False)
            print(f"  {q!r}: 检索 {r['results']} / 新 {r['new_papers']} / "
                  f"缓存命中 {r['cache_hits']} / LLM {r['llm_calls']} 次，"
                  f"耗时 {_time.perf_counter() - tq:.1f}s")
        except RuntimeError as e:
            print(f"  {q!r}: 失败——{str(e)[:120]}")
    ingest_s = _time.perf_counter() - t0

    db.init_db()
    with db.conn() as c:
        n = c.execute("SELECT COUNT(*) n FROM papers").fetchone()["n"]
        dup = c.execute("""SELECT COUNT(*) n FROM (
                             SELECT norm_key FROM papers GROUP BY norm_key
                             HAVING COUNT(*) > 1)""").fetchone()["n"]
        src = c.execute("SELECT source, COUNT(*) n FROM papers GROUP BY source").fetchall()
    lat = []
    for q in queries[:3]:
        for i in range(5):
            t = _time.perf_counter()
            with db.conn() as c:
                db.search_fts(c, q, 10)
            lat.append((_time.perf_counter() - t) * 1000)
    lat.sort()
    p50, p95 = lat[len(lat) // 2], lat[int(len(lat) * 0.95) - 1]
    print(f"\n[scale 报告] 总论文 {n} 篇 | norm_key 重复 {dup} | 采集总耗时 {ingest_s:.1f}s"
          f"（{n / max(ingest_s, 0.001):.1f} 篇/s）")
    print(f"  来源分布: {[(r['source'], r['n']) for r in src]}")
    print(f"  FTS 查询延迟: p50 {p50:.1f}ms / p95 {p95:.1f}ms（{len(lat)} 次，查询 {queries[:3]}）")


def cmd_recard(args):
    if not llm.available():
        print("未配置 LLM（.env 里填 LLM_API_BASE / LLM_API_KEY / LLM_MODEL），无法重刷。")
        sys.exit(1)
    n = cards.recard_all(args.topic, args.limit, force_all=args.all)
    print(f"已重刷 {n} 张卡片。")


def cmd_eval(args):
    from pathlib import Path as _P
    from papernest import eval as ev
    suffix = args.retrieval + (f"_{args.qa_mode}" if args.qa and args.qa_mode != "plain" else "")
    out = _P(args.out) if args.out else config.DATA_DIR / f"eval_report_{suffix}.json"
    report = ev.run_eval(k=args.k, run_qa=args.qa, limit=args.limit,
                         out_path=out, retrieval=args.retrieval,
                         qa_mode=args.qa_mode)
    ev.print_report(report)
    print(f"\n报告已存：{out}")


def cmd_demo(args):
    """离线演示：全程不连外网、不调 LLM——查询缓存 + FTS + 词面匹配兜底链路。"""
    from papernest import agent

    db.init_db()
    line = "─" * 62
    print(f"{line}\n PaperNest 离线演示（无外网 / 无 LLM 也能跑的链路）\n{line}")

    with db.conn() as c:
        n = c.execute("SELECT COUNT(*) n FROM papers").fetchone()["n"]
        q = c.execute("SELECT query FROM query_cache LIMIT 3").fetchall()
    print(f"\n[1] 本地库：{n} 篇论文（SQLite + 全文索引，零外网依赖）")
    if q:
        print(f"    查询缓存：{len(q)}+ 条命中记录——同查询重放连外网请求都不出")

    goal = "库里有哪些关于评测智能体的文献？"
    plan = agent.build_plan(goal)
    print(f"\n[2] Planner（确定性路由，模型不参与）：{goal!r}")
    for s in plan:
        print(f"    step{s.id}: {s.tool}  ——  {s.reason}")

    query = "agent evaluation"
    with db.conn() as c:
        rows = db.search_fts(c, query, 3)
    print(f"\n[3] FTS 检索（向量模型的离线兜底）：{query!r} 命中 {len(rows)} 篇")
    for r in rows:
        print(f"    [{r['id']}] {r['title'][:58]}")

    para = ("Large language model agents are increasingly deployed, "
            "yet systematic evaluation of their tool-use reliability remains challenging.")
    rec = cite.recommend(para, 3)
    print(f"\n[4] 引用推荐（词面重叠兜底模式）→ {len(rec['candidates'])} 条候选")
    for cd in rec["candidates"]:
        flag = "✓核验" if cd["verified"] else "×未核"
        print(f"    [{cd['paper_id']}] {flag} {cd['title'][:52]}")
    if rec.get("degraded"):
        print(f"    （降级说明：{rec['degraded'][:60]}）")

    ids = [cd["paper_id"] for cd in rec["candidates"][:1]]
    if ids:
        print(f"\n[5] 导出 RIS（可直接导入 EndNote/Zotero）：")
        print("\n".join("    " + l for l in cite.export(ids, "ris").splitlines()[:8]))

    print(f"\n{line}\n 演示结束。配好 key 后：eval / survey / L2 精读 / RAG 全链路激活。\n{line}")


def cmd_serve(args):
    import uvicorn
    host = getattr(args, "host", None) or "127.0.0.1"
    # 如实告诉应用绑在哪：它据此决定「无口令是否可以放行」。
    # 应用自己问不到 uvicorn 的绑定地址，只能由启动方声明。
    os.environ["PAPERNEST_BIND_HOST"] = host
    uvicorn.run(__import__("papernest.api", fromlist=["app"]).app,
                host=host, port=args.port, log_level="warning")


# ── 二期：多 Agent 写作流水线（CLI 同步跑各阶段任务，检查点人工确认）──

def _write_job(run_id: str, stage: str, extra: dict | None = None) -> dict:
    """同步执行一个流水线阶段任务（复用 jobs 基建，进度落库）。"""
    from papernest import jobs, pipeline
    job_id = jobs.create_job("write", {"run_id": run_id, "stage": stage} | (extra or {}))
    pipeline.attach_job(run_id, job_id)
    jobs.execute_job(job_id)
    job = jobs.get_job(job_id)
    if job["status"] != "done":
        print(f"[write 失败] {job.get('error')}")
        sys.exit(1)
    return job["result"]


def _print_topics(run):
    print(f"写作任务 {run['id']}（课题：{run['topic'][:40]}）")
    print(f"选题 Agent 产出 {len(run['topics'])} 个候选（检查点①：选定后生成大纲）：")
    for i, t in enumerate(run["topics"], 1):
        print(f"  [{i}] {t.get('title', '')}")
        print(f"      缺口：{t.get('gap', '')[:90]}")
        print(f"      切入：{t.get('angle', '')[:90]}  风险：{t.get('risks', '')[:60]}")
        for ev in t.get("evidence") or []:
            mark = "✓已核" if ev.get("verified") else "×待核"
            print(f"      {mark} 文献#{ev.get('paper_id')}：{str(ev.get('quote', ''))[:80]}")


def _print_outline(run_id):
    from papernest import pipeline
    o = pipeline.get_outline(run_id)
    if not o:
        print("（大纲还没生成）")
        return None
    print(f"大纲 v{o['version']}（{'人工改过' if o['edited'] else 'Agent 生成'}）：{o['title']}")
    for s in o["sections"]:
        print(f"  {s['no']}. {s['title']}（约 {s.get('words', 0)} 字｜"
              f"拟引文献 {s.get('paper_ids') or ('无·no_support' if s.get('no_support') else '待文献Agent核验')}）")
        for p in s.get("points") or []:
            print(f"     · {p[:80]}")
    if o.get("warnings"):
        print("  机械校验备注：")
        for w in o["warnings"]:
            print(f"   - {w}")
    return o


def cmd_write(args):
    from papernest import pipeline
    run = pipeline.create_run(args.topic, target_words=args.words,
                              max_rewrites=args.rewrites)
    r = _write_job(run["id"], "topic")
    if r.get("degraded"):
        print(f"（{r['degraded']}）")
    _print_topics(pipeline.get_run(run["id"]))
    print(f"\n下一步：python cli.py write-pick {run['id']} \"题目或候选序号对应题目\"")


def cmd_write_pick(args):
    from papernest import pipeline
    title = args.title
    if title.isdigit():  # 允许直接用候选序号
        run = pipeline.get_run(args.run_id)
        cands = run["topics"] if run else []
        idx = int(title) - 1
        if not (0 <= idx < len(cands)):
            print(f"[write-pick 失败] 候选序号 1-{len(cands)}")
            sys.exit(1)
        title = cands[idx]["title"]
    pipeline.select_topic(args.run_id, title)
    r = _write_job(args.run_id, "outline")
    if r.get("degraded"):
        print(f"（{r['degraded']}）")
    _print_outline(args.run_id)
    print(f"\n下一步：python cli.py write-go {args.run_id}")


def cmd_write_outline(args):
    _print_outline(args.run_id)


def cmd_write_go(args):
    from papernest import pipeline
    pipeline.save_outline(args.run_id, None)
    r = _write_job(args.run_id, "sections")
    reused = r.get("cache_reused", 0)
    print(f"逐节撰写完成：{r['sections']} 节（缓存复用 {reused} 节，失败 {len(r.get('failed') or [])} 节）")
    for no, score in (r.get("scores") or {}).items():
        print(f"  第 {no} 节 评审分 {score if score is not None else '—（离线不评）'}")
    print(f"下一步：python cli.py write-finish {args.run_id}")


def cmd_write_status(args):
    from papernest import pipeline
    state = pipeline.get_run_state(args.run_id)
    if not state:
        print("写作任务不存在")
        sys.exit(1)
    run, outline, sections = state["run"], state["outline"], state["sections"]
    print(f"run {run['id']}｜状态 {run['status']}｜阶段 {run.get('stage') or '—'}")
    print(f"  题目：{run.get('title') or '（未定题）'}")
    if run.get("error"):
        print(f"  ✗ {run['error']}")
    if outline:
        print(f"  大纲 v{outline['version']}（{len(outline['sections'])} 节）")
    for s in sections:
        chip = {"done": "✓", "reused": "♻", "failed": "✗", "pending": "·",
                "drafting": "…", "pending_user": "⏸"}.get(s["status"], s["status"])
        ncit = len(s.get("citations") or {})
        print(f"  {chip} 节{s['sec_no']} {s['title'][:30]}｜评审 {s['score'] if s['score'] is not None else '—'}"
              f"｜{s['attempt']} 稿｜引用 {ncit}｜{len(s.get('content') or '')} 字")


def cmd_write_finish(args):
    from papernest import pipeline
    _write_job(args.run_id, "polish")
    rep = pipeline.get_run(args.run_id).get("result") or {}
    rate = rep.get("verified_rate")
    print(f"终检报告：{rep.get('words', 0)} 字｜引用 {rep.get('citation_count', 0)} 条｜"
          f"核验率 {rate if rate is not None else '—'}｜"
          f"悬空 {len(rep.get('dangling') or [])}｜待补证据 {rep.get('pending_evidence', 0)}")
    for f in (rep.get("files") or {}).values():
        print(f"  已导出：{f}")
    if not rep.get("ok"):
        print("  ⚠ 终检未全过——悬空/未核验编号见 write-status 与报告详情")


def cmd_write_export(args):
    from papernest import pipeline
    run = pipeline.get_run(args.run_id)
    if not run:
        print("写作任务不存在")
        sys.exit(1)
    files = (run.get("result") or {}).get("files") or {}
    path = files.get(args.fmt)
    if not path or not Path(path).exists():
        print(f"没有 {args.fmt} 导出文件（先 write-finish）")
        sys.exit(1)
    if args.fmt == "docx":
        print(path)
    else:
        print(Path(path).read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser(prog="papernest")
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("verify")
    v.add_argument("--query", default="large language model agent evaluation")
    v.add_argument("--limit", type=int, default=20)
    v.set_defaults(fn=cmd_verify)

    i = sub.add_parser("ingest")
    i.add_argument("query")
    i.add_argument("--limit", type=int, default=50)
    i.add_argument("--topic", default=None)
    i.add_argument("--no-cards", action="store_true",
                   help="只入元数据（L0），不生成 L1 卡片——规模实验用，0 次 LLM")
    i.set_defaults(fn=cmd_ingest)

    sub.add_parser("stats").set_defaults(fn=cmd_stats)

    hp = sub.add_parser("health", help="检索健康探针（配置/索引/活体/检索四项）")
    hp.add_argument("--offline", action="store_true",
                    help="只跑配置与索引比对，0 次网络请求（CI 门禁用）")
    hp.set_defaults(fn=cmd_health)

    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("--limit", type=int, default=20)
    s.set_defaults(fn=cmd_search)

    sub.add_parser("models").set_defaults(fn=cmd_models)

    sub.add_parser("embed").set_defaults(fn=cmd_embed)

    ci = sub.add_parser("cite")
    ci.add_argument("text")
    ci.add_argument("--topk", type=int, default=5)
    ci.set_defaults(fn=cmd_cite)

    ex = sub.add_parser("export")
    ex.add_argument("ids", help="逗号分隔的 paper_id，如 1,2,5")
    ex.add_argument("--fmt", default="bibtex",
                    choices=["bibtex", "ris", "gbt7714", "ieee"])
    ex.set_defaults(fn=cmd_export)

    rd = sub.add_parser("read")
    rd.add_argument("paper_id", type=int)
    rd.add_argument("--topic", default=None)
    rd.set_defaults(fn=cmd_read)

    rc = sub.add_parser("recard")
    rc.add_argument("--limit", type=int, default=None)
    rc.add_argument("--topic", default=None)
    rc.add_argument("--all", action="store_true", help="重刷全部卡片（提示词变更后用）")
    rc.set_defaults(fn=cmd_recard)

    ev = sub.add_parser("eval")
    ev.add_argument("--k", type=int, default=5)
    ev.add_argument("--qa", action="store_true", help="跑无证据率（需 LLM key，会调真模型）")
    ev.add_argument("--retrieval", default="auto",
                    choices=["auto", "fts", "deep", "deep-rrf"],
                    help="auto=线上同款混合检索；fts=强制纯 FTS（消融）；"
                         "deep=迭代检索·派生查询补位（0 token）；"
                         "deep-rrf=派生查询 RRF 融合（消融：实测更差，复现负结果用）")
    ev.add_argument("--qa-mode", dest="qa_mode", default="plain",
                    choices=["plain", "rcs"],
                    help="无证据率的作答口径：plain=朴素 RAG；rcs=重排+定向摘要（对照用）")
    ev.add_argument("--limit", type=int, default=None, help="每类条目上限（冒烟用）")
    ev.add_argument("--out", default=None, help="报告输出路径（默认 data/eval_report.json）")
    ev.set_defaults(fn=cmd_eval)

    sub.add_parser("reindex", help="重建页级全文索引 pages_fts（排障/兜底用）"
                   ).set_defaults(fn=cmd_reindex)

    vc = sub.add_parser("vec", help="向量后端：状态 / 重建索引 / 给章节块补向量")
    vc.add_argument("action", choices=["status", "rebuild", "embed-chunks"])
    vc.add_argument("--limit", type=int, default=None, help="只处理前 N 块")
    vc.add_argument("--dry-run", action="store_true", help="只试算成本，不调接口")
    vc.set_defaults(fn=cmd_vec)

    sub.add_parser("rechunk", help="把已有全文重切成章节级检索单元（存量库跑一次）"
                   ).set_defaults(fn=cmd_rechunk)

    idc = sub.add_parser("import-doc",
                         help="多格式入库：Word/PPT/HTML/Markdown/纯文本（PDF 用 import-pdf）")
    idc.add_argument("paths", nargs="+", help="文件或目录（目录递归找支持的格式）")
    idc.add_argument("--topic", default=None)
    idc.add_argument("--no-cards", action="store_true", help="只入正文，不生成卡片（0 token）")
    idc.set_defaults(fn=cmd_import_doc)

    st = sub.add_parser("sectiontree", help="章节树索引与两级检索（先选范围、再检索）")
    st.add_argument("action", choices=["build", "stats", "search"])
    st.add_argument("query", nargs="?", default="")
    st.add_argument("--top-k", dest="top_k", type=int, default=5)
    st.set_defaults(fn=cmd_sectiontree)

    ob = sub.add_parser("export-obsidian",
                        help="导出 Obsidian 文献笔记（citekey 命名 + 双链）")
    ob.add_argument("out", help="目标目录（建议 vault 里的独立子目录，同名文件会被覆盖）")
    ob.add_argument("--ids", default=None, help="逗号分隔 paper_id；不填导全库")
    ob.set_defaults(fn=cmd_export_obsidian)

    ak = sub.add_parser("ask", help="命令行问答（--deep 走迭代自反馈闭环）")
    ak.add_argument("question")
    ak.add_argument("--top-k", dest="top_k", type=int, default=5)
    ak.add_argument("--deep", action="store_true",
                    help="迭代问答：作答→机械找无证据论断→补检索→重答（多次模型调用）")
    ak.add_argument("--rounds", type=int, default=2,
                    help="迭代上限（仅 --deep，取值 1-5，超出会被夹紧）")
    ak.add_argument("--rcs", action="store_true",
                    help="RCS 增强：宽检索→LLM 重排→逐篇定向摘要（带依据句机械校验）→作答")
    ak.set_defaults(fn=cmd_ask)

    dg = sub.add_parser("digest", help="新论文摘报：按库内画像给 arXiv 新文打分")
    dg.add_argument("--days", type=int, default=3, help="回看几天的新提交")
    dg.add_argument("--top-k", dest="top_k", type=int, default=10)
    dg.add_argument("--categories", default=None,
                    help="逗号分隔的 arXiv 分类（不填则从画像猜，猜不出会如实报告）")
    dg.add_argument("--repeat-after", dest="repeat_after", type=int, default=30,
                    help="多少天内推过的不再重复推（默认 30）")
    dg.add_argument("--llm", action="store_true", help="对前若干条加一层模型重排")
    dg.add_argument("--profile", action="store_true", help="只看兴趣画像，不拉新论文")
    dg.set_defaults(fn=cmd_digest)

    mx = sub.add_parser("matrix", help="跨论文结构化对比矩阵（可导出 csv/markdown/latex）")
    mx.add_argument("ids", help="逗号分隔的 paper_id")
    mx.add_argument("--fmt", default="markdown", choices=["markdown", "csv", "latex"])
    mx.add_argument("--llm", dest="llm", action="store_true", default=None,
                    help="强制走 LLM 增强抽取（默认有 key 就用、没有就离线）")
    mx.add_argument("--no-llm", dest="llm", action="store_false",
                    help="强制离线抽取（0 token）")
    mx.set_defaults(fn=cmd_matrix)

    st = sub.add_parser("structure", help="看某篇本地 PDF 的章节结构与参考文献")
    st.add_argument("paper_id", type=int)
    st.set_defaults(fn=cmd_structure)

    ip = sub.add_parser("import-pdf", help="本地 PDF 入库（可传目录，按 sha256 与 norm_key 去重）")
    ip.add_argument("paths", nargs="+", help="PDF 文件或目录，可多个")
    ip.add_argument("--topic", default=None, help="课题上下文（写进卡片的「与我课题的关系」）")
    ip.add_argument("--no-cards", action="store_true", help="只入库不生成卡片（0 次 LLM）")
    ip.set_defaults(fn=cmd_import_pdf)

    fm = sub.add_parser("fix-meta", help="修正抽错的元数据（改 title/doi 会重算 norm_key）")
    fm.add_argument("paper_id", type=int)
    fm.add_argument("--title", default=None)
    fm.add_argument("--authors", default=None, help="分号分隔，如 \"Alice Smith;Bob Lee\"")
    fm.add_argument("--year", type=int, default=None)
    fm.add_argument("--venue", default=None)
    fm.add_argument("--doi", default=None)
    fm.add_argument("--arxiv-id", dest="arxiv_id", default=None)
    fm.set_defaults(fn=cmd_fix_meta)

    ib = sub.add_parser("import-bib", help="导入 BibTeX / RIS / Zotero CSV")
    ib.add_argument("path", help=".bib / .ris / .csv 文件路径")
    ib.add_argument("--fmt", default="auto", choices=["auto", "bibtex", "ris", "csv"])
    ib.add_argument("--enrich", action="store_true",
                    help="用 Semantic Scholar 补缺失摘要（联网、受限流影响，条目多时会慢）")
    ib.set_defaults(fn=cmd_import_bib)

    gr = sub.add_parser("graph", help="引文网络：拉边 / 阅读缺口 / 相关推荐")
    gr.add_argument("action",
                    choices=["fetch", "local-refs", "gaps", "related", "stats"])
    gr.add_argument("--ids", default=None, help="逗号分隔 paper_id（fetch/related）")
    gr.add_argument("--limit", type=int, default=20, help="fetch 不指定 ids 时取被引最高的前 N 篇")
    gr.add_argument("--per-paper", dest="per_paper", type=int, default=100,
                    help="每篇最多拉多少条边")
    gr.add_argument("--direction", default="both",
                    choices=["both", "references", "citations"])
    gr.add_argument("--top-k", dest="top_k", type=int, default=20)
    gr.set_defaults(fn=cmd_graph)

    pt = sub.add_parser("ppt")
    pt.add_argument("ids", help="逗号分隔的 paper_id，如 1,2,3（单篇=论文汇报，多篇=文献汇报）")
    pt.add_argument("--topic", default="", help="汇报课题（默认取 .env 的 RESEARCH_TOPIC）")
    pt.set_defaults(fn=cmd_ppt)

    sub.add_parser("demo").set_defaults(fn=cmd_demo)

    qp = sub.add_parser("qasper")
    qp.add_argument("action", choices=["download", "import", "eval", "ctxab"])
    qp.add_argument("--n-papers", type=int, default=30)
    qp.add_argument("--k", type=int, default=5)
    qp.add_argument("--sec-k", type=int, default=2)
    qp.add_argument("--limit", type=int, default=None, help="评测题数上限（冒烟用）")
    qp.set_defaults(fn=cmd_qasper)

    sc = sub.add_parser("scale")
    sc.add_argument("queries", help="分号分隔的多组查询，如 \"llm agents;retrieval augmented generation\"")
    sc.add_argument("--limit", type=int, default=100)
    sc.set_defaults(fn=cmd_scale)

    wr = sub.add_parser("write")
    wr.add_argument("topic", help="课题描述（给选题 Agent 的输入）")
    wr.add_argument("--words", type=int, default=3000, help="目标总字数")
    wr.add_argument("--rewrites", type=int, default=2, help="每节评审不达标重写上限")
    wr.set_defaults(fn=cmd_write)

    wp = sub.add_parser("write-pick")
    wp.add_argument("run_id")
    wp.add_argument("title", help="题目全文，或候选序号（如 2）")
    wp.set_defaults(fn=cmd_write_pick)

    wo = sub.add_parser("write-outline")
    wo.add_argument("run_id")
    wo.set_defaults(fn=cmd_write_outline)

    wg = sub.add_parser("write-go")
    wg.add_argument("run_id")
    wg.set_defaults(fn=cmd_write_go)

    ws = sub.add_parser("write-status")
    ws.add_argument("run_id")
    ws.set_defaults(fn=cmd_write_status)

    wf = sub.add_parser("write-finish")
    wf.add_argument("run_id")
    wf.set_defaults(fn=cmd_write_finish)

    we = sub.add_parser("write-export")
    we.add_argument("run_id")
    we.add_argument("--fmt", default="md", choices=["md", "docx"])
    we.set_defaults(fn=cmd_write_export)

    se = sub.add_parser("serve")
    se.add_argument("--port", type=int, default=8765)
    # 默认只绑回环。要对外提供服务必须显式指定 --host，而那条路径上
    # 应用会强制要求 PAPERNEST_API_KEY（见 api._check_exposure）。
    se.add_argument("--host", default="127.0.0.1",
                    help="绑定地址（默认 127.0.0.1 仅本机；对外需同时设 PAPERNEST_API_KEY）")
    se.set_defaults(fn=cmd_serve)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
