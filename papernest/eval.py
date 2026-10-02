"""评测三指标：Recall@K / 引用准确率 / 无证据率。

评测集在项目根 eval_set.json，gold 标到 norm_key（DOI/arXiv 归一化主键），
不随重新采集换 id。三个指标全部机械可算，不靠人眼：
- Recall@K        ：检索 top-K 命中 gold 的比例（检索链路与线上同一套：向量优先 / FTS 兜底）
- 引用准确率       ：推荐候选里证据句通过机械回取校验的比例 + 对 gold 的 precision@K
- 无证据率        ：回答的论断句里没有 [n] 引用支撑的比例（越低越好）
QA 指标要调真模型（--qa 才跑），检索与引用指标离线可算。
"""
import json
import re
import time
from pathlib import Path

from . import db, intent as intent_mod, tools

# 意图评测集路径——与 eval_set.json 分开，因为 gold 字段（gold_intent vs gold_keys）
# 不同；放一起会让 load_eval_set 的断言更难维护。两条入口共用同一个 load_eval_set
# 机制，但 type=intent 的条目被 metric_intent_routing 单挑出来。
INTENT_EVAL_SET_PATH = Path(__file__).resolve().parent.parent / "intent_eval_set.json"

# 意图分类的 5 个合法值。metric_intent_routing 用这个做混淆矩阵的轴。
INTENT_LABELS = ("cite", "survey", "read", "qa", "unknown")

EVAL_SET_PATH = Path(__file__).resolve().parent.parent / "eval_set.json"

# 论断句判定：这些是「免责/元话术」，不算需要引用支撑的论断
_META_PAT = re.compile(
    r"未覆盖|通识解释|建议检索|抱歉|无法回答|基于你的问题|根据上下文|以下是|为您"
    r"|当前上下文|上下文未|库里没有|库内没有|无法列出|检索关键词")
_CITE_PAT = re.compile(r"\[\d{1,2}\]")


def load_eval_set(path: Path | None = None) -> list[dict]:
    p = Path(path) if path else EVAL_SET_PATH
    entries = json.loads(p.read_text(encoding="utf-8"))
    for e in entries:
        assert e.get("type") in ("retrieval", "citation", "qa"), f"坏条目：{e.get('id')}"
        assert e.get("gold_keys"), f"{e.get('id')} 缺 gold_keys"
    return entries


def load_intent_eval_set(path: Path | None = None) -> list[dict]:
    """加载意图评测集。每条必须有 gold_intent 字段（5 类之一）。

    与 eval_set.json 的关键差异：
    - 没有 gold_keys（意图评测与具体论文无关）
    - type 字段固定为 "intent"
    - gold_intent 必须 ∈ INTENT_LABELS
    """
    p = Path(path) if path else INTENT_EVAL_SET_PATH
    entries = json.loads(p.read_text(encoding="utf-8"))
    for e in entries:
        assert e.get("type") == "intent", f"意图评测集只接受 type=intent，{e.get('id')} 是 {e.get('type')}"
        gold = e.get("gold_intent")
        assert gold in INTENT_LABELS, f"{e.get('id')} gold_intent={gold!r} 不在 {INTENT_LABELS}"
        assert e.get("query"), f"{e.get('id')} 缺 query"
    return entries


def default_retrieve(query: str, top_k: int) -> tuple[list[int], str | None]:
    """与线上同一套检索链路（tools.retrieve_library：RRF 混合 / FTS 兜底）。"""
    r = tools.retrieve_library(query, top_k)
    return r["paper_ids"], r.get("retrieval_mode") or ("fts" if r.get("degraded") else None)


def fts_only_retrieve(query: str, top_k: int) -> tuple[list[int], str | None]:
    """消融用：强制纯 FTS（与线上共用同一 search_fts 容错实现）。"""
    from . import db
    with db.conn() as c:
        rows = db.search_fts(c, query, top_k)
    return [r["id"] for r in rows], "fts"


def deep_retrieve(query: str, top_k: int) -> tuple[list[int], str | None]:
    """迭代检索口径：单轮召回 + 伪相关反馈派生查询「补位」（0 token）。

    与 auto/fts 口径跑同一套评测集，差值就是迭代检索的净贡献。
    """
    from . import deepsearch
    ids, mode, _trace = deepsearch.deep_retrieve(query, top_k)
    return ids, mode


def deep_rrf_retrieve(query: str, top_k: int) -> tuple[list[int], str | None]:
    """消融口径：派生查询用 RRF 融进主排序。留着是为了让那个负结果可复现——
    实测每个 k 都不如单轮（query drift），所以线上不用它。"""
    from . import deepsearch
    ids, mode, _trace = deepsearch.deep_retrieve(query, top_k, fusion="rrf")
    return ids, mode


RETRIEVERS = {"auto": default_retrieve, "fts": fts_only_retrieve,
              "deep": deep_retrieve, "deep-rrf": deep_rrf_retrieve}


def metric_recall(entries: list[dict], k: int = 5,
                  retrieve=default_retrieve) -> dict:
    """Recall@K：|topK ∩ gold| / |gold|，对条目取宏平均。"""
    items, modes = [], {}
    for e in entries:
        ids, mode = retrieve(e["query"], k)
        modes[mode or "unknown"] = modes.get(mode or "unknown", 0) + 1
        gold = set(e["gold_keys"])
        hit = _keys_of(ids) & gold
        items.append({"id": e["id"], "query": e["query"], "recall": round(len(hit) / len(gold), 3),
                      "hit": sorted(hit), "miss": sorted(gold - hit)})
    mean = round(sum(i["recall"] for i in items) / max(len(items), 1), 4)
    mode_str = "+".join(sorted(modes)) or "unknown"
    return {"recall_at_k": mean, "k": k, "n": len(items), "retrieval_mode": mode_str,
            "items": items}


def metric_citation(entries: list[dict], top_k: int = 5) -> dict:
    """引用准确率：对带引用需求的段落跑 cite.recommend。
    - evidence_verified_rate：top-K 候选中证据句通过机械回取校验的比例（防引用幻觉的核心）
    - precision_at_k        ：top-K 中命中 gold 的比例（推荐得准不准）
    """
    from . import cite
    items = []
    for e in entries:
        r = cite.recommend(e["query"], top_k)
        cands = r["candidates"]
        gold = set(e["gold_keys"])
        top_ids = {c["paper_id"] for c in cands[:top_k]}
        gold_ids = _ids_of_keys(gold)
        verified_n = sum(1 for c in cands[:top_k] if c["verified"])
        items.append({
            "id": e["id"], "query": e["query"][:60],
            "precision": round(len(top_ids & gold_ids) / max(len(cands[:top_k]), 1), 3),
            "verified_rate": round(verified_n / max(len(cands[:top_k]), 1), 3),
            "n_strong": sum(1 for c in cands[:top_k] if c["strong"]),
        })
    n = max(len(items), 1)
    return {
        "evidence_verified_rate": round(sum(i["verified_rate"] for i in items) / n, 4),
        "precision_at_k": round(sum(i["precision"] for i in items) / n, 4),
        "top_k": top_k, "n": len(items), "items": items,
    }


def metric_intent_routing(entries: list[dict],
                        classify=intent_mod.classify) -> dict:
    """意图分类准确率 + 每类 precision/recall + 5×5 混淆矩阵。

    比 metric_recall 那种「top-K 命中 gold」式评测多一层：
    - accuracy  : 整体正确率
    - per_intent: 每个类的 P/R/F1
    - confusion : 5×5 矩阵，shape=[gold, predicted]，便于一眼看出
                  cite/survey 之间有没有互相吃（build_plan 注释里讲的
                  「Survey 标题被读成用户意图」就是这类问题）

    classify 参数化：测试可以注入假分类器；生产用 intent_mod.classify。
    """
    items = []
    confusion = {g: {p: 0 for p in INTENT_LABELS} for g in INTENT_LABELS}
    for e in entries:
        got = classify(e["query"])
        # classify 可返回 Intent（属性访问）/ dict（key 访问）/ str（旧 stub）
        if hasattr(got, "primary"):
            predicted = got.primary
            signals = getattr(got, "signals", [])
            confidence = getattr(got, "confidence", None)
        elif isinstance(got, dict):
            predicted = got.get("primary", "unknown")
            signals = got.get("signals", [])
            confidence = got.get("confidence")
        else:
            predicted = str(got)
            signals = []
            confidence = None
        gold = e["gold_intent"]
        ok = (predicted == gold)
        confusion[gold][predicted] = confusion[gold].get(predicted, 0) + 1
        items.append({
            "id": e["id"],
            "query": e["query"][:60],
            "gold": gold,
            "predicted": predicted,
            "ok": ok,
            "signals": signals,
            "confidence": confidence,
            "category": e.get("category", ""),
        })

    n = max(len(items), 1)
    accuracy = round(sum(i["ok"] for i in items) / n, 4)

    # per_intent P/R/F1
    per_intent = {}
    for label in INTENT_LABELS:
        # gold 是 label 的条目
        gold_n = sum(1 for i in items if i["gold"] == label)
        # predicted 是 label 的条目
        pred_n = sum(1 for i in items if i["predicted"] == label)
        # 命中
        tp = sum(1 for i in items if i["gold"] == label and i["predicted"] == label)
        precision = round(tp / pred_n, 4) if pred_n else 0.0
        recall = round(tp / gold_n, 4) if gold_n else 0.0
        f1 = round(2 * precision * recall / (precision + recall), 4)             if (precision + recall) else 0.0
        per_intent[label] = {
            "gold_n": gold_n, "pred_n": pred_n, "tp": tp,
            "precision": precision, "recall": recall, "f1": f1,
        }

    worst = sorted([i for i in items if not i["ok"]],
                   key=lambda i: i.get("confidence") or 0)[:5]

    return {
        "accuracy": accuracy,
        "n": len(items),
        "per_intent": per_intent,
        "confusion": confusion,
        "worst_items": worst,
        "items": items,
    }


def split_claims(answer: str) -> tuple[list[str], list[str]]:
    """把回答切成 (有引用支撑的论断句, 无引用支撑的论断句)。免责/元话术不计。"""
    text = re.sub(r"。(\[\d{1,2}\])", r"\1。", answer or "")  # 角标贴回它支撑的前句
    supported, unsupported = [], []
    for s in re.split(r"(?<=[。！？!?])\s*|\n+", text):
        s = s.strip().lstrip("-*•· ").strip().replace("**", "")
        if len(s) < 15 or _META_PAT.search(s):
            continue
        (supported if _CITE_PAT.search(s) else unsupported).append(s)
    return supported, unsupported


def metric_no_evidence(entries: list[dict], top_k: int = 5,
                       mode: str = "plain") -> dict:
    """无证据率：调真模型回答 QA 条目，论断句无 [n] 支撑的比例（越低越好）。
    顺带算 cited_gold_rate：gold 文献被实际引用进回答的比例。

    mode="rcs" 走 RCS 增强路径（重排 + 逐篇定向摘要），与 plain 跑同一批题，
    差值即 RCS 的净贡献。
    """
    from . import llm, rag, rcs
    if not llm.available():
        return {"skipped": "未配置 LLM，无证据率需要真模型回答（配 key 后 --qa 重跑）"}
    items = []
    for e in entries:
        if mode == "rcs":
            r = rcs.answer(e["query"], top_k=top_k)
        else:
            r = rag.answer([{"role": "user", "content": e["query"]}], top_k=top_k)
        supported, unsupported = split_claims(r.get("answer", ""))
        cited_gold = _cited_gold(r.get("sources") or [], set(e["gold_keys"]))
        item = {
            "id": e["id"], "query": e["query"][:60],
            "n_claims": len(supported) + len(unsupported),
            "n_unsupported": len(unsupported),
            "cited_gold": cited_gold,
            "answer_head": (r.get("answer") or "")[:80],
        }
        if mode == "rcs":
            st = (r.get("trace") or {}).get("summary") or {}
            item["quotes_verified"] = st.get("quotes_verified", 0)
            item["quotes_total"] = st.get("quotes_total", 0)
            item["filtered_out"] = (r.get("trace") or {}).get("filtered_out", 0)
            item["not_selected"] = (r.get("trace") or {}).get("not_selected_by_rerank", 0)
        items.append(item)
    total = sum(i["n_claims"] for i in items)
    unsup = sum(i["n_unsupported"] for i in items)
    gold_n = sum(len(e["gold_keys"]) for e in entries)
    gold_hit = sum(i["cited_gold"] for i in items)
    out = {
        "mode": mode,
        "no_evidence_rate": round(unsup / max(total, 1), 4),
        "cited_gold_rate": round(gold_hit / max(gold_n, 1), 4),
        "n": len(items), "n_claims": total, "n_unsupported": unsup, "items": items,
    }
    if mode == "rcs":
        qt = sum(i.get("quotes_total", 0) for i in items)
        qv = sum(i.get("quotes_verified", 0) for i in items)
        out["quote_verified_rate"] = round(qv / qt, 4) if qt else None
        out["quotes_total"] = qt
        # 进了 top_k 又被摘要判为不相关而扔掉的（危险信号，值得盯）
        out["filtered_out"] = sum(i.get("filtered_out", 0) for i in items)
        # 候选池里重排没选中的（正常，池子本来就比 top_k 大）
        out["not_selected"] = sum(i.get("not_selected", 0) for i in items)
    return out


def _cited_gold(sources: list[dict], gold_keys: set[str]) -> int:
    keys = set()
    with db.conn() as c:
        for s in sources:
            row = c.execute("SELECT norm_key FROM papers WHERE id=?", (s.get("paper_id"),)).fetchone()
            if row and row["norm_key"] in gold_keys:
                keys.add(row["norm_key"])
    return len(keys)


def _keys_of(paper_ids: list[int]) -> set[str]:
    if not paper_ids:
        return set()
    with db.conn() as c:
        q = ",".join("?" * len(paper_ids))
        rows = c.execute(f"SELECT norm_key FROM papers WHERE id IN ({q})", paper_ids).fetchall()
    return {r["norm_key"] for r in rows}


def _ids_of_keys(keys: set[str]) -> set[int]:
    if not keys:
        return set()
    with db.conn() as c:
        q = ",".join("?" * len(keys))
        rows = c.execute(f"SELECT id FROM papers WHERE norm_key IN ({q})", list(keys)).fetchall()
    return {r["id"] for r in rows}


def run_eval(k: int = 5, run_qa: bool = False, limit: int | None = None,
             out_path: Path | None = None, retrieval: str = "auto",
             qa_mode: str = "plain") -> dict:
    entries = load_eval_set()
    if limit:
        by_type: dict[str, list] = {}
        for e in entries:
            by_type.setdefault(e["type"], []).append(e)
        entries = [e for lst in by_type.values() for e in lst[:limit]]
    retrieve = RETRIEVERS.get(retrieval, default_retrieve)
    # 两类查询形态**分开报**，不合成一个数：
    #   keyword —— 空格分隔的关键词串（评测集最初的全部形态）
    #   natural —— 同一批 gold，query 改写成用户真会打的自然语言提问（带问号/疑问词）
    # 合成一个平均值会让「口径变了」被一个没变的指标名盖住；而两者的差距本身
    # 就是一个要盯的指标——它衡量的是「查询形态惩罚」有多大。
    retrieval_entries = [e for e in entries if e["type"] == "retrieval"]
    keyword = [e for e in retrieval_entries if e.get("form", "keyword") == "keyword"]
    natural = [e for e in retrieval_entries if e.get("form") == "natural"]
    report = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "k": k,
        "retrieval": retrieval,
        "library": _library_snapshot(),
        # 保持历史可比：这一项的口径与加入 natural 形态之前完全一致
        "retrieval_metric": metric_recall(keyword, k, retrieve),
        "citation": metric_citation([e for e in entries if e["type"] == "citation"], k),
    }
    # 意图评测独立加载——不与 retrieval/citation/qa 共用一套数据，避免某条 gold
    # 缺失把整个 run 拖红。找不到文件就跳过（评测集是可选的）。
    intent_path = Path(__file__).resolve().parent.parent / "intent_eval_set.json"
    if intent_path.exists():
        intent_entries = load_intent_eval_set(intent_path)
        report["intent_routing"] = metric_intent_routing(intent_entries)
    else:
        report["intent_routing"] = {"skipped": "intent_eval_set.json 不存在"}
    if natural:
        report["retrieval_metric_natural"] = metric_recall(natural, k, retrieve)
        report["form_gap"] = round(
            report["retrieval_metric"]["recall_at_k"]
            - report["retrieval_metric_natural"]["recall_at_k"], 4)
    if run_qa:
        report["qa"] = metric_no_evidence([e for e in entries if e["type"] == "qa"],
                                          k, mode=qa_mode)
    else:
        report["qa"] = {"skipped": "未跑（--qa 开启；需 LLM key）"}
    if out_path:
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _library_snapshot() -> dict:
    with db.conn() as c:
        n = c.execute("SELECT COUNT(*) n FROM papers").fetchone()["n"]
        vectors = c.execute("SELECT COUNT(DISTINCT paper_id) n FROM vectors WHERE kind='paper'").fetchone()["n"]
    return {"papers": n, "papers_with_vectors": vectors}


def print_report(r: dict) -> None:
    print(f"PaperNest 评测报告  {r['ts']}  （库内 {r['library']['papers']} 篇，"
          f"{r['library']['papers_with_vectors']} 篇有向量 · 检索口径 {r.get('retrieval', 'auto')}）")
    ret = r["retrieval_metric"]
    print(f"\n① Recall@{ret['k']}          : {ret['recall_at_k']}"
          f"   （关键词串形态 {ret['n']} 条 · 检索链路 {ret['retrieval_mode']}）")
    nat = r.get("retrieval_metric_natural")
    if nat:
        # 两个形态的差距要摆在一起看：它衡量的是「用户按自然语言提问要付多少代价」。
        # 差距大 = 查询预处理有问题，而不是检索算法有问题。
        print(f"   Recall@{nat['k']} (自然语言): {nat['recall_at_k']}"
              f"   （{nat['n']} 条 · 同一批 gold，只换查询形态）"
              f"   形态差距 {r.get('form_gap')}")
    worst = sorted(ret["items"], key=lambda i: i["recall"])[:3]
    for i in worst:
        print(f"    最差 {i['id']} recall={i['recall']}  miss={len(i['miss'])}  {i['query'][:40]}")
    if nat:
        for i in sorted(nat["items"], key=lambda x: x["recall"])[:3]:
            print(f"    最差(自然语言) {i['id']} recall={i['recall']}  {i['query'][:40]}")
    cit = r["citation"]
    print(f"\n② 引用准确率        : 证据核验率 {cit['evidence_verified_rate']}"
          f" | precision@{cit['top_k']} {cit['precision_at_k']}   （{cit['n']} 条）")
    qa = r["qa"]
    if qa.get("skipped"):
        print(f"\n③ 无证据率          : 跳过——{qa['skipped']}")
    else:
        print(f"\n③ 无证据率          : {qa['no_evidence_rate']}"
              f"（{qa['n_unsupported']}/{qa['n_claims']} 条论断无引用）"
              f" | gold 引用覆盖 {qa['cited_gold_rate']}"
              f" | 口径 {qa.get('mode', 'plain')}")
        if qa.get("mode") == "rcs":
            print(f"    RCS：依据句机械回取 {qa.get('quote_verified_rate')}"
                  f"（{qa.get('quotes_total', 0)} 句）"
                  f" | 入选后被摘要判为不相关而剔除 {qa.get('filtered_out', 0)} 篇"
                  f"（重排未选中 {qa.get('not_selected', 0)} 篇属正常落选）")
    intent = r.get("intent_routing", {})
    if intent.get("skipped"):
        print(f"\n④ 意图路由          : 跳过——{intent['skipped']}")
    else:
        print(f"\n④ 意图路由          : accuracy {intent['accuracy']}（{intent['n']} 条）")
        for label in ("cite", "survey", "read", "qa", "unknown"):
            pi = intent["per_intent"].get(label, {})
            if pi.get("gold_n", 0) == 0 and pi.get("pred_n", 0) == 0:
                continue
            print(f"    {label:8s} P={pi['precision']:.2f} R={pi['recall']:.2f} F1={pi['f1']:.2f}"
                  f"  (gold={pi['gold_n']} pred={pi['pred_n']} tp={pi['tp']})")
        for w in intent.get("worst_items", [])[:3]:
            print(f"    ✗ {w['id']} gold={w['gold']} pred={w['predicted']} {w['query']!r}")
