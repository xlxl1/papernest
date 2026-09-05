"""QASPER 公开基准接入：把「自建评测」升级为「他证」。

QASPER（Das et al., AI2 2021）：5,049 个问题、1,585 篇 NLP 论文，每题带 gold
证据片段（evidence-bearing QA）——与 PaperNest 的证据链定位天然对齐。

三步：
1. download：拉取 QASPER dev（S3，落盘 data/qasper/，一次即可）
2. import_papers：论文全文按「节」导入 papers+pages（norm_key=qasper:<id>，零 LLM 零 PDF）
3. eval：两层指标，全部机械可算
   - paper_hit@k   ：问题在全库（含 QASPER 篇互为干扰项）检索中，gold 论文进 top-k 的比例
   - evidence_recall@k ：gold 论文内按相关性排节，top-k 节里含 gold 证据片段的比例
   （配 LLM 时可再加无证据率；未配时以上指标离线可跑）

用法：
  python cli.py qasper download
  python cli.py qasper import --n-papers 30
  python cli.py qasper eval --k 5
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from . import config, db, http

QASPER_DIR = config.DATA_DIR / "qasper"
# AI2 官方 S3（QASPER 论文与 qasper-led-baseline 仓库给的就是这个地址）。
# 原来配的三个源（GitHub raw / ai2-public-datasets / hf-mirror）现在全部 404——
# 数据集换了发布位置，「已接入公开基准」的说法当时已经名存实亡了。
QASPER_URLS = [
    "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz",
]
MAX_PAPERS = 200  # 导入上限：个人库场景的合理规模


def dev_path() -> Path:
    return QASPER_DIR / "qasper-dev-v1.json"


def _extract_dev(buf: bytes, out: Path) -> bool:
    """从下载到的载荷里刨出 dev 集 JSON。支持三种载荷：裸 JSON / zip / tar.gz。

    官方 tgz 里同时有 train 和 dev，只要 dev（train 十几万行，个人库场景用不上）。
    """
    import io
    import tarfile
    import zipfile

    if buf[:1] == b"{":
        out.write_bytes(buf)
        return True
    if buf[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(buf)) as z:
            name = next((n for n in z.namelist()
                         if n.endswith(".json") and "dev" in n.lower()), None)
            name = name or next(n for n in z.namelist() if n.endswith(".json"))
            out.write_bytes(z.read(name))
        return True
    if buf[:2] == b"\x1f\x8b":                       # gzip 魔数
        with tarfile.open(fileobj=io.BytesIO(buf), mode="r:gz") as t:
            names = [n for n in t.getnames() if n.endswith(".json")]
            name = next((n for n in names if "dev" in n.lower()), None) or (names[0] if names else None)
            if not name:
                return False
            fh = t.extractfile(name)
            if fh is None:
                return False
            out.write_bytes(fh.read())
        return True
    return False


def download() -> dict:
    """下载 QASPER dev。已存在则跳过；载荷可以是裸 JSON / zip / tar.gz。"""
    QASPER_DIR.mkdir(parents=True, exist_ok=True)
    out = dev_path()
    if out.exists():
        data = json.loads(out.read_text(encoding="utf-8"))
        return {"papers_available": len(data), "path": str(out), "cached": True}
    t0 = time.perf_counter()
    errors = []
    for url in QASPER_URLS:
        buf = bytearray()
        try:
            with http.client(timeout=180) as client:
                with client.stream("GET", url) as r:
                    if r.status_code != 200:
                        errors.append(f"{url.split('/')[2]}: HTTP {r.status_code}")
                        continue
                    for chunk in r.iter_bytes(1 << 16):
                        buf.extend(chunk)
                        if len(buf) > 300 * 1024 * 1024:
                            return {"error": "超过 300MB 上限，异常文件已放弃"}
        except Exception as e:
            errors.append(f"{url.split('/')[2]}: {type(e).__name__}")
            continue
        try:
            if not _extract_dev(bytes(buf), out):
                errors.append(f"{url.split('/')[2]}: 包里没有找到 JSON")
                continue
        except Exception as e:
            errors.append(f"{url.split('/')[2]}: 载荷解不开（{type(e).__name__}，可能是拦截页）")
            continue
        data = json.loads(out.read_text(encoding="utf-8"))
        return {"papers_available": len(data), "path": str(out),
                "source": url.split("/")[2],
                "download_s": round(time.perf_counter() - t0, 1)}
    return {"error": "全部下载源不可用（" + "；".join(errors)[:200] + "）——检查网络/代理后重试"}


def _sections_of(paper: dict) -> list[tuple[str, str]]:
    """论文全文 → [(节名, 正文)]。

    v0.3 的 full_text 是 [{"section_name":..., "paragraphs":[...]}]；
    更早的格式是 {"section_names":[...], "sections":[...]}。两种都认，认不出返回空
    （宁可这篇没有全文，也不要把结构猜错塞进库里）。
    """
    full = paper.get("full_text")
    out: list[tuple[str, str]] = []
    if isinstance(full, list):
        for sec in full:
            if not isinstance(sec, dict):
                continue
            paras = sec.get("paragraphs") or []
            body = "\n".join(p for p in paras if isinstance(p, str) and p.strip()).strip()
            if body:
                out.append(((sec.get("section_name") or "").strip() or "Section", body))
    elif isinstance(full, dict):
        for name, text in zip(full.get("section_names") or [], full.get("sections") or []):
            body = (text or "").strip() if isinstance(text, str) else ""
            if body:
                out.append(((name or "").strip() or "Section", body))
    return out


def import_papers(max_papers: int = 30) -> dict:
    """导入 QASPER 论文：norm_key=qasper:<id>，全文按节入 pages，level 标 2。零 LLM。"""
    p = dev_path()
    if not p.exists():
        return {"error": "先运行 qasper download"}
    data = json.loads(p.read_text(encoding="utf-8"))
    db.init_db()
    imported = sections_total = 0
    with db.conn() as c:
        for paper_id, paper in list(data.items())[:min(max_papers, MAX_PAPERS)]:
            key = f"qasper:{paper_id}"
            if db.get_by_norm_key(c, key):
                imported += 1
                continue
            pid = db.insert_l0(c, {
                "norm_key": key,
                "title": paper.get("title") or f"QASPER {paper_id}",
                "abstract": (paper.get("abstract") or "").strip(),
                "year": 2021, "venue": "QASPER-dev", "authors": [],
                "doi": None, "arxiv_id": None, "source": "qasper",
            })
            c.commit()
            for sec_no, (sec_name, body) in enumerate(_sections_of(paper), 1):
                c.execute("INSERT OR REPLACE INTO pages(paper_id,page_no,text) VALUES(?,?,?)",
                          (pid, sec_no, f"{sec_name}\n{body}"))
                sections_total += 1
            c.execute("UPDATE papers SET level=2 WHERE id=?", (pid,))
            db.reindex_pages(c, pid)        # 节级全文进全库检索索引
            # QASPER 按节导入，一个 "page" 本来就是一节 → 直接作检索单元
            db.replace_chunks(c, pid, db.chunks_from_pages(c, pid))
            c.commit()
            imported += 1
    return {"imported": imported, "sections": sections_total}


# ── 评测 ──

def _norm(s: str) -> str:
    return re.sub(r"\s+", "", (s or "")).lower()


def _evidence_map(paper: dict) -> dict[str, tuple[str, list[str]]]:
    """qid → (question, gold 证据片段列表)。

    v0.3 的结构是 qas[i].answers[j].answer.{evidence, highlighted_evidence, extractive_spans}，
    证据句在 `evidence`（原文段落）与 `highlighted_evidence`（标注者划的句子）里；
    `extractive_spans` 常常只是 "BIBREF19" 这种引用标记，当证据片段用会污染指标，
    只在它足够长时才收。原来的实现读的是 qa["evidence"]（更早的格式），在 v0.3 上恒为空。
    """
    out = {}
    for qa in paper.get("qas", []):
        spans: list[str] = []
        for ans in qa.get("answers") or []:
            a = ans.get("answer") if isinstance(ans, dict) else None
            if not isinstance(a, dict):
                continue
            for key in ("evidence", "highlighted_evidence"):
                spans.extend(s for s in (a.get(key) or []) if isinstance(s, str))
            spans.extend(s for s in (a.get("extractive_spans") or [])
                         if isinstance(s, str) and len(s.strip()) >= 20)
        spans = [s for s in dict.fromkeys(spans) if s.strip()]
        if spans:
            out[qa["question_id"]] = (qa.get("question") or "", spans)
    return out


def _load(max_papers: int) -> dict:
    p = dev_path()
    if not p.exists():
        raise FileNotFoundError("先运行 qasper download")
    return json.loads(p.read_text(encoding="utf-8"))


def run_eval(k: int = 5, sec_k: int = 2, max_papers: int = 30,
             max_questions: int | None = None) -> dict:
    """两层证据指标。有 gold 证据的问题构成评测集；干扰项来自全库（含其他 QASPER 篇）。"""
    from . import tools

    data = _load(max_papers)
    db.init_db()
    paper_items, sec_items = [], []
    n_q = 0
    with db.conn() as c:
        for paper_id, paper in list(data.items())[:min(max_papers, MAX_PAPERS)]:
            key = f"qasper:{paper_id}"
            row = c.execute("SELECT id FROM papers WHERE norm_key=?", (key,)).fetchone()
            if not row:
                continue
            pid = row["id"]
            sections = {r["page_no"]: r["text"] for r in c.execute(
                "SELECT page_no,text FROM pages WHERE paper_id=?", (pid,)).fetchall()}
            sec_norms = {no: _norm(t) for no, t in sections.items()}
            for qid, (question, golds) in _evidence_map(paper).items():
                if not question:
                    continue
                n_q += 1
                if max_questions and n_q > max_questions:
                    break
                gold_norms = [_norm(g) for g in golds if len(_norm(g)) >= 20]
                if not gold_norms:
                    continue

                # ① 论文级：gold 论文是否进入全库检索 top-k（干扰项 = 全库）
                r = tools.retrieve_library(question, k)
                paper_hit = 1.0 if pid in r["paper_ids"][:k] else 0.0
                paper_items.append({"paper": paper_id, "qid": qid, "hit": paper_hit,
                                    "mode": r.get("retrieval_mode") or "fts"})

                # ② 节级：gold 论文内按关键词重叠排节，top-sec_k 节是否含 gold 证据
                words = [w for w in re.findall(r"[a-zA-Z][a-zA-Z-]{2,}", question)][:8]
                def sec_score(no_txt):
                    no, t = no_txt
                    tl = t.lower()
                    return sum(tl.count(w.lower()) for w in words)
                ranked = sorted(sec_norms.items(), key=sec_score, reverse=True)[:sec_k]
                ev_hit = 0.0
                for no, _ in ranked:
                    if any(g in sec_norms[no] for g in gold_norms):
                        ev_hit = 1.0
                        break
                sec_items.append({"paper": paper_id, "qid": qid, "hit": ev_hit})
            if max_questions and n_q >= max_questions:
                break
    return {
        "paper_hit_at_k": round(sum(i["hit"] for i in paper_items) / max(len(paper_items), 1), 4),
        "evidence_recall_at_sec_k": round(sum(i["hit"] for i in sec_items) / max(len(sec_items), 1), 4),
        "k": k, "sec_k": sec_k, "n_questions": len(paper_items),
        "retrieval_mode": paper_items[0]["mode"] if paper_items else "unknown",
        "paper_items": paper_items[:30], "sec_items": sec_items[:30],
    }


# ── 上下文组装 A/B：证据到底有没有进上下文 ──────────────────────────────────

def _norm_ws(s: str) -> str:
    """只压缩空白、不删空白。

    与 `_norm`（删掉**全部**空白）配套使用。两个口径必须一起报：`_norm` 对
    「词间空格被吃掉」完全不敏感——`Thissectionpresents` 与 `This section presents`
    在它眼里得分相同，而线上的 chunks_fts（trigram）与 embedding 都对空格敏感。
    只报 `_norm` 会把「文本被粘连」这类缺陷记成零损失。
    """
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def context_ab(budgets=(2000, 4000, 8000, 16000), max_papers: int = 30,
               max_questions: int | None = None) -> dict:
    """等上下文预算下，比较两种上下文组装口径的**证据召回**。

    - ``chunk``：用检索实际命中的章节块组装（当前线上口径）
    - ``page`` ：拿 paper_id 回到 pages 表按词频重新选页（改造前的口径）

    检索被固定成「gold 论文已在候选里」，因此量到的差异只来自**组装**，
    与召回强弱无关——这正是「按章节切」那条结论所声称的东西。

    每个预算给两个口径 × 两种归一（strip=删全部空白、ws=只压缩空白）。
    0 token、0 外网。
    """
    from . import db, rag

    data = _load(max_papers)
    rows = []
    with db.conn() as c:
        for paper_id, paper in list(data.items())[:max_papers]:
            r = c.execute("SELECT id FROM papers WHERE norm_key=?",
                          (f"qasper:{paper_id}",)).fetchone()
            if not r:
                continue
            pid = r["id"]
            for _qid, (question, golds) in _evidence_map(paper).items():
                if not question:
                    continue
                g = [x for x in golds if len(_norm(x)) >= 20]
                if g:
                    rows.append((pid, question, g))
                if max_questions and len(rows) >= max_questions:
                    break
            if max_questions and len(rows) >= max_questions:
                break

    out: dict = {"n_questions": len(rows), "budgets": {}}
    for budget in budgets:
        per = {"chunk": {"strip": 0, "ws": 0}, "page": {"strip": 0, "ws": 0}}
        for pid, question, golds in rows:
            for mode in ("chunk", "page"):
                # page 口径 = 强制走改造前的 _page_context（按页选块）
                ctx = rag.prepare([{"role": "user", "content": question}],
                                  top_k=1, candidate_ids=[pid],
                                  max_context_chars=budget,
                                  _ctx_unit=mode)
                text = ctx.system
                if any(_norm(g) in _norm(text) for g in golds):
                    per[mode]["strip"] += 1
                if any(_norm_ws(g) in _norm_ws(text) for g in golds):
                    per[mode]["ws"] += 1
        n = max(len(rows), 1)
        out["budgets"][budget] = {
            m: {k: round(v / n, 4) for k, v in d.items()} for m, d in per.items()}
    return out
