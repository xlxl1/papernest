"""L2 全文精读：OA PDF 下载 → 按页抽取 → 带页码的精读卡片（机械回取校验）。

合规：只下载 openAccessPdf 声明的 OA 链接；%PDF 魔数校验；付费墙永不触碰。
精读卡片的 key_findings 逐条做机械校验——claim 的连续片段必须真的出现在
其声称的那一页文本里，验不上的标 verified=false（不删，如实标注）。
"""
import json
import re

import httpx

from . import config, db, http, llm, netguard

PDF_DIR = config.DATA_DIR / "pdf"

L2_SYSTEM = """你是学术文献精读助手。基于给定的带页码全文，输出 JSON：
- tldr: 三句话精读总结
- key_findings: [{ "claim": 该文的关键结论（原文依据，不要改写数字）, "page": 页码整数 }]
- method_detail: 方法细节（比摘要层更具体）
- limitations: 局限
- relation_to_topic: 结合我的课题，这篇文献能具体用在哪
硬性要求：key_findings 只能来自原文，claim 里保留原文的关键数字与术语；
page 必须是该结论所在页。只输出 JSON。"""


def fetch_pdf(paper_id: int) -> dict:
    """下载 OA PDF 落盘。返回 {path} 或 {error}。

    硬约束：整体 120s 截止 + 25MB 上限（socket 级超时挡不住慢速滴流下载，
    没有 total deadline 会无限挂起——实测踩过的坑）。
    """
    import time
    with db.conn() as c:
        row = c.execute("SELECT * FROM papers WHERE id=?", (paper_id,)).fetchone()
    if not row:
        return {"error": f"论文 {paper_id} 不存在"}
    url = row["oa_pdf_url"]
    # arXiv 兜底的条件不能只看「有没有 url」：netguard 只放行 https，而 S2 的
    # openAccessPdf 会原样给出 `http://arxiv.org/pdf/...`（真库 5 篇）。
    # 只判 `not url` 的话，这些论文会被我们**自己的** SSRF 防线永久挡死，
    # 而错误文案「只允许 https 公网地址」还会把人引去排查网络。
    if row["arxiv_id"] and (not url or not url.lower().startswith("https://")):
        url = f"https://arxiv.org/pdf/{row['arxiv_id']}"  # arXiv 恒为 OA，且支持 https
    if not url:
        return {"error": "该论文没有 OA PDF 链接（可能付费墙，按合规红线不获取）"}
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    path = PDF_DIR / f"{paper_id}.pdf"
    if path.exists() and path.stat().st_size > 1024:
        return {"path": str(path)}
    deadline = time.monotonic() + 120
    max_bytes = 25 * 1024 * 1024
    try:
        # **逐跳校验 + 自己跟重定向**。`http.client()` 默认 follow_redirects=True，
        # 那样一个合法外域 302 到 169.254.169.254 就绕过了入口校验——所以这里
        # 关掉自动跟随，每一跳都重新过一次 netguard.check_url。
        # 目标 URL 可由用户经 .bib 导入写进 oa_pdf_url，必须当成不可信输入。
        with http.client(timeout=httpx.Timeout(connect=15, read=30, write=30, pool=15),
                         headers={"User-Agent": "PaperNest/0.1"},
                         follow_redirects=False) as client:
            target = netguard.check_url(url)
            for _hop in range(netguard.MAX_REDIRECTS + 1):
                with client.stream("GET", target) as r:
                    if r.is_redirect:
                        loc = r.headers.get("location") or ""
                        target = netguard.check_url(str(r.url.join(loc)))
                        continue
                    if r.status_code != 200:
                        # **不回吐上游状态码**：它是一个精确的内网探测 oracle
                        # （200/401/403/连接失败各不相同）。只说下载失败。
                        return {"error": f"下载失败（来源不可用），来源 {url[:80]}"}
                    buf = bytearray()
                    for chunk in r.iter_bytes(65536):
                        buf.extend(chunk)
                        if len(buf) > max_bytes:
                            return {"error": "文件超过 25MB 上限，放弃下载"}
                        if time.monotonic() > deadline:
                            return {"error": "下载超过 120 秒截止，放弃（站点过慢或被阻断）"}
                    if bytes(buf[:5]) != b"%PDF-":
                        return {"error": f"返回内容不是 PDF（可能是拦截页），来源 {url[:80]}"}
                    path.write_bytes(buf)
                    return {"path": str(path)}
            return {"error": f"重定向超过 {netguard.MAX_REDIRECTS} 跳，放弃"}
    except netguard.BlockedURL:
        # 拒绝原因（解析到哪个 IP 段）不回吐：它本身就是一条内网探测的反馈
        return {"error": "目标地址不允许访问（只允许 https 公网地址）"}
    except Exception as e:
        return {"error": f"下载失败：{e}（本机网络可能屏蔽该站点，开代理后重试）"}


def extract_pages(path: str, paper_id: int, max_pages: int = 40) -> int:
    """PyMuPDF 按页抽文本入 pages，同时按**章节**切一份 chunks 作检索单元。

    两个单元各司其职，别合并：
    - pages 按物理页 —— 精读卡片的 key_findings 要报准页码，机械回取也按页验；
    - chunks 按章节 —— 检索要的是语义完整的块。实测（25 篇真实 arXiv PDF、
      69 道 QASPER 带 gold 证据的题）等上下文预算下证据召回是按页切的 2.2~3.2 倍。

    章节识别失败（扫描件、纯图 PDF 是常态）时 chunks 退化成按页，如实降级不报错。
    返回实际处理的页数（0 页 PDF 返回 0，不炸）。
    """
    import pymupdf
    doc = pymupdf.open(path)
    n = 0
    try:
        with db.conn() as c:
            c.execute("DELETE FROM pages WHERE paper_id=?", (paper_id,))
            for i, page in enumerate(doc, 1):
                if i > max_pages:
                    break
                n = i
                text = page.get_text("text").strip()
                if text:
                    c.execute(
                        "INSERT OR REPLACE INTO pages(paper_id,page_no,text) VALUES(?,?,?)",
                        (paper_id, i, text))
            c.execute("UPDATE papers SET pdf_path=? WHERE id=?", (path, paper_id))
            db.reindex_pages(c, paper_id)   # 精读拿到的正文要进得了全库检索
            db.replace_chunks(c, paper_id, build_chunks(path, c, paper_id))
    finally:
        doc.close()
    return n


def build_chunks(path: str, c, paper_id: int) -> list[dict]:
    """按章节切检索单元，表格另行成块；识别不到结构就退化成按页（不抛异常）。

    表格必须单独成块并**整行切分、表头跨块重复**：论文里的结果表被页边界或
    字数上限拦腰砍断后，下半张没有表头，检索到也读不懂——那是 chunk 质量最隐蔽的
    杀手之一。表格块标 kind='table'，检索结果里可以据此换一种呈现。
    """
    from . import structure, tables
    try:
        cs = structure.section_chunks(path)
    except Exception:
        cs = []
    out: list[dict] = []
    if cs:
        out = [{"text": ch.get("text") or "",
                "section_path": ch.get("section_path") or ch.get("section_title") or "",
                "level": ch.get("level") or 1,
                "start_page": ch.get("start_page"), "end_page": ch.get("end_page"),
                "kind": "text"}
               for ch in cs]
    else:
        out = db.chunks_from_pages(c, paper_id)

    try:
        for t in tables.detect_tables(path):
            for part in tables.chunk_table(t):
                out.append({
                    "text": part["text"],
                    "section_path": f"表格（第 {t['page_no']} 页）"
                                    + (f" {part['part']}/{part['of']}"
                                       if part.get("of", 1) > 1 else ""),
                    "level": 1, "start_page": t["page_no"], "end_page": t["page_no"],
                    "kind": "table"})
    except Exception:
        pass        # 表格识别失败不影响正文块——正文才是主路径
    return out


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", (s or "").lower())


def _verify_claim(claim: str, page_no: int, pages: dict[int, str]) -> bool:
    """机械回取：claim 的连续 12 字符片段必须真的出现在声称的那一页。"""
    c_n, p_n = _norm(claim), _norm(pages.get(page_no, ""))
    if len(c_n) < 12 or not p_n:
        return False
    step = max(len(c_n) // 8, 8)
    probes = [c_n[i:i + 12] for i in range(0, min(len(c_n), 200), step)]
    hit = sum(1 for pr in probes if pr in p_n)
    return hit >= max(1, len(probes) // 3)


def read_paper(paper_id: int, topic: str | None = None, progress=None) -> dict:
    db.init_db()
    if progress:
        progress(0.1, "download", "下载 OA PDF 中")
    dl = fetch_pdf(paper_id)
    if "error" in dl:
        return dl
    if progress:
        progress(0.45, "extract", "按页抽取全文")
    n_pages = extract_pages(dl["path"], paper_id)
    with db.conn() as c:
        pages = {r["page_no"]: r["text"]
                 for r in c.execute("SELECT page_no,text FROM pages WHERE paper_id=?",
                                    (paper_id,)).fetchall()}
        row = c.execute("SELECT title FROM papers WHERE id=?", (paper_id,)).fetchone()
    if not llm.available():
        return {"paper_id": paper_id, "pages_stored": len(pages),
                "card": None, "note": "全文已抽取入库（可在对话中检索引用）；L2 精读卡片需要配置 LLM"}

    if progress:
        progress(0.6, "card", "重档模型生成精读卡片")
    full = "\n\n".join(f"【第 {pno} 页】\n{txt[:4000]}" for pno, txt in sorted(pages.items()))
    user = (f"我的课题：{topic or config.RESEARCH_TOPIC}\n\n"
            f"标题：{row['title']}\n\n全文（带页码）：\n{full[:80000]}")
    text = llm.chat(L2_SYSTEM, user, purpose="card_l2", paper_id=paper_id,
                    temperature=0.2, model=config.heavy_model())
    card = llm.extract_json(text)
    for f in card.get("key_findings") or []:
        f["verified"] = _verify_claim(f.get("claim", ""), int(f.get("page", 0) or 0), pages)
    verified_n = sum(1 for f in card.get("key_findings") or [] if f.get("verified"))
    card["_verified_summary"] = f"{verified_n}/{len(card.get('key_findings') or [])} 条关键结论通过机械回取校验"
    with db.conn() as c:
        c.execute("""UPDATE papers SET card_json=?, card_model=?, level=MAX(level,2),
                     updated_at=datetime('now','localtime') WHERE id=?""",
                  (json.dumps(card, ensure_ascii=False), config.heavy_model(), paper_id))
        c.commit()
    if progress:
        progress(0.98, "verify", "机械回取校验完成")
    return {"paper_id": paper_id, "pages_stored": len(pages), "card": card}


def search_pages(paper_id: int, keyword: str, limit: int = 3):
    with db.conn() as c:
        like = f"%{keyword}%"
        return c.execute("""SELECT page_no, text FROM pages WHERE paper_id=? AND text LIKE ?
                            ORDER BY page_no LIMIT ?""", (paper_id, like, limit)).fetchall()
