# -*- coding: utf-8 -*-
"""插图理解：把论文里的图变成能被检索到的文字。

## 现状

全仓**零图片抽取**。论文里的图对这个系统完全不存在——用户问「那个架构图长什么样」
「模型结构是怎么连的」，检索侧一个字都没有。

## 为什么以图题为锚，而不是 `page.get_images()`

先量了一遍真库（26 篇 233 页）：位图 148 张、图题 70 条。看着位图更多，
但两边指的根本不是一回事：

- **论文 460 有 4 条图题、0 张位图**；462 是 3 条图题、0 张位图。
  它们的图是**矢量**画的（matplotlib / TikZ 出的 PDF 图元），`get_images()` 看不见。
- 反过来，论文 1 有 93 张位图却只有 5 条图题——那 93 张多半是图里的小块和装饰。

所以按位图抓，会**漏掉整类矢量图**，同时抓回一堆碎片。按图题抓才对得上
「读者眼里的一张图」。代价是没有图题的图抓不到，那种图本来也没法给它写说明。

## 图题判据

`^(Figure|Fig.|图|FIGURE)\\s*\\d+\\s*[:：.]`——**数字后面必须跟冒号或句点**。
不要这一条的话会把正文里提到图的句子一起收进来：真库上宽判据 75 条、严判据 70 条，
多出的 5 条逐条看过，全是「Figure 1 presents our organizational framework…」
「Figure 3 shows that two baselines…」这种正文句子，一条真图题都没有。

## 图区怎么定

图题**上方**、到上一个文本块底边（或栏顶）之间的矩形，按栏宽收口。
渲染这块区域，矢量图和位图一并进来——这正是以图题为锚的好处。

## 为什么提示词里**严禁读数**

第一版让模型「说明它想说明的关键点」，跑真图核对时抓到一条实质性错误：
论文 459 图 3（三组余弦相似度折线图），模型说
「绿色线（MLM+BRLM-SA）在第 1–5 层接近 1.0，**第 6 层骤降**」。
对着图看，绿线恰恰是**唯一不掉**的那条（1.0 → 0.98），骤降的是红线和灰/蓝线。
而「BRLM-SA 能把相似度保持住」正是这篇论文的论点——模型把图的结论说反了。

同一段说明里，结构性的断言（三组子图、坐标轴名称与范围、四条曲线的图例）
**全部正确**。错的只集中在「读数值 / 判走势」这一类。

所以现在的分工是：**图负责被检索到，不负责给结论**。
说明只写图的类型、成分、比较关系；具体数值和趋势交给正文和表格——
那些能逐字核验（`matrix` / `rcs` 回取 `pages` 原文），模型从像素里读出来的数字不能。

加了这条约束后重跑同样三张图，读数完全消失。但**残留一类更轻的错误**：
同一张图 3 的说明里写「每组含**五条**曲线」，紧接着列的却是四种模型（图里是四条）
——数量会数错。所以贴进块里的说明必须带 `MARK` 前缀：它是检索的入口，
不是可引用的事实来源。

## 存哪

跟表格摘要同一套：按内容哈希存在 `figure_summaries`，不进 `chunks.text` 主存储。
`db.replace_chunks` 会把一篇的 chunks 整个换掉，跟着存就等于每次重切都重新买一遍。
贴进检索块的那份带 `MARK` 前缀——**它是模型看图写的，不是论文原文**。
"""
from __future__ import annotations

import hashlib
import os
import re

from . import config, db

#: 图题：数字后必须跟冒号/句点，否则会把正文里提到图的句子一起收进来（见模块注释）。
CAPTION_RE = re.compile(r"^\s*(?:Figure|Fig\.?|FIGURE|图)\s*\.?\s*(\d+[a-z]?)\s*[:：.]",
                        re.I)

#: 渲染图区的 DPI。150 够模型看清结构图的连线与标签，再高只是把 token 烧掉。
RENDER_DPI = 150

#: 图区的最小尺寸（点）。比这还小的多半是被误判的行内公式或角标。
MIN_W = 80.0
MIN_H = 50.0

#: 图区向上最多回溯多少点。防止一页只有图题时把整页正文都当成图。
MAX_UP = 620.0

#: 贴进块正文的出处标记。**不要改**：`decorate` 靠它判断「已经贴过了」。
MARK = "【插图说明｜模型看图生成，非原文】"

#: 提示词的核心约束是**不许读数**，理由见下面的实测。
_SYSTEM = (
    "你是学术论文的插图解读助手。给你一张论文插图和它的图题，"
    "用中文写一段检索用的说明，只写这三件事："
    "① 这是什么类型的图（架构图 / 流程图 / 折线图 / 柱状图 / 示例图 / 热力图……）；"
    "② 图里有哪些主要成分——模块名、节点、图例项、坐标轴的**名称**、子图的划分；"
    "③ 图在结构上呈现的对比关系是什么（谁和谁在比、沿什么维度比）。"
    "\n\n"
    "**严禁读数**：不要报任何坐标值、峰值、百分比、"
    "也不要说某条线「上升 / 下降 / 骤降 / 最高」。"
    "曲线的具体走势请一律略过，只说明这张图在比较哪些对象。"
    "结论性的话交给正文和表格，那些是可以逐字核验的，你读出来的数字不是。"
    "\n\n"
    "200 字以内，只输出说明本身。图里没有的东西一个字都不要编，看不清就说看不清。"
    "图题已经说过的话不要重复，要补的是图题里没有的视觉信息。")


def figure_hash(paper_id: int, page_no: int, caption: str) -> str:
    """一张图的稳定标识：论文 + 页码 + 图题（归一化后）。

    用图题而不是像素：重新渲染、换 DPI 都不该让说明作废；而同一篇同一页上
    图题一样的两张图不存在。
    """
    cap = re.sub(r"\s+", " ", caption or "").strip()
    raw = f"{int(paper_id)}\x1e{int(page_no)}\x1e{cap}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _is_prose(block) -> bool:
    """这个文本块是不是**正文段落**（而不是图里的标签）。

    图区的上边界不能取「上方最近的任意文本块」：论文插图里的坐标轴刻度、
    节点名、图例都是独立文本块，而且就落在图**内部**——取它们当上界，
    圈出来的图区会只剩图题上方那一薄条（实测 459 第 1 页只圈到 86pt 高，
    真图有 300 多）。正文段落长、词多；图内标签短而碎，这是两者最稳的差别。
    """
    t = (block[4] or "").strip()
    return len(t) >= 60 and len(t.split()) >= 10


def _column_left_right(blocks, cap_rect, page_rect):
    """图题所在栏的左右边界。双栏论文里不收口的话会把邻栏正文一起渲进来。"""
    x0, x1 = cap_rect[0], cap_rect[2]
    mid = (x0 + x1) / 2.0
    same = [b for b in blocks
            if b[0] < mid < b[2] or (abs(b[0] - x0) < 24 and abs(b[2] - x1) < 24)]
    if same:
        x0 = min(x0, min(b[0] for b in same))
        x1 = max(x1, max(b[2] for b in same))
    return max(page_rect[0], x0 - 6), min(page_rect[2], x1 + 6)


def find_figures(pdf_path: str) -> list[dict]:
    """按图题定位插图。返回 `[{page_no, caption, rect, ...}]`，**不渲染、不花钱**。"""
    import pymupdf
    out = []
    doc = pymupdf.open(pdf_path)
    try:
        for i, page in enumerate(doc, 1):
            blocks = [b for b in page.get_text("blocks") if (b[4] or "").strip()]
            blocks.sort(key=lambda b: (b[1], b[0]))
            pr = tuple(page.rect)
            for n, b in enumerate(blocks):
                text = (b[4] or "").strip()
                m = CAPTION_RE.match(text)
                if not m:
                    continue
                cap_top = b[1]
                x0, x1 = _column_left_right(blocks, b, pr)
                # 上边界：同栏里、图题上方最近的一个**正文段落**的底边。
                # 只认正文段落，不认任意文本块——见 `_is_prose`。
                above = [o[3] for o in blocks
                         if o is not b and o[3] <= cap_top + 1
                         and o[0] < x1 and o[2] > x0 and _is_prose(o)]
                top = max(above) if above else pr[1]
                top = max(top, cap_top - MAX_UP, pr[1])
                if cap_top - top < MIN_H or x1 - x0 < MIN_W:
                    continue        # 图题上面没有空间：多半是接排的续行，不是图
                # 下边界多给 4pt：文本块的 bbox 比字面稍紧，正好卡在图题上沿会
                # 削掉图最底下一排元素（实测 472 图 3 底部一排圆点被切了半截）。
                # 多切进去的那一条最多带上图题第一行，无害——图题本来就要单独给模型。
                bottom = min(pr[3], cap_top + 4.0)
                out.append({"page_no": i, "caption": re.sub(r"\s+", " ", text).strip(),
                            "label": m.group(1),
                            "rect": (float(x0), float(top), float(x1), float(bottom)),
                            "width": float(x1 - x0), "height": float(bottom - top)})
    finally:
        doc.close()
    return out


def render_figure(pdf_path: str, page_no: int, rect, dpi: int = RENDER_DPI) -> bytes:
    """把图区渲染成 PNG。矢量图和位图一并进来——这是按图题定位的好处。"""
    import pymupdf
    doc = pymupdf.open(pdf_path)
    try:
        page = doc[page_no - 1]
        clip = pymupdf.Rect(*rect)
        return page.get_pixmap(dpi=dpi, clip=clip).tobytes("png")
    finally:
        doc.close()


def get_many(paper_id: int) -> dict:
    """一篇的全部插图说明，`{figure_hash: summary}`。`build_chunks` 用它一次取完。"""
    with db.conn() as c:
        return {r["figure_hash"]: r["summary"] for r in c.execute(
            "SELECT figure_hash,summary FROM figure_summaries WHERE paper_id=?",
            (paper_id,))}


def pending(paper_ids=None) -> list[dict]:
    """还没有说明的图。要重新解析 PDF，所以慢——但它**不花钱**。"""
    db.init_db()
    with db.conn() as c:
        rows = c.execute(
            "SELECT id,pdf_path,title FROM papers "
            "WHERE pdf_path IS NOT NULL AND pdf_path <> '' ORDER BY id").fetchall()
        have = {r["figure_hash"] for r in c.execute(
            "SELECT figure_hash FROM figure_summaries")}
    want = set(paper_ids) if paper_ids else None
    out = []
    for r in rows:
        if want is not None and r["id"] not in want:
            continue
        if not os.path.exists(r["pdf_path"]):
            continue
        try:
            figs = find_figures(r["pdf_path"])
        except Exception:                               # noqa: BLE001
            continue
        for f in figs:
            h = figure_hash(r["id"], f["page_no"], f["caption"])
            if h in have:
                continue
            out.append({**f, "figure_hash": h, "paper_id": r["id"],
                        "title": r["title"], "pdf_path": r["pdf_path"]})
    return out


def estimate(items) -> dict:
    """动手前的账。**一次接口都不调。**

    视觉模型的 prompt token 主要是图片本身，按渲染后的像素面积估
    （约 28x28 像素 1 token，是这一族模型的通行口径）；输出用实测中位数
    （见 `tablesum.MEASURED_COMPLETION_TOKENS` 那条：思考 token 也计费）。
    """
    from . import tablesum
    items = list(items)
    px = sum(int(i["width"] * RENDER_DPI / 72) * int(i["height"] * RENDER_DPI / 72)
             for i in items)
    est_in = px // (28 * 28) + len(items) * 200
    est_out = len(items) * tablesum.MEASURED_COMPLETION_TOKENS
    return {"n_figures": len(items), "n_papers": len({i["paper_id"] for i in items}),
            "est_prompt_tokens": est_in, "est_completion_tokens": est_out,
            "est_total_tokens": est_in + est_out,
            "note": "图片 token 按 28x28 像素/token 估；输出按实测中位数"}


def describe(items, dry_run: bool = True, progress=None, model=None) -> dict:
    """给每张图写一段说明并落库。**默认 dry_run**——要花钱得显式说。"""
    items = list(items)
    est = estimate(items)
    if dry_run:
        return {"written": 0, "skipped": len(items), "failed": [],
                "dry_run": True, "estimate": est}
    if not (config.VISION_MODEL and config.LLM_API_KEY and config.LLM_API_BASE):
        raise RuntimeError("未配置视觉模型（PAPERNEST_VISION_MODEL / LLM_API_*）")
    db.init_db()
    m = model or config.VISION_MODEL
    written, failed = 0, []
    for n, it in enumerate(items, 1):
        if progress:
            progress(n / max(1, len(items)),
                     f"插图 {n}/{len(items)}（论文 {it['paper_id']} 第 {it['page_no']} 页 "
                     f"图 {it['label']}）")
        try:
            png = render_figure(it["pdf_path"], it["page_no"], it["rect"])
            text = _ask_vision(png, it, m).strip()
        except Exception as exc:                        # noqa: BLE001
            failed.append({"figure_hash": it["figure_hash"],
                           "paper_id": it["paper_id"], "page_no": it["page_no"],
                           "error": f"{type(exc).__name__}: {exc}"})
            continue
        if not text:
            failed.append({"figure_hash": it["figure_hash"],
                           "paper_id": it["paper_id"], "page_no": it["page_no"],
                           "error": "模型返回空"})
            continue
        with db.conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO figure_summaries(figure_hash,paper_id,page_no,"
                "label,caption,summary,model,is_model_written) VALUES(?,?,?,?,?,?,?,1)",
                (it["figure_hash"], it["paper_id"], it["page_no"], it["label"],
                 it["caption"], text, m))
        written += 1
    return {"written": written, "skipped": 0, "failed": failed,
            "dry_run": False, "estimate": est}


def _ask_vision(png: bytes, item: dict, model: str) -> str:
    """一次视觉调用。**把图题一起给模型**：它是唯一可信的锚，
    能挡住「看图说话说岔了」——模型至少知道这张图应该在讲什么。"""
    import base64
    import json
    import time
    import urllib.request

    from . import budget
    budget.check("插图理解")
    b64 = base64.b64encode(png).decode("ascii")
    payload = {"model": model, "temperature": 0.2,
               "messages": [{"role": "system", "content": _SYSTEM},
                            {"role": "user", "content": [
                                {"type": "image_url",
                                 "image_url": {"url": f"data:image/png;base64,{b64}"}},
                                {"type": "text",
                                 "text": f"论文：{item.get('title') or ''}\n"
                                         f"图题：{item['caption']}"}]}]}
    req = urllib.request.Request(
        config.LLM_API_BASE.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": "Bearer " + config.LLM_API_KEY,
                 "Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.loads(r.read().decode("utf-8"))
    txt = ((d.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    if isinstance(txt, list):               # 有的网关把 content 返成分段列表
        txt = "".join(p.get("text", "") for p in txt if isinstance(p, dict))
    u = d.get("usage") or {}
    with db.conn() as c:                    # 视觉调用也要进账本，否则闸门看不见它
        c.execute("INSERT INTO llm_calls(ts,purpose,model,paper_id,prompt_tokens,"
                  "completion_tokens,latency_ms) VALUES(datetime('now','localtime'),"
                  "'figure_summary',?,?,?,?,?)",
                  (model, item.get("paper_id"), u.get("prompt_tokens") or 0,
                   u.get("completion_tokens") or 0, int((time.time() - t0) * 1000)))
    return txt


def decorate(chunk_text: str, summary) -> str:
    """把说明贴到插图块正文前面。已经贴过、或没有说明，都原样返回（幂等）。"""
    if not chunk_text:
        return chunk_text
    text = str(summary or "").strip()
    if not text:
        return chunk_text
    if chunk_text.lstrip().startswith(MARK):
        return chunk_text
    return f"{MARK}{text}\n\n{chunk_text}"
