# -*- coding: utf-8 -*-
"""表格摘要：让表格用**自然语言**参与语义检索。

## 为什么需要

库里 93 个 kind='table' 的块存的是原样 Markdown，然后整块拿去做 embedding：

    | Model | BLEU | chrF |
    | --- | --- | --- |
    | Baseline | 24.1 | 51.3 |
    | Ours | 27.8 | 55.0 |

用户问「哪个方法效果最好」「这篇的 BLEU 提升了多少」，跟这堆数字在语义空间里
几乎没有距离可言——表格里最关键的那几件事（谁跟谁比、比的是什么、谁赢了）
一个字都没写出来，它只存在于读者脑子里。词面那一路也救不了：
「效果最好」在表格里同样是零命中。

所以给每张表配一段模型写的说明，和表格**放在同一个块里**一起嵌入：
说明负责被语义检索找到，原表负责回答具体是多少。

## 三条约束

**① 摘要是模型写的，必须标出来。** 贴进块正文的那份带
`【表格摘要｜模型生成，非原文】` 前缀。逐字核验那几条路（`matrix` / `rcs`）
读的是 `pages` 表的整页原文，天然碰不到这段字；但人会看到上下文，模型也会看到，
所以出处要写在脸上。`table_summaries.is_model_written` 是同一件事在数据层的记号。

**② 按表的内容哈希存，不跟着 chunks 走。** `db.replace_chunks` 会把一篇的 chunks
整个换掉（重切块、改切分参数、`cli.py rechunk` 都会），摘要要是只存在
`chunks.text` 里，每重切一次就得重新买一遍。表的识别是确定性的——同一张表
重切前后哈希不变——所以摘要能跨重切复用，重切后由 `build_chunks` 自动贴回去。

**③ 花钱前先看账。** `estimate()` 一次接口都不调；`summarize()` 默认 `dry_run=True`。

**④ 表被抽坏时不许给结论。** 真库 86 张表里有 24 张（28%）列结构塌了——
多级表头被压平，本该分属几列的数字挤进同一个格。在那种表上让模型判「谁最好」，
实测会把论文的论点说反：论文 463 第 3 页 Bengali 行是 `6.72 8.83 9.19`
（No Pre-Order / Pre-Order HT / Pre-Order G），预排序把 BLEU 从 6.72 提到 8.83，
模型却写「预排序反而降低了翻译得分」。`looks_mangled()` 认出这种表并换一套
提示词，只写「在报告什么、比什么、用什么指标」，`structure_ok` 列记下这件事。
和 `figures.py` 那条「严禁读数」是同一条原则：**模型负责让内容被检索到，
不负责给可引用的结论**；区别只是表结构完好时结论可信，所以不必一刀切关掉。
"""
from __future__ import annotations

import hashlib
import os
import re

from . import config, db, llm

#: 送进模型的表格文本上限。绝大多数表远小于这个数；截断只是防一张病态的巨表
#: 把单次调用撑爆。
MAX_TABLE_CHARS = 6000

#: 摘要的目标长度。太短说不清「谁跟谁比、比什么、谁赢」；太长会稀释掉原表在
#: 同一个块里的权重——这个块的向量是摘要和表格一起算出来的。
TARGET_CHARS = 200

#: 值得摘要的下限。真库 102 张检出表里有 14 张只有 1 个数据行——那不是表，
#: 是检出噪声，模型对它们只会如实回答「表格内容缺失，无法判断」（真跑过，
#: 论文 482 第 5 页那张的全部内容就是一个 "ei"）。花钱买这句话，还要把它塞进
#: 检索块里当噪声，两头都不划算。
MIN_ROWS = 2
MIN_CHARS = 80

#: 单表实测用量（副本上 5 次调用，glm-5.2）：
#:     prompt 187 / 216 / 326 / 910 / 187      completion 478 / 787 / 1259 / 808 / 588
#: **completion 远大于可见输出**（约 200 字的说明却记了 500~1300 token）——
#: 这类模型的思考 token 也计费。按 chars/4 估输出会低估 3~4 倍，
#: 所以这里用实测中位数，而不是拿目标字数折算。
#:
#: **换模型要重测这个数**：同样一张表，`qwen3.5-omni-plus` 只用 93 completion token
#: （它不出思考 token），全程 547 token/表，是 glm-5.2 实测均值 1427 的 38%。
#: 用 `PAPERNEST_TABLE_MODEL` 指定模型时，这个估值会偏高——偏高比偏低安全，
#: 但报账时要说清楚它是按哪个模型估的。
MEASURED_COMPLETION_TOKENS = 790

#: 表格摘要用哪个模型。留空 = 跟 `config.LLM_MODEL` 走。
#: 单独开一个口子是因为**它和主模型的可用性可能不同**：2026-09-09 账户欠费后，
#: 249 个模型里只剩 8 个能调，主模型 glm-5.2 不在其中。
TABLE_MODEL = os.environ.get("PAPERNEST_TABLE_MODEL", "")

#: 贴进块正文的出处标记。**不要改**：`decorate` 靠它判断「已经贴过了」，
#: 改了会贴第二遍。
MARK = "【表格摘要｜模型生成，非原文】"

_BASE = ("你是学术论文的表格解读助手。给定一张论文里的表格，用中文写一段检索用的说明。"
         f"控制在 {TARGET_CHARS} 字以内，只输出这段说明本身，不要标题、不要 Markdown、"
         "不要复述整张表的数字。表里没有的信息一个字都不要编。")

#: 表格结构完好时用：可以给结论——「谁最好」正是检索最需要的那句话。
_SYSTEM_CLEAN = _BASE + (
    "必须写清楚四件事：这张表在报告什么、被比较的对象有哪些、用的指标是什么、"
    "以及数据显示的最主要结论（谁最好 / 趋势如何）。"
    "看不出结论就直说「表中未体现明确结论」。")

#: 表格被抽坏时用：**不许给结论**。理由见 `looks_mangled` 与模块注释。
_SYSTEM_MANGLED = _BASE + (
    "这张表是从 PDF 里自动抽取的，**列结构已经损坏**："
    "多级表头被压平了，一个单元格里可能挤着本该分属几列的数字。"
    "所以只写三件事：这张表在报告什么、被比较的对象有哪些、用的指标是什么。"
    "\n\n"
    "**严禁给结论**：不要说谁最好、谁高于谁、哪个方法有提升、趋势如何。"
    "你无法确定某个数字属于哪一列，据此下结论就是猜。"
    "结尾请写「（表格列结构在抽取中受损，未据此下结论）」。")


def looks_mangled(t: dict) -> bool:
    """这张表的列结构是不是在抽取中塌了。

    判据：**有两个以上单元格里挤着多个数字**。多级表头被压平时，本该分属
    几列的数值会挤进同一个格——真库 86 张表里有 24 张（28%）是这样。

    为什么必须区分：论文 463 第 3 页那张表的 Bengali 行是
    `6.72 8.83 9.19`（No Pre-Order / Pre-Order HT / Pre-Order G），
    预排序把 BLEU 从 6.72 提到 8.83。模型写出来的结论却是
    「所有语言在无预排序条件下的得分均高于预排序条件，预排序反而降低了翻译得分」
    ——**把这篇论文的论点说反了**。它不是在瞎编，是在一张列已经错位的表上
    没法判断哪个数字属于哪一列。

    这跟 `figures.py` 那条「严禁读数」是同一条原则：
    **模型负责让内容被检索到，不负责给可引用的结论**——区别只是表格结构完好时
    结论是可信的（那正是检索最需要的一句话），所以只在塌了的时候关掉它。
    """
    n = 0
    for row in (t.get("rows") or []):
        for cell in (row or []):
            if _MULTI_NUM_RE.match((cell or "").strip()):
                n += 1
                if n >= 2:
                    return True
    return False


#: 一个单元格里挤了两个以上数字——列被压平的信号。
_MULTI_NUM_RE = re.compile(r"^[-+]?\d[\d.,]*\s+[-+]?\d[\d.,]*(\s|$)")


def table_hash(paper_id: int, t: dict) -> str:
    """一张表的稳定标识：论文 + 页码 + 全部单元格文本（归一化后）。

    **不含 bbox 与行列数**：同一张表在不同版本的检出里 bbox 会微动，
    但只要单元格文本一样就还是同一张表，摘要就该复用。
    """
    cells = "\x1f".join(
        re.sub(r"\s+", " ", (c or "")).strip()
        for r in (t.get("rows") or []) for c in (r or []))
    raw = f"{int(paper_id)}\x1e{t.get('page_no')}\x1e{cells}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def markdown_of(t: dict) -> str:
    """一张表的完整 Markdown（`chunk_table` 的各块拼回去）。"""
    from . import tables
    try:
        parts = tables.chunk_table(t)
    except Exception:                                   # noqa: BLE001
        parts = []
    if parts:
        return "\n\n".join(p.get("text") or "" for p in parts)
    return "\n".join(" | ".join((c or "") for c in (row or []))
                     for row in (t.get("rows") or []))


def get(hash_: str) -> str | None:
    with db.conn() as c:
        r = c.execute("SELECT summary FROM table_summaries WHERE table_hash=?",
                      (hash_,)).fetchone()
    return r["summary"] if r else None


def get_many(paper_id: int) -> dict:
    """一篇的全部摘要，`{table_hash: summary}`。`build_chunks` 用它一次取完。"""
    with db.conn() as c:
        return {r["table_hash"]: r["summary"] for r in c.execute(
            "SELECT table_hash,summary FROM table_summaries WHERE paper_id=?",
            (paper_id,))}


def pending(paper_ids=None) -> list[dict]:
    """还没有摘要的表。要重新识别 PDF，所以慢——但它**不花钱**。

    只看有 PDF 文件在的论文：表格识别读的是 PDF，不是库里的文本。
    """
    from . import tables
    db.init_db()
    with db.conn() as c:
        rows = c.execute(
            "SELECT id,pdf_path,title FROM papers "
            "WHERE pdf_path IS NOT NULL AND pdf_path <> '' ORDER BY id").fetchall()
        have = {r["table_hash"] for r in c.execute(
            "SELECT table_hash FROM table_summaries")}
    want = set(paper_ids) if paper_ids else None
    out = []
    for r in rows:
        if want is not None and r["id"] not in want:
            continue
        if not os.path.exists(r["pdf_path"]):
            continue
        try:
            found = tables.detect_tables(r["pdf_path"])
        except Exception:                               # noqa: BLE001
            continue            # 识别不了就是这篇没有表，不是本模块该处理的事
        for t in found:
            h = table_hash(r["id"], t)
            if h in have:
                continue
            md = markdown_of(t)
            if not md.strip():
                continue
            if (t.get("n_rows") or 0) < MIN_ROWS or len(md) < MIN_CHARS:
                continue        # 检出噪声，见 MIN_ROWS
            out.append({"table_hash": h, "paper_id": r["id"], "title": r["title"],
                        "page_no": t.get("page_no"), "n_rows": t.get("n_rows"),
                        "n_cols": t.get("n_cols"), "markdown": md,
                        "n_chars": len(md), "mangled": looks_mangled(t)})
    return out


def estimate(items) -> dict:
    """动手前的账。**一次接口都不调。**

    输入按 chars/4 粗估（表格以英文数字为主）。输出**不按目标字数折算**，
    用实测中位数——第一版就是那么估的，结果实跑下来是估值的 2.7 倍：
    这类模型的思考 token 也计费，200 字的说明记了 500~1300 token。
    真实账单以 `llm_calls` 为准。
    """
    items = list(items)
    in_chars = sum(min(i["n_chars"], MAX_TABLE_CHARS) for i in items)
    # 输入：表格是英文数字为主，chars/4 够用；system 提示词是中文，按 chars/1.5 折。
    est_in = in_chars // 4 + len(items) * int(len(_SYSTEM_MANGLED) / 1.5)
    # 输出：用实测值，不用目标字数——见 MEASURED_COMPLETION_TOKENS。
    est_out = len(items) * MEASURED_COMPLETION_TOKENS
    return {"n_tables": len(items), "n_papers": len({i["paper_id"] for i in items}),
            "total_chars": sum(i["n_chars"] for i in items),
            "est_prompt_tokens": est_in, "est_completion_tokens": est_out,
            "est_total_tokens": est_in + est_out,
            "note": "输出按实测中位数 %d token/表（含思考 token）；真实用量以 llm_calls 为准"
                    % MEASURED_COMPLETION_TOKENS}


def summarize(items, dry_run: bool = True, progress=None, model=None) -> dict:
    """给每张表生成一段说明并落库。**默认 dry_run**——要花钱得显式说。

    单张表失败不影响其余：记进 `failed` 接着走。返回
    `{written, skipped, failed, dry_run, estimate}`。
    """
    items = list(items)
    est = estimate(items)
    if dry_run:
        return {"written": 0, "skipped": len(items), "failed": [],
                "dry_run": True, "estimate": est}
    if not llm.available():
        raise llm.LLMUnavailable("未配置 LLM，无法生成表格摘要")
    db.init_db()
    written, failed = 0, []
    m = model or TABLE_MODEL or config.LLM_MODEL
    for n, it in enumerate(items, 1):
        if progress:
            progress(n / max(1, len(items)),
                     f"表格摘要 {n}/{len(items)}"
                     f"（论文 {it['paper_id']} 第 {it['page_no']} 页）")
        user = (f"论文标题：{it.get('title') or ''}\n"
                f"表格位置：第 {it['page_no']} 页"
                f"（{it['n_rows']} 行 × {it['n_cols']} 列）\n\n"
                f"{it['markdown'][:MAX_TABLE_CHARS]}")
        try:
            sys_prompt = _SYSTEM_MANGLED if it.get("mangled") else _SYSTEM_CLEAN
            text = llm.chat(sys_prompt, user, purpose="table_summary",
                            paper_id=it["paper_id"], temperature=0.2,
                            model=m).strip()
        except Exception as exc:                        # noqa: BLE001
            failed.append({"table_hash": it["table_hash"],
                           "paper_id": it["paper_id"], "page_no": it["page_no"],
                           "error": f"{type(exc).__name__}: {exc}"})
            continue
        if not text:
            failed.append({"table_hash": it["table_hash"],
                           "paper_id": it["paper_id"], "page_no": it["page_no"],
                           "error": "模型返回空"})
            continue
        with db.conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO table_summaries(table_hash,paper_id,page_no,"
                "n_rows,n_cols,summary,model,is_model_written,structure_ok)"
                " VALUES(?,?,?,?,?,?,?,1,?)",
                (it["table_hash"], it["paper_id"], it["page_no"], it["n_rows"],
                 it["n_cols"], text, m, 0 if it.get("mangled") else 1))
        written += 1
    return {"written": written, "skipped": 0, "failed": failed,
            "dry_run": False, "estimate": est}


def decorate(chunk_text: str, summary) -> str:
    """把摘要贴到表格块正文前面。已经贴过、或压根没摘要，都原样返回（幂等）。"""
    if not chunk_text:
        return chunk_text
    # 全空白的摘要按「没有」处理：不然会贴出一个后面什么都没有的空标记，
    # 那比不贴更糟——它宣称这里有段模型写的说明，其实一个字都没有。
    text = str(summary or "").strip()
    if not text:
        return chunk_text
    if chunk_text.lstrip().startswith(MARK):
        return chunk_text
    return f"{MARK}{text}\n\n{chunk_text}"


def papers_with_unapplied_summaries() -> list[int]:
    """有摘要、但库里的表格块还没贴上的论文——这些需要重切一次块才生效。

    只读，不改任何东西：什么时候重切、要不要顺带重嵌，是调用方（和人）的决定。
    """
    db.init_db()
    with db.conn() as c:
        return [r["paper_id"] for r in c.execute(
            "SELECT DISTINCT s.paper_id FROM table_summaries s "
            "JOIN chunks ch ON ch.paper_id = s.paper_id AND ch.kind = 'table' "
            "WHERE ch.text NOT LIKE ?", (MARK + "%",))]
