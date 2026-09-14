"""库内 RAG 问答：检索（向量优先 / FTS 兜底）→ 组装带编号上下文 → 严格引用作答。

来源铁律：回答只能基于检索到的文献上下文；每条论断带 [n] 角标；
上下文没覆盖的要直说「库内文献未覆盖」，不许编。
"""
import json
import os
import math
import re
from typing import NamedTuple

from . import chat, config, db, degrade, embeddings, http, llm


SYSTEM = """你是科研文献库的问答助手。仅可依据「检索到的文献上下文」回答问题：
1. 每条来自文献的论断末尾标注来源编号，如 [1]、[2]；编号与上下文一一对应。
2. 上下文不足以回答时，明确说「库内文献未覆盖这个问题」，并可建议检索关键词。
3. 解释术语时优先引用上下文里文献的原句；上下文没有该术语时才允许用通识解释，并注明「（通识解释，非库内来源）」。
4. 润色/写作类请求不强制加角标，但涉及文献事实的部分仍须标注。
5. 有「之前的对话」时，用它理解「它/这篇/第 2 篇」这类指代；但事实仍只能来自文献上下文，
   不要把之前轮次里自己说过的话当成已证实的来源。
用中文回答。"""


def _vector_candidates(q: str, top_k: int
                       ) -> tuple[list[int], list[degrade.Degradation], dict]:
    """检索候选 + 本次检索的降级记录。

    降级判定**不再由这里反推**。原来是 `if mode == "fts" and available()`——
    而 mode 在有 chunk 命中时是 'fts+chunks'，这个等号恒为假，于是向量整路挂掉时
    一句提示都不会出现（默认配置下 chunk 路是开的，正是最常见的情况）。
    现在 search_hybrid 直接把结构化的降级记录返回出来，原样透传即可。
    """
    res = embeddings.search_hybrid(q, top_k)
    return res.ids, list(res.degraded), dict(res.chunk_hits or {})


#: 页面取词：ASCII 按词，CJK 按 2-gram。原来是 `q.split()`——中文问句没有空格，
#: 整句被当成一个「词」，拿去 `t.count()` 数英文正文恒为 0，于是中文提问时
#: L2 全文页**永远**进不了上下文（实测：库内 45 篇有全文的论文里 40 篇正文是英文）。
#: 上下文的字符预算。篇数由调用方的 top_k 决定、单篇由 TL;DR + key_findings + 摘要 +
#: 页块累加，两头原来都没有上限；`llm.chat` 拼 payload 时也不做长度检查，超模型窗口时
#: 服务端返 400，而 400 是「立即失败不重试」——于是整个任务在烧掉前几轮 token 之后
#: 失败，用户只看到一句透传的 provider 报错，看不出是上下文超了。
#: 24000 字符按中文口径 ≈13k token。两个口径的实测：`llm_calls` 里 22 次真实问答的
#: prompt_tokens 区间是 211–6451；而修好 `_page_context`（中文提问下全文页原本恒不进
#: 上下文）之后，合成的最坏情况（12 篇全带 L2 全文）到 12.6k token。
#: 预算与 `deepsearch.MAX_CONTEXT_PAPERS=12` 对齐：12 篇正好是一篇都不丢的点。
MAX_CONTEXT_CHARS = 24000

_ASCII_TOK = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_CJK_RUN = re.compile(r"[一-鿿]+")


def _query_terms(text: str, limit: int = 12) -> list[str]:
    """切出给页面打分用的检索词。中英分治，去重保序后截断，结果稳定可复现。

    不额外过滤中文虚词：在所有页上均匀出现的词对**页间排序**没有贡献、会自己抵消，
    多一张停用词表反而多一处要维护的口径。
    """
    terms: list[str] = []
    for w in _ASCII_TOK.findall(text or ""):
        if len(w) >= 2:
            terms.append(w.lower())
    for run in _CJK_RUN.findall(text or ""):
        terms += [run[i:i + 2] for i in range(len(run) - 1)]
    return list(dict.fromkeys(terms))[:limit]


def _window(text: str, terms: list[str], cap: int) -> str:
    """截到 cap 字符，**名义上**围绕命中位置取窗口。

    原来是 `txt[:cap]`：选中一页之后从头截，实测中位数只保留了 29%——
    而命中往往在页中部，于是「选中了正确的页、却把证据切掉了」。

    ## 但它多数时候并没有真的居中（2026-09-10 实测）

    `terms` 来自 `_query_terms`，**带停用词**。`min(find(t))` 取的是任意词的首个
    出现位置，而 the/of 几乎必然出现在开头：241 个真实块上量下来，
    **首个命中的中位位置是 51 字符、68% 落在前 100 字符内**，
    于是 `start = max(0, pos - cap//3)` = 0——等价于回到 `text[:cap]`。

    ## 试过按「加权命中密度」选窗口，**没能证明更好**

    每次出现按 1/tf 记分（在这块里出现 100 次的词每次只值 0.01），
    取密度最高的一段。274 篇 / 924 题 / 4566 条 gold 探针：

        截窗留存 42.7% → 45.4%   片段召回 0.1334 → 0.1419   问题召回 0.2803 → 0.3063
        问题级 p=0.0667（160 题变化）   探针级 p=0.0834（**263 好 / 224 坏**）

    263 对 224 基本是抛硬币。这次样本量是够的（n=4566），所以不能再说「判不出来」
    ——**就是不可靠**。机制上明显更合理不等于实测更好，没有采用。

    真要改善这一环，方向多半不在「窗口取哪一段」，而在**块本身有多大**：
    4000 字的块截到 800 字，丢掉 80% 是结构性的。见 `qasper.chunk_sweep`。
    """
    if len(text) <= cap:
        return text
    low = text.lower()
    pos = min((p for p in (low.find(t) for t in terms) if p >= 0), default=-1)
    if pos < 0:
        return text[:cap]
    start = max(0, pos - cap // 3)
    seg = text[start:start + cap]
    return ("…" + seg) if start else seg


def _pick_pages(pages: dict[int, str], terms: list[str],
                per_paper: int, cap: int) -> list[str]:
    """按词频选页。**分数除以 sqrt(页长)**，否则最长的那页恒赢。

    原来是绝对词频求和，长页天然占优：真库实测 41 篇里 34 篇（83%）打分第一的
    就是该篇字符数最多的页（往往是相关工作/参考文献/附录）。长度归一之后
    比较的是「密度」而不是「体量」。
    """
    if not terms:
        return []
    scored = []
    for pno, txt in pages.items():
        t = (txt or "").lower()
        raw = sum(t.count(k) for k in terms)
        scored.append((raw / ((len(t) or 1) ** 0.5), raw, pno, txt))
    # 排序键必须全序：并列时按页码决胜。原来只按分数排，并列页的先后取决于
    # SELECT 的返回顺序（没有 ORDER BY），同一条问句可能选出不同的页。
    scored.sort(key=lambda x: (-x[0], x[2]))
    return [f"    【第 {pno} 页】{_window(txt, terms, cap // per_paper)}"
            for _d, raw, pno, txt in scored[:per_paper] if raw > 0]


#: 上下文的组装单元：page（默认）| chunk。**默认 page 是量出来的，不是没接线。**
#:
#: 把上下文换成「按章节块组装」是本项目一条**负结果**。三臂配对实验
#: （QASPER 88 道带 gold 证据的题，等上下文预算，纯 FTS 底座，0 token）：
#:
#: | 口径 | 证据召回(删空白) | 证据召回(保留空白) | 变化的题 | p |
#: |---|---|---|---|---|
#: | A 改造前（绝对词频 + 页首硬截） | 0.2273 | 0.2273 | — | — |
#: | B 只改选块（密度归一 + 命中窗口） | **0.2614** | **0.2614** | 7/88 | 0.452 |
#: | C 再换成章节块 | 0.2045 | **0.1136** | 17/88 | 0.331 |
#:
#: B→C 方向为负（−0.0568，不显著），而**保留空白口径下腰斩**（0.2614→0.1136）——
#: 根因不是「章节 vs 页」这个单元本身，而是 `chunks` 表的文本是 pages 的**有损再推导**：
#: 同一篇论文 pages=24 而 chunks=20、pages=19 而 chunks=31，文本已不是原文逐字。
#: 这会直接打断本项目的机械回取校验（证据句要能在原文里逐字找到）。
#:
#: 所以：**在 chunk 文本的逐字保真修好之前，上下文继续按页组装。**
#: 检索侧的 chunk 命中信息已经完整接通（`RetrievalResult.chunk_hits`）并有测试，
#: 换单元只差把这个开关打开——但不能在数字变好之前打开。
#: 复现：`python cli.py qasper ctxab`
CONTEXT_UNIT = os.environ.get("PAPERNEST_CONTEXT_UNIT", "page").strip().lower()

#: 检索命中的块在密度排序上的加成。**是加成，不是闸门**——
#: 第一版把上下文限制成「只有 chunks_fts 命中的那 ≤4 块」，实测证据召回
#: 从 0.2614 掉到 0.1818：命中集之外的块再相关也进不来，而按密度扫全篇能捞到它们。
#: 检索信号该用来「排得更准」，不该用来「把别的候选删掉」。
#: 单个块窗口的目标字符数。QASPER 等预算扫描下片段召回在这里见顶
#: （800 字 0.0961；600 字 0.0915；400 字 0.0832）。`per_paper` 由它反推。
TARGET_WINDOW_CHARS = 800
#: 一篇最多取几个块。再多只是把上下文剁碎——问题召回还在涨，片段召回已经在掉。
MAX_CHUNKS_PER_PAPER = 3

HIT_BOOST = 1.6


def _rank_chunks(chunks: list[dict], hits: list[dict], terms: list[str],
                 per_paper: int) -> tuple[list[dict], bool]:
    """从**全部**章节块里挑 per_paper 个，检索命中的享加成。

    返回 (选中的块, 是否只能靠检索信号)。后者为 True 时说明问句在这篇正文里
    词面命中恒为 0（典型是中文问句 × 英文正文）——此时检索命中是唯一的信号。

    ## 为什么要 IDF：原来这里基本是在「按块长排序」

    `terms` 来自 `_query_terms`，它**故意不过滤停用词**，理由是「均匀出现的词对
    排序没有贡献、会自己抵消」。那句话对长度相近的页成立，对这里不成立：
    原来的打分是 `count / sqrt(len)`，长块的停用词计数随长度线性涨、分母只涨 sqrt。

    实测（真库 26 篇 / 692 块）：只喂十个停用词，选出的 50 个块里 **34 个（68%）
    是该篇最长的前 5 块**（随机约 21%）；英文查询的词频命中里**中位 94%**
    来自「出现在过半块里」的词（中文查询 0%）。

    ## 这个改动一度被判「无效」——那是评测集太小

    2026-09-09 第一次试时评测集只有 25 篇 / 69 题，`p=0.219`，按本仓口径没采用。
    把 QASPER dev 的 PDF 补齐后（274 篇 / **917 题** / 4556 条 gold），
    同一个改动 `p=5e-05`。**当时不是改动没用，是判不出来。**

        per=2 无 IDF（原始）             问题 0.1723  片段 0.0904
        per=3 无 IDF                    问题 0.1974  片段 0.0961
        per=3 + log(1+N/(1+df))         问题 0.2203  片段 0.1036   ← 形式选错的那版
        per=3 + BM25 式 IDF             问题 **0.2704**  片段 **0.1247**

    IDF 的**形式**也试错过一次：`log(1+N/(1+df))` 在 df=N 时仍有 log(2)≈0.69，
    压不住停用词的几百次重复，只拿到一半的收益（两者相差 p=5e-05）。
    """
    hit_nos = {h["chunk_no"] for h in hits}
    lowered = [(ch, (ch.get("text") or "").lower()) for ch in chunks]
    # IDF 在**候选集自身**上算：一个词若这篇的每个块都有，它就分不出块。
    # 不用语料级 IDF——这个函数的任务就是「在这一篇里选块」，本篇的文档频率
    # 才是对的归一化，而且不用额外扫全库。
    #
    # 形式用 BM25 的那一支。**这里试错过一次**：先写的是 `log(1 + N/(1+df))`，
    # 它在 df=N 时仍有 log(2)≈0.69，压不住停用词的几百次重复——
    # QASPER 917 题上只把问题召回从 0.1974 抬到 0.2203，而 BM25 式抬到 0.2704
    # （两者相差 p=5e-05，显著）。`log(N/df)` 截 0 的效果与 BM25 式打平
    # （0.2694 vs 0.2704），选后者是因为它不需要额外的截 0 补丁。
    n_docs = len(lowered) or 1
    idf = {}
    for k in {t.lower() for t in terms}:
        df = sum(1 for _, t in lowered if k in t)
        idf[k] = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
    scored = []
    for ch, t in lowered:
        raw = sum(t.count(k) for k in idf)          # 仍按原始命中判「词面全 0」
        weighted = sum(t.count(k) * idf[k] for k in idf)
        density = weighted / ((len(t) or 1) ** 0.5)
        if ch["chunk_no"] in hit_nos:
            density *= HIT_BOOST
        scored.append((density, raw, ch["chunk_no"], ch))
    if any(s[1] for s in scored):
        scored.sort(key=lambda x: (-x[0], x[2]))   # 全序：并列按 chunk_no 决胜
        return [s[3] for s in scored[:per_paper]], False
    # 词面全 0：只剩检索信号（向量那一路知道哪段语义上贴题）
    return hits[:per_paper], True


def _chunk_context(chunks: list[dict], hits: list[dict], terms: list[str],
                   per_paper: int, cap: int) -> tuple[str, bool]:
    """把章节块铺成上下文。

    这是「按章节切」这条结论真正落地的地方。此前 chunks 只影响「哪篇论文被召回」，
    组装上下文时一行都不读它——`_page_context` 拿 paper_id 回到 pages 表按词频
    重新猜一遍，等于把检索已经算出来的答案扔掉再猜一次。
    带上 section_path 与 start_page：引用要能溯源到章节，页码要能机械回取。
    """
    picked, lexical_blind = _rank_chunks(chunks, hits, terms, per_paper)
    out = []
    for h in picked:
        head = h.get("section_path") or "正文"
        page = h.get("start_page")
        loc = f"{head}·第 {page} 页" if page else head
        out.append(f"    【{loc}】{_window(h.get('text') or '', terms, cap // per_paper)}")
    return ("\n" + "\n".join(out) if out else ""), lexical_blind


def _paper_terms(c, paper_id: int) -> list[str]:
    """这篇论文自己的标题词与卡片关键词——跨语言兜底选页用。"""
    r = c.execute("SELECT title, card_json FROM papers WHERE id=?", (paper_id,)).fetchone()
    if not r:
        return []
    card = cards_safe(r["card_json"])
    kws = [k for k in (card.get("keywords") or []) if isinstance(k, str)]
    return _query_terms(" ".join([r["title"] or ""] + kws))


def _evidence_context(paper_id: int, q: str, hits: list[dict] | None,
                      per_paper: int, cap: int,
                      unit: str = "chunk") -> tuple[str, str | None]:
    """这一篇的证据块。**优先用检索真正命中的 chunk**，没有才退回按页猜。

    口径（会如实进 degraded）：
      None       —— 用了检索命中的章节块，或问句正常选出了页
      "fallback" —— 问句在这篇正文里一个词都没命中，退回按本篇关键词选页
      "miss"     —— 有全文却一页都没选出，只有摘要进上下文
    """
    if unit != "chunk":          # A/B 用：强制走改造前的「按页选块」口径
        return _page_context(paper_id, q, per_paper, cap, hits)
    with db.conn() as c:
        chunks = [dict(r) for r in c.execute(
            "SELECT chunk_no, section_path, start_page, text FROM chunks "
            # 参考文献块不是本篇的证据。这里必须单独挡一道：`_rank_chunks` 是直接
            # 从 chunks 表拉全部块按词频密度排的，不靠 FTS 命中，所以把它们排除出
            # chunks_fts 并不足以阻止它们被当成证据写进 LLM 上下文。
            "WHERE paper_id=? AND kind IS NOT 'reference' "
            "ORDER BY chunk_no", (paper_id,))]
    if chunks:
        block, blind = _chunk_context(chunks, hits or [], _query_terms(q),
                                      per_paper, cap)
        if block:
            # 词面全 0 时选出来的块只由检索信号决定——与按页选块的 "fallback"
            # 是同一类事实，同样要如实上报，不能因为换了单元就假装没降级。
            return block, ("fallback" if blind else None)
    return _page_context(paper_id, q, per_paper, cap, hits)


def _pages_from_hits(pages: dict[int, str], hits: list[dict] | None,
                     terms: list[str], per_paper: int, cap: int) -> list[str]:
    """用检索**语义**命中的块所在页来选页——跨语言时唯一有信号的那一路。

    `_pick_pages` 是纯字面 `t.count(k)`，中文问句在英文正文里恒为 0：真库实测
    45 篇有全文的论文 × 50 条中文问句，**98.5% 走 fallback**（同一批英文问句只有
    27.8%）。而 `search_hybrid` 早就把向量赢下的那一块取回来了（`chunk_hits` 里
    via='vector' 的项，`db.chunks_by_no` 带 `start_page`），`embeddings.py` 那句注释
    写得很清楚：「词面命中恒为 0 的跨语言场景下，这是唯一知道该喂哪段的信息来源」
    ——只是这条信号一直没传到按页组装这条路上。

    保持命中顺序（`_merge_hits` 已把语义命中排在词面命中前面），同一页只取一次。
    窗口仍围绕问句词面开：跨语言时词面为空，`_window` 自会退回页首。
    """
    if not hits:
        return []
    out, seen = [], set()
    for h in hits:
        pno = h.get("start_page")
        if pno is None or pno in seen or pno not in pages:
            continue
        seen.add(pno)
        out.append(f"    【第 {pno} 页】{_window(pages[pno] or '', terms, cap // per_paper)}")
        if len(out) >= per_paper:
            break
    return out


def _page_context(paper_id: int, q: str, per_paper: int = 2, cap: int = 2400,
                  hits: list[dict] | None = None) -> tuple[str, str | None]:
    """论文已有 L2 全文时，挑与问句最相关的几页，带页码进上下文。

    三级判据，依次退让：
      ① 问句词面选页（`_pick_pages`）→ 口径 None；
      ② 词面全 0 时用**检索的语义命中**定位页（`_pages_from_hits`）→ 口径同样是
         None：选出来的页确实是**因为这次提问**才被选中的，不是降级；
      ③ 都不行才退回按本篇自己的关键词选页 → ``"fallback"``，选出来的是「本篇核心」
         而不是「问句相关」，必须标注让上层如实上报；
      ④ 一页都没选出 → ``"miss"``。

    **不再静默**：原来 ②③④ 一律返回空串，调用方无从区分「这篇没有全文」和
    「有全文但一个词都没匹配上」。
    """
    with db.conn() as c:
        pages = {r["page_no"]: r["text"] for r in
                 c.execute("SELECT page_no,text FROM pages WHERE paper_id=?", (paper_id,))}
        if not pages:
            return "", None          # 这篇本来就没入 L2 全文，不是异常
        terms = _query_terms(q)
        blocks = _pick_pages(pages, terms, per_paper, cap)
        if blocks:
            return "\n" + "\n".join(blocks), None
        blocks = _pages_from_hits(pages, hits, terms, per_paper, cap)
        if blocks:
            return "\n" + "\n".join(blocks), None
        # 连语义命中都没有：退回按本篇关键词选页——选出来的是「本篇核心」
        # 而不是「问句相关」，所以必须标注，让上层如实上报。
        blocks = _pick_pages(pages, _paper_terms(c, paper_id), per_paper, cap)
    if blocks:
        return "\n" + "\n".join(blocks), "fallback"
    return "", "miss"


def _carry_over(session_id: str | None, question: str, ids: list[int],
                top_k: int) -> tuple[list[int], list[degrade.Degradation]]:
    """追问回指会话里的某个 [n] 时，把被指的论文拉回上下文并排在最前。

    「第 2 篇的方法讲细一点」——如果本轮检索没召回那篇，模型就只能瞎编或者装傻。

    编号按**会话全局**分配、落在 `chat_sessions.source_map_json`（`chat.assign_indices`），
    所以反查也必须查那张表。原来查的是 `chat.last_sources()`——只有最近一条有来源的
    assistant 消息里出现过的编号，中间隔一轮换话题 [2] 就查不到，且完全静默。

    pin 是**无条件插队**的，而回指的编号个数由用户输入决定（粘一段带 [1]..[99] 的
    相关工作就是几十个），所以必须封顶、并且至少给本轮检索留一个名额——否则
    「[1][2][3] 和 X 比起来怎么样」里的 X 一篇都进不了上下文。这三种失败
    （解析不到 / 被截断 / 把本轮检索挤空）都要如实上报，不许静默。
    """
    if not session_id:
        return ids, []
    wanted = chat.referenced_indices(question)
    if not wanted:                        # 没有回指：一次库都不用查
        return ids, []
    by_idx = chat.papers_by_index(session_id)
    pinned = list(dict.fromkeys(pid for i in wanted if (pid := by_idx.get(i))))
    if not pinned:
        # 会话里还一个编号都没有时不报：用户不可能见过任何 [n]，
        # 多半是把正文里的引文角标（「[12] 是什么意思」）抄了进来。
        if by_idx:
            return ids, [degrade.Degradation(
                degrade.REF_INDEX_UNRESOLVED,
                f"回指的编号 {wanted} 不在本会话的引用表里"
                f"（当前 1–{max(by_idx)}），未拉回任何文献")]
        return ids, []

    notes: list[degrade.Degradation] = []
    cap = max(1, top_k - 1)               # 至少给本轮检索留一个名额
    if len(pinned) > cap:
        notes.append(degrade.Degradation(
            degrade.REF_PIN_TRUNCATED,
            f"回指了 {len(pinned)} 篇，上下文只放得下 {cap} 篇，"
            f"其余未纳入"))
        pinned = pinned[:cap]
    merged = list(dict.fromkeys(pinned + list(ids)))
    out = merged[:max(top_k, len(pinned))]
    if ids and not set(out) & set(ids):
        notes.append(degrade.Degradation(
            degrade.REF_PIN_EVICTED,
            "本轮检索到的文献被回指的论文全部挤出上下文"))
    return out, notes


class PreparedContext(NamedTuple):
    """`prepare` 的返回值。

    比原来的 4 元组多一个 `notes`：`degraded` 那句中文是给人看的（前端 10 处消费点
    都按字符串渲染，契约不能动），`notes` 是同一批信号的结构化形式，用于落库、
    做监控聚合、以及判断有没有 CRITICAL 级降级。
    """

    system: str
    query: str
    sources: list[dict]
    degraded: str | None
    notes: list[degrade.Degradation]


def prepare(messages: list[dict], top_k: int = 5,
            candidate_ids: list[int] | None = None,
            session_id: str | None = None,
            max_context_chars: int = MAX_CONTEXT_CHARS,
            _ctx_unit: str | None = None
            ) -> PreparedContext:
    """检索 + 组装上下文，返回 `PreparedContext`。

    供流式回答使用：检索先行可立刻把来源推给前端，再逐 token 出正文。
    传 session_id 时启用服务端会话记忆：历史以库里的为准（前端可以不发），
    追问会把上文问题拼进检索查询，[n] 编号在整个会话内保持指向同一篇论文。
    """
    q = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
    turns = chat.history(session_id, 12) if session_id else []
    prior_qs = [t["content"] for t in turns if t["role"] == "user"]
    # 客户端历史只在没有会话时兜底（老前端仍能工作），有会话一律以服务端为准
    if not turns and len(messages) > 1:
        turns = [{"role": m.get("role", "user"), "content": m.get("content", "")}
                 for m in messages[:-1]][-12:]
        prior_qs = [t["content"] for t in turns if t["role"] == "user"]

    search_q, rewrite_note = chat.rewrite_query(q, prior_qs)
    if candidate_ids is None:
        ids, notes, chunk_hits = _vector_candidates(search_q, top_k)
        ids, ref_notes = _carry_over(session_id, q, ids, top_k)
        notes += ref_notes
    else:
        # 上游（agent / deep_answer / RCS 兜底）已经检索过并把候选传进来，这里不重复检索。
        # 上游那次检索自己的降级记录由上游负责上报，这里没有可报的。
        # **但块级命中要补一次**：不补的话这几条路径就退回按页猜，
        # 「按章节切」的收益只在直接问答那一条路上生效——同一个功能两条路两种行为。
        # 一次 chunks_fts 查询，0 token。
        ids, notes = list(dict.fromkeys(candidate_ids))[:top_k], []
        with db.conn() as c:
            hits = db.search_chunks_hits(c, search_q, max(top_k * 3, 15))
        chunk_hits = {p: h for p, h in hits.items() if p in set(ids)}

    numbering = chat.assign_indices(session_id, list(ids))
    # 会话编号只增不减（`source_map_json`），而渲染层与回指解析都只认 3 位数。
    # 越界不是「不好看」——引用角标和「[n]」回指会**同时**静默失效，所以要报出来。
    if numbering and max(numbering.values()) > chat.MAX_RENDERABLE_INDEX:
        notes.append(degrade.Degradation(
            degrade.REF_INDEX_OVERFLOW,
            f"本会话的来源编号已到 [{max(numbering.values())}]，超出渲染上限 "
            f"{chat.MAX_RENDERABLE_INDEX}：引用角标与「第 n 篇」回指会失效，建议新开一段对话"))
    blocks, sources, page_notes = [], [], []
    used, over_budget = 0, []
    # 页块配额按篇数摊。宁可让每篇的原文引用短一些，也不要整篇论文连同它的 [n]
    # 一起被预算挤掉——丢一篇是丢掉一条可引用的来源，缩一篇只是证据少一点。
    # 0.55 是页块能占的份额，其余留给标题 / TL;DR / key_findings / 摘要。
    page_cap = max(600, min(2400, int(max_context_chars * 0.55) // max(len(ids), 1)))
    # 每篇取几个块：**由窗口下限反推**，不拍常数。QASPER 上等预算扫描
    # （274 篇 / 917 题 / 4556 条 gold）：片段召回在每窗 800 字时见顶，
    #   per=1 窗2400 → 0.0935 | per=2 窗1200 → 0.0904 | **per=3 窗800 → 0.0961**
    #   per=4 窗600  → 0.0915 | per=6 窗400  → 0.0832
    # 问题召回一路涨到 per=6（0.2159），但那是把上下文剁成 400 字碎片换来的——
    # 证据总量反而在掉，而且短碎片更难支撑逐字引用。所以取「窗口不低于 800」这条线。
    per_paper = max(1, min(MAX_CHUNKS_PER_PAPER, page_cap // TARGET_WINDOW_CHARS))
    with db.conn() as c:
        for pid in ids:
            r = c.execute("SELECT * FROM papers WHERE id=?", (pid,)).fetchone()
            if not r:
                continue
            i = numbering.get(pid, len(sources) + 1)
            card = cards_safe(r["card_json"])
            lines = [f"[{i}] {r['title']}（{r['venue'] or ''} {r['year'] or ''}）"]
            if card.get("tldr"):
                lines.append(f"    TL;DR：{card['tldr']}")
            if card.get("key_findings"):
                for f in card["key_findings"][:4]:
                    lines.append(f"    结论（第{f.get('page','?')}页{'，已核验' if f.get('verified') else ''}）：{f.get('claim','')}")
            if r["abstract"]:
                lines.append(f"    摘要原句：{r['abstract'][:500]}")
            page_block, note = _evidence_context(pid, search_q,
                                                 chunk_hits.get(pid),
                                                 per_paper, page_cap,
                                                 unit=_ctx_unit or CONTEXT_UNIT)
            if page_block:
                lines.append(page_block)
            paper_block = "\n".join(lines)
            # 超预算的论文**同时**不进 blocks 和不进 sources：只丢其一的话，
            # sources 里会有一个上下文里没有对应块的 [n]，模型会去引用一个看不见的编号。
            # 首篇无条件保留——宁可超一点，也不能交出空上下文。
            if blocks and used + len(paper_block) > max_context_chars:
                # 记下**是哪几篇**，不只是几篇。编号在预算过滤之前就分配好了
                # （上面的 assign_indices），被挤掉的 [n] 会在 sources 里变成空洞，
                # 只报篇数的话用户和运维都对不上号、事后无法复盘。
                over_budget.append((i, r["title"]))
                continue
            used += len(paper_block)
            if note:
                page_notes.append(note)
            blocks.append(paper_block)
            sources.append({"idx": i, "paper_id": pid, "title": r["title"],
                            "year": r["year"], "venue": r["venue"]})
    sources.sort(key=lambda s: s["idx"])
    context = "\n\n".join(blocks) or "（库内没有检索到相关文献）"
    sys = SYSTEM
    hist_block = chat.history_block(turns)
    if hist_block:
        sys += f"\n\n之前的对话（用于理解指代，不是事实来源）：\n{hist_block}"
    sys += f"\n\n检索到的文献上下文：\n{context}"
    if over_budget:
        # 点名到具体哪几篇。标题截 24 字符、最多 3 篇、多于 3 篇补「等」——
        # 前端是原样渲染这句中文的自由文本，长度必须自己封死。
        # 前缀逐字保留，前端 10 处按字符串消费的地方与前缀匹配的脚本都不受影响。
        who = "、".join(f"[{n}]{(t or '')[:24]}" for n, t in over_budget[:3])
        notes.append(degrade.Degradation(
            degrade.CONTEXT_OVER_BUDGET,
            f"{len(over_budget)} 篇超出上下文预算（{max_context_chars} 字符）未纳入："
            f"{who}{' 等' if len(over_budget) > 3 else ''}"))
    if page_notes.count("fallback"):
        notes.append(degrade.Degradation(
            degrade.PAGE_PICK_FALLBACK,
            f"{page_notes.count('fallback')} 篇按本篇关键词选页"
            "（问句未命中其正文，常见于中文问句×英文论文）"))
    if page_notes.count("miss"):
        notes.append(degrade.Degradation(
            degrade.PAGE_PICK_MISS,
            f"{page_notes.count('miss')} 篇有全文却未命中任何页，仅摘要进上下文"))
    if rewrite_note:
        notes.append(degrade.Degradation(degrade.QUERY_REWRITTEN, rewrite_note))
    return PreparedContext(sys, q, sources, degrade.render(notes), notes)


def answer(messages: list[dict], top_k: int = 5,
           candidate_ids: list[int] | None = None,
           session_id: str | None = None) -> dict:
    """Answer a question from the local library.

    ``candidate_ids`` lets an orchestrator expose retrieval as a first-class
    tool and pass its result into the writer without a hidden second search.
    Existing callers keep the old behaviour.
    """
    if not llm.available():
        raise llm.LLMUnavailable("未配置 LLM（.env 的 LLM_API_BASE / LLM_API_KEY / LLM_MODEL）")
    ctx = prepare(messages, top_k, candidate_ids, session_id)
    resp = llm.chat(ctx.system, ctx.query, purpose="chat", temperature=0.4)
    if session_id:
        chat.append(session_id, "user", ctx.query)
        chat.append(session_id, "assistant", resp, ctx.sources,
                    ctx.degraded, ctx.notes)
    return {"answer": resp, "sources": ctx.sources, "degraded": ctx.degraded,
            "degraded_detail": degrade.as_dicts(ctx.notes),
            "session_id": session_id}


def cards_safe(raw):
    try:
        return json.loads(raw or "{}")
    except Exception:
        return {}
