"""对话会话的服务端记忆：历史落库 + 会话内引用编号稳定 + 追问改写。

原来的「多轮」是假的：前端确实把最近 12 条历史发上来了，但 rag.prepare 只取
最后一条 user 消息，其余全部丢弃，也不落库——于是「它和刚才那篇比呢」这类指代
必然失效，刷新页面历史就没了。这里把会话变成一等对象：

- 历史以服务端为准（前端可以不发），每轮的 user/assistant 与**该轮实际进上下文的文献**
  一起落库，对话可回放、可审计；
- 会话内 [n] 编号固定：第一轮里 [2] 是哪篇，第五轮里 [2] 还是那篇。
  否则用户说「第 2 篇讲得更细一点」时，模型看到的 [2] 早就换人了；
- 追问改写默认零成本（词面拼接上一轮问题），可切到 LLM 改写做对照实验。
"""
from __future__ import annotations

import json
import os
import re
import uuid
from typing import Any

from . import db, degrade

# 追问改写口径：heuristic（默认，0 token）| llm（调轻档模型）| off
REWRITE_MODE = os.environ.get("PAPERNEST_QUERY_REWRITE", "heuristic").strip().lower()

# 指代标记：命中即认为这句依赖上文，检索时要把上一轮问题拼进来。
# 中文按子串匹配没问题；**英文必须走词边界**——原来两者混在一个元组里做子串匹配，
# "it" 会命中 limitations / suite / critical / benchmark suite，
# 于是「多模态大模型的 limitations 有哪些」被判成追问，上一轮的问题被拼进检索查询，
# 上一轮主题的论文直接进本轮上下文。这是默认问答路径上的高频漂移，且完全静默。
_REF_MARKERS = (
    "它", "它们", "他们", "这篇", "那篇", "这些", "那些", "这个", "那个", "此文",
    "上面", "上述", "刚才", "前面", "之前", "继续", "再说", "展开", "详细讲",
    "为什么呢", "怎么做的", "有什么区别", "对比一下", "第一篇", "第二篇", "第三篇",
)
# 上面这张表里混着两类词，判定口径不能一样：
# **硬标记**是光杆代词与篇级指示词，出现就一定依赖上文，句子里还剩多少实词都不改变
# 这一点（「它的局限是什么」「这篇论文的方法是什么」）。「其它」不是代词，必须排除，
# 否则「还有其它 Massive MIMO 信道估计方法吗」被判成追问。
_HARD_REF = re.compile(
    r"(?<!其)它们?|他们|此文|这篇|那篇|上述|刚才"
    # 指示词**紧跟通用名词**时同样是光杆回指：「这些论文里哪个效果最好」
    # 「这个方法的复杂度呢」——剩下的实词（论文/效果、方法/复杂度）看着像检索信号，
    # 其实指向的是上一轮的东西。要求名词紧跟指示词（只允许空白），所以
    # 「这个 IP-MCMC-PF 目标跟踪方法…」不受影响：那里跟着的是一个具体型号。
    r"|[这那][些个]\s*(?:论文|文章|方法|模型|工作|研究|结果|实验|数据集)")
# **软标记**（其余的）既是指代词、又是常见学术实词/句式：「展开说说 RIS 辅助的宽带
# 信道估计」「对比一下 ZF 和 MMSE 检测器」「继续讲 SC-FDMA 的半盲信道估计」都是自足
# 查询。它们只能在句子本身**没有**检索信号时才算追问。表以 _REF_MARKERS 为唯一真源，
# 派生而来，不需要两处同步。
_SOFT_MARKERS = tuple(m for m in _REF_MARKERS if not _HARD_REF.fullmatch(m))
_ASCII_REF = re.compile(
    r"\b(it|its|they|them|these|those|the above|this paper|that paper)\b", re.I)

#: 疑问词与虚词。去掉它们之后还剩下实词，就说明这句自己带检索信号、能独立检索。
#: 这替代了原来的 `len(q) <= 12 一律判追问`——「近场信道估计的最新进展」11 个字，
#: 是一条完整的独立查询，却被当成追问拼上了上一轮的问题。
_FILLER = re.compile(
    r"为什么|怎么样|怎么做|怎么|如何|什么|哪些|讲讲|说说|展开|继续|详细|一下|一点|"
    r"\b(why|how|what|which|compare|more|detail|details|explain|again|continue|please)\b|"
    r"[的了吗呢吧啊呀嘛，。？！、,.?!\s]", re.I)
# 「第 3 篇」「[2]」这类明确回指某条来源的说法
#: 会话编号只增不减，上界是库内论文数（当前 503 篇）。原来这里和前端 7 处渲染
#: 正则都写的是 `\d{1,2}`——编号一过 99，引用渲染与「[n]」回指**同时静默失效**。
#: 放宽到 3 位（覆盖到 999）；再往上会被 `MAX_RENDERABLE_INDEX` 的降级如实报出来，
#: 不会像原来那样悄悄不认。不写 `\d+` 是为了不把正文里的年份「[2024]」当成引用。
_ORDINAL_RE = re.compile(r"\[(\d{1,3})\]|第\s*([0-9一二三四五六七八九十]{1,3})\s*篇")

#: 渲染层能认的最大编号（前端与上面的正则都是 3 位）。超过就不是「不好看」，
#: 而是引用角标和回指一起失效——必须上报，不能静默。
MAX_RENDERABLE_INDEX = 999
_CN_NUM = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
           "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


# ── 会话 CRUD ──

def ensure_session(session_id: str | None, first_message: str = "") -> str:
    """返回一个可用的 session_id；传 None 或未知 id 都会新建（不报错，对话不该被打断）。"""
    db.init_db()
    with db.conn() as c:
        if session_id:
            row = c.execute("SELECT id FROM chat_sessions WHERE id=?",
                            (session_id,)).fetchone()
            if row:
                return session_id
        new_id = session_id or uuid.uuid4().hex[:12]
        c.execute("INSERT OR IGNORE INTO chat_sessions(id,title) VALUES(?,?)",
                  (new_id, (first_message or "").strip()[:60]))
    return new_id


def append(session_id: str, role: str, content: str,
           sources: list[dict] | None = None, degraded: str | None = None,
           notes=None):
    """落一轮消息。`degraded` 是给人看的那句中文，`notes` 是同一批信号的结构化形式。

    两个都存：前端 10 处消费点按字符串渲染（契约不能动），而运维要的是能
    `GROUP BY code` 的那一份。`notes` 省略时按 `degraded` 兜成 code='unknown'——
    宁可标不准，也不要让还没改造完的调用方把信号整条丢掉。
    """
    if notes is None:
        notes = degrade.from_legacy(degraded)
    with db.conn() as c:
        c.execute(
            """INSERT INTO chat_messages(session_id,role,content,sources_json,
                                         degraded,degraded_json)
               VALUES(?,?,?,?,?,?)""",
            (session_id, role, content or "",
             json.dumps(sources or [], ensure_ascii=False), degraded,
             json.dumps(degrade.as_dicts(notes), ensure_ascii=False)))
        c.execute("UPDATE chat_sessions SET updated_at=datetime('now','localtime') "
                  "WHERE id=?", (session_id,))


def history(session_id: str, max_turns: int = 12) -> list[dict[str, Any]]:
    """取最近 max_turns 条消息（按时间正序返回，方便直接铺进 prompt）。

    带上 degraded：这条信号本来就落了库，但一直没被读出来——会话回放里
    「这轮是在降级状态下答的」看不见，前端那行 ⚠ 渲染永远是空的。
    """
    with db.conn() as c:
        rows = c.execute(
            """SELECT role, content, sources_json, degraded, degraded_json, created_at
               FROM chat_messages
               WHERE session_id=? ORDER BY id DESC LIMIT ?""",
            (session_id, max_turns)).fetchall()
    out = []
    for r in reversed(rows):
        out.append({"role": r["role"], "content": r["content"],
                    "sources": json.loads(r["sources_json"] or "[]"),
                    "degraded": r["degraded"],
                    "degraded_detail": json.loads(r["degraded_json"] or "[]"),
                    "created_at": r["created_at"]})
    return out


def last_sources(session_id: str) -> list[dict]:
    """上一轮回答实际用到的来源（会话回放 / 调试用）。

    **不要拿它反查 [n]**：编号是会话全局的（`source_map_json`），这里只看得见
    最近一条有来源的 assistant 消息里出现过的那几个。回指要用 `papers_by_index`。
    """
    with db.conn() as c:
        row = c.execute(
            """SELECT sources_json FROM chat_messages
               WHERE session_id=? AND role='assistant' AND sources_json != '[]'
               ORDER BY id DESC LIMIT 1""", (session_id,)).fetchone()
    return json.loads(row["sources_json"]) if row else []


def list_sessions(limit: int = 30) -> list[dict]:
    db.init_db()
    with db.conn() as c:
        rows = c.execute(
            """SELECT s.id, s.title, s.created_at, s.updated_at,
                      (SELECT COUNT(*) FROM chat_messages m WHERE m.session_id=s.id) n
               FROM chat_sessions s ORDER BY s.updated_at DESC LIMIT ?""",
            (limit,)).fetchall()
    return [dict(r) for r in rows]


def get_session(session_id: str) -> dict | None:
    with db.conn() as c:
        row = c.execute("SELECT * FROM chat_sessions WHERE id=?",
                        (session_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["source_map"] = json.loads(d.pop("source_map_json") or "{}")
    d["messages"] = history(session_id, 200)
    return d


def delete_session(session_id: str) -> bool:
    with db.conn() as c:
        cur = c.execute("DELETE FROM chat_sessions WHERE id=?", (session_id,))
        return cur.rowcount > 0


# ── 会话内稳定编号 ──

def _load_map(row) -> dict[int, int]:
    """把 source_map_json 解析成 {paper_id: 编号}。两处读取共用同一口径。"""
    raw = (row["source_map_json"] if row else "{}") or "{}"
    return {int(k): int(v) for k, v in json.loads(raw).items()}


def assign_indices(session_id: str | None, paper_ids: list[int]) -> dict[int, int]:
    """给本轮文献分配 [n]：会话里出现过的沿用旧号，新文献顺延。

    不这么做的话，第 1 轮的 [2] 和第 3 轮的 [2] 可能是两篇不同的论文，
    而用户和模型都在按编号指代——错位的引用比没有引用更糟。
    """
    if not session_id:
        return {pid: i for i, pid in enumerate(paper_ids, 1)}
    # 先用**普通读连接**看一眼：绝大多数轮次里这些论文早就有编号了，什么都不用写。
    # 无条件走 BEGIN IMMEDIATE 会让这些只读调用也去抢 SQLite 写锁——而它在
    # `rag.prepare` 的必经路径上，一旦有长写事务（如 fulltext 在写事务里重解析 PDF，
    # 实测 26 页约 0.87s）就要排队，最坏等满 BUSY_TIMEOUT_S=30s 抛 OperationalError，
    # api.py 再把它变成 500「检索失败」。修 lost update 是必要的，但闸门不该开这么宽。
    with db.conn() as c:
        row = c.execute("SELECT source_map_json FROM chat_sessions WHERE id=?",
                        (session_id,)).fetchone()
        mapping = _load_map(row)
    if all(pid in mapping for pid in paper_ids):
        return {pid: mapping[pid] for pid in paper_ids}

    # 确有新论文要编号才升级成写事务。**事务内必须重读** mapping：
    # 上面那次读是在锁外做的，期间别人可能已经改过。
    with db.conn(immediate=True) as c:
        row = c.execute("SELECT source_map_json FROM chat_sessions WHERE id=?",
                        (session_id,)).fetchone()
        mapping = _load_map(row)
        nxt = max(mapping.values(), default=0) + 1
        changed = False
        for pid in paper_ids:
            if pid not in mapping:
                mapping[pid] = nxt
                nxt += 1
                changed = True
        if changed:
            c.execute("UPDATE chat_sessions SET source_map_json=? WHERE id=?",
                      (json.dumps({str(k): v for k, v in mapping.items()}), session_id))
    return {pid: mapping[pid] for pid in paper_ids}


def papers_by_index(session_id: str | None) -> dict[int, int]:
    """会话内 [n] → paper_id 的反查：`assign_indices` 的读取侧。

    `source_map_json` 存的是 {paper_id: 编号}，而用户说「第 2 篇」时要的是反方向。
    编号是**会话全局**的，所以回指只能查这张表——查 `last_sources()` 只看得见
    最近一轮出现过的编号，中间隔一轮换话题就查不到（合成多轮会话上实测：
    第 3 轮起平均只剩 34.8% 的编号还解析得到，且完全静默）。
    """
    if not session_id:
        return {}
    with db.conn() as c:
        row = c.execute("SELECT source_map_json FROM chat_sessions WHERE id=?",
                        (session_id,)).fetchone()
    return {idx: pid for pid, idx in _load_map(row).items()}


# ── 追问识别与检索查询改写 ──

def _has_retrieval_signal(q: str) -> bool:
    """去掉疑问词和虚词之后还剩下实词 → 这句自己就能检索，不该被拼上上一轮的问题。"""
    return len(_FILLER.sub("", q or "").strip()) >= 3


def is_followup(question: str, prior_user_turns: list[str]) -> bool:
    q = (question or "").strip()
    if not prior_user_turns or not q:
        return False
    low = q.lower()
    # 1) 显式回指某条来源（「[3] 这篇」「第 2 篇」）：无条件算追问。
    #    漏判这一类最贵——用户点了名，检索却按字面去找「第 2 篇」。
    if referenced_indices(q):
        return True
    # 2) 光杆代词 / 英文代词：同样无条件，句子里还剩多少实词都不改变它依赖上文这件事。
    if _HARD_REF.search(low) or _ASCII_REF.search(low):
        return True
    # 3) 软标记必须**让位给检索信号**：把标记本身抠掉，剩下的句子还能独立检索
    #    （「对比一下 ZF 和 MMSE 检测器」去掉「对比一下」还剩两个型号）就不是追问。
    #    原来这里是「标记命中即 return True」，把下面这条判断整个短路了——
    #    上面第 39 行注释声称检索信号判断替代了 `len(q)<=12` 的一刀切，
    #    但它对**带标记的句子根本不生效**。真库实测 14 条自足查询误判 8 条（57%），
    #    其中两条被拼上上一轮问题后 top10 存活率 0/10：整个候选集换成了上一轮的主题。
    rest = low
    for m in _SOFT_MARKERS:
        if m in rest:
            rest = rest.replace(m, " ")
    # 只有「去掉疑问词和虚词后什么都不剩」的句子才算追问（「为什么？」「展开讲讲」「why?」）。
    # 判错的代价是不对称的：误判追问会把好好的查询整体换成上一轮的主题、还自信地引
    # [n]；漏判时历史块仍在 system prompt 里（模型能自己解指代），显式 [n] 也另有
    # `rag._carry_over` 兜底，且检索 0 行是**响的**失败。所以偏向「宁可不当追问」。
    return not _has_retrieval_signal(rest)


def referenced_indices(question: str) -> list[int]:
    """从「第 2 篇」「[3]」里解析出用户回指的来源编号。"""
    out = []
    for m in _ORDINAL_RE.finditer(question or ""):
        if m.group(1):
            out.append(int(m.group(1)))
        elif m.group(2):
            token = m.group(2)
            out.append(int(token) if token.isdigit() else _CN_NUM.get(token, 0))
    return [i for i in out if i > 0]


def rewrite_query(question: str, prior_user_turns: list[str]) -> tuple[str, str | None]:
    """返回 (用于检索的查询, 改写说明)。改写说明为 None 表示原样使用。

    heuristic 口径把最近两轮用户问题拼在当前问题前面——纯词面拼接，0 token，
    对「它的局限是什么」这类空查询效果立竿见影。想做口径对照就切 llm。
    """
    q = (question or "").strip()
    if REWRITE_MODE == "off" or not is_followup(q, prior_user_turns):
        return q, None
    if REWRITE_MODE == "llm":
        rewritten = _llm_rewrite(q, prior_user_turns)
        if rewritten:
            return rewritten, f"追问改写（LLM）：{rewritten[:60]}"
    context = " ".join(prior_user_turns[-2:])
    merged = f"{context} {q}".strip()
    return merged, "追问改写（词面拼接上文问题，0 token）"


def _llm_rewrite(question: str, prior_user_turns: list[str]) -> str | None:
    from . import llm
    if not llm.available():
        return None
    system = ("把用户的追问改写成一个不依赖上文、可独立检索的完整问题。"
              "只输出改写后的问题本身，不要解释，不要加引号。")
    user = ("对话中之前的问题：\n- " + "\n- ".join(prior_user_turns[-3:])
            + f"\n\n用户现在的追问：{question}")
    try:
        out = llm.chat(system, user, purpose="chat_rewrite", temperature=0.0).strip()
    except Exception:
        return None                        # 改写失败不该让整轮对话失败，退回词面拼接
    return out.splitlines()[0][:200] if out else None


def _shorten_turn(role: str, text: str, head: int = 200, tail: int = 120) -> str:
    """助手回答**两端保留**；用户消息整条保留（截到 400）。

    助手的免责与限定句——「但库内文献未覆盖 X」「该结论未经核验」——按 rag.SYSTEM
    的要求几乎总在结尾，原来 `text[:400]` 硬截把它们丢掉，**留下的恰好是最自信的
    前半段**，随后又以「之前的对话」的身份进 system prompt。用户消息本来就短，
    而且是约束（「只看 2023 年之后」）的载体，不该被两端截。
    """
    if role == "用户":
        return text[:400]
    if len(text) <= head + tail:
        return text
    return f"{text[:head]}……（中略）……{text[-tail:]}"


def history_block(turns: list[dict], max_chars: int = 1800) -> str:
    """把历史压成 prompt 里的一段。越久远的越靠前。

    超长时**先丢最早的助手行**，用户行留到最后再丢：用户在第一轮立下的约束
    原来是最先被 `pop(0)` 丢掉的，而它恰恰是整段历史里最该留住的东西。
    """
    items: list[tuple[str, str]] = []
    for t in turns:
        role = "用户" if t["role"] == "user" else "助手"
        text = re.sub(r"\s+", " ", t["content"] or "").strip()
        if text:
            items.append((role, _shorten_turn(role, text)))

    def render(rows):
        return "\n".join(f"{r}：{x}" for r, x in rows)

    block = render(items)
    while len(block) > max_chars and items:
        drop = next((i for i, (r, _x) in enumerate(items) if r == "助手"), 0)
        items.pop(drop)
        block = render(items)
    return block
