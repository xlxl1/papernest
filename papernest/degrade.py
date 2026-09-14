"""降级信号的结构化表示：一处定义、到处可查。

**为什么需要它**：项目里识别降级的能力一直不差（三态选页、向量空索引诊断、
上下文超预算计数都写得很细），但这些信号一律被 `f"{a}；{b}"` 拼成一句中文，
落进 `chat_messages.degraded` 这个 TEXT 列就没了。后果是：
- 没法回答「上周有多少次问答是在向量挂掉的状态下答的」这种运维问题；
- 没法按类型设闸门（「向量整路失败时不许出答案」和「有 1 篇超预算」不该同权）；
- 拼串是单向的，谁也没法从那句话里把类型解析回来。

这里给每种降级一个**稳定的 code**（英文、可 grep、可入库、可做聚合），
外加一句面向用户的中文 message。`render()` 产出的字符串与原来逐字一致——
前端 10 处消费点全是字符串，契约不能动，所以结构化只做**增量**。

用法：

    notes = []
    notes.append(Degradation(VECTOR_CALL_FAILED, "向量检索不可用（...），已降级 FTS"))
    degraded = render(notes)          # 给前端与旧接口的那句中文
    detail   = as_dicts(notes)        # 落库 / 给监控的结构化形式
"""
from __future__ import annotations

from typing import NamedTuple

# ── 降级类型。改这里就是改运维口径，所以集中定义、不许在调用点现编字符串。──

#: 配了 EMBED_MODEL，但一条向量都没匹配上（最常见成因：模型名与建索引时不一致）
VECTOR_INDEX_EMPTY = "vector_index_empty"
#: 向量检索抛异常（网络、维度不一致、端点 404……）
VECTOR_CALL_FAILED = "vector_call_failed"
#: 配置的向量后端不可用、或派生索引落后于真相，本次查询退回 SQLite 真相来源现算
VECTOR_BACKEND_DEGRADED = "vector_backend_degraded"
#: 检索到的论文因上下文预算被挤掉
CONTEXT_OVER_BUDGET = "context_over_budget"
#: 问句在该篇正文里一个词都没命中，退回按本篇自己的关键词选页
PAGE_PICK_FALLBACK = "page_pick_fallback"
#: 有全文却一页都没选出，只有摘要进上下文
PAGE_PICK_MISS = "page_pick_miss"
#: 追问被改写后才拿去检索
QUERY_REWRITTEN = "query_rewritten"

#: 用户回指的 [n] 不在本会话的编号表里，没有任何论文被拉回上下文
REF_INDEX_UNRESOLVED = "ref_index_unresolved"
#: 回指的论文太多，被截断了——用户点名的东西没能全进上下文
REF_PIN_TRUNCATED = "ref_pin_truncated"
#: 回指 pin 把本轮检索结果整体挤出了上下文
REF_PIN_EVICTED = "ref_pin_evicted"
#: 会话编号已经超出渲染层能认的位数（引用角标与 [n] 回指会同时失效）
REF_INDEX_OVERFLOW = "ref_index_overflow"
#: 章节树建索引时拿不到结构化章节，退回「一页一节」（章节路径成了「（第 N 页）」）
SECTION_TREE_PAGE_FALLBACK = "section_tree_page_fallback"
#: 流式输出中途断开，答案是半截的
STREAM_INTERRUPTED = "stream_interrupted"

#: 会让**回答本身不可信**的降级——适合当闸门，而不是只当提示。
#: 语义检索整路没生效时，召回质量的下降是系统性的（真库实测：纯 FTS 口径
#: Recall@5 0.6771 vs 混合 0.9427），与「少了一篇上下文」不是一个量级。
CRITICAL = frozenset({VECTOR_INDEX_EMPTY, VECTOR_CALL_FAILED,
                      STREAM_INTERRUPTED})


class Degradation(NamedTuple):
    """一条降级记录。`code` 给机器，`message` 给人。"""

    code: str
    message: str

    @property
    def critical(self) -> bool:
        return self.code in CRITICAL


def render(notes) -> str | None:
    """压成给前端/旧接口用的那句中文。空列表返回 None（与原来「无降级即 None」一致）。"""
    parts = [n.message for n in (notes or []) if n and n.message]
    return "；".join(parts) if parts else None


def as_dicts(notes) -> list[dict]:
    """结构化形式：落库、进 SSE、给监控做聚合。"""
    return [{"code": n.code, "message": n.message, "critical": n.critical}
            for n in (notes or []) if n]


def from_dicts(rows) -> list[Degradation]:
    """`as_dicts` 的逆运算。用于信号跨越 dict 边界（trace、SSE、库里的 JSON）后还原。"""
    return [Degradation(r.get("code") or "unknown", r.get("message") or "")
            for r in (rows or []) if isinstance(r, dict)]


def merge(*groups) -> list[Degradation]:
    """按 code 去重合并多批记录，保序。

    多段链路各自产生降级时（检索一段、组装上下文一段、迭代补检索又一段），
    同一个 code 报两遍只是噪声——用户要知道的是「向量挂了」，不是「向量挂了三次」。
    """
    out: list[Degradation] = []
    seen: set[str] = set()
    for g in groups:
        for n in (g or []):
            if n and n.code not in seen:
                seen.add(n.code)
                out.append(n)
    return out


def has_critical(notes) -> bool:
    return any(n.critical for n in (notes or []) if n)


def from_legacy(text: str | None) -> list[Degradation]:
    """把一句已经拼好的中文兜回结构化形式。

    给**还没改造完**的调用方留的过渡口径：宁可打上 `unknown` 也不要丢掉这条信号。
    新代码不要用它——直接构造 Degradation。
    """
    if not text:
        return []
    return [Degradation("unknown", part) for part in text.split("；") if part.strip()]
