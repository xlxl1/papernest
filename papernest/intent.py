"""意图分类：把用户一句话映射到工具路由。

把这一层从 agent.build_plan 里拆出来，是因为它有三件事需要独立成立：

1. **可以独立评测**——意图分类跑得准不准，跟后续工具跑得准不准是两件事，
   混在一起就是「agent 错了」一个归因，修都不知道往哪修。这里返回结构化的
   Intent 和 signals，tests/test_intent_routing.py 可以单独压。

2. **可以叠加兜底**——启发式是冷启动的种子（零 LLM 开销、可复现），但总会有
   边界 case。`PAPERNEST_INTENT=llm` 切到 LLM 兜底（轻档模型 + 严格 JSON 输出），
   启发式置信度低时升级。LLM 不可用就退回启发式，不破坏现有 LLMUnavailable 兜底。

3. **可以被 Replanner 复用**——agent.py 在每一步执行后调 replan.decide()，
   那一步也要看「这个 goal 当时被我分到了哪条意图，现在执行结果是不是符合预期」。
   把意图分类独立成纯函数，主循环和 Replanner 共用同一份认知，避免重复维护词表。

## 设计口径

- **领域词保守、动词确认**：原 build_plan 注释里讲过「'A Survey' 标题里的
  Survey 不算用户意图」。所以领域词只标「在谈论这件事」，动词才是「请你做这件事」。
  cite 用 cite 的动词、survey 用 survey 的动词，否则「推荐几篇 agent 综述」
  会被 survey 分支吃掉。
- **survey 触发刻意保守**：generate_survey 是全库 LLM 长文生成，误触发一次的
  代价远大于漏触发一次。所以必须在「领域词 + 动词」之外**额外**没有 cite 信号
  才认。
- **置信度是连续值，不是 0/1**：多个信号叠加加分；引号标题 / paper_id 这种
  硬信号直接 0.9+；默认路径 0.5。Replanner 据此决定是否要 ask_user。
"""

from __future__ import annotations

import os
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from . import llm
from .schemas import Intent  # API 契约；拆出来便于测试与跨模块复用

# ── 词表 ───────────────────────────────────────────────────────────────
# 抽出来不放在类属性里：模块级常量在 import 时构建一次，分类热路径不再触发属性查找。
# 同时让 tests/test_intent_routing.py 可以 `from . import intent` 然后直接覆盖
# 这几个元组做参数化测试。

# 领域词：用户「在谈论」这件事，但不等于「请你做这件事」。
# ASCII 词在 _match_words 里走词边界，避免 cite 含于 cited / cite 含于 excited
# 这类屈折形假命中；CJK 仍按子串，因为中文没有词边界概念。
_CITE_DOMAINS = ("引用", "参考文献", "引文", "citation", "cite", "citations",
                 "支持这段", "支持一下", "推荐文献", "推荐几篇", "找几篇", "找文献",
                 "段落", "这段话", "这段", "那段", "本段",
                 "paragraph", "this paragraph", "this section", "above", "below")
_SURVEY_DOMAINS = ("综述", "survey", "研究现状", "研究进展", "文献总结", "literature review")
_READ_DOMAINS = ("精读", "全文", "阅读论文", "阅读一下", "读一下", "read paper",
                 "详细分析", "细读")

# 动词：「请你做这件事」。两侧词边界更严格——recommend 含于 recommendation、
# add 含于 Addressing、find 含于 findings，库里的论文标题会自己变成祈使句。
_CITE_VERBS = ("推荐", "补充", "补上", "找", "找出", "给出", "标注",
               "加几篇", "加上",
               "recommend", "recommends", "suggest", "suggests", "find", "add")
_SURVEY_VERBS = ("写", "生成", "起草", "做一篇", "整理成", "综述一下", "总结一下",
                 "写一篇", "产出",
                 "generate", "write", "draft", "compose", "produce")

# ── 硬信号正则 ─────────────────────────────────────────────────────────
# 这些信号比关键词叠加更可靠：**只要出现，就强烈暗示某条意图**。

# 引号标题：「《A Survey of LLMs》」「"Attention is all you need"」
# 强 read_paper 信号——用户大概率想读这篇具体的论文。
_QUOTED_TITLE = re.compile(r"[《「“]([^》」”]{4,80})[》」”]")

# paper_id 提及：「精读论文 12」「read #3」「paper 7」
# 强 read_paper 信号。
_PAPER_ID = re.compile(r"(?:论文|paper|#)\s*([0-9]{1,4})", re.I)

# 指代编号：「[2]」「第 3 篇」——强 follow-up / cite 信号
# 引用 chat._ORDINAL_RE 的口径：3 位数，配套 chat.py 的渲染层。
_ORDINAL = re.compile(r"\[(\d{1,3})\]|第\s*([0-9一二三四五六七八九十]{1,3})\s*篇")

# 纯追问：句子里只剩疑问词 / 虚词
# 与 chat.py 的 _has_retrieval_signal 同口径，避免一个标"自足查询"一个标"追问"。
_FILLER_ONLY = re.compile(
    r"^\s*(?:为什么|怎么样|怎么做|怎么|如何|什么|哪些|讲讲|说说|展开|继续|详细|"
    r"一下|一点|对比一下|为什么呢|怎么做的|有什么区别|"
    r"(?:why|how|what|which|compare|more|detail|details|explain|again|continue)"
    r"|[的了吗呢吧啊呀嘛，。？！、,.?!\s])+\s*$", re.I)

def _match_words(text: str, words: tuple[str, ...], whole: bool) -> list[str]:
    """子串匹配；ASCII 走词边界。

    whole=False：领域词用，左边界卡掉 eli-cite-d 之类的「cite 只是词中间一段」
                 假命中（真库 pages 里有 elicited，chunks 里有 citeseer）。
    whole=True ：动词用，两侧都卡，不然 recommend 含于 recommendation。
    """
    hits: list[str] = []
    low = text.lower()
    for w in words:
        wl = w.lower()
        if wl.isascii():
            pat = r"(?<![a-z0-9])" + re.escape(wl) + (r"(?![a-z0-9])" if whole else "")
            if re.search(pat, low):
                hits.append(w)
        elif wl in low:
            hits.append(w)
    return hits


def _extract_paper_id(text: str) -> int | None:
    m = _PAPER_ID.search(text or "")
    return int(m.group(1)) if m else None


def _has_ordinal(text: str) -> bool:
    return bool(_ORDINAL.search(text or ""))


def _has_quoted_title(text: str) -> bool:
    return bool(_QUOTED_TITLE.search(text or ""))


def _is_pure_followup(text: str) -> bool:
    """句子里只有疑问词 / 虚词——典型追问，需要回拉历史。"""
    return bool(_FILLER_ONLY.match(text or ""))


def classify(goal: str) -> Intent:
    """纯启发式意图分类。

    故意不调 LLM——这是 fast-path，零开销、可复现、随叫随到。
    启发式信心不足时由 classify_with_llm_fallback 升级。
    """
    g = goal or ""
    signals: list[str] = []

    # 1. 硬信号：引号标题 + read 动词 → 强 read
    # 引号标题单独出现**不是** read 信号——「和《X》相关的文献」「《X》被引用
    # 了多少次」里 X 只是 topic anchor 或检索目标，不是「请你读这篇」。
    # 必须配合 read 动词（「读/精读/全文/详细分析」等）才认。这一点与原
    # build_plan 注释里讲过的「Survey 标题不算用户意图」是同一类陷阱——
    # 中文没有词边界，引号只能靠配对动词才能确认意图。
    quoted = _QUOTED_TITLE.search(g)
    if quoted:
        read_verbs_near = _match_words(g, _READ_DOMAINS, whole=False)
        if read_verbs_near:
            signals.append("quoted_title")
            signals += [f"read_domain:{w}" for w in read_verbs_near]
            pid = _extract_paper_id(g)
            return Intent(
                primary="read",
                confidence=0.9,
                signals=signals + ([f"paper_id={pid}"] if pid else []),
                paper_id=pid,
                topic_hint=quoted.group(1),
            )

    # 2. 硬信号：paper_id + read 域 → read
    pid = _extract_paper_id(g)
    read_hits = _match_words(g, _READ_DOMAINS, whole=False)
    if pid is not None and read_hits:
        signals.append("paper_id")
        signals += [f"read_domain:{w}" for w in read_hits]
        return Intent(primary="read", confidence=0.9,
                      signals=signals, paper_id=pid,
                      topic_hint=g[:200] if g else None)

    # 3. 硬信号：paper_id 单独（无 read 词）→ 也判 read，但置信度低一点
    if pid is not None and not read_hits:
        signals.append("paper_id_only")
        return Intent(primary="read", confidence=0.65,
                      signals=signals, paper_id=pid,
                      topic_hint=g[:200] if g else None)

    # 4. 强 cite：cite 域 + cite 动词
    cite_dom = _match_words(g, _CITE_DOMAINS, whole=False)
    cite_vb = _match_words(g, _CITE_VERBS, whole=True)
    if cite_dom and cite_vb:
        signals += [f"cite_domain:{w}" for w in cite_dom]
        signals += [f"cite_verb:{w}" for w in cite_vb]
        ordinal_bonus = 0.1 if _has_ordinal(g) else 0.0
        return Intent(primary="cite", confidence=min(0.85 + ordinal_bonus, 0.95),
                      signals=signals, topic_hint=g[:200] if g else None)

    # 5. survey：survey 域 + survey 动词（必须在 cite 之后，否则「推荐几篇 agent 综述」被吃掉）
    # 刻意保守：survey 域必须完全独立于 cite 域（前者是「写综述」语义、后者是
    # 「加引用」语义），不能是 cite 域词库的一部分。
    survey_dom = _match_words(g, _SURVEY_DOMAINS, whole=False)
    survey_vb = _match_words(g, _SURVEY_VERBS, whole=True)
    if survey_dom and survey_vb:
        signals += [f"survey_domain:{w}" for w in survey_dom]
        signals += [f"survey_verb:{w}" for w in survey_vb]
        return Intent(primary="survey", confidence=0.8,
                      signals=signals, topic_hint=g[:200] if g else None)

    # 6. 纯追问（句子里只剩疑问词）→ qa + 标记需要历史
    if _is_pure_followup(g):
        signals.append("pure_followup")
        return Intent(primary="qa", confidence=0.7,
                      signals=signals, topic_hint=None)

    # 7. 默认：qa
    signals.append("default")
    return Intent(primary="qa", confidence=0.5,
                  signals=signals, topic_hint=g[:200] if g else None)


# LLM 兜底：仅当启发式 confidence < 阈值时升级。
# 切到 LLM 的开关：PAPERNEST_INTENT=llm。不设就跟以前一样只用启发式。
_LLM_MODE = os.environ.get("PAPERNEST_INTENT", "").strip().lower() == "llm"
_LLM_FALLBACK_THRESHOLD = 0.7

_INTENT_SYSTEM = """你是论文库的意图路由器。把用户问题分到下面 5 类之一：
- cite       用户要把引用/参考文献加到一段文字里（「推荐几篇支持这段的文献」）
- survey     用户要基于库内文献生成一篇综述/总结（「写一篇 Agent 综述」）
- read       用户想精读某一篇具体的论文（「精读论文 12」「读一下《...》」）
- qa         用户要问答/检索某个主题（「LLM Agent 方法论」）
- unknown    完全无法判断，需要向用户澄清

只输出 JSON，字段：
  primary: 上述 5 类之一
  confidence: 0-1 浮点
  reason: 一句话说明判据
不要输出其它字段，不要解释。"""


def classify_with_llm_fallback(goal: str) -> Intent:
    """启发式为主；confidence 低且 LLM 可用时升级。

    LLMUnavailable（缺 key）一律退回启发式，不抛异常——这是与全仓一致的
    「LLM 不可用就当 mock 用」口径（见 llm.LLMUnavailable 的处理约定）。
    """
    fast = classify(goal)
    if not _LLM_MODE:
        return fast
    if fast.confidence >= _LLM_FALLBACK_THRESHOLD:
        return fast
    if not llm.available():
        return fast
    try:
        out = llm.chat(_INTENT_SYSTEM, f"用户问题：{goal}",
                       purpose="intent_classify", temperature=0.0).strip()
        parsed = llm.extract_json(out)
        primary = str(parsed.get("primary") or "").strip().lower()
        if primary not in {"cite", "survey", "read", "qa", "unknown"}:
            return fast
        conf = float(parsed.get("confidence") or 0.5)
        conf = max(0.0, min(1.0, conf))
        # LLM 给的置信度若还不如启发式，直接用启发式
        if conf <= fast.confidence:
            return fast
        return Intent(
            primary=primary,
            confidence=conf,
            signals=fast.signals + [f"llm:{parsed.get('reason', '')[:60]}"],
            paper_id=fast.paper_id,
            topic_hint=fast.topic_hint,
        )
    except (llm.LLMUnavailable, llm.LLMError, ValueError, KeyError):
        # 任何 LLM 兜底失败都退回启发式，不影响主流程
        return fast
