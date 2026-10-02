"""Replanner：每步执行后重新决策下一步是什么。

## 为什么需要这一层

agent.build_plan 是**一次性**的：拿到用户一句话就拍板一串步骤，然后一条道走到黑。
问题在于：

- 意图是「猜」的——「A Survey」标题里的 Survey 不算用户意图，注释里花了大段讲这件事。
  即便把词表维护到极致，启发式仍然有边界 case。
- 工具结果是「事后才知道」的——recommend_citations 是不是 0 verified、retrieve_library
  是不是空、answer_question 是不是回「库内文献未覆盖」，这些**只能跑完才知道**，
  而一拍板的 plan 没法拿这些信号回头。
- 用户没说清的事「装作听清了」——现有 build_plan 只能 fall through 到默认的
  retrieve+answer，输出一个看似自信但其实答非所问的回答。

Replanner 把这件事拆成两步：**先拍板一个种子 plan，每步跑完看结果再决定下一步**。
种子 plan 还是 build_plan()——它的零开销、可复现、可单测是宝贵的，别浪费。

## 决策空间

- CONTINUE  ：当前 plan 还能继续，下一步照原计划执行
- REVISE    ：当前 plan 有问题，改 plan（在尾部加步骤 / 替换某个步骤 / 删步骤）
- ABORT     ：整个 plan 都不对，整盘重新规划
- ASK_USER  ：意图或参数真的搞不清，向用户要澄清
- FINISH    ：已经拿到用户想要的答案，提前结束
- FAIL      ：执行不下去（超时 / 配置缺失 / 工具反复失败），返回失败

## 默认零 LLM 调用

所有规则都是结构化信号判定：不调 LLM、不读工具实现细节、只看工具返回 dict。
`PAPERNEST_REPLANNER=llm` 时升级到 LLM 兜底，但**默认关闭**——多一次 LLM 调用
的代价（200ms+ 延迟 + token 费用）在启发式已经能处理 95% case 的情况下不值。

## 决策顺序（重要）

规则之间不是并列的，**优先级决定行为**：

  0.   意图置信度极低     → ASK_USER（执行前）
  1.   已生成答案         → FINISH（早停）
  2.   累计 timeout ≥ 2  → FAIL（别再 ask 造成死循环）
  3.   单次 timeout/fail  → CONTINUE（让工具层重试，不抢意图层的活）
  4.   retrieve 空 + 非 read → ASK_USER（可能是关键词不对）
  5.   answer 未覆盖 + survey 域 → REVISE 升 survey
  6.   cite 全 unverified → REVISE 升 survey
  7.   默认              → CONTINUE

为什么 2/3 优先于 4：超时已经表明这条路径在工具层有问题，再去问用户「要不要换
关键词」是错位——用户也救不了模型超时。timeout/fail 应该先收敛。

## 可测

decide() 是纯函数：相同 state → 相同 decision。tests/test_intent_routing.py
可以构造假 state 压回归，不用真跑工具。
"""

from __future__ import annotations

import os
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from . import intent as intent_mod
from . import llm
from .schemas import ClarificationQuestion, PlanStep
from .schemas import PlanStep

# 切换：PAPERNEST_REPLANNER=llm 时使用 LLM 兜底；默认规则优先
_LLM_MODE = os.environ.get("PAPERNEST_REPLANNER", "").strip().lower() == "llm"

# 置信度阈值：低于这个值的 intent 在执行前就应该 ASK_USER。
# 故意设得很低——0.5 是 default qa 路径的「正常」置信度，常见的「LLM Agent 方法论」
# 这种问题就会卡 0.5，太激进会把所有 qa 都拽到澄清。
# 只在「真的不知道」（primary=unknown）或非常冷门（<0.4）时才主动 ask。
_LOW_CONFIDENCE = 0.4



class ReplanDecision(BaseModel):
    action: Literal["continue", "revise", "abort", "ask_user", "finish", "fail"]
    reason: str
    new_steps: list[PlanStep] = Field(default_factory=list)
    questions: list[ClarificationQuestion] = Field(default_factory=list)
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


def _build_steps_for(intent_obj: intent_mod.Intent, goal: str, top_k: int) -> list[PlanStep]:
    """把 Intent 翻译成具体的 PlanStep 序列。

    与 build_plan 同形但分开：build_plan 是冷启动种子，
    这里支持运行时根据 intent 重建步骤（Abort 时用）。
    """
    p = intent_obj.primary
    if p == "cite":
        return [PlanStep(
            id=1, tool="retrieve_library",
            reason="先召回候选，再喂给引用推荐。",
            args={"query": goal, "top_k": top_k},
        ), PlanStep(
            id=2, tool="recommend_citations",
            reason="基于召回结果匹配可支撑段落的文献。",
            args={"paragraph": goal, "top_k": top_k},
        )]
    if p == "survey":
        return [PlanStep(
            id=1, tool="retrieve_library",
            reason="先召回综述主题相关的候选论文。",
            args={"query": goal, "top_k": max(top_k, 8)},
        ), PlanStep(
            id=2, tool="generate_survey",
            reason="基于召回结果生成带引用结构化综述。",
            args={"topic": goal, "top_k": max(top_k, 8)},
        )]
    if p == "read" and intent_obj.paper_id is not None:
        return [PlanStep(
            id=1, tool="read_paper",
            reason=f"按 paper_id 精读论文 #{intent_obj.paper_id}。",
            args={"paper_id": intent_obj.paper_id},
        )]
    if p == "read":
        # read 意图但没 paper_id：用 retrieve 找标题最接近的那篇
        topic = intent_obj.topic_hint or goal
        return [PlanStep(
            id=1, tool="retrieve_library",
            reason="read 意图但未指定 paper_id，先按标题/主题召回候选。",
            args={"query": topic, "top_k": max(3, min(top_k, 8))},
        ), PlanStep(
            id=2, tool="answer_question",
            reason="让 LLM 在召回里指出最匹配的那篇并精读。",
            args={"goal": goal, "top_k": max(3, min(top_k, 8))},
        )]
    return [PlanStep(
        id=1, tool="retrieve_library",
        reason="按用户问题召回相关候选。",
        args={"query": goal, "top_k": top_k},
    ), PlanStep(
        id=2, tool="answer_question",
        reason="基于召回结果生成带来源回答。",
        args={"goal": goal, "top_k": top_k},
    )]


def _retrieval_is_empty(observation: dict) -> bool:
    """retrieve_library 返回 0 论文，且没降级（降级可能是临时问题）。"""
    if not isinstance(observation, dict):
        return False
    papers = observation.get("papers") or observation.get("paper_ids") or []
    if papers:
        return False
    if observation.get("degraded") or observation.get("degraded_detail"):
        return False
    return True


def _answer_says_no_coverage(observation: dict) -> bool:
    """answer_question 回了「库内文献未覆盖」之类的话。

    不靠「未覆盖」字符串做精确匹配——改个文案就破。改成判 answer 里**没有**
    任何 [n] 引用编号（无引用支撑的回答通常就是这种兜底）。
    """
    if not isinstance(observation, dict):
        return False
    answer = observation.get("answer") or ""
    sources = observation.get("sources") or []
    if sources:
        return False
    if re.search(r"\[\d{1,3}\]", answer):
        return False
    if len(answer.strip()) < 40:
        return True
    return True


def _cites_all_unverified(observation: dict) -> bool:
    if not isinstance(observation, dict):
        return False
    cands = observation.get("candidates") or []
    if not cands:
        return True
    return all(not c.get("verified") for c in cands)


def decide(state: dict[str, Any]) -> ReplanDecision:
    """主入口：拿当前 state，返回下一步怎么走。

    state 字段约定（agent.py 负责维护）：
      - intent: intent_mod.Intent（首次分类）
      - steps_done: list[(PlanStep, observation_dict)]
      - last_observation: dict | None
      - last_step: PlanStep | None
      - last_status: "completed" | "failed" | "timeout" | ...
      - has_answer: bool
      - timeout_count: int
      - failed_tools: list[str]
      - plan_remaining: int  # 还有多少步骤没跑；rule 4/5/6 据此判定是否「终端步骤」

    故意做成纯函数：相同 state → 相同 decision，好测。
    """
    intent_obj: intent_mod.Intent = state.get("intent")
    last_obs: dict = state.get("last_observation") or {}
    last_step = state.get("last_step")
    last_status = state.get("last_status", "completed")
    steps_done = state.get("steps_done") or []
    has_answer = state.get("has_answer", False)
    timeout_count = state.get("timeout_count", 0)
    goal = state.get("goal", "")
    top_k = state.get("top_k", 5)

    # ── 规则 0：执行前的意图信心检查 ──
    # 仅当 primary=unknown（启发式明确放弃）或 confidence 极低时才主动问。
    # 「default qa 路径」置信度就是 0.5，不是低，是基线。
    if (not steps_done and intent_obj
            and (intent_obj.primary == "unknown"
                 or intent_obj.confidence < _LOW_CONFIDENCE)):
        return ReplanDecision(
            action="ask_user",
            reason=f"意图分类置信度 {intent_obj.confidence:.2f} 偏低，"
                   f"信号 {intent_obj.signals}。",
            questions=[ClarificationQuestion(
                qid="intent",
                question="你想做什么？",
                options=[
                    {"label": "推荐引用", "value": "cite"},
                    {"label": "写综述", "value": "survey"},
                    {"label": "精读某篇", "value": "read"},
                    {"label": "检索/问答", "value": "qa"},
                ],
            )],
            confidence=1.0,
        )

    # ── 规则 1：已经拿到答案就早停 ──
    if has_answer and last_status == "completed":
        return ReplanDecision(action="finish", reason="已生成回答，提前结束。",
                              confidence=1.0)

    # ── 规则 2：累计超时 → fail（**优先于 ask**，避免 ask-fail-ask 死循环）──
    if timeout_count >= 2:
        return ReplanDecision(
            action="fail",
            reason=f"累计 {timeout_count} 次步骤超时，停止重试避免继续烧资源。",
            confidence=1.0,
        )

    # ── 规则 3：单次超时/失败 → continue（让工具层重试/降级）──
    if last_status in ("failed", "timeout"):
        return ReplanDecision(
            action="continue",
            reason=f"上一步 {last_status}，意图层不调整，按原 plan 继续。",
            confidence=0.6,
        )

    # ── 规则 4：retrieve_library 空召回 + 非 read 意图 → ASK_USER ──
    # 只在 last_status == completed 时触发——超时/失败时空召回可能是工具问题，
    # 不该让用户背锅。
    # **plan_remaining == 0 闸门**：retrieve 是 plan 的**最后一步**时空召回才问；
    # 如果后面还有 answer_question 等步骤，让它们自己处理空召回。
    # 这避免「retrieve 空 → 直接问用户」打断正常的「retrieve + answer」流水线
    # （answer_question 自己会回「库内未覆盖」，再走 rule 5/6 升级）。
    plan_remaining = state.get("plan_remaining", 0)
    if (last_status == "completed"
            and plan_remaining == 0
            and last_step and last_step.tool == "retrieve_library"
            and _retrieval_is_empty(last_obs)
            and intent_obj and intent_obj.primary != "read"):
        return ReplanDecision(
            action="ask_user",
            reason=f"按当前问题在库里没召回任何论文（intent={intent_obj.primary}）。"
                   f"可能是主题过窄、关键词不对，或者你想做的是别的。",
            questions=[ClarificationQuestion(
                qid="no_match",
                question="库里没找到相关论文，你想怎么继续？",
                options=[
                    {"label": "换个关键词重试", "value": "retry_rephrase"},
                    {"label": "基于全库做综述", "value": "survey_all"},
                    {"label": "我自己确认下", "value": "abort"},
                ],
            )],
            confidence=0.9,
        )

    # ── 规则 5：answer_question 兜底 + survey 域词 → REVISE 到 survey ──
    # 用户问「LLM Agent 综述」，启发式可能没识别出 survey（缺动词），
    # 跑了 retrieve+answer 拿到空回答，自动升级到 survey 救场。
    if (last_status == "completed"
            and last_step and last_step.tool == "answer_question"
            and _answer_says_no_coverage(last_obs)
            and intent_obj and intent_obj.primary != "survey"):
        # 直接看 goal 里有没有 survey 域词，不强求动词——「LLM Agent 综述」
        # 这种没有动词、但用户提到综述的句子，默认 qa 拿不到结果时，
        # 给一次 survey 救场比直接报「库内未覆盖」要好。
        probe = intent_mod.classify(goal)
        probe_dom = intent_mod._match_words(
            goal, intent_mod._SURVEY_DOMAINS, whole=False)
        if probe.primary == "survey" or probe_dom:
            return ReplanDecision(
                action="revise",
                reason="默认 qa 路径没覆盖，goal 含 survey 域词，升级到 survey。",
                new_steps=[PlanStep(
                    id=len(steps_done) + 1, tool="generate_survey",
                    reason="默认回答未覆盖，升级到综述生成（基于已召回结果）。",
                    args={"topic": goal, "top_k": max(top_k, 8)},
                )],
                confidence=0.8,
            )

    # ── 规则 6：recommend_citations 全 unverified → REVISE 到 survey ──
    if (last_status == "completed"
            and last_step and last_step.tool == "recommend_citations"
            and (last_obs.get("candidates") or [])
            and _cites_all_unverified(last_obs)):
        return ReplanDecision(
            action="revise",
            reason="引用候选全部未通过证据校验，库内可能没有强支撑的文献，"
                   "升级为基于全库的综述。",
            new_steps=[PlanStep(
                id=len(steps_done) + 1, tool="generate_survey",
                reason="引用推荐失败兜底：基于库内文献生成综述。",
                args={"topic": goal, "top_k": max(top_k, 8)},
            )],
            confidence=0.7,
        )

    # ── 默认：照原 plan 继续 ──
    return ReplanDecision(action="continue", reason="按原 plan 继续。", confidence=1.0)


def revise(state: dict[str, Any]) -> ReplanDecision:
    """Abort 后的整盘重规划：拿当前 state，从零再生成 plan。"""
    intent_obj: intent_mod.Intent = (state.get("intent")
                                     or intent_mod.classify(state.get("goal", "")))
    if intent_obj.confidence < _LOW_CONFIDENCE:
        return decide(state)
    top_k = state.get("top_k", 5)
    return ReplanDecision(
        action="abort",
        reason=f"按 intent={intent_obj.primary} 重新生成步骤。",
        new_steps=_build_steps_for(intent_obj, state.get("goal", ""), top_k),
        confidence=0.9,
    )


def decide_with_llm_fallback(state: dict[str, Any]) -> ReplanDecision:
    """规则版为主；置信度低时升级到 LLM。

    与 intent.classify_with_llm_fallback 同样的口径：
    PAPERNEST_REPLANNER=llm 才升级；LLM 失败一律退回规则。
    """
    fast = decide(state)
    if not _LLM_MODE:
        return fast
    if fast.confidence >= 0.7:
        return fast
    if fast.action in ("ask_user", "fail"):
        return fast
    if not llm.available():
        return fast
    try:
        sys_prompt = (
            "你是论文库 Agent 的 Replanner。基于当前 state 决定下一步：\n"
            "- continue: 继续原 plan\n"
            "- revise: 加一步（说明 tool 与 reason）\n"
            "- ask_user: 需要澄清（说明问题）\n"
            "- fail: 失败\n"
            "只输出 JSON：{action, reason, confidence, new_step_tool?, new_step_reason?}"
        )
        user = (
            f"用户目标：{state.get('goal', '')[:200]}\n"
            f"意图：{state.get('intent').primary if state.get('intent') else 'unknown'}"
            f" (conf={state.get('intent').confidence if state.get('intent') else 0})\n"
            f"已完成步骤数：{len(state.get('steps_done') or [])}\n"
            f"上一步：{state.get('last_step').tool if state.get('last_step') else 'none'}"
            f" status={state.get('last_status')}\n"
            f"上一步返回：{str(state.get('last_observation') or {})[:300]}"
        )
        out = llm.chat(sys_prompt, user, purpose="replan", temperature=0.0).strip()
        parsed = llm.extract_json(out)
        action = str(parsed.get("action") or "").strip().lower()
        if action not in {"continue", "revise", "ask_user", "fail", "finish"}:
            return fast
        conf = float(parsed.get("confidence") or 0.5)
        conf = max(0.0, min(1.0, conf))
        if conf <= fast.confidence:
            return fast
        return ReplanDecision(
            action=action,
            reason=f"LLM 兜底：{parsed.get('reason', '')[:120]}",
            confidence=conf,
        )
    except (llm.LLMUnavailable, llm.LLMError, ValueError, KeyError):
        return fast
