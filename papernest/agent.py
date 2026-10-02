"""研究 Agent 编排器：Plan-Execute-Replan 架构。

## 整体设计

```
用户输入 goal
    │
    ▼
[意图分类 intent.classify_with_llm_fallback] ──→ 启发式 + 可选 LLM 兜底
    │
    ▼
[种子 plan = build_plan(goal)] ──→ 保持原行为，dry_run 时直接返回
    │
    ▼
[主循环：state machine]
    while step_idx < len(plan) and step_idx < max_steps:
        执行 step（线程池 + 协作式超时 + 一次重试）
        if success:
            更新 state
            [Replanner replan.decide_with_llm_fallback] ──→ 看 observation 决定下一步
                ├─ continue: step_idx += 1
                ├─ revise:   plan.extend(new_steps)
                ├─ abort:    plan = replan.revise(state).new_steps
                ├─ ask_user: return needs_clarification
                ├─ finish:   return completed
                └─ fail:     return failed
        if failure:
            return failed/blocked（保留原行为；规则上 timeout_count 也记入 state）
```

## 为什么不直接用 LLM 做规划

build_plan 是一次性、零 LLM、可复现的「冷启动种子」。它适合做**第一刀**——
零开销、能挡住 80% 明显 query。但它一旦选错就没有回头路（注释里写的那堆
边界 case 都属于这一层）。

把 Replanner 加在外层，**保留 build_plan 做冷启动 + 加事后观察**——
意图是猜的不可靠，但「retrieve_library 拿到 0 篇」这件事是确定的，可观测的。
两件事各有擅长的位置，叠在一起比单层更稳。

## 与原版的差异

1. **意图可观测**：每次 run 在 AgentResponse.intent 字段返回结构化 Intent。
   tests/test_intent_routing.py 可以单压意图层，不用真跑工具。
2. **可中途改 plan**：原版 plan 一旦生成就不变；新版在每步执行后由 Replanner
   决定继续 / 追加 / 整盘重来。
3. **新增 needs_clarification 状态**：意图真的搞不清时主动 ask_user，
   不再 fall through 到默认 qa 路径硬答。前端按 clarification_questions 渲染。
4. **保留一切原有行为**：build_plan 签名不变、ThreadPoolExecutor 协作式取消、
   累积降级、event emission、db.save_agent_run、blocked/failed 状态都照旧。
"""

from __future__ import annotations

import concurrent.futures
import re
import time
import uuid
from typing import Any

from . import deadline, degrade, intent, llm, replan, tools
from .schemas import AgentResponse, PlanStep, ToolEvent


# ── 关键词词表 ─────────────────────────────────────────────────────────
# 保留在 agent.py 里**而不是**迁去 intent.py，是因为 build_plan 是 agent.py
# 的对外契约（api.py 在 dry_run 时会直接调它），迁走就破坏向后兼容。
# 真正的「启发式分类」在 intent.py（多了置信度、信号链、quoted_title 等硬信号）；
# 这里留一份**最简版**用于 dry_run 与向后兼容。

_CITE_WORDS = ("引用", "参考文献", "citation", "cite", "支持这段", "推荐文献")
_SURVEY_WORDS = ("综述", "survey", "研究现状", "研究进展", "文献总结")
_READ_WORDS = ("精读", "全文", "阅读论文", "read paper", "详细分析")

# 上面三组是**领域词**：既可能是用户的意图，也可能只是被用户引述的库内容。
# 这个库的主题正是 LLM Agent，503 篇里有 5 篇标题含 Survey/综述，而「贴一段标题
# 来提问」是最常见的用法——于是「被引述内容里的领域词」被读成了「用户的意图」。
# 真库实测：「《…A Survey…》被引用了多少次」被路由去做引用推荐（答案 209 就在
# papers.citation_count 里）；「精读论文 1：…A Survey…」被路由去写全库综述。
#
# 加词边界**修不掉这一层**——"A Survey" 本来就是独立单词。所以领域词只用来判断
# 「在谈论这件事」，还必须有一个**祈使动词**表明「请你做这件事」才触发昂贵分支。
# 动词按分支分开：推荐是引用的动词、不是综述的动词，否则「推荐几篇 agent 综述」
# 会被 survey 分支吃掉。
#
# 这两张表会漂移（用户说「攒一篇综述」就落回默认路径）。这是**刻意的不对称**：
# generate_survey 是全库 LLM 长文生成，误触发一次的代价远大于漏触发一次，
# 而默认路径至少还会给一个带来源的回答。扩表时按这个口径加词。
_CITE_VERBS = ("推荐", "补充", "补上", "找", "给出", "标注",
               "recommend", "recommends", "suggest", "suggests", "find", "add")
_SURVEY_VERBS = ("写", "生成", "起草", "做一篇", "整理成", "综述一下", "总结一下",
                 "generate", "write", "draft", "compose", "produce")


def _has_any(text: str, words: tuple[str, ...], whole: bool = False) -> bool:
    """子串匹配；ASCII 词加词边界，中日韩没有词边界故仍按子串。

    领域词只卡**左**边界（`whole=False`）：放过 citation / cited / surveys 这些屈折形，
    卡掉 `eli-cite-d` / `ex-cite-d` 这类「cite 只是词中间一段」的假命中
    （真库 pages 里有 elicited，chunks 里有 citeseer）。
    动词卡**两侧**边界（`whole=True`）：不然 recommend 含于 recommendation、
    add 含于 Addressing、find 含于 findings，库里的论文标题会自己变成祈使句。
    """
    lowered = text.lower()
    for word in words:
        w = word.lower()
        if w.isascii():
            pat = r"(?<![a-z0-9])" + re.escape(w) + (r"(?![a-z0-9])" if whole else "")
            if re.search(pat, lowered):
                return True
        elif w in lowered:
            return True
    return False


def _paper_id_from_goal(goal: str) -> int | None:
    # Accept common forms such as "精读论文 12" and "read #12".
    match = re.search(r"(?:论文|paper|#)\s*([0-9]+)", goal, flags=re.I)
    return int(match.group(1)) if match else None


def build_plan(goal: str, top_k: int = 5) -> list[PlanStep]:
    """Build a predictable plan from the user's intent.

    保留作为对外契约（dry_run / 兼容旧调用方）。真实的意图分类在 intent.classify()。
    这里与原版完全等价——一份**最小可工作**的种子 plan。
    """
    if _has_any(goal, _CITE_WORDS) and _has_any(goal, _CITE_VERBS, whole=True):
        return [PlanStep(
            id=1,
            tool="recommend_citations",
            reason="检测到引用/参考文献需求，先匹配可支持该段落的文献。",
            args={"paragraph": goal, "top_k": top_k},
        )]
    if _has_any(goal, _SURVEY_WORDS) and _has_any(goal, _SURVEY_VERBS, whole=True):
        return [PlanStep(
            id=1,
            tool="generate_survey",
            reason="检测到综述需求，基于库内文献生成带引用的结构化综述。",
            args={"topic": goal, "top_k": max(top_k, 8)},
        )]
    if _has_any(goal, _READ_WORDS):
        paper_id = _paper_id_from_goal(goal)
        if paper_id is not None:
            return [PlanStep(
                id=1,
                tool="read_paper",
                reason=f"检测到全文精读需求，读取指定论文 #{paper_id}。",
                args={"paper_id": paper_id},
            )]
    return [
        PlanStep(
            id=1,
            tool="retrieve_library",
            reason="先从本地文献库召回与问题最相关的候选论文。",
            args={"query": goal, "top_k": top_k},
        ),
        PlanStep(
            id=2,
            tool="answer_question",
            reason="使用召回结果生成带来源标记的回答。",
            args={"goal": goal, "top_k": top_k},
        ),
    ]


def _with_deadline(fn, timeout_s: float, args: dict):
    """在工作线程里给这次工具调用设一个预算，再执行。

    **不要把它说成「硬截止」**：Python 无法中断已开跑的线程，`future.cancel()`
    对 running 任务恒为 False。原来超时只截断了调用方——run 立刻返回 failed，
    而这条线程还在跑完整个工具：继续下载、继续写库、继续调重档模型花钱
    （实测 `timeout_s=2` 的步骤，工具线程在 6s 处照样跑完）。

    这里做的是**协作式取消**：把预算放进线程本地，长耗时组件（`llm.chat` 的重试
    循环与退避 sleep）在每个可中断点自己检查并停手。覆盖的是时间成本的大头——
    单次 LLM 调用最坏 433s 里有 153s 纯粹在退避 sleep。单次阻塞的 C 调用
    （PyMuPDF 解析）仍然中断不了，这一点如实标注。
    """
    with deadline.scope(timeout_s):
        return fn(**args)


def _short_summary(data: Any) -> str:
    if not isinstance(data, dict):
        return str(data)[:160]
    if "answer" in data:
        return f"已生成回答，来源 {len(data.get('sources') or [])} 篇"
    if "candidates" in data:
        return f"已找到 {len(data.get('candidates') or [])} 个引用候选"
    if "survey" in data:
        return f"已生成综述，来源 {len(data.get('sources') or [])} 篇"
    if "paper_ids" in data:
        return f"召回 {len(data.get('paper_ids') or [])} 篇论文"
    if "card" in data:
        return f"已处理论文 #{data.get('paper_id')}，抽取 {data.get('pages_stored', 0)} 页"
    return ", ".join(str(k) for k in list(data)[:4])[:160]


def _build_replan_event(step_id: int, decision) -> dict[str, Any]:
    """Replanner 决策的事件结构。存进 replan_events 给前端渲染。"""
    return {
        "after_step": step_id,
        "action": decision.action,
        "reason": decision.reason,
        "confidence": decision.confidence,
        "new_steps": [s.model_dump() for s in (decision.new_steps or [])],
        "questions": [q.model_dump() for q in (decision.questions or [])],
    }


def run_agent(goal: str, top_k: int = 5, max_steps: int = 5,
              topic: str | None = None, dry_run: bool = False,
              timeout_s: int = 240, on_event=None,
              persist: bool = True,
              intent_hint: str | None = None) -> AgentResponse:
    """Run a bounded plan with Replanner-driven step decisions.

    执行状态机：
      - planned → running → completed | failed | blocked | needs_clarification
      - 每步带超时与一次重试（LLMUnavailable 不重试——缺配置不是瞬态故障）
      - 超时是**协作式**的：调用方不再等待，同时给工作线程下发预算
      - 每步执行完后调 Replanner，按决策 continue/revise/abort/finish/fail/ask_user
      - 每个事件同步给 on_event 回调（SSE 进度用），整条轨迹落 agent_runs 表

    Plan-Execute-Replan 改造：除了原本的 planned/completed/failed/blocked，新增
    needs_clarification 状态与 replan_events 字段。build_plan 签名保持不变。
    """
    run_id = uuid.uuid4().hex
    t_start = time.perf_counter()
    unconfigured = False

    # ── 1. 意图分类（启发式 + 可选 LLM 兜底）──
    # 不影响 build_plan / dry_run，只是把意图显式记录下来便于调试与评测。
    intent_obj = intent.classify_with_llm_fallback(goal)

    # ── 1.5 intent_hint 覆盖 ──
    # 前端用户在 clarification_questions 选了答案后回传 intent_hint——直接锁定意图。
    # 这样做的好处：
    # 1. 用户的选择比 LLM 更准（人知道自己想干啥）
    # 2. 跳过 Replanner 的低置信度 ASK_USER（已经问过了）
    # 3. confidence=1.0 让 Replanner rule 0 不会再次触发 ASK_USER（避免循环）
    # 非法 label 静默忽略，让启发式兜底——前端如果传了乱码也不会崩。
    if intent_hint in ("cite", "survey", "read", "qa"):
        intent_obj = intent.Intent(
            primary=intent_hint,
            confidence=1.0,
            signals=(intent_obj.signals if intent_obj else []) + [f"user_hint:{intent_hint}"],
            paper_id=intent_obj.paper_id if intent_obj else None,
            topic_hint=intent_obj.topic_hint if intent_obj else None,
        )

    # ── 2. pre_execution Replanner 检查 ──
    # 这一步**必须在 dry_run 之前**，否则 dry_run 拿不到 replan_events。
    # 用空 steps_done 触发「执行前」规则（低置信度 → ask_user）。
    pre_state = {
        "goal": goal, "intent": intent_obj, "top_k": top_k,
        "steps_done": [], "last_step": None, "last_observation": None,
        "last_status": "completed", "has_answer": False,
        "timeout_count": 0, "failed_tools": [],
    }
    pre_decision = replan.decide_with_llm_fallback(pre_state)
    pre_event = {"after_step": 0, "phase": "pre_execution",
                 **pre_decision.to_dict()}

    # ── 3. 种子 plan ──
    # 两种来源：
    # - 默认（无 intent_hint）：build_plan()，保留原行为
    # - intent_hint 已提供：replan._build_steps_for()，按锁定意图构造步骤
    #   这样前端用户回答 clarification 后，plan 真的会跟着变（之前 build_plan
    #   仍按原始 goal 词表分类，intent_hint 没真正接管）。
    if intent_hint in ("cite", "survey", "read", "qa"):
        plan = replan._build_steps_for(intent_obj, goal, top_k)
    else:
        plan = build_plan(goal, top_k)
    if dry_run:
        # dry_run 也带上 replan_events，前端能预览 Replanner 的初判。
        return AgentResponse(
            run_id=run_id, status="planned", goal=goal, plan=plan,
            intent=intent_obj, replan_events=[pre_event],
        )

    events: list[ToolEvent] = []
    callback_errors: list[str] = []
    replan_events: list[dict[str, Any]] = []  # Replanner 决策记录，可观测性
    clarification_questions = []  # 仅当 needs_clarification 时填充

    # Replanner 用的 state。每次步骤执行完更新。
    replan_state: dict[str, Any] = {
        "goal": goal,
        "intent": intent_obj,
        "top_k": top_k,
        "steps_done": [],          # list[(PlanStep, observation_dict)]
        "last_step": None,
        "last_observation": None,
        "last_status": "completed",
        "has_answer": False,
        "timeout_count": 0,
        "failed_tools": [],
        "plan_remaining": len(plan),  # Replanner 据此判定「当前步骤是不是 plan 最后一步」
    }

    final_answer: str | None = None
    sources: list[dict[str, Any]] = []
    result: dict[str, Any] = {}
    degraded: str | None = None
    notes: list = []          # 结构化降级记录，跨步骤累积
    error: str | None = None
    status = "completed"

    def emit(ev: ToolEvent):
        events.append(ev)
        if on_event:
            try:
                on_event(ev.model_dump())
            except Exception as exc:
                # 回调故障不该连累执行，但也不能咽掉：SSE 断了前端时间线会缺步，
                # run 却仍报 completed，用户据此以为某一步压根没跑过。
                callback_errors.append(type(exc).__name__)

    def _finish(run_status: str, run_error: str | None) -> AgentResponse:
        resp = AgentResponse(
            run_id=run_id, status=run_status, goal=goal,
            plan=plan, events=events, answer=final_answer,
            sources=sources, result=result, degraded=degraded,
            error=run_error,
            total_ms=round((time.perf_counter() - t_start) * 1000),
            intent=intent_obj,
            clarification_questions=clarification_questions,
            replan_events=replan_events,
        )
        if callback_errors:
            note = f"进度回调失败 {len(callback_errors)} 次（前端时间线可能缺步）"
            resp.degraded = f"{resp.degraded}；{note}" if resp.degraded else note
        if persist:
            try:
                from . import db
                db.save_agent_run(run_id, goal, run_status,
                                  [p.model_dump() for p in plan],
                                  [e.model_dump() for e in events],
                                  final_answer, run_error, resp.total_ms)
            except Exception as exc:
                # 落库失败最可能发生在库出问题 / 磁盘满 / 并发写冲突的时刻，
                # 也就是**最需要这条轨迹**的时刻。原来接口照常返回 run_id 和
                # completed，事后按 id 回查却是空的，docstring 承诺的可回放在此失效。
                note = f"执行轨迹落库失败：{type(exc).__name__}"
                resp.degraded = f"{resp.degraded}；{note}" if resp.degraded else note
        return resp

    # ── 4. 执行前的 Replanner 检查（意图信心极低 → 直接 ask_user）──
    # pre_decision 在前面就已经算过了，这里把它的副作用（replan_events 记录）
    # 与 ask_user/fail 的早期返回做完。
    replan_events.append(pre_event)
    if pre_decision.action == "ask_user":
        clarification_questions = pre_decision.questions
        return _finish("needs_clarification", pre_decision.reason)
    if pre_decision.action == "fail":
        return _finish("failed", pre_decision.reason)

    # ── 4. 步骤执行主循环 ──
    # 用 while + step_idx 而非 for-range，是为了支持 replan.abort 后重置 step_idx。
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    abandoned = False
    try:
        step_idx = 0
        while step_idx < len(plan):
            if step_idx >= max_steps:
                error = f"达到 max_steps={max_steps}，仍有计划步骤未执行"
                return _finish("blocked", error)

            step = plan[step_idx]
            timed_out = False
            tool_error = None
            started = time.perf_counter()
            data = None

            fn = tools.TOOL_REGISTRY.get(step.tool)
            if fn is None:
                tool_error = f"未知工具：{step.tool}"
                emit(ToolEvent(step_id=step.id, tool=step.tool, status="failed",
                               latency_ms=0, attempt=1,
                               summary="工具未注册", error=tool_error))
                error = tool_error
                return _finish("failed", error)

            # 注意：原版 agent.py 在步骤**开始时不发** running 事件，只在重试时
            # 发"running + 重试中"。新加的 running 事件会让 test_agent 的事件计数
            # 断言破。保持原行为；前端靠 completed / failed / timeout 三个终态
            # 渲染进度已经够用。

            for attempt in (1, 2):
                try:
                    future = pool.submit(_with_deadline, fn, timeout_s, step.args)
                    # The caller's step budget is the actual wait budget.  The
                    # worker also receives the cooperative deadline, but a
                    # blocking tool must not get an extra grace period that
                    # lets the next step start after the plan has expired.
                    data = future.result(timeout=max(0.01, timeout_s))
                    if not isinstance(data, dict):
                        # 工具约定返回 dict；不是 dict 当作 bug 处理
                        raise TypeError(f"工具返回非 dict：{type(data).__name__}")
                    latency_ms = round((time.perf_counter() - started) * 1000)
                    emit(ToolEvent(step_id=step.id, tool=step.tool,
                                   status="completed",
                                   latency_ms=latency_ms, attempt=attempt,
                                   summary=_short_summary(data)))
                    break
                except concurrent.futures.TimeoutError:
                    timed_out = True
                    abandoned = True
                    future.cancel()  # 尚未开跑才取消得掉；已在跑的线程只能放弃等待
                    tool_error = (f"工具执行超过 {timeout_s}s，已放弃等待。"
                                  f"该线程已收到取消信号，会在下一个可中断点"
                                  f"（重试/退避之间）停手；单次阻塞调用无法中断")
                    break  # 超时不重试：耗时工具重试只会更久
                except llm.LLMUnavailable as exc:  # 配置缺失不是瞬态故障，不重试
                    tool_error = str(exc)
                    unconfigured = True
                    break
                except Exception as exc:
                    tool_error = f"{type(exc).__name__}: {str(exc)[:300]}"
                    if attempt == 1:
                        emit(ToolEvent(
                            step_id=step.id, tool=step.tool, status="running",
                            latency_ms=round((time.perf_counter() - started) * 1000),
                            attempt=attempt,
                            summary="首次尝试失败，重试中", error=tool_error))
                        continue

            if data is not None:
                # ── 步骤成功：更新 state 并调 Replanner ──
                replan_state["last"] = data
                replan_state["steps_done"].append((step, data))
                replan_state["last_step"] = step
                replan_state["last_observation"] = data
                replan_state["last_status"] = "completed"
                replan_state["plan_remaining"] = len(plan) - step_idx - 1

                if step.tool == "retrieve_library":
                    replan_state["retrieval"] = data
                if data.get("degraded") or data.get("degraded_detail"):
                    # **累积，不是覆盖**。原来是后写覆盖：第 1 步 retrieve_library
                    # 报出的「向量整路失败」（CRITICAL）会被第 2 步 answer_question
                    # 的低级降级整串顶掉，用户看到的只剩轻微降级。
                    notes = degrade.merge(
                        notes,
                        degrade.from_dicts(data.get("degraded_detail") or [])
                        or degrade.from_legacy(data.get("degraded")))
                    degraded = degrade.render(notes)
                if step.tool in {"answer_question", "generate_survey"}:
                    final_answer = data.get("answer") or data.get("survey")
                    sources = data.get("sources") or []
                    replan_state["has_answer"] = bool(final_answer)
                if step.tool == "recommend_citations":
                    sources = data.get("candidates") or []
                result = data

                # ── Replanner 决策 ──
                decision = replan.decide_with_llm_fallback(dict(replan_state))
                replan_events.append(_build_replan_event(step.id, decision))

                if decision.action == "ask_user":
                    clarification_questions = decision.questions
                    return _finish("needs_clarification", decision.reason)
                if decision.action == "finish":
                    return _finish("completed", None)
                if decision.action == "fail":
                    return _finish("failed", decision.reason)
                if decision.action == "revise":
                    # 把新步骤追加到 plan 尾部（保持原步骤的 id 不冲突，给新步骤重置 id）
                    next_id = max(p.id for p in plan) + 1
                    for new_step in decision.new_steps:
                        new_step.id = next_id
                        next_id += 1
                    plan.extend(decision.new_steps)
                    step_idx += 1
                    continue
                if decision.action == "abort":
                    # 整盘重规划：丢弃剩余步骤，按新 intent 重建 plan
                    new_decision = replan.revise(dict(replan_state))
                    if new_decision.action == "ask_user":
                        clarification_questions = new_decision.questions
                        return _finish("needs_clarification", new_decision.reason)
                    new_plan = list(new_decision.new_steps)
                    for i, p in enumerate(new_plan, start=1):
                        p.id = i
                    plan = new_plan
                    replan_state["plan_remaining"] = len(plan)
                    replan_events.append({"after_step": step.id, "phase": "aborted",
                                          "new_plan": [p.model_dump() for p in plan]})
                    step_idx = 0
                    continue
                # continue 或规则未匹配：照原 plan 继续
                step_idx += 1
                continue

            # ── 步骤失败：保留原行为直接返回 ──
            # 这里**不**调 Replanner 的 continue/revise——失败原因大概率在工具层
            # （网络/超时/配置），意图层调整救不了它。但要把 timeout_count 与
            # failed_tools 累进 replan_state，未来步骤 Replanner 仍能感知。
            if timed_out:
                replan_state["timeout_count"] = replan_state.get("timeout_count", 0) + 1
            else:
                replan_state.setdefault("failed_tools", []).append(step.tool)


            if timed_out:
                emit(ToolEvent(step_id=step.id, tool=step.tool, status="timeout",
                               latency_ms=timeout_s * 1000, summary="步骤超时", error=tool_error))
                error = tool_error
                return _finish("failed", error)
            emit(ToolEvent(step_id=step.id, tool=step.tool, status="failed",
                           latency_ms=round((time.perf_counter() - started) * 1000),
                           attempt=2,
                           summary="工具执行失败（已重试）" if len(events) > 1 else "工具执行失败",
                           error=tool_error))
            error = tool_error
            status = "blocked" if unconfigured else "failed"
            return _finish(status, error)
    finally:
        # ThreadPoolExecutor 的 `with` 出口默认 wait=True——超时步骤会把「硬截止」
        # 变回「等到天荒地老」。超时过就不等了，让那条线程自生自灭。
        pool.shutdown(wait=not abandoned, cancel_futures=True)

    return _finish("completed", None)
