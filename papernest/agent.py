"""A small, deterministic Research Agent orchestrator.

This is intentionally a bounded workflow rather than an unconstrained
multi-agent loop.  The planner selects a known tool sequence, the executor
passes typed state between tools, and every invocation is recorded as an event
that the API can render as an execution timeline.
"""
from __future__ import annotations

import concurrent.futures
import re
import time
import uuid
from typing import Any

from . import deadline, degrade, llm, tools
from .schemas import AgentResponse, PlanStep, ToolEvent


_CITE_WORDS = ("引用", "参考文献", "citation", "cite", "支持这段", "推荐文献")
_SURVEY_WORDS = ("综述", "survey", "研究现状", "研究进展", "文献总结")
_READ_WORDS = ("精读", "全文", "阅读论文", "read paper", "详细分析")


def _has_any(text: str, words: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(word.lower() in lowered for word in words)


def _paper_id_from_goal(goal: str) -> int | None:
    # Accept common forms such as "精读论文 12" and "read #12".
    match = re.search(r"(?:论文|paper|#)\s*([0-9]+)", goal, flags=re.I)
    return int(match.group(1)) if match else None


def build_plan(goal: str, top_k: int = 5) -> list[PlanStep]:
    """Build a predictable plan from the user's intent.

    An LLM planner can be added later, but this baseline is reproducible and
    keeps arbitrary model output away from the tool execution boundary.
    """
    if _has_any(goal, _CITE_WORDS):
        return [PlanStep(
            id=1,
            tool="recommend_citations",
            reason="检测到引用/参考文献需求，先匹配可支持该段落的文献。",
            args={"paragraph": goal, "top_k": top_k},
        )]
    if _has_any(goal, _SURVEY_WORDS):
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


def run_agent(goal: str, top_k: int = 5, max_steps: int = 5,
              topic: str | None = None, dry_run: bool = False,
              timeout_s: int = 240, on_event=None,
              persist: bool = True) -> AgentResponse:
    """Run a bounded plan and return a serializable execution trace.

    执行状态机：planned → (running: step 尝试/重试) → completed | failed | blocked。
    每步带超时与一次重试（LLMUnavailable 不重试——缺配置不是瞬态故障）。
    超时是**协作式**的：调用方不再等待，同时给工作线程下发预算，
    长耗时组件在重试/退避之间自行停手（见 `deadline`）。Python 中断不了
    已开跑的线程，所以单次阻塞调用仍会跑完——不要把它说成「硬截止」。
    每个事件同步给 on_event 回调（SSE 进度用），整条轨迹落 agent_runs 表。
    """
    run_id = uuid.uuid4().hex
    t_start = time.perf_counter()
    plan = build_plan(goal, top_k)
    if dry_run:
        return AgentResponse(run_id=run_id, status="planned", goal=goal, plan=plan)

    events: list[ToolEvent] = []
    callback_errors: list[str] = []
    state: dict[str, Any] = {"topic": topic, "retrieval": {}, "last": {}}
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
        resp = AgentResponse(run_id=run_id, status=run_status, goal=goal,
                             plan=plan, events=events, answer=final_answer,
                             sources=sources, result=result, degraded=degraded,
                             error=run_error,
                             total_ms=round((time.perf_counter() - t_start) * 1000))
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

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1,
                                                 thread_name_prefix="papernest-tool")
    abandoned = False  # 有步骤超时：其后台线程可能还在收尾，不能等它
    try:
        for step in plan[:max_steps]:
            args = dict(step.args)
            if step.tool == "answer_question":
                args["candidate_ids"] = state["retrieval"].get("paper_ids") or None
            data, tool_error, timed_out = None, None, False

            for attempt in (1, 2):  # 每步最多一次重试
                started = time.perf_counter()
                future = pool.submit(_with_deadline, tools.TOOL_REGISTRY[step.tool],
                                     timeout_s, args)
                try:
                    data = future.result(timeout=timeout_s)
                    if not isinstance(data, dict):
                        data = {"value": data}
                    tool_error = None
                    emit(ToolEvent(
                        step_id=step.id, tool=step.tool, status="completed",
                        latency_ms=round((time.perf_counter() - started) * 1000),
                        attempt=attempt, summary=_short_summary(data)))
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
                state["last"] = data
                if step.tool == "retrieve_library":
                    state["retrieval"] = data
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
                if step.tool == "recommend_citations":
                    sources = data.get("candidates") or []
                result = data
                continue

            if timed_out:
                emit(ToolEvent(step_id=step.id, tool=step.tool, status="timeout",
                               latency_ms=timeout_s * 1000, summary="步骤超时", error=tool_error))
                error = tool_error
                return _finish("failed", error)
            emit(ToolEvent(step_id=step.id, tool=step.tool, status="failed",
                           latency_ms=round((time.perf_counter() - t_start) * 1000),
                           summary="工具执行失败（已重试）" if len(events) > 1 else "工具执行失败",
                           error=tool_error))
            error = tool_error
            status = "blocked" if "未配置" in (tool_error or "") else "failed"
            return _finish(status, error)
    finally:
        # ThreadPoolExecutor 的 `with` 出口默认 wait=True——超时步骤会把「硬截止」
        # 变回「等到天荒地老」。超时过就不等了，让那条线程自生自灭。
        pool.shutdown(wait=not abandoned, cancel_futures=True)

    if len(plan) > max_steps:
        error = f"达到 max_steps={max_steps}，仍有计划步骤未执行"
        return _finish("blocked", error)
    return _finish("completed", None)

