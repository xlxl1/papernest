"""Plan-Execute-Replan 改造的回归测试。

覆盖三层：

1. **意图分类（intent.classify）**——把 build_plan 里的词表 + 硬信号拆出来后的
   单测。parity 测试：每条 case 既要走 intent 又要走 build_plan，验证分类
   与路由结果一致。

2. **Replanner 决策（replan.decide）**——state machine 的 9 个 rule 都要有 case。
   全部用纯 state 字典构造，不依赖真工具、不依赖 DB，跑得快、好定位。

3. **API 契约（schemas.AgentResponse）**——新增字段（intent / clarification_questions /
   replan_events / needs_clarification 状态）必须存在；旧字段（plan / events /
   answer / sources）必须保留。

不做端到端的 run_agent 真实执行——那是 test_agent.py 的职责。
本测试覆盖的是「不动工具也能压」的那部分。
"""

import unittest

from papernest.agent import build_plan
from papernest.intent import Intent, classify
from papernest.replan import (
    ClarificationQuestion,
    ReplanDecision,
    decide,
)
from papernest.schemas import AgentResponse, PlanStep, ToolEvent


def _state(goal, steps_done_specs, last_status="completed", timeout_count=0,
           has_answer=False, intent_obj=None, plan_remaining=0):
    """构造 replan.decide 用的 state 字典。

    plan_remaining=0 是默认：Replanner 只有在「retrieve 是 plan 最后一步」时
    才会因空召回 ASK_USER；中间步骤不触发。
    """
    if intent_obj is None:
        intent_obj = classify(goal)
    steps = []
    for tool, obs in steps_done_specs:
        s = PlanStep(id=len(steps) + 1, tool=tool, reason="", args={})
        steps.append((s, obs))
    last = steps[-1] if steps else None
    return {
        "goal": goal, "intent": intent_obj, "top_k": 5,
        "steps_done": steps,
        "last_step": last[0] if last else None,
        "last_observation": last[1] if last else None,
        "last_status": last_status,
        "has_answer": has_answer,
        "timeout_count": timeout_count,
        "failed_tools": [],
        "plan_remaining": plan_remaining,
    }


class IntentClassificationTests(unittest.TestCase):
    """意图分类单元测试——覆盖主要路由 + 故意造出的 tricky case。"""

    def test_hard_signal_quoted_title_is_read(self):
        i = classify('读一下《Attention is all you need》')
        self.assertEqual(i.primary, "read")
        self.assertIn("quoted_title", i.signals)
        self.assertEqual(i.topic_hint, "Attention is all you need")

    def test_quoted_title_alone_is_not_read(self):
        """引号标题单独出现**不是** read 信号——只是 topic anchor。
        「和《X》相关的文献」「《X》被引用了多少次」都不该被路由去读 X。
        这正是 build_plan 注释里讲的「A Survey 不算用户意图」同类陷阱。
        """
        for g in [
            chr(0x300a) + "A Survey of LLMs" + chr(0x300b) + " 被引用了多少次？",
            "和" + chr(0x300a) + "A Survey of LLMs" + chr(0x300b) + " 相关的文献",
        ]:
            i = classify(g)
            self.assertNotEqual(i.primary, "read",
                                f"goal={g!r} should not be read: signals={i.signals}")

    def test_paper_id_with_read_word_is_read(self):
        i = classify("精读论文 12")
        self.assertEqual(i.primary, "read")
        self.assertEqual(i.paper_id, 12)
        self.assertGreaterEqual(i.confidence, 0.85)

    def test_paper_id_alone_is_read_low_confidence(self):
        """只有 paper_id 没有 read 动词也判 read，但置信度要低于有动词的。"""
        i = classify("paper 7 讲了什么")
        self.assertEqual(i.primary, "read")
        self.assertEqual(i.paper_id, 7)
        self.assertLess(i.confidence, 0.85)  # 显式比硬信号低一档

    def test_cite_requires_both_domain_and_verb(self):
        # 没有 cite domain 也没有 cite verb → 不算 cite
        i = classify("Agent 论文最新进展")
        self.assertNotEqual(i.primary, "cite")

    def test_survey_does_not_eat_cite(self):
        """「推荐几篇 agent 综述」必须有 cite_verb 才能进 cite，不能被 survey 吃掉。

        这是原 build_plan 注释里强调的：survey 域词被库标题污染得很厉害。
        """
        i = classify("推荐几篇 agent 综述")
        self.assertEqual(i.primary, "cite")

    def test_survey_needs_verb(self):
        """survey 域词 + survey 动词才能进 survey，缺动词落回 qa。"""
        i = classify("写一篇 LLM Agent 综述")
        self.assertEqual(i.primary, "survey")
        i = classify("LLM Agent 综述")  # 没有「写」/「生成」/「整理」等动词
        self.assertNotEqual(i.primary, "survey")

    def test_default_qa_path(self):
        i = classify("LLM Agent 方法论")
        self.assertEqual(i.primary, "qa")
        # 默认 qa 路径置信度 0.5——是基线，不是低
        self.assertEqual(i.confidence, 0.5)

    def test_pure_followup_is_qa_with_higher_confidence(self):
        """纯追问（只剩疑问词）比默认 qa 略高。"""
        i = classify("为什么呢")
        self.assertEqual(i.primary, "qa")
        self.assertGreater(i.confidence, 0.5)

    def test_english_word_boundary_prevents_substring_hits(self):
        """英文 domain 词必须走词边界——"find" 不能命中 "findings"。"""
        i = classify("What are the findings on RAG?")
        # "findings" 不该被 cite 动词 "find" 命中
        self.assertNotEqual(i.primary, "cite")

    def test_intent_object_serializable(self):
        """Intent 必须能 round-trip（dict）——前端、评测、trace 都靠这个。"""
        i = classify("精读论文 42")
        d = i.model_dump()
        i2 = Intent(**d)
        self.assertEqual(i.primary, i2.primary)
        self.assertEqual(i.confidence, i2.confidence)
        self.assertEqual(i.paper_id, i2.paper_id)


class IntentParityWithBuildPlanTests(unittest.TestCase):
    """每个意图分类 case 对应的 build_plan 工具，必须落到同一工具。"""

    def _expected_first_tool(self, goal: str) -> str:
        plan = build_plan(goal)
        return plan[0].tool

    def test_parity_read(self):
        for g in ["精读论文 12", "读一下《X》"]:
            intent_obj = classify(g)
            expected = "read_paper" if intent_obj.paper_id else self._expected_first_tool(g)
            self.assertEqual(self._expected_first_tool(g), expected)

    def test_parity_cite(self):
        for g in ["推荐支持某段的引用", "找几篇参考文献"]:
            intent_obj = classify(g)
            self.assertEqual(intent_obj.primary, "cite")
            self.assertEqual(self._expected_first_tool(g), "recommend_citations")

    def test_parity_survey(self):
        for g in ["写一篇综述", "生成综述", "整理成综述"]:
            intent_obj = classify(g)
            self.assertEqual(intent_obj.primary, "survey")
            self.assertEqual(self._expected_first_tool(g), "generate_survey")

    def test_parity_default_qa(self):
        for g in ["LLM Agent 方法论", "什么是 channel estimation"]:
            intent_obj = classify(g)
            self.assertEqual(intent_obj.primary, "qa")
            self.assertEqual(self._expected_first_tool(g), "retrieve_library")


class ReplanDecisionTests(unittest.TestCase):
    """Replanner state machine 的 9 个 rule 各一条 case。"""

    def test_rule_0_unknown_intent_asks_user(self):
        """primary=unknown → ASK_USER。"""
        state = _state("xyz", [], intent_obj=Intent(
            primary="unknown", confidence=0.3, signals=[]))
        d = decide(state)
        self.assertEqual(d.action, "ask_user")
        self.assertTrue(d.questions)

    def test_rule_0_low_confidence_asks_user(self):
        """置信度 < 0.4 也 ASK_USER（仅当无 steps_done）。"""
        state = _state("...", [], intent_obj=Intent(
            primary="qa", confidence=0.3, signals=[]))
        d = decide(state)
        self.assertEqual(d.action, "ask_user")

    def test_rule_0_normal_qa_does_NOT_ask(self):
        """默认 qa 路径置信度 0.5 不应触发 ASK_USER。"""
        state = _state("LLM Agent 方法论", [])
        d = decide(state)
        self.assertEqual(d.action, "continue")

    def test_rule_1_has_answer_finishes(self):
        """已生成答案 + 步骤成功 → FINISH（早停）。"""
        state = _state("LLM Agent", [
            ("retrieve_library", {"paper_ids": [1, 2], "papers": [{}, {}]}),
            ("answer_question", {"answer": "基于 [1] ...", "sources": [{"id": 1}]}),
        ], has_answer=True)
        d = decide(state)
        self.assertEqual(d.action, "finish")

    def test_rule_2_timeout_count_fails(self):
        """累计 timeout ≥ 2 → FAIL，不再问用户。"""
        state = _state("LLM Agent",
                       [("retrieve_library", {"paper_ids": [], "papers": []})],
                       last_status="timeout", timeout_count=2)
        d = decide(state)
        self.assertEqual(d.action, "fail")

    def test_rule_3_single_timeout_continues(self):
        """单次 timeout → CONTINUE，不抢工具层的活。"""
        state = _state("LLM Agent",
                       [("retrieve_library", {"paper_ids": [], "papers": []})],
                       last_status="timeout", timeout_count=1)
        d = decide(state)
        self.assertEqual(d.action, "continue")

    def test_rule_4_empty_retrieval_asks_user(self):
        """retrieve_library 空召回 + 非 read 意图 + 是 plan 最后一步 → ASK_USER。"""
        state = _state("推荐支持 LLM 的文献", [
            ("retrieve_library", {"paper_ids": [], "papers": []}),
        ], plan_remaining=0)
        d = decide(state)
        self.assertEqual(d.action, "ask_user")
        self.assertEqual(d.questions[0].qid, "no_match")

    def test_rule_4_empty_retrieval_not_last_step_continues(self):
        """retrieve 空召回但后面还有步骤（如 answer_question）→ 不主动问，
        让后面的步骤自然处理空召回（answer 会回「未覆盖」，触发 rule 5）。"""
        state = _state("LLM Agent 方法论", [
            ("retrieve_library", {"paper_ids": [], "papers": []}),
        ], plan_remaining=1)  # 后面还有 1 步
        d = decide(state)
        self.assertNotEqual(d.action, "ask_user")

    def test_rule_4_empty_retrieval_read_does_not_ask(self):
        """read 意图下空召回是正常的（paper_id 不存在），不再追问。"""
        state = _state("精读论文 999", [
            ("retrieve_library", {"paper_ids": [], "papers": []}),
        ])
        # read 意图：primary 是 "read"，_ask 不触发
        d = decide(state)
        self.assertNotEqual(d.action, "ask_user")

    def test_rule_5_qa_no_coverage_with_survey_word_revises(self):
        """qa 路径未覆盖 + goal 含 survey 域词 → REVISE 到 survey。"""
        state = _state("LLM Agent 综述", [
            ("retrieve_library", {"paper_ids": [1], "papers": [{"id": 1}]}),
            ("answer_question", {"answer": "库内未覆盖。", "sources": []}),
        ])
        d = decide(state)
        self.assertEqual(d.action, "revise")
        self.assertEqual(d.new_steps[0].tool, "generate_survey")

    def test_rule_5_qa_no_coverage_without_survey_word_continues(self):
        """qa 路径未覆盖但 goal 没 survey 域词 → CONTINUE（不强制升级）。"""
        state = _state("LLM Agent 评测", [
            ("retrieve_library", {"paper_ids": [1], "papers": [{"id": 1}]}),
            ("answer_question", {"answer": "库内未覆盖。", "sources": []}),
        ])
        d = decide(state)
        self.assertEqual(d.action, "continue")

    def test_rule_6_cites_all_unverified_revises_to_survey(self):
        """引用候选全部 unverified → REVISE 到 survey 救场。"""
        state = _state("推荐支持某段的引用", [
            ("retrieve_library", {"paper_ids": [1, 2], "papers": [{}, {}]}),
            ("recommend_citations", {"candidates": [
                {"paper_id": 1, "verified": False},
                {"paper_id": 2, "verified": False},
            ]}),
        ])
        d = decide(state)
        self.assertEqual(d.action, "revise")
        self.assertEqual(d.new_steps[0].tool, "generate_survey")

    def test_rule_6_partial_verified_does_not_revise(self):
        """部分 verified → CONTINUE，让用户看到候选。"""
        state = _state("推荐支持某段的引用", [
            ("retrieve_library", {"paper_ids": [1, 2], "papers": [{}, {}]}),
            ("recommend_citations", {"candidates": [
                {"paper_id": 1, "verified": True},
                {"paper_id": 2, "verified": False},
            ]}),
        ])
        d = decide(state)
        self.assertNotEqual(d.action, "revise")

    def test_default_continue(self):
        """没有任何 rule 命中 → CONTINUE。"""
        state = _state("推荐几篇引用", [
            ("retrieve_library", {"paper_ids": [1], "papers": [{"id": 1}]}),
            ("recommend_citations", {"candidates": [
                {"paper_id": 1, "verified": True},
            ]}),
        ])
        d = decide(state)
        self.assertEqual(d.action, "continue")


class ReplanRulesOrderingTests(unittest.TestCase):
    """规则优先级测试——timeout/fail 应该优先于 ask_user。"""

    def test_timeout_fails_before_asking_on_empty_retrieval(self):
        """retrieve 空 + timeout_count=2 → fail（rule 2 优先于 rule 4）。"""
        state = _state("推荐文献",
                       [("retrieve_library", {"paper_ids": [], "papers": []})],
                       last_status="timeout", timeout_count=2)
        d = decide(state)
        self.assertEqual(d.action, "fail")

    def test_failed_step_continues_without_replanning(self):
        """单次失败时 Replanner 不抢活——继续执行原 plan。"""
        state = _state("LLM Agent",
                       [("retrieve_library", {"paper_ids": [], "papers": []})],
                       last_status="failed")
        d = decide(state)
        self.assertEqual(d.action, "continue")


class AgentResponseSchemaTests(unittest.TestCase):
    """AgentResponse 新增字段必须存在 + 旧字段必须保留（向后兼容）。"""

    def test_new_fields_exist(self):
        r = AgentResponse(run_id="x", status="planned", goal="g")
        self.assertTrue(hasattr(r, "intent"))
        self.assertTrue(hasattr(r, "clarification_questions"))
        self.assertTrue(hasattr(r, "replan_events"))

    def test_old_fields_preserved(self):
        """plan / events / answer / sources 不能丢——前端 SSE 渲染靠这些。"""
        r = AgentResponse(run_id="x", status="planned", goal="g",
                          plan=[PlanStep(id=1, tool="t", reason="", args={})],
                          events=[ToolEvent(step_id=1, tool="t", status="completed")],
                          answer="ans", sources=[{"id": 1}])
        self.assertEqual(r.plan[0].tool, "t")
        self.assertEqual(r.events[0].status, "completed")
        self.assertEqual(r.answer, "ans")
        self.assertEqual(r.sources, [{"id": 1}])

    def test_needs_clarification_status_is_valid(self):
        """新增 needs_clarification 状态必须能被 Pydantic 接受。"""
        r = AgentResponse(run_id="x", status="needs_clarification", goal="g",
                          clarification_questions=[
                              ClarificationQuestion(
                                  qid="intent", question="what?",
                                  options=[{"label": "a", "value": "qa"}],
                              ),
                          ])
        self.assertEqual(r.status, "needs_clarification")
        self.assertEqual(len(r.clarification_questions), 1)
        self.assertEqual(r.clarification_questions[0].qid, "intent")

    def test_intent_default_none(self):
        """未传 intent 时为 None——保留旧 AgentResponse 构造方式。"""
        r = AgentResponse(run_id="x", status="completed", goal="g")
        self.assertIsNone(r.intent)
        self.assertEqual(r.replan_events, [])
        self.assertEqual(r.clarification_questions, [])


class AgentRunAgentDryRunTests(unittest.TestCase):
    """dry_run 走通新管线：意图字段填充、replan_events 记录、plan 保持原 build_plan 行为。"""

    def test_dry_run_fills_intent(self):
        from papernest.agent import run_agent
        r = run_agent("精读论文 12", dry_run=True, persist=False)
        self.assertEqual(r.status, "planned")
        self.assertIsNotNone(r.intent)
        self.assertEqual(r.intent.primary, "read")
        self.assertEqual(r.intent.paper_id, 12)

    def test_dry_run_records_pre_execution_replan(self):
        """dry_run 也至少有一次 replan_events 记录（pre_execution 阶段）。"""
        from papernest.agent import run_agent
        r = run_agent("LLM Agent 方法论", dry_run=True, persist=False)
        self.assertGreaterEqual(len(r.replan_events), 1)
        self.assertEqual(r.replan_events[0].get("phase"), "pre_execution")

    def test_dry_run_keeps_build_plan_compat(self):
        """build_plan 的 plan 字段必须仍是 build_plan 的输出，不被 Replanner 替换。"""
        from papernest.agent import run_agent
        r = run_agent("推荐支持某段的引用", dry_run=True, persist=False)
        # 旧 build_plan 行为：cite 走 recommend_citations（单步）
        self.assertEqual(r.plan[0].tool, "recommend_citations")
        self.assertEqual(len(r.plan), 1)


if __name__ == "__main__":
    unittest.main()
