"""澄清问题端到端流程的测试。

模拟前端流程：
1. 用户提问 → Replanner 决定 ASK_USER → 返回 needs_clarification + questions
2. 用户选 intent=cite → 重跑带 intent_hint=cite → 走 cite 路径

不依赖真 LLM（mock TOOL_REGISTRY）、不依赖真 DB（TempDbTestCase）。
"""

import unittest

from fastapi.testclient import TestClient

from papernest import agent, tools
from papernest.api import app
from papernest.schemas import PlanStep
from tests.support import TempDbTestCase


class ClarificationFlowTests(TempDbTestCase):
    """needs_clarification + intent_hint 完整流程。"""

    def setUp(self):
        super().setUp()
        # 用一个会触发 ASK_USER 的目标：默认 qa 路径，但 goal 没领域词
        # → intent confidence 0.5，正常情况 rule 0 不触发 ASK_USER
        # 但 retrieve 返回空 + plan_remaining=0 会触发 ASK_USER（rule 4）
        # 简化：用 mock 让 retrieve 返回空论文 + 不再后续工具
        # 让 default qa 路径走到 retrieve → 空 → 触发 ASK_USER
        self._saved = dict(tools.TOOL_REGISTRY)

    def tearDown(self):
        tools.TOOL_REGISTRY.clear()
        tools.TOOL_REGISTRY.update(self._saved)
        super().tearDown()

    def test_step1_needs_clarification_when_retrieve_empty(self):
        """retrieve_library 返回空 + 是 plan 最后一步 → Replanner 触发 ASK_USER。

        通过把 build_plan patch 成「只 retrieve 一步」，让 plan_remaining=0，
        rule 4（empty retrieval + plan last step）的判定条件命中。
        """
        tools.TOOL_REGISTRY["retrieve_library"] = lambda query, top_k=5: {
            "paper_ids": [], "papers": [], "retrieval_mode": "fts", "degraded": None
        }
        # 把 build_plan 临时改成只返回 retrieve 一步
        original = agent.build_plan
        agent.build_plan = lambda goal, top_k=5: [
            PlanStep(id=1, tool="retrieve_library", reason="test",
                     args={"query": goal, "top_k": top_k})
        ]
        try:
            r = agent.run_agent("LLM Agent 方法论", dry_run=False, persist=False, timeout_s=5)
            self.assertEqual(r.status, "needs_clarification")
            self.assertTrue(r.clarification_questions)
            q = r.clarification_questions[0]
            self.assertEqual(q.qid, "no_match")
        finally:
            agent.build_plan = original

    def test_step2_user_hint_resumes_run(self):
        """用户答完 cite 后重跑，intent_hint=cite → 走 cite plan → 不会再次 ASK_USER。"""
        # cite plan 是 [retrieve_library, recommend_citations] 两步
        # 用 mock 让 retrieve 返回空 papers（不触发 degraded），让 recommend 返回成功
        tools.TOOL_REGISTRY["retrieve_library"] = lambda query, top_k=5: {
            "paper_ids": [], "papers": [], "retrieval_mode": "fts", "degraded": None
        }
        tools.TOOL_REGISTRY["recommend_citations"] = lambda paragraph, top_k=5: {
            "candidates": [{"paper_id": 1, "title": "X", "verified": True}]
        }
        # 用户的原始目标——模糊（不指定 cite 动词），但 intent_hint=cite 锁定
        r = agent.run_agent("处理这一段", dry_run=False, persist=False,
                             timeout_s=5, intent_hint="cite")
        # 应该走完 cite 路径，状态 completed 或 failed（不是 needs_clarification）
        self.assertNotEqual(r.status, "needs_clarification",
                            f"intent_hint 应该避免再次 ASK_USER，但状态是 {r.status}")
        self.assertEqual(r.intent.primary, "cite")
        self.assertEqual(r.intent.confidence, 1.0)

    def test_intent_hint_persists_across_steps(self):
        """intent_hint 让所有步骤都按锁定意图走，不会中途被 Replanner 改成别的。"""
        tools.TOOL_REGISTRY["retrieve_library"] = lambda query, top_k=5: {
            "paper_ids": [1], "papers": [{"id": 1}]
        }
        tools.TOOL_REGISTRY["recommend_citations"] = lambda paragraph, top_k=5: {
            "candidates": [{"paper_id": 1, "verified": True}]
        }
        r = agent.run_agent("处理这一段", dry_run=False, persist=False,
                             timeout_s=5, intent_hint="cite")
        # plan 应该是 cite 路径（retrieve + recommend）
        tools_used = [s.tool for s in r.plan]
        self.assertIn("recommend_citations", tools_used)


class ClarificationAPIContractTests(TempDbTestCase):
    """API 层：AgentRequest 接受 intent_hint，透传到 run_agent。"""

    def test_agent_run_endpoint_accepts_intent_hint(self):
        # 不真跑工具，只验证请求体格式与 AgentRequest 字段对齐
        from papernest.schemas import AgentRequest
        req = AgentRequest(goal="g", intent_hint="cite")
        # Pydantic 验证
        self.assertEqual(req.intent_hint, "cite")
        # run_agent 接受
        r = agent.run_agent(req.goal, req.top_k, req.max_steps, req.topic,
                             dry_run=True, persist=False,
                             intent_hint=req.intent_hint)
        self.assertEqual(r.intent.primary, "cite")


if __name__ == "__main__":
    unittest.main()
