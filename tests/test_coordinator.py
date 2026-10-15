"""Coordinator contract tests: persistence, replay planning and parallel evidence."""
from __future__ import annotations

import time
import unittest

from papernest import coordinator, db, pipeline
from .support import TempDbTestCase


class CoordinatorTests(TempDbTestCase):
    prefix = "papernest_coordinator"

    def setUp(self):
        super().setUp()
        self.run = pipeline.create_run("coordinator test topic")
        self.c = coordinator.Coordinator(self.run["id"])

    def test_state_and_message_are_replayable(self):
        self.c.initialize(("topic", "writer"))
        self.c.set_state("topic", "running", input={"topic": "x"}, increment_attempt=True)
        self.c.set_state("topic", "done", output={"topics": 2})
        msg = self.c.send("topic", "outline", "topics_ready", {"count": 2},
                          confidence=0.9, requires_review=True, message_id="m1")
        self.assertEqual(msg["message_id"], "m1")
        self.assertEqual(self.c.states()["topic"]["status"], "done")
        self.assertEqual(self.c.states()["topic"]["attempt"], 1)
        message = next(m for m in self.c.messages() if m["message_id"] == "m1")
        self.assertEqual(message["payload"], {"count": 2})
        self.assertTrue(message["requires_review"])

    def test_plan_selects_first_pending_or_failed_agent(self):
        self.c.initialize()
        self.assertEqual(self.c.plan()["next"], "topic")
        self.c.set_state("topic", "done")
        self.assertEqual(self.c.plan()["next"], "outline")
        self.c.set_state("outline", "failed", error="temporary")
        self.assertTrue(self.c.plan()["resumable"])
        self.assertEqual(self.c.plan()["next"], "outline")

    def test_parallel_evidence_preserves_order_and_isolates_failures(self):
        sections = [{"no": 1, "title": "a"}, {"no": 2, "title": "b"},
                    {"no": 3, "title": "c"}]

        def build(section):
            time.sleep(0.01 * (4 - section["no"]))
            if section["no"] == 2:
                raise RuntimeError("bad section")
            return {"section": section["no"]}

        results, errors = self.c.parallel_evidence(sections, build, max_workers=3)
        self.assertEqual(list(results), [1, 3])
        self.assertEqual(results[1], {"section": 1})
        self.assertIn(2, errors)
        self.assertEqual(self.c.states()["evidence:2"]["status"], "failed")
        self.assertEqual(self.c.states()["evidence:1"]["status"], "done")

    def test_running_stage_is_not_replanned_before_completion(self):
        self.c.set_state("topic", "running")
        self.assertEqual(self.c.plan()["next"], "topic")
        self.assertEqual(self.c.plan()["action"], "wait")
        self.assertFalse(self.c.plan()["resumable"])

    def test_major_audit_replans_only_affected_sections(self):
        for agent in coordinator.DEFAULT_AGENTS:
            self.c.set_state(agent, "done")
        self.c.set_state("auditor", "done", output={
            "issues": [{"severity": "major", "issue_type": "numeric_conflict"}],
            "rewrite_sections": [3, 1, 3],
        })
        plan = self.c.plan()
        self.assertEqual(plan["next"], "writer")
        self.assertEqual(plan["action"], "rewrite_sections")
        self.assertEqual(plan["sections"], [1, 3])
        self.assertTrue(plan["requires_review"])
        # A fresh, clean audit clears the repair plan after the local rewrite.
        self.c.set_state("auditor", "done", output={"issues": [], "rewrite_sections": []})
        self.assertIsNone(self.c.plan()["next"])

    def test_major_document_issue_without_section_requires_audit_review(self):
        for agent in coordinator.DEFAULT_AGENTS:
            self.c.set_state(agent, "done")
        self.c.set_state("auditor", "done", output={
            "issues": [{"severity": "major", "issue_type": "claim_conflict"}],
        })
        self.assertEqual(self.c.plan()["action"], "review_audit")
        self.assertEqual(self.c.plan()["next"], "auditor")


if __name__ == "__main__":
    unittest.main()
