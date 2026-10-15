"""Offline contract tests for the P1 evidence agent.

The fixtures deliberately use a temporary SQLite database and only the persisted
paper/page/chunk/artifact tables.  No embedding service or LLM is required.
"""
from __future__ import annotations

import json
import unittest
from unittest import mock

from papernest import db, evidence, pipeline

try:                                        # tests/ is not a package
    from .support import TempDbTestCase
except ImportError:                         # discover -s tests
    from support import TempDbTestCase


class EvidenceAgentOfflineTests(TempDbTestCase):
    prefix = "papernest_evidence"

    def setUp(self):
        super().setUp()
        db.init_db()
        with db.conn() as c:
            self.p1 = db.insert_l0(c, {
                "norm_key": "fixture:p1", "title": "Tool-use evaluation",
                "abstract": "We evaluate long-horizon tool use reliability.",
                "year": 2024, "venue": "Fixture", "authors": ["A. Author"],
                "doi": None, "arxiv_id": None, "source": "fixture",
            })
            self.p2 = db.insert_l0(c, {
                "norm_key": "fixture:p2", "title": "Unrelated paper",
                "abstract": "Ocean temperature records are analysed.",
                "year": 2023, "venue": "Fixture", "authors": ["B. Author"],
                "doi": None, "arxiv_id": None, "source": "fixture",
            })
            # Pages are the source of truth for quote verification.
            c.execute("INSERT INTO pages(paper_id,page_no,text) VALUES(?,?,?)", (
                self.p1, 4, "Long-horizon agents need robust tool-use evaluation."))
            c.execute("INSERT INTO pages(paper_id,page_no,text) VALUES(?,?,?)", (
                self.p1, 5, "Success rate by planning horizon: 82% at 4 steps. "
                "R = successes / trials"))
            c.execute("INSERT INTO pages(paper_id,page_no,text) VALUES(?,?,?)", (
                self.p1, 6, "Success rate versus planning horizon."))
            # Populate the FTS mirror as a real import would do.
            db.replace_chunks(c, self.p1, [
                {"section_path": "3 Evaluation", "start_page": 4,
                 "end_page": 4, "kind": "text", "block_type": "text",
                 "text": "Long-horizon agents need robust tool-use evaluation.",
                 "block_id": "p1-c1"},
                {"section_path": "4 Results", "start_page": 5,
                 "end_page": 5, "kind": "table", "block_type": "table",
                 "text": "Success rate by planning horizon: 82% at 4 steps.",
                 "block_id": "p1-table-1"},
            ])
            c.execute("""INSERT INTO formulas(
                         paper_id,block_id,page_no,raw_text,formula_no,source,
                         confidence,latex,latex_confidence,status,metadata_json,parser_version)
                         VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""", (
                self.p1, "p1-formula-1", 5, "R = successes / trials", "(1)",
                "pdf_text", 0.98, r"R = \\frac{successes}{trials}", 0.98,
                "resolved", "{}", "fixture"))
            c.execute("""INSERT INTO table_summaries(
                         table_hash,paper_id,page_no,n_rows,n_cols,summary,model,
                         is_model_written,structure_ok)
                         VALUES(?,?,?,?,?,?,?,?,?)""", (
                "table-hash-1", self.p1, 5, 2, 2,
                "Model summary: success declines with horizon.", "fixture", 1, 1))
            c.execute("""INSERT INTO figure_summaries(
                         figure_hash,paper_id,page_no,label,caption,summary,model,
                         is_model_written)
                         VALUES(?,?,?,?,?,?,?,?)""", (
                "figure-hash-1", self.p1, 6, "Fig. 1",
                "Success rate versus planning horizon.",
                "Model summary: performance declines.", "fixture", 1))
            c.execute("INSERT INTO pages(paper_id,page_no,text) VALUES(?,?,?)", (
                self.p2, 1, "Ocean temperature records are analysed."))
            c.commit()

    def _build(self, section, candidate_ids=None):
        # Keep the public helper under test; this also makes failures clear if
        # the implementation accidentally only exposes the class method.
        return evidence.build_evidence_bundle(section, candidate_ids=candidate_ids)

    def test_bundle_has_structured_claims_and_verified_text_evidence(self):
        section = {"no": 2, "title": "Tool-use evaluation",
                   "points": ["long-horizon tool-use evaluation"],
                   "claims": ["Long-horizon agents need robust evaluation."]}
        bundle = self._build(section, [self.p1])
        self.assertEqual(bundle.section_id, "section-2")
        self.assertEqual(bundle.agent, "evidence_agent")
        self.assertGreater(len(bundle.evidence), 0)
        self.assertGreater(len(bundle.claims), 0)
        self.assertTrue(all(item.paper_id == self.p1 for item in bundle.evidence))
        self.assertTrue(all(item.quote.strip() for item in bundle.evidence))
        self.assertTrue(any(item.block_type == "text" for item in bundle.evidence))
        self.assertTrue(any(item.verified for item in bundle.evidence
                            if item.block_type in {"text", "table", "formula"}))
        for item in bundle.evidence:
            if "Model summary:" in item.quote:
                self.assertFalse(item.verified,
                                 "generated artifact summaries are not source quotes")
        self.assertGreater(bundle.coverage, 0.0)
        # Pydantic serialization is the hand-off format between agents.
        payload = bundle.model_dump()
        self.assertEqual(type(payload["evidence"]), list)
        self.assertEqual(type(payload["claims"]), list)
        self.assertEqual(evidence.EvidenceBundle.model_validate(payload), bundle)

    def test_candidate_ids_are_a_hard_paper_boundary(self):
        section = {"no": 1, "title": "temperature", "points": ["ocean"]}
        bundle = self._build(section, [999999])
        self.assertEqual(bundle.evidence, [])
        self.assertEqual(bundle.coverage, 0.0)
        unrestricted = self._build(section, [self.p2])
        self.assertTrue(all(x.paper_id == self.p2 for x in unrestricted.evidence))

    def test_table_formula_and_figure_rows_keep_source_metadata(self):
        section = {"no": 3, "title": "success rate planning horizon",
                   "points": ["success rate", "formula", "figure"]}
        bundle = self._build(section, [self.p1])
        kinds = {x.block_type for x in bundle.evidence}
        self.assertIn("table", kinds)
        self.assertIn("formula", kinds)
        # Figure captions are source text; generated figure summaries must not
        # be silently presented as verified paper quotes.
        self.assertIn("figure", kinds)
        by_kind = {k: next(x for x in bundle.evidence if x.block_type == k)
                   for k in kinds}
        self.assertTrue(by_kind["table"].page)
        self.assertTrue(by_kind["formula"].page)
        self.assertTrue(by_kind["figure"].page)
        self.assertTrue(by_kind["figure"].quote)
        self.assertTrue(any(x.verified for x in bundle.evidence
                            if x.block_type in {"table", "formula", "figure"}))

    def test_empty_section_is_safe_and_does_not_touch_external_services(self):
        with mock.patch("papernest.embeddings.embed_texts",
                        side_effect=AssertionError("network/embedding call")), \
             mock.patch("papernest.llm.chat",
                        side_effect=AssertionError("network/LLM call")):
            bundle = self._build({"no": 4, "title": "", "points": []}, [])
        self.assertEqual(bundle.evidence, [])
        self.assertEqual(bundle.claims, [])
        self.assertEqual(bundle.coverage, 0.0)

    def test_pipeline_wrapper_returns_same_contract(self):
        section = {"no": 5, "title": "tool-use evaluation",
                   "points": ["long-horizon"]}
        with mock.patch.object(pipeline, "build_evidence_bundle",
                               wraps=evidence.build_evidence_bundle) as wrapped:
            bundle = pipeline.build_evidence_bundle(section, [self.p1])
        self.assertEqual(bundle.section_id, "section-5")
        wrapped.assert_called_once_with(section, [self.p1])

    def test_write_one_persists_bundle_for_replay(self):
        """A section hand-off must survive process restarts in sections."""
        run = pipeline.create_run("tool-use evaluation")
        section = {"no": 1, "title": "Tool-use evaluation",
                   "points": ["long-horizon tool-use evaluation"],
                   "paper_ids": [self.p1], "words": 120}
        with db.conn() as c:
            c.execute("INSERT INTO outlines(run_id,version,title,outline_json) "
                      "VALUES(?,?,?,?)",
                      (run["id"], 1, "Tool use", json.dumps([section])))
        result = pipeline._write_one(pipeline.get_run(run["id"]), section, 1, force=True)
        self.assertEqual(result["status"], "done")
        row = pipeline._latest_sections(run["id"])[0]
        stored = json.loads(row["evidence_json"])
        self.assertEqual(stored["section_id"], "section-1")
        self.assertGreaterEqual(len(stored["evidence"]), 1)
        self.assertIn("claims", stored)

    def test_p2_writer_boundary_keeps_only_verified_claim_sources(self):
        bundle = {"claims": [
            {"claim_id": "c1", "text": "supported", "evidence_ids": ["e1", "e2"]},
        ], "evidence": [
            {"evidence_id": "e1", "paper_id": self.p1, "quote": "verified", "verified": True},
            {"evidence_id": "e2", "paper_id": self.p2, "quote": "unverified", "verified": False},
        ]}
        self.assertEqual(pipeline._claim_bound_paper_ids(bundle), {self.p1})
        self.assertEqual(pipeline._claim_bound_paper_ids({"claims": [], "evidence": []}), set())

    def test_writer_prompt_omits_unverified_bound_hits(self):
        captured = {}
        run = {"id": "run", "glossary": {}}
        section = {"no": 1, "title": "Evidence", "points": ["supported"]}
        bundle = {"claims": [{"claim_id": "c1", "text": "supported",
                              "evidence_ids": ["e1", "e2"]}],
                  "evidence": [
                      {"evidence_id": "e1", "quote": "verified quote", "verified": True,
                       "page": 1, "block_type": "text"},
                      {"evidence_id": "e2", "quote": "do not expose", "verified": False,
                       "page": 2, "block_type": "text"},
                  ]}
        with mock.patch.object(pipeline.writing, "_paper_context", return_value=("", [])), \
             mock.patch.object(pipeline.llm, "chat",
                               side_effect=lambda system, user, **kwargs:
                               captured.setdefault("user", user) or "draft"):
            pipeline._draft_section(run, section, [], 1, None, bundle)
        self.assertIn("verified quote", captured["user"])
        self.assertNotIn("do not expose", captured["user"])


if __name__ == "__main__":
    unittest.main()


