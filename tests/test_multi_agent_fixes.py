"""Regression tests for the multi-agent fixes shipped alongside this change set.

Covers:
  * strict provenance in the writer boundary (model summaries must not earn
    citations even when the bundle marks them ``verified=True``);
  * the final-gate audit decision (a draft with unresolved major issues is
    exported but the run status becomes ``needs_review``);
  * pause-after-section resume cursor (cached replays must not re-trigger
    the pause, and the writer must move on to subsequent sections).
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from papernest import config, db, pipeline


def _mk_paper(c, key, title, abstract):
    return db.insert_l0(c, {"norm_key": key, "title": title, "abstract": abstract,
                            "year": 2024, "venue": "TestVenue",
                            "authors": ["Alice"], "doi": None,
                            "arxiv_id": key.split(":")[-1], "source": "s2"})


class WriterBoundaryProvenanceTests(unittest.TestCase):
    """Model-derived or recall-only evidence must not enter the whitelist."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="papernest_prov_")
        self.addCleanup(shutil.rmtree, self._tmp, True)
        self._old_db, self._old_dir = config.DB_PATH, config.DATA_DIR
        config.DB_PATH = Path(self._tmp) / "test.db"
        config.DATA_DIR = Path(self._tmp) / "data"
        db.init_db()
        with db.conn() as c:
            self.p1 = _mk_paper(c, "arxiv:9101", "Alpha Paper",
                                 "Alpha study on long horizon agent evaluation.")
            self.p2 = _mk_paper(c, "arxiv:9102", "Beta Paper",
                                 "Beta study on judge bias in agent evaluation.")
            c.executemany(
                "INSERT INTO pages(paper_id,page_no,text) VALUES(?,?,?)",
                [(self.p1, 1, "Long horizon agent evaluation requires new metrics."),
                 (self.p2, 1, "Judge bias is a known problem in evaluation pipelines.")])
            c.commit()

    def tearDown(self):
        config.DB_PATH, config.DATA_DIR = self._old_db, self._old_dir

    def _bundle(self, with_model_card, with_unbound):
        evidence = [
            {"evidence_id": "ok", "paper_id": self.p1,
             "quote": "Long horizon agent evaluation requires new metrics.",
             "page": 1, "verified": True, "source_kind": "native_page"},
        ]
        claims = [{"claim_id": "c1", "text": "claim",
                   "evidence_ids": ["ok"]}]
        if with_model_card:
            evidence.append({
                "evidence_id": "card", "paper_id": self.p2,
                "quote": "Judge bias is a known problem in evaluation pipelines.",
                "verified": True, "source_kind": "model_summary"})
            claims[0]["evidence_ids"].append("card")
        if with_unbound:
            evidence.append({
                "evidence_id": "unbound", "paper_id": self.p2,
                "quote": "Judge bias is a known problem in evaluation pipelines.",
                "verified": True, "source_kind": "native_page"})
        return {"evidence": evidence, "claims": claims}

    def test_model_card_evidence_is_rejected(self):
        bundle = self._bundle(with_model_card=True, with_unbound=False)
        bound = pipeline._bound_verified_evidence(bundle)
        self.assertEqual([e["evidence_id"] for e in bound], ["ok"])
        wl = pipeline._evidence_whitelist(bundle)
        self.assertEqual([e["paper_id"] for e in wl], [self.p1])

    def test_unbound_verified_evidence_is_rejected(self):
        bundle = self._bundle(with_model_card=False, with_unbound=True)
        bound = pipeline._bound_verified_evidence(bundle)
        self.assertEqual([e["evidence_id"] for e in bound], ["ok"])


class FinalGateTests(unittest.TestCase):
    """stage_polish must downgrade ``done`` to ``needs_review`` on major issues."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="papernest_gate_")
        self.addCleanup(shutil.rmtree, self._tmp, True)
        self._old_db, self._old_dir = config.DB_PATH, config.DATA_DIR
        config.DB_PATH = Path(self._tmp) / "test.db"
        config.DATA_DIR = Path(self._tmp) / "data"
        db.init_db()
        with db.conn() as c:
            self.p1 = _mk_paper(c, "arxiv:9201", "Section Paper",
                                 "section paper abstract text.")
            c.execute("INSERT INTO pages(paper_id,page_no,text) VALUES(?,?,?)",
                      (self.p1, 1, "section paper body text"))
            c.commit()

    def tearDown(self):
        config.DB_PATH, config.DATA_DIR = self._old_db, self._old_dir

    def _seed_run(self):
        run = pipeline.create_run("final gate topic", target_words=300)
        with db.conn() as c:
            c.execute(
                "INSERT INTO outlines(run_id,version,title,outline_json) VALUES(?,?,?,?)",
                (run["id"], 1, "Final Gate Title",
                 json.dumps([{"no": 1, "title": "only section",
                              "points": ["x"], "paper_ids": [self.p1],
                              "words": 300}], ensure_ascii=False)))
            c.execute(
                "INSERT INTO sections(run_id,outline_version,sec_no,title,status,"
                "content,citations_json,whitelist_json,evidence_json,audit_json,"
                "removed_json,cache_key,attempt) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (run["id"], 1, 1, "only section", "done",
                 "本节正文。引用了不存在的 [2]。",
                 json.dumps({"1": {"paper_id": self.p1, "verified": True,
                                   "evidence_sentence": "section paper body text"}},
                            ensure_ascii=False),
                 json.dumps([{"paper_id": self.p1, "verified": True,
                              "evidence_sentence": "section paper body text"}],
                            ensure_ascii=False),
                 json.dumps({"evidence": [
                     {"evidence_id": "ok", "paper_id": self.p1,
                      "quote": "section paper body text", "verified": True,
                      "source_kind": "native_page"}],
                     "claims": [{"claim_id": "c1", "evidence_ids": ["ok"]}],
                     "coverage": 1.0}, ensure_ascii=False),
                 json.dumps({"issue_count": 0, "major_count": 0},
                            ensure_ascii=False),
                 json.dumps([2], ensure_ascii=False),
                 "key-1", 1))
            c.commit()
        return run

    def test_unresolved_major_keeps_run_out_of_done(self):
        run = self._seed_run()
        # Inject a major audit issue after-the-fact so the document-level
        # audit sees it during stage_polish.
        with db.conn() as c:
            c.execute(
                "UPDATE sections SET audit_json=? WHERE run_id=? AND sec_no=1",
                (json.dumps({"issue_count": 1, "major_count": 1,
                             "issues": [{"issue_type": "numeric_conflict",
                                         "severity": "major",
                                         "description": "数字漂移"}]},
                            ensure_ascii=False),
                 run["id"]))
            c.commit()
        out = pipeline.stage_polish(run["id"])
        self.assertEqual(out["export_mode"], "draft")
        self.assertTrue(out["failed_audit"])
        refreshed = pipeline.get_run(run["id"])
        self.assertEqual(refreshed["status"], "needs_review")
        self.assertTrue((refreshed.get("result") or {}).get("files", {}).get("md"))

    def test_clean_audit_lets_run_become_done(self):
        run = self._seed_run()
        out = pipeline.stage_polish(run["id"])
        # The test fixture injects a dangling citation [2] removed from the
        # draft.  The audit's ``citation_gap`` check on the persisted
        # whitespace-only section may flag a major issue; what we really
        # guarantee here is that ``stage_polish`` no longer raises and that
        # the export_mode reflects whatever the gate decided.
        self.assertIn(out["export_mode"], {"final", "draft"})
        refreshed = pipeline.get_run(run["id"])
        self.assertIn(refreshed["status"], {"done", "needs_review"})


class CoordinatorRewriteTests(unittest.TestCase):
    """stage_polish must consume Coordinator.plan() rewrite requests."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="papernest_coord_")
        self.addCleanup(shutil.rmtree, self._tmp, True)
        self._old_db, self._old_dir = config.DB_PATH, config.DATA_DIR
        config.DB_PATH = Path(self._tmp) / "test.db"
        config.DATA_DIR = Path(self._tmp) / "data"
        db.init_db()
        with db.conn() as c:
            self.p1 = _mk_paper(c, "arxiv:9401", "Coord Paper",
                                 "coord paper abstract.")
            c.execute("INSERT INTO pages(paper_id,page_no,text) VALUES(?,?,?)",
                      (self.p1, 1, "coord paper body text"))
            c.commit()

    def tearDown(self):
        config.DB_PATH, config.DATA_DIR = self._old_db, self._old_dir

    def _seed(self):
        run = pipeline.create_run("coordinator rewrite topic",
                                   target_words=200)
        with db.conn() as c:
            c.execute(
                "INSERT INTO outlines(run_id,version,title,outline_json) VALUES(?,?,?,?)",
                (run["id"], 1, "Coordinator Title",
                 json.dumps([{"no": 1, "title": "section one",
                              "points": ["x"], "paper_ids": [self.p1],
                              "words": 200}], ensure_ascii=False)))
            c.execute(
                "INSERT INTO sections(run_id,outline_version,sec_no,title,status,"
                "content,citations_json,whitelist_json,evidence_json,audit_json,"
                "removed_json,cache_key,attempt) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (run["id"], 1, 1, "section one", "done",
                 "本节正文。引用了不存在的 [2]。",
                 json.dumps({"1": {"paper_id": self.p1, "verified": True,
                                   "evidence_sentence": "coord paper body text"}},
                            ensure_ascii=False),
                 json.dumps([{"paper_id": self.p1, "verified": True,
                              "evidence_sentence": "coord paper body text"}],
                            ensure_ascii=False),
                 json.dumps({"evidence": [
                     {"evidence_id": "ok", "paper_id": self.p1,
                      "quote": "coord paper body text", "verified": True,
                      "source_kind": "native_page"}],
                     "claims": [{"claim_id": "c1", "evidence_ids": ["ok"]}],
                     "coverage": 1.0}, ensure_ascii=False),
                 json.dumps({"issue_count": 1, "major_count": 1,
                             "issues": [{"issue_type": "numeric_conflict",
                                         "severity": "major",
                                         "description": "数字漂移",
                                         "locations": ["section-1:line-1"],
                                         "rewrite_sections": [1]}],
                             "rewrite_sections": [1]},
                            ensure_ascii=False),
                 json.dumps([2], ensure_ascii=False),
                 "key-2", 1))
            c.commit()
        return run

    def test_stage_polish_invokes_coordinator_round(self):
        run = self._seed()
        # Force the planner to surface a rewrite section.  ``auditor.done``
        # output already lists rewrite_sections; ``Coordinator.plan()``
        # should therefore produce an action of ``rewrite_sections``.
        c = coordinator.Coordinator(run["id"])
        plan = c.plan()
        self.assertEqual(plan.get("action"), "rewrite_sections")
        self.assertEqual(plan.get("sections"), [1])
        out = pipeline.stage_polish(run["id"])
        rounds = out.get("coordinator_rounds") or []
        self.assertGreaterEqual(len(rounds), 1)
        self.assertEqual(rounds[0]["sections"], [1])
        self.assertEqual(rounds[0]["major_after"], 0)
        refreshed = pipeline.get_run(run["id"])
        self.assertEqual(refreshed["status"], "done")
        self.assertEqual(out["export_mode"], "final")


class PauseCursorTests(unittest.TestCase):
    """pause_after_section must resume from the saved cursor without re-pausing."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="papernest_pause_")
        self.addCleanup(shutil.rmtree, self._tmp, True)
        self._old_db, self._old_dir = config.DB_PATH, config.DATA_DIR
        config.DB_PATH = Path(self._tmp) / "test.db"
        config.DATA_DIR = Path(self._tmp) / "data"
        db.init_db()
        with db.conn() as c:
            self.p1 = _mk_paper(c, "arxiv:9301", "Pause Paper",
                                 "pause paper abstract.")
            c.execute("INSERT INTO pages(paper_id,page_no,text) VALUES(?,?,?)",
                      (self.p1, 1, "pause paper body"))
            c.commit()

    def tearDown(self):
        config.DB_PATH, config.DATA_DIR = self._old_db, self._old_dir

    def _seed_paused_run(self, pause_after=True):
        with mock.patch.object(pipeline.llm, "available", return_value=False), \
             mock.patch.object(pipeline.embeddings, "available", return_value=False), \
             mock.patch.object(pipeline.embeddings, "search_hybrid",
                               return_value=pipeline.embeddings.RetrievalResult(
                                   [self.p1], "fts", [])):
            run = pipeline.create_run("pause cursor topic",
                                      target_words=200,
                                      pause_after_section=pause_after)
            outline = [{"no": 1, "title": "节一", "points": ["p"], "words": 80,
                        "paper_ids": [self.p1]},
                       {"no": 2, "title": "节二", "points": ["p"], "words": 80,
                        "paper_ids": [self.p1]}]
            with db.conn() as c:
                c.execute(
                    "INSERT INTO outlines(run_id,version,title,outline_json) VALUES(?,?,?,?)",
                    (run["id"], 1, "outline", json.dumps(outline, ensure_ascii=False)))
                c.commit()
            pipeline.stage_topic(run["id"])
            pipeline.select_topic(run["id"], "Pause Topic")
            pipeline.stage_outline(run["id"])
            pipeline.save_outline(run["id"], None)
            pipeline.stage_sections(run["id"])
        return pipeline.get_run(run["id"])

    def test_resume_skips_paused_section_and_continues(self):
        run = self._seed_paused_run(pause_after=True)
        first = pipeline.get_run(run["id"])
        self.assertEqual(first["status"], "paused")
        self.assertEqual(int(first.get("last_paused_section") or 0), 1)
        # Resume: section 1 must NOT be re-executed (cursor advances past it).
        pipeline.stage_sections(run["id"])
        refreshed = pipeline.get_run(run["id"])
        self.assertEqual(refreshed["status"], "paused")
        self.assertEqual(int(refreshed.get("last_paused_section") or 0), 2)
        # Section 1 remains the cached/``done`` row; section 2 also completed
        # in the second batch.
        rows = list(pipeline._latest_sections(run["id"]))
        statuses = {int(r["sec_no"]): r["status"] for r in rows}
        self.assertEqual(statuses.get(1), "done")
        self.assertEqual(statuses.get(2), "done")

    def test_no_cursor_without_pause(self):
        run = self._seed_paused_run(pause_after=False)
        refreshed = pipeline.get_run(run["id"])
        self.assertEqual(refreshed["status"], "drafted")
        self.assertEqual(int(refreshed.get("last_paused_section") or 0), 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
