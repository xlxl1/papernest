import tempfile
import unittest
from pathlib import Path

import pymupdf

from papernest import config, db, formulas, fulltext
from tests.support import TempDbTestCase


def _formula_pdf(path: Path):
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 80), "1 Method", fontsize=14, fontname="hebo")
    page.insert_text((72, 120), "We minimize the following objective.", fontsize=10)
    # Separate small superscript spans give the detector geometric script evidence.
    page.insert_text((145, 180), "L = x", fontsize=13, fontname="symbol")
    page.insert_text((183, 174), "2", fontsize=8)
    page.insert_text((197, 180), "+ y", fontsize=13, fontname="symbol")
    page.insert_text((233, 174), "2", fontsize=8)
    page.insert_text((480, 180), "(1)", fontsize=10)
    page.insert_text((72, 220), "where x and y are normalized variables.", fontsize=10)
    doc.save(path)
    doc.close()


class FormulaDetectionTests(unittest.TestCase):
    def test_detects_formula_with_page_bbox_and_audit_status(self):
        with tempfile.TemporaryDirectory(prefix="papernest_formula_") as tmp:
            path = Path(tmp) / "formula.pdf"
            _formula_pdf(path)
            result = formulas.detect_pdf(path)
        self.assertEqual(result["source"], "pdf_text")
        self.assertTrue(result["formulas"])
        item = result["formulas"][0]
        self.assertEqual(item["page_no"], 1)
        self.assertEqual(len(item["bbox"]), 4)
        self.assertEqual(item["status"], "unresolved")
        self.assertIsNone(item["latex"])

    def test_unresolved_formula_is_not_emitted_as_latex(self):
        item = {
            "block_id": "f1", "page_no": 2, "bbox": (1, 2, 3, 4),
            "raw_text": "x 2 + y 2", "source": "pdf_text",
            "confidence": .7, "status": "unresolved",
            "latex": None, "latex_confidence": None,
        }
        md = formulas.canonical_markdown([item | {"type": "formula"}])
        self.assertIn("Unresolved formula", md)
        self.assertIn("x 2 + y 2", md)
        self.assertNotIn("$$", md)
        self.assertIn("page=2", md)


class FormulaPersistenceTests(TempDbTestCase):
    prefix = "papernest_formula_db"

    def setUp(self):
        super().setUp()
        db.init_db()
        self.path = self.tmp / "formula.pdf"
        _formula_pdf(self.path)
        with db.conn() as c:
            self.pid = db.insert_l0(c, {
                "norm_key": "title:formula-test", "title": "Formula Test",
                "abstract": "objective function", "year": 2026, "venue": None,
                "authors": [], "doi": None, "arxiv_id": None,
                "citation_count": None, "oa_pdf_url": None,
                "source": "test", "s2_id": None,
            })

    def test_extract_is_idempotent_and_builds_canonical_markdown(self):
        fulltext.extract_pages(str(self.path), self.pid)
        fulltext.extract_pages(str(self.path), self.pid)
        with db.conn() as c:
            rows = db.formulas_for_paper(c, self.pid)
            artifact = db.document_artifact(c, self.pid, "canonical_markdown",
                                            formulas.PARSER_VERSION)
            formula_chunks = c.execute(
                "SELECT COUNT(*) n FROM chunks WHERE paper_id=? AND block_type='formula'",
                (self.pid,)).fetchone()["n"]
        self.assertTrue(rows)
        self.assertEqual(len({r["block_id"] for r in rows}), len(rows))
        self.assertEqual(formula_chunks, len(rows))
        self.assertIsNotNone(artifact)
        self.assertIn("papernest:block", artifact["content"])
        self.assertIn("Unresolved formula", artifact["content"])
        self.assertIn("# 1 Method", artifact["content"])

    def test_rebuild_after_loading_chunks_preserves_formula_evidence(self):
        fulltext.extract_pages(str(self.path), self.pid)
        with db.conn() as c:
            markdown = fulltext.rebuild_canonical_markdown(c, self.pid)
        self.assertIn("Unresolved formula", markdown)
        self.assertNotIn("Raw PDF text", markdown)
        self.assertIn("page=1", markdown)


if __name__ == "__main__":
    unittest.main()
