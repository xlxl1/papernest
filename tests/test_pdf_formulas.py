import json
import shutil
import tempfile
import unittest
from pathlib import Path

from papernest import config, db, formulas


class FormulaDetectionTests(unittest.TestCase):
    def test_formula_number_and_superscript_are_detected(self):
        raw = {"blocks": [{"type": 0, "lines": [{
            "dir": (1, 0), "bbox": (70, 100, 300, 125),
            "spans": [
                {"text": "E = mc", "font": "STIXGeneral", "size": 12,
                 "bbox": (70, 100, 145, 120)},
                {"text": "2", "font": "STIXGeneral", "size": 8,
                 "bbox": (146, 96, 151, 106)},
                {"text": " (3)", "font": "Times", "size": 12,
                 "bbox": (260, 100, 300, 120)},
            ]}]}]}
        items = formulas.candidates_from_page_dict(raw, 2)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["formula_no"], "3")
        self.assertIn("super_or_subscript", items[0]["reasons"])
        self.assertEqual(items[0]["status"], "unresolved")
        self.assertIsNone(items[0]["latex"])

    def test_unresolved_formula_never_uses_display_math(self):
        block = {"block_id": "f1", "type": "formula", "page_no": 3,
                 "bbox": (1, 2, 3, 4), "source": "pdf_text",
                 "confidence": .82, "raw_text": "x 2 + y 2",
                 "status": "unresolved", "latex": None}
        md = formulas.canonical_markdown([block])
        self.assertIn("id=f1 type=formula page=3 bbox=1.00,2.00,3.00,4.00", md)
        self.assertIn("Unresolved formula", md)
        self.assertNotIn("$$", md)


class FormulaMigrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="papernest_formula_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.old = config.DB_PATH, config.DATA_DIR
        config.DATA_DIR = Path(self.tmp)
        config.DB_PATH = config.DATA_DIR / "test.db"
        self.addCleanup(self._restore)

    def _restore(self):
        config.DB_PATH, config.DATA_DIR = self.old

    def test_migration_and_formula_replace_are_idempotent(self):
        db.init_db(force=True)
        db.init_db(force=True)
        with db.conn() as c:
            cols = {r[1] for r in c.execute("PRAGMA table_info(formulas)")}
            self.assertIn("parser_version", cols)
            item = {"block_id": "f1", "page_no": 1, "bbox": (1, 2, 3, 4),
                    "raw_text": "x=1 (1)", "formula_no": "1",
                    "source": "pdf_text", "confidence": .8,
                    "latex": None, "latex_confidence": None,
                    "status": "unresolved"}
            db.replace_formulas(c, 99, [item], formulas.PARSER_VERSION)
            db.replace_formulas(c, 99, [item], formulas.PARSER_VERSION)
            n = c.execute("SELECT COUNT(*) n FROM formulas WHERE paper_id=99").fetchone()["n"]
            self.assertEqual(n, 1)
            db.save_document_artifact(c, 99, "formula_audit", formulas.PARSER_VERSION,
                                      json.dumps(item), {"degraded": "ocr_no_geometry"})
            row = db.document_artifact(c, 99, "formula_audit")
            self.assertIn("ocr_no_geometry", row["metadata_json"])


if __name__ == "__main__":
    unittest.main()
