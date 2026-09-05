"""Obsidian 文献笔记导出：citekey 命名、frontmatter、双链、不编造。"""
import shutil
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from papernest import config, db, graph, obsidian


class ObsidianExportTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="papernest_obs_")
        self.addCleanup(shutil.rmtree, self._tmp, True)
        self._old_db, self._old_dir = config.DB_PATH, config.DATA_DIR
        config.DB_PATH = Path(self._tmp) / "test.db"
        config.DATA_DIR = Path(self._tmp) / "data"
        self._embed = mock.patch("papernest.embeddings.available", return_value=False)
        self._embed.start()
        db.init_db()
        graph.ensure_schema()
        card = {"tldr": "一句话总结。", "method": "提出了某方法。",
                "limitations": "样本量有限。",
                "keywords": ["channel estimation", "mimo"],
                "key_findings": [{"claim": "MSE 下降 30%", "page": 4, "verified": True},
                                 {"claim": "推理更快", "page": 6, "verified": False}]}
        with db.conn() as c:
            self.p1 = db.insert_l0(c, {
                "norm_key": "doi:10.1/aa", "title": "Deep Channel Estimation",
                "abstract": "We study channel estimation.", "year": 2024,
                "venue": "TWC", "authors": ["Alice Smith", "Bob Lee"],
                "doi": "10.1/aa", "arxiv_id": "2401.00001", "source": "s2"})
            c.execute("UPDATE papers SET card_json=? WHERE id=?",
                      (json.dumps(card, ensure_ascii=False), self.p1))
            self.p2 = db.insert_l0(c, {
                "norm_key": "doi:10.1/bb", "title": "Massive MIMO Fundamentals",
                "abstract": "Fundamentals.", "year": 2010, "venue": "TWC",
                "authors": ["Carol Wu"], "doi": "10.1/bb", "arxiv_id": None,
                "source": "s2"})
            # p1 引用 p2（都在库内 → 应生成双链）；再加一条指向库外的边（不应生成）
            c.execute("INSERT INTO citation_edges(src_key,dst_key) VALUES(?,?)",
                      ("doi:10.1/aa", "doi:10.1/bb"))
            c.execute("INSERT INTO citation_edges(src_key,dst_key) VALUES(?,?)",
                      ("doi:10.1/aa", "doi:10.9/external"))

    def tearDown(self):
        self._embed.stop()
        config.DB_PATH, config.DATA_DIR = self._old_db, self._old_dir

    def test_note_has_citekey_filename_and_frontmatter(self):
        fn, text = obsidian.note_markdown(self.p1)
        self.assertEqual(fn, "smith2024deepchannelestimatio.md")
        self.assertTrue(text.startswith("---\n"))
        self.assertIn('title: "Deep Channel Estimation"', text)
        self.assertIn("year: 2024", text)
        self.assertIn('doi: "10.1/aa"', text)
        self.assertIn("tags: [", text)

    def test_card_fields_render_and_missing_ones_are_omitted(self):
        _fn, text = obsidian.note_markdown(self.p1)
        self.assertIn("## TL;DR", text)
        self.assertIn("## 局限", text)
        self.assertNotIn("## 结果", text)          # 卡片没有 results → 小节不出现，不编造
        self.assertIn("- ✓ (p.4) MSE 下降 30%", text)
        self.assertIn("- ? (p.6) 推理更快", text)  # 未核验的如实标 ?

    def test_in_library_citation_becomes_wikilink(self):
        _fn, text = obsidian.note_markdown(self.p1)
        self.assertIn("[[wu2010massivemimofundamen", text)
        self.assertNotIn("external", text)         # 库外引用不生成双链

    def test_cited_by_direction(self):
        _fn, text = obsidian.note_markdown(self.p2)
        self.assertIn("## 被引于（库内）", text)
        self.assertIn("[[smith2024deepchannelestimatio]]", text)

    def test_export_vault_writes_files(self):
        out = Path(self._tmp) / "vault"
        r = obsidian.export_vault(out)
        self.assertEqual(r["written"], 2)
        files = sorted(p.name for p in out.glob("*.md"))
        self.assertEqual(len(files), 2)
        content = (out / r["files"][0]).read_text(encoding="utf-8")
        self.assertIn("source: PaperNest", content)

    def test_export_subset_and_missing_id(self):
        out = Path(self._tmp) / "vault2"
        r = obsidian.export_vault(out, [self.p1, 99999])
        self.assertEqual(r["written"], 1)
        self.assertEqual(len(r["errors"]), 1)

    def test_doi_url_form_is_normalised(self):
        """OpenAlex 把 DOI 存成完整 URL——库里 455 篇一大半是这个形态（引文图踩过同坑）。"""
        with db.conn() as c:
            pid = db.insert_l0(c, {
                "norm_key": "doi:10.5/cc", "title": "Url Doi Paper",
                "abstract": None, "year": 2020, "venue": None, "authors": ["Dan Qi"],
                "doi": "https://doi.org/10.5/cc", "arxiv_id": None, "source": "openalex"})
        _fn, text = obsidian.note_markdown(pid)
        self.assertIn('doi: "10.5/cc"', text)
        self.assertIn("[DOI](https://doi.org/10.5/cc)", text)
        self.assertNotIn("doi.org/https", text)     # 不许拼出套娃链接

    def test_filename_is_safe_for_windows(self):
        with db.conn() as c:
            pid = db.insert_l0(c, {
                "norm_key": "title:x1", "title": "标题里有：冒号/斜杠?",
                "abstract": None, "year": None, "venue": None,
                "authors": [], "doi": None, "arxiv_id": None, "source": "upload"})
        fn, _text = obsidian.note_markdown(pid)
        for ch in '<>:"/\\|?*':
            self.assertNotIn(ch, fn.replace(".md", "").replace("_", ""))


if __name__ == "__main__":
    unittest.main()
