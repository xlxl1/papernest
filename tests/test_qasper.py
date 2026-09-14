import shutil
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from papernest import config, db, qasper


EVIDENCE = "Simulation results show our method reduces MSE by 30 percent compared with least squares."


def _fixture_paper(pid: str, title: str) -> dict:
    """合成一篇迷你 QASPER 论文，**结构与官方 v0.3 一致**。

    这个夹具原来用的是更早的格式（full_text 是 {section_names, sections}、
    证据挂在 qa["evidence"] 上）。真实 v0.3 里 full_text 是 list、证据埋在
    qas[i].answers[j].answer 下——夹具与真实数据形态脱节，于是解析器对着
    真数据一条全文都导不进来、一条证据都读不出来，而四个用例全绿。
    """
    return {
        "title": title,
        "abstract": "We study near-field channel estimation with deep generative models.",
        "full_text": [
            {"section_name": "Introduction",
             "paragraphs": ["Near-field channel estimation for extremely large-scale MIMO "
                            "requires new models because the spherical wavefront "
                            "invalidates far-field assumptions."]},
            {"section_name": "Method",
             "paragraphs": ["We propose a deep generative model with visibility-region priors.",
                            EVIDENCE]},
        ],
        "qas": [{
            "question_id": "q1",
            "question": "How much does the proposed method reduce MSE compared with least squares?",
            "answers": [{
                "annotation_id": "a1", "worker_id": "w1",
                "answer": {"unanswerable": False, "yes_no": None,
                           "free_form_answer": "30 percent",
                           "extractive_spans": ["BIBREF3"],      # 太短，不该被当证据
                           "evidence": [EVIDENCE],
                           "highlighted_evidence": [EVIDENCE]},
            }],
        }],
    }


def _legacy_fixture_paper(title: str) -> dict:
    """更早的 QASPER 格式——解析器要向后兼容它。"""
    return {
        "title": title, "abstract": "legacy abstract",
        "full_text": {"section_names": ["Introduction", "Method"],
                      "sections": ["legacy intro body", "legacy method body"]},
        "qas": [],
    }


class QasperTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="papernest_qa_")
        self.addCleanup(shutil.rmtree, self._tmp, True)
        self._old_db, self._old_dir = config.DB_PATH, qasper.QASPER_DIR
        config.DB_PATH = Path(self._tmp) / "test.db"
        qasper.QASPER_DIR = Path(self._tmp)
        db.init_db()
        # 直接落一个合成 dev.json，绕过下载
        data = {"p1": _fixture_paper("p1", "Deep Generative Near-Field Estimation")}
        (Path(self._tmp) / "qasper-dev-v1.json").write_text(
            json.dumps(data), encoding="utf-8")

    def tearDown(self):
        config.DB_PATH, qasper.QASPER_DIR = self._old_db, self._old_dir

    def test_download_uses_cache(self):
        r = qasper.download()
        self.assertTrue(r["cached"])
        self.assertEqual(r["papers_available"], 1)

    def test_import_creates_paper_with_sections(self):
        r = qasper.import_papers(10)
        self.assertEqual(r["imported"], 1)
        self.assertEqual(r["sections"], 2)
        with db.conn() as c:
            row = c.execute("SELECT id, level, norm_key FROM papers WHERE norm_key='qasper:p1'").fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row["level"], 2)
            n = c.execute("SELECT COUNT(*) n FROM pages WHERE paper_id=?",
                          (row["id"],)).fetchone()["n"]
            self.assertEqual(n, 2)

    def test_eval_finds_evidence(self):
        qasper.import_papers(10)
        r = qasper.run_eval(k=3, sec_k=1, max_papers=10)
        self.assertEqual(r["n_questions"], 1)
        # 合成数据里证据就在与问题重合度最高的 Method 节，两层都应命中
        self.assertEqual(r["paper_hit_at_k"], 1.0)
        self.assertEqual(r["evidence_recall_at_sec_k"], 1.0)

    def test_eval_requires_import(self):
        # dev.json 存在但未导入库 → 0 题，指标应为 0 而非崩溃
        r = qasper.run_eval(k=3, max_papers=10)
        self.assertEqual(r["n_questions"], 0)
        self.assertEqual(r["paper_hit_at_k"], 0)

    # ── 解析器与真实数据形态 ──

    def test_sections_parse_v03_list_format(self):
        secs = qasper._sections_of(_fixture_paper("p1", "T"))
        self.assertEqual([name for name, _ in secs], ["Introduction", "Method"])
        self.assertIn(EVIDENCE, secs[1][1])          # 同一节的多个段落要拼在一起

    def test_sections_parse_legacy_dict_format(self):
        secs = qasper._sections_of(_legacy_fixture_paper("T"))
        self.assertEqual([name for name, _ in secs], ["Introduction", "Method"])

    def test_sections_of_unknown_shape_returns_empty(self):
        """结构认不出时宁可没有全文，也不要把猜错的东西塞进库。"""
        self.assertEqual(qasper._sections_of({"full_text": "一段裸字符串"}), [])
        self.assertEqual(qasper._sections_of({}), [])

    def test_evidence_map_reads_v03_answer_objects(self):
        ev = qasper._evidence_map(_fixture_paper("p1", "T"))
        self.assertIn("q1", ev)
        _q, spans = ev["q1"]
        self.assertIn(EVIDENCE, spans)
        self.assertNotIn("BIBREF3", spans)           # 短的 extractive_span 不算证据

    def test_import_indexes_pages_for_library_search(self):
        """QASPER 全文导入后必须能被全库检索命中，否则这批正文白导。"""
        qasper.import_papers(10)
        with db.conn() as c:
            self.assertTrue(db.search_pages_fts(c, "visibility-region priors", 5))


if __name__ == "__main__":
    unittest.main()


class GoldProbesStripWhatPdfCanNeverContainTests(unittest.TestCase):
    """QASPER 的 gold 证据抽自论文的 **LaTeX 源码**，我们的正文抽自 **PDF**。

    LaTeX 里的占位 token 在 PDF 里是渲染好的东西，两边**永远对不上**：

        BIBREF19        → (Zoph et al., 2016)
        $p(w_i|t_j)$    → 排版好的公式字形
        DBLP:conf/...   → 正常的引用文字

    真库实测（405 条 gold）：35.3% 含 BIBREF 一类占位符；含数学记号的探针
    达成率只有 **3.9%（13/334）**。不切开就等于把这些条目白白算进分母——
    **`span_recall` 的天花板被自己压低了**，据它调检索参数就是在追假象。

    切开之后（274 篇 / 4542 条探针）：相对上界的达成率 **55.8% → 59.9%**，
    问题召回 0.2704 → 0.2803，片段召回 0.1247 → 0.1306。

    只取**最长**的那一段是防灌水：一条 gold 被引用切成几截时，
    短截片（"we compare our approaches with"）到处都能命中。
    """

    def test_bibref_is_stripped(self):
        p = qasper.gold_probes(
            "we compare with BIBREF19, and cross-lingual transfer without pretraining")
        self.assertTrue(p)
        self.assertNotIn("bibref", p[0])

    def test_inline_math_is_stripped(self):
        p = qasper.gold_probes(
            "sorted by $p(w_i|t_j)$, the probability of word i in topic j here")
        self.assertTrue(p)
        self.assertNotIn("$", p[0])

    def test_latex_commands_are_stripped(self):
        p = qasper.gold_probes(
            r"we set \alpha to 0.5 and evaluate on the standard benchmark suite")
        self.assertTrue(p)
        self.assertNotIn("alpha", p[0])

    def test_dblp_keys_are_stripped_case_insensitively(self):
        p = qasper.gold_probes(
            "DBLP:conf/naacl/AnastasopoulosC18 proposed a triangle multi-task strategy")
        self.assertTrue(p)
        self.assertNotIn("dblp", p[0])

    def test_a_bracketed_reference_is_not_mistaken_for_latex(self):
        r"""`\[a-zA-Z]+` 少写一个反斜杠就会把 `table [3]` 当成 LaTeX 命令切掉。"""
        p = qasper.gold_probes(
            "results in table [3] show a clear improvement over the baseline system")
        self.assertTrue(p)
        self.assertIn("table[3]", p[0])

    def test_plain_prose_is_untouched(self):
        s = "a perfectly ordinary sentence of sufficient length with nothing special"
        self.assertEqual(qasper.gold_probes(s), [qasper._norm(s)])

    def test_only_the_longest_fragment_is_kept(self):
        """短截片到处都能命中，留着会把指标灌水。"""
        p = qasper.gold_probes(
            "we show BIBREF1 that the proposed multilingual transfer approach "
            "consistently improves over every baseline we tried BIBREF2 ok")
        self.assertEqual(len(p), 1)
        self.assertIn("consistentlyimproves", p[0])

    def test_a_span_that_is_all_placeholders_yields_nothing(self):
        self.assertEqual(qasper.gold_probes("BIBREF1 BIBREF2 FLOAT SELECTED"), [])

    def test_fragments_below_the_floor_are_dropped(self):
        self.assertGreaterEqual(qasper.MIN_PROBE_CHARS, 20)
        self.assertEqual(qasper.gold_probes("BIBREF1 short bit BIBREF2"), [])


class EvalFixturesStayOutOfTheLibraryTests(unittest.TestCase):
    """274 份评测 PDF **不进用户的文献库**。

    它们是夹具：混进 `papers` 表会污染检索（凭空多出几百篇 NLP 论文参与召回），
    也会让库里的统计口径失真。`eval_tasks()` 直接从目录按 arXiv id 读。
    """

    def test_eval_tasks_reads_the_directory_directly(self):
        import inspect
        src = inspect.getsource(qasper.eval_tasks)
        self.assertIn("EVAL_PDF_DIR", src)
        self.assertIn("glob", src)

    def test_the_fetcher_does_not_tell_you_to_import_them(self):
        src = (config.ROOT / "tools" / "fetch_qasper_pdfs.py").read_text(encoding="utf-8")
        self.assertNotIn("cli.py import-pdf", src,
                         "又在提示把评测夹具导进文献库了")

    def test_the_directory_is_gitignored(self):
        gi = (config.ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("data/qasper_pdf", gi)
