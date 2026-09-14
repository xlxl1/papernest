"""RCS（重排 + 逐篇定向摘要）：重点钉住它引入的**新失败模式**。

RCS 是加在「检索完成之后、作答之前」的增强件，它比朴素 RAG 多了两处模型判断，
也就多了两处出错的地方。这个文件测的主要不是 happy path，而是：
- 重排失败/乱答/漏评/全同分 → 不能让增强件变成新的故障点；
- 定向摘要把有用的论文判成不相关 → 这是 RCS 最危险的失效（朴素 RAG 至少把原文摆着，
  作答模型自己还能捞；RCS 判错就是彻底消失），必须可见、可降级；
- 依据句校验既不能被 PDF 排版伪影误杀，也不能被跨页拼接骗过。
"""
import shutil
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from papernest import config, db, embeddings, rcs


def _fake_llm(routes):
    """按 purpose 路由的假模型。routes: purpose 前缀 → 返回值（str 或 callable(user)）。"""
    def chat(system, user, purpose="", paper_id=None, temperature=0.3, model=None):
        for prefix, val in routes.items():
            if purpose.startswith(prefix):
                return val(user) if callable(val) else val
        return "{}"
    return chat


def _scores(*pairs) -> str:
    return json.dumps({"scores": [{"id": i, "score": s, "why": "理由"}
                                  for i, s in pairs]}, ensure_ascii=False)


def _summary(relevant=True, summary="要点。", quotes=()) -> str:
    return json.dumps({"relevant": relevant, "summary": summary,
                       "quotes": list(quotes)}, ensure_ascii=False)


class RcsTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="papernest_rcs_")
        self.addCleanup(shutil.rmtree, self._tmp, True)
        self._old_db, self._old_dir = config.DB_PATH, config.DATA_DIR
        config.DB_PATH = Path(self._tmp) / "test.db"
        config.DATA_DIR = Path(self._tmp) / "data"
        self._p = [mock.patch("papernest.embeddings.available", return_value=False)]
        for p in self._p:
            p.start()
        db.init_db()
        with db.conn() as c:
            self.p1 = db.insert_l0(c, {
                "norm_key": "arxiv:1", "title": "Pilot Contamination Survey",
                "abstract": "Pilot contamination limits massive MIMO capacity.",
                "year": 2015, "venue": "V", "authors": [], "doi": None,
                "arxiv_id": "1", "source": "s2"})
            self.p2 = db.insert_l0(c, {
                "norm_key": "arxiv:2", "title": "Unrelated Ocean Study",
                "abstract": "Ocean temperature records from buoys.",
                "year": 2020, "venue": "V", "authors": [], "doi": None,
                "arxiv_id": "2", "source": "s2"})
            self.p3 = db.insert_l0(c, {
                "norm_key": "arxiv:3", "title": "Beamforming Design",
                "abstract": "Hybrid beamforming for large arrays.",
                "year": 2022, "venue": "V", "authors": [], "doi": None,
                "arxiv_id": "3", "source": "s2"})
            c.execute("INSERT INTO pages(paper_id,page_no,text) VALUES(?,?,?)",
                      (self.p1, 4, "Pilot decontamination via subspace projection "
                                   "reduces inter-cell interference substantially."))
            db.reindex_pages(c, self.p1)

    def tearDown(self):
        for p in self._p:
            p.stop()
        config.DB_PATH, config.DATA_DIR = self._old_db, self._old_dir

    def _on(self, routes):
        return mock.patch.multiple(
            "papernest.llm", available=mock.Mock(return_value=True),
            chat=mock.Mock(side_effect=_fake_llm(routes)))


class RerankFailureModeTests(RcsTestBase):
    """重排是增强件，任何失败都必须退回检索原序，绝不能让问答挂掉。"""

    def test_rerank_reorders_by_model_score(self):
        with self._on({"rcs_rerank": _scores((1, 2), (2, 0), (3, 9))}):
            ids, tr = rcs.rerank("导频污染", [self.p1, self.p2, self.p3], 2)
        self.assertEqual(ids, [self.p3, self.p1])
        self.assertTrue(tr["used"])
        self.assertEqual(tr["scores"][0]["score"], 9)

    def test_model_error_falls_back_to_retrieval_order(self):
        with self._on({"rcs_rerank": mock.Mock(side_effect=RuntimeError("网络断了"))}):
            ids, tr = rcs.rerank("q", [self.p1, self.p2, self.p3], 2)
        self.assertEqual(ids, [self.p1, self.p2])     # 检索原序
        self.assertFalse(tr["used"])
        self.assertIn("RuntimeError", tr["error"])

    def test_garbage_json_falls_back(self):
        with self._on({"rcs_rerank": "这不是 JSON"}):
            ids, tr = rcs.rerank("q", [self.p1, self.p2], 2)
        self.assertEqual(ids, [self.p1, self.p2])
        self.assertFalse(tr["used"])

    def test_out_of_range_ids_are_discarded_not_crashing(self):
        """模型编出不存在的候选编号——丢弃，不能拿它去索引数组。"""
        with self._on({"rcs_rerank": _scores((99, 10), (1, 3))}):
            ids, tr = rcs.rerank("q", [self.p1, self.p2], 2)
        self.assertTrue(tr["used"])
        self.assertEqual([s["paper_id"] for s in tr["scores"]], [self.p1])

    def test_missing_candidates_keep_retrieval_order_after_scored_ones(self):
        """模型漏评的不能凭空猜分，也不能因为漏评就把这篇丢掉。"""
        with self._on({"rcs_rerank": _scores((2, 8))}):
            ids, tr = rcs.rerank("q", [self.p1, self.p2, self.p3], 3)
        self.assertEqual(ids[0], self.p2)                    # 唯一被评分的排最前
        self.assertEqual(ids[1:], [self.p1, self.p3])        # 其余保持检索原序
        self.assertEqual(tr["missing"], [self.p1, self.p3])

    def test_all_same_score_is_reported_as_no_signal(self):
        """全同分 = 重排没产生任何信息，白花一次调用。不能只写「已生效」。"""
        with self._on({"rcs_rerank": _scores((1, 5), (2, 5), (3, 5))}):
            ids, tr = rcs.rerank("q", [self.p1, self.p2, self.p3], 3)
        self.assertEqual(ids, [self.p1, self.p2, self.p3])   # 同分保持原序
        self.assertIn("no_signal", tr)

    def test_ordering_is_deterministic(self):
        with self._on({"rcs_rerank": _scores((1, 5), (2, 5), (3, 9))}):
            runs = {tuple(rcs.rerank("q", [self.p1, self.p2, self.p3], 3)[0])
                    for _ in range(5)}
        self.assertEqual(len(runs), 1)


class QuoteVerificationTests(RcsTestBase):
    """依据句校验：不能被排版伪影误杀，也不能被跨页拼接骗过。"""

    def _item(self, pid):
        return rcs._load([pid])[pid]

    def test_verbatim_quote_from_page_verifies_with_page_number(self):
        ok, page = rcs._verify(
            "Pilot decontamination via subspace projection", self._item(self.p1))
        self.assertTrue(ok)
        self.assertEqual(page, 4)

    def test_quote_from_abstract_verifies_without_page(self):
        ok, page = rcs._verify(
            "Pilot contamination limits massive MIMO capacity", self._item(self.p1))
        self.assertTrue(ok)
        self.assertIsNone(page)

    def test_fabricated_quote_fails(self):
        ok, _p = rcs._verify("这句话原文里根本不存在的编造内容", self._item(self.p1))
        self.assertFalse(ok)

    def test_line_break_hyphen_artifact_does_not_cause_false_negative(self):
        """PDF 换行断词：模型照抄了，不该因为一个连字符就判「未核验」。"""
        ok, _p = rcs._verify(
            "Pilot de-\ncontamination via subspace projection", self._item(self.p1))
        self.assertTrue(ok)

    def test_ligature_is_normalised(self):
        with db.conn() as c:
            pid = db.insert_l0(c, {
                "norm_key": "arxiv:9", "title": "Ligature Paper",
                "abstract": "We conﬁrm the ﬁnding across ﬁve datasets.",
                "year": 2024, "venue": "V", "authors": [], "doi": None,
                "arxiv_id": "9", "source": "s2"})
        ok, _p = rcs._verify("We confirm the finding across five datasets",
                             self._item(pid))
        self.assertTrue(ok)

    def test_too_short_quote_is_rejected(self):
        """太短的片段随便都能命中，不构成证据。"""
        self.assertFalse(rcs._verify("Pilot", self._item(self.p1))[0])

    def test_cross_page_splice_is_not_accepted(self):
        """前半在一页、后半在另一页，拼起来「验过」是假证据——必须逐页比对。"""
        with db.conn() as c:
            pid = db.insert_l0(c, {
                "norm_key": "arxiv:8", "title": "Two Page Paper", "abstract": "",
                "year": 2024, "venue": "V", "authors": [], "doi": None,
                "arxiv_id": "8", "source": "s2"})
            c.execute("INSERT INTO pages(paper_id,page_no,text) VALUES(?,?,?)",
                      (pid, 1, "The proposed estimator reduces the mean squared error"))
            c.execute("INSERT INTO pages(paper_id,page_no,text) VALUES(?,?,?)",
                      (pid, 2, "by thirty percent under low signal to noise ratio"))
        ok, _p = rcs._verify(
            "The proposed estimator reduces the mean squared error "
            "by thirty percent under low signal to noise ratio", self._item(pid))
        self.assertFalse(ok)


class SummaryFilteringTests(RcsTestBase):
    """定向摘要的过滤是 RCS 最危险的一步：判错就是这篇论文彻底消失。"""

    def test_irrelevant_papers_are_dropped_and_counted(self):
        routes = {"rcs_rerank": _scores((1, 9), (2, 1)),
                  "rcs_summary": lambda user: (
                      _summary(True, "导频污染要点。",
                               ["Pilot contamination limits massive MIMO capacity"])
                      if "Pilot" in user else _summary(False, "", []))}
        with self._on(routes):
            ctx, sources, tr = rcs.build_context("导频污染", [self.p1, self.p2], 2)
        self.assertEqual(len(sources), 1)
        self.assertEqual(tr["summary"]["dropped_irrelevant"], 1)
        # 「入选后被摘要扔掉」与「重排未选中」必须分开报——混成一个数会严重误导：
        # 前者是危险信号（这篇本来排进来了，是被模型主动扔的），后者是正常落选。
        self.assertEqual(tr["filtered_out"], 1)
        self.assertEqual(tr["not_selected_by_rerank"], 0)

    def test_rerank_dropouts_are_not_counted_as_filtered(self):
        """候选池比 top_k 大是设计如此，落选不该被算进「被扔掉」。"""
        routes = {"rcs_rerank": _scores((1, 9), (2, 5), (3, 1)),
                  "rcs_summary": _summary(True, "要点。", [])}
        with self._on(routes):
            _ctx, sources, tr = rcs.build_context(
                "q", [self.p1, self.p2, self.p3], 1)
        self.assertEqual(len(sources), 1)
        self.assertEqual(tr["not_selected_by_rerank"], 2)
        self.assertEqual(tr["filtered_out"], 0)

    def test_all_filtered_out_degrades_instead_of_empty_answer(self):
        """全部被判不相关 → 退回朴素上下文，而不是让作答模型面对空上下文。"""
        with self._on({"rcs_rerank": _scores((1, 5), (2, 5)),
                       "rcs_summary": _summary(False, "", [])}):
            ctx, sources, tr = rcs.build_context("q", [self.p1, self.p2], 2)
        self.assertEqual(ctx, "")
        self.assertEqual(sources, [])
        self.assertIn("degraded", tr)

    def test_unverified_quote_is_kept_but_marked(self):
        """验不过的依据句不静默丢弃——标出来，并在 prompt 里禁止当事实引用。"""
        with self._on({"rcs_rerank": _scores((1, 9)),
                       "rcs_summary": _summary(True, "要点。", ["模型编造的依据句内容"])}):
            ctx, sources, tr = rcs.build_context("q", [self.p1], 1)
        self.assertIn("未通过原文回取校验", ctx)
        self.assertFalse(sources[0]["quotes"][0]["verified"])
        self.assertEqual(tr["summary"]["quotes_verified"], 0)
        self.assertEqual(tr["summary"]["quotes_total"], 1)

    def test_summary_failure_on_one_paper_does_not_sink_the_rest(self):
        calls = {"n": 0}

        def summary(user):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("这篇挂了")
            return _summary(True, "要点。",
                            ["Hybrid beamforming for large arrays"])
        with self._on({"rcs_rerank": _scores((1, 9), (2, 8)),
                       "rcs_summary": summary}):
            ctx, sources, tr = rcs.build_context("q", [self.p1, self.p3], 2)
        self.assertEqual(len(sources), 1)
        self.assertEqual(len(tr["summary"]["errors"]), 1)

    def test_one_call_per_paper_not_per_field(self):
        """成本上限要钉死：N 篇就是 N 次摘要调用，不能变成 N×字段数。

        RCS 的代价本来就是「每次问答多 1+N 次调用」，如果这里退化成按字段调用，
        成本会再翻几倍——这是这个模块最需要守住的一条线。
        """
        spy = mock.Mock(side_effect=_fake_llm(
            {"rcs_rerank": _scores((1, 9), (2, 8), (3, 7)),
             "rcs_summary": _summary(True, "要点。", [])}))
        with mock.patch("papernest.llm.available", return_value=True), \
             mock.patch("papernest.llm.chat", spy):
            rcs.build_context("q", [self.p1, self.p2, self.p3], 3)
        purposes = [c.kwargs.get("purpose") for c in spy.call_args_list]
        self.assertEqual(sum(1 for p in purposes if p == "rcs_summary"), 3)
        self.assertEqual(sum(1 for p in purposes if p == "rcs_rerank"), 1)
        self.assertEqual(len(purposes), 4)      # 3 篇 → 1 + 3，一次不多


class RcsAnswerTests(RcsTestBase):
    def test_requires_llm(self):
        with mock.patch("papernest.llm.available", return_value=False):
            with self.assertRaises(Exception):
                rcs.answer("q")

    def test_answer_reports_verified_rate(self):
        routes = {"rcs_rerank": _scores((1, 9)),
                  "rcs_summary": _summary(
                      True, "要点。",
                      ["Pilot contamination limits massive MIMO capacity"]),
                  "rcs_answer": "导频污染会限制容量 [1]。"}
        with mock.patch("papernest.embeddings.search_hybrid",
                        return_value=embeddings.RetrievalResult([self.p1], "fts", [])):
            with self._on(routes):
                r = rcs.answer("导频污染", top_k=1)
        self.assertIn("[1]", r["answer"])
        self.assertEqual(r["verified_rate"], 1.0)
        self.assertEqual(len(r["sources"]), 1)
        self.assertIn("rcs_summary", r["sources"][0])


if __name__ == "__main__":
    unittest.main()


class BodyTextEvidenceWindowTest(RcsTestBase):
    """定向摘要必须看得到「问句真正命中的那一页」，而不是恒定的前 3 页。

    形态照真库量出来的数字构造（data/papernest.db 实测）：
      · 45 篇有全文的论文平均 12.6 页（中位 12）；
      · QASPER 侧单节中位 1227 字符；abstract+卡片中位 868、p90 1223 字符；
      · 122 道带 gold 证据的题里，gold 证据首次出现的页(节)序号中位数 = 5、p90 = 10，
        只有 43.4% 落在前 3 页——而前 3 节典型是 Abstract/Intro/Related Work。
    所以这里：12 页 × ~1200 字符，abstract ~1200 字符，证据落在第 9 页。

    三个 filler 页与 evidence 页等长，所以 `_pick_pages` 的密度归一不会因页长差异
    而作弊——第 9 页胜出只能是因为它含 bleu/wmt14/decoder 这些问句词。
    """

    def _make_paper(self):
        filler = ("This section reviews prior work on unrelated topics and "
                  "provides general background discussion for completeness. ")
        evidence = ("We evaluate the proposed decoder on the WMT14 corpus and "
                    "observe a BLEU improvement of 2.3 points over the baseline.")
        abstract = ("We study neural sequence models. " * 40)[:1200]
        with db.conn() as c:
            pid = db.insert_l0(c, {
                "norm_key": "arxiv:9001", "title": "Neural Decoder Study",
                "abstract": abstract, "year": 2021, "venue": "V", "authors": [],
                "doi": None, "arxiv_id": "9001", "source": "s2"})
            for pno in range(1, 13):
                if pno == 9:
                    text = (filler * 5) + evidence + " " + (filler * 5)
                else:
                    text = filler * 10
                c.execute("INSERT INTO pages(paper_id,page_no,text) VALUES(?,?,?)",
                          (pid, pno, text[:1200]))
            db.reindex_pages(c, pid)
        return pid, evidence

    def test_evidence_page_reaches_the_summary_model(self):
        pid, _evidence = self._make_paper()
        seen = []

        def capture(user):
            seen.append(user)
            return _summary(True, "要点。", [])

        question = "What BLEU improvement does the proposed decoder achieve on WMT14?"
        with self._on({"rcs_summary": capture}):
            rcs.summarize_for(question, [pid])

        self.assertEqual(len(seen), 1, "摘要调用没发出去")
        prompt = seen[0]
        # ① 钉住缺陷本身：命中页必须进得了喂给模型的正文
        self.assertIn(
            "BLEU improvement of 2.3 points", prompt,
            "第 9 页（问句唯一真正命中的页）没有进入 _body_text 的输出——"
            "定向摘要模型看不到证据，它判出来的「不相关」不是模型过滤激进，是输入里就没有")
        # ② 钉住「不许靠拆掉预算蒙混过关」
        self.assertLessEqual(len(prompt), rcs.SUMMARY_CHARS + 400,
                             "正文超出 SUMMARY_CHARS 预算，等于把上限拆了")
        # ③ 钉住预算天花板：防止后人「再改大一点」无依据地放大 token 成本
        self.assertLessEqual(rcs.SUMMARY_CHARS, 8000,
                             "SUMMARY_CHARS 超过 8000：token 成本没有依据地放大")

    def test_page_markers_do_not_break_quote_verification(self):
        """`_pick_pages` 插的「【第 N 页】」被模型连着抄回来时，不许判成「未核验」。

        那是**假阴性**——把真证据标成未核验，比假阳性更隐蔽，因为它让人不敢用。
        """
        self.assertEqual(rcs._norm_quote("【第 15 页】We evaluate the decoder…"),
                         rcs._norm_quote("We evaluate the decoder"))
