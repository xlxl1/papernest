"""写作流水线的**在线**路径（配了 LLM 才走的那一半）。

为什么单独一个文件：原有 13 个流水线用例全部跑在 `llm.available()=False` 下，
而 `_write_one` 在离线分支就提前 return 了——在线分支里 `return {... "status": final_status ...}`
引用了一个从未赋值的名字，只要配上真 key，每写完一节就 NameError、整个 run 判 failed。
一个用例都没碰到那行。这里用假模型把在线分支跑起来，把「撰写⇄评审⇄重写」闭环焊死。
"""
import shutil
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from papernest import config, db, pipeline


def _review(score, majors: int = 0) -> str:
    return json.dumps({
        "overall": "评审意见",
        "score": score,
        "strengths": ["结构清楚"],
        "issues": [{"severity": "major", "location": "第一段",
                    "problem": "论断缺证据", "fix": "补白名单内引用"}] * majors,
        "checklist": ["补证据"],
    }, ensure_ascii=False)


REVIEW_PASS = _review(9)
REVIEW_FAIL = _review(4, majors=1)


class FakeLLM:
    """按 purpose 路由的假模型：草稿 / 评审 / 润色各自可控，并记录调用序列。

    列表形式的 drafts/reviews 按调用次序消费，用完后一直返回最后一个。
    """

    def __init__(self, drafts="本节正文 [1]。", reviews=REVIEW_PASS, polish=None):
        self._drafts = drafts if isinstance(drafts, list) else [drafts]
        self._reviews = reviews if isinstance(reviews, list) else [reviews]
        self._polish = polish
        self.calls: list[str] = []

    @staticmethod
    def _take(seq, i):
        return seq[min(i, len(seq) - 1)]

    def chat(self, system, user, purpose="", paper_id=None,
             temperature=0.3, model=None):
        self.calls.append(purpose)
        if purpose.startswith("write_"):
            return self._take(self._drafts,
                              sum(1 for p in self.calls if p.startswith("write_")) - 1)
        if purpose == "writing_review":
            return self._take(self._reviews,
                              sum(1 for p in self.calls if p == "writing_review") - 1)
        if purpose == "writing_polish":
            return self._polish or json.dumps(
                {"revised": "润色后的全文 [1]。", "changes": []}, ensure_ascii=False)
        return "{}"

    def n(self, prefix: str) -> int:
        return sum(1 for p in self.calls if p.startswith(prefix))


class WriteOnlineTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="papernest_online_")
        self.addCleanup(shutil.rmtree, self._tmp, True)
        self._old_db, self._old_dir = config.DB_PATH, config.DATA_DIR
        config.DB_PATH = Path(self._tmp) / "test.db"
        config.DATA_DIR = Path(self._tmp) / "data"
        # 向量强制不可用：否则开发机 .env 里配了 EMBED_MODEL 时，测试会真的去打
        # embeddings 接口——又慢又花钱，结果还随网络状况漂移。词面兜底是确定性的。
        self._embed_patch = mock.patch("papernest.embeddings.available",
                                       return_value=False)
        self._embed_patch.start()
        db.init_db()
        with db.conn() as c:
            self.p1 = db.insert_l0(c, {
                "norm_key": "arxiv:8001", "title": "Agent Evaluation Survey",
                "abstract": "Agents need evaluation. Tool use reliability is hard to measure.",
                "year": 2024, "venue": "TestVenue", "authors": ["Alice Smith"],
                "doi": None, "arxiv_id": "8001", "source": "s2"})
            self.p2 = db.insert_l0(c, {
                "norm_key": "arxiv:8002", "title": "LLM Judge Bias Study",
                "abstract": "LLM judges show bias in evaluation pipelines.",
                "year": 2024, "venue": "TestVenue", "authors": ["Bob Lee"],
                "doi": None, "arxiv_id": "8002", "source": "s2"})

    def tearDown(self):
        self._embed_patch.stop()
        config.DB_PATH, config.DATA_DIR = self._old_db, self._old_dir

    # ── 夹具 ──

    def _online(self, fake: FakeLLM):
        """把 LLM 打开并接上假模型（pipeline 与 writing 共用 papernest.llm）。"""
        return mock.patch.multiple("papernest.llm",
                                   available=mock.Mock(return_value=True),
                                   chat=mock.Mock(side_effect=fake.chat))

    def _seed(self, max_rewrites: int = 2, sections=None):
        run = pipeline.create_run("agent evaluation methods", max_rewrites=max_rewrites)
        # 节标题/论点用与摘要同源的英文措辞：词面兜底才找得到证据句，
        # 白名单非空——否则测的就不是「引用被剥离」而是「压根没有白名单」了。
        secs = sections or [{"no": 1, "title": "Agent evaluation background",
                             "points": ["tool use reliability is hard to measure"],
                             "paper_ids": [self.p1], "words": 400}]
        with db.conn() as c:
            c.execute(
                "INSERT INTO outlines(run_id,version,title,outline_json) VALUES(?,?,?,?)",
                (run["id"], 1, "Agent 评测综述", json.dumps(secs, ensure_ascii=False)))
        return pipeline.get_run(run["id"]), secs

    def test_whitelist_is_non_empty_in_fixture(self):
        """夹具自检：白名单为空会让下面几个「引用」用例变成假绿。"""
        wl, err = pipeline._whitelist(
            "Agent evaluation background tool use reliability is hard to measure",
            k=6, extra_ids=[self.p1])
        self.assertIsNone(err)
        self.assertGreaterEqual(len(wl), 1)
        self.assertTrue(all(w["verified"] and w["evidence_sentence"] for w in wl))

    # ── 回归：在线分支必须能返回 ──

    def test_write_one_online_returns_done(self):
        """曾经在这里抛 NameError('final_status')——在线路径的主干。"""
        run, secs = self._seed()
        fake = FakeLLM()
        with self._online(fake):
            r = pipeline._write_one(run, secs[0], 1, force=True)
        self.assertEqual(r["status"], "done")
        self.assertTrue(r["met_bar"])
        self.assertEqual(r["attempt"], 1)          # 一稿过审，不多花 token
        self.assertEqual(fake.n("write_"), 1)
        self.assertEqual(fake.n("writing_review"), 1)
        rows = pipeline._latest_sections(run["id"])
        self.assertEqual(rows[0]["status"], "done")
        self.assertEqual(rows[0]["score"], 9)

    def test_low_score_rewrites_then_delivers_honestly(self):
        """评分始终不达标：重写到上限后如实交付（不是 failed，也不假装达标）。"""
        run, secs = self._seed(max_rewrites=2)
        fake = FakeLLM(reviews=REVIEW_FAIL)
        with self._online(fake):
            r = pipeline._write_one(run, secs[0], 1, force=True)
        self.assertEqual(r["attempt"], 3)          # 初稿 + 2 次重写
        self.assertEqual(fake.n("write_"), 3)
        self.assertFalse(r["met_bar"])             # 没达标就说没达标
        self.assertEqual(r["status"], "done")      # 但仍如实交付
        self.assertEqual(r["score"], 4)

    def test_rewrite_stops_as_soon_as_bar_is_met(self):
        run, secs = self._seed(max_rewrites=2)
        fake = FakeLLM(drafts=["初稿 [1]。", "改后 [1]。"],
                       reviews=[REVIEW_FAIL, REVIEW_PASS])
        with self._online(fake):
            r = pipeline._write_one(run, secs[0], 1, force=True)
        self.assertEqual(r["attempt"], 2)
        self.assertTrue(r["met_bar"])
        self.assertEqual(fake.n("write_"), 2)      # 达标即停，不跑满上限
        self.assertEqual(pipeline._latest_sections(run["id"])[0]["content"], "改后 [1]。")

    def test_max_rewrites_zero_never_rewrites(self):
        run, secs = self._seed(max_rewrites=0)
        fake = FakeLLM(reviews=REVIEW_FAIL)
        with self._online(fake):
            r = pipeline._write_one(run, secs[0], 1, force=True)
        self.assertEqual(r["attempt"], 1)
        self.assertFalse(r["met_bar"])

    # ── 引用白名单在在线路径同样生效 ──

    def test_out_of_whitelist_citations_stripped_online(self):
        """模型编出白名单外的 [9]：生成期剥离并如实计数，不留进正文。"""
        run, secs = self._seed()
        fake = FakeLLM(drafts="有据的论断 [1]，编造的引用 [9]。")
        with self._online(fake):
            r = pipeline._write_one(run, secs[0], 1, force=True)
        self.assertEqual(r["removed"], 1)
        content = pipeline._latest_sections(run["id"])[0]["content"]
        self.assertNotIn("[9]", content)
        self.assertIn("[1]", content)

    def test_citations_recorded_only_for_used_numbers(self):
        run, secs = self._seed()
        with self._online(FakeLLM(drafts="只用第一篇 [1]。")):
            pipeline._write_one(run, secs[0], 1, force=True)
        cit = json.loads(pipeline._latest_sections(run["id"])[0]["citations_json"])
        self.assertEqual(list(cit), ["1"])
        self.assertTrue(cit["1"]["verified"])

    # ── 缓存：在线路径重跑未变节 0 次调用 ──

    def test_unchanged_section_reuses_cache_with_zero_llm_calls(self):
        run, secs = self._seed()
        fake = FakeLLM()
        with self._online(fake):
            pipeline._write_one(run, secs[0], 1, force=True)
            before = len(fake.calls)
            r = pipeline._write_one(run, secs[0], 1, force=False)
        self.assertEqual(r["status"], "reused")
        self.assertTrue(r["cache_hit"])
        self.assertEqual(len(fake.calls), before)   # 一次模型调用都没多花

    # ── 整阶段 ──

    def test_stage_sections_online_reports_below_bar(self):
        secs = [{"no": 1, "title": "Agent evaluation background",
                 "points": ["tool use reliability is hard to measure"],
                 "paper_ids": [self.p1], "words": 300},
                {"no": 2, "title": "LLM judge bias",
                 "points": ["llm judges show bias in evaluation pipelines"],
                 "paper_ids": [self.p2], "words": 300}]
        run, _ = self._seed(max_rewrites=0, sections=secs)
        # 第 1 节达标、第 2 节不达标（每节各一次撰写 + 一次评审）
        fake = FakeLLM(reviews=[REVIEW_PASS, REVIEW_FAIL])
        with self._online(fake):
            out = pipeline.stage_sections(run["id"])
        self.assertEqual(out["sections"], 2)
        self.assertEqual(out["failed"], [])
        self.assertEqual(out["below_bar"], [2])     # 如实点名没过审的节
        self.assertEqual(pipeline.get_run(run["id"])["status"], "drafted")

    def test_stage_polish_online_exports_and_checks(self):
        run, secs = self._seed()
        fake = FakeLLM()
        with self._online(fake):
            pipeline.stage_sections(run["id"])
            out = pipeline.stage_polish(run["id"])
        self.assertIn("md", out["files"])
        self.assertTrue(Path(out["files"]["md"]).exists())
        self.assertEqual(pipeline.get_run(run["id"])["status"], "done")
        report = pipeline.get_run(run["id"])["result"]
        self.assertIn("verified_rate", report)
        self.assertEqual(report["dangling"], [])

    def test_run_stage_online_end_to_end(self):
        """走任务分发入口，确认在线全流程不再把 run 判 failed。"""
        run, _ = self._seed()
        with self._online(FakeLLM()):
            pipeline.run_stage(run["id"], {"stage": "sections"})
            pipeline.run_stage(run["id"], {"stage": "polish"})
        final = pipeline.get_run(run["id"])
        self.assertEqual(final["status"], "done")
        self.assertIsNone(final["error"])


if __name__ == "__main__":
    unittest.main()
