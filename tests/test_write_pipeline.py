"""二期写作流水线：机械部分全部离线可测——大纲校验、引用白名单、缓存、
局部→全局编号重排、机械终检、导出（md/docx）。LLM 一律 mock 为不可用。"""
import shutil
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from papernest import config, db, pipeline


def _mk_paper(c, key, title, abstract, card=None):
    pid = db.insert_l0(c, {"norm_key": key, "title": title, "abstract": abstract,
                           "year": 2024, "venue": "TestVenue",
                           "authors": ["Alice", "Bob"], "doi": None,
                           "arxiv_id": key.split(":")[-1], "source": "s2"})
    if card is not None:
        c.execute("UPDATE papers SET card_json=? WHERE id=?",
                  (json.dumps(card, ensure_ascii=False), pid))
    return pid


P1_LIM = ("Evaluation of multi-step tool use reliability is limited and "
          "lacks systematic benchmarks for long horizon tasks.")


class PipelineOfflineTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="papernest_pipe_")
        self.addCleanup(shutil.rmtree, self._tmp, True)
        self._old_db, self._old_dir = config.DB_PATH, config.DATA_DIR
        config.DB_PATH = Path(self._tmp) / "test.db"
        config.DATA_DIR = Path(self._tmp) / "data"
        db.init_db()
        with db.conn() as c:
            self.p1 = _mk_paper(c, "arxiv:9001", "Agent Evaluation Survey",
                                "agents need evaluation. tool use reliability is hard.",
                                {"tldr": "agent evaluation survey",
                                 "limitations": P1_LIM,
                                 "keywords": ["agent evaluation", "benchmark"]})
            self.p2 = _mk_paper(c, "arxiv:9002", "LLM Judge Bias Study",
                                "llm judges show bias in evaluation pipelines.",
                                {"tldr": "judge bias study",
                                 "limitations": "Human agreement analysis is missing.",
                                 "keywords": ["llm-as-judge"]})
            self.p3 = _mk_paper(c, "arxiv:9003", "Unrelated Paper",
                                "about ocean temperature records.",
                                {"tldr": "ocean warming", "keywords": []})
            c.commit()
        # 全程离线：LLM 不可用、检索走固定 ids（不打向量接口）。
        # embeddings.available 必须一并按住——_whitelist → cite.recommend → embed_texts
        # 这条链绕过了 search_hybrid，开发机 .env 配了 EMBED_MODEL 时会真的发请求。
        self._patches = [mock.patch.object(pipeline.llm, "available", return_value=False),
                         mock.patch("papernest.embeddings.available", return_value=False),
                         mock.patch.object(pipeline.embeddings, "search_hybrid",
                                           return_value=pipeline.embeddings.RetrievalResult(
                                             [self.p1, self.p2, self.p3],
                                             "fts", []))]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        config.DB_PATH, config.DATA_DIR = self._old_db, self._old_dir

    # ── 选题 Agent（离线：局限栏确定性抽取）──

    def test_stage_topic_offline_creates_candidates(self):
        run = pipeline.create_run("agent evaluation methods")
        r = pipeline.stage_topic(run["id"])
        run = pipeline.get_run(run["id"])
        self.assertEqual(run["status"], "pending_user_topic")
        self.assertGreaterEqual(len(run["topics"]), 1)
        self.assertEqual(r["degraded"] and "未配置 LLM" in r["degraded"], True)
        for t in run["topics"]:
            for ev in t["evidence"]:
                self.assertTrue(ev["verified"])  # 证据句逐字回取到原文

    def test_topic_evidence_unverifiable_marked(self):
        run = pipeline.create_run("agent evaluation methods")
        pipeline.stage_topic(run["id"])
        with db.conn() as c:
            c.execute("UPDATE writing_runs SET topics_json=? WHERE id=?",
                      (json.dumps([{"title": "T", "gap": "g", "angle": "a", "risks": "r",
                                    "evidence": [{"paper_id": self.p1,
                                                  "quote": "这句原文里根本不存在"}]}]),
                       run["id"]))
            c.commit()
        # 直接走核验函数：编造的 quote 必须判 False
        self.assertFalse(pipeline._verify_quote(self.p1, "这句原文里根本不存在"))
        self.assertTrue(pipeline._verify_quote(self.p1, P1_LIM[:40]))

    # ── 检查点① → 大纲 Agent ──

    def test_select_topic_and_outline_flow(self):
        run = pipeline.create_run("agent evaluation methods")
        pipeline.stage_topic(run["id"])
        run = pipeline.select_topic(run["id"], "Agent 评测方法综述")
        self.assertEqual(run["status"], "outline_running")
        r = pipeline.stage_outline(run["id"])
        o = pipeline.get_outline(run["id"])
        self.assertEqual(o["version"], 1)
        self.assertGreaterEqual(len(o["sections"]), 2)
        self.assertTrue(all("words" in s for s in o["sections"]))
        run = pipeline.get_run(run["id"])
        self.assertEqual(run["status"], "pending_user_outline")
        self.assertIsInstance(run["glossary"], dict)
        # 无支撑节必须被机械校验补标 no_support（第 4 节拟引为空）
        last = o["sections"][-1]
        if not last.get("paper_ids"):
            self.assertTrue(last.get("no_support"))
            self.assertTrue(any("no_support" in w for w in r["warnings"]))

    def test_validate_outline_fixes(self):
        sections = [{"no": 1, "title": "A", "points": [], "paper_ids": [999], "words": 50},
                    {"no": 3, "title": "", "points": [], "paper_ids": [], "words": 60000}]
        issues = pipeline.validate_outline(sections, {self.p1})
        self.assertEqual([s["no"] for s in sections], [1, 2])       # 编号重排
        self.assertEqual(sections[0]["paper_ids"], [])              # 非法拟引剔除
        self.assertTrue(sections[1]["no_support"])                  # 无拟引补标
        self.assertTrue(any("字数预算" in i for i in issues))

    def test_save_outline_versions(self):
        run = pipeline.create_run("agent evaluation methods")
        pipeline.stage_topic(run["id"])
        pipeline.select_topic(run["id"], "Agent 评测方法综述")
        pipeline.stage_outline(run["id"])
        pipeline.save_outline(run_id := run["id"], None)
        self.assertEqual(pipeline.get_run(run_id)["status"], "drafting")
        self.assertEqual(pipeline.get_outline(run_id)["version"], 1)
        edited = pipeline.get_outline(run_id)["sections"]
        edited[0]["title"] = "改过的标题"
        pipeline.save_outline(run_id, edited)
        self.assertEqual(pipeline.get_outline(run_id)["version"], 2)
        self.assertEqual(pipeline.get_outline(run_id)["edited"], 1)

    # ── 文献 Agent（白名单）──

    def test_whitelist_only_verified(self):
        wl, err = pipeline._whitelist("tool use reliability evaluation of agents",
                                      k=3, extra_ids=[999999])
        self.assertIsNone(err)
        for w in wl:
            self.assertTrue(w["verified"])
            self.assertTrue(w["evidence_sentence"])
        self.assertTrue(all(w["paper_id"] != 999999 for w in wl))  # 不存在的文献进不来

    def test_whitelist_failure_is_not_the_same_as_no_support(self):
        """「cite.recommend 抛异常」和「库里真的没有可支撑文献」原来产出完全一样，

        都渲染成「本节无支撑」写进草稿并落库。前者是要修的故障，后者是该如实交付的
        事实——一次向量服务抖动就会让某一节永久变成无引用段，且没有任何地方记下它。
        """
        with mock.patch("papernest.cite.recommend", side_effect=RuntimeError("boom")):
            wl, err = pipeline._whitelist("tool use reliability", k=3)
        self.assertEqual(wl, [])
        self.assertIsNotNone(err)
        self.assertIn("白名单构造失败", err)

        # 对照：库里确实没有可支撑文献时，白名单同样为空，但 err 必须是 None
        wl2, err2 = pipeline._whitelist("完全无关的量子引力弦论主题", k=3)
        self.assertEqual(wl2, [])
        self.assertIsNone(err2)

    # ── 缓存 ──

    def test_cache_key_stable_and_sensitive(self):
        k1 = pipeline._cache_key("标题", ["论点1"], 600, {"术语": "x"})
        k2 = pipeline._cache_key("标题", ["论点1"], 600, {"术语": "x"})
        k3 = pipeline._cache_key("标题", ["论点2"], 600, {"术语": "x"})
        k4 = pipeline._cache_key("标题", ["论点1"], 800, {"术语": "x"})
        self.assertEqual(k1, k2)
        self.assertNotEqual(k1, k3)
        self.assertNotEqual(k1, k4)

    def test_section_cache_reuse_zero_llm(self):
        run = pipeline.create_run("agent evaluation methods")
        with db.conn() as c:  # 节级状态挂在当前大纲版本下
            c.execute("INSERT INTO outlines(run_id,version,title,outline_json) VALUES(?,?,?,?)",
                      (run["id"], 1, "t", "[]"))
            c.commit()
        sec = {"no": 1, "title": "研究背景", "points": ["背景论点"], "words": 600}
        key = pipeline._cache_key(sec["title"], sec["points"], sec["words"], {})
        pipeline._upsert_section(run["id"], 1, sec, whitelist_json=[],
                                 cache_key=key, content="已缓存内容 [1]",
                                 citations_json={}, removed_json=[], status="done",
                                 attempt=1, score=8, review_json=None)
        r = pipeline._write_one(pipeline.get_run(run["id"]), sec, 1, force=False)
        self.assertEqual(r["status"], "reused")     # 命中缓存：LLM 若被调用会抛 LLMUnavailable
        rows = pipeline._latest_sections(run["id"])
        self.assertEqual(rows[0]["content"], "已缓存内容 [1]")

    # ── 白名单外编号剥离 ──

    def test_sanitize_removes_out_of_whitelist(self):
        text, removed = pipeline._sanitize("甲 [1] 乙 [3] 丙 [12] 丁 [2]", [None] * 2)
        self.assertEqual(removed, [3, 12])
        self.assertEqual(text, "甲 [1] 乙  丙  丁 [2]")

    # ── 装配与机械终检 ──

    def _seed_sections(self, run_id):
        o = {"title": "测试综述", "sections": []}
        with db.conn() as c:
            c.execute("INSERT INTO outlines(run_id,version,title,outline_json) VALUES(?,?,?,?)",
                      (run_id, 1, "测试综述", json.dumps(o["sections"])))
            c.commit()
        e1, e2 = (pipeline._wl_entry(self.p1, "evidence one"),
                  pipeline._wl_entry(self.p2, "evidence two"))
        pipeline._upsert_section(run_id, 1, {"no": 1, "title": "背景", "points": []},
                                 whitelist_json=[], cache_key="k1",
                                 content="背景论断 [1] 与 [2]。",
                                 citations_json={"1": e1, "2": e2},
                                 removed_json=[], status="done", attempt=1, score=8,
                                 review_json=None)
        pipeline._upsert_section(run_id, 1, {"no": 2, "title": "相关工作", "points": []},
                                 whitelist_json=[], cache_key="k2",
                                 content="后续工作 [1] 继续推进（待补证据）。",
                                 citations_json={"1": e2},
                                 removed_json=[], status="done", attempt=1, score=8,
                                 review_json=None)
        return e1, e2

    def test_assemble_renumbers_global(self):
        run = pipeline.create_run("agent evaluation methods")
        self._seed_sections(run["id"])
        body, refs, infos = pipeline.assemble(run["id"])
        # 节1 的 [1][2] 不变；节2 的局部 [1]（p2）应重排为全局 [2]
        self.assertIn("背景论断 [1] 与 [2]。", body)
        self.assertIn("后续工作 [2] 继续推进", body)
        self.assertEqual(set(refs), {1, 2})
        self.assertEqual(refs[1]["paper_id"], self.p1)
        self.assertEqual(refs[2]["paper_id"], self.p2)
        rep = pipeline.final_check(body, refs)
        self.assertTrue(rep["ok"])
        self.assertEqual(rep["citation_count"], 3)
        self.assertEqual(rep["verified_rate"], 1.0)
        self.assertEqual(rep["pending_evidence"], 1)

    def test_final_check_reports_dangling(self):
        body, refs, _ = "正文 [1] 与悬空 [5]。", {1: {"paper_id": 1, "verified": True}}, []
        rep = pipeline.final_check(body, refs)
        self.assertFalse(rep["ok"])
        self.assertEqual(rep["dangling"], [5])
        # 悬空编号没有证据支撑：核验率如实被拉低（1/2），不假装通过
        self.assertEqual(rep["verified_rate"], 0.5)

    # ── 润色 + 终检 + 导出（离线：润色原样返回并如实标注）──

    def test_stage_polish_offline_exports(self):
        run = pipeline.create_run("agent evaluation methods")
        self._seed_sections(run["id"])
        r = pipeline.stage_polish(run["id"])
        run = pipeline.get_run(run["id"])
        self.assertEqual(run["status"], "done")
        self.assertIsNotNone(r["verified_rate"])
        md = Path(r["files"]["md"]).read_text(encoding="utf-8")
        self.assertIn("## 参考文献", md)
        self.assertIn("AI 参与说明", md)
        self.assertIn("不生成实验数据", md)
        self.assertIn("[1]", md)
        self.assertTrue(Path(r["files"]["docx"]).exists())

    def test_run_stage_unknown_stage_fails_run(self):
        run = pipeline.create_run("agent evaluation methods")
        with self.assertRaises(ValueError):
            pipeline.run_stage(run["id"], {"stage": "nonsense"})
        self.assertEqual(pipeline.get_run(run["id"])["status"], "failed")

    def test_get_run_state_shape(self):
        run = pipeline.create_run("agent evaluation methods")
        self._seed_sections(run["id"])
        st = pipeline.get_run_state(run["id"])
        self.assertIn("run", st) and self.assertIn("outline", st) and self.assertIn("sections", st)
        self.assertEqual(st["sections"][0]["citations"]["1"]["paper_id"], self.p1)


if __name__ == "__main__":
    unittest.main()
