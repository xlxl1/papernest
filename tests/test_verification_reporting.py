# -*- coding: utf-8 -*-
"""可信度信号必须作用在产物上，且「验不了」不能记成「验不过」。

survey 是唯一输出不可逐句溯源的产品线，而改之前它对自己产物的可信度判断是坏的：
- `best_sentence` 在一篇论文没有句级向量时返回 (None, 0.0) 而**不抛异常**，
  except 接不住，于是 `bool(0.0 >= 0.35) = False`——「本库无句级向量」被记成
  「全部核验未通过」，是个把整篇综述判死刑的假警报；那句本该兜底的
  「本库无句级向量，未做核验」因此**永远不可达**。
- checks 只是一个平行列表，返回的 survey 仍是原文本，未通过核验的句子带着它的 [n]
  留在正文里、形态与通过核验的句子完全一致；下游 `agent.py` 只取裸文本，平行信息丢失。

顺带覆盖多轮历史截断的方向问题（chat.history_block）。
"""
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from papernest import chat, config, db, embeddings, survey


class SurveyVerificationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="papernest_surveyver_")
        self.addCleanup(shutil.rmtree, self._tmp, True)
        self._old_db, self._old_dir = config.DB_PATH, config.DATA_DIR
        config.DB_PATH = Path(self._tmp) / "test.db"
        config.DATA_DIR = Path(self._tmp) / "data"
        db.init_db()
        with db.conn() as c:
            self.pid = db.insert_l0(c, {
                "norm_key": "arxiv:7101", "title": "Near-Field Channel Estimation",
                "abstract": "Polar-domain dictionary for spherical wavefronts.",
                "year": 2023, "venue": "TWC", "authors": ["A B"],
                "doi": None, "arxiv_id": "7101", "source": "s2"})
        self.text = "近场信道估计依赖极化域字典来刻画球面波前 [1]。"

    def tearDown(self):
        config.DB_PATH, config.DATA_DIR = self._old_db, self._old_dir

    def _generate(self, best_sentence_result):
        with mock.patch("papernest.llm.available", return_value=True), \
             mock.patch("papernest.embeddings.search_hybrid",
                        return_value=embeddings.RetrievalResult([self.pid], "fts", [])), \
             mock.patch("papernest.llm.chat", return_value=self.text), \
             mock.patch("papernest.embeddings.embed_texts",
                        return_value=[[0.0, 0.0, 0.0, 0.0]]), \
             mock.patch("papernest.embeddings.best_sentence",
                        return_value=best_sentence_result):
            return survey.generate("近场信道估计", top_k=1)

    def test_no_sentence_vectors_is_unverified_not_failed(self):
        """(None, 0.0) 是「验不了」。原来被算成 ok=False，报「0/N 通过」——假警报。"""
        r = self._generate((None, 0.0))
        self.assertEqual([c["ok"] for c in r["checks"]], [None])
        self.assertEqual(r["checks"][0]["reason"], "该文献无句级向量")
        self.assertIn("未核验", r["verified_summary"])
        self.assertNotIn("未通过证据核验", r["survey"])   # 验不了不等于验不过，不该标注

    def test_low_similarity_is_annotated_on_the_product(self):
        """未通过核验的句子必须在产物上有痕迹，不能只躺在平行的 checks 里。"""
        r = self._generate(("some evidence", 0.21))
        self.assertEqual([c["ok"] for c in r["checks"]], [False])
        self.assertIn("未通过证据核验", r["survey"])
        self.assertIn("0.21", r["survey"])
        self.assertIn("未通过", r["verified_summary"])

    def test_annotation_adds_and_never_removes(self):
        """标注不删句：原文必须原样保留，口径同 rcs。"""
        r = self._generate(("some evidence", 0.21))
        self.assertTrue(r["survey"].startswith(self.text.rstrip("。")[:20]))
        self.assertEqual(r["survey_raw"], self.text)

    def test_passing_claims_are_left_alone(self):
        r = self._generate(("some evidence", 0.92))
        self.assertEqual([c["ok"] for c in r["checks"]], [True])
        self.assertEqual(r["survey"], self.text)
        self.assertIn("1 条通过", r["verified_summary"])


class HistoryBlockTests(unittest.TestCase):
    """历史截断的方向：原来两个方向都在做最坏选择。"""

    def test_assistant_caveats_at_the_tail_survive(self):
        """免责句几乎总在结尾，硬截 [:400] 恰好留下最自信的前半段。"""
        body = "近场信道估计有三类方法。" * 40
        tail = "以上第三点在库内文献未覆盖，未经核验。"
        turns = [{"role": "assistant", "content": body + tail}]
        block = chat.history_block(turns, max_chars=5000)
        self.assertIn("库内文献未覆盖", block)
        self.assertIn("中略", block)          # 两端保留，中间省略

    def test_user_constraints_are_dropped_last(self):
        """用户第一轮立下的约束原来是最先被 pop(0) 丢掉的——恰恰最该留住。"""
        turns = [{"role": "user", "content": "只看 2023 年之后的文献"}]
        for i in range(6):
            turns.append({"role": "assistant", "content": f"第{i}轮回答。" * 30})
        block = chat.history_block(turns, max_chars=300)
        self.assertIn("只看 2023 年之后", block)

    def test_short_assistant_message_is_not_mangled(self):
        turns = [{"role": "assistant", "content": "很短的一句回答。"}]
        self.assertIn("很短的一句回答。", chat.history_block(turns))
        self.assertNotIn("中略", chat.history_block(turns))


class FollowupDetectionTests(unittest.TestCase):
    """英文标记原来是子串匹配：'it' 命中 limitations / suite / critical。"""

    PRIOR = ["RAG 是什么"]

    def test_reference_markers_still_detected(self):
        for q in ("它的局限是什么", "为什么？", "第二篇讲得更细一点",
                  "展开讲讲", "继续", "why?", "compare these two"):
            self.assertTrue(chat.is_followup(q, self.PRIOR), q)

    def test_english_substring_no_longer_misfires(self):
        for q in ("多模态大模型的 limitations 有哪些",
                  "有哪些 benchmark suite 可用",
                  "critical path 分析怎么做"):
            self.assertFalse(chat.is_followup(q, self.PRIOR), q)

    def test_short_but_self_contained_question_is_not_a_followup(self):
        """「近场信道估计的最新进展」11 个字，是一条完整的独立查询，
        原来被 len(q) <= 12 一刀切判成追问，上一轮的问题被拼进检索。"""
        self.assertFalse(chat.is_followup("近场信道估计的最新进展", self.PRIOR))

    def test_no_prior_turns_is_never_a_followup(self):
        self.assertFalse(chat.is_followup("它的局限是什么", []))


if __name__ == "__main__":
    unittest.main()
