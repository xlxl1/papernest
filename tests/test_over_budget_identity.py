# -*- coding: utf-8 -*-
"""上下文预算把论文挤掉时，degraded 必须说出「是哪几篇」。

背景（真库实测）：超预算时只报篇数——「3 篇超出上下文预算（24000 字符）未纳入」。
而 [n] 是在预算过滤**之前**按检索名次分配的（`chat.assign_indices`），被挤掉的编号
就此消失：真库上 top_k=30 跑 eval_set r02「agent evaluation benchmark framework」，
sources idx 是 [1..26, 29]，27/28/30 是空洞，而那句提示里既没有编号也没有标题。
用户看到「来源 …26、29」和「3 篇未纳入」，两者对不上；运维拿 degraded_json 聚合
也只知道有几篇、不知道是哪几篇，事后无法复盘。

**审计另一半的指控不成立，所以没有改 `continue`**：审计说 best-fit「丢掉更相关的
论文而放进更靠后的短论文」。但 `continue` 相对 `break` 只会**多**放论文——两者在
第一次溢出之前逐条相同，`break` 就此停下，`continue` 只做 append，所以 best-fit 的
入选集合恒为严格 top-k 截断的**超集**（170 问题 × 5 档 top_k × 5 档预算 = 4250 个
组合，反例 0）。被丢掉的那篇在严格截断口径下同样进不来——它是被预算丢的，
不是被 `continue` 丢的。QASPER 88 题的六个配置 A/B 也全部 n_changed=0、p=1.0。
"""
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from papernest import config, db, rag


class OverBudgetIdentityTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="papernest_overbudget_")
        self.addCleanup(shutil.rmtree, self._tmp, True)
        self._old_db, self._old_dir = config.DB_PATH, config.DATA_DIR
        config.DB_PATH = Path(self._tmp) / "test.db"
        config.DATA_DIR = Path(self._tmp) / "data"
        # 向量 / LLM 按住：开发机 .env 里配的是真 key，不摁住会真发请求
        self._patches = [mock.patch("papernest.embeddings.available", return_value=False),
                         mock.patch("papernest.llm.available", return_value=False)]
        for p in self._patches:
            p.start()
        db.init_db()
        titles = ["AgentGym: Evolving Large Language Model-based Agents",
                  "Retrieval-Augmented Generation for Knowledge-Intensive NLP",
                  "A Survey of Web Navigation Agents"]
        self.ids = []
        with db.conn() as c:
            for n, t in enumerate(titles, 1):
                self.ids.append(db.insert_l0(c, {
                    "norm_key": f"arxiv:80{n}", "title": t,
                    "abstract": "agent evaluation benchmark framework. " * 20,
                    "year": 2024, "venue": "ACL", "authors": ["A B"],
                    "doi": None, "arxiv_id": f"80{n}", "source": "s2"}))

    def tearDown(self):
        for p in self._patches:
            p.stop()
        config.DB_PATH, config.DATA_DIR = self._old_db, self._old_dir

    def _prepare(self):
        return rag.prepare([{"role": "user", "content": "agent evaluation benchmark"}],
                           top_k=3, candidate_ids=self.ids, max_context_chars=700)

    def test_over_budget_note_names_the_dropped_papers(self):
        ctx = self._prepare()
        dropped = sorted({s["idx"] for s in ctx.sources} ^ {1, 2, 3})
        self.assertEqual(dropped, [2, 3],
                         "前置条件没成立：这条用例要求第 2、3 篇被预算挤掉")
        self.assertIsNotNone(ctx.degraded)
        self.assertIn("超出上下文预算", ctx.degraded)
        # 编号：sources 里出现的空洞必须能在降级提示里对上号
        for n in dropped:
            self.assertIn(f"[{n}]", ctx.degraded,
                          f"degraded 没说编号 [{n}] 被挤掉了：{ctx.degraded!r}")
        # 身份：光有编号还不够——被丢掉的论文不在 sources 里，编号查无此篇
        self.assertIn("Retrieval-Augmented", ctx.degraded,
                      f"degraded 没说被丢掉的是哪篇论文：{ctx.degraded!r}")

    def test_over_budget_note_still_reports_the_count(self):
        """加身份不能把原来的篇数口径弄丢（前端 10 处按字符串渲染这句话）。"""
        ctx = self._prepare()
        self.assertIn("2 篇超出上下文预算（700 字符）未纳入", ctx.degraded)


if __name__ == "__main__":
    unittest.main()
