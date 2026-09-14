# -*- coding: utf-8 -*-
"""论文里的图，对这个系统原本完全不存在（全仓零图片抽取）。

用户问「那个架构图长什么样」「模型结构怎么连的」，检索侧一个字都没有。
`papernest/figures.py` 把图变成能被检索到的文字。这个文件钉住四件事，
每一件都是在真 PDF 上跑出来的，不是想出来的。

## ① 以图题为锚，不是 `page.get_images()`

真库 26 篇 233 页：位图 148 张、图题 70 条。看着位图多，指的却不是一回事——
**论文 460 有 4 条图题、0 张位图**（462 是 3 条 / 0 张），它们的图是矢量画的，
`get_images()` 根本看不见；反过来论文 1 有 93 张位图只有 5 条图题，那 93 张
多半是图里的小块和装饰。按位图抓会**漏掉整类矢量图**又抓回一堆碎片。

## ② 图题判据要求数字后跟冒号/句点

宽判据（`Figure N`）在真库上命中 75 条，严判据 70 条。多出的 5 条逐条看过，
全是正文句子：「Figure 1 presents our organizational framework…」
「Figure 3 shows that two baselines…」——一条真图题都没有。

## ③ 图区上界只认「正文段落」

不能取「图题上方最近的任意文本块」：插图里的坐标轴刻度、节点名、图例都是
独立文本块，而且落在图**内部**。取它们当上界，圈出来的只剩薄薄一条——
实测论文 459 第 1 页只圈到 86pt 高（真图 300 多），4 条图题只定位出 2 张。
改成只认正文段落后，459 的 4 张图全部定位到，高度 156/319/182/287pt。

## ④ 说明里**不许读数**

第一版让模型「说明关键点」，跑真图核对时抓到一条实质性错误：
论文 459 图 3（三组余弦相似度折线图），模型说「绿色线（MLM+BRLM-SA）在
第 1–5 层接近 1.0，**第 6 层骤降**」。对着图看，绿线恰恰是唯一**不掉**的那条
（1.0 → 0.98）——而「BRLM-SA 能把相似度保持住」正是这篇论文的论点，
模型把图的结论说反了。同一段里结构性的断言（子图划分、坐标轴名、四条曲线的
图例）全部正确，错的只集中在读数值这一类。

加约束后重跑，读数完全消失；但**残留一类更轻的错误**：说明里写「每组含五条
曲线」紧接着列了四种模型（图里是四条）——数量会数错。所以贴进检索块的说明
必须带 `MARK` 前缀：它是检索的入口，不是可引用的事实来源。
"""
import re
import unittest
from unittest import mock

from papernest import config, figures

PDF_DIR = config.ROOT / "data" / "pdf"


class CaptionRegexTests(unittest.TestCase):
    """② 数字后必须跟冒号或句点。"""

    def test_real_captions_match(self):
        for s in ("Figure 2: The overview of BRLM.",
                  "Fig. 1: An overview of the ecosystem",
                  "Figure 4a. Cosine similarity",
                  "图 3：模型结构",
                  "FIGURE 10: Results"):
            with self.subTest(s=s):
                self.assertTrue(figures.CAPTION_RE.match(s), s)

    def test_sentences_that_merely_mention_a_figure_do_not(self):
        """真库上正是这 5 条把宽判据的 75 撑到比严判据的 70 多出来。"""
        for s in ("Figure 1 presents our organizational framework for understanding",
                  "Figure 1 shows an example of OLAC metadata.",
                  "Figure 1 illustrates the overall architecture of the model",
                  "Figure 1a also shows the cities mentioned most often",
                  "Figure 3 shows that two baselines achieve similar tradeoff"):
            with self.subTest(s=s):
                self.assertIsNone(figures.CAPTION_RE.match(s), s)


class ProseGuardTests(unittest.TestCase):
    """③ 图区上界只认正文段落——图内标签不算。"""

    def _b(self, text):
        return (0.0, 0.0, 100.0, 10.0, text, 0, 0)

    def test_axis_labels_are_not_prose(self):
        for t in ("Encoder Layer", "0.9", "MLM+BRLM-SA", "Aligner Tool", "y1"):
            self.assertFalse(figures._is_prose(self._b(t)), t)

    def test_a_real_paragraph_is_prose(self):
        self.assertTrue(figures._is_prose(self._b(
            "To the best of our knowledge, no work has addressed word order "
            "divergence in transfer learning for multilingual NMT.")))


@unittest.skipUnless((PDF_DIR / "459.pdf").exists(), "需要真 PDF 459")
class RealPdfFigureLocationTests(unittest.TestCase):
    """③ 真 PDF 回归：459 的四张图都要定位到，且不能只圈出一条缝。"""

    @classmethod
    def setUpClass(cls):
        cls.figs = figures.find_figures(str(PDF_DIR / "459.pdf"))

    def test_all_four_figures_are_found(self):
        self.assertEqual(len(self.figs), 4,
                         f"只定位到 {len(self.figs)} 张（应为 4）："
                         f"{[f['label'] for f in self.figs]}")

    def test_none_of_them_is_a_sliver(self):
        """只认「上方最近的任意文本块」时，第 1 页只圈到 86pt 高。"""
        thin = [(f["label"], round(f["height"])) for f in self.figs
                if f["height"] < 120]
        self.assertEqual(thin, [], f"这些图只圈出一条缝：{thin}")

    def test_regions_stay_inside_one_column(self):
        """双栏论文里不收口的话会把邻栏正文一起渲进来。"""
        wide = [(f["label"], round(f["width"])) for f in self.figs
                if f["width"] > 560]
        self.assertEqual(wide, [], f"图区横跨了整页：{wide}")

    def test_hash_is_stable_across_reruns(self):
        again = figures.find_figures(str(PDF_DIR / "459.pdf"))
        a = [figures.figure_hash(459, f["page_no"], f["caption"]) for f in self.figs]
        b = [figures.figure_hash(459, f["page_no"], f["caption"]) for f in again]
        self.assertEqual(a, b, "哈希不稳定的话，重切一次块摘要就得重新买")


class NoNumbersInThePromptContractTests(unittest.TestCase):
    """④ 「不许读数」这条约束是靠提示词兜的，它必须真的写在提示词里。"""

    def test_the_prompt_forbids_reading_values(self):
        p = figures._SYSTEM
        self.assertIn("严禁读数", p)
        for word in ("骤降", "峰值", "百分比"):
            self.assertIn(word, p, f"提示词没点名要禁的「{word}」这类说法")

    def test_the_prompt_says_where_conclusions_come_from(self):
        """只说「不许」不够——得告诉模型结论该由谁来给，否则它会绕着写。"""
        self.assertIn("正文和表格", figures._SYSTEM)


class ProvenanceTests(unittest.TestCase):
    def test_decorate_marks_model_written(self):
        got = figures.decorate("Figure 2: The overview of BRLM.", "这是一张架构图。")
        self.assertTrue(got.startswith(figures.MARK))
        self.assertIn("Figure 2", got, "图题（论文原文）被说明顶掉了")

    def test_mark_distinguishes_model_text_from_the_paper(self):
        self.assertIn("非原文", figures.MARK)
        self.assertIn("模型", figures.MARK)

    def test_decorate_is_idempotent(self):
        once = figures.decorate("cap", "说明")
        self.assertEqual(figures.decorate(once, "说明"), once)

    def test_blank_summary_does_not_leave_an_empty_mark(self):
        """空标记比不贴更糟：它宣称这里有段模型写的说明，其实一个字都没有。"""
        for empty in (None, "", "   "):
            self.assertEqual(figures.decorate("cap", empty), "cap")


class SpendingIsGatedTests(unittest.TestCase):
    def test_describe_defaults_to_dry_run(self):
        items = [{"paper_id": 1, "page_no": 2, "label": "1", "caption": "c",
                  "figure_hash": "h", "width": 400.0, "height": 300.0,
                  "pdf_path": "x.pdf", "title": "t"}]
        with mock.patch.object(figures, "_ask_vision",
                               side_effect=AssertionError("默认就花钱了")):
            r = figures.describe(items)
        self.assertTrue(r["dry_run"])
        self.assertEqual(r["written"], 0)

    def test_vision_model_defaults_to_off(self):
        """按图片像素计费的东西，不能悄悄给一个默认值替用户决定花钱。"""
        import os
        if os.environ.get("PAPERNEST_VISION_MODEL"):
            self.skipTest("本机显式配了视觉模型")
        self.assertEqual(config.VISION_MODEL, "")

    def test_estimate_calls_nothing(self):
        items = [{"paper_id": 1, "page_no": 2, "width": 400.0, "height": 300.0}]
        with mock.patch.object(figures, "_ask_vision",
                               side_effect=AssertionError("估算不该调模型")):
            e = figures.estimate(items)
        self.assertEqual(e["n_figures"], 1)
        self.assertGreater(e["est_total_tokens"], 0)


class OnlyDescribedFiguresBecomeChunksTests(unittest.TestCase):
    def test_build_chunks_skips_figures_without_a_summary(self):
        """没有说明的话，块里就只剩图题——而图题本来就在正文块里，白占一个检索单元。"""
        import inspect

        from papernest import fulltext
        src = inspect.getsource(fulltext.build_chunks)
        self.assertIn("figures", src)
        i = src.index("figures.find_figures")
        self.assertIn("continue", src[i:i + 400],
                      "没有说明的图也成块了")


if __name__ == "__main__":
    unittest.main()
