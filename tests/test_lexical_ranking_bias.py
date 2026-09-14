# -*- coding: utf-8 -*-
"""词面选块原来基本是在「按块长排序」，现在按 IDF 加权。

## 原来的毛病

`_query_terms` **故意不过滤停用词**，理由是「均匀出现的词对排序没有贡献、
会自己抵消」。那句话对长度相近的页成立；对 `_rank_chunks` 不成立——
它的打分是 `count / sqrt(len)`，长块的停用词计数随长度线性涨、分母只涨 sqrt，
于是块长本身变成了分数。

真库实测（26 篇 / 692 块）：**只喂十个停用词**，选出的 50 个块里 34 个（68%）
是该篇最长的前 5 块（随机约 21%）；英文查询的词频命中里**中位 94%**
来自「出现在过半块里」的词。

## 为什么当初没改

2026-09-09 第一次试 IDF 时，评测集只有 25 篇 / 69 题——`p=0.219`，判不出来。
按本仓口径（小评测集必须做显著性检验）**没有采用**，并把负结果记在了这里。

后来把 QASPER dev 的 PDF 补齐（`tools/fetch_qasper_pdfs.py`，274 篇 / 917 题 /
4556 条 gold），同一个改动 **p=5e-05**。**当时不是改动没用，是样本量判不出来。**

## 现在的口径（917 题实测）

    配置                                问题召回   片段召回
    per=2, 无 IDF（原始）                0.1723   0.0904
    per=3, 无 IDF                       0.1974   0.0961
    per=3, log(1+N/(1+df))              0.2203   0.1036
    per=3, log(N/df) 截 0               0.2694   0.1234
    per=3, BM25 式  ← 现行               0.2704   0.1247

IDF 的**形式**也试错过一次：先写的 `log(1+N/(1+df))` 在 df=N 时仍有 log(2)≈0.69，
压不住停用词的几百次重复，只拿到一半的收益（0.2203 vs 0.2704，两者 p=5e-05）。

`per_paper` 不再是常数，改成由窗口下限反推（`TARGET_WINDOW_CHARS=800`）：
等预算扫描下片段召回在每窗 800 字见顶（800→0.0961 / 600→0.0915 / 400→0.0832）。
问题召回一路涨到 per=6，但那是把上下文剁成 400 字碎片换来的——证据总量在掉。
"""
import math
import unittest

from papernest import rag

STOPWORDS = ["the", "of", "is", "in", "and", "to", "we", "for", "as", "by"]


def _stopword_chunk(no, n_chars):
    filler = "the of is in and to we for as by "
    return {"chunk_no": no, "kind": "text",
            "text": (filler * (n_chars // len(filler) + 1))[:n_chars]}


class UbiquitousTermsAreDiscountedTests(unittest.TestCase):
    """出现在每个候选块里的词，IDF 应当把它压到接近 0。"""

    def test_a_term_in_every_chunk_carries_almost_no_weight(self):
        n, df = 20, 20
        idf_all = math.log(1 + (n - df + 0.5) / (df + 0.5))
        idf_rare = math.log(1 + (n - 1 + 0.5) / (1 + 0.5))
        self.assertLess(idf_all, 0.05, "满篇都有的词还有权重")
        self.assertGreater(idf_rare / max(idf_all, 1e-9), 50,
                           "稀有词与满篇词的权重差不到 50 倍——压不住几百次重复；"
                           "log(1+N/(1+df)) 就是栽在这里（收益只有一半）")

    def test_rank_chunks_uses_an_idf_that_vanishes(self):
        """闸门：换回 df=N 时不归零的形式，收益会掉一半（实测 0.2704 → 0.2203）。"""
        import inspect
        src = inspect.getsource(rag._rank_chunks)
        self.assertIn("idf", src)
        self.assertIn("df", src)
        self.assertNotIn("log(1 + n_docs / (1 + df))", src.replace(" ", " "),
                         "又换回了 df=N 时不归零的那个形式")

    def test_a_rare_term_beats_stopword_repetition_when_there_is_a_corpus(self):
        """候选集够大时（真实场景是一篇 10~70 块），停用词被压平，稀有词说了算。"""
        chunks = [_stopword_chunk(i, 3000) for i in range(12)]
        chunks.append({"chunk_no": 99, "kind": "figure",
                       "text": "cosine similarity across encoder layers " * 3})
        terms = rag._query_terms("cosine similarity of the encoder layers")
        picked, _ = rag._rank_chunks(chunks, [], terms, 1)
        self.assertEqual(picked[0]["chunk_no"], 99,
                         "12 个纯停用词的长块仍然压过了唯一相关的那个块")

    def test_lexically_blind_is_still_reported(self):
        """词面全 0 时要如实说「只能靠向量」——IDF 不能把这个信号弄丢。"""
        chunks = [{"chunk_no": 1, "kind": "text", "text": "totally unrelated prose"}]
        hits = [{"chunk_no": 7, "text": "from the vector side"}]
        picked, blind = rag._rank_chunks(chunks, hits, ["量子", "纠缠"], 1)
        self.assertTrue(blind)
        self.assertEqual(picked[0]["chunk_no"], 7)


class WindowFloorDrivesChunkCountTests(unittest.TestCase):
    """`per_paper` 由窗口下限反推，不是拍的常数。"""

    def test_the_floor_is_the_measured_peak(self):
        self.assertEqual(rag.TARGET_WINDOW_CHARS, 800,
                         "改这个数要先重跑等预算扫描：片段召回在 800 字见顶"
                         "（800→0.0961 / 600→0.0915 / 400→0.0832）")
        self.assertEqual(rag.MAX_CHUNKS_PER_PAPER, 3)

    def test_prepare_derives_it_from_the_budget(self):
        import inspect
        src = inspect.getsource(rag.prepare)
        self.assertIn("TARGET_WINDOW_CHARS", src, "又写回常数了")
        self.assertIn("MAX_CHUNKS_PER_PAPER", src)


class TheMeasurementsAreRecordedTests(unittest.TestCase):
    """数字必须留在代码里——否则下一个人只能重跑一遍才知道为什么是这样。"""

    def test_rank_chunks_records_the_bias_and_the_fix(self):
        d = rag._rank_chunks.__doc__ or ""
        for token in ("68%", "94%", "0.2704", "0.2203"):
            self.assertIn(token, d, f"docstring 里没有 {token} 这个实测")

    def test_it_records_that_the_small_eval_set_could_not_decide(self):
        """这条最容易被后人误读成「试过没用」。"""
        d = rag._rank_chunks.__doc__ or ""
        self.assertIn("69", d)
        self.assertIn("917", d, "没写清楚是扩样本之后才判出来的")



class WindowCenteringIsMostlyNominalTests(unittest.TestCase):
    """`_window` 声称「围绕命中位置取窗口」，实际多数时候等于 `text[:cap]`。

    `terms` 带停用词，`min(find(t))` 取的是任意词的首个出现位置，而 the/of
    几乎必然在开头：241 个真实块上，**首个命中中位在 51 字符、68% 落在前 100 内**，
    于是 `start = max(0, pos - cap//3)` = 0。

    ## 试过按加权命中密度选窗口，**没能证明更好**

    274 篇 / 924 题 / 4566 条探针：截窗留存 42.7% → 45.4%，
    问题级 p=0.0667，探针级 p=0.0834（**263 好 / 224 坏**）。
    263 对 224 就是抛硬币，而 n=4566 已经够有功效——**不能再说「判不出来」**。
    机制上更合理 ≠ 实测更好，所以没采用。

    这个文件不改行为，只把「试过、没用」钉住，免得下一个人重做一遍——
    也免得有人只看见「窗口没真的居中」就动手。
    """

    def test_a_stopword_at_the_start_pins_the_window(self):
        text = "The " + "x" * 2000 + " cosine similarity across encoder layers " + "y" * 2000
        got = rag._window(text, ["the", "cosine", "similarity"], 800)
        self.assertFalse(got.startswith("…"),
                         "窗口居中生效了？那本文件记录的实测结论需要重跑")
        self.assertNotIn("cosine", got, "证据在块中部，被截掉了——这正是现状")

    def test_without_stopwords_it_does_centre(self):
        """机制本身是对的，被喂进来的词毁了——这条区分「实现坏了」和「输入坏了」。"""
        text = "The " + "x" * 2000 + " cosine similarity across encoder layers " + "y" * 2000
        got = rag._window(text, ["cosine", "similarity"], 800)
        self.assertTrue(got.startswith("…"))
        self.assertIn("cosine", got)

    def test_the_negative_result_is_recorded(self):
        d = rag._window.__doc__ or ""
        self.assertIn("263", d, "没记「263 好 / 224 坏」——下个人会重做一遍")
        self.assertIn("0.0834", d)
        self.assertIn("没有采用", d)

if __name__ == "__main__":
    unittest.main()
