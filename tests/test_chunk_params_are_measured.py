# -*- coding: utf-8 -*-
"""切片参数（`max_chars` / `overlap`）：默认值是**量出来的**，别凭直觉改。

## 现行：max_chars=1000（2026-09-10 实测）

274 篇 / **924 题** / 4566 条 gold 探针，等预算 2400，口径与生产一致
（`rag._rank_chunks` 选块、`rag._window` 取窗口、`per_paper=3` 由窗口下限反推）：

    块上限   平均块数   问题召回   片段召回   vs 4000
      300     158.2    0.1894    0.0655   p=5e-05 显著更差
      500      94.9    0.2532    0.1067   p=0.110
      700      68.6    0.3149    0.1432   p=0.038 显著
     1000      49.1  **0.3366    0.1656** p=0.0004 显著   ← 峰值
     1500      34.7    0.3236    0.1551   p=0.0024 显著
     4000      19.5    0.2803    0.1334   ← 原默认

倒 U 形。机制是**截窗**：单块窗口 800 字，4000 字的块要丢掉 80%，
1000 字的块基本能整块进上下文；小到 300/500 反而更差——块比窗口还小，
每块贡献不足 800 字，总文本量下降，gold 片段也更容易跨块被切断。

第二个评测集（自建中文集，纯 FTS 口径）改前改后**逐项一致**
（Recall@5 0.7552 / 核验率 0.9 / precision@5 0.225），零回归。

## 这条结论**推翻了 2026-09-07 那次的**，原因要记清楚

上一次扫描（69 题）得到的是完全相反的排序：1000→17.4%、2000→21.7%、
**4000→26.1%**，据此把 4000 定成「拐点」。三个前提后来全变了：

  ① **样本量** 69 → 924。旧扫描里 1000 vs 4000 的 p=0.145，本来就判不出来。
  ② **gold 口径**：35% 的 gold 含 LaTeX 占位符（BIBREF/`$…$`/DBLP），
     与 PDF 永远匹配不上，等于把天花板压低了 17 个点（见 `qasper.gold_probes`）。
  ③ **选块与窗口**：旧口径是 `per_paper=2` / 窗口 1200、且排序被块长主导；
     现在是 `per_paper=3` / 窗口 800 + BM25 式 IDF。

③ 尤其关键：**块大小和 per_paper/窗口是一对，不能分开调**。
旧口径下窗口 1200 > 块 1000，小块整块进上下文但只进 2 块（2400 字）；
新口径下 3 块 × 800 = 2400 字，同样预算装下更多**不同**的块。
所以下次改 `rag.TARGET_WINDOW_CHARS` 或 `MAX_CHUNKS_PER_PAPER`，
**必须连这张表一起重跑**。

## overlap 仍然不加

4000 档上 overlap=10% 与 0% 逐题完全相同（p=1.0），25% 更差。
加一个查不出收益的参数，代价是每次重切都多付一份存储与嵌入费用。
（这条是在旧口径下测的，新口径下没有重测——真要加 overlap 得先重扫。）
"""
import unittest

from papernest import db, docimport, embeddings, structure


class ChunkDefaultsAreTheMeasuredOnesTests(unittest.TestCase):
    def test_the_default_is_the_measured_peak(self):
        """改这个值之前先重跑块大小扫描——而且要连窗口参数一起扫（见文件注释）。"""
        self.assertEqual(structure.DEFAULT_CHUNK_CHARS, 1000)
        import inspect
        sig = inspect.signature(structure.section_chunks)
        self.assertEqual(sig.parameters["max_chars"].default,
                         structure.DEFAULT_CHUNK_CHARS,
                         "签名默认值和常量对不上，改一处漏一处")

    def test_the_hard_cap_is_separate_from_the_target(self):
        """`MAX_CHUNK_CHARS` 是**不变量**（三条产 chunk 的路都不许越过），
        `DEFAULT_CHUNK_CHARS` 是**目标值**。两者不是一回事：目标 1000 仍然满足上限 4000。
        混成一个数的话，调切分参数会顺手改掉一条数据完整性约束。"""
        self.assertEqual(db.MAX_CHUNK_CHARS, 4000)
        self.assertEqual(docimport.MAX_CHUNK_CHARS, db.MAX_CHUNK_CHARS,
                         "三条产 chunk 的路必须同一个上限")
        self.assertLessEqual(structure.DEFAULT_CHUNK_CHARS, db.MAX_CHUNK_CHARS)

    def test_chunk_size_is_coupled_to_the_window(self):
        """块长的最优值取决于窗口——4000 曾经最好，正是因为当时窗口是 1200 / 每篇 2 块。

        这条闸门在提醒：动 `TARGET_WINDOW_CHARS` 或 `MAX_CHUNKS_PER_PAPER`
        就必须重扫块长。
        """
        from papernest import rag
        self.assertGreaterEqual(
            structure.DEFAULT_CHUNK_CHARS, rag.TARGET_WINDOW_CHARS,
            "块比窗口还小的话，每块贡献不足一个窗口，总文本量下降"
            "（实测 300/500 就是这么掉下去的）")

    def test_there_is_still_no_overlap_parameter(self):
        """没有 overlap 是**测过之后的决定**，不是漏了。

        4000 档上 overlap=10% 与 0% 逐题完全相同（p=1.0），25% 更差。
        真要加，先让 chunksweep 显示出显著收益。
        """
        import inspect
        for fn in (structure.section_chunks, structure._section_chunks, structure._pack):
            with self.subTest(fn=fn.__name__):
                self.assertNotIn("overlap", inspect.signature(fn).parameters,
                                 f"{fn.__name__} 加了 overlap 参数——"
                                 f"请附上 chunksweep 的显著性结果")

    def test_cap_stays_below_the_embedding_budget(self):
        """这是硬约束不是调参：超过嵌入侧上限的块，块尾静默不进向量。"""
        self.assertLess(db.MAX_CHUNK_CHARS, embeddings.INPUT_CHARS)

    def test_the_sweep_is_runnable(self):
        """结论必须带可复现的命令——本仓被审计点名过「头条实验无脚本」。"""
        from papernest import qasper
        self.assertTrue(callable(getattr(qasper, "chunk_sweep", None)),
                        "qasper.chunk_sweep 没了，那上面那张表就无从复现")
        import inspect
        src = inspect.getsource(qasper.chunk_sweep)
        self.assertIn("sign_flip_test", src, "扫参没做显著性检验")


if __name__ == "__main__":
    unittest.main()
