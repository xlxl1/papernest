# -*- coding: utf-8 -*-
"""表格文本在库里存了两份。

`fulltext.build_chunks` 走两条腿：`structure.section_chunks` 切正文块，
`tables.chunk_table` 切表格块。可这两条腿读的是同一页 PDF——表格里的每一行，
在正文块里还有**一模一样的一份**。真库量下来一字不差：

    表格区 39,027 字符  ==  全部 kind='table' 块的字符数 39,027

`tables.split_page_text` 本来就是为解决这件事写的（还配了 13 个测试），
但它**生产零调用**——没有任何一条入库路径用过它。

## 这有什么关系（不是「多占点空间」）

拿表头当查询去检索（也就是用户真正问到某张表的时候），上下文里同时出现
「表格块 + 含同样内容的正文块」的比例（真库 86 次表头查询）：

    每篇取 3 块  26%      每篇取 5 块  45%      每篇取 8 块  60%

问到表的时候，接近一半的上下文预算在读同一段内容两遍。

## 为什么不能直接用 split_page_text

它是**按页**切的。换上去等于把章节切分丢掉，而章节切分是本仓测出来最硬的收益
（等预算下证据召回 2.2~3.2 倍）。所以走的是另一条路：让 `section_chunks`
接一个 `detected_tables`，把落在表格区的 block 从正文块里摘掉，章节结构不动。

## 摘的判据（三条同时满足）

① 几何：block 垂直中心落在表框内；② 文本：`tables._line_owner` 也认；
③ 不是表题/图题。

单用①会把表题连坐（`chunk_table` 不带表题，摘了就丢）；单用②会把参考文献条目和
致谢摘走（真库上抓到 9 条，词面偶然命中）。两条叠加后，630 个被摘的 block 里
只剩 1 条是真正的正文误杀（0.16%）。

**误杀不等于丢失**：`pages` / `pages_fts` 里的整页原文不动，词面那一路照样召回。
丢的是语义检索的那一路，不是全部——这是这个取舍能接受的前提，
所以下面有一条用例专门钉住「原文还在 pages 里」。
"""
import re
import unittest

from papernest import config, structure, tables

PDF_DIR = config.ROOT / "data" / "pdf"
sq = lambda s: re.sub(r"\s+", "", s or "")


def _block(idx, page, x0, y0, text, h=9.0):
    return {"block_index": idx, "page_no": page, "x0": float(x0),
            "y0": float(y0), "y1": float(y0 + h), "text": text}


class TableOwnedBlocksTests(unittest.TestCase):
    """判据本身：合成数据，不依赖真 PDF。"""

    def setUp(self):
        self.table = {"page_no": 1, "bbox": (80.0, 100.0, 300.0, 200.0),
                      "n_rows": 3, "n_cols": 3,
                      "rows": [["Model", "BLEU", "chrF"],
                               ["Baseline", "24.1", "51.3"],
                               ["Ours", "27.8", "55.0"]]}

    def test_a_row_inside_the_box_is_table_owned(self):
        b = _block(0, 1, 82, 150, "Baseline 24.1 51.3")
        self.assertEqual(tables.table_owned_blocks([b], [self.table]), {0})

    def test_a_caption_touching_the_box_is_kept(self):
        """表格块不带表题，摘掉就等于把「这张表是干什么的」弄丢了。"""
        b = _block(0, 1, 82, 195, "Table 2: BLEU and chrF on the test set")
        self.assertEqual(tables.table_owned_blocks([b], [self.table]), set(),
                         "表题被当成表格行摘走了")

    def test_prose_outside_the_box_is_kept_even_if_words_match(self):
        """只靠词面时，参考文献与致谢会被误摘——几何这一条就是为挡它加的。"""
        b = _block(0, 1, 82, 600, "Baseline 24.1 51.3 is what we report in Table 2.")
        self.assertEqual(tables.table_owned_blocks([b], [self.table]), set())

    def test_prose_inside_the_box_that_shares_no_cells_is_kept(self):
        """只靠几何时，被表框圈住的正文会被连坐——文本这一条挡它。"""
        b = _block(0, 1, 82, 150,
                   "We now describe the evaluation protocol used throughout.")
        self.assertEqual(tables.table_owned_blocks([b], [self.table]), set())

    def test_no_tables_means_nothing_is_dropped(self):
        b = _block(0, 1, 82, 150, "Baseline 24.1 51.3")
        self.assertEqual(tables.table_owned_blocks([b], []), set())


class SectionChunksStayStructuredTests(unittest.TestCase):
    """去重不能以丢掉章节切分为代价——那是本仓最硬的那条收益。"""

    @classmethod
    def setUpClass(cls):
        cls.pdf = next((p for p in (PDF_DIR / "463.pdf", PDF_DIR / "461.pdf")
                        if p.exists()), None)
        if cls.pdf is None:
            raise unittest.SkipTest("需要真 PDF")
        cls.found = tables.detect_tables(str(cls.pdf))
        cls.plain = structure.section_chunks(str(cls.pdf))
        cls.deduped = structure.section_chunks(str(cls.pdf), detected_tables=cls.found)

    def test_dedup_actually_removes_something(self):
        a = sum(len(sq(c["text"])) for c in self.plain)
        b = sum(len(sq(c["text"])) for c in self.deduped)
        self.assertLess(b, a, "表格行没有从正文块里摘掉")

    def test_section_paths_survive(self):
        """摘的是 block，不是章节。章节路径必须一条不少。"""
        pa = {c.get("section_path") for c in self.plain}
        pb = {c.get("section_path") for c in self.deduped}
        self.assertEqual(pa - pb, set(), f"去重把整个章节摘没了：{pa - pb}")

    def test_not_passing_tables_keeps_the_old_behaviour(self):
        """默认不去重——调用方要显式选择，老调用点行为不变。"""
        again = structure.section_chunks(str(self.pdf))
        self.assertEqual([c["text"] for c in again], [c["text"] for c in self.plain])

    def test_dedup_does_not_eat_a_third_of_the_body(self):
        """闸门：真库全量降幅 2.1%，单篇最高 7.8%。摘掉超过三成一定是判据坏了。"""
        a = sum(len(sq(c["text"])) for c in self.plain)
        b = sum(len(sq(c["text"])) for c in self.deduped)
        self.assertGreater(b / max(1, a), 0.7,
                           f"正文被摘掉了 {1 - b/max(1,a):.0%}——判据失控了")


class DroppedTextIsStillReachableTests(unittest.TestCase):
    """去重的前提：摘掉的只是 chunk，整页原文还在 `pages` 里。

    这条一旦不成立，误杀就从「少一路检索」变成「彻底丢失」，
    上面那个 0.16% 的取舍就不再成立了。
    """

    def test_extract_pages_stores_untouched_page_text(self):
        import inspect
        src = inspect.getsource(__import__("papernest.fulltext", fromlist=["x"])
                                .extract_pages)
        self.assertIn('page.get_text("text")', src)
        self.assertIn("INSERT OR REPLACE INTO pages", src,
                      "整页原文不再入库的话，被摘掉的正文就真的没了")
        self.assertIn("reindex_pages", src, "页文本没进 FTS，词面那一路也救不回来")


if __name__ == "__main__":
    unittest.main()
