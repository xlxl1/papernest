# -*- coding: utf-8 -*-
"""表格块存的是原样 Markdown，然后整块拿去做 embedding。

    | Model | BLEU | chrF |
    | Baseline | 24.1 | 51.3 |
    | Ours | 27.8 | 55.0 |

用户问「哪个方法效果最好」，跟这堆数字在语义空间里几乎没有距离——
「谁跟谁比、比的是什么、谁赢了」在表里一个字都没写，它只存在于读者脑子里。
词面那一路也救不了：「效果最好」在表格里同样是零命中。

`papernest/tablesum.py` 给每张表配一段模型写的说明，和原表放在**同一个块**里
一起嵌入：说明负责被找到，原表负责回答具体是多少。副本实跑的样子（论文 468）：

    「……使用的评估指标为词错误率（WER）……数据显示，LangID-All 模型在所有
      指标上表现最好，且带有语言识别的模型整体优于不带语言识别的模型及 wFST 基线。」

这个文件钉住五件事，每一件都是实跑时踩到的：

① **出处**。摘要是模型生成的，贴进块正文时必须带 `MARK` 前缀。逐字核验那几条路
   （`matrix` / `rcs`）读的是 `pages` 表的整页原文，天然碰不到这段字；但人和模型
   都会在上下文里看到它，出处要写在脸上。

② **幂等**。`decorate` 贴第二遍会让同一段摘要在块里出现两次，
   而 `build_chunks` 每次重切都会调它。

③ **不跟着 chunks 走**。摘要按表的内容哈希存。`db.replace_chunks` 会把一篇的
   chunks 整个换掉，摘要要是只存在 `chunks.text` 里，每重切一次就得重新买一遍。

④ **账要估得准**。第一版把输出按目标字数（200 字）折算，实跑下来是估值的
   **2.7 倍**——这类模型的思考 token 也计费，200 字的说明记了 500~1300 token。
   低估账单比不估更糟：人是看着这个数按下回车的。

⑤ **表被抽坏时不许下结论**。真库 86 张表里 24 张（28%）列结构塌了，
   在那种表上模型会把论文的论点说反。首轮 60 条里有 17 条出自这类表、
   其中 14 条带结论性措辞，已全部作废重生成。详见最后一组用例。
"""
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from papernest import config, db, tablesum


class TableHashIsStableTests(unittest.TestCase):
    """哈希要跨「重新检出」稳定，否则摘要复用不了，每次重切都重新花钱。"""

    T = {"page_no": 3, "n_rows": 2, "n_cols": 2, "bbox": (10.0, 20.0, 100.0, 60.0),
         "rows": [["Model", "BLEU"], ["Ours", "27.8"]]}

    def test_bbox_and_counts_do_not_affect_the_hash(self):
        """同一张表在不同版本的检出里 bbox 会微动——那不该让摘要作废。"""
        moved = dict(self.T, bbox=(10.4, 20.1, 100.2, 60.3), n_rows=99)
        self.assertEqual(tablesum.table_hash(7, self.T),
                         tablesum.table_hash(7, moved))

    def test_whitespace_in_cells_is_normalised(self):
        spaced = dict(self.T, rows=[["Model ", " BLEU"], ["Ours", "27.8"]])
        self.assertEqual(tablesum.table_hash(7, self.T),
                         tablesum.table_hash(7, spaced))

    def test_different_content_gets_a_different_hash(self):
        other = dict(self.T, rows=[["Model", "BLEU"], ["Ours", "27.9"]])
        self.assertNotEqual(tablesum.table_hash(7, self.T),
                            tablesum.table_hash(7, other))

    def test_the_same_table_in_two_papers_is_two_tables(self):
        self.assertNotEqual(tablesum.table_hash(7, self.T),
                            tablesum.table_hash(8, self.T))


class ProvenanceIsOnTheFaceOfItTests(unittest.TestCase):
    """摘要是模型写的。上下文里的每一个字，读的人都要能分清谁写的。"""

    def test_decorate_marks_the_text_as_model_written(self):
        got = tablesum.decorate("| Model | BLEU |", "这张表比较了两个模型的 BLEU。")
        self.assertTrue(got.startswith(tablesum.MARK), "摘要没有标出处")
        self.assertIn("| Model | BLEU |", got, "原表被摘要顶掉了")

    def test_decorate_is_idempotent(self):
        """`build_chunks` 每次重切都会调它——贴两遍就是同一段话说两次。"""
        once = tablesum.decorate("| a | b |", "摘要")
        twice = tablesum.decorate(once, "摘要")
        self.assertEqual(once, twice)
        self.assertEqual(twice.count(tablesum.MARK), 1)

    def test_no_summary_means_no_change(self):
        for empty in (None, "", "   "):
            self.assertEqual(tablesum.decorate("| a | b |", empty), "| a | b |")

    def test_the_mark_says_it_is_not_the_original_text(self):
        """标记的措辞是这条约束的全部载体——只写「表格摘要」是不够的。"""
        self.assertIn("非原文", tablesum.MARK)
        self.assertIn("模型生成", tablesum.MARK)


class SummariesSurviveRechunkingTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="papernest_tsum_")
        self.addCleanup(shutil.rmtree, self._tmp, True)
        self._old = (config.DB_PATH, config.DATA_DIR)
        config.DATA_DIR = Path(self._tmp) / "data"
        config.DB_PATH = config.DATA_DIR / "t.db"
        self.addCleanup(self._restore)
        db.init_db()

    def _restore(self):
        config.DB_PATH, config.DATA_DIR = self._old

    def test_the_table_is_its_own_key_not_a_chunk_row(self):
        """存在自己的表里、按内容哈希索引，`replace_chunks` 碰不到它。"""
        with db.conn() as c:
            cols = {r[1] for r in c.execute("PRAGMA table_info(table_summaries)")}
        self.assertIn("table_hash", cols)
        self.assertIn("is_model_written", cols,
                      "数据层没有记「这是模型写的」，日后没法把它和原文区分开")
        self.assertNotIn("chunk_no", cols,
                         "按 chunk_no 存的话，重切一次块摘要就得重新买一遍")

    def test_round_trip(self):
        t = {"page_no": 2, "n_rows": 3, "n_cols": 2, "rows": [["a", "b"], ["1", "2"]]}
        h = tablesum.table_hash(5, t)
        with db.conn() as c:
            c.execute("INSERT INTO table_summaries(table_hash,paper_id,page_no,"
                      "n_rows,n_cols,summary,model,is_model_written)"
                      " VALUES(?,?,?,?,?,?,?,1)", (h, 5, 2, 3, 2, "一段说明", "m"))
        self.assertEqual(tablesum.get(h), "一段说明")
        self.assertEqual(tablesum.get_many(5), {h: "一段说明"})
        self.assertEqual(tablesum.get_many(6), {})

    def test_build_chunks_reattaches_the_summary(self):
        """重切之后摘要要自己贴回去——否则「跨重切复用」只是句空话。"""
        import inspect
        from papernest import fulltext
        src = inspect.getsource(fulltext.build_chunks)
        self.assertIn("tablesum", src, "build_chunks 没去取已存的摘要")
        self.assertIn("decorate", src, "取了但没贴到表格块上")


class TheBillIsHonestTests(unittest.TestCase):
    """账单是人按下回车的依据。低估比不估更糟。"""

    ITEMS = [{"paper_id": 1, "page_no": 2, "n_rows": 5, "n_cols": 3,
              "n_chars": 400, "markdown": "x" * 400, "table_hash": "h1"},
             {"paper_id": 1, "page_no": 3, "n_rows": 9, "n_cols": 4,
              "n_chars": 800, "markdown": "y" * 800, "table_hash": "h2"}]

    def test_output_is_estimated_from_measurement_not_target_length(self):
        """实测：约 200 字的说明记了 478~1259 completion token（含思考 token）。
        按 200 字折算会低估 3~4 倍，第一版就是这么错的。"""
        self.assertGreater(tablesum.MEASURED_COMPLETION_TOKENS,
                           tablesum.TARGET_CHARS * 2,
                           "输出估算又退回按目标字数折算了")
        est = tablesum.estimate(self.ITEMS)
        self.assertEqual(est["est_completion_tokens"],
                         2 * tablesum.MEASURED_COMPLETION_TOKENS)

    def test_estimate_spends_nothing(self):
        with mock.patch.object(tablesum.llm, "chat",
                               side_effect=AssertionError("估算不该调模型")):
            tablesum.estimate(self.ITEMS)

    def test_summarize_defaults_to_dry_run(self):
        with mock.patch.object(tablesum.llm, "chat",
                               side_effect=AssertionError("默认就花钱了")):
            r = tablesum.summarize(self.ITEMS)
        self.assertTrue(r["dry_run"])
        self.assertEqual(r["written"], 0)
        self.assertEqual(r["skipped"], 2)

    def test_one_bad_table_does_not_abort_the_batch(self):
        """一张表失败就整批中断的话，前面已经花掉的钱全白花。"""
        calls = []

        def fake(system, user, purpose, **kw):
            calls.append(purpose)
            if len(calls) == 1:
                raise RuntimeError("boom")
            return "第二张的说明"

        with mock.patch.object(tablesum.llm, "chat", fake), \
             mock.patch.object(tablesum.llm, "available", return_value=True), \
             mock.patch.object(tablesum.db, "init_db"), \
             mock.patch.object(tablesum.db, "conn"):
            r = tablesum.summarize(self.ITEMS, dry_run=False)
        self.assertEqual(r["written"], 1)
        self.assertEqual(len(r["failed"]), 1)
        self.assertIn("boom", r["failed"][0]["error"])


class DetectionNoiseIsNotWorthPayingForTests(unittest.TestCase):
    """真库 102 张检出表里有 14 张只有 1 个数据行。

    对它们模型只会如实回答「表格内容缺失，无法判断」——实跑过，论文 482
    第 5 页那张的全部内容就是一个 "ei"。花钱买这句话，还要把它塞进检索块里
    当噪声，两头都不划算。
    """

    def test_a_one_row_detection_is_skipped(self):
        self.assertGreaterEqual(tablesum.MIN_ROWS, 2)

    def test_the_floor_also_covers_tiny_but_multi_row_tables(self):
        self.assertGreater(tablesum.MIN_CHARS, 0,
                           "只卡行数的话，2 行 x 2 列的 20 个字符照样会被买单")


if __name__ == "__main__":
    unittest.main()


class MangledTablesGetNoConclusionTests(unittest.TestCase):
    """从 PDF 抽出来的表，28% 列结构是塌的——在那种表上不许让模型下结论。

    真库 86 张表里 24 张有「一个单元格里挤着多个数字」的格子，那是多级表头
    被压平的信号。论文 463 第 3 页那张的 Bengali 行就是 `6.72 8.83 9.19`
    （No Pre-Order / Pre-Order HT / Pre-Order G）——预排序把 BLEU 从 6.72
    提到 8.83，而模型写出来的结论是「所有语言在无预排序条件下的得分均高于
    预排序条件，预排序反而降低了翻译得分」，**把这篇论文的论点说反了**。

    它不是在瞎编：在一张列已经错位的表上，它没法知道哪个数字属于哪一列。
    所以这不是提示词写得不够严的问题，是**不该问这个问题**。

    首轮实跑落库的 60 条里，17 条出自这类表，其中 14 条带结论性措辞
    （最好 / 优于 / 提升 / 降低）——已全部作废重生成。
    """

    def _t(self, rows):
        return {"page_no": 1, "n_rows": len(rows), "n_cols": len(rows[0]),
                "rows": rows}

    def test_a_collapsed_multi_level_header_is_detected(self):
        t = self._t([["Language", "BLEU", "LeBLEU"],
                     ["Bengali", "6.72 8.83 9.19", "37.10 41.50 42.01"],
                     ["Tamil", "4.86 6.04 6.00", "29.38 30.77 31.33"]])
        self.assertTrue(tablesum.looks_mangled(t))

    def test_a_clean_table_is_not_flagged(self):
        t = self._t([["Model", "BLEU", "chrF"],
                     ["Baseline", "24.1", "51.3"],
                     ["Ours", "27.8", "55.0"]])
        self.assertFalse(tablesum.looks_mangled(t))

    def test_one_stray_multi_number_cell_is_not_enough(self):
        """年份区间、`95% CI 1.2 3.4` 这种偶发格子不该把整张表判死。"""
        t = self._t([["Method", "Score", "Years"],
                     ["A", "24.1", "2019 2020"],
                     ["B", "27.8", "2021"]])
        self.assertFalse(tablesum.looks_mangled(t))

    def test_the_two_prompts_differ_on_conclusions(self):
        clean, mangled = tablesum._SYSTEM_CLEAN, tablesum._SYSTEM_MANGLED
        self.assertIn("最主要结论", clean)
        self.assertIn("严禁给结论", mangled)
        self.assertNotIn("最主要结论", mangled)

    def test_the_mangled_prompt_says_why(self):
        """只说「不许」，模型会绕着写。得让它知道原因是列错位了。"""
        self.assertIn("列结构", tablesum._SYSTEM_MANGLED)

    def test_summarize_picks_the_prompt_by_the_flag(self):
        seen = []

        def fake(system, user, purpose, **kw):
            seen.append(system)
            return "说明"

        items = [{"paper_id": 1, "page_no": 1, "n_rows": 3, "n_cols": 3,
                  "n_chars": 100, "markdown": "x" * 100, "table_hash": "h1",
                  "mangled": True},
                 {"paper_id": 1, "page_no": 2, "n_rows": 3, "n_cols": 3,
                  "n_chars": 100, "markdown": "y" * 100, "table_hash": "h2",
                  "mangled": False}]
        with mock.patch.object(tablesum.llm, "chat", fake), \
             mock.patch.object(tablesum.llm, "available", return_value=True), \
             mock.patch.object(tablesum.db, "init_db"), \
             mock.patch.object(tablesum.db, "conn"):
            tablesum.summarize(items, dry_run=False)
        self.assertEqual(seen, [tablesum._SYSTEM_MANGLED, tablesum._SYSTEM_CLEAN])

    def test_the_db_records_which_summaries_dare_conclude(self):
        """事后要能把不可信的那批捞出来重做——首轮就靠这个捞出了 17 条。"""
        import shutil
        import tempfile
        from pathlib import Path

        from papernest import config, db
        tmp = tempfile.mkdtemp(prefix="papernest_sok_")
        self.addCleanup(shutil.rmtree, tmp, True)
        old = (config.DB_PATH, config.DATA_DIR)
        self.addCleanup(lambda: setattr(config, "DB_PATH", old[0]))
        self.addCleanup(lambda: setattr(config, "DATA_DIR", old[1]))
        config.DATA_DIR = Path(tmp) / "data"
        config.DB_PATH = config.DATA_DIR / "t.db"
        db.init_db()
        with db.conn() as c:
            cols = {r[1] for r in c.execute("PRAGMA table_info(table_summaries)")}
        self.assertIn("structure_ok", cols)
