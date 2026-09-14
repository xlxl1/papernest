"""自定义列文献对比矩阵（papernest/matrix.py）。

全程离线且确定性：DB 指到临时目录，embeddings / llm 一律摁死，
在线路径用假模型注入——一个 token 都不真花，跨进程重跑结果一致。
"""
import shutil
import csv
import io
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from papernest import config, db, matrix

# ── 夹具文本：卡片字段与原文的「对得上 / 对不上」都是故意设计的 ──

P1_ABSTRACT = ("我们提出 RetroFormer，一种检索增强的 Transformer。"
               "在 WikiText-103 上困惑度从 18.4 降到 15.2。")
P1_PAGES = {
    1: "1 Introduction. We propose RetroFormer, a retrieval-augmented Transformer.",
    3: ("5 Experiments. We evaluate on the WikiText-103 dataset and report perplexity. "
        "Accuracy and F1 are also reported."),
    4: ("6 Limitations. 仅在英文语料上验证，未测试低资源语言。"
        "Future work includes multilingual evaluation."),
}
P1_CARD = {
    "tldr": "检索增强的 Transformer。",
    "method": "提出 RetroFormer，一种检索增强的 Transformer。",   # 摘要里逐字有 → 可回取
    "results": "在 WikiText-103 上困惑度从 18.4 降到 15.2。",      # 摘要里逐字有 → 可回取
    "limitations": "仅在英文语料上验证，未测试低资源语言。",        # 第 4 页里逐字有 → 可回取
    "relation_to_topic": "可直接用于本课题的检索增强模块。",        # 原文没有 → 未核验
    "keywords": ["retrieval", "transformer"],
}


class MatrixTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="papernest_matrix_")
        self.addCleanup(shutil.rmtree, self._tmp, True)
        self._old_db, self._old_dir = config.DB_PATH, config.DATA_DIR
        config.DB_PATH = Path(self._tmp) / "test.db"
        config.DATA_DIR = Path(self._tmp) / "data"
        # 开发机 .env 真配了 EMBED_MODEL / LLM key，不摁住就会真发请求
        self._patches = [
            mock.patch("papernest.embeddings.available", return_value=False),
            mock.patch("papernest.llm.available", return_value=False),
        ]
        for p in self._patches:
            p.start()
        db.init_db()
        matrix.ensure_schema(force=True)
        self._seed()

    def tearDown(self):
        for p in reversed(self._patches):
            p.stop()
        config.DB_PATH, config.DATA_DIR = self._old_db, self._old_dir

    # ── 夹具 ──

    def _add(self, key, title, abstract, card=None, pages=None, year=2024):
        with db.conn() as c:
            pid = db.insert_l0(c, {
                "norm_key": key, "title": title, "abstract": abstract, "year": year,
                "venue": "TestVenue", "authors": ["Alice Smith"], "doi": None,
                "arxiv_id": None, "source": "test"})
            if card is not None:
                db.save_card(c, pid, card, "test-fixture")
            for no, text in (pages or {}).items():
                c.execute("INSERT INTO pages(paper_id,page_no,text) VALUES(?,?,?)",
                          (pid, no, text))
        return pid

    def _seed(self):
        self.p1 = self._add("arxiv:9001", "RetroFormer", P1_ABSTRACT,
                            card=P1_CARD, pages=P1_PAGES)
        # 卡片里塞占位串「（摘要未提及）」：这类内容计进覆盖率就是自欺
        self.p2 = self._add("arxiv:9002", "Baseline Study",
                            "A baseline study on English corpora.",
                            card={"tldr": "基线研究。", "method": "（摘要未提及）"})
        # 完全没有卡片、也没有全文：只能靠摘要兜底
        self.p3 = self._add("arxiv:9003", "SQuAD Eval",
                            "We evaluate on the SQuAD benchmark.")
        # 覆盖率手算专用的两篇
        self.pc = self._add("arxiv:9004", "DualNet", "本文提出 DualNet 双塔结构。",
                            card={"method": "本文提出 DualNet 双塔结构。"})
        self.pd = self._add("arxiv:9005", "CatNet",
                            "Totally different English text about cats.",
                            card={"method": "一段与原文完全无关的改写。"})


# ── 列定义 ──

class ColumnTests(MatrixTestBase):
    def test_default_columns_shape(self):
        keys = [c["key"] for c in matrix.DEFAULT_COLUMNS]
        self.assertEqual(keys, ["method", "dataset", "metric", "results",
                                "limitations", "relation"])
        for col in matrix.DEFAULT_COLUMNS:
            self.assertTrue(col["label"] and col["hint"])

    def test_normalize_accepts_bare_strings(self):
        cols = matrix.normalize_columns(["method", "novelty"])
        self.assertEqual([c["key"] for c in cols], ["method", "novelty"])
        self.assertEqual(cols[1]["label"], "novelty")  # 没给 label 就用 key

    def test_normalize_rejects_duplicate_and_empty(self):
        with self.assertRaises(ValueError):
            matrix.normalize_columns(["method", "method"])
        with self.assertRaises(ValueError):
            matrix.normalize_columns([])
        with self.assertRaises(ValueError):
            matrix.normalize_columns([{"label": "无 key"}])
        with self.assertRaises(ValueError):
            matrix.normalize_columns("method")

    def test_relation_column_does_not_keyword_search(self):
        """「与课题关系」是判断不是检索——显式配空关键词，不许拿列名去全文乱撞。"""
        col = matrix.normalize_columns(["relation"])[0]
        self.assertEqual(matrix._keywords_for(col), [])


# ── 离线抽取 ──

class OfflineTests(MatrixTestBase):
    def _m(self, ids=None, columns=None):
        return matrix.extract_offline(ids or [self.p1], columns)

    def test_card_fields_are_taken_directly(self):
        cells = self._m()["rows"][0]["cells"]
        self.assertEqual(cells["method"]["value"], P1_CARD["method"])
        self.assertEqual(cells["method"]["source"], "card:method")
        self.assertEqual(cells["results"]["value"], P1_CARD["results"])
        self.assertEqual(cells["results"]["source"], "card:results")
        self.assertEqual(cells["limitations"]["source"], "card:limitations")
        self.assertEqual(cells["relation"]["source"], "card:relation_to_topic")

    def test_card_cell_verified_only_when_text_really_backtracks(self):
        cells = self._m()["rows"][0]["cells"]
        self.assertTrue(cells["method"]["verified"])       # 摘要里逐字有
        self.assertTrue(cells["limitations"]["verified"])  # 第 4 页里逐字有
        self.assertFalse(cells["relation"]["verified"])    # 原文没有，如实标未核验
        self.assertIn("未通过回取校验", cells["relation"]["note"])
        self.assertIsNone(cells["relation"]["quote"])

    def test_page_fallback_carries_page_number(self):
        """卡片没有 dataset / metric 字段 → 全文关键词定位原句，且必须带页码。"""
        cells = self._m()["rows"][0]["cells"]
        self.assertEqual(cells["dataset"]["source"], "page:3")
        self.assertEqual(cells["dataset"]["page"], 3)
        self.assertIn("WikiText-103 dataset", cells["dataset"]["value"])
        # 逐字摘录路径不报 verified：这句就是从该页切出来的，回校验同一页恒真，
        # 报 True 是空转断言（真库穷举 17492 个候选，零反例）。
        self.assertIsNone(cells["dataset"]["verified"])
        self.assertEqual(cells["metric"]["page"], 3)
        self.assertIn("Accuracy", cells["metric"]["value"])
        # 取到的必须是原文逐字片段
        self.assertIn(cells["dataset"]["value"], P1_PAGES[3])

    def test_abstract_fallback_when_no_pages(self):
        cells = self._m([self.p3])["rows"][0]["cells"]
        self.assertEqual(cells["dataset"]["source"], "abstract")
        self.assertIsNone(cells["dataset"]["page"])
        self.assertEqual(cells["dataset"]["value"], "We evaluate on the SQuAD benchmark.")

    def test_missing_field_is_empty_with_honest_note(self):
        cells = self._m([self.p3])["rows"][0]["cells"]
        for key in ("method", "results", "limitations", "relation"):
            with self.subTest(col=key):
                self.assertIsNone(cells[key]["value"])
                self.assertIsNone(cells[key]["source"])
                self.assertFalse(cells[key]["verified"])
                self.assertEqual(cells[key]["note"], matrix.EMPTY_NOTE)

    def test_placeholder_card_value_is_not_counted_as_covered(self):
        """mock 卡片写死的「（摘要未提及）」不算覆盖——否则覆盖率是自欺。"""
        cells = self._m([self.p2])["rows"][0]["cells"]
        self.assertIsNone(cells["method"]["value"])
        self.assertEqual(cells["method"]["note"], matrix.EMPTY_NOTE)

    def test_row_order_follows_caller_order(self):
        m = self._m([self.p3, self.p1, self.p2])
        self.assertEqual([r["paper_id"] for r in m["rows"]],
                         [self.p3, self.p1, self.p2])

    def test_offline_is_deterministic_across_runs(self):
        a = matrix.extract_offline([self.p1, self.p2, self.p3])
        b = matrix.extract_offline([self.p1, self.p2, self.p3])
        self.assertEqual(json.dumps(a, ensure_ascii=False, sort_keys=True),
                         json.dumps(b, ensure_ascii=False, sort_keys=True))

    def test_custom_column_with_explicit_keywords(self):
        cols = [{"key": "future", "label": "未来工作",
                 "hint": "作者说的下一步", "keywords": ["Future work"]}]
        cells = self._m([self.p1], cols)["rows"][0]["cells"]
        self.assertEqual(cells["future"]["page"], 4)
        self.assertIn("Future work", cells["future"]["value"])

    def test_custom_column_hits_same_named_card_field(self):
        pid = self._add("arxiv:9101", "Novel", "abs", card={"novelty": "首次把 A 与 B 合并。"})
        cells = matrix.extract_offline([pid], ["novelty"])["rows"][0]["cells"]
        self.assertEqual(cells["novelty"]["value"], "首次把 A 与 B 合并。")
        self.assertEqual(cells["novelty"]["source"], "card:novelty")

    def test_offline_marks_itself_degraded(self):
        self.assertIn("离线", self._m()["degraded"])
        self.assertEqual(self._m()["mode"], "offline")


# ── coverage / verified_rate ──

class MetricsTests(MatrixTestBase):
    def test_hand_computable_coverage_and_verified_rate(self):
        """2 篇 × 2 列 = 4 格；填 2 格（两篇的 method），其中 1 格能回取。"""
        m = matrix.extract_offline([self.pc, self.pd], ["method", "limitations"])
        self.assertEqual(m["cells_total"], 4)
        self.assertEqual(m["cells_filled"], 2)
        self.assertEqual(m["cells_verified"], 1)
        self.assertEqual(m["coverage"], 0.5)        # 2/4
        self.assertEqual(m["verified_rate"], 0.5)   # 1/2（分母是非空格）

    def test_full_paper_metrics(self):
        m = matrix.extract_offline([self.p1])
        # 6 格全满，但只有 4 格**需要**回取校验（走卡片路径，3 过 1 不过）；
        # 另 2 格是从全文里逐字摘录的，校验对它们不适用，不进分子也不进分母。
        self.assertEqual((m["cells_total"], m["cells_filled"], m["cells_verified"]),
                         (6, 6, 3))
        self.assertEqual(m["cells_checkable"], 4)
        self.assertEqual(m["cells_excerpted"], 2)   # dataset / metric 走 _locate
        self.assertEqual(m["coverage"], 1.0)
        self.assertEqual(m["verified_rate"], round(3 / 4, 4))

    def test_empty_table_metrics_do_not_divide_by_zero(self):
        m = matrix.extract_offline([424242])
        self.assertEqual(m["rows"], [])
        self.assertEqual(m["coverage"], 0.0)
        # 没有可校验的格子 → None（显示「—」），不是 0.0：
        # 报 0% 会被读成「全都没通过校验」，比报恒真的 100% 更糟。
        self.assertIsNone(m["verified_rate"])


# ── 输入校验 ──

class InputTests(MatrixTestBase):
    def test_empty_paper_ids_raises(self):
        for bad in ([], None, "3"):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                matrix.build(bad)

    def test_non_integer_paper_id_raises(self):
        with self.assertRaises(ValueError):
            matrix.build(["abc"])

    def test_missing_paper_recorded_in_errors_and_others_survive(self):
        m = matrix.build([self.p1, 999999], use_llm=False)
        self.assertEqual([r["paper_id"] for r in m["rows"]], [self.p1])
        self.assertEqual(len(m["errors"]), 1)
        self.assertEqual(m["errors"][0]["paper_id"], 999999)
        self.assertIn("不存在", m["errors"][0]["error"])

    def test_duplicate_ids_collapse(self):
        m = matrix.build([self.p1, self.p1], use_llm=False)
        self.assertEqual(len(m["rows"]), 1)


# ── LLM 增强路径（假模型，不真调）──

class FakeLLM:
    """按 paper_id 返回预置 JSON 的假模型；记录每次调用，供「每篇一次」断言。"""

    def __init__(self, payloads=None, fail_for=()):
        self.calls: list[dict] = []
        self._payloads = payloads or {}
        self._fail = set(fail_for)

    def chat(self, system, user, purpose="", paper_id=None,
             temperature=0.3, model=None):
        self.calls.append({"paper_id": paper_id, "purpose": purpose, "user": user})
        if paper_id in self._fail:
            raise RuntimeError("模拟模型故障")
        return json.dumps(self._payloads.get(paper_id, {"cells": {}}),
                          ensure_ascii=False)

    def ids(self):
        return [c["paper_id"] for c in self.calls]


def online(fake: FakeLLM):
    return mock.patch.multiple("papernest.llm",
                               available=mock.Mock(return_value=True),
                               chat=mock.Mock(side_effect=fake.chat))


REAL_QUOTE = "We propose RetroFormer, a retrieval-augmented Transformer."
FAKE_QUOTE = "这句话在原文里根本不存在，纯属编造的依据句。"


class LLMTests(MatrixTestBase):
    def test_one_call_per_paper_not_per_cell(self):
        """6 列 × 2 篇：必须是 2 次调用，不是 12 次——每 cell 一调会把 token 烧穿。"""
        fake = FakeLLM()
        with online(fake):
            m = matrix.build([self.p1, self.p2])
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(fake.ids(), [self.p1, self.p2])
        self.assertEqual(m["mode"], "llm")
        self.assertEqual(len(m["columns"]), 6)

    def test_prompt_carries_every_column_key(self):
        fake = FakeLLM()
        with online(fake):
            matrix.build([self.p1], ["method", "dataset"])
        user = fake.calls[0]["user"]
        self.assertIn("method", user)
        self.assertIn("dataset", user)
        self.assertIn("第 3 页", user)  # 分页全文进了上下文

    def test_real_quote_is_verified(self):
        fake = FakeLLM({self.p1: {"cells": {
            "method": {"value": "检索增强 Transformer", "quote": REAL_QUOTE, "page": 1}}}})
        with online(fake):
            cell = matrix.build([self.p1], ["method"])["rows"][0]["cells"]["method"]
        self.assertEqual(cell["value"], "检索增强 Transformer")
        self.assertEqual(cell["source"], "llm")
        self.assertEqual(cell["page"], 1)
        self.assertTrue(cell["verified"])
        self.assertIsNone(cell["note"])

    def test_fabricated_quote_is_kept_but_marked_unverified(self):
        fake = FakeLLM({self.p1: {"cells": {
            "method": {"value": "编造的方法描述", "quote": FAKE_QUOTE, "page": 1}}}})
        with online(fake):
            m = matrix.build([self.p1], ["method"])
        cell = m["rows"][0]["cells"]["method"]
        self.assertFalse(cell["verified"])
        self.assertEqual(cell["value"], "编造的方法描述")   # 不静默丢弃
        self.assertEqual(cell["quote"], FAKE_QUOTE)         # 依据句原样留证
        self.assertIn("回取", cell["note"])
        self.assertEqual(m["verified_rate"], 0.0)
        self.assertEqual(m["coverage"], 1.0)

    def test_missing_quote_is_unverified(self):
        fake = FakeLLM({self.p1: {"cells": {
            "method": {"value": "有值但没给依据句"}}}})
        with online(fake):
            cell = matrix.build([self.p1], ["method"])["rows"][0]["cells"]["method"]
        self.assertFalse(cell["verified"])
        self.assertIn("未给出原文依据句", cell["note"])

    def test_null_value_stays_empty(self):
        fake = FakeLLM({self.p1: {"cells": {
            "method": {"value": None, "quote": None, "page": None}}}})
        with online(fake):
            cell = matrix.build([self.p1], ["method"])["rows"][0]["cells"]["method"]
        self.assertIsNone(cell["value"])
        self.assertIn("未覆盖", cell["note"])

    def test_column_absent_from_model_output_is_empty(self):
        fake = FakeLLM({self.p1: {"cells": {}}})
        with online(fake):
            cell = matrix.build([self.p1], ["method"])["rows"][0]["cells"]["method"]
        self.assertIsNone(cell["value"])
        self.assertIn("模型未给出该列", cell["note"])

    def test_one_paper_failure_does_not_break_the_others(self):
        fake = FakeLLM(payloads={self.p1: {"cells": {
            "method": {"value": "检索增强", "quote": REAL_QUOTE, "page": 1}}}},
            fail_for=[self.p2])
        with online(fake):
            m = matrix.build([self.p1, self.p2], ["method"])
        self.assertEqual(len(m["rows"]), 2)
        self.assertEqual(m["rows"][0]["cells"]["method"]["value"], "检索增强")
        self.assertNotIn("fallback", m["rows"][0])
        self.assertEqual(m["rows"][1]["fallback"], "offline")   # 如实标降级
        self.assertEqual(len(m["errors"]), 1)
        self.assertEqual(m["errors"][0]["paper_id"], self.p2)
        self.assertIn("模拟模型故障", m["errors"][0]["error"])
        self.assertIn("1/2", m["degraded"])

    def test_unparseable_json_falls_back_offline(self):
        with online(FakeLLM()) as _:
            with mock.patch("papernest.llm.chat", return_value="这不是 JSON"):
                m = matrix.build([self.p1], ["method"])
        self.assertEqual(m["rows"][0]["fallback"], "offline")
        self.assertEqual(m["rows"][0]["cells"]["method"]["source"], "card:method")
        self.assertEqual(len(m["errors"]), 1)

    def test_build_auto_picks_llm_when_available(self):
        fake = FakeLLM()
        with online(fake):
            self.assertEqual(matrix.build([self.p1])["mode"], "llm")
        self.assertEqual(len(fake.calls), 1)

    def test_build_auto_picks_offline_when_not_available(self):
        m = matrix.build([self.p1])
        self.assertEqual(m["mode"], "offline")

    def test_explicit_use_llm_without_config_raises(self):
        from papernest import llm
        with self.assertRaises(llm.LLMUnavailable):
            matrix.build([self.p1], use_llm=True)
        with self.assertRaises(llm.LLMUnavailable):
            matrix.extract_llm([self.p1])

    def test_truncated_context_is_reported_not_swallowed(self):
        """超长全文被截进上下文时，空格就不一定是「原文未覆盖」——必须说出来。"""
        pid = self._add("arxiv:9200", "Long Paper", "abs",
                        pages={1: "x" * (matrix._PAGE_CHARS + 10)})
        fake = FakeLLM()
        with online(fake):
            m = matrix.build([pid], ["method"])
        self.assertTrue(m["rows"][0]["context_truncated"])
        self.assertIn("截断", m["degraded"])

    def test_short_context_is_not_flagged(self):
        fake = FakeLLM()
        with online(fake):
            m = matrix.build([self.p1], ["method"])
        self.assertNotIn("context_truncated", m["rows"][0])
        self.assertIsNone(m["degraded"])

    def test_progress_callback_is_called(self):
        seen = []
        with online(FakeLLM()):
            matrix.build([self.p1, self.p2],
                         progress=lambda f, s, msg: seen.append((f, s)))
        self.assertEqual(len(seen), 2)
        self.assertTrue(all(0.0 <= f <= 1.0 for f, _ in seen))


# ── 导出 ──

NASTY_TITLE = "A|B study 50% off_x"
NASTY_METHOD = "pipe | and\nnewline"
NASTY_NOTES = "specials &%$#_~^ end"


def _md_cells(line: str) -> list[str]:
    """按未转义的 | 切一行 Markdown 表格（\\| 属于单元格内容）。"""
    return re.split(r"(?<!\\)\|", line)


def nasty_matrix() -> dict:
    cols = matrix.normalize_columns([
        {"key": "method", "label": "方法"},
        {"key": "notes", "label": "备注 & 说明"},
    ])
    rows = [{"paper_id": 7, "title": NASTY_TITLE, "year": 2024, "venue": None,
             "cells": {
                 "method": matrix._cell(value=NASTY_METHOD, source="page:2", page=2,
                                        quote=NASTY_METHOD, verified=True),
                 "notes": matrix._cell(value=NASTY_NOTES, source="llm", verified=False,
                                       note="未核验"),
             }}]
    return matrix._finish(cols, rows, [], mode="offline", degraded=None)


class ExportTests(MatrixTestBase):
    def setUp(self):
        super().setUp()
        self.m = nasty_matrix()
        self.real = matrix.extract_offline([self.p1, self.p2])

    # CSV

    def test_csv_has_utf8_bom(self):
        text = matrix.to_csv(self.m)
        self.assertTrue(text.startswith("﻿"))
        self.assertTrue(text.encode("utf-8").startswith(b"\xef\xbb\xbf"))

    def test_csv_reads_back_with_pipes_and_newlines(self):
        text = matrix.to_csv(self.m)
        rows = list(csv.reader(io.StringIO(text.lstrip("﻿"))))
        header, body = rows[0], rows[1]
        self.assertEqual(header, ["ID", "文献", "年份", "方法", "备注 & 说明"])
        self.assertEqual(len(body), 5)
        self.assertEqual(body[0], "7")
        self.assertEqual(body[1], NASTY_TITLE)
        self.assertEqual(body[3], NASTY_METHOD + "【p.2】")     # 换行原样保留在字段内
        self.assertEqual(body[4], NASTY_NOTES + "【未核验】")

    def test_csv_footer_states_coverage_and_empty_legend(self):
        text = matrix.to_csv(self.m)
        self.assertIn("覆盖率 100.0%", text)
        self.assertIn("回取校验通过率 50.0%", text)
        self.assertIn(matrix.EMPTY_LEGEND, text)

    def test_csv_without_marks_is_clean(self):
        rows = list(csv.reader(io.StringIO(
            matrix.to_csv(self.m, with_marks=False).lstrip("﻿"))))
        self.assertEqual(rows[1][3], NASTY_METHOD)

    # Markdown

    def test_markdown_table_keeps_column_count(self):
        md = matrix.to_markdown(self.m)
        table = [ln for ln in md.splitlines() if ln.startswith("|")]
        self.assertEqual(len(table), 3)                     # 表头 + 分隔 + 1 行
        # 只按「未转义的 |」切——转义后的 \| 是单元格内容，渲染器不会当分隔符
        widths = sorted({len(_md_cells(ln)) for ln in table})
        self.assertEqual(widths, [5 + 2])                   # 5 列 → 首尾各一个空片段

    def test_markdown_escapes_pipe_and_newline(self):
        md = matrix.to_markdown(self.m)
        self.assertIn(r"pipe \| and<br>newline", md)
        self.assertNotIn("pipe | and", md)
        self.assertIn(r"A\|B study 50% off_x", md)

    def test_markdown_footer(self):
        md = matrix.to_markdown(self.m)
        self.assertIn("> 覆盖率 100.0%", md)
        self.assertIn(matrix.EMPTY_LEGEND, md)
        self.assertIn(matrix.MARK_LEGEND, md)

    def test_markdown_empty_cell_is_blank(self):
        md = matrix.to_markdown(self.real)
        row = [ln for ln in md.splitlines() if ln.startswith("| ") and "Baseline" in ln][0]
        fields = [f.strip() for f in _md_cells(row)[1:-1]]
        self.assertEqual(fields[3:], [""] * 6)              # 空 = 原文未覆盖

    # LaTeX

    def test_latex_escapes_every_special_char(self):
        tex = matrix.to_latex(self.m)
        self.assertNotIn(NASTY_NOTES, tex)
        self.assertIn(r"specials \&\%\$\#\_\textasciitilde{}\textasciicircum{} end", tex)
        self.assertIn(r"备注 \& 说明", tex)
        # 除对齐用的 & 外，所有特殊字符都必须紧跟在反斜杠后面
        for mm in re.finditer(r"[%$#_^~]", tex):
            with self.subTest(pos=mm.start(), ch=mm.group()):
                self.assertEqual(tex[mm.start() - 1], "\\")

    def test_latex_ampersands_are_only_alignment(self):
        tex = matrix.to_latex(self.m)
        n_cols = 3 + len(self.m["columns"])
        n_lines = 1 + len(self.m["rows"])                   # 表头 + 数据行
        self.assertEqual(len(re.findall(r"(?<!\\)&", tex)), (n_cols - 1) * n_lines)

    def test_latex_uses_booktabs_and_states_metrics(self):
        tex = matrix.to_latex(self.m)
        for token in (r"\toprule", r"\midrule", r"\bottomrule",
                      r"\begin{tabular}", r"\end{table}"):
            self.assertIn(token, tex)
        self.assertIn(r"覆盖率 100.0\%", tex)
        self.assertIn(matrix.EMPTY_LEGEND, tex)

    def test_latex_row_ends_are_intact(self):
        tex = matrix.to_latex(self.m)
        # 单元格里的换行被压成空格，不会把 tabular 的行结束符冲掉
        self.assertEqual(tex.count(r"\\"), 1 + len(self.m["rows"]))

    # 统一入口 / 空表

    def test_export_dispatch_and_unknown_format(self):
        for fmt in ("csv", "markdown", "latex"):
            self.assertTrue(matrix.export(self.m, fmt))
        with self.assertRaises(ValueError):
            matrix.export(self.m, "xlsx")

    def test_exports_survive_empty_table(self):
        empty = matrix.extract_offline([424242])
        for fmt in ("csv", "markdown", "latex"):
            with self.subTest(fmt=fmt):
                self.assertIn("覆盖率 0.0", matrix.export(empty, fmt))


# ── 列预设 ──

class PresetTests(MatrixTestBase):
    def test_save_get_list_delete(self):
        cols = [{"key": "method", "label": "方法"}, {"key": "cost", "label": "算力开销"}]
        matrix.save_preset("我的对比列", cols)
        got = matrix.get_preset("我的对比列")
        self.assertEqual([c["key"] for c in got], ["method", "cost"])
        self.assertEqual([p["name"] for p in matrix.list_presets()], ["我的对比列"])
        matrix.save_preset("我的对比列", ["method"])         # 同名覆盖
        self.assertEqual(len(matrix.get_preset("我的对比列")), 1)
        self.assertTrue(matrix.delete_preset("我的对比列"))
        self.assertIsNone(matrix.get_preset("我的对比列"))
        self.assertFalse(matrix.delete_preset("我的对比列"))

    def test_preset_name_required(self):
        with self.assertRaises(ValueError):
            matrix.save_preset("  ", ["method"])

    def test_preset_feeds_build(self):
        matrix.save_preset("三列", ["method", "dataset", "limitations"])
        m = matrix.build([self.p1], matrix.get_preset("三列"), use_llm=False)
        self.assertEqual([c["key"] for c in m["columns"]],
                         ["method", "dataset", "limitations"])


# ── 回取校验本身 ──

class VerifyTests(unittest.TestCase):
    def test_page_scoped_verify_matches_fulltext_caliber(self):
        """口径对齐 fulltext._verify_claim：校验的是「在它声称的那一页」。"""
        ctx = {"pages": [(1, matrix._norm("We propose RetroFormer, a Transformer.")),
                         (2, matrix._norm("Unrelated page two."))],
               "by_page": {}, "abstract": "", "hay": ""}
        self.assertEqual(matrix._resolve_page("We propose RetroFormer", ctx, 1),
                         (1, "claimed"))
        self.assertEqual(matrix._resolve_page("We propose RetroFormer", ctx, 9),
                         (1, "relocated"))     # 声称页错了 → 更正为真实页
        self.assertEqual(matrix._resolve_page(FAKE_QUOTE, ctx, 1), (None, "none"))

    def test_whitespace_is_ignored(self):
        hay = matrix._norm("We propose\n  RetroFormer, a retrieval-augmented Transformer.")
        self.assertTrue(matrix.verify_quote("We propose RetroFormer,", hay))

    def test_unrelated_text_fails(self):
        hay = matrix._norm("Totally different English text about cats and dogs.")
        self.assertFalse(matrix.verify_quote(FAKE_QUOTE, hay))
        self.assertFalse(matrix.verify_quote("", hay))

    def test_cross_chunk_concatenation_does_not_create_false_hits(self):
        """摘要末尾 + 下一页开头拼起来的串，不能被判成「原文里有」。"""
        hay = "\n".join([matrix._norm("...ends with alpha"),
                         matrix._norm("beta begins here...")])
        self.assertFalse(matrix.verify_quote("endswithalphabetabegins", hay))


# ── 复核补测：页码归属 / 真实卡片形态 / 词面巧合 / 导出注入 ──
#
# 前一轮夹具太理想：卡片文本被写成与摘要逐字一致、模型给的 page 恰好也对，
# 于是「页码从没被校验过」这件事在 57 个用例里一次都没暴露出来。
# 下面这些用例专挑「夹具刚好对上」的地方拆。


class PageAttributionTests(MatrixTestBase):
    """页码是出处的一半。模型/卡片自称的页码必须逐页验，验不上就别写。"""

    def test_model_page_is_corrected_not_trusted(self):
        """依据句真在第 1 页、模型标 p.9 —— 不能给个绿标就把 p.9 印进表里。"""
        fake = FakeLLM({self.p1: {"cells": {"method": {
            "value": "检索增强 Transformer", "quote": REAL_QUOTE, "page": 9}}}})
        with online(fake):
            cell = matrix.build([self.p1], ["method"])["rows"][0]["cells"]["method"]
        self.assertTrue(cell["verified"])
        self.assertEqual(cell["page"], 1)                    # 按原文更正
        self.assertIn("p.9", cell["note"])
        self.assertIn("更正", cell["note"])
        self.assertEqual(matrix._cell_text(cell), "检索增强 Transformer【p.1】")

    def test_unverifiable_quote_drops_the_claimed_page(self):
        """验不过的依据句 + 它自称的页码，凑不出出处——页码必须丢掉。"""
        fake = FakeLLM({self.p1: {"cells": {"method": {
            "value": "编造的方法", "quote": FAKE_QUOTE, "page": 3}}}})
        with online(fake):
            cell = matrix.build([self.p1], ["method"])["rows"][0]["cells"]["method"]
        self.assertFalse(cell["verified"])
        self.assertIsNone(cell["page"])
        self.assertEqual(cell["quote"], FAKE_QUOTE)          # 证据仍留着
        self.assertEqual(matrix._cell_text(cell), "编造的方法【未核验】")

    def test_missing_quote_drops_the_claimed_page(self):
        fake = FakeLLM({self.p1: {"cells": {
            "method": {"value": "有值没依据", "page": 4}}}})
        with online(fake):
            cell = matrix.build([self.p1], ["method"])["rows"][0]["cells"]["method"]
        self.assertIsNone(cell["page"])
        self.assertIn("p.4", cell["note"])

    def test_abstract_quote_does_not_borrow_a_page_number(self):
        """依据句只在摘要里 → 页码如实为 None，不采信模型标的页。"""
        quote = "在 WikiText-103 上困惑度从 18.4 降到 15.2。"
        fake = FakeLLM({self.p1: {"cells": {
            "results": {"value": "困惑度 18.4→15.2", "quote": quote, "page": 2}}}})
        with online(fake):
            cell = matrix.build([self.p1], ["results"])["rows"][0]["cells"]["results"]
        self.assertTrue(cell["verified"])
        self.assertIsNone(cell["page"])
        self.assertIn("摘要", cell["note"])

    def test_probes_may_not_be_stitched_across_pages(self):
        """探针散落在不同页、加起来凑够 2/3 —— 全文合集会放行，逐页校验不放行。"""
        page1, page2 = "a" * 12 + "b" * 12, "c" * 12 + "d" * 12
        quote = page1 + page2                                # 4 个探针，每页各占 2 个
        pid = self._add("arxiv:9300", "Stitch", "", pages={1: page1, 2: page2})
        hay = "\n".join([matrix._norm(page1), matrix._norm(page2)])
        self.assertTrue(matrix.verify_quote(quote, hay))     # 旧口径：会误判为「原文里有」
        fake = FakeLLM({pid: {"cells": {
            "method": {"value": "拼接出来的", "quote": quote, "page": 1}}}})
        with online(fake):
            cell = matrix.build([pid], ["method"])["rows"][0]["cells"]["method"]
        self.assertFalse(cell["verified"])                   # 新口径：逐页都不够 2/3
        self.assertIsNone(cell["page"])


class RealCardShapeTests(MatrixTestBase):
    """按库里**真实**的卡片形态建夹具，而不是按「刚好能过」的形态。"""

    # 真实 L2 卡片（fulltext.read_paper 落库的形状）：key_findings 是带页码的对象数组
    L2_CARD = {
        "tldr": "L2 精读卡片。",
        "key_findings": [
            {"claim": "Accuracy and F1 are also reported.", "page": 3},
            {"claim": "Future work includes multilingual evaluation.", "page": 4},
        ],
        "method_detail": "We propose RetroFormer, a retrieval-augmented Transformer.",
        "limitations": "仅在英文语料上验证，未测试低资源语言。",
        "relation_to_topic": "可用于检索增强模块。",
    }

    def test_l2_key_findings_list_is_joined_without_a_fake_page(self):
        """两条结论分别在 p.3 / p.4，拼成一格后不能拿第一条的 p.3 当整格出处。"""
        pid = self._add("arxiv:9401", "L2", P1_ABSTRACT,
                        card=self.L2_CARD, pages=P1_PAGES)
        cells = matrix.extract_offline([pid], ["results", "method"])["rows"][0]["cells"]
        self.assertEqual(cells["results"]["source"], "card:key_findings")
        self.assertIn("Accuracy and F1", cells["results"]["value"])
        self.assertIn("Future work", cells["results"]["value"])
        self.assertIsNone(cells["results"]["page"])           # 页码不一致 → 不给页码
        # method_detail 是第 1 页的原句逐字 → 页码能自己定位出来
        self.assertEqual(cells["method"]["source"], "card:method_detail")
        self.assertEqual(cells["method"]["page"], 1)
        self.assertTrue(cells["method"]["verified"])

    def test_single_finding_keeps_and_corrects_its_page(self):
        card = {"key_findings": [
            {"claim": "Accuracy and F1 are also reported.", "page": 8}]}   # 真实在 p.3
        pid = self._add("arxiv:9402", "L2b", P1_ABSTRACT, card=card, pages=P1_PAGES)
        cell = matrix.extract_offline([pid], ["results"])["rows"][0]["cells"]["results"]
        self.assertEqual(cell["page"], 3)
        self.assertIn("更正", cell["note"])

    def test_real_l1_card_is_a_paraphrase_and_is_marked_unverified(self):
        """库里真实的 L1 卡片是「英文摘要 → 中文改写」，回取必然失败。

        这不是 bug，是本模块的诚实之处；钉住它，防止有人为了让
        verified_rate 好看而把校验口径放松成「差不多就算过」。
        """
        pid = self._add("arxiv:9403", "Real L1",
                        "We present a retrieval-augmented Transformer for long-context "
                        "language modeling, and evaluate it on WikiText-103.",
                        card={"method": "提出了一种面向长上下文语言建模的检索增强 Transformer。",
                              "limitations": "（摘要未提及）"})
        m = matrix.extract_offline([pid], ["method", "limitations"])
        cell = m["rows"][0]["cells"]["method"]
        self.assertEqual(cell["value"],
                         "提出了一种面向长上下文语言建模的检索增强 Transformer。")
        self.assertFalse(cell["verified"])
        self.assertIsNone(cell["page"])
        self.assertIsNone(cell["quote"])
        self.assertIn("非原文逐字", cell["note"])
        # 「（摘要未提及）」是库里最常见的占位串，不许计进覆盖率
        self.assertIsNone(m["rows"][0]["cells"]["limitations"]["value"])
        self.assertEqual(m["coverage"], 0.5)
        self.assertEqual(m["verified_rate"], 0.0)

    def test_malformed_card_json_does_not_crash(self):
        pid = self._add("arxiv:9404", "Bad", "We evaluate on the SQuAD benchmark.")
        for bad in ("{not json", "[1,2,3]", "null", '"a string"'):
            with self.subTest(card_json=bad):
                with db.conn() as c:
                    c.execute("UPDATE papers SET card_json=? WHERE id=?", (bad, pid))
                cells = matrix.extract_offline([pid], ["method", "dataset"])["rows"][0]["cells"]
                self.assertIsNone(cells["method"]["value"])
                self.assertEqual(cells["dataset"]["source"], "abstract")

    def test_stringify_handles_scalar_dict_and_list_shapes(self):
        self.assertEqual(matrix._stringify(42), ("42", None))
        self.assertEqual(matrix._stringify(True), (None, None))
        self.assertEqual(matrix._stringify({"claim": "abc", "page": "5"}), ("abc", 5))
        self.assertEqual(matrix._stringify({"page": 5}), (None, None))
        self.assertEqual(matrix._stringify([{"claim": "a", "page": 2},
                                            {"claim": "b", "page": 2}]), ("a；b", 2))
        self.assertEqual(matrix._stringify([{"claim": "a", "page": 2},
                                            {"claim": "b"}]), ("a；b", None))
        self.assertEqual(matrix._stringify(["（mock）", "真内容"]), ("真内容", None))


class LexicalMatchTests(MatrixTestBase):
    def test_column_name_keyword_hit_is_flagged_as_lexical_only(self):
        """自定义列没给关键词时拿列名去全文撞，撞到的是词面不是语义——必须说清楚。"""
        pid = self._add("arxiv:9500", "KW", "abs",
                        pages={1: "The cost function is convex. We use 3 GPU-hours."})
        cell = matrix.extract_offline(
            [pid], [{"key": "cost", "label": "算力开销"}])["rows"][0]["cells"]["cost"]
        self.assertEqual(cell["value"], "The cost function is convex.")
        self.assertEqual(cell["page"], 1)
        self.assertIn("词面", cell["note"])
        self.assertIn("非语义判定", cell["note"])

    def test_curated_and_explicit_keyword_columns_carry_no_such_note(self):
        cells = matrix.extract_offline(
            [self.p1], ["dataset", {"key": "future", "keywords": ["Future work"]}]
        )["rows"][0]["cells"]
        self.assertIsNone(cells["dataset"]["note"])          # COLUMN_KEYWORDS 精选过
        self.assertIsNone(cells["future"]["note"])           # 用户显式给的词

    def test_long_paragraph_is_clipped_around_the_keyword(self):
        """PDF 里整段没有句号是常态——截出来的必须仍是原文逐字且含关键词。"""
        page = "x" * 900 + " the WikiText-103 dataset was used " + "y" * 900
        pid = self._add("arxiv:9501", "Long", "abs", pages={2: page})
        cell = matrix.extract_offline([pid], ["dataset"])["rows"][0]["cells"]["dataset"]
        self.assertLessEqual(len(cell["value"]), matrix._MAX_SENT)
        self.assertIn("dataset", cell["value"].lower())
        self.assertIn(cell["value"], page)                   # 逐字子串，没被改写
        self.assertEqual(cell["page"], 2)
        self.assertIsNone(cell["verified"])          # 逐字摘录，不报回取结论


class RobustnessTests(MatrixTestBase):
    def test_model_output_without_cells_wrapper_is_still_used(self):
        """模型常直接返回 {"method": {...}}；能对上列 key 就采信，别整行白白降级。"""
        fake = FakeLLM({self.p1: {
            "method": {"value": "检索增强", "quote": REAL_QUOTE, "page": 1}}})
        with online(fake):
            m = matrix.build([self.p1], ["method"])
        self.assertNotIn("fallback", m["rows"][0])
        self.assertEqual(m["errors"], [])
        self.assertEqual(m["rows"][0]["cells"]["method"]["value"], "检索增强")

    def test_json_without_any_known_column_still_falls_back(self):
        fake = FakeLLM({self.p1: {"summary": "答非所问"}})
        with online(fake):
            m = matrix.build([self.p1], ["method"])
        self.assertEqual(m["rows"][0]["fallback"], "offline")
        self.assertIn("cells", m["errors"][0]["error"])

    def test_hostile_paper_ids_are_rejected(self):
        for bad in ([True], [3.9], [{1: 2}], {self.p1, self.p2}):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                matrix.build(bad, use_llm=False)

    def test_offline_progress_callback_is_called(self):
        seen = []
        matrix.extract_offline([self.p1, self.p2], ["method"],
                               progress=lambda f, s, msg: seen.append((f, s)))
        self.assertEqual([s for _, s in seen], ["offline", "offline"])
        self.assertTrue(all(0.0 <= f <= 1.0 for f, _ in seen))


class ExportHardeningTests(MatrixTestBase):
    def _one(self, title, value):
        cols = matrix.normalize_columns([{"key": "m", "label": "方法"}])
        rows = [{"paper_id": 1, "title": title, "year": 2024, "venue": None,
                 "cells": {"m": matrix._cell(value=value, source="llm", verified=True)}}]
        return matrix._finish(cols, rows, [], mode="offline", degraded=None)

    def test_csv_formula_injection_is_neutralised(self):
        """CSV 卖点是「Excel 双击打开」，= + - @ 开头的字段会被当公式执行。"""
        m = self._one("=cmd|'/c calc'!A1", "-1+1")
        row = list(csv.reader(io.StringIO(matrix.to_csv(m).lstrip("﻿"))))[1]
        self.assertTrue(row[1].startswith("'="))
        self.assertTrue(row[3].startswith("'-"))

    def test_ordinary_values_are_not_touched_by_the_csv_guard(self):
        m = self._one("A normal title", "一个普通的中文值")
        row = list(csv.reader(io.StringIO(matrix.to_csv(m).lstrip("﻿"))))[1]
        self.assertEqual(row[1], "A normal title")
        self.assertEqual(row[3], "一个普通的中文值")

    def test_mark_legend_is_omitted_when_marks_are_off(self):
        m = self._one("t", "v")
        for fmt in ("csv", "markdown", "latex"):
            with self.subTest(fmt=fmt):
                self.assertNotIn("【p.N】", matrix.export(m, fmt, with_marks=False))
                self.assertIn("【p.N】", matrix.export(m, fmt, with_marks=True))
                self.assertIn(matrix.EMPTY_LEGEND, matrix.export(m, fmt, with_marks=False))

    def test_latex_survives_a_literal_backslash_in_a_cell(self):
        m = self._one(r"a \alpha b", r"O(n^2) \& 100% done_x")
        tex = matrix.to_latex(m)
        self.assertIn(r"\textbackslash{}alpha", tex)
        self.assertEqual(tex.count("\\\\"), 1 + len(m["rows"]))   # 行结束符没被冲掉
        for mm in re.finditer(r"[%$#_^~]", tex):
            with self.subTest(pos=mm.start()):
                self.assertEqual(tex[mm.start() - 1], "\\")


class DefensiveBranchTests(MatrixTestBase):
    """把「提前 return」的防御分支逐条走一遍。

    本项目踩过：13 个用例全在提前 return 之前就折返了，漏掉一个必现的 NameError。
    这些分支平时不走，一走就是线上出事的时候。
    """

    def test_verify_quote_rejects_too_short_and_empty(self):
        hay = matrix._norm("We propose RetroFormer, a retrieval-augmented Transformer.")
        self.assertFalse(matrix.verify_quote("zzz", hay))      # 太短且不是子串
        self.assertFalse(matrix.verify_quote("abc", ""))
        self.assertFalse(matrix.verify_quote("   ", hay))
        self.assertTrue(matrix.verify_quote("propose", hay))   # 短，但整串命中

    def test_resolve_page_with_empty_quote(self):
        ctx = {"pages": [(1, "abc")], "by_page": {1: "abc"}, "abstract": "", "hay": "abc"}
        self.assertEqual(matrix._resolve_page("", ctx, 1), (None, "none"))
        self.assertEqual(matrix._resolve_page(None, ctx, None), (None, "none"))

    def test_clip_without_the_keyword_falls_back_to_a_prefix(self):
        long = "z" * 1200
        self.assertEqual(len(matrix._clip(long, "nothing-here")), matrix._MAX_SENT)
        self.assertEqual(matrix._clip("short", "no"), "short")

    def test_is_placeholder_substring_branch(self):
        self.assertTrue(matrix._is_placeholder("（无摘要，仅元数据）"))
        self.assertTrue(matrix._is_placeholder("方法：（摘要未提及）"))
        self.assertTrue(matrix._is_placeholder("（mock 模式：未接 LLM，本卡从摘要直取）"))
        # 有真内容、只是带个说明性括号的，不能误判成占位串（库里真有这种）
        self.assertFalse(matrix._is_placeholder(
            "仿真验证了所提方法的鲁棒性。（摘要未提供具体数值指标）"))

    def test_card_lookup_tolerates_a_non_dict_card(self):
        self.assertEqual(matrix._card_lookup(None, "method"), (None, None))
        self.assertEqual(matrix._card_lookup(["a"], "method"), (None, None))

    def test_stringify_falls_through_on_unknown_types(self):
        self.assertEqual(matrix._stringify({1, 2}), (None, None))
        self.assertEqual(matrix._stringify(object()), (None, None))
        self.assertEqual(matrix._stringify(None), (None, None))

    def test_normalize_columns_rejects_non_string_non_dict(self):
        for bad in ([1], [None], [["method"]]):
            with self.subTest(value=bad), self.assertRaises(ValueError) as cm:
                matrix.normalize_columns(bad)
            self.assertIn("必须是字符串或对象", str(cm.exception))

    def test_normalize_columns_accepts_keywords_as_a_bare_string(self):
        col = matrix.normalize_columns([{"key": "a", "keywords": "Future work"}])[0]
        self.assertEqual(col["keywords"], ["Future work"])
        self.assertEqual(matrix._keywords_for(col), ["Future work"])
        self.assertFalse(matrix._derived_keywords(col))

    def test_llm_cell_accepts_a_bare_string_column(self):
        fake = FakeLLM({self.p1: {"cells": {"method": "模型直接给了一句话"}}})
        with online(fake):
            cell = matrix.build([self.p1], ["method"])["rows"][0]["cells"]["method"]
        self.assertEqual(cell["value"], "模型直接给了一句话")
        self.assertFalse(cell["verified"])
        self.assertIn("未给出原文依据句", cell["note"])

    def test_page_zero_and_garbage_pages_are_ignored(self):
        for bad in (0, -3, "第三页", None, [3], 3.7):
            with self.subTest(page=bad):
                self.assertIsNone(matrix._page_of({"page": bad}))
        self.assertEqual(matrix._page_of({"page": "5"}), 5)


if __name__ == "__main__":
    unittest.main()
