"""PDF 表格识别与「整行不截断」切分的离线测试。

测试 PDF 全部由 pymupdf 现场生成（坐标 / 字号 / 字体写死），不依赖任何外部样本，
因此启发式的判定是确定性的、跨机器可复现。本模块不联网、不调 LLM、不写库，
但仍按项目约定把 DB_PATH/DATA_DIR 指到临时目录、把 llm/embeddings 摁成不可用——
防止将来有人往 tables.py 里加了这类调用而测试不报警（开发机 .env 真配了 EMBED_MODEL）。

夹具刻意贴近真实形态：结果表的数据列是真数字（表头判定靠的就是「表头非数字、
数据行有数字」），参考文献夹具带悬挂缩进（这是几何兜底最容易误判的东西），
split_page_text 的页文本同时覆盖「一行一格」与「一行一排格」两种抽取形态。
"""
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pymupdf

from papernest import config, tables

BODY = 10.0
BOLD, PLAIN = "hebo", "helv"
CJK = "china-s"


# ── 夹具：程序化生成 PDF ──

def _save(doc, tmp: Path, name: str) -> str:
    p = tmp / name
    doc.save(str(p))
    doc.close()
    return str(p)


#: 一张真实形态的结果表：表头是词、数据是小数。
RESULT_TABLE = [
    ["Model", "Recall", "MRR", "nDCG"],
    ["BM25", "0.612", "0.401", "0.455"],
    ["Dense", "0.708", "0.502", "0.560"],
    ["Hybrid", "0.771", "0.548", "0.601"],
    ["Ours", "0.812", "0.573", "0.634"],
]


def _draw_grid(page, x0, y0, cw, rh, n_rows, n_cols):
    for r in range(n_rows + 1):
        page.draw_line((x0, y0 + r * rh), (x0 + n_cols * cw, y0 + r * rh))
    for c in range(n_cols + 1):
        page.draw_line((x0 + c * cw, y0), (x0 + c * cw, y0 + n_rows * rh))


def _fill(page, data, x0, y0, cw, rh, dy, bold_first=True, fontname=PLAIN):
    for r, row in enumerate(data):
        for c, v in enumerate(row):
            fn = BOLD if (bold_first and r == 0 and fontname == PLAIN) else fontname
            page.insert_text((x0 + c * cw, y0 + r * rh + dy), v, fontsize=BODY, fontname=fn)


def bordered_pdf(tmp: Path, data=None, name="bordered.pdf") -> str:
    """有框线的表：find_tables() 的主场。"""
    data = data or RESULT_TABLE
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    _draw_grid(page, 72, 100, 110, 24, len(data), len(data[0]))
    _fill(page, data, 76, 100, 110, 24, 16)
    return _save(doc, tmp, name)


def borderless_pdf(tmp: Path, name="borderless.pdf") -> str:
    """无框线的空格对齐表（学术论文的三线表就长这样）：几何兜底的主场。

    上下各留一段正文，用来验证候选区不会把正文吞进去。
    """
    data = [["Variant", "Recall", "MRR", "nDCG"],
            ["full", "0.812", "0.573", "0.634"],
            ["-rerank", "0.744", "0.510", "0.571"],
            ["-header", "0.690", "0.470", "0.522"],
            ["-tables", "0.655", "0.441", "0.498"]]
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 70), "Table 2. Ablation study on five benchmarks.",
                     fontsize=BODY, fontname=PLAIN)
    _fill(page, data, 72, 100, 110, 20, 0)
    page.insert_text((72, 100 + len(data) * 20 + 40),
                     "The ablation shows that removing repetition costs the most.",
                     fontsize=BODY, fontname=PLAIN)
    return _save(doc, tmp, name)


def cjk_borderless_pdf(tmp: Path) -> str:
    """中文无框线表：单元格是中文，列宽比英文窄，验证几何兜底不挑语言。"""
    data = [["方法", "召回率", "准确率", "备注"],
            ["词面检索", "0.612", "0.401", "基线"],
            ["向量检索", "0.708", "0.502", "需要模型"],
            ["混合检索", "0.771", "0.548", "本文"]]
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    _fill(page, data, 72, 100, 110, 20, 0, bold_first=False, fontname=CJK)
    return _save(doc, tmp, "cjk.pdf")


def headerless_pdf(tmp: Path) -> str:
    """没有表头的表：每一行都是数字。has_header 必须如实为 False。"""
    data = [[f"{i}.{j}0" for j in range(1, 5)] for i in range(1, 6)]
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    _fill(page, data, 72, 100, 110, 20, 0, bold_first=False)
    return _save(doc, tmp, "headerless.pdf")


def prose_pdf(tmp: Path) -> str:
    """纯正文：一张表都不该有。"""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    y = 72
    for i in range(25):
        page.insert_text((72, y), f"This is an ordinary body line number {i:02d} here.",
                         fontsize=BODY, fontname=PLAIN)
        y += 14
    return _save(doc, tmp, "prose.pdf")


def references_pdf(tmp: Path) -> str:
    """参考文献段：编号列 + 悬挂缩进正文列，x 区间对齐得比真表还整齐。

    这是几何兜底最经典的误报源——GEOM_MIN_COLS=3 就是为它定的，必须测。
    """
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    y = 72
    page.insert_text((72, y), "References", fontsize=12.5, fontname=BOLD)
    y += 20
    for i in range(1, 9):
        page.insert_text((72, y), f"[{i}]", fontsize=BODY, fontname=PLAIN)
        page.insert_text((100, y), f"A. Author{i} and B. Coauthor. Paper title {i}.",
                         fontsize=BODY, fontname=PLAIN)
        y += 14
        page.insert_text((100, y), f"Journal of Testing, 20{10 + i}.",
                         fontsize=BODY, fontname=PLAIN)
        y += 16
    return _save(doc, tmp, "refs.pdf")


def mixed_page_pdf(tmp: Path) -> str:
    """正文 + 表 + 正文的一页：split_page_text 的被测对象。"""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    y = 72
    page.insert_text((72, y), "3 Experiments", fontsize=14.0, fontname=BOLD)
    y += 22
    for s in ("We evaluate our chunking strategy on five retrieval benchmarks.",
              "Every run uses the same encoder and the same index configuration."):
        page.insert_text((72, y), s, fontsize=BODY, fontname=PLAIN)
        y += 16
    y += 14
    page.insert_text((72, y), "Table 2. Ablation study on five benchmarks.",
                     fontsize=BODY, fontname=PLAIN)
    y += 24
    data = [["Variant", "Recall", "MRR", "nDCG"],
            ["full", "0.812", "0.573", "0.634"],
            ["-rerank", "0.744", "0.510", "0.571"],
            ["-header", "0.690", "0.470", "0.522"],
            ["-tables", "0.655", "0.441", "0.498"]]
    _fill(page, data, 72, y, 110, 20, 0)
    y += len(data) * 20 + 30
    for s in ("Removing header repetition costs the most recall of all ablations.",
              "This confirms that a table chunk without its header is nearly useless."):
        page.insert_text((72, y), s, fontsize=BODY, fontname=PLAIN)
        y += 16
    return _save(doc, tmp, "mixed.pdf")


def cross_page_pdf(tmp: Path) -> str:
    """一张表被页边界劈成两半：第 2 页那半没有表头——本模块存在的理由。"""
    doc = pymupdf.open()
    p1 = doc.new_page(width=595, height=842)
    head = ["Model", "Recall", "MRR", "nDCG"]
    top = [[f"m{i:02d}", f"0.{600 + i}", f"0.{400 + i}", f"0.{500 + i}"] for i in range(6)]
    _fill(p1, [head] + top, 72, 700, 110, 20, 0)
    p2 = doc.new_page(width=595, height=842)
    bottom = [[f"m{i:02d}", f"0.{600 + i}", f"0.{400 + i}", f"0.{500 + i}"] for i in range(6, 12)]
    _fill(p2, bottom, 72, 72, 110, 20, 0, bold_first=False)
    return _save(doc, tmp, "crosspage.pdf")


def two_page_pdf(tmp: Path) -> str:
    """两页各一张表：验证 page_no 过滤。"""
    doc = pymupdf.open()
    for tag in ("A", "B"):
        page = doc.new_page(width=595, height=842)
        data = [[f"{tag}col1", f"{tag}col2", f"{tag}col3"]] + \
               [[f"{tag}{i}", f"0.{100 + i}", f"0.{200 + i}"] for i in range(4)]
        _fill(page, data, 72, 100, 110, 20, 0)
    return _save(doc, tmp, "twopage.pdf")


def blank_page_pdf(tmp: Path) -> str:
    doc = pymupdf.open()
    doc.new_page()
    return _save(doc, tmp, "blank.pdf")


def zero_page_pdf(tmp: Path) -> str:
    """0 页 PDF：pymupdf 存不出来（cannot save with zero pages），手写最小结构。"""
    p = tmp / "zero.pdf"
    p.write_bytes(b"%PDF-1.4\n"
                  b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
                  b"2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\n"
                  b"trailer\n<< /Root 1 0 R /Size 3 >>\n%%EOF\n")
    return str(p)


def broken_pdf(tmp: Path) -> str:
    p = tmp / "broken.pdf"
    p.write_bytes(b"this is definitely not a pdf at all")
    return str(p)


# ── 把 Markdown 表解析回来（验证「没散架」）──

#: 未被转义的竖线才是列分隔符；`\|` 是单元格内容。
_SPLIT_RE = re.compile(r"(?<!\\)\|")


def parse_md(md: str) -> list[list[str]]:
    """只认 `|` 开头的行；返回不含分隔行的单元格矩阵。"""
    rows = []
    for line in md.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in _SPLIT_RE.split(line)[1:-1]]
        if all(set(c) <= set("-: ") and c for c in cells):
            continue  # 分隔行
        rows.append(cells)
    return rows


def unescape(cell: str) -> str:
    return cell.replace("<br>", "\n").replace("\\|", "|").replace("\\\\", "\\")


def make_table(rows, has_header=True, n_cols=None) -> dict:
    return {"page_no": 1, "bbox": (0.0, 0.0, 100.0, 100.0),
            "n_rows": len(rows), "n_cols": n_cols or max(len(r) for r in rows),
            "rows": rows, "detected_by": "geometry", "has_header": has_header}


class _Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="papernest_tables_")
        self.addCleanup(shutil.rmtree, self._tmp, True)
        self.tmp = Path(self._tmp)
        self._old_db, self._old_dir = config.DB_PATH, config.DATA_DIR
        config.DATA_DIR = self.tmp / "data"
        config.DB_PATH = config.DATA_DIR / "test.db"
        self._patches = [
            mock.patch("papernest.embeddings.available", return_value=False),
            mock.patch("papernest.llm.available", return_value=False),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        config.DB_PATH, config.DATA_DIR = self._old_db, self._old_dir
        shutil.rmtree(self._tmp, ignore_errors=True)


# ── ① 识别 ──

class DetectTests(_Base):
    def test_bordered_table_uses_find_tables(self):
        got = tables.detect_tables(bordered_pdf(self.tmp))
        self.assertEqual(len(got), 1)
        t = got[0]
        self.assertEqual(t["detected_by"], "find_tables")
        self.assertEqual((t["n_rows"], t["n_cols"]), (5, 4))
        self.assertEqual(t["rows"], RESULT_TABLE)
        self.assertTrue(t["has_header"])
        self.assertEqual(t["page_no"], 1)
        self.assertEqual(len(t["bbox"]), 4)
        self.assertLess(t["bbox"][0], t["bbox"][2])
        self.assertLess(t["bbox"][1], t["bbox"][3])

    def test_borderless_table_falls_back_to_geometry(self):
        got = tables.detect_tables(borderless_pdf(self.tmp))
        self.assertEqual(len(got), 1, f"应只识别出 1 张表，实际 {got}")
        t = got[0]
        self.assertEqual(t["detected_by"], "geometry")
        self.assertEqual((t["n_rows"], t["n_cols"]), (5, 4))
        self.assertEqual(t["rows"][0], ["Variant", "Recall", "MRR", "nDCG"])
        self.assertEqual(t["rows"][-1], ["-tables", "0.655", "0.441", "0.498"])
        self.assertTrue(t["has_header"])
        # 上下的正文不能被吞进表格区
        flat = " ".join(c for r in t["rows"] for c in r)
        self.assertNotIn("Table 2", flat)
        self.assertNotIn("ablation shows", flat)

    def test_detected_by_is_honest_when_find_tables_missing(self):
        """老版本 pymupdf 没有 find_tables：必须优雅降级到几何，并如实标注。"""
        path = bordered_pdf(self.tmp)
        with mock.patch.object(pymupdf.Page, "find_tables", None):
            got, warns = tables._detect(path)
        self.assertTrue(got, "没有 find_tables 时几何兜底也应识别出这张表")
        self.assertTrue(all(t["detected_by"] == "geometry" for t in got))
        self.assertTrue(any("find_tables" in w for w in warns), warns)

    def test_find_tables_exception_degrades_not_raises(self):
        path = bordered_pdf(self.tmp)

        def boom(self, *a, **kw):
            raise RuntimeError("模拟 find_tables 内部炸了")

        with mock.patch.object(pymupdf.Page, "find_tables", boom):
            got, warns = tables._detect(path)
        self.assertTrue(got, "find_tables 抛异常后应由几何兜底接住")
        self.assertTrue(all(t["detected_by"] == "geometry" for t in got))
        self.assertTrue(any("find_tables()" in w for w in warns), warns)

    def test_cjk_table_is_detected(self):
        got = tables.detect_tables(cjk_borderless_pdf(self.tmp))
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]["n_cols"], 4)
        self.assertEqual(got[0]["rows"][0], ["方法", "召回率", "准确率", "备注"])

    def test_headerless_table_reports_has_header_false(self):
        got = tables.detect_tables(headerless_pdf(self.tmp))
        self.assertEqual(len(got), 1)
        self.assertFalse(got[0]["has_header"], "全是数字的表不该被说成有表头")

    def test_prose_pdf_has_no_tables(self):
        self.assertEqual(tables.detect_tables(prose_pdf(self.tmp)), [])

    def test_reference_list_is_not_mistaken_for_a_table(self):
        """悬挂缩进的参考文献 x 对齐很整齐，是几何兜底的头号误报源。"""
        got = tables.detect_tables(references_pdf(self.tmp))
        self.assertEqual(got, [], f"参考文献被误判成了表格：{got}")

    def test_page_no_filter(self):
        path = two_page_pdf(self.tmp)
        all_t = tables.detect_tables(path)
        self.assertEqual(len(all_t), 2)
        self.assertEqual([t["page_no"] for t in all_t], [1, 2])
        p2 = tables.detect_tables(path, page_no=2)
        self.assertEqual(len(p2), 1)
        self.assertEqual(p2[0]["page_no"], 2)
        self.assertEqual(p2[0]["rows"][0][0], "Bcol1")
        self.assertEqual(tables.detect_tables(path, page_no=9), [])

    def test_cross_page_table_each_half_is_detected(self):
        """本模块存在的理由：第 2 页那半没有表头，但仍要被识别成表格。"""
        got = tables.detect_tables(cross_page_pdf(self.tmp))
        self.assertEqual([t["page_no"] for t in got], [1, 2])
        self.assertTrue(got[0]["has_header"])
        self.assertFalse(got[1]["has_header"], "下半张表本来就没表头，不能假装有")
        self.assertEqual(got[1]["rows"][0][0], "m06")

    def test_blank_and_zero_page_pdf_are_graceful(self):
        self.assertEqual(tables.detect_tables(blank_page_pdf(self.tmp)), [])
        got, warns = tables._detect(zero_page_pdf(self.tmp))
        self.assertEqual(got, [])
        self.assertTrue(any("0" in w for w in warns), warns)

    def test_unopenable_pdf_raises_table_error(self):
        with self.assertRaises(tables.TableError):
            tables.detect_tables(broken_pdf(self.tmp))
        with self.assertRaises(tables.TableError):
            tables.detect_tables(self.tmp / "does_not_exist.pdf")

    def test_two_runs_are_identical(self):
        """排序键必须全序：同一 PDF 连跑两次结果要逐字节一致。"""
        path = mixed_page_pdf(self.tmp)
        a = tables.detect_tables(path)
        b = tables.detect_tables(path)
        self.assertEqual(a, b)
        self.assertEqual(tables.summarize(path), tables.summarize(path))
        self.assertEqual(tables.chunk_table(a[0]), tables.chunk_table(b[0]))


# ── ② Markdown ──

class MarkdownTests(_Base):
    def test_plain_table_round_trips(self):
        md = tables.table_to_markdown(make_table(RESULT_TABLE))
        self.assertEqual(parse_md(md), RESULT_TABLE)
        self.assertIn("| --- | --- | --- | --- |", md)

    def test_pipe_newline_and_cjk_do_not_break_the_grid(self):
        rows = [["列|名", "第二列", "第三列"],
                ["a|b", "第一行\n第二行", "中文 内容"],
                ["C:\\path", "", "尾"]]
        md = tables.table_to_markdown(make_table(rows))
        parsed = parse_md(md)
        self.assertTrue(all(len(r) == 3 for r in parsed), f"列数散架了：{parsed}")
        self.assertEqual(unescape(parsed[0][0]), "列|名")
        self.assertEqual(unescape(parsed[1][0]), "a|b")
        self.assertEqual(unescape(parsed[1][1]), "第一行\n第二行")
        self.assertEqual(unescape(parsed[2][0]), "C:\\path")
        self.assertNotIn("\n第二行", md.split("\n")[2].split("|")[0])  # 换行没漏出去
        self.assertEqual(len(md.splitlines()), 4)  # 表头 + 分隔 + 2 数据行

    def test_missing_header_borrows_first_row_and_says_so(self):
        rows = [["1.0", "2.0"], ["3.0", "4.0"]]
        md = tables.table_to_markdown(make_table(rows, has_header=False))
        self.assertTrue(md.startswith(tables.NOTE_BORROWED_HEADER))
        self.assertEqual(parse_md(md), rows)
        # 有表头时不加这句
        self.assertNotIn(tables.NOTE_BORROWED_HEADER,
                         tables.table_to_markdown(make_table(rows, has_header=True)))

    def test_ragged_rows_are_padded_not_dropped(self):
        rows = [["a", "b", "c"], ["only one"], ["x", "y", "z", "extra"]]
        md = tables.table_to_markdown(make_table(rows, n_cols=3))
        parsed = parse_md(md)
        self.assertTrue(all(len(r) == 3 for r in parsed), parsed)
        self.assertEqual(parsed[1], ["only one", "", ""])
        self.assertEqual(parsed[2][2], "z extra", "多出来的格要并进最后一格，不能丢")

    def test_empty_table_returns_empty_string(self):
        self.assertEqual(tables.table_to_markdown(make_table([[]], n_cols=0)), "")
        self.assertEqual(tables.table_to_markdown({"rows": [], "n_cols": 0}), "")


# ── ③ 切分（核心）──

def _rows(n: int, cols=3) -> list[list[str]]:
    head = ["Model", "Recall", "MRR"][:cols]
    return [head] + [[f"m{i:02d}"] + [f"0.{500 + i + j}" for j in range(cols - 1)]
                     for i in range(n)]


class ChunkTests(_Base):
    def test_60_rows_split_into_two_and_second_carries_header(self):
        """核心断言：60 数据行按 max_rows=50 切成 2 块，第 2 块带表头。"""
        t = make_table(_rows(60))
        chunks = tables.chunk_table(t, max_rows=50)
        self.assertEqual(len(chunks), 2)
        self.assertEqual([c["n_rows"] for c in chunks], [50, 10])
        self.assertEqual([(c["part"], c["of"]) for c in chunks], [(1, 2), (2, 2)])
        self.assertFalse(chunks[0]["header_repeated"])
        self.assertTrue(chunks[1]["header_repeated"])
        second = parse_md(chunks[1]["text"])
        self.assertEqual(second[0], ["Model", "Recall", "MRR"],
                         "第 2 块必须自带表头，否则它被单独检索到时不可读")
        self.assertEqual(second[1][0], "m50")
        self.assertEqual(len(second), 11)  # 表头 + 10 数据行
        # 数据行一条不丢、不重
        got = [r[0] for c in chunks for r in parse_md(c["text"])[1:]]
        self.assertEqual(got, [f"m{i:02d}" for i in range(60)])

    def test_repeat_header_false_keeps_columns_but_drops_names(self):
        t = make_table(_rows(60))
        chunks = tables.chunk_table(t, max_rows=50, repeat_header=False)
        self.assertEqual(len(chunks), 2)
        self.assertFalse(chunks[1]["header_repeated"])
        self.assertIn(tables.NOTE_CONT_NO_HEADER, chunks[1]["text"])
        second = parse_md(chunks[1]["text"])
        self.assertEqual(second[0], ["", "", ""], "续块要用空表头占位，保持列数")
        self.assertTrue(all(len(r) == 3 for r in second))

    def test_char_limit_splits_even_under_row_limit(self):
        """双限：行数没到上限，字符数到了也要切。"""
        t = make_table([["A", "B"]] + [[f"r{i:02d}", "x" * 100] for i in range(20)])
        chunks = tables.chunk_table(t, max_rows=50, max_chars=600)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(c["n_rows"] < 20 for c in chunks))
        self.assertTrue(all(len(c["text"]) <= 600 for c in chunks),
                        [len(c["text"]) for c in chunks])
        self.assertEqual(sum(c["n_rows"] for c in chunks), 20)

    def test_oversized_row_is_never_truncated(self):
        """一行超长：整行单独成块 + warning，绝不截断。"""
        long_cell = "超长单元格内容" * 100
        rows = [["A", "B"], ["short", "ok"], ["big", long_cell], ["tail", "end"]]
        chunks = tables.chunk_table(make_table(rows), max_rows=50, max_chars=200)
        flagged = [c for c in chunks if "warning" in c]
        self.assertEqual(len(flagged), 1, [c.get("warning") for c in chunks])
        self.assertEqual(flagged[0]["n_rows"], 1, "超长行必须自己单独成块")
        self.assertIn(long_cell, flagged[0]["text"], "内容被截断了")
        self.assertGreater(len(flagged[0]["text"]), 200, "这块本来就该超限")
        self.assertIn("不截断", flagged[0]["warning"])
        # 其余行不受影响，也没丢
        cells = [r[0] for c in chunks for r in parse_md(c["text"])[1:]]
        self.assertEqual(cells, ["short", "big", "tail"])

    def test_borrowed_header_note_only_on_first_chunk(self):
        t = make_table(_rows(60), has_header=False)
        chunks = tables.chunk_table(t, max_rows=50)
        self.assertIn(tables.NOTE_BORROWED_HEADER, chunks[0]["text"])
        self.assertNotIn(tables.NOTE_BORROWED_HEADER, chunks[1]["text"])
        self.assertIn(tables.NOTE_CONT_WITH_HEADER, chunks[1]["text"])

    def test_small_table_is_one_chunk(self):
        chunks = tables.chunk_table(make_table(RESULT_TABLE))
        self.assertEqual(len(chunks), 1)
        self.assertEqual((chunks[0]["part"], chunks[0]["of"]), (1, 1))
        self.assertEqual(chunks[0]["n_rows"], 4)
        self.assertFalse(chunks[0]["header_repeated"])

    def test_header_only_and_empty_tables(self):
        only = tables.chunk_table(make_table([["A", "B"]]))
        self.assertEqual(len(only), 1)
        self.assertEqual(only[0]["n_rows"], 0)
        self.assertEqual(tables.chunk_table({"rows": [], "n_cols": 0}), [])

    def test_degenerate_limits_do_not_hang_or_crash(self):
        chunks = tables.chunk_table(make_table(_rows(5)), max_rows=0, max_chars=0)
        self.assertEqual(len(chunks), 5)
        self.assertTrue(all(c["n_rows"] == 1 for c in chunks))


# ── ④ 一页拆块 ──

class SplitPageTests(_Base):
    def _page_text(self, path: str, page_no=1) -> str:
        doc = pymupdf.open(path)
        try:
            return doc[page_no - 1].get_text("text")
        finally:
            doc.close()

    def test_table_and_body_never_share_a_block(self):
        path = mixed_page_pdf(self.tmp)
        found = tables.detect_tables(path, page_no=1)
        self.assertEqual(len(found), 1)
        blocks = tables.split_page_text(self._page_text(path), found)
        kinds = [b["kind"] for b in blocks]
        self.assertIn("table", kinds)
        self.assertIn("text", kinds)
        self.assertEqual([b["index"] for b in blocks], list(range(len(blocks))))
        table_blocks = [b for b in blocks if b["kind"] == "table"]
        text_blocks = [b for b in blocks if b["kind"] == "text"]
        # 表格数据不能出现在正文块里
        for b in text_blocks:
            for cell in ("0.812", "-rerank", "nDCG"):
                self.assertNotIn(cell, b["text"], f"表格数据漏进正文块：{b['text']!r}")
        # 正文不能出现在表格块里
        for b in table_blocks:
            self.assertNotIn("We evaluate", b["text"])
            self.assertNotIn("Removing header repetition", b["text"])
        # 正文两段都在，顺序保持
        joined = "\n".join(b["text"] for b in text_blocks)
        self.assertIn("We evaluate our chunking strategy", joined)
        self.assertIn("Removing header repetition", joined)
        self.assertLess(blocks.index(text_blocks[0]), blocks.index(table_blocks[0]))

    def test_row_per_line_extraction_is_also_recognised(self):
        """另一种常见抽取形态：一整排单元格挤在同一行文本里。"""
        t = make_table(RESULT_TABLE)
        page_text = (
            "We report the main results below.\n"
            "Model    Recall   MRR     nDCG\n"
            "BM25     0.612    0.401   0.455\n"
            "Dense    0.708    0.502   0.560\n"
            "Hybrid   0.771    0.548   0.601\n"
            "Ours     0.812    0.573   0.634\n"
            "Our hybrid model wins on every dataset in the table above.\n")
        blocks = tables.split_page_text(page_text, [t])
        text_join = "\n".join(b["text"] for b in blocks if b["kind"] == "text")
        self.assertIn("We report the main results", text_join)
        self.assertIn("Our hybrid model wins", text_join)
        self.assertNotIn("0.612", text_join)
        self.assertNotIn("BM25", text_join)
        self.assertEqual(sum(1 for b in blocks if b["kind"] == "table"), 1)

    def test_body_line_sharing_words_with_cells_stays_body(self):
        """覆盖率兜底：一句正文里偶然含两个单元格词，不能被摘进表格区。"""
        t = make_table([["Recall", "Full", "MRR"],
                        ["0.612", "0.401", "0.455"],
                        ["0.708", "0.502", "0.560"]])
        line = "We measure recall on the full benchmark suite and report the mean."
        blocks = tables.split_page_text(line + "\n0.612 0.401 0.455\n", [t])
        text_join = "\n".join(b["text"] for b in blocks if b["kind"] == "text")
        self.assertIn("We measure recall on the full benchmark", text_join)

    def test_unmatched_table_is_appended_not_lost(self):
        t = make_table(RESULT_TABLE)
        blocks = tables.split_page_text("完全无关的一页正文，跟表格一个字都对不上。", [t])
        self.assertEqual([b["kind"] for b in blocks], ["text", "table"])
        self.assertIn("| BM25 |", blocks[1]["text"])

    def test_long_body_is_split_on_paragraph_boundaries(self):
        paras = [f"这是第 {i} 段正文，讲的是表格切分对检索的影响。" * 6 for i in range(6)]
        blocks = tables.split_page_text("\n\n".join(paras), [], max_chars=400)
        text_blocks = [b for b in blocks if b["kind"] == "text"]
        self.assertGreater(len(text_blocks), 1)
        self.assertTrue(all(b["n_chars"] <= 400 for b in text_blocks),
                        [b["n_chars"] for b in text_blocks])
        for i in range(6):
            self.assertIn(f"这是第 {i} 段正文", "\n".join(b["text"] for b in text_blocks))

    def test_no_tables_means_pure_text_blocks(self):
        blocks = tables.split_page_text("just one short paragraph here", [])
        self.assertEqual([b["kind"] for b in blocks], ["text"])
        self.assertEqual(tables.split_page_text("", []), [])

    def test_table_chunks_keep_their_metadata(self):
        t = make_table(_rows(60))
        blocks = tables.split_page_text("", [t], max_chars=900)
        table_blocks = [b for b in blocks if b["kind"] == "table"]
        self.assertGreater(len(table_blocks), 1)
        self.assertEqual([b["part"] for b in table_blocks],
                         list(range(1, len(table_blocks) + 1)))
        self.assertTrue(all(b["of"] == len(table_blocks) for b in table_blocks))
        self.assertTrue(all(b["table_index"] == 0 for b in table_blocks))
        self.assertTrue(table_blocks[1]["header_repeated"])


# ── ⑤ 汇总 ──

class SummarizeTests(_Base):
    def test_summarize_counts_by_detector(self):
        s = tables.summarize(bordered_pdf(self.tmp))
        self.assertEqual(s["n_tables"], 1)
        self.assertEqual(s["pages_with_tables"], [1])
        self.assertEqual(s["by_detector"], {"find_tables": 1, "geometry": 0})
        self.assertEqual(s["warnings"], [])
        self.assertFalse(s["degraded"])
        self.assertEqual(s["n_pages"], 1)

        g = tables.summarize(borderless_pdf(self.tmp))
        self.assertEqual(g["by_detector"], {"find_tables": 0, "geometry": 1})

    def test_summarize_on_pdf_without_tables(self):
        s = tables.summarize(prose_pdf(self.tmp))
        self.assertEqual(s["n_tables"], 0)
        self.assertEqual(s["pages_with_tables"], [])
        self.assertEqual(s["by_detector"], {"find_tables": 0, "geometry": 0})

    def test_summarize_marks_degraded_when_find_tables_unavailable(self):
        path = bordered_pdf(self.tmp)
        with mock.patch.object(pymupdf.Page, "find_tables", None):
            s = tables.summarize(path)
        self.assertTrue(s["degraded"])
        self.assertTrue(s["warnings"])
        self.assertEqual(s["by_detector"]["find_tables"], 0)

    def test_summarize_multi_page(self):
        s = tables.summarize(cross_page_pdf(self.tmp))
        self.assertEqual(s["n_tables"], 2)
        self.assertEqual(s["pages_with_tables"], [1, 2])
        self.assertEqual(s["n_pages"], 2)


if __name__ == "__main__":
    unittest.main()
