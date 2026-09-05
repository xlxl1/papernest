"""PDF 版面结构解析的离线测试。

测试 PDF 全部由 pymupdf 现场生成（字号/字体/坐标写死），不依赖任何外部样本文件，
因此启发式的判定是确定性的、跨机器可复现。
本模块不联网、不调 LLM、不写库，但仍按项目约定把 DB_PATH/DATA_DIR 指到临时目录、
把 llm/embeddings 摁成不可用——防止将来有人在 structure.py 里加了这类调用而测试不报警。
"""
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pymupdf

from papernest import config, structure

BODY = 10.0          # 正文字号
H1, H2 = 15.0, 12.5  # 一级/二级标题字号


def _write(doc, tmp: Path, name: str) -> str:
    p = tmp / name
    doc.save(str(p))
    doc.close()
    return str(p)


class _Builder:
    """按「从上往下写」的方式造 PDF，自动换页，返回落盘路径。"""

    def __init__(self, tmp: Path, name: str, width=595, height=842):
        self.doc = pymupdf.open()
        self.tmp, self.name = tmp, name
        self.w, self.h = width, height
        self._new_page()

    def _new_page(self):
        self.page = self.doc.new_page(width=self.w, height=self.h)
        self.y = 72.0

    def line(self, text, size=BODY, bold=False, x=72.0, cjk=False):
        if self.y + size + 6 > self.h - 60:
            self._new_page()
        font = "china-s" if cjk else ("hebo" if bold else "helv")
        self.page.insert_text((x, self.y), text, fontsize=size, fontname=font)
        self.y += size + 4
        return self

    def gap(self, dy=14.0):
        self.y += dy
        return self

    def save(self) -> str:
        return _write(self.doc, self.tmp, self.name)


def paper_pdf(tmp: Path) -> str:
    """一篇「长得像论文」的 PDF：大字号标题 + 加粗编号章节 + 正文里的编号反例。"""
    b = _Builder(tmp, "paper.pdf")
    b.line("Section-Aware Chunking for Scientific PDFs", size=17.0, bold=True)
    b.line("Alice Smith, Bob Lee", size=11.0).gap()
    b.line("Abstract", size=H2, bold=True)
    b.line("We propose a layout aware chunking method for scientific papers.")
    b.line("It uses font size and numbering heuristics only.").gap()
    b.line("1 Introduction", size=H1, bold=True)
    b.line("Chunking by page splits a section into two halves and hurts retrieval.")
    # 反例：正文里的编号行，绝不能被判成标题
    b.line("1. First we note that the loss decreases fast in early epochs.")
    b.line("2. Second, the numbering above is body text, not a heading at all.")
    b.line("Results show a consistent gain across five datasets and two models.").gap()
    b.line("2 Related Work", size=H1, bold=True)
    b.line("GROBID is the standard tool but requires a Java service to run.").gap()
    b.line("3 Method", size=H1, bold=True)
    b.line("We combine font size, boldness, numbering and keyword evidence.").gap()
    b.line("3.1 Block Extraction", size=H2, bold=True)
    b.line("Spans are merged into lines and then into blocks by font and spacing.").gap()
    b.line("3.2 Training Details", size=H2, bold=True)
    b.line("We train with Adam for one hundred epochs and a cosine schedule.")
    b.line("Figure 1. The training curve of our model on the validation split.")
    return b.save()


def long_section_pdf(tmp: Path, n_paragraphs=40) -> str:
    """一个超长章节：用来验证按 max_chars 二次切分后文本不丢、标题一致。"""
    b = _Builder(tmp, "long.pdf")
    b.line("4 Experiments", size=H1, bold=True)
    for i in range(n_paragraphs):
        b.line(f"Paragraph {i:02d} reports the accuracy of the model on split {i:02d}.")
    return b.save()


def no_heading_pdf(tmp: Path) -> str:
    """完全没有标题线索的 PDF（扫描 OCR 出来的纯正文常长这样）：两页等宽正文。"""
    b = _Builder(tmp, "flat.pdf")
    for i in range(30):
        b.line(f"plain body line {i:02d} with nothing that looks like a heading here")
    b.line("and one more page of the very same uniform body text follows below")
    for i in range(30, 60):
        b.line(f"plain body line {i:02d} with nothing that looks like a heading here")
    return b.save()


def references_pdf(tmp: Path) -> str:
    """带 References 段的 PDF：方括号编号 + 折行 DOI + arXiv id + 中文条目。"""
    b = _Builder(tmp, "refs.pdf")
    b.line("1 Introduction", size=H1, bold=True)
    b.line("Some body text before the reference section starts here.").gap()
    b.line("References", size=H2, bold=True)
    b.line("[1] A. Smith and B. Lee. Deep learning for channel estimation.")
    b.line("IEEE Trans. Signal Process., 2023. doi:10.1109/tsp.2023.123456")
    # 折行 DOI：行尾停在 "/" 上，下一行接着写——必须能接回来
    b.line("[2] C. Wang. A survey of retrieval augmented generation. ACM")
    b.line("Computing Surveys, 2024. https://doi.org/10.1145/")
    b.line("3649506.3649517")
    b.line("[3] D. Zhang, E. Liu. Scaling laws for neural language models.")
    b.line("arXiv:2401.12345, 2024.")
    # 折行 arXiv id：断在小数点后
    b.line("[4] F. Chen. Long context transformers without retrieval. arXiv:2312.")
    b.line("09876, 2023.")
    b.line("[5] Zhang San, Li Si. A method with no external identifier at all.")
    b.line("Journal of Nothing, 2019.")
    return b.save()


def numdot_refs_pdf(tmp: Path) -> str:
    """另一种排版：条目用 "1." 编号，且 References 标题与首条挤在同一行之外。"""
    b = _Builder(tmp, "refs2.pdf")
    b.line("Bibliography", size=H2, bold=True)
    b.line("1. A. Author. First entry title here. Venue, 2020. doi:10.1000/xyz123")
    b.line("2. B. Author. Second entry title here. Venue, 2021.")
    b.line("3. C. Author. Third entry title here. Venue, 2022.")
    return b.save()


def chinese_pdf(tmp: Path) -> str:
    b = _Builder(tmp, "zh.pdf")
    b.line("第一章 绪论", size=H1, bold=False, cjk=True)
    b.line("按页切分会把一个章节切成两半，检索精度因此下降。", cjk=True).gap()
    b.line("第二节 相关工作", size=H2, cjk=True)
    b.line("已有工作大多依赖外部服务，对个人工具而言过重。", cjk=True).gap()
    b.line("参考文献", size=H2, cjk=True)
    b.line("[1] 张三, 李四. 一种新的信道估计方法. 电子学报, 2021.", cjk=True)
    b.line("[2] 王五. 大模型综述. 计算机学报, 2023.", cjk=True)
    return b.save()


def cross_page_pdf(tmp: Path) -> str:
    """一个章节横跨两页——按页切会把它切成两半，这正是本模块要解决的问题。"""
    b = _Builder(tmp, "cross.pdf")
    b.line("2 Method", size=H1, bold=True)
    for i in range(90):   # 一页装不下，必然跨页
        b.line(f"method line {i:02d} explains one more step of the proposed pipeline")
    b.line("3 Conclusion", size=H1, bold=True)
    b.line("We conclude that section aware chunking helps retrieval.")
    return b.save()


def running_header_pdf(tmp: Path) -> str:
    """每页都有加粗页眉的 5 页文档：页眉不能被当成章节标题。"""
    b = _Builder(tmp, "header.pdf")
    for page in range(5):
        if page:
            b._new_page()
        b.line("Preprint. Under review.", size=11.5, bold=True)
        b.line(f"1.{page} Numbered Section", size=H2, bold=True)
        for i in range(6):
            b.line(f"page {page} body line {i} with ordinary running text in it")
    return b.save()


def roman_heading_pdf(tmp: Path) -> str:
    """IEEE 风格：罗马数字 + 全大写标题，字号与正文相同、只有加粗。"""
    b = _Builder(tmp, "roman.pdf")
    b.line("I. INTRODUCTION", size=BODY, bold=True)
    b.line("Massive MIMO has been widely studied in the last decade.").gap()
    b.line("II. SYSTEM MODEL", size=BODY, bold=True)
    b.line("We consider a single cell uplink system with many antennas.")
    return b.save()


def hanging_indent_refs_pdf(tmp: Path) -> str:
    """APA 风格：没有 [n] 也没有 "1."，靠悬挂缩进区分条目。"""
    b = _Builder(tmp, "apa.pdf")
    b.line("References", size=H2, bold=True)
    b.line("Smith, A., & Lee, B. Deep learning for wireless. Nature, 2022.", x=72.0)
    b.line("Retrieved from https://example.org/paper-one", x=92.0)
    b.line("Wang, C. Retrieval augmented generation at scale. JMLR, 2024.", x=72.0)
    b.line("doi:10.5555/abc.2024.999", x=92.0)
    return b.save()


def three_page_header_pdf(tmp: Path) -> str:
    """只有 3 页的短文（扩展摘要/workshop paper）也有页眉——页数少不是不过滤的理由。"""
    b = _Builder(tmp, "header3.pdf")
    for page in range(3):
        if page:
            b._new_page()
        b.line("Preprint. Under review.", size=11.5, bold=True)
        b.line(f"2.{page} Numbered Section", size=H2, bold=True)
        for i in range(6):
            b.line(f"page {page} body line {i} with ordinary running text in it")
    return b.save()


def repeated_template_headings_pdf(tmp: Path) -> str:
    """学位论文式排版：每章都是「Chapter N」+「同模板的章名」。

    归一化页码后这些标题的 key 完全相同，粗暴的「跨页重复=页眉」会把整篇的
    章节标题一次性删光，全文退化成按页切——这是最贵的一种静默失败。
    """
    b = _Builder(tmp, "thesis.pdf")
    for c in range(1, 6):
        if c > 1:
            b._new_page()
        b.line(f"Chapter {c}", size=16.0, bold=True)
        b.line(f"Topic Number {c}", size=13.0, bold=True)
        for i in range(12):
            b.line(f"chapter {c} body line {i:02d} with some ordinary prose here")
    return b.save()


def theorem_pdf(tmp: Path) -> str:
    """数学味重的论文：定理环境又粗又短，最容易被当成章节标题。

    每个定理头后面都跟一句正文（真实排版就是这样），否则连续的粗体行会被
    MuPDF 并成同一个块、因「行数超限」被丢掉——那样测的就不是题注否决了。
    """
    b = _Builder(tmp, "thm.pdf")
    b.line("3 Results", size=H1, bold=True)
    b.line("The accuracy improves across all datasets we tested here.").gap()
    b.line("Theorem 1 The bound is tight", size=BODY, bold=True)
    b.line("We now prove the statement with a standard argument here.").gap()
    b.line("Lemma 2. A useful inequality", size=BODY, bold=True)
    b.line("The inequality follows from convexity of the objective here.").gap()
    b.line("Definition 3: Coherence", size=BODY, bold=True)
    b.line("Coherence measures the maximal correlation between columns.").gap()
    b.line("Table 2: Accuracy on five datasets", size=BODY, bold=True)
    b.line("Numbers are averaged over three random seeds in this table.").gap()
    b.line("定理 4 收敛性", size=H2, cjk=True)   # 中文定理头：靠字号本可以过线
    b.line("该算法在温和条件下收敛到一个稳定点。", cjk=True)
    return b.save()


def arxiv_url_refs_pdf(tmp: Path) -> str:
    """现实中的 arXiv 引用大量只写 URL，不写 "arXiv:"。"""
    b = _Builder(tmp, "arxivurl.pdf")
    b.line("References", size=H2, bold=True)
    b.line("[1] A. Vaswani et al. Attention is all you need. "
           "https://arxiv.org/abs/1706.03762")
    b.line("[2] B. Lee. Some paper title. Available: http://arxiv.org/pdf/2401.12345v2")
    b.line("[3] C. Wang. Third paper title. https://doi.org/10.1145/3649506.3649517")
    return b.save()


def uniform_heading_pdf(tmp: Path) -> str:
    """没有独立题名的文稿：所有标题同字号，第一节是真章节，不能当题名删掉。"""
    b = _Builder(tmp, "uniform.pdf")
    b.line("Overview", size=13.0, bold=True)
    b.line("This document has no separate title block at all above here.").gap()
    b.line("Rollout Plan", size=13.0, bold=True)
    b.line("The rollout proceeds in three stages over the next two quarters.")
    return b.save()


def blank_page_pdf(tmp: Path) -> str:
    d = pymupdf.open()
    d.new_page()
    return _write(d, tmp, "blank.pdf")


def zero_page_pdf(tmp: Path) -> str:
    """0 页 PDF：pymupdf 存不出来（cannot save with zero pages），手写最小结构。"""
    p = tmp / "zero.pdf"
    p.write_bytes(b"%PDF-1.4\n"
                  b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
                  b"2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\n"
                  b"trailer\n<< /Root 1 0 R /Size 3 >>\n%%EOF\n")
    return str(p)


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s)


class StructureTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="papernest_structure_")
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

    # ── ① 块抽取 ──

    def test_extract_blocks_carries_span_evidence(self):
        blocks = structure.extract_blocks(paper_pdf(self.tmp))
        self.assertTrue(blocks)
        self.assertEqual([b["block_index"] for b in blocks], list(range(len(blocks))))
        for b in blocks:
            self.assertGreaterEqual(b["page_no"], 1)
            self.assertTrue(b["lines"])
            self.assertEqual(b["n_lines"], len(b["lines"]))
            for ln in b["lines"]:
                self.assertIn("size", ln)
                self.assertIn("font", ln)
                self.assertEqual(len(ln["bbox"]), 4)
                self.assertEqual(ln["page_no"], b["page_no"])
        head = next(b for b in blocks if b["text"].startswith("Section-Aware"))
        self.assertTrue(head["bold"])
        self.assertGreater(head["size"], BODY)
        self.assertIn("Bold", head["font"])

    def test_body_font_size_is_the_mode(self):
        blocks = structure.extract_blocks(paper_pdf(self.tmp))
        self.assertEqual(structure.body_font_size(blocks), BODY)

    # ── ② 章节识别 ──

    def test_detect_sections_titles_and_levels(self):
        secs = structure.detect_sections(paper_pdf(self.tmp))
        titles = [s["title"] for s in secs]
        for expect in ("Abstract", "1 Introduction", "2 Related Work", "3 Method",
                       "3.1 Block Extraction", "3.2 Training Details"):
            self.assertIn(expect, titles, f"没识别出章节：{expect}；实得 {titles}")
        by = {s["title"]: s for s in secs}
        self.assertEqual(by["3.2 Training Details"]["level"], 2)
        self.assertEqual(by["3.1 Block Extraction"]["level"], 2)
        self.assertEqual(by["3 Method"]["level"], 1)
        self.assertEqual(by["Abstract"]["level"], 1)
        self.assertEqual([s["block_index"] for s in secs],
                         sorted(s["block_index"] for s in secs))

    def test_matched_by_reflects_real_evidence(self):
        secs = {s["title"]: s for s in structure.detect_sections(paper_pdf(self.tmp))}
        m = secs["3.2 Training Details"]["matched_by"]
        self.assertEqual(set(m), {"font_size", "bold", "numbering", "short_line"})
        a = secs["Abstract"]["matched_by"]
        self.assertIn("keyword", a)
        self.assertIn("bold", a)
        self.assertNotIn("numbering", a)   # Abstract 没编号，不能凭空报出这一条
        self.assertIn("numbering", secs["1 Introduction"]["matched_by"])
        self.assertIn("keyword", secs["2 Related Work"]["matched_by"])

    def test_numbered_body_lines_are_not_headings(self):
        """最容易假绿的地方：正文里的 "1. First we note that..." 不是标题。"""
        secs = structure.detect_sections(paper_pdf(self.tmp))
        titles = [s["title"] for s in secs]
        for t in titles:
            self.assertNotIn("First we note", t)
            self.assertNotIn("Second, the numbering", t)
            self.assertFalse(t.startswith("Results show"),
                             f"关键词 Results 命中了正文句子：{t!r}")
            self.assertFalse(t.startswith("Figure 1"), f"图题被当成章节：{t!r}")

    def test_chinese_sections(self):
        secs = structure.detect_sections(chinese_pdf(self.tmp))
        by = {s["title"]: s for s in secs}
        self.assertIn("第一章 绪论", by)
        self.assertIn("第二节 相关工作", by)
        self.assertIn("参考文献", by)
        self.assertEqual(by["第一章 绪论"]["level"], 1)
        self.assertEqual(by["第二节 相关工作"]["level"], 2)
        self.assertIn("numbering", by["第一章 绪论"]["matched_by"])
        self.assertIn("keyword", by["参考文献"]["matched_by"])

    def test_detection_is_deterministic(self):
        """跨两次调用完全一致——排序键必须是全序，不能靠遍历 set。"""
        path = paper_pdf(self.tmp)
        a = structure.detect_sections(path)
        b = structure.detect_sections(path)
        self.assertEqual(a, b)
        self.assertEqual([c["chunk_index"] for c in structure.section_chunks(path)],
                         [c["chunk_index"] for c in structure.section_chunks(path)])

    # ── ③ 按章节切 chunk ──

    def test_section_chunks_carry_path_and_pages(self):
        chunks = structure.section_chunks(paper_pdf(self.tmp))
        self.assertTrue(chunks)
        self.assertEqual([c["chunk_index"] for c in chunks], list(range(len(chunks))))
        for c in chunks:
            self.assertNotIn("degraded", c)
            self.assertLessEqual(c["start_page"], c["end_page"])
        deep = next(c for c in chunks if c["section_title"] == "3.2 Training Details")
        self.assertEqual(deep["section_path"], "3 Method > 3.2 Training Details")
        self.assertEqual(deep["level"], 2)
        self.assertIn("cosine schedule", deep["text"])
        intro = next(c for c in chunks if c["section_title"] == "1 Introduction")
        self.assertEqual(intro["section_path"], "1 Introduction")
        # 正文里的编号行留在正文 chunk 里，没有被切成章节
        self.assertIn("First we note", intro["text"])
        head = chunks[0]
        self.assertEqual(head["section_title"], "")
        self.assertEqual(head["section_path"], "（文首，章节之前）")
        self.assertIn("Alice Smith", head["text"])

    def test_long_section_split_keeps_title_and_loses_no_text(self):
        path = long_section_pdf(self.tmp)
        whole = structure.section_chunks(path, max_chars=100000)
        self.assertEqual(len(whole), 1)
        full = whole[0]["text"]
        parts = structure.section_chunks(path, max_chars=400)
        self.assertGreater(len(parts), 3)
        self.assertEqual({c["section_title"] for c in parts}, {"4 Experiments"})
        self.assertEqual({c["section_path"] for c in parts}, {"4 Experiments"})
        self.assertEqual([c["part"] for c in parts], list(range(1, len(parts) + 1)))
        self.assertEqual({c["n_parts"] for c in parts}, {len(parts)})
        for c in parts:
            self.assertLessEqual(c["n_chars"], 400)
        self.assertEqual(_norm("".join(c["text"] for c in parts)), _norm(full))
        self.assertIn("Paragraph 00", parts[0]["text"])
        self.assertIn("Paragraph 39", parts[-1]["text"])

    def test_section_spanning_a_page_break_stays_one_chunk(self):
        """本模块存在的理由：跨页的章节不该被页边界切成两半。"""
        path = cross_page_pdf(self.tmp)
        chunks = structure.section_chunks(path, max_chars=100000)
        method = [c for c in chunks if c["section_title"] == "2 Method"]
        self.assertEqual(len(method), 1)
        self.assertEqual(method[0]["start_page"], 1)
        self.assertEqual(method[0]["end_page"], 2)   # 一个 chunk 横跨两页
        self.assertIn("method line 00", method[0]["text"])
        self.assertIn("method line 89", method[0]["text"])
        conclusion = [c for c in chunks if c["section_title"] == "3 Conclusion"]
        self.assertEqual(len(conclusion), 1)
        self.assertNotIn("method line", conclusion[0]["text"])

    def test_running_header_is_not_a_section(self):
        secs = structure.detect_sections(running_header_pdf(self.tmp))
        titles = [s["title"] for s in secs]
        self.assertNotIn("Preprint. Under review.", titles)
        self.assertEqual(titles, [f"1.{i} Numbered Section" for i in range(5)])
        self.assertEqual({s["level"] for s in secs}, {2})

    def test_roman_all_caps_headings_without_size_change(self):
        """字号与正文相同、只有加粗 + 罗马编号 + 全大写，也要认得出来。"""
        secs = structure.detect_sections(roman_heading_pdf(self.tmp))
        titles = [s["title"] for s in secs]
        self.assertEqual(titles, ["I. INTRODUCTION", "II. SYSTEM MODEL"])
        m = secs[0]["matched_by"]
        self.assertNotIn("font_size", m)      # 字号确实没变大，不能虚报这条判据
        self.assertIn("bold", m)
        self.assertIn("all_caps", m)
        self.assertIn("numbering", m)
        self.assertIn("keyword", m)

    def test_paper_title_is_not_reported_as_a_section(self):
        secs = structure.detect_sections(paper_pdf(self.tmp))
        self.assertNotIn("Section-Aware Chunking for Scientific PDFs",
                         [s["title"] for s in secs])

    def test_no_heading_degrades_to_page_chunks(self):
        chunks = structure.section_chunks(no_heading_pdf(self.tmp))
        self.assertTrue(chunks)
        for c in chunks:
            self.assertEqual(c["degraded"], "未识别到章节结构，已退化为按页切分")
            self.assertEqual(c["section_title"], "")
            self.assertEqual(c["level"], 0)
            self.assertEqual(c["start_page"], c["end_page"])
        self.assertEqual(sorted({c["start_page"] for c in chunks}), [1, 2])

    def test_page_fallback_still_respects_max_chars(self):
        chunks = structure.section_chunks(no_heading_pdf(self.tmp), max_chars=300)
        self.assertTrue(chunks)
        for c in chunks:
            self.assertLessEqual(c["n_chars"], 300)
            self.assertEqual(c["degraded"], "未识别到章节结构，已退化为按页切分")

    # ── ④ 参考文献 ──

    def test_references_doi_arxiv_and_folded_doi(self):
        r = structure.extract_references(references_pdf(self.tmp))
        self.assertEqual(r["n_entries"], 5, r["entries"])
        self.assertEqual(r["split_by"], "bracket")
        by = {e["index"]: e for e in r["entries"]}
        self.assertEqual(by[1]["doi"], "10.1109/tsp.2023.123456")
        # 折行 DOI：https://doi.org/10.1145/ + 3649506.3649517 必须接回来
        self.assertEqual(by[2]["doi"], "10.1145/3649506.3649517")
        self.assertEqual(by[3]["arxiv_id"], "2401.12345")
        # 折行 arXiv：arXiv:2312. + 09876
        self.assertEqual(by[4]["arxiv_id"], "2312.09876")
        self.assertEqual(by[5]["doi"], "")
        self.assertEqual(by[5]["arxiv_id"], "")
        self.assertIsNone(by[5]["norm_key"])
        self.assertEqual(r["with_doi"], 2)
        self.assertEqual(r["with_arxiv"], 2)
        self.assertEqual(r["with_norm_key"], 4)
        self.assertEqual(by[1]["norm_key"], "doi:10.1109/tsp.2023.123456")
        self.assertEqual(by[3]["norm_key"], "arxiv:2401.12345")
        self.assertEqual(by[1]["year"], 2023)
        self.assertEqual(by[5]["year"], 2019)
        self.assertIn("Deep learning for channel estimation", by[1]["title_guess"])
        self.assertIn("survey of retrieval augmented generation", by[2]["title_guess"])
        # 作者段（拼音全名、没有 A. 缩写）不能被当成标题印出去
        self.assertEqual(by[5]["title_guess"], "A method with no external identifier at all")
        self.assertIsNone(r["degraded"])
        for e in r["entries"]:
            self.assertTrue(e["raw"])
            self.assertGreaterEqual(e["page_no"], 1)

    def test_references_numdot_layout(self):
        r = structure.extract_references(numdot_refs_pdf(self.tmp))
        self.assertEqual(r["split_by"], "numdot")
        self.assertEqual(r["n_entries"], 3)
        self.assertEqual([e["index"] for e in r["entries"]], [1, 2, 3])
        self.assertEqual(r["entries"][0]["doi"], "10.1000/xyz123")
        self.assertEqual(r["entries"][0]["year"], 2020)
        self.assertEqual(r["entries"][2]["doi"], "")

    def test_references_hanging_indent_layout(self):
        r = structure.extract_references(hanging_indent_refs_pdf(self.tmp))
        self.assertEqual(r["split_by"], "hanging-indent")
        self.assertEqual(r["n_entries"], 2, r["entries"])
        self.assertTrue(r["entries"][0]["raw"].startswith("Smith, A."))
        self.assertIn("Retrieved from", r["entries"][0]["raw"])  # 缩进的续行并进同一条
        self.assertEqual(r["entries"][1]["doi"], "10.5555/abc.2024.999")
        self.assertEqual(r["entries"][1]["year"], 2024)

    def test_references_chinese(self):
        r = structure.extract_references(chinese_pdf(self.tmp))
        self.assertEqual(r["n_entries"], 2)
        self.assertIn("电子学报", r["entries"][0]["raw"])
        self.assertEqual(r["entries"][0]["year"], 2021)
        self.assertEqual(r["entries"][0]["title_guess"], "一种新的信道估计方法")
        self.assertEqual(r["with_doi"], 0)

    def test_no_references_section_is_reported_not_faked(self):
        r = structure.extract_references(long_section_pdf(self.tmp))
        self.assertEqual(r["n_entries"], 0)
        self.assertEqual(r["entries"], [])
        self.assertEqual(r["with_doi"], 0)
        self.assertEqual(r["with_arxiv"], 0)
        self.assertIn("未定位到", r["degraded"])

    def test_references_section_is_not_swallowed_by_body(self):
        """References 段本身不该混进正文 chunk 之外的章节里（起始页要对得上）。"""
        path = references_pdf(self.tmp)
        r = structure.extract_references(path)
        self.assertEqual(r["start_page"], 1)
        secs = [s["title"] for s in structure.detect_sections(path)]
        self.assertIn("References", secs)

    # ── ⑤ summarize 与异常输入 ──

    def test_summarize_returns_everything(self):
        s = structure.summarize(references_pdf(self.tmp))
        self.assertEqual(s["body_font_size"], BODY)
        self.assertIsNone(s["degraded"])
        self.assertGreater(s["n_chunks"], 0)
        self.assertEqual(s["n_sections"], len(s["sections"]))
        self.assertEqual(s["references"]["n_entries"], 5)
        self.assertEqual(s["n_pages"], 1)
        self.assertEqual(s["warnings"], [])

    def test_summarize_marks_page_fallback(self):
        s = structure.summarize(no_heading_pdf(self.tmp))
        self.assertEqual(s["sections"], [])
        self.assertEqual(s["degraded"], "未识别到章节结构，已退化为按页切分")
        self.assertGreater(s["n_chunks"], 0)
        self.assertTrue(any("参考文献" in w for w in s["warnings"]))

    def test_blank_page_pdf_does_not_explode(self):
        path = blank_page_pdf(self.tmp)
        self.assertEqual(structure.extract_blocks(path), [])
        self.assertEqual(structure.detect_sections(path), [])
        self.assertEqual(structure.section_chunks(path), [])
        r = structure.extract_references(path)
        self.assertEqual(r["n_entries"], 0)
        s = structure.summarize(path)
        self.assertIn("没有可提取的文本层", s["degraded"])
        self.assertEqual(s["n_pages"], 1)
        self.assertEqual(s["body_font_size"], 0.0)

    def test_zero_page_pdf_does_not_explode(self):
        path = zero_page_pdf(self.tmp)
        self.assertEqual(structure.extract_blocks(path), [])
        self.assertEqual(structure.section_chunks(path), [])
        s = structure.summarize(path)
        self.assertEqual(s["n_pages"], 0)
        self.assertEqual(s["n_chunks"], 0)
        self.assertIn("没有可提取的文本层", s["degraded"])

    def test_unopenable_file_raises_structure_error(self):
        bad = self.tmp / "bad.pdf"
        bad.write_bytes(b"this is definitely not a pdf")
        for fn in (structure.extract_blocks, structure.detect_sections,
                   structure.section_chunks, structure.extract_references,
                   structure.summarize):
            with self.assertRaises(structure.StructureError):
                fn(str(bad))
        with self.assertRaises(structure.StructureError):
            structure.summarize(str(self.tmp / "does-not-exist.pdf"))

    def test_structure_error_is_value_error(self):
        """调用方 except ValueError 也应兜得住（与 pdfimport.PdfImportError 一致）。"""
        self.assertTrue(issubclass(structure.StructureError, ValueError))

    # ── 内部判据的定向测试（防回归）──

    def test_numbering_levels(self):
        self.assertEqual(structure._numbering("3.2 Training")[1], 2)
        self.assertEqual(structure._numbering("3.2.1 Warmup")[1], 3)
        self.assertEqual(structure._numbering("4 Experiments")[1], 1)
        self.assertEqual(structure._numbering("IV. RESULTS")[1], 1)
        self.assertEqual(structure._numbering("第三章 方法")[1], 1)
        self.assertEqual(structure._numbering("第二节 相关工作")[1], 2)
        self.assertIsNone(structure._numbering("Introduction"))

    def test_sentence_veto(self):
        self.assertTrue(structure._looks_like_sentence(
            "1. First we note that the loss decreases fast."))
        self.assertTrue(structure._looks_like_sentence("本文提出了一种新的信道估计方法。"))
        self.assertFalse(structure._looks_like_sentence("3.2 Training Details"))
        self.assertFalse(structure._looks_like_sentence("Abstract"))

    def test_split_entries_inline_brackets(self):
        """[1]...[2]... 全挤在同一行时也要切开，且编号不能变成假条目。"""
        seg = [{"text": "[1] A. Smith. First paper title. Venue, 2020. "
                        "[2] B. Lee. Second paper title. Venue, 2021.",
                "page_no": 3, "x0": 72.0, "y0": 100.0, "y1": 110.0}]
        groups, how = structure._split_entries(seg)
        self.assertEqual(how, "bracket-inline")
        self.assertEqual(len(groups), 2)
        self.assertTrue(groups[0][0]["text"].startswith("[1]"))
        self.assertTrue(groups[1][0]["text"].startswith("[2]"))

    def test_body_sentence_starting_with_references_is_not_the_ref_section(self):
        blocks = [
            {"block_index": 0, "page_no": 1,
             "lines": [{"text": "References to prior work show that chunking matters.",
                        "page_no": 1, "x0": 72.0, "y0": 100.0, "y1": 110.0}]},
        ]
        r = structure._extract_references(blocks)
        self.assertEqual(r["n_entries"], 0)
        self.assertIn("未定位到", r["degraded"])

    def test_join_entry_does_not_glue_complete_doi(self):
        """DOI 已经写完时不能无脑粘下一行——那会造出一个假 DOI。"""
        raw = structure._join_entry(
            ["A. Smith. Title. Venue, 2023. doi:10.1109/tsp.2023.123456",
             "Smith, J. Another line of the same entry."])
        self.assertEqual(structure._find_doi(raw), "10.1109/tsp.2023.123456")
        self.assertIn("456 Smith", raw)

    def test_join_entry_does_not_glue_doi_ending_with_sentence_period(self):
        """DOI 以句号收尾、下一行是大写开头的新句子——粘回去就是造假 DOI。"""
        raw = structure._join_entry(
            ["A. Smith. Title. Venue. doi:10.1109/tsp.2023.123456.",
             "Smith, J. Trailing note of the same entry."])
        self.assertEqual(structure._find_doi(raw), "10.1109/tsp.2023.123456")
        self.assertNotIn("123456.Smith", raw)

    def test_join_entry_repairs_word_break_hyphen(self):
        raw = structure._join_entry(["A. Smith. Retrieval augmen-", "ted generation."])
        self.assertIn("augmented generation", raw)

    # ── 回归：折行修复不能造出假 DOI（续行不是大写开头的情形）──

    def test_join_entry_does_not_glue_doi_to_lowercase_continuation(self):
        """DOI 写完 + 句号，续行是 "pp. 1234-1245."。

        续行以小写字母开头就无脑粘，会造出 10.1109/tsp.2021.123456.pp ——
        这个假 DOI 会当作 norm_key 写进引文图，比抽不出来有害得多。
        """
        raw = structure._join_entry(
            ["[3] X. Li. A title here. IEEE Trans., 2021. doi:10.1109/tsp.2021.123456.",
             "pp. 1234-1245."])
        self.assertEqual(structure._find_doi(raw), "10.1109/tsp.2021.123456")
        self.assertNotIn("123456.pp", raw)

    def test_join_entry_does_not_glue_doi_to_digit_continuation(self):
        """续行以数字开头也不行：下一条 "4. W. Wang..." 会被接成 10.1000/xyz123.4。"""
        raw = structure._join_entry(
            ["3. Z. Chen. Third entry. Venue, 2020. doi:10.1000/xyz123.",
             "4. W. Wang. Fourth entry. Venue, 2021."])
        self.assertEqual(structure._find_doi(raw), "10.1000/xyz123")
        self.assertNotIn("xyz123.4", raw)

    def test_join_entry_does_not_glue_doi_to_page_range(self):
        raw = structure._join_entry(
            ["A. Author. Title. Venue. https://doi.org/10.1145/3649506.3649517.",
             "1234-1245, 2024."])
        self.assertEqual(structure._find_doi(raw), "10.1145/3649506.3649517")

    def test_join_entry_still_repairs_a_genuine_fold_after_a_dot(self):
        """反方向：真折行（DOI 断在小数点后、续行是长数字串）仍要接得回来。"""
        raw = structure._join_entry(
            ["A. Author. Title. ACM Computing Surveys, 2024. https://doi.org/10.1145/3649506.",
             "3649517"])
        self.assertEqual(structure._find_doi(raw), "10.1145/3649506.3649517")

    # ── 回归：页眉过滤不能连真标题一起删 ──

    def test_running_header_on_a_three_page_pdf_is_filtered(self):
        """页眉过滤原先要求 >= 4 页，3 页的短文里页眉会变成 3 个假章节。"""
        secs = structure.detect_sections(three_page_header_pdf(self.tmp))
        titles = [s["title"] for s in secs]
        self.assertNotIn("Preprint. Under review.", titles)
        self.assertEqual(titles, [f"2.{i} Numbered Section" for i in range(3)])

    def test_same_template_headings_are_not_deleted_as_running_headers(self):
        """「Chapter N」式的同模板标题归一化页码后 key 相同，不能整批当页眉删掉。"""
        path = repeated_template_headings_pdf(self.tmp)
        titles = [s["title"] for s in structure.detect_sections(path)]
        for c in range(1, 6):
            self.assertIn(f"Topic Number {c}", titles, f"章节标题被当页眉删了：{titles}")
        chunks = structure.section_chunks(path)
        self.assertTrue(chunks)
        for c in chunks:
            self.assertNotIn("degraded", c)   # 不该退化成按页切

    # ── 回归：定理环境不是章节 ──

    def test_theorem_environments_are_not_sections(self):
        titles = [s["title"] for s in structure.detect_sections(theorem_pdf(self.tmp))]
        self.assertEqual(titles, ["3 Results"], f"定理/引理/定义被当成章节：{titles}")

    # ── 回归：arXiv 的 URL 写法也要能接进引文图 ──

    def test_arxiv_url_forms_are_recovered(self):
        r = structure.extract_references(arxiv_url_refs_pdf(self.tmp))
        by = {e["index"]: e for e in r["entries"]}
        self.assertEqual(by[1]["arxiv_id"], "1706.03762")
        self.assertEqual(by[2]["arxiv_id"], "2401.12345")   # /pdf/…v2 也要认
        self.assertEqual(by[2]["norm_key"], "arxiv:2401.12345")
        self.assertEqual(by[3]["doi"], "10.1145/3649506.3649517")
        self.assertEqual(r["with_norm_key"], 3)

    # ── 回归：所有标题同字号时，第一节不是题名 ──

    def test_first_section_is_kept_when_no_title_stands_out(self):
        titles = [s["title"] for s in structure.detect_sections(uniform_heading_pdf(self.tmp))]
        self.assertEqual(titles, ["Overview", "Rollout Plan"])

    def test_is_bold_reads_latex_subset_font_names(self):
        """LaTeX 嵌入子集字体：cmbx/sfbx 是粗体，随机六字母前缀不能当证据。"""
        for name in ("GHIJKL+CMBX12", "YZABCD+SFBX1000", "ABCDEF+CMSSBX10",
                     "ABCDEF+NimbusRomNo9L-Medi", "STUVWX+TimesNewRomanPS-BoldMT"):
            self.assertTrue(structure._is_bold({"font": name, "flags": 0}), name)
        for name in ("ABCDEF+NimbusRomNo9L-Regu", "MNOPQR+CMR10", "ABCDEF+CMTI10"):
            self.assertFalse(structure._is_bold({"font": name, "flags": 0}), name)
        self.assertTrue(structure._is_bold({"font": "MNOPQR+CMR10", "flags": 16}))


if __name__ == "__main__":
    unittest.main()
