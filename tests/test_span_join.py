"""span 拼接必须补回词间空格。

审计确认的 blocker：`structure._line_of` 与 `pdfimport` 都是
`spans = [s for s in ... if s.text.strip()]` 之后 `"".join(...)`——
先把纯空白的 span 过滤掉，再无分隔拼接，于是相邻的词被粘死。

真库实测：走这条路的 `chunks` 里 **435/784（55.5%）** 含 25 字母以上的粘连词
（`pioNER:DatasetsandBaselinesforArmenian`、`ntelligenceisenteringapivotalerawiththe`），
而同一批 PDF 走 `page.get_text("text")` 的 `pages` 表是 **0%**。

**为什么此前 30 多条切分用例全绿**：夹具用 `page.insert_text((x,y), text)` 整行写入，
PyMuPDF 回读时一行只有**一个**含空格的 span，`"".join` 恰好等价于正确拼接——
合成夹具把真实数据的形态整个排除在外了。所以这里：
- 纯函数用**手工构造的多 span 行**测（那才是真实 PDF 的形态）；
- 另有一条用仓库里真实 PDF 的集成用例（没有该文件就跳过）。
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

from papernest import structure

ROOT = Path(__file__).resolve().parent.parent
GLUE = re.compile(r"[a-z]{25,}")


def _span(text, x0, x1, size=10.0):
    return {"text": text, "size": size, "bbox": (x0, 0.0, x1, size)}


class JoinSpans(unittest.TestCase):

    def test_gap_between_words_becomes_a_space(self):
        """真实 PDF 里最常见的形态：字体切换把一行切成多个 span，
        词间空格只体现为 bbox 之间的水平间距。"""
        spans = [_span("Large", 0, 30), _span("Language", 34, 80),
                 _span("Model", 84, 115)]
        self.assertEqual(structure.join_spans(spans), "Large Language Model")

    def test_no_gap_means_no_space(self):
        """同一个词被切成两段（连字/上标）时不能插空格。"""
        spans = [_span("Trans", 0, 30), _span("former", 30.2, 60)]
        self.assertEqual(structure.join_spans(spans), "Transformer")

    def test_whitespace_only_span_is_kept(self):
        """PyMuPDF 有时把空格单独放一个 span——先 strip 过滤就是把词边界丢掉。"""
        spans = [_span("Named", 0, 32), _span(" ", 32, 35), _span("Entity", 35, 70)]
        self.assertEqual(structure.join_spans(spans), "Named Entity")

    def test_existing_trailing_space_is_not_doubled(self):
        spans = [_span("Named ", 0, 35), _span("Entity", 40, 70)]
        self.assertEqual(structure.join_spans(spans), "Named Entity")

    def test_hyphenated_linebreak_is_not_split(self):
        """断词连字符结尾不补空格，否则 'estima-tion' 会变成 'estima- tion'。"""
        spans = [_span("estima-", 0, 40), _span("tion", 45, 70)]
        self.assertEqual(structure.join_spans(spans), "estima-tion")

    def test_the_real_failure_case(self):
        """审计里那条真实样本的形态。"""
        words = ["Intelligence", "is", "entering", "a", "pivotal", "era", "with", "the"]
        spans, x = [], 0.0
        for w in words:
            spans.append(_span(w, x, x + len(w) * 5))
            x += len(w) * 5 + 4        # 词间留 4pt 间距（> 0.12 * 10）
        out = structure.join_spans(spans)
        self.assertEqual(out, " ".join(words))
        self.assertIsNone(GLUE.search(out.lower()), "仍然粘连")

    def test_threshold_is_proportional_to_font_size(self):
        """大字号标题的词间距按绝对值更大，阈值必须按字号成比例。"""
        spans = [_span("Deep", 0, 60, size=24.0), _span("Learning", 63, 140, size=24.0)]
        self.assertEqual(structure.join_spans(spans), "Deep Learning")

    def test_empty_and_malformed_spans_do_not_crash(self):
        self.assertEqual(structure.join_spans([]), "")
        self.assertEqual(structure.join_spans(None), "")
        self.assertEqual(structure.join_spans([{"text": "a"}, {"text": "b"}]), "ab")
        self.assertEqual(
            structure.join_spans([{"text": "x", "bbox": None, "size": None}]), "x")

    def test_old_implementation_would_have_glued(self):
        """把旧写法摆在旁边，说明这条用例确实咬得住那个 bug。"""
        spans = [_span("Named", 0, 32), _span("Entity", 35, 70),
                 _span("Recognition", 74, 130)]
        old = "".join(s["text"] for s in spans if s["text"].strip())
        self.assertEqual(old, "NamedEntityRecognition")
        self.assertEqual(structure.join_spans(spans), "Named Entity Recognition")


class RealPdfHasNoGluedWords(unittest.TestCase):
    """集成层：拿仓库里真实的 PDF 走一遍，粘连词必须为 0。

    合成夹具测不出这个 bug（`insert_text` 一行只产出一个含空格的 span），
    所以这条必须用真 PDF。文件不在就跳过，不让干净 checkout 变红。
    """

    def test_no_glued_lines(self):
        pdfs = sorted((ROOT / "data" / "pdf").glob("*.pdf"))
        if not pdfs:
            self.skipTest("data/pdf 下没有真实 PDF 样本")
        try:
            import pymupdf
        except ImportError:
            self.skipTest("未安装 pymupdf")

        glued, total = [], 0
        doc = pymupdf.open(pdfs[0])
        try:
            for pno, page in enumerate(doc, 1):
                if pno > 8:
                    break
                for block in page.get_text("dict").get("blocks", []):
                    for ln in block.get("lines", []):
                        text = structure._clean(
                            structure.join_spans(ln.get("spans", []) or []))
                        if not text:
                            continue
                        total += 1
                        if GLUE.search(text.lower()):
                            glued.append(text[:60])
        finally:
            doc.close()

        self.assertGreater(total, 50, "样本 PDF 抽不出足够的行，用例没有说服力")
        self.assertEqual(glued, [],
                         f"{len(glued)}/{total} 行仍有粘连词，样例：{glued[:3]}")


if __name__ == "__main__":
    unittest.main()
