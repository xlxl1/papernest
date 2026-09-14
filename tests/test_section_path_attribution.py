# -*- coding: utf-8 -*-
r"""漏检子标题 → section_path 给出错误出处。

审计原话是「`HEADING_MIN_SCORE=3` 太高，子标题拿不满 3 分被丢弃」。**实测不成立**：
真库里被标错的那一节（1.pdf 的 6.2）评分正好是 3，够门槛；它是被评分**之前**的
两道否决项拦掉的。把门槛降到 2 一个都救不回来，却会多出 436 个假章节
（26 份真实 PDF 上误检 67 → 409，6.1 倍）。

真正的两条成因（26 份真实 PDF 上各自可复现）：
① `_looks_like_sentence`：标题末尾带句号（"6.2 Memory Constraints and
   Long-Term Adaptation."）+ token 数 ≥4 → 被当成正文句子否决。
② `n_lines > MAX_HEADING_LINES`：LaTeX 的 \section 把节号排成独立的行盒，与标题
   首行**同基线**（482.pdf 的 '4.4' 与 'Unsupervised Dependency Parsing' 都是
   y0=65.4），MuPDF 按行盒计数，两行标题报成 3 行 → 被长度否决。这条远比 ① 常见。

两种情况下 `_section_chunks` 的边界完全由 sections 决定，漏掉那一节的正文被并进
上一节，section_path 从此给出错误出处——这是「引用带页码/章节号」这个卖点的
**正确性**问题，不是召回问题。真库实测 13/804 个 chunk（1.6%）带着错误的出处。

**必须用真实 PDF**：合成夹具的 `insert_text` 一行只产出一个行盒，既造不出
「节号独立行盒」的形态，也不会自然写出带句号的标题；原有 169 条版面用例
一条都没照出这个问题。文件不在就跳过，不让干净 checkout 变红。
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

from papernest import structure

ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / "data" / "pdf"


def _pdf(name: str) -> str:
    p = PDF_DIR / name
    if not p.exists():
        raise unittest.SkipTest(f"没有真实样本 {p}")
    return str(p)


def _blocks_and_sections(path: str):
    blocks, n_pages = structure._read(path)
    if not blocks:
        raise unittest.SkipTest(f"{path} 没有文本层")
    body = structure.body_font_size(blocks)
    return blocks, structure._detect_sections(blocks, body, n_pages)


def _chunk_holding(chunks: list[dict], needle: str) -> dict:
    for c in chunks:
        if needle in c["text"]:
            return c
    raise AssertionError(f"真实样本里找不到 {needle!r}，用例前提已变")


class TrailingPeriodHeadingTests(unittest.TestCase):
    """成因①：标题末尾的句号让它被当成正文句子。"""

    HEAD = "6.2 Memory Constraints and Long-Term Adaptation."
    BODY62 = "Maintaining coherence across multi-turn dialogues"

    def test_heading_is_detected(self):
        _, secs = _blocks_and_sections(_pdf("1.pdf"))
        titles = [s["title"] for s in secs]
        self.assertIn("6.1 Scalability and Coordination", titles,
                      "6.1 都没检出，样本或上游解析变了，本用例失去意义")
        self.assertIn(
            self.HEAD, titles,
            f"6.2 被漏检；实际检出的 6.x：{[t for t in titles if t.startswith('6')]}")

    def test_body_is_not_attributed_to_the_previous_section(self):
        chunks = structure.section_chunks(_pdf("1.pdf"), 4000)
        c = _chunk_holding(chunks, self.BODY62)
        self.assertIn("6.2", c["section_path"] or "",
                      f"6.2 的正文被标成了别处的出处：{c['section_path']!r}")
        self.assertNotIn("6.1 Scalability", c["section_path"] or "")

    def test_heading_text_does_not_leak_into_body(self):
        """标题行本身不该出现在任何 chunk 的正文里——出现了就说明它没被当成边界。"""
        chunks = structure.section_chunks(_pdf("1.pdf"), 4000)
        leaked = [c["section_path"] for c in chunks if self.HEAD in c["text"]]
        self.assertEqual(leaked, [],
                         f"标题行漏进了正文，说明它没被识别为章节边界：{leaked}")


class HangingSectionNumberTests(unittest.TestCase):
    """成因②：节号是同基线上的独立行盒，把 n_lines 顶过 MAX_HEADING_LINES。"""

    HEAD = "4.4 Unsupervised Dependency Parsing without gold POS tags"

    def test_block_really_has_the_hanging_number_shape(self):
        """先把成因坐实：这个块的行盒数 > 2，但视觉基线只有 2 条。

        这条**修法前后都必须绿**——它断言的是成因存在，用来保证下面那条用例
        真的咬在这个 bug 上，而不是碰巧通过。
        """
        blocks, _ = _blocks_and_sections(_pdf("482.pdf"))
        hit = [b for b in blocks if b["text"] == self.HEAD]
        self.assertEqual(len(hit), 1, "样本变了，找不到 4.4 这一块")
        b = hit[0]
        self.assertGreater(b["n_lines"], structure.MAX_HEADING_LINES,
                           "样本已不是「节号独立行盒」的形态，本用例失去意义")
        baselines = {round(ln["y0"], 0) for ln in b["lines"]}
        self.assertLessEqual(len(baselines), structure.MAX_HEADING_LINES,
                             "视觉行也超过 2 行，那就不是这条 bug")

    def test_heading_is_detected(self):
        _, secs = _blocks_and_sections(_pdf("482.pdf"))
        self.assertIn(self.HEAD, [s["title"] for s in secs])


class NumberingContinuityTests(unittest.TestCase):
    """全语料不变量：检出的编号序列不该跳号，而被跳过的那个号**就摆在原文里**。

    判据不依赖任何外部真值，也不自证：缺口完全由「已检出的前后两个兄弟编号」定义，
    中间那个块必须同字号、够短、有字母、不是题注。命中即说明有一节正文正挂在
    错误的 section_path 下。
    """

    def _gaps(self, path: str):
        blocks, secs = _blocks_and_sections(path)
        seen = {}
        for s in secs:
            m = structure._NUM_ARABIC_RE.match(s["title"])
            if m and m.group(1) not in seen:
                seen[m.group(1)] = s
        out = []
        for num, s in seen.items():
            parts = num.split(".")
            nxt = ".".join(parts[:-1] + [str(int(parts[-1]) + 1)])
            after = ".".join(parts[:-1] + [str(int(parts[-1]) + 2)])
            if nxt in seen or after not in seen:
                continue
            lo, hi = s["block_index"], seen[after]["block_index"]
            for b in blocks:
                if not (lo < b["block_index"] < hi):
                    continue
                m = structure._NUM_ARABIC_RE.match(b["text"])
                if (m and m.group(1) == nxt
                        and len(b["text"]) <= structure.MAX_HEADING_CHARS
                        and re.search(r"[A-Za-z一-鿿]", b["text"])
                        and not structure._CAPTION_RE.match(b["text"])
                        and abs(b["size"] - s["font_size"]) <= 0.1):
                    out.append((nxt, b["page_no"], b["text"][:70]))
                    break
        return out

    def test_no_numbering_gap_in_any_real_pdf(self):
        pdfs = sorted(PDF_DIR.glob("*.pdf"))
        if len(pdfs) < 5:
            self.skipTest("data/pdf 下真实样本太少，用例没有说服力")
        bad = []
        for p in pdfs:
            for num, page, text in self._gaps(str(p)):
                bad.append(f"{p.name} p{page} 漏检 {num}：{text!r}")
        self.assertEqual(bad, [], "编号跳号 = 有整节正文挂在错误的出处上：\n  "
                                  + "\n  ".join(bad))


if __name__ == "__main__":
    unittest.main()
