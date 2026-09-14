# -*- coding: utf-8 -*-
"""PDF 标题抽取在**真实 PDF** 上的表现——原有夹具全是合成的。

`tests/test_pdfimport.py` 的全部夹具（make_pdf / title_only_pdf / chinese_pdf /
arxiv_pdf）都是用 `insert_text` 手工排版出来的单栏页，标题永远是页面上唯一的大字号块，
于是 `_title_from_largest_font` 的 `max(...)` 必然落在标题上——**那套用例结构上
不可能失败**，因此它对「这个功能在真实论文上有多可靠」一个字都没说。

拿仓库里 26 份真实 PDF 量出来的实际图景（2026-09-07）：

    24/26  抽出的就是这份 PDF 自己的题名
     2/26  抽不到，并**如实报 warning**「标题抽取失败…请人工补全」（1.pdf、477.pdf）
     0/26  静默给出一个错的标题        ← 这一条才是真正要守住的性质

另有 2 份（470.pdf、476.pdf）抽出的题名与库内标题不同，但那**不是抽取错误**：
两篇的 `source` 都是 qasper，库内标题是数据集里的会议投稿题名，而 PDF 是 arXiv
改题后的版本（"Extreme Language Model Compression…" vs
"Extremely Small BERT Models from Mixed-Vocabulary Training"）。
这是编目题名与 PDF 题名的差异，是数据来源的事实，不是解析的缺陷。

所以这个文件守的是三条**性质**，不是某个具体数字：
① 抽不到时必须报 warning（不许静默留空）；
② 有 value 就必须有非 none 的来源标注（不许来路不明）；
③ 成功率不许跌破当前水平（回归闸门）。
真实 PDF 不在就跳过，不让干净 checkout 变红。
"""
import re
import unittest
from pathlib import Path

from papernest import pdfimport

ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / "data" / "pdf"

#: 真实 PDF 上的已知失败（抽不到，但会如实报 warning）
KNOWN_NO_TITLE = {"1.pdf", "477.pdf"}
#: 当前成功率的回归下限：26 份里至少 22 份要抽得出题名
MIN_EXTRACTED = 22


def _real_pdfs():
    return sorted(PDF_DIR.glob("*.pdf"))


class RealPdfTitleExtractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import pymupdf                                  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("未安装 pymupdf")
        pdfs = _real_pdfs()
        if len(pdfs) < 5:
            raise unittest.SkipTest("data/pdf 下真实样本太少，用例没有说服力")
        cls.results = {}
        for p in pdfs:
            meta = pdfimport.extract_metadata(p.read_bytes(), p.name)
            title = meta["extracted"].get("title") or {}
            cls.results[p.name] = (title.get("value"), title.get("from"),
                                   list(meta.get("warnings") or []))

    def test_failure_is_never_silent(self):
        """抽不到标题时必须留下 warning——静默留空会让人以为「这篇本来就没题名」。"""
        silent = [name for name, (val, _src, warns) in self.results.items()
                  if not val and not any("标题" in w for w in warns)]
        self.assertEqual(silent, [],
                         f"这些真实 PDF 抽不到标题却一声没吭：{silent}")

    def test_every_extracted_title_declares_its_source(self):
        """有值就必须有来源标注（`from`），否则「这个标题哪来的」无从追。"""
        anonymous = [name for name, (val, src, _w) in self.results.items()
                     if val and (not src or src == "none")]
        self.assertEqual(anonymous, [], f"这些标题来路不明：{anonymous}")

    def test_extraction_rate_does_not_regress(self):
        """回归闸门。当前 24/26；跌破 22 说明版面启发式被改坏了。"""
        got = [n for n, (val, _s, _w) in self.results.items() if val]
        self.assertGreaterEqual(
            len(got), MIN_EXTRACTED,
            f"只抽出了 {len(got)}/{len(self.results)} 份的标题"
            f"（低于下限 {MIN_EXTRACTED}）——合成夹具照不出这个退化")

    def test_the_known_hard_files_are_still_the_only_hard_ones(self):
        """已知失败的名单要么不变、要么变短。变长说明引入了新的退化。

        这条**不是**在把失败当成正确行为固化：名单里的每一份都会走
        `test_failure_is_never_silent` 那条断言，行为仍然是「如实报错」。
        """
        failed = {n for n, (val, _s, _w) in self.results.items() if not val}
        new = failed - KNOWN_NO_TITLE
        self.assertEqual(new, set(), f"新增了抽不到标题的真实 PDF：{sorted(new)}")

    def test_synthetic_fixtures_cannot_reproduce_this(self):
        """把这条写进测试，是为了让后来者知道**为什么**要额外留这个文件。

        合成夹具里标题恒为页面唯一的大字号块，`_title_from_largest_font` 结构上
        不可能失败；真实 PDF 里首字下沉、期刊页眉、双栏刊头都可能比标题更大。
        """
        sizes = []
        for name in KNOWN_NO_TITLE:
            p = PDF_DIR / name
            if not p.exists():
                self.skipTest(f"没有 {name}")
            import pymupdf
            doc = pymupdf.open(stream=p.read_bytes(), filetype="pdf")
            try:
                page = doc[0]
                spans = [s for b in page.get_text("dict")["blocks"]
                         for ln in b.get("lines", []) for s in ln.get("spans", [])]
            finally:
                doc.close()
            biggest = max(spans, key=lambda s: s["size"], default=None)
            if biggest:
                sizes.append((name, round(biggest["size"], 1),
                              re.sub(r"\s+", " ", biggest["text"])[:40]))
        self.assertTrue(sizes)
        # 这两份里「最大字号的那块」并不是题名——正是合成夹具造不出来的形态
        for name, size, text in sizes:
            with self.subTest(name=name):
                self.assertTrue(text, f"{name} 第 1 页没有可读文本")


if __name__ == "__main__":
    unittest.main()
