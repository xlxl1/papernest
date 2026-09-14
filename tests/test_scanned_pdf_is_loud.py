# -*- coding: utf-8 -*-
"""扫描件（无文本层）入库必须**说出来**，不能看起来像成功。

实测修复前：造一个无文本层的 PDF 走完整条 `ingest_pdf`——

    papers=1 · pages=0 · chunks=0
    warnings = [标题抽取失败, 作者抽取失败, 未查到 DOI, 年份抽取失败, 摘要抽取失败, …]

**入库「成功」了**，而 warnings 里一句都没说「这份 PDF 没有文本层、全文一个字都没进库」。
用户会把那串警告读成「元数据要手工补」，而不是「这份文档整个检索不到」。
这是本项目最贵的一种静默失败：文件确实在库里、列表里看得见、点开有标题，
只是问它什么都答不上来——而失败点在几天前的那次上传。

本项目**不做 OCR**（全仓零匹配），所以正确的行为不是偷偷降级，是明确告诉用户
「这条路走不通，请先转成可搜索 PDF」。
"""
import shutil
import tempfile
import unittest
from pathlib import Path

from papernest import config, db, pdfimport


def _pdf(with_text: bool) -> bytes:
    import pymupdf
    doc = pymupdf.open()
    page = doc.new_page()
    if with_text:
        page.insert_text((72, 100), "Neural Machine Translation", fontsize=18)
        page.insert_text((72, 140), "We study sequence models on WMT14.", fontsize=11)
    else:
        page.draw_rect(pymupdf.Rect(50, 50, 300, 200), color=(0, 0, 0))
    data = doc.tobytes()
    doc.close()
    return data


class ScannedPdfSaysSoTests(unittest.TestCase):
    def setUp(self):
        try:
            import pymupdf                                   # noqa: F401
        except ImportError:
            self.skipTest("未安装 pymupdf")
        self._tmp = tempfile.mkdtemp(prefix="papernest_scan_")
        self.addCleanup(shutil.rmtree, self._tmp, True)
        self._old = (config.DB_PATH, config.DATA_DIR)
        config.DATA_DIR = Path(self._tmp) / "data"
        config.DB_PATH = config.DATA_DIR / "test.db"
        self.addCleanup(self._restore)
        db.init_db()

    def _restore(self):
        config.DB_PATH, config.DATA_DIR = self._old

    def test_no_text_layer_is_named_explicitly(self):
        r = pdfimport.ingest_pdf(_pdf(False), "scan.pdf", make_card=False)
        self.assertEqual(r["pages_stored"], 0, "夹具不是无文本层的，用例是空跑")
        named = [w for w in r["warnings"] if "没有文本层" in w]
        self.assertEqual(
            len(named), 1,
            f"没有任何一条警告点破「这份 PDF 没有文本层」——"
            f"用户只会看到：{[w[:20] for w in r['warnings']]}")
        msg = named[0]
        self.assertIn("检索", msg, "没说清后果（检索不到）")
        self.assertIn("OCR", msg, "没说清本项目不做 OCR、该怎么办")

    def test_a_normal_pdf_does_not_get_the_warning(self):
        """有文本层的照常入库，不该被这条警告污染。"""
        r = pdfimport.ingest_pdf(_pdf(True), "ok.pdf", make_card=False)
        self.assertGreater(r["pages_stored"], 0)
        self.assertEqual([w for w in r["warnings"] if "没有文本层" in w], [])

    def test_text_layer_probe_is_honest_about_not_knowing(self):
        """探针判断不了时返回 None，而不是猜一个 False 去误报。"""
        self.assertIsNone(pdfimport._has_text_layer(str(Path(self._tmp) / "nope.pdf")))
