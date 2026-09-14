# -*- coding: utf-8 -*-
"""没有文本层的 PDF，原来是一条死路。

它走完整条入库流程会得到 `papers=1 / pages=0 / chunks=0`，而提示只能说
「本项目不做 OCR，请自己找外部工具转成可搜索 PDF」。`papernest/ocr.py` 把这条路接上。

## 引擎是实测选的，不是拍的

真库里**一份扫描件都没有**（26 篇 233 页，0 字页与 <50 字页各 0 个，
98.3% 的页超过 300 字；此前笔记里「1.pdf / 477.pdf 无文本层」经实测不成立）。
没有对象就没法评，所以改用**真论文合成扫描件**：按 DPI 渲染成位图、去掉文本层，
原来的文本层就是逐字 ground truth（`tools/ocr_bench.py`）。

两篇论文各 4 页，200 DPI：

    引擎                      词面召回      词面精确率     token/页   耗时/页
    RapidOCR（本地）          91.3 / 90.8%  85.9 / 86.4%      0        17s
    qwen-vl-ocr（云端）       92.7 / 93.7%  95.8 / 95.0%   ~4850     13~24s

差距主要在**精确率 +9~10pt**，而且不是「云端在编字」：是 RapidOCR 按视觉顺序
吐碎片，把 `ad-\\ndressed` 拆成两个原文里没有的词面，云端会还原阅读顺序、
把连字拼回去。对本项目这直接等于检索命中率。

**给 RapidOCR 加到 300 DPI 没有帮助**（召回 91.2%、精确率 85.4%），耗时反而 1.7 倍
——差距不是分辨率造成的。

**这些数字是上界**：合成件没有倾斜、噪点、装订阴影、印章，真扫描件只会更差。

## 这个文件钉住什么

① 默认关闭——云端一页约 4850 token、本地 17s/页，都不该在用户没说话时发生；
② OCR 出来的文本要标出来，它不是文本层，有错字；
③ 单页失败不能丢掉整份（扫描件里夹一张纯图插页很常见）；
④ 引擎实现只有一份——评测脚本从 `papernest.ocr` 复用，不许自己再写一遍。
   本仓刚因为「同名函数定义两份、后定义的旧实现才生效」栽过一次（`llm.extract_json`），
   评测脚本和生产各写一份是同一个坑的更隐蔽版本：评出来的数字将不代表生产行为。
"""
import unittest
from unittest import mock

from papernest import config, ocr


class DefaultsToOffTests(unittest.TestCase):
    """云端要花钱、本地要 17s/页，两样都不该在用户没说话时发生。"""

    def test_mode_is_off_unless_asked(self):
        import os
        if os.environ.get("PAPERNEST_OCR"):
            self.skipTest("本机显式开了 OCR")
        self.assertEqual(ocr.MODE, "off")
        self.assertFalse(ocr.enabled())

    def test_ocr_pdf_refuses_when_disabled(self):
        """**显式抛**，不是静默返回空——静默会让人以为 OCR 做过了。"""
        with mock.patch.object(ocr, "MODE", "off"):
            with self.assertRaises(ocr.OcrUnavailable) as cm:
                ocr.ocr_pdf("whatever.pdf")
        self.assertIn("PAPERNEST_OCR", str(cm.exception),
                      "报错没说怎么打开它")

    def test_available_reports_without_calling_anything(self):
        a = ocr.available()
        self.assertIn("local", a)
        self.assertIn("cloud", a)
        self.assertEqual(a["mode"], ocr.MODE)


class TheBillIsHonestTests(unittest.TestCase):
    def test_cloud_estimate_uses_the_measured_rate(self):
        """实测 8 页 38,731 token。按「一页几百 token」估会低估一个数量级。"""
        self.assertGreater(ocr.MEASURED_CLOUD_TOKENS_PER_PAGE, 1000)
        e = ocr.estimate(20, "cloud")
        self.assertEqual(e["est_total_tokens"],
                         20 * ocr.MEASURED_CLOUD_TOKENS_PER_PAGE)

    def test_local_costs_nothing_but_takes_time(self):
        e = ocr.estimate(20, "local")
        self.assertEqual(e["est_total_tokens"], 0)
        self.assertGreater(e["est_seconds"], 0,
                           "本地不花钱不等于免费——17s/页也要先告诉人")

    def test_default_dpi_is_the_measured_one(self):
        """300 DPI 实测对本地引擎没有改善、耗时 1.7 倍，所以默认是 200。"""
        self.assertEqual(ocr.DPI, 200)


class OcrTextIsLabelledTests(unittest.TestCase):
    def test_mark_says_it_is_not_the_text_layer(self):
        self.assertIn("OCR", ocr.MARK)
        self.assertIn("非文本层", ocr.MARK)

    def test_pages_carry_the_mark(self):
        with mock.patch.object(ocr, "MODE", "local"), \
             mock.patch.object(ocr, "render_page", return_value=b"png"), \
             mock.patch.object(ocr, "ocr_local", return_value="识别出来的正文"), \
             mock.patch("pymupdf.open") as op:
            op.return_value.page_count = 2
            pages = ocr.ocr_pdf("x.pdf")
        self.assertEqual(len(pages), 2)
        for _, text in pages:
            self.assertTrue(text.startswith(ocr.MARK))


class OnePageFailureDoesNotSinkTheFileTests(unittest.TestCase):
    """扫描件里夹一张纯图插页很常见，为它丢掉整份不值。"""

    def test_a_failing_page_is_skipped(self):
        calls = []

        def flaky(png):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("这页崩了")
            return "正文"

        with mock.patch.object(ocr, "MODE", "local"), \
             mock.patch.object(ocr, "render_page", return_value=b"png"), \
             mock.patch.object(ocr, "ocr_local", flaky), \
             mock.patch("pymupdf.open") as op:
            op.return_value.page_count = 3
            pages = ocr.ocr_pdf("x.pdf")
        self.assertEqual([p for p, _ in pages], [1, 3])

    def test_a_missing_engine_still_raises(self):
        """单页失败要跳过，但**引擎压根用不了**是配置问题，必须冒出去。"""
        with mock.patch.object(ocr, "MODE", "local"), \
             mock.patch.object(ocr, "render_page", return_value=b"png"), \
             mock.patch.object(ocr, "ocr_local",
                               side_effect=ocr.OcrUnavailable("没装")), \
             mock.patch("pymupdf.open") as op:
            op.return_value.page_count = 3
            with self.assertRaises(ocr.OcrUnavailable):
                ocr.ocr_pdf("x.pdf")

    def test_blank_pages_are_dropped(self):
        with mock.patch.object(ocr, "MODE", "local"), \
             mock.patch.object(ocr, "render_page", return_value=b"png"), \
             mock.patch.object(ocr, "ocr_local", return_value="   "), \
             mock.patch("pymupdf.open") as op:
            op.return_value.page_count = 2
            self.assertEqual(ocr.ocr_pdf("x.pdf"), [])


class IngestPathTests(unittest.TestCase):
    """接进入库路径的方式：补救成功要说清楚，补救不了要退回原来的大声报错。"""

    def test_the_warning_now_offers_a_way_out(self):
        from papernest import pdfimport
        w = pdfimport.NO_TEXT_LAYER_WARNING
        self.assertIn("PAPERNEST_OCR", w, "还在说「本项目不做 OCR」")
        self.assertIn("默认关闭", w, "没说清楚它不会自己跑起来")

    def test_the_success_note_admits_ocr_has_errors(self):
        from papernest import pdfimport
        note = pdfimport.OCR_APPLIED_NOTE
        self.assertIn("错字", note,
                      "没说 OCR 有错字，用户会拿它当原文逐字引用")

    def test_the_success_note_explains_the_metadata_warnings(self):
        """元数据在 OCR **之前**抽，所以扫描件必然带一串「抽取失败」。

        实测一份合成扫描件入库拿到 7 条 warning，其中 6 条是元数据抽取失败。
        不解释的话用户会读成两个独立故障，进而怀疑 OCR 也没成。
        """
        from papernest import pdfimport
        note = pdfimport.OCR_APPLIED_NOTE
        self.assertIn("预期", note)
        self.assertIn("fix-meta", note, "没告诉人怎么补元数据")

    def test_rescue_is_a_no_op_when_ocr_is_off(self):
        from papernest import pdfimport
        with mock.patch.object(ocr, "MODE", "off"):
            self.assertEqual(pdfimport._ocr_rescue("x.pdf", 1, []), 0)

    def test_rescue_reports_instead_of_raising(self):
        """补救路径炸了不该把整次上传拖下水——论文元数据已经有价值了。"""
        from papernest import pdfimport
        warns = []
        with mock.patch.object(ocr, "enabled", return_value=True), \
             mock.patch.object(ocr, "ocr_pdf", side_effect=RuntimeError("boom")):
            n = pdfimport._ocr_rescue("x.pdf", 1, warns)
        self.assertEqual(n, 0)
        self.assertTrue(any("OCR" in w and "boom" in w for w in warns), warns)


class OneImplementationOnlyTests(unittest.TestCase):
    """④ 评测脚本必须复用生产的引擎，不许自己写一份。

    本仓刚栽过：`llm.extract_json` 同名定义了两份，后定义的旧实现才生效、
    新版成了死代码。评测脚本和生产各写一份 OCR 调用是同一个坑的更隐蔽版本
    ——评出来的数字将不再代表生产行为。
    """

    def test_bench_imports_the_production_module(self):
        src = (config.ROOT / "tools" / "ocr_bench.py").read_text(encoding="utf-8")
        self.assertIn("from papernest import config, ocr", src)

    def test_bench_does_not_construct_its_own_rapidocr(self):
        src = (config.ROOT / "tools" / "ocr_bench.py").read_text(encoding="utf-8")
        self.assertNotIn("RapidOCR()", src,
                         "评测脚本又自己实例化了一个引擎")

    def test_bench_shares_the_prompt(self):
        """提示词不同 = 评的不是生产行为。"""
        src = (config.ROOT / "tools" / "ocr_bench.py").read_text(encoding="utf-8")
        self.assertIn("ocr._PROMPT", src)


if __name__ == "__main__":
    unittest.main()


class KnownWeakModelsAreFlaggedTests(unittest.TestCase):
    """`qwen3.5-ocr` 会**整片跳过页面里的表格**，而且不吭一声。

    账户欠费后只剩免费额度，`qwen3.5-ocr` 是仅存能调的几个模型之一，所以专门测了它。
    整页口径看着还行（463 召回 86.5%、480 召回 92.8%，精确率 95~96%），
    **按数值词面单看就露馅了**：

        463 第 3 页（正文 + 两张结果表）  数值召回 21.5%(14/65)   本地 RapidOCR 100%(65/65)
        463 第 5 页（几乎整页是表）      数值召回 100%(100/100)  本地 RapidOCR  98%(98/100)

    第 3 页的输出里 `Table` / `BLEU` / `LeBLEU` / `|` **各出现 0 次**——它把正文转完
    就停在脚注，`finish_reason=stop`（不是截断），**同一张图跑三次一字不差**。
    表格占满整页时它躲不开，正文里夹着表就直接不转。

    这是本仓一直在打的那类失败：**静默丢内容**。扫描论文的价值大半在结果表里。
    """

    def test_the_weak_model_is_listed(self):
        self.assertIn("qwen3.5-ocr", ocr.WEAK_ON_TABLES)

    def test_selecting_it_produces_a_warning(self):
        w = ocr.model_warning("qwen3.5-ocr")
        self.assertIn("表格", w)
        self.assertIn("21.5%", w, "没给出实测数字，读的人没法判断严重程度")
        self.assertIn("qwen-vl-ocr-latest", w, "没告诉人该换成什么")
        self.assertIn("local", w, "没给出不花钱的退路")

    def test_a_good_model_gets_no_warning(self):
        self.assertEqual(ocr.model_warning("qwen-vl-ocr-latest"), "")

    def test_the_default_is_not_a_weak_model(self):
        self.assertNotIn(ocr.CLOUD_MODEL, ocr.WEAK_ON_TABLES,
                         "默认模型是已知会丢表格的那个")

    def test_the_warning_reaches_the_caller(self):
        """光在文档里写没用——真跑的时候得看见。"""
        seen = []
        with mock.patch.object(ocr, "MODE", "cloud"), \
             mock.patch.object(ocr, "CLOUD_MODEL", "qwen3.5-ocr"), \
             mock.patch.object(ocr, "render_page", return_value=b"png"), \
             mock.patch.object(ocr, "ocr_cloud", return_value="正文"), \
             mock.patch("pymupdf.open") as op:
            op.return_value.page_count = 1
            ocr.ocr_pdf("x.pdf", progress=lambda f, m: seen.append(m))
        self.assertTrue(any("表格" in m for m in seen),
                        f"跑起来没有提示这个模型会丢表格：{seen}")
