# -*- coding: utf-8 -*-
"""两栏排版页上，右栏正文被当成表格的一列粘进来。

真 PDF 463 第 5 页是「左栏一张密集数值表 + 右栏正文（含致谢与参考文献条目）」。
`_column_bands` **找到了中缝**（x=[292,306]，宽 14pt，居中偏差 0.2pt，
右栏填充率 0.95——明摆着的正文栏），却被最后那道比例闸毙掉：
右栏 61 行 < `GUTTER_MIN_SIDE_RATIO`(0.25) × 全页 290 行 = 72。

比例闸默认两栏行数相当。可**一栏是密集表格、另一栏是正文**恰恰是最该分栏的
情形，也恰恰是两侧行数最悬殊的情形——它把自己最该管的那类页面放过去了。
于是右栏正文成了表格的第 6 列：

    ['-', '-', '8.77', '10.18', '10.30', 'national Conference on Learning Re…']
    ['3000', '0.0', '20.51', '20.76', '20.29', 'Bhattacharyya. 2014. Supertag Base…']

今天这只是往表格块里塞垃圾（正文块里还有一份完整的）。但它挡在两件事前面：
① **表格摘要**——把这种表喂给模型，会得到一段自信的胡话；
② **表格/正文去重**——一旦按检出结果把表格行从正文里摘掉，这 38 段正文碎片
   就真的从检索里消失了。所以这条必须先修。

修法：绝对行数下限（`GUTTER_MIN_SIDE_LINES`）照旧一票否决；比例不够时，
只要这一侧自己就是一栏撑满宽度的正文（`_fill_ratio >= GUTTER_PROSE_FILL`）就认。
表格的列永远撑不满栏宽，冒名不进来——这一点原本就是本模块最后一道闸的判据。

实测口径（26 篇真 PDF，改前 vs 改后）：检出表数 102 → 102（一张没多没少），
散文列 29 → 27，受影响论文**只有 463 一篇**（散文列 2 → 0，38 个正文碎片移出表格）。
剩下的 27 条散文列是合法的说明列（`Category | Method | Key Contribution` 这种），
不是缺陷——所以这条修复是外科手术式的，不是把阈值一撸了之。
"""
import unittest
from pathlib import Path

from papernest import config, tables

PDF_463 = config.ROOT / "data" / "pdf" / "463.pdf"


def _line(x0, x1, y, text="x"):
    return {"x0": float(x0), "x1": float(x1), "y0": float(y), "y1": float(y + 8),
            "text": text, "size": 9.0, "spans": []}


class ColumnBandsSplitAsymmetricPagesTests(unittest.TestCase):
    """不依赖真 PDF：合成一页「左栏密集表 + 右栏正文」，行数悬殊。"""

    def _page(self, n_table_rows: int, n_prose_lines: int):
        lines = []
        y = 100.0
        for i in range(n_table_rows):          # 左栏：4 列窄单元格，撑不满栏宽
            for x0 in (81, 130, 180, 230):
                lines.append(_line(x0, x0 + 24, y))
            y += 10
        y = 100.0
        for i in range(n_prose_lines):         # 右栏：撑满栏宽的正文行
            lines.append(_line(312, 525, y, "a full width sentence of prose text"))
            y += 12
        return lines

    def test_a_dense_table_column_does_not_swallow_the_prose_column(self):
        """左 229 行 / 右 61 行——正是 463 第 5 页的形状。"""
        bands = tables._column_bands(self._page(57, 61))   # 57*4 = 228 行
        self.assertGreaterEqual(
            len(bands), 2,
            "两栏没被分开：右栏正文会被当成左栏那张表的额外一列粘进去")
        widest = max(bands, key=lambda b: max(l["x1"] for l in b)
                     - min(l["x0"] for l in b))
        self.assertLess(
            max(l["x1"] for l in widest) - min(l["x0"] for l in widest), 300,
            "分出来的组仍横跨整页，说明中缝没起作用")

    def test_a_stray_handful_of_lines_is_still_not_a_column(self):
        """放宽的是**比例**，不是绝对下限——零星几行不能凭填充率高就算一栏。"""
        bands = tables._column_bands(self._page(57, tables.GUTTER_MIN_SIDE_LINES - 1))
        self.assertEqual(len(bands), 1,
                         f"少于 {tables.GUTTER_MIN_SIDE_LINES} 行的一侧不该被认成栏")

    def test_a_full_page_table_is_not_split_by_its_own_gutter(self):
        """原注释警告过的反向风险：占满整页的多列表被自己的列间空白劈成两半，
        每半只剩两列、双双被 GEOM_MIN_COLS 毙掉，整张表凭空消失。"""
        lines, y = [], 100.0
        for _ in range(40):                    # 6 列全页宽，列间有空白但没有正文栏
            for x0 in (81, 150, 220, 320, 390, 460):
                lines.append(_line(x0, x0 + 40, y))
            y += 10
        self.assertEqual(len(tables._column_bands(lines)), 1,
                         "整张表被它自己的列间空白劈开了")


@unittest.skipUnless(PDF_463.exists(), "需要真 PDF 463")
class RealTwoColumnPageTests(unittest.TestCase):
    """真 PDF 回归：463 第 5 页不许再有正文碎片混进表格。"""

    @classmethod
    def setUpClass(cls):
        cls.tabs = [t for t in tables.detect_tables(str(PDF_463))
                    if t["page_no"] == 5]

    def test_no_prose_column_survives(self):
        offenders = []
        for t in self.tabs:
            for j in range(t["n_cols"]):
                vals = [(r[j] or "").strip() for r in t["rows"] if j < len(r)]
                vals = [v for v in vals if v]
                if len(vals) < 3:
                    continue
                prose = [v for v in vals if len(v) >= 25 and len(v.split()) >= 4
                         and not tables._numericish(v)]
                if len(prose) / len(vals) >= 0.6:
                    offenders.append((t["n_rows"], t["n_cols"], j, prose[:2]))
        self.assertEqual(offenders, [],
                         f"表格里仍有整列是正文：{offenders}")

    def test_the_table_no_longer_spans_both_columns(self):
        """粘进来的那一列会把 bbox 撑到整页宽（81→525）。"""
        for t in self.tabs:
            x0, _, x1, _ = t["bbox"]
            self.assertLess(x1 - x0, 340,
                            f"表框宽 {x1-x0:.0f}pt，横跨了整页两栏：{t['n_rows']}x{t['n_cols']}")

    def test_the_acknowledgements_did_not_become_table_cells(self):
        """最刺眼的那条：致谢整句成了一个单元格。"""
        cells = " ".join(tables._squash(c) for t in self.tabs
                         for r in t["rows"] for c in r)
        self.assertNotIn("wewouldliketothank", cells)


if __name__ == "__main__":
    unittest.main()
