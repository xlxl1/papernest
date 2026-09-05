"""对比矩阵的展示口径：长格子截断，但 CSV 保留全文。

真实数据反推的规则：库里的卡片字段常常是整段（「与我课题的关系」实测 600+ 字），
原样铺进 markdown 表格后一行占满整屏，「一眼横向对比」——这张表存在的唯一理由——
就没了。但 CSV 是拿去做分析的，在那里截断等于静默丢数据。
"""
import csv
import io
import unittest

from papernest import matrix

LONG = "这是一段很长的卡片字段。" * 30          # 360 字
SHORT = "简短结论。"


def _matrix(*values: str) -> dict:
    cols = [{"key": f"c{i}", "label": f"列{i}", "hint": ""} for i in range(len(values))]
    cells = {f"c{i}": {"value": v, "source": "card", "page": None, "verified": True,
                       "note": None}
             for i, v in enumerate(values)}
    return {"columns": cols,
            "rows": [{"paper_id": 1, "title": "测试论文", "year": 2024, "cells": cells}],
            "coverage": 1.0, "verified_rate": 1.0,
            "cells_total": len(values), "cells_filled": len(values),
            "cells_verified": len(values), "degraded": None, "errors": []}


class TruncationTests(unittest.TestCase):
    def test_markdown_truncates_long_cells(self):
        out = matrix.to_markdown(_matrix(LONG))
        row = [ln for ln in out.splitlines() if ln.startswith("| 1 |")][0]
        self.assertIn("…", row)
        self.assertLess(len(row), 400)

    def test_markdown_leaves_short_cells_alone(self):
        out = matrix.to_markdown(_matrix(SHORT))
        self.assertIn(SHORT, out)
        self.assertNotIn("…", out)

    def test_truncation_is_announced_only_when_it_happens(self):
        self.assertIn("已截断", matrix.to_markdown(_matrix(LONG)))
        self.assertNotIn("已截断", matrix.to_markdown(_matrix(SHORT)))

    def test_csv_keeps_full_text_by_default(self):
        """CSV 是拿去分析的，默认截断就是静默丢数据。"""
        out = matrix.to_csv(_matrix(LONG))
        rows = list(csv.reader(io.StringIO(out.lstrip("﻿"))))
        body = next(r for r in rows if r and r[0] == "1")
        self.assertIn(LONG, body[3])
        self.assertNotIn("…", body[3])

    def test_csv_can_opt_into_truncation(self):
        out = matrix.to_csv(_matrix(LONG), max_chars=60)
        self.assertIn("…", out)

    def test_marks_survive_truncation(self):
        """截断绝不能把出处一起切掉——那比长格子更糟。"""
        m = _matrix(LONG)
        m["rows"][0]["cells"]["c0"].update(page=7, verified=False)
        out = matrix.to_markdown(m)
        self.assertIn("【p.7 未核验】", out)

    def test_truncation_prefers_a_clause_boundary(self):
        text = "第一句在这里结束。" + "后面还有很多内容需要被截掉" * 20
        short, cut = matrix._shorten(text, 40)
        self.assertTrue(cut)
        self.assertTrue(short.endswith("…"))
        self.assertLessEqual(len(short), 41)

    def test_shorten_does_not_rewrite_text_it_does_not_truncate(self):
        """不截断时原样返回：顺手压掉换行会让 markdown 的 <br> 失效。"""
        text = "第一行\n第二行"
        self.assertEqual(matrix._shorten(text, 100), (text, False))

    def test_latex_truncates_and_stays_valid(self):
        out = matrix.to_latex(_matrix(LONG, LONG))
        self.assertIn("…", out)
        body = [ln for ln in out.splitlines() if ln.endswith(r"\\") and "测试论文" in ln]
        self.assertEqual(len(body), 1)
        self.assertEqual(body[0].count("&"), 4)     # 5 列 → 4 个分隔符，没被截散架

    def test_export_max_chars_override(self):
        self.assertIn("…", matrix.export(_matrix(LONG), "csv", max_chars=50))
        self.assertNotIn("…", matrix.export(_matrix(LONG), "markdown", max_chars=0))


if __name__ == "__main__":
    unittest.main()
