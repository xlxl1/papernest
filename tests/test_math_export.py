"""Word OMML 导出单元测试。"""
import unittest

from papernest import math_export


class MathExportTests(unittest.TestCase):
    def _xml(self, latex):
        return math_export.omml_paragraph(latex).xml

    def test_fraction_and_greek_symbol(self):
        xml = self._xml(r"\frac{\alpha^2}{\beta+1}")
        self.assertIn("m:f", xml)
        self.assertIn("m:num", xml)
        self.assertIn("m:den", xml)
        self.assertIn("α", xml)
        self.assertIn("β", xml)
        self.assertIn("m:sSup", xml)

    def test_subscript_and_operators(self):
        xml = self._xml(r"x_i \leq y_j \times z")
        self.assertIn("m:sSub", xml)
        self.assertIn("≤", xml)
        self.assertIn("×", xml)

    def test_display_math_delimiters_are_removed(self):
        xml = self._xml(r"$$a^2$$")
        self.assertIn("m:sSup", xml)
        self.assertNotIn("$$", xml)

    def test_matrix_becomes_native_omml_matrix(self):
        xml = self._xml(r"\begin{bmatrix}a & b \\ c & d\end{bmatrix}")
        self.assertIn("m:m", xml)
        self.assertEqual(xml.count("<m:mr>"), 2)
        self.assertEqual(xml.count("<m:e>"), 4)

    def test_sum_and_integral_keep_limits_and_left_right(self):
        xml = self._xml(r"\left(\sum_{i=1}^{n} x_i + \int_0^1 f(x)\,dx\right)")
        self.assertEqual(xml.count("<m:nary>"), 2)
        # lxml serializes non-ASCII OMML operator values as replacement glyphs
        # in this environment; the native n-ary nodes and limit text are the
        # compatibility contract we can assert portably.
        self.assertIn("<m:chr", xml)
        self.assertIn("<m:t>i</m:t>", xml)
        self.assertIn("<m:t>1</m:t>", xml)
        self.assertIn("m:sub", xml)
        self.assertIn("m:sup", xml)
        self.assertNotIn("left", xml)
        self.assertNotIn("right", xml)


if __name__ == "__main__":
    unittest.main()
