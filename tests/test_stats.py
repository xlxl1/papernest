"""配对显著性检验：小评测集上「方向为正」与「显著」必须分得开。

项目自己定的规矩是「32 题里一道题就是 0.031，小评测集必须做显著性检验」，
但此前仓库里没有任何检验实现，README 报出去的 p 值没有出处也重算不出来。
这些用例钉住三件事：结果可复现、不显著时不许说显著、逐题无变化时不许编 p 值。
"""
from __future__ import annotations

import unittest

from papernest.stats import describe, sign_flip_test


class SignFlipTests(unittest.TestCase):

    def test_identical_scores_report_no_difference(self):
        s = [1.0, 0.0, 0.5] * 10
        r = sign_flip_test(s, list(s))
        self.assertEqual(r["mean_diff"], 0.0)
        self.assertEqual(r["p_value"], 1.0)
        self.assertFalse(r["significant"])
        self.assertEqual(r["n_changed"], 0)

    def test_one_flipped_item_out_of_32_is_not_significant(self):
        """这正是项目定这条规矩的场景：32 题里一道题 = 0.031，绝不能报成显著。"""
        before = [0.0] + [1.0] * 31
        after = [1.0] + [1.0] * 31
        r = sign_flip_test(before, after)
        self.assertAlmostEqual(r["mean_diff"], 0.0312, places=3)
        self.assertFalse(r["significant"], f"一道题就报显著（p={r['p_value']}）")

    def test_a_large_consistent_improvement_is_significant(self):
        before = [0.0] * 20 + [1.0] * 12
        after = [1.0] * 32
        r = sign_flip_test(before, after)
        self.assertTrue(r["significant"])
        self.assertLess(r["p_value"], 0.01)

    def test_regression_is_detected_two_sided(self):
        """双侧检验：变差同样要被判显著，不能只对改善敏感。"""
        before = [1.0] * 32
        after = [0.0] * 20 + [1.0] * 12
        r = sign_flip_test(before, after)
        self.assertLess(r["mean_diff"], 0)
        self.assertTrue(r["significant"])

    def test_result_is_reproducible(self):
        before = [0.0, 1.0] * 16
        after = [1.0, 1.0] * 16
        self.assertEqual(sign_flip_test(before, after)["p_value"],
                         sign_flip_test(before, after)["p_value"])

    def test_p_value_is_never_zero(self):
        """加一平滑：报 p=0 是在宣称「绝无可能」，重采样给不出这个结论。"""
        r = sign_flip_test([0.0] * 200, [1.0] * 200)
        self.assertGreater(r["p_value"], 0)

    def test_length_mismatch_is_an_error(self):
        with self.assertRaises(ValueError):
            sign_flip_test([1.0, 0.0], [1.0])

    def test_empty_input_does_not_crash(self):
        r = sign_flip_test([], [])
        self.assertEqual(r["n"], 0)
        self.assertFalse(r["significant"])

    def test_describe_says_not_significant_out_loud(self):
        before = [0.0] + [1.0] * 31
        r = sign_flip_test(before, [1.0] * 32)
        self.assertIn("不显著", describe(r, "自建集 Recall@5"))

    def test_describe_reports_how_many_items_changed(self):
        r = sign_flip_test([0.0, 0.0] + [1.0] * 30, [1.0, 1.0] + [1.0] * 30)
        self.assertIn("2 题发生变化", describe(r))


if __name__ == "__main__":
    unittest.main()
