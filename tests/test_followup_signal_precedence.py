# -*- coding: utf-8 -*-
"""追问判定的优先级：指代标记不许短路检索信号判断。

`is_followup` 原来是「命中 `_REF_MARKERS` 就立刻 return True」，而那张表里混着两类词：
光杆代词（它 / 这篇 / 上述）和**既是指代词、又是常见学术句式**的词
（继续 / 展开 / 详细讲 / 对比一下 / 这个 / 这些）。中文用户问自己库里的英文论文时，
「对比一下 ZF 和 MMSE 检测器」「展开说说 RIS 辅助的宽带信道估计」都是自足查询，
却全被判成追问、被拼上上一轮的问题去检索。

`chat.py` 的注释声称检索信号判断替代了「`len(q)<=12` 一律判追问」的一刀切，
但它排在标记检查后面，**对带标记的句子根本不生效**。

真库实测（问句里的实词全部取自库内真实论文标题）：14 条自足查询误判 8 条 = 57%；
其中「对比一下 ZF 检测器和 MMSE 检测器…」与「这些年 Massive MIMO 导频污染有哪些解法」
被拼上上一轮问题后，原查询 top10 的存活率是 **0/10**——整个候选集换成了上一轮的主题，
而用户拿到的是自信引着 [n] 的错主题回答，全程静默。
"""
import unittest

from papernest import chat


class FollowupSignalPrecedenceTests(unittest.TestCase):
    PRIOR = ["近场信道估计的最新进展"]

    #: 实词出处（data/papernest.db，只读核对过）：
    #: #166 RIS-Aided Massive MIMO / ZF Detectors、#49 Wideband Near-Field XL-MIMO、
    #: #218 IRS Aided mmWave、#124 Mixed-Resolution ADCs、#25 IP-MCMC-PF 目标跟踪、
    #: #30 MIMO SC-FDMA 半盲信道估计、#143 Pilot Contamination Survey
    SELF_CONTAINED = (
        "对比一下 ZF 检测器和 MMSE 检测器在 Massive MIMO 下的性能",
        "详细讲讲 XL-MIMO 的近场信道协方差估计",
        "展开说说 RIS 辅助 mmWave Massive MIMO 的宽带信道估计",
        "除了深度学习，还有其它 Massive MIMO 信道估计方法吗",
        "深度学习之前，混合分辨率 ADC 的信道估计是怎么做的",
        "这个 IP-MCMC-PF 目标跟踪方法用的约束知识是什么",
        "继续讲 MIMO SC-FDMA 系统的半盲信道估计新方法",
        "这些年 Massive MIMO 导频污染有哪些解法",
    )

    def test_marker_must_not_short_circuit_retrieval_signal(self):
        """这些句子自己就带检索信号，标记不该把它们短路成追问。"""
        for q in self.SELF_CONTAINED:
            with self.subTest(q=q):
                self.assertTrue(chat._has_retrieval_signal(q), q)
                self.assertFalse(chat.is_followup(q, self.PRIOR), q)

    def test_self_contained_query_is_not_rewritten(self):
        """判定错了还不够——真正的危害是检索串被污染，这里钉住最终产物。"""
        q = "对比一下 ZF 检测器和 MMSE 检测器在 Massive MIMO 下的性能"
        search_q, note = chat.rewrite_query(q, self.PRIOR)
        self.assertEqual(search_q, q, "自足查询被拼上了上一轮的问题")
        self.assertIsNone(note)

    def test_explicit_source_reference_is_always_a_followup(self):
        """显式点名某条来源：漏判最贵（按字面去检索「第 2 篇」），无条件算追问。"""
        for q in ("[3] 这篇的实验设置是什么", "第 2 篇怎么做的",
                  "第二篇讲得更细一点", "[1] 和 [2] 在信道估计上的差别"):
            with self.subTest(q=q):
                self.assertTrue(chat.is_followup(q, self.PRIOR), q)

    def test_bare_pronoun_is_still_a_followup(self):
        """正题不能被改坏：光杆代词与「什么都不剩」的句子仍要算追问。"""
        for q in ("它的局限是什么", "它们的复杂度呢", "这篇论文的方法是什么",
                  "上述方法的假设是什么", "刚才那篇用的数据集是什么",
                  "展开讲讲", "继续", "为什么？"):
            with self.subTest(q=q):
                self.assertTrue(chat.is_followup(q, self.PRIOR), q)

    def test_demonstrative_plus_generic_noun_is_still_a_followup(self):
        """「这些论文」「这个方法」——剩下的实词看着像检索信号，指的却是上一轮的东西。

        这是**修这条 bug 时踩出来的新退化**：把「这个/这些」整体降成软标记之后，
        「这些论文里哪个效果最好」抠掉标记还剩「论文/效果」，就被判成了自足查询。
        所以指示词**紧跟通用名词**时要走硬标记；跟着具体型号时（「这个 IP-MCMC-PF
        目标跟踪方法」）不受影响，那一条由上面的用例守着。
        """
        for q in ("这些论文里哪个效果最好", "这个方法的复杂度呢",
                  "那这个方法的复杂度呢", "这些实验的设置是什么"):
            with self.subTest(q=q):
                self.assertTrue(chat.is_followup(q, self.PRIOR), q)

    def test_bare_intensifier_is_a_followup(self):
        """「再展开一点」「再详细一点」没有任何检索信号，只能是追问。"""
        for q in ("再展开一点", "再详细一点"):
            with self.subTest(q=q):
                self.assertTrue(chat.is_followup(q, self.PRIOR), q)

    def test_english_pronouns_keep_word_boundaries(self):
        """英文走词边界这条既有性质不能被这次改动带塌。"""
        self.assertTrue(chat.is_followup("compare these two", self.PRIOR))
        self.assertFalse(
            chat.is_followup("多模态大模型的 limitations 有哪些", self.PRIOR),
            "'it' 又子串命中 limitations 了")


if __name__ == "__main__":
    unittest.main()
