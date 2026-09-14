# -*- coding: utf-8 -*-
"""离线 matrix 的两条缺陷：verified 构造性恒真 + COLUMN_KEYWORDS 裸子串。

两条是同一个「词面即证据」假设的两半，**相乘**才有杀伤力：
裸子串抽出一句与该列无关的原文，恒真的 verified 保证它一定拿到「已核验」，
导出层因而不打任何警告——读者看到的是「一句原文 +【p.1】+ 脚注回取校验通过率 100%」，
一个语义上完全错误的格子，穿着满分证据的外衣。

真库实测（45 篇有全文的论文 × 6 默认列）：
  · 176 个非空格子里，`verify_quote` 真正做过工作的只有 4 个（全走卡片路径），
    98.3% 这个自证分数的分子里 171/173 是恒真产物；
  · 穷举证明恒真：566 个非空页切出的 13876 句 + 478 篇摘要的 3616 句，
    `verify_quote(sent, 同一页)` 全为 True，反例 0；
  · 45 篇里 38 篇的页文本含 BIBREF 占位符，18/172 个离线格子是被 'f1' ⊂ 'BIBREF1'
    这类误命中抽出来的（「评测指标」列 17/42 = 40.5%），且因为占位符几乎总在第 1 页
    Introduction，坏命中系统性地压在 p.1 上。

夹具是真库论文 485（Inexpensive Domain Adaptation of Pretrained Language Models）的
**逐字**页文本——不是自造的：合成夹具（干净的、不含 BIBREF 占位符的页）正好会掩盖这两条。
"""
from papernest import matrix
from tests.test_matrix import MatrixTestBase

# ── 真库 paper 485 的逐字页文本（SELECT text FROM pages WHERE paper_id=485）──
REAL_P1 = ("Introduction\n"
           "Pretrained Language Models (PTLMs) such as BERT BIBREF1 have spearheaded "
           "advances on many NLP tasks.")
REAL_P13 = (
    "Experiment 1: Biomedical NER ::: Results and discussion\n"
    "Table TABREF7 (bottom) shows entity-level precision, recall and F1. For ease of "
    "visualization, Figure FIGREF13 shows what portion of the BioBERT - BERT F1 delta "
    "is covered. We improve over general-domain BERT on all tasks with varying effect "
    "sizes.")
REAL_ABSTRACT = (
    "Domain adaptation of Pretrained Language Models (PTLMs) is typically achieved by "
    "pretraining on in-domain text. While successful, this approach is expensive in "
    "terms of hardware, runtime and CO_2 emissions.")
BIBREF_SENT = ("Pretrained Language Models (PTLMs) such as BERT BIBREF1 have "
               "spearheaded advances on many NLP tasks.")


class OfflineEvidenceRegressions(MatrixTestBase):
    def _metric_cell(self):
        pid = self._add("arxiv:485real", "Inexpensive Domain Adaptation of PTLMs",
                        REAL_ABSTRACT, card=None,
                        pages={1: REAL_P1, 13: REAL_P13})
        m = matrix.extract_offline([pid], ["metric"])
        return m, m["rows"][0]["cells"]["metric"]

    # ── (2) COLUMN_KEYWORDS 裸子串：'f1' 命中引文占位符 'BIBREF1' ──

    def test_bibref_placeholder_does_not_hijack_the_metric_column(self):
        """'f1' 不许命中 'BIBREF1'：那句话讲的是 BERT 的来历，不是评测指标。"""
        _m, cell = self._metric_cell()
        self.assertNotEqual(cell["value"], BIBREF_SENT)
        self.assertNotIn("BIBREF1", cell["value"] or "")
        # 该论文真正讲指标的句子在 p.13
        self.assertEqual(cell["page"], 13)
        self.assertIn("precision, recall and F1", cell["value"])

    def test_word_form_growth_is_still_matched(self):
        """右边界不能挡：真库 39 个 dataset 格子里 38 个命中的是 'datasets'。"""
        self.assertTrue(matrix._kw_re("dataset").search("we release two datasets"))
        self.assertTrue(matrix._kw_re("limitation").search("known limitations are"))
        # 但前缀被更长的 token 吞掉就是噪声
        self.assertIsNone(matrix._kw_re("f1").search("bert bibref1 spearheaded"))
        self.assertIsNone(matrix._kw_re("limitation").search("word delimitation here"))
        self.assertIsNone(
            matrix._kw_re("accuracy").search("cited as flickinger2011accuracy"))

    # ── (1) 离线 _locate 路径的 verified 构造性恒真 ──

    def test_offline_excerpt_does_not_claim_mechanical_verification(self):
        """原句是从该页逐字切出来的，回校验同一页恒真——这条路径不能报 verified。"""
        m, cell = self._metric_cell()
        self.assertEqual(cell["source"], "page:13")
        self.assertIsNone(cell["verified"],
                          "离线逐字摘录的 verified 必须是 None（不适用），不能是恒真的 True")
        # 恒真的格子不许进校验通过率的分子/分母
        self.assertEqual(m["cells_filled"], 1)
        self.assertEqual(m["cells_verified"], 0)
        self.assertEqual(m["cells_excerpted"], 1)
        self.assertEqual(m["cells_checkable"], 0)
        self.assertIsNone(m["verified_rate"],
                          "全表没有可校验的格子时，通过率应为 None（—），不是 100% 也不是 0%")

    def test_export_marks_the_excerpt_as_an_excerpt(self):
        """导出层必须如实标注：既不能什么都不标（读者会当成已核验），也不能标未核验。"""
        m, _cell = self._metric_cell()
        md = matrix.to_markdown(m)
        self.assertIn("【p.13 逐字摘录】", md)
        self.assertIn("逐字摘录", matrix.MARK_LEGEND)
        self.assertNotIn("回取校验通过率 100", md)

    # ── 根因存档：钉住「为什么不能在这条路径上做回取」 ──

    def test_verify_quote_against_its_own_page_is_constructively_true(self):
        """`_norm` 是逐字符映射，页文本的任何切片归一化后必是该页归一化结果的子串。

        这条**修法前后都绿**——它不是在测缺陷，是在存档缺陷的成因：
        只要 `_locate` 返回的还是原文切片，就不该拿 `verify_quote` 去「校验」它。
        """
        for page_text in (REAL_P1, REAL_P13, REAL_ABSTRACT):
            hay = matrix._norm(page_text)
            for sent in matrix._split_sentences(page_text):
                self.assertTrue(matrix.verify_quote(sent, hay), sent)
