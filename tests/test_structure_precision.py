"""章节识别的**精确率**：只靠一个信号不足以判定标题。

这条规则是拿真实数据反推的。一篇 26 页的真实综述在门槛=2 时被识别出 **125 个章节**，
其中二十余个是分类图里又粗又短的图元标签（Self-Learning / Centralized /
Memory Mechanism / Hybrid Architecture …）——它们只命中 `bold` 一个信号就达标了。
而 bold 在图表标签、表头、行内强调里遍地都是。把门槛提到 3 之后是 29 个，
层级也对上了真实目录（1 → 2.1/2.2/2.3 → 3.1/3.2 …）。

合成 PDF 天然不会有分类图，所以原有 44 个用例一个都没照出这个问题——
这里专门把那个版面形态造出来。
"""
import shutil
import tempfile
import unittest
from pathlib import Path

import pymupdf

from papernest import structure

BODY = 9.6


def _builder(tmp: Path, name: str):
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    state = {"y": 72.0, "page": page}

    def line(text, size=BODY, bold=False, x=72.0):
        if state["y"] + size + 6 > 842 - 60:
            state["page"] = doc.new_page(width=595, height=842)
            state["y"] = 72.0
        state["page"].insert_text((x, state["y"]), text, fontsize=size,
                                  fontname="hebo" if bold else "helv")
        state["y"] += size + 4

    def save():
        p = tmp / name
        doc.save(str(p))
        doc.close()
        return str(p)

    return line, save


class HeadingPrecisionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="papernest_struct_prec_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def _taxonomy_pdf(self) -> str:
        """一页正常章节 + 一页「分类图」：图里全是又粗又短的标签。"""
        line, save = _builder(self.tmp, "taxonomy.pdf")
        line("1 Introduction", size=13.0, bold=True)
        line("Agents need a systematic methodology for construction and evaluation.")
        line("This paragraph is ordinary body text with nothing special about it.")
        # 分类图里的图元标签：又粗又短，但都不是章节
        for label in ("Self-Learning", "Multi-agent Co-Evolution", "External Resource",
                      "Centralized", "Decentralized Collaboration", "Hybrid Architecture",
                      "Profile Definition", "Memory Mechanism", "Planning Capability",
                      "Action Execution"):
            line(label, size=BODY, bold=True)
        line("2 Agent Methodology", size=13.0, bold=True)
        line("We organise the literature along three axes described below.")
        line("2.1 Agent Construction", size=11.0, bold=True)
        line("Construction covers profile, memory, planning and action execution.")
        return save()

    def test_bold_short_labels_are_not_sections(self):
        secs = structure.detect_sections(self._taxonomy_pdf())
        titles = {s["title"] for s in secs}
        for label in ("Self-Learning", "Centralized", "Memory Mechanism",
                      "Hybrid Architecture", "Action Execution"):
            self.assertNotIn(label, titles, f"图元标签 {label!r} 被当成了章节")

    def test_real_sections_survive_the_tighter_threshold(self):
        """收紧门槛不能把真章节一起误杀——这才是这次改动的风险所在。"""
        secs = structure.detect_sections(self._taxonomy_pdf())
        titles = {s["title"] for s in secs}
        for want in ("1 Introduction", "2 Agent Methodology", "2.1 Agent Construction"):
            self.assertIn(want, titles)

    def test_numbered_subsection_keeps_its_level(self):
        secs = structure.detect_sections(self._taxonomy_pdf())
        by_title = {s["title"]: s for s in secs}
        self.assertEqual(by_title["2 Agent Methodology"]["level"], 1)
        self.assertEqual(by_title["2.1 Agent Construction"]["level"], 2)

    def test_single_signal_never_qualifies(self):
        """把规则本身钉住：任何只命中一个信号的块都不该成为标题。

        版面信号（font_size / bold）各 2 分、文本信号各 1 分、门槛 3 分，
        所以「一个信号」最多 2 分，永远不够。"""
        secs = structure.detect_sections(self._taxonomy_pdf())
        for s in secs:
            signals = [m for m in s["matched_by"] if m != "short_line"]
            self.assertGreaterEqual(len(signals), 2,
                                    f"{s['title']!r} 只靠 {signals} 就被判成标题了")

    def test_precision_on_a_figure_heavy_page(self):
        """量化一下：10 个图元标签一个都不许进，真章节 3 个一个都不许丢。"""
        secs = structure.detect_sections(self._taxonomy_pdf())
        self.assertEqual(len(secs), 3)


if __name__ == "__main__":
    unittest.main()
