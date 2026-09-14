# -*- coding: utf-8 -*-
"""自建评测集的**成分**：有多少条其实是「标题查找」而不是检索。

审计说「32 道检索题的 query 几乎全是 gold 论文标题的改写」。真库上量下来，
这个说法**说过头了**，但方向是对的，而且能说得更准（2026-09-07 实测，
判据 = 查询里的词面有多少落在 gold 论文标题里；中文按二元切、英文按词切）：

    keyword 形态  n=32  中位 0.50 / 均值 0.45  |  >=80% 的有 **10 条（31%）**，其中 5 条是 1.00
    natural 形态  n=32  中位 0.38 / 均值 0.32  |  >=80% 的有 **0 条**

也就是说：**keyword 形态里约三分之一基本等于标题查找**（r07/r08/r14/r22/r32 这几条
的查询词是 gold 标题的真子集），而 2026-09-03 补的 natural 改写形态没有这个问题。
覆盖率为 0 的那几条（r03/r06/r09）是中文问句配英文标题，这个判据对它们不适用，
不代表它们「更难」。

**这意味着什么**：keyword 形态的 Recall 数字里含有一块「把标题抄回去」的水分，
不能拿它单独证明检索质量；两个形态一起报、并说明这个成分，才是诚实的口径。

**这个文件不修数据**，它把成分钉住：
① 高重叠条目不许再增加（新加题不能继续从标题里抄）；
② natural 形态必须保持 0 条高重叠；
③ 两个形态的条目要一一对应（`derived_from`），否则「形态对比」就不是配对实验。
"""
import ast
import json
import re
import sqlite3
import unittest
from pathlib import Path

from papernest import config

EVAL = config.ROOT / "eval_set.json"
REAL_DB = config.ROOT / "data" / "papernest.db"

#: 判为「基本就是标题查找」的阈值
HIGH = 0.8
#: 当前 keyword 形态里高重叠的条数——**只许减少，不许增加**
MAX_HIGH_OVERLAP_KEYWORD = 10


def _tok(s: str) -> set:
    s = s or ""
    return (set(re.findall(r"[a-z0-9]+", s.lower()))
            | {s[i:i + 2] for i in range(len(s) - 1)
               if re.fullmatch(r"[一-鿿]{2}", s[i:i + 2])})


class EvalSetCompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not EVAL.exists():
            raise unittest.SkipTest("没有 eval_set.json")
        cls.items = json.loads(EVAL.read_text(encoding="utf-8"))
        cls.titles = {}
        if REAL_DB.exists():
            conn = sqlite3.connect(f"file:{REAL_DB.as_posix()}?mode=ro", uri=True)
            try:
                cls.titles = {k: (t or "") for k, t in
                              conn.execute("SELECT norm_key, title FROM papers")}
            finally:
                conn.close()

    def _overlaps(self, form: str):
        if not self.titles:
            self.skipTest("没有真库，取不到 gold 论文的标题")
        out = []
        for q in self.items:
            if q.get("type") != "retrieval" or q.get("form") != form:
                continue
            keys = q.get("gold_keys") or []
            if isinstance(keys, str):
                keys = ast.literal_eval(keys)
            qt = _tok(q["query"])
            if not qt:
                continue
            tt = _tok(" ".join(self.titles.get(k, "") for k in keys))
            out.append((q["id"], len(qt & tt) / len(qt)))
        return out

    def test_keyword_form_title_lookup_share_does_not_grow(self):
        """新加的题不许继续从 gold 标题里抄词面——那测的是「能不能把标题查回来」。"""
        high = [i for i, r in self._overlaps("keyword") if r >= HIGH]
        self.assertLessEqual(
            len(high), MAX_HIGH_OVERLAP_KEYWORD,
            f"keyword 形态里查询词≥{HIGH:.0%} 落在 gold 标题里的有 {len(high)} 条"
            f"（上限 {MAX_HIGH_OVERLAP_KEYWORD}）：{high}")

    def test_natural_form_is_not_a_title_lookup(self):
        """natural 形态存在的**理由**就是不抄标题；破了这条它就白加了。"""
        high = [i for i, r in self._overlaps("natural") if r >= HIGH]
        self.assertEqual(high, [], f"natural 形态里出现了标题查找题：{high}")

    def test_the_two_forms_are_paired(self):
        """两个形态是配对实验（同一批 gold、只改问法），不配对就不能做形态对比。"""
        kw = {q["id"] for q in self.items
              if q.get("type") == "retrieval" and q.get("form") == "keyword"}
        nat = [q for q in self.items
               if q.get("type") == "retrieval" and q.get("form") == "natural"]
        self.assertTrue(kw and nat, "两个形态至少要各有一条")
        orphan = [q["id"] for q in nat if q.get("derived_from") not in kw]
        self.assertEqual(orphan, [],
                         f"这些 natural 题找不到对应的 keyword 题：{orphan}")

    def test_gold_keys_all_resolve(self):
        """gold 指向的论文必须还在库里，否则这条题的分母是错的。"""
        if not self.titles:
            self.skipTest("没有真库")
        dangling = []
        for q in self.items:
            keys = q.get("gold_keys") or []
            if isinstance(keys, str):
                keys = ast.literal_eval(keys)
            dangling += [(q["id"], k) for k in keys if k not in self.titles]
        self.assertEqual(dangling, [], f"gold 指向了库里没有的论文：{dangling[:5]}")


if __name__ == "__main__":
    unittest.main()
