import sqlite3
import unittest

from papernest import config, eval as ev


class SplitClaimsTests(unittest.TestCase):
    def test_cited_claims_are_supported(self):
        answer = ("AgentGym 提供了多环境训练框架并评测了多种智能体的能力 [1]。"
                  "它同时给出了统一的训练方法 [2]。")
        supported, unsupported = ev.split_claims(answer)
        self.assertEqual(len(supported), 2)
        self.assertEqual(unsupported, [])

    def test_uncited_claims_count_as_no_evidence(self):
        answer = ("近场信道估计需要考虑球面波前的影响，这与远场自由空间传播假设完全不同。"
                  "库内文献未覆盖这个问题，建议检索关键词 sphere wavefront。")
        supported, unsupported = ev.split_claims(answer)
        # 第二句是免责/元话术，不计入论断
        self.assertEqual(supported, [])
        self.assertEqual(len(unsupported), 1)

    def test_short_fragments_are_ignored(self):
        supported, unsupported = ev.split_claims("好的。是的。")
        self.assertEqual((supported, unsupported), ([], []))


class MetricMathTests(unittest.TestCase):
    def test_recall_macro_average_with_fake_retriever(self):
        entries = [
            {"id": "a", "type": "retrieval", "query": "q1", "gold_keys": ["k1"]},
            {"id": "b", "type": "retrieval", "query": "q2", "gold_keys": ["k1", "k2"]},
        ]
        fake_ids = {  # q1 召回的论文不在 gold 里，q2 命中 k2
            "q1": ([101], None),
            "q2": ([102], None),
        }
        orig_keys = ev._keys_of
        fake_keys = {101: {"k0"}, 102: {"k2"}}
        try:
            ev._keys_of = lambda ids: set().union(*[fake_keys[i] for i in ids]) if ids else set()
            r = ev.metric_recall(entries, k=5, retrieve=lambda q, k: fake_ids[q])
        finally:
            ev._keys_of = orig_keys
        self.assertEqual(r["items"][0]["recall"], 0.0)
        self.assertEqual(r["items"][1]["recall"], 0.5)
        self.assertEqual(r["recall_at_k"], 0.25)


class EvalSetIntegrityTests(unittest.TestCase):
    """评测集 gold 键必须在当前库里真实存在——curation 出错时这里报警。

    这是全套里唯一一条**有意**读真实库的用例（其余都走临时库），所以它必须
    只读打开。原来它调的是 `db.init_db()`：那是写路径，会对用户 40MB 的生产库
    执行 `executescript(SCHEMA)` + `_migrate()`——跑一次测试就顺带做了一次
    schema 迁移，实测触发过（user_version 5→6）。
    库不存在时跳过，而不是失败：干净 checkout 上没有这个库，不该因此变红。
    """

    def test_all_gold_keys_exist_in_library(self):
        if not config.DB_PATH.exists():
            self.skipTest(f"本机没有文献库（{config.DB_PATH}），跳过 gold 键校验")
        uri = f"file:{config.DB_PATH.as_posix()}?mode=ro"
        con = sqlite3.connect(uri, uri=True)
        try:
            con.row_factory = sqlite3.Row
            known = {r["norm_key"] for r in con.execute("SELECT norm_key FROM papers")}
        finally:
            con.close()
        entries = ev.load_eval_set()
        for e in entries:
            missing = set(e["gold_keys"]) - known
            self.assertFalse(missing, f"{e['id']} 的 gold 键不在库中：{missing}")

    def test_query_forms_are_paired_one_to_one(self):
        """natural 形态必须与某条 keyword 形态共享 gold——它测的是「换个问法还找不找得到」，
        不是新造 gold。gold 一旦独立生成，就会掉进「用系统自己的召回当标准答案」的自证陷阱。
        """
        entries = ev.load_eval_set()
        by_id = {e["id"]: e for e in entries}
        keyword = [e for e in entries
                   if e["type"] == "retrieval" and e.get("form", "keyword") == "keyword"]
        natural = [e for e in entries
                   if e["type"] == "retrieval" and e.get("form") == "natural"]
        self.assertTrue(keyword, "关键词形态的检索题不该为空")
        for e in natural:
            src = by_id.get(e.get("derived_from"))
            self.assertIsNotNone(src, f"{e['id']} 的 derived_from 指向不存在的条目")
            self.assertEqual(sorted(e["gold_keys"]), sorted(src["gold_keys"]),
                             f"{e['id']} 的 gold 与来源 {src['id']} 不一致")
            self.assertNotEqual(e["query"], src["query"], f"{e['id']} 没有真的改写查询")

    def test_ids_are_unique(self):
        ids = [e["id"] for e in ev.load_eval_set()]
        self.assertEqual(len(ids), len(set(ids)))


if __name__ == "__main__":
    unittest.main()
