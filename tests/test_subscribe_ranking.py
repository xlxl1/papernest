"""订阅打分的两条排序规则：批内 IDF 与「广度折扣」。

都是拿真实数据打出来的问题反推的规则，所以用例直接照着那两个症状写：
- 库画像里权重最高的往往是 learning / deep 这种泛词（库内很多论文带它们），
  但它们同时出现在当天 arXiv 的半数论文里，毫无区分度 —— 结果推来的是
  「城市交通预测」「核岭回归」「农业 Web 系统」，而库真正的主题一篇没中。
- 只靠一个泛词命中的论文，不该排在命中多个不同概念的论文前面。
"""
import unittest

from papernest import subscribe


def _paper(aid: str, title: str, abstract: str = "") -> dict:
    return {"arxiv_id": aid, "norm_key": f"arxiv:{aid}",
            "title": title, "abstract": abstract}


def _profile(*terms: tuple[str, float]) -> dict:
    return {"terms": [{"term": t, "weight": w} for t, w in terms],
            "arxiv_categories": [], "n_papers": 10}


class BatchIdfTests(unittest.TestCase):
    """批内 IDF：用这批论文自己算区分度，不需要任何外部语料。"""

    def _batch(self, n_common: int = 28, n_rare: int = 2) -> list[dict]:
        # 大批量里「learning」几乎人人都有，「beamforming」只有两篇
        out = [_paper(f"26{i:04d}", f"Learning Method Number {i}") for i in range(n_common)]
        out += [_paper(f"27{i:04d}", f"Beamforming Design Number {i}") for i in range(n_rare)]
        return out

    def test_common_term_gets_lower_idf_than_rare_term(self):
        batch = self._batch()
        prof = _profile(("learning", 1.0), ("beamforming", 1.0))
        scored = subscribe.score_papers(batch, prof)
        idf = {}
        for it in scored:
            for r in it["reasons"]:
                idf[r["term"]] = r["idf"]
        self.assertLess(idf["learning"], idf["beamforming"])
        self.assertLess(idf["learning"], 0.5)       # 占满全批 → 被压得很低
        self.assertGreater(idf["beamforming"], 0.6)  # 只占 2/30 → 基本原样保留

    def test_rare_term_paper_outranks_common_term_paper(self):
        """同样的画像权重、同样是标题命中一个词——罕见词那篇必须排前面。"""
        batch = self._batch()
        scored = subscribe.score_papers(batch, _profile(("learning", 1.0),
                                                        ("beamforming", 1.0)))
        top = scored[0]
        self.assertIn("Beamforming", top["title"])

    def test_small_batch_is_not_penalised(self):
        """批太小时 IDF 没有统计意义，不该拿几篇论文去断定一个词是不是泛词。"""
        batch = [_paper("2600001", "Learning Things"),
                 _paper("2600002", "Beamforming Things")]
        scored = subscribe.score_papers(batch, _profile(("learning", 1.0),
                                                        ("beamforming", 1.0)))
        for it in scored:
            for r in it["reasons"]:
                self.assertEqual(r["idf"], 1.0)

    def test_idf_is_reported_for_every_reason(self):
        """推荐理由要能自证：用户得看到这个词到底有没有区分度。"""
        scored = subscribe.score_papers(self._batch(), _profile(("learning", 1.0)))
        for it in scored:
            for r in it["reasons"]:
                self.assertIn("idf", r)
                self.assertGreater(r["idf"], 0.0)

    def test_scoring_is_deterministic_across_runs(self):
        batch = self._batch()
        prof = _profile(("learning", 1.0), ("beamforming", 1.0), ("design", 0.5))
        runs = {tuple((i["arxiv_id"], i["score"]) for i in
                      subscribe.score_papers(batch, prof)) for _ in range(5)}
        self.assertEqual(len(runs), 1)


class BreadthDiscountTests(unittest.TestCase):
    """广度折扣：一个泛词命中不构成推荐理由。"""

    def _batch_with(self, *extra: dict) -> list[dict]:
        filler = [_paper(f"28{i:04d}", f"Unrelated Topic {i}") for i in range(25)]
        return list(extra) + filler

    def test_single_concept_match_is_flagged_and_discounted(self):
        one = _paper("2900001", "Beamforming Alone")
        two = _paper("2900002", "Beamforming And Mimo Together")
        scored = subscribe.score_papers(self._batch_with(one, two),
                                        _profile(("beamforming", 1.0), ("mimo", 1.0)))
        by_id = {i["arxiv_id"]: i for i in scored}
        self.assertEqual(by_id["2900001"]["n_concepts"], 1)
        self.assertEqual(by_id["2900002"]["n_concepts"], 2)
        self.assertIn("只命中一个词项", by_id["2900001"]["weak_match"])
        self.assertNotIn("weak_match", by_id["2900002"])
        self.assertLess(by_id["2900001"]["score"], by_id["2900002"]["score"])

    def test_raw_score_stays_equal_to_the_sum_of_reasons(self):
        """折扣只作用于 score：raw_score 必须仍等于各条理由之和，
        否则「理由能解释分数」就断了，而那是这个模块的立身之本。"""
        one = _paper("2900003", "Beamforming Alone")
        scored = subscribe.score_papers(self._batch_with(one), _profile(("beamforming", 1.0)))
        item = next(i for i in scored if i["arxiv_id"] == "2900003")
        self.assertAlmostEqual(item["raw_score"],
                               sum(r["contribution"] for r in item["reasons"]), places=6)
        self.assertLess(item["score"], item["raw_score"])    # 但 score 打了折

    def test_no_match_has_no_weak_flag(self):
        """一个词都没命中的论文不该被标成「弱匹配」——它根本不是匹配。"""
        scored = subscribe.score_papers(self._batch_with(), _profile(("beamforming", 1.0)))
        for it in scored:
            self.assertEqual(it["score"], 0.0)
            self.assertNotIn("weak_match", it)

    def test_breadth_beats_a_single_strong_hit(self):
        """这就是真实数据里那个症状：标题里有个 deep，此外与本库毫无关系。"""
        generic = _paper("2900004", "Deep Learning For Agriculture Web Systems")
        onTopic = _paper("2900005", "Beamforming Design", "mimo channel estimation study")
        batch = self._batch_with(generic, onTopic)
        batch += [_paper(f"2A{i:04d}", f"Deep Learning Study {i}") for i in range(20)]
        scored = subscribe.score_papers(batch, _profile(
            ("deep", 1.0), ("learning", 1.0), ("beamforming", 0.8), ("mimo", 0.8)))
        order = [i["arxiv_id"] for i in scored]
        self.assertLess(order.index("2900005"), order.index("2900004"))


if __name__ == "__main__":
    unittest.main()
