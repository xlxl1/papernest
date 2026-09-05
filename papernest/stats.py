"""配对显著性检验。零依赖（不引 scipy），结果可复现（固定种子的确定性重采样）。

**为什么必须有这个文件**：本项目的评测集只有 32~69 题，一道题的权重就是 0.031~0.014。
README 里那些 +0.0156 / +0.031 的差异，在这个样本量下完全可能是噪声——
而此前仓库里没有任何检验实现，报出去的 p 值没有出处、事后也重算不出来。
现在改任何影响检索排序的参数，都能用同一个函数给出可复现的 p 值。

用法：

    from papernest.stats import sign_flip_test
    r = sign_flip_test(before_scores, after_scores)   # 两个等长的逐题得分列表
    print(r["mean_diff"], r["p_value"], r["significant"])
"""
from __future__ import annotations

import random

#: 默认重采样次数。20000 次下 p 的分辨率约 5e-5，足够支撑「显著/不显著」的判断，
#: 且在 32~69 题的规模上是毫秒级。
DEFAULT_ROUNDS = 20000
#: 默认种子。**固定**是刻意的：同一批数据两次跑必须给出同一个 p，
#: 否则「显著性」本身就成了不可复现的数字。
DEFAULT_SEED = 20260903


def sign_flip_test(before: list[float], after: list[float],
                   rounds: int = DEFAULT_ROUNDS, seed: int = DEFAULT_SEED,
                   alpha: float = 0.05) -> dict:
    """配对符号翻转检验（双侧）。

    原假设：改动对每一题的影响方向是随机的，即每个差值 d_i 取 ±|d_i| 等概率。
    重采样时随机翻转每个差值的符号，统计 |重采样均值| ≥ |实际均值| 的比例。

    为什么用它而不是配对 t 检验：逐题得分是 0/1 或少数几个离散值（Recall@k 的分母
    通常是 1~3），分布远非正态，样本量又只有几十。符号翻转不假设分布形状，
    对这种数据更稳。

    差值恒为 0 的题**不参与**重采样（它们对任何符号分配都贡献 0），但仍计入
    `n` 与均值的分母——「32 题里只有 3 题变了」这个事实必须留在报告里。
    """
    if len(before) != len(after):
        raise ValueError(f"两组得分长度不一致：{len(before)} vs {len(after)}")
    n = len(before)
    if n == 0:
        return {"n": 0, "n_changed": 0, "mean_before": 0.0, "mean_after": 0.0,
                "mean_diff": 0.0, "p_value": 1.0, "significant": False,
                "rounds": 0, "note": "空评测集"}

    diffs = [a - b for b, a in zip(before, after)]
    observed = sum(diffs) / n
    nonzero = [d for d in diffs if d != 0]
    if not nonzero:
        return {"n": n, "n_changed": 0,
                "mean_before": round(sum(before) / n, 4),
                "mean_after": round(sum(after) / n, 4),
                "mean_diff": 0.0, "p_value": 1.0, "significant": False,
                "rounds": 0, "note": "逐题得分完全相同，无可检验的差异"}

    rng = random.Random(seed)
    hits = 0
    target = abs(observed) - 1e-12          # 容忍浮点误差，否则 |x| >= |x| 可能判假
    for _ in range(rounds):
        s = sum(d if rng.random() < 0.5 else -d for d in nonzero)
        if abs(s / n) >= target:
            hits += 1
    p = (hits + 1) / (rounds + 1)           # 加一平滑：p 不该报成 0
    return {
        "n": n,
        "n_changed": len(nonzero),
        "mean_before": round(sum(before) / n, 4),
        "mean_after": round(sum(after) / n, 4),
        "mean_diff": round(observed, 4),
        "p_value": round(p, 5),
        "significant": p < alpha,
        "rounds": rounds,
        "seed": seed,
    }


def describe(r: dict, label: str = "") -> str:
    """给报告用的一句话结论。**样本量不足时明说，不要把方向当成结论。**"""
    head = f"{label}：" if label else ""
    if not r["n"]:
        return f"{head}无数据"
    delta = r["mean_diff"]
    arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
    core = (f"{r['mean_before']} → {r['mean_after']} "
            f"({arrow}{abs(delta):.4f}，n={r['n']}，{r['n_changed']} 题发生变化)")
    if not r["n_changed"]:
        return f"{head}{core}，逐题完全一致"
    # 不显著时**必须看符号**再措辞。原来这里硬编码「方向为正」，于是变差的结果
    # 也会被描述成方向为正——而这个模块存在的唯一理由就是如实报告负结果，
    # 在它唯一为之而生的场景上撒谎是最糟的形态。
    direction = "为正" if delta > 0 else ("为负" if delta < 0 else "持平")
    verdict = (f"p={r['p_value']}，显著" if r["significant"]
               else f"p={r['p_value']}，**不显著——方向{direction}但样本量不足以判定**")
    return f"{head}{core}，{verdict}"
