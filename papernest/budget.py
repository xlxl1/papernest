# -*- coding: utf-8 -*-
"""每日用量闸门：**防失控，不是防滥用**。

## 为什么不是限流 / 配额

限流解决的是「别人打爆我的服务」。本项目是**自己本机跑**的单用户工具，
没有「别人」——`docker-compose` 绑 `127.0.0.1`，`api._check_exposure()` 在绑非回环
又没口令时直接拒绝启动。所以按 IP / 按用户限速在这里是纯负担。

真正咬过人的是另一件事：**失控的循环把自己的额度烧光**。
2026-09-03 那次 embedding 免费额度被跑光，之后线上口径只能退化成纯 FTS
（Recall@5 从 0.9427 掉到 0.7552），而当时没有任何东西提前拦一下。

## 为什么按 token 计而不按钱

`llm_calls.cost_usd` 依赖 `config.LLM_PRICES_JSON`，而它在 `.env.example` 里是注释掉的
——真库 **3886 条调用里 cost_usd 非空的是 0 条**。也就是说「成本证据链」目前是空的。
价格表是 provider 相关、会变的东西，保持可选是对的；但闸门不能建在一个默认为空的字段上。
`prompt_tokens` / `completion_tokens` 每次调用都真实落账，所以按它计。

## 上限怎么定的

真库实测的每日用量（`llm_calls` 按 `date(ts)` 汇总）：

    08-29  136 次   217,518      08-31  209 次  1,172,694   ← 观测到的峰值
    08-30   16 次    29,417      09-01  170 次  1,100,357
    09-02 1865 次   406,107      09-03  317 次    188,028
    09-04  136 次    75,354      09-07 1037 次    763,671

取 **300 万 / 天**：约为观测峰值的 2.5 倍——正常的重活（全库重嵌、批量精读）撞不到，
而一个失控循环会在几分钟内冲过去。`PAPERNEST_DAILY_TOKEN_BUDGET=0` 显式关闭。

这是**闸门不是预算管理**：它只保证「不会在你没注意时把额度烧光」，
不告诉你花了多少钱——那需要价格表，是另一件事。
"""
from __future__ import annotations

from . import config, db


class BudgetExceeded(RuntimeError):
    """今日用量已达上限。**故意是个显式异常**：静默降级会让人以为模型变笨了。"""


#: 每日 token 上限（prompt + completion 之和，含 embedding）。0 = 不限。
DAILY_TOKEN_BUDGET = config._env_int(3_000_000, "PAPERNEST_DAILY_TOKEN_BUDGET",
                                     minimum=0)


def spent_today() -> int:
    """今天已经用掉的 token。库不可用时返回 0——**闸门坏了不该把功能一起锁死**。"""
    try:
        with db.conn() as c:
            row = c.execute(
                "SELECT COALESCE(SUM(COALESCE(prompt_tokens,0)"
                " + COALESCE(completion_tokens,0)), 0) n FROM llm_calls "
                "WHERE date(ts) = date('now','localtime')").fetchone()
        return int(row["n"] if row else 0)
    except Exception:                                   # noqa: BLE001
        return 0


def remaining() -> int | float:
    """今天还剩多少 token；未设上限时是 `float('inf')`。"""
    if DAILY_TOKEN_BUDGET <= 0:
        return float("inf")
    return max(0, DAILY_TOKEN_BUDGET - spent_today())


def check(what: str = "调用") -> None:
    """要发一次付费请求之前调它。超了就抛 `BudgetExceeded`。

    只在**将要花钱**的入口调（`llm.chat` / `llm.chat_stream` / `embeddings.embed_texts`），
    不要撒到每个函数里——那会让一次问答查十几遍库，而这个闸门的精度本来就只到「今天」。
    """
    if DAILY_TOKEN_BUDGET <= 0:
        return
    used = spent_today()
    if used >= DAILY_TOKEN_BUDGET:
        raise BudgetExceeded(
            f"今日 token 用量已达上限（{used:,} / {DAILY_TOKEN_BUDGET:,}），"
            f"已停止{what}以免烧光额度。"
            f"确认是正常用量就调高 PAPERNEST_DAILY_TOKEN_BUDGET，"
            f"或设成 0 关闭这道闸门。")
