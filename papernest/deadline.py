"""协作式截止时间：让「被放弃的工作」真的停下来。

**为什么需要它**：`agent.run_agent` 用 `future.result(timeout=)` 实现所谓「硬截止」，
但 Python **无法中断已经开跑的线程**——`future.cancel()` 对 running 任务恒为 False。
超时后 run 立刻返回 failed，而那条线程仍在跑完整个工具：继续下载 PDF、继续写库、
继续调重档模型花钱。实测：`timeout_s=2` 的步骤，工具线程在 6s 处照样跑完。

更糟的是预算根本对不上：`llm.chat` 最坏是 4 次尝试 × 90s 超时 + 8/20/45 的退避 ≈ 433s，
而 agent 的默认步骤超时是 240s。**单次 LLM 调用就能超过步骤超时**，硬截止必然踩到。

这里给出的是**协作式**取消，不是抢占式：
- 执行方在工作线程里 `set(deadline)`；
- 长耗时组件（`llm.chat` 的重试循环、退避 sleep）在每个可中断点检查 `expired()`，
  过期就立刻抛 `DeadlineExceeded`，并且**退避 sleep 不会睡过截止时间**。

覆盖的是时间成本的大头（重试与退避，最坏 433s 里有 153s 纯粹在 sleep）。
**不覆盖**的是单次阻塞的 C 调用（PyMuPDF 解析一页、一次 socket read）——
那些只能等它自己返回，Python 层面没有办法。这一点必须如实说，不要把协作式
取消说成「硬截止」。
"""
from __future__ import annotations

import threading
import time

_local = threading.local()


class DeadlineExceeded(Exception):
    """当前线程的工作预算已用尽。上层应当把它当成超时处理，而不是重试。"""


def set(seconds: float | None) -> None:
    """给**当前线程**设一个从现在起 `seconds` 秒的截止时间。None 表示不限。"""
    _local.at = None if seconds is None else time.monotonic() + seconds


def clear() -> None:
    _local.at = None


def at() -> float | None:
    return getattr(_local, "at", None)


def remaining() -> float:
    """剩余秒数；没设截止时间返回 `inf`。"""
    a = at()
    return float("inf") if a is None else a - time.monotonic()


def expired() -> bool:
    return remaining() <= 0


def check() -> None:
    """到点就抛。放在每个可中断点（重试之间、发请求之前）。"""
    if expired():
        raise DeadlineExceeded("工作预算已用尽（步骤超时），停止重试")


def sleep(seconds: float) -> None:
    """退避睡眠，但**绝不睡过截止时间**。睡完仍然过期就抛。

    原来退避是裸 `time.sleep(80)`——步骤早就超时返回了，这条线程还要再睡 80 秒
    然后发一次没人要的请求。
    """
    time.sleep(max(0.0, min(seconds, remaining())))
    check()


class scope:
    """`with deadline.scope(30): ...` —— 进入时设，退出时还原。"""

    def __init__(self, seconds: float | None):
        self.seconds = seconds

    def __enter__(self):
        self._prev = at()
        set(self.seconds)
        return self

    def __exit__(self, *exc):
        _local.at = self._prev
        return False
