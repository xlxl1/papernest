# -*- coding: utf-8 -*-
"""日志：**够用就好**，因为这是一个单用户的本机工具。

## 为什么不是结构化 JSON + request-id + trace

那套东西解决的是「多实例、多租户、要在 Grafana 里按 trace 串起来」的问题。
本项目的部署设想是**自己本机跑**（`docker-compose` 也绑在 `127.0.0.1`，
`api._check_exposure()` 在绑非回环又没口令时直接拒绝启动）。
在这个前提下，日志要回答的问题只有一个：

    「刚才那次为什么失败？」——事后还查得到吗？

所以这里只做三件事：写到一个会轮转的文件、带时间戳和 traceback、
默认同时打到 stderr（前台跑 `cli.py serve` 时能直接看见）。
**没做**的：结构化字段、request-id、采样、远端上报——它们在本机单用户下
是纯负担，真要多用户部署时再加，那时也该连鉴权/租户一起重新设计。

## 为什么需要它

全仓 225 个 `except` 处理器里有 161 个（72%）**既不重抛也不记录**。
其中大多数是正当的（「这条解析策略不行就换下一条」），但有一类不是：
API 端点把异常吞成 `HTTP 200 + {"error": ...}`——前端只显示一句话，
traceback 连同栈帧一起消失，事后完全无从复盘。这个模块就是给那一类用的。

## 用法

    from . import log
    logger = log.get("api")
    ...
    except Exception:
        logger.exception("survey 生成失败 topic=%s", topic)   # 带 traceback
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import threading

from . import config

#: 日志文件。跟库放在一起，`data/` 本来就是这个项目的可写目录。
LOG_PATH = config.DATA_DIR / "papernest.log"
#: 单文件上限与保留份数：本机工具不该让日志吃掉磁盘。
MAX_BYTES = 5 * 1024 * 1024
BACKUPS = 3
#: 默认 INFO；调试时 `PAPERNEST_LOG_LEVEL=DEBUG`。非法值退回 INFO，不炸。
LEVEL = os.environ.get("PAPERNEST_LOG_LEVEL", "INFO").strip().upper()

_lock = threading.Lock()
_ready = False


def _setup() -> None:
    """幂等初始化。**只挂在 `papernest` 这个 logger 上**，不碰 root——
    动 root 会顺手改掉 uvicorn / httpx 的日志行为，那不是这个模块该管的事。"""
    global _ready
    if _ready:
        return
    with _lock:
        if _ready:
            return
        root = logging.getLogger("papernest")
        root.setLevel(getattr(logging, LEVEL, logging.INFO))
        root.propagate = False          # 免得再被 root 打一遍
        fmt = logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S")
        try:
            config.DATA_DIR.mkdir(parents=True, exist_ok=True)
            fh = logging.handlers.RotatingFileHandler(
                LOG_PATH, maxBytes=MAX_BYTES, backupCount=BACKUPS,
                encoding="utf-8")
            fh.setFormatter(fmt)
            root.addHandler(fh)
        except OSError:
            # 只读挂载 / 权限不足：**不能因为写不了日志就让程序起不来**
            pass
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(fmt)
        root.addHandler(sh)
        _ready = True


def get(name: str) -> logging.Logger:
    """取一个子 logger（`papernest.<name>`）。第一次调用时完成初始化。"""
    _setup()
    return logging.getLogger("papernest").getChild(name)
