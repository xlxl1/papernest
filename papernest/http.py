"""统一 HTTP 客户端工厂。

默认 trust_env=False：不让系统环境变量里的代理（可能指向没开的 Clash）
悄悄劫持所有请求。需要代理时显式设置 PAPERNEST_PROXY（.env 亦可）。
"""
import httpx

from . import config


def client(timeout: float | None = None, **kw) -> httpx.Client:
    """统一构造 httpx.Client。**每个默认值都可被调用方覆盖**。

    原来 `follow_redirects=True` 是硬写在关键字里的，调用方再传一次就是
    `TypeError: got multiple values for keyword argument`——而这个异常会被
    调用点的 `except Exception` 吞掉，表现成一句「下载失败」。
    出站校验那条路径正需要关掉自动重定向（自己逐跳校验），于是「校验没跑到」
    被伪装成了「校验生效了」。默认值用 setdefault 给，不要写死。
    """
    kw.setdefault("trust_env", False)
    kw.setdefault("proxy", config.PAPERNEST_PROXY or None)
    kw.setdefault("follow_redirects", True)
    return httpx.Client(timeout=timeout or config.HTTP_TIMEOUT, **kw)
