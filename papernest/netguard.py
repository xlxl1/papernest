"""出站请求的目标校验：挡住 SSRF。

**为什么需要它**：`fulltext.fetch_pdf` 直接拿 `papers.oa_pdf_url` 发服务端 GET，
而那个字段是可以被用户写入的——`POST /api/import/bibliography` 上传一份 .bib，
里面的 `url` 只要以 `.pdf` 结尾就会被 `bibimport` 收进 `oa_pdf_url`。
于是「上传 .bib → 触发精读」就是一条完整的服务端请求伪造链路，
在默认无口令的部署形态下可被任意人触发。

两条关键设计：

1. **按解析出的 IP 判，不是按域名判**。`http://内网主机.example.com/x.pdf` 的域名
   看着完全正常，解析出来是 10.0.0.5。域名黑名单挡不住 DNS 重绑定之外的任何情况。
2. **必须逐跳校验**。`follow_redirects=True` 时，一个合法外域 302 到 169.254.169.254
   同样能打进来——所以调用方要关掉自动重定向，每一跳都过一次 `check_url`。

不做的事：不解决 DNS rebinding（校验时解析一次、httpx 连接时再解析一次，中间存在
TOCTOU 窗口）。要彻底堵住得自己建连并把已校验的 IP 传给 socket，代价与收益不成比例——
本项目的威胁模型是「别把内网端口和云元数据服务暴露出去」，这一层足够。
"""
from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

#: 只允许 https。arXiv / S2 / OpenAlex 的 OA 链接全部支持 https，
#: 放行 http 只会多一条明文且更容易被中间人改写的路径。
ALLOWED_SCHEMES = ("https",)
#: 只允许标准 https 端口。允许任意端口等于把内网端口扫描的能力直接交出去。
ALLOWED_PORTS = (443,)
#: 跟随重定向的最大跳数（每一跳都要重新校验）
MAX_REDIRECTS = 3


class BlockedURL(Exception):
    """目标不允许访问。消息面向日志，**不要原样回吐给调用方**——
    「拒绝了哪个 IP 段」本身就是一条内网探测的反馈。"""


def _ip_is_public(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return not (addr.is_private or addr.is_loopback or addr.is_link_local
                or addr.is_reserved or addr.is_multicast or addr.is_unspecified)


def resolve(host: str) -> list[str]:
    """解析出全部 A/AAAA 记录。一个都解析不出来就当作不可访问。"""
    try:
        return sorted({info[4][0] for info in
                       socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)})
    except OSError:
        return []


def check_url(url: str) -> str:
    """校验一个出站 URL，通过则原样返回，否则抛 `BlockedURL`。

    校验项：scheme、端口、host 能解析、且**解析出的每一个 IP 都是公网地址**。
    「每一个」是刻意的——只要有一条 A 记录指向内网就拒绝，不给部分命中留缝。
    """
    p = urlparse(url or "")
    if p.scheme not in ALLOWED_SCHEMES:
        raise BlockedURL(f"scheme 不允许：{p.scheme!r}（只允许 {'/'.join(ALLOWED_SCHEMES)}）")
    if not p.hostname:
        raise BlockedURL("URL 里没有主机名")
    port = p.port or 443
    if port not in ALLOWED_PORTS:
        raise BlockedURL(f"端口不允许：{port}")
    ips = resolve(p.hostname)
    if not ips:
        raise BlockedURL(f"主机名解析不到任何地址：{p.hostname}")
    bad = [ip for ip in ips if not _ip_is_public(ip)]
    if bad:
        raise BlockedURL(f"{p.hostname} 解析到非公网地址：{bad}")
    return url
