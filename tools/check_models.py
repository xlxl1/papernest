#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""模型连通性自检：一次最小调用，把「配没配对」说清楚。

接模型之前先跑这个。它只发**一次**最小请求（LLM 几个 token、嵌入一个短句），
花费可以忽略，但能把下面这几种最常见的错配区分开——
它们的报错长得很像，靠肉眼看 401 是分不出来的：

  · key 与 base_url 不配套（换了平台只改了 key）→ 401 INVALID_API_KEY
  · 模型名在这个平台上不存在              → 404 / model not found
  · key 过期或额度用尽                    → 401 / 403 / 429
  · base_url 少了 /v1 或多了 /chat/completions → 404

用法：python tools/check_models.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from papernest import config, embeddings, llm  # noqa: E402

# Windows 的 GBK 控制台遇到非 GBK 字符会直接抛 UnicodeEncodeError，
# 把诊断工具自己搞崩——它恰恰是出问题时才跑的，不能挑控制台。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")


def _mask(v: str) -> str:
    if not v:
        return "（空）"
    return f"{v[:6]}…{v[-4:]}（{len(v)} 字符）"


def _diagnose(exc: Exception, base: str, model: str) -> list[str]:
    """把报错翻译成「该去改哪一行」。"""
    msg = str(exc)
    tips: list[str] = []
    if "401" in msg or "INVALID_API_KEY" in msg or "Unauthorized" in msg:
        tips.append("key 被拒。最常见的原因是**换了平台只改了 key、base_url 还是旧的**——")
        tips.append(f"  当前 base_url 是 {base}")
        tips.append("  确认这个 key 就是这个域名签发的；换平台时 base_url / 模型名要一起换。")
        tips.append("  其次才考虑：key 复制时少了尾字符、已过期、或额度用尽。")
    elif "404" in msg or "not found" in msg.lower():
        tips.append(f"路径或模型名不对。当前模型 {model!r}，base_url {base}")
        tips.append("  · base_url 应当只到 /v1，代码会自己拼 /chat/completions")
        tips.append("  · 模型名要用平台模型列表里那个**可复制的 ID**，不是显示名")
    elif "429" in msg:
        tips.append("被限流或额度用尽。等一会儿再试，或换一个还有额度的模型。")
    elif "Connect" in type(exc).__name__ or "Timeout" in type(exc).__name__:
        tips.append(f"连不上 {urlparse(base).netloc}。检查网络 / 代理（PAPERNEST_PROXY）。")
    return tips


def main() -> int:
    print(f"配置文件：{config.ROOT / '.env'}")
    print()
    print("── LLM ──")
    print(f"  base  {config.LLM_API_BASE or '（空）'}")
    print(f"  key   {_mask(config.LLM_API_KEY)}")
    print(f"  model {config.LLM_MODEL or '（空）'}")
    ok_llm = False
    if not llm.available():
        print("  [FAIL] 未配置齐全（base / key / model 三样都要有）")
    else:
        t0 = time.perf_counter()
        try:
            out = llm.chat("只回答一个词。", "回答：ok", purpose="probe", temperature=0)
            print(f"  [OK]  通了（{time.perf_counter() - t0:.1f}s）返回 {out.strip()[:40]!r}")
            ok_llm = True
        except Exception as exc:                                # noqa: BLE001
            print(f"  [FAIL] {type(exc).__name__}: {str(exc)[:160]}")
            for line in _diagnose(exc, config.LLM_API_BASE, config.LLM_MODEL):
                print(f"    {line}")

    print()
    print("── 嵌入 ──")
    print(f"  base  {config.EMBED_API_BASE or '（空）'}")
    print(f"  model {config.EMBED_MODEL or '（空）'}")
    ok_emb = False
    if not embeddings.available():
        print("  [FAIL] 未配置齐全")
    else:
        t0 = time.perf_counter()
        try:
            vec = embeddings.embed_texts(["连通性自检"], purpose="probe")[0]
            print(f"  [OK]  通了（{time.perf_counter() - t0:.1f}s）维度 {len(vec)}")
            ok_emb = True
        except Exception as exc:                                # noqa: BLE001
            print(f"  [FAIL] {type(exc).__name__}: {str(exc)[:160]}")
            for line in _diagnose(exc, config.EMBED_API_BASE, config.EMBED_MODEL):
                print(f"    {line}")

    print()
    print(f"结论：LLM {'可用' if ok_llm else '不可用'} · 嵌入 {'可用' if ok_emb else '不可用'}")
    if ok_llm and ok_emb:
        print("两条都通了，可以开始跑功能验证。")
    return 0 if (ok_llm and ok_emb) else 1


if __name__ == "__main__":
    raise SystemExit(main())
