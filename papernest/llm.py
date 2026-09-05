"""OpenAI 兼容 LLM 客户端。

不配 key 时 available()=False，上层走 mock 摘要直取——流水线先行，
接 key 后新论文自动出真卡片，旧 mock 卡片用 cli.py recard 重刷。
非重试边界按成因划分：400 立即失败不重试，429/5xx 才退避（QC-Bench F2）。
"""
import json
import random
import time

import httpx

from . import config, db, deadline, http


class LLMUnavailable(Exception):
    pass


class LLMError(Exception):
    pass


def available() -> bool:
    return bool(config.LLM_API_BASE and config.LLM_API_KEY and config.LLM_MODEL)


def chat(system: str, user: str, purpose: str, paper_id: int | None = None,
         temperature: float = 0.3, model: str | None = None) -> str:
    if not available():
        raise LLMUnavailable("未配置 LLM_API_BASE / LLM_API_KEY / LLM_MODEL")
    url = config.LLM_API_BASE.rstrip("/") + "/chat/completions"
    payload = {
        "model": model or config.LLM_MODEL,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": temperature,
    }
    headers = {"Authorization": f"Bearer {config.LLM_API_KEY}"}
    # 中转站过载时常表现为 TLS 随机中断：4 次指数退避 + 抖动，
    # 单次成功率 ~60% 时整体成功率 1-0.4^4 ≈ 97%
    delays = [d + random.uniform(0, d * 0.3) for d in (8, 20, 45, 80)]
    last = None
    _t0 = time.perf_counter()
    # 单次调用最坏 4×90s 超时 + 8/20/45 退避 ≈ 433s，远超 agent 的步骤超时（240s）。
    # 这里在每个可中断点看一眼工作预算：过期就停手，而不是继续烧钱、继续占线程。
    with http.client(timeout=min(90, max(1.0, deadline.remaining()))) as client:
        for attempt in range(4):
            deadline.check()
            try:
                r = client.post(url, json=payload, headers=headers)
            except httpx.HTTPError as e:
                last = e
                if attempt < 3:
                    deadline.sleep(delays[attempt])
                continue
            if r.status_code == 400:
                raise LLMError(f"LLM 请求被拒（400，不重试）：{r.text[:200]}")
            if r.status_code in (429, 500, 502, 503, 504):
                last = f"HTTP {r.status_code}"
                if attempt < 3:
                    deadline.sleep(delays[attempt])
                continue
            if r.status_code >= 400:  # 401/403/404：配置问题，重试无意义
                raise LLMError(f"LLM 请求被拒（HTTP {r.status_code}，不重试）：{r.text[:200]}")
            data = r.json()
            usage = data.get("usage") or {}
            # 成本证据链：每次调用落账 tokens / latency / cost（无价格表则 cost=NULL）
            latency_ms = (time.perf_counter() - _t0) * 1000
            price = config.model_price(payload["model"])
            cost = None
            if price and usage:
                cost = (usage.get("prompt_tokens", 0) * price[0]
                        + usage.get("completion_tokens", 0) * price[1]) / 1e6
            with db.conn() as c:
                db.add_llm_call(c, purpose, paper_id,
                                usage.get("prompt_tokens", 0),
                                usage.get("completion_tokens", 0),
                                payload["model"], latency_ms=round(latency_ms, 1),
                                cost_usd=round(cost, 8) if cost is not None else None)
                c.commit()
            return data["choices"][0]["message"]["content"]
    raise LLMError(f"LLM 连续 4 次失败：{last}")


def chat_stream(system: str, user: str, purpose: str, paper_id: int | None = None,
                temperature: float = 0.3, model: str | None = None):
    """流式对话：逐段 yield 文本 chunk；落账与非流式一致。

    重试边界：首 chunk 到达前失败可整体重试；已经开始输出后再断流，
    重试会导致重复文本，直接抛错交给上层处理。
    """
    if not available():
        raise LLMUnavailable("未配置 LLM_API_BASE / LLM_API_KEY / LLM_MODEL")
    url = config.LLM_API_BASE.rstrip("/") + "/chat/completions"
    payload = {
        "model": model or config.LLM_MODEL,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    headers = {"Authorization": f"Bearer {config.LLM_API_KEY}"}
    delays = [d + random.uniform(0, d * 0.3) for d in (8, 20, 45, 80)]
    last = None
    _t0 = time.perf_counter()
    chunks: list[str] = []  # 跨 attempt 保留：首 chunk 后断流不重试，避免重复输出
    for attempt in range(4):
        deadline.check()
        try:
            with http.client(timeout=httpx.Timeout(
                    connect=15, read=min(180, max(1.0, deadline.remaining())),
                    write=30, pool=15)) as client:
                with client.stream("POST", url, json=payload, headers=headers) as r:
                    if r.status_code == 400 and "stream_options" in payload:
                        payload.pop("stream_options")  # 个别兼容端点不认 usage 选项
                        last = "HTTP 400 (stream_options)"
                        continue
                    if r.status_code in (429, 500, 502, 503, 504):
                        last = f"HTTP {r.status_code}"
                        if attempt < 3:
                            deadline.sleep(delays[attempt])
                        continue
                    if r.status_code >= 400:
                        # 400/401/403 是请求本身的问题，重试只是把同一个错误再犯 4 遍。
                        # （原来这里走 raise_for_status()，HTTPStatusError 属于 httpx.HTTPError，
                        #   会被下面的 except 接住去退避重试，与非流式的「400 不重试」自相矛盾。）
                        detail = r.read().decode("utf-8", "replace")[:200]
                        raise LLMError(
                            f"LLM 请求被拒（HTTP {r.status_code}，不重试）：{detail}")
                    usage: dict | None = None
                    for line in r.iter_lines():
                        if not line.startswith("data:"):
                            continue
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            obj = json.loads(data_str)
                        except ValueError:
                            continue
                        if obj.get("usage"):
                            usage = obj["usage"]
                        delta = ((obj.get("choices") or [{}])[0].get("delta") or {})
                        t = delta.get("content")
                        if t:
                            chunks.append(t)
                            yield t
                    latency_ms = round((time.perf_counter() - _t0) * 1000, 1)
                    price = config.model_price(payload["model"])
                    cost = None
                    if price and usage:
                        cost = (usage.get("prompt_tokens", 0) * price[0]
                                + usage.get("completion_tokens", 0) * price[1]) / 1e6
                    with db.conn() as c:
                        db.add_llm_call(c, purpose, paper_id,
                                        (usage or {}).get("prompt_tokens", 0),
                                        (usage or {}).get("completion_tokens", 0),
                                        payload["model"], latency_ms=latency_ms,
                                        cost_usd=round(cost, 8) if cost is not None else None)
                        c.commit()
                    return
        except httpx.HTTPError as e:
            last = e
            if chunks:
                raise LLMError(f"输出中断（已产出 {len(chunks)} 段，不自动重试避免重复）：{e}") from e
            if attempt < 3:
                deadline.sleep(delays[attempt])
            continue
    raise LLMError(f"LLM 连续 4 次失败：{last}")


def extract_json(text: str) -> dict:
    """从回复中提取第一个完整 JSON 对象；兼容模型加的 ```json 围栏。"""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    start = text.find("{")
    if start == -1:
        raise LLMError("回复中没有 JSON 对象")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise LLMError("JSON 未闭合")
