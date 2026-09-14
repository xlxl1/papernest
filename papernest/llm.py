"""OpenAI 兼容 LLM 客户端。

不配 key 时 available()=False，上层走 mock 摘要直取——流水线先行，
接 key 后新论文自动出真卡片，旧 mock 卡片用 cli.py recard 重刷。
非重试边界按成因划分：400 立即失败不重试，429/5xx 才退避（QC-Bench F2）。
"""
import json
import re
import random
import time

import httpx

from . import budget, config, db, deadline, http


class LLMUnavailable(Exception):
    pass


class LLMError(Exception):
    pass


def available() -> bool:
    return bool(config.LLM_API_BASE and config.LLM_API_KEY and config.LLM_MODEL)


def _content_of(data) -> str | None:
    """从 OpenAI 兼容响应里取正文；取不到返回 None（**不抛裸异常**）。

    `data["choices"][0]["message"]["content"]` 这条链上每一环都可能缺：
    中转站过载时常见 HTTP 200 + `{"error": {...}}`、`{"choices": []}`、
    甚至 `choices[0]` 里只有 `delta` 没有 `message`。原来直接下标，抛的是
    KeyError / IndexError / TypeError——**都不是 LLMError**，于是绕过上层所有
    降级分支，以 500 + 裸 traceback 冒到用户面前。
    """
    try:
        choices = data["choices"]
        msg = choices[0].get("message") or choices[0].get("delta") or {}
        content = msg.get("content")
    except (KeyError, IndexError, TypeError, AttributeError):
        return None
    return content if isinstance(content, str) and content.strip() else None


def _explain_400(body: str) -> str:
    """把 400 的响应体翻译成「我该做什么」。

    400 是个筐：参数错、模型名错、内容被安全策略拦、**账户欠费**，全在里面。
    原来一律报「LLM 请求被拒（400，不重试）」再跟一段 JSON——批量任务跑到一半
    炸出几十条一模一样的长英文，人得自己去读 JSON 才知道是钱的问题。

    2026-09-09 实测：一批表格摘要跑到第 61 张时账户欠费，剩下 23 张全是
    `"type":"Arrearage"`。那不是代码问题，也不该让人从堆栈里刨。
    """
    low = (body or "").lower()
    hint = ""
    if "arrearage" in low or "overdue" in low or "good standing" in low:
        hint = ("**账户欠费**，模型调用已被服务方停掉。去控制台充值后重试；"
                "已经完成的部分都落库了，重跑会跳过它们（各模块的 pending() 是幂等的）。")
    elif "model" in low and ("not found" in low or "not exist" in low
                            or "invalid" in low):
        hint = "模型名可能不对。用 `python tools/check_models.py` 核一下 LLM_MODEL。"
    elif "data_inspection" in low or "content" in low and "policy" in low:
        hint = "内容被服务方的安全策略拦下了，换一段输入或换模型。"
    head = "LLM 请求被拒（400，不重试）："
    if hint:
        return head + hint + "\n原始响应：" + (body or "")[:200]
    return head + (body or "")[:200]


def _error_of(data) -> str | None:
    """响应体里 provider 自己写的错误说明（有就说明重试没有意义）。"""
    if not isinstance(data, dict):
        return None
    err = data.get("error")
    if isinstance(err, dict):
        return str(err.get("message") or err)
    if isinstance(err, str) and err.strip():
        return err
    for k in ("message", "detail"):
        v = data.get(k)
        if isinstance(v, str) and v.strip() and "choices" not in data:
            return v
    return None


def chat(system: str, user: str, purpose: str, paper_id: int | None = None,
         temperature: float = 0.3, model: str | None = None) -> str:
    if not available():
        raise LLMUnavailable("未配置 LLM_API_BASE / LLM_API_KEY / LLM_MODEL")
    budget.check("模型调用")     # 每日用量闸门：防失控循环烧光额度，不是限流
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
                raise LLMError(_explain_400(r.text))
            if r.status_code in (429, 500, 502, 503, 504):
                last = f"HTTP {r.status_code}"
                if attempt < 3:
                    deadline.sleep(delays[attempt])
                continue
            if r.status_code >= 400:  # 401/403/404：配置问题，重试无意义
                raise LLMError(f"LLM 请求被拒（HTTP {r.status_code}，不重试）：{r.text[:200]}")
            try:
                data = r.json()
            except ValueError:
                # 中转站过载时也可能回 HTTP 200 + 一段 HTML 错误页
                last = f"HTTP 200 但响应体不是 JSON：{r.text[:120]}"
                if attempt < 3:
                    deadline.sleep(delays[attempt])
                continue
            text = _content_of(data)
            if text is None:
                # **HTTP 200 + 错误体**是中转站最常见的失败形态。原来这里直接
                # `data["choices"][0]["message"]["content"]`，抛的是裸 KeyError——
                # 它不是 LLMError，于是绕过上层所有 `except LLMError` 的降级分支，
                # 以 500 + 裸 traceback 的形式冒到用户面前。
                err = _error_of(data)
                if err:
                    # provider 明确告诉我们错在哪（模型名错、余额不足…），重试无意义
                    raise LLMError(f"LLM 返回 HTTP 200 但带错误体（不重试）：{err[:200]}")
                last = f"HTTP 200 但响应里没有可用的 choices：{str(data)[:120]}"
                if attempt < 3:
                    deadline.sleep(delays[attempt])
                continue
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
            return text
    raise LLMError(f"LLM 连续 4 次失败：{last}")


def chat_stream(system: str, user: str, purpose: str, paper_id: int | None = None,
                temperature: float = 0.3, model: str | None = None):
    """流式对话：逐段 yield 文本 chunk；落账与非流式一致。

    重试边界：首 chunk 到达前失败可整体重试；已经开始输出后再断流，
    重试会导致重复文本，直接抛错交给上层处理。
    """
    if not available():
        raise LLMUnavailable("未配置 LLM_API_BASE / LLM_API_KEY / LLM_MODEL")
    budget.check("模型调用")     # 每日用量闸门：防失控循环烧光额度，不是限流
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


#: 尾逗号：`{"a": 1,}` / `[1, 2,]`。只在**紧跟右括号**时才算，不会误伤字符串里的逗号。
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def _loads_tolerant(blob: str):
    """解析模型给的 JSON，**所有失败都收敛成 `LLMError`**。

    这里原来是裸 `json.loads`。它对「括号配平但格式非法」抛的是
    `json.JSONDecodeError`（`ValueError` 子类，**不是 `LLMError`**），
    而全仓 6 个调用点只 `except llm.LLMError`——
    `cards.py:31`（L1 卡片）/ `pipeline.py:251,375`（选题、大纲）/
    `tablegen.py:40` / `writing.py:61,93`。也就是说模型只要犯一次
    **单引号**或**尾逗号**这两个最常见的毛病，这些功能就会直接崩，
    而它们各自旁边就摆着写好的离线兜底路径（`_offline_topics` / `_offline_outline` …）
    ——兜底进不去，等于白写。

    所以按「先严后宽、最后干净失败」三级：
      ① 严格 `json.loads`；
      ② 去掉尾逗号再试（模型最常见的毛病之一）；
      ③ `ast.literal_eval`（吃单引号与 Python 风格的 True/False/None；
         它**不执行代码**，只认字面量，比正则替换引号安全得多）。
    三级都不行才抛 `LLMError`，让调用方的兜底真的跑起来。
    """
    try:
        return json.loads(blob)
    except ValueError:
        pass
    try:
        return json.loads(_TRAILING_COMMA_RE.sub(r"", blob))
    except ValueError:
        pass
    try:
        import ast
        got = ast.literal_eval(blob)
        if isinstance(got, (dict, list)):
            return got
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        pass
    raise LLMError(f"JSON 解析失败：{blob[:160]}")


def _scan_balanced(text: str, open_ch: str, close_ch: str) -> str | None:
    r"""从第一个 `open_ch` 起扫到配对的 `close_ch`，**跳过字符串内部**。

    不认字符串的扫描器会被字符串里的括号骗到：符号抽取返回的
    `$\mathcal{T}_\mathrm{LM}$`、代码片段、LaTeX 公式里全是括号，
    提前截断之后拿到半截 JSON，症状看起来像「模型没按格式输出」。
    """
    start = text.find(open_ch)
    if start == -1:
        return None
    depth = 0
    in_str = escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == chr(92):
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _strip_fence(text: str) -> str:
    """剥掉模型爱加的 ```json 围栏。"""
    text = (text or "").strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) > 1:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:]
    return text.strip()


def extract_json_array(text: str) -> list:
    """从回复里取第一个 JSON 数组。**解析不了就抛 `LLMError`，不返回空列表。**

    这个函数原来住在 `symbols.py` 里，是 `extract_json` 的平行实现，
    带着同样的两个洞：括号扫描不认字符串、`json.loads` 失败就静默 `return []`。
    后者最要命——`symbols.extract_for_paper` 是「先 DELETE 再 INSERT」，
    静默的空列表会把用户已有的符号表清空后换成空的，而返回值只说 `count: 0`。
    所以这里改成抛异常，让调用方**必须**决定拿它怎么办。
    """
    blob = _scan_balanced(_strip_fence(text), "[", "]")
    if blob is None:
        raise LLMError("回复中没有 JSON 数组")
    got = _loads_tolerant(blob)
    if not isinstance(got, list):
        raise LLMError(f"期望 JSON 数组，拿到 {type(got).__name__}")
    return got


def extract_json(text: str) -> dict:
    """从回复中提取第一个完整 JSON 对象；兼容模型加的 ```json 围栏。

    扫描与容错解析都走 `_scan_balanced` / `_loads_tolerant`——**这里不许再写一遍**。
    这个函数原本把 `_scan_balanced` 的括号计数逐行抄了一份在自己体内，
    两份实现修一处漏一处；合并的时候我又把新版加在了文件最开头，
    于是同一个名字定义了两次：后定义的（旧实现）才生效，新版成了死代码，
    而排在模块 docstring 前面的那个 def 还把 `llm.__doc__` 变成了 None。
    """
    blob = _scan_balanced(_strip_fence(text), "{", "}")
    if blob is None:
        raise LLMError("回复中没有 JSON 对象")
    got = _loads_tolerant(blob)
    if not isinstance(got, dict):
        raise LLMError(f"期望 JSON 对象，拿到 {type(got).__name__}")
    return got
