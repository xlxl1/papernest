# -*- coding: utf-8 -*-
"""扫描件 OCR：给没有文本层的 PDF 一条出路。

## 在此之前

没有文本层的 PDF 走完整条入库流程会得到 `papers=1 / pages=0 / chunks=0`，
而 `NO_TEXT_LAYER_WARNING` 只能告诉用户「本项目不做 OCR，请自己找外部工具转」。
这个模块把那条死路接上。

## 两个引擎，实测对比过

真库里**一份扫描件都没有**（26 篇 233 页，0 字页与 <50 字页各 0 个），
所以不能拿库里的文件评。改用**真论文合成扫描件**：按 DPI 渲染成位图、
去掉文本层，原来的文本层就是逐字 ground truth（`tools/ocr_bench.py`）。

两篇论文各 4 页，200 DPI：

    引擎                        词面召回   词面精确率   token/页   耗时/页
    RapidOCR（本地，onnxruntime） 91.3/90.8%  85.9/86.4%      0      17s
    qwen-vl-ocr（云端）           92.7/93.7%  95.8/95.0%   ~4850    13~24s

**看召回和精确率，不要看 CER**：双栏论文里 PDF 文本层按栏读、OCR 按视觉读，
同一批字换个顺序编辑距离就爆掉（同一页 CER 0.66 而召回 96%）。

差距主要在**精确率 +9~10pt**，而且不是「云端在编字」——是 RapidOCR 按视觉顺序
吐碎片，把 `ad-\\ncessed` 这种断词拆成两个原文里没有的词面；云端会还原阅读顺序、
把连字拼回去。对本项目这直接等于检索命中率。

**给 RapidOCR 加到 300 DPI 没有帮助**（召回 91.2%、精确率 85.4%），耗时反而 1.7 倍
——所以差距不是分辨率造成的，这条堵住了「参数没给公平」的质疑。

**这些数字是上界**：合成扫描件没有倾斜、噪点、装订阴影和印章，真扫描件只会更差。

## `qwen3.5-ocr` 会**静默丢掉表格**——别拿它读论文

账户欠费后只剩免费额度可用，`qwen3.5-ocr` 是仅存的几个能调的模型之一，
所以专门测了它。整页口径看着还行（463 召回 86.5%、480 召回 92.8%，精确率 95~96%），
**但按数值词面单看就露馅了**：

    463 第 3 页（正文 + 两张结果表）   数值召回  qwen3.5-ocr 21.5%(14/65)   RapidOCR 100%(65/65)
    463 第 5 页（几乎整页是表）       数值召回  qwen3.5-ocr 100%(100/100)  RapidOCR  98%(98/100)

第 3 页的输出里 `Table` / `BLEU` / `LeBLEU` / `|` **各出现 0 次**：它把正文转完就停在脚注，
整片跳过了两张表，`finish_reason=stop`（不是截断），**同一张图跑三次结果一字不差**。
表格挤满整页时它躲不开（第 5 页 100%），正文里夹着表就直接不转。

对本项目这是最坏的一类失败：**扫描论文的价值大半在结果表里，而它丢得悄无声息**。
所以 `qwen3.5-ocr` 进了 `WEAK_ON_TABLES`，选它会给一条明确警告。

## 默认关闭

云端要花钱（一页约 4850 token，一篇 20 页的扫描件约 9.7 万），本地要花 17s/页。
两样都不该在用户没说话的时候发生，所以 `PAPERNEST_OCR` 默认 `off`。
"""
from __future__ import annotations

import base64
import io
import json
import os
import time

from . import config

#: 云端 OCR 模型。默认 qwen-vl-ocr-latest——实测里它是唯一两页都稳的。
CLOUD_MODEL = os.environ.get("PAPERNEST_OCR_CLOUD_MODEL", "qwen-vl-ocr-latest")

#: 实测会**静默丢掉表格**的模型：整页召回看着正常，数值词面掉到 21.5%，
#: 而且完全可复现（同图三次一字不差）。选它必须先把这句话说清楚。
WEAK_ON_TABLES = {"qwen3.5-ocr"}

WEAK_ON_TABLES_WARNING = (
    "⚠ {model} 实测会**整片跳过页面里的表格**：正文照常转，表格一个字不出，"
    "也不报错（463 第 3 页数值召回 21.5%，同图跑三次一字不差）。"
    "论文的结果大多在表里——要读带表的扫描件，改用 "
    "PAPERNEST_OCR_CLOUD_MODEL=qwen-vl-ocr-latest，或退回 PAPERNEST_OCR=local"
    "（本地精确率低些，但数值召回 98~100%）。")


def model_warning(model: str | None = None) -> str:
    """选的模型有没有已知的坑。没有就返回空串。"""
    m = model or CLOUD_MODEL
    return WEAK_ON_TABLES_WARNING.format(model=m) if m in WEAK_ON_TABLES else ""

#: 渲染 DPI。实测 300 对本地引擎没有改善而耗时 1.7 倍，所以 200 是默认。
DPI = config._env_int(200, "PAPERNEST_OCR_DPI", minimum=72)

#: `off`（默认）/ `local`（RapidOCR）/ `cloud`（qwen-vl-ocr）。
MODE = (os.environ.get("PAPERNEST_OCR", "off") or "off").strip().lower()

#: 单页实测用量（合成扫描件，qwen-vl-ocr-latest，200 DPI）：8 页共 38,731 token。
MEASURED_CLOUD_TOKENS_PER_PAGE = 4850

#: OCR 出来的文本要标出来：它不是 PDF 的文本层，是识别出来的，有错字。
MARK = "【OCR 识别，非文本层】"

_PROMPT = ("把这一页的文字**原样**转成纯文本，保持阅读顺序，"
           "不要总结、不要翻译、不要加任何说明。")

_rapid = None


class OcrUnavailable(RuntimeError):
    """选的引擎装不上 / 没配好。**显式抛**：静默跳过会让人以为 OCR 做过了。"""


def available() -> dict:
    """哪些引擎现在能用。不做任何调用，不花钱。"""
    local = False
    try:
        import rapidocr_onnxruntime  # noqa: F401
        local = True
    except Exception:                                   # noqa: BLE001
        local = False
    cloud = bool(config.LLM_API_BASE and config.LLM_API_KEY)
    return {"local": local, "cloud": cloud, "mode": MODE,
            "cloud_model": CLOUD_MODEL, "dpi": DPI}


def enabled() -> bool:
    return MODE in ("local", "cloud")


def estimate(n_pages: int, mode: str | None = None) -> dict:
    """动手前的账。**一次接口都不调。**"""
    m = (mode or MODE).lower()
    if m == "cloud":
        tok = n_pages * MEASURED_CLOUD_TOKENS_PER_PAGE
        return {"mode": m, "n_pages": n_pages, "est_total_tokens": tok,
                "est_seconds": n_pages * 20,
                "note": f"按实测 {MEASURED_CLOUD_TOKENS_PER_PAGE} token/页"}
    return {"mode": m, "n_pages": n_pages, "est_total_tokens": 0,
            "est_seconds": n_pages * 17,
            "note": "本地引擎不花钱，按实测 17s/页"}


def render_page(pdf_path: str, page_no: int, dpi: int | None = None) -> bytes:
    """把一页渲染成 PNG。"""
    import pymupdf
    doc = pymupdf.open(pdf_path)
    try:
        return doc[page_no - 1].get_pixmap(dpi=dpi or DPI).tobytes("png")
    finally:
        doc.close()


def ocr_local(png: bytes) -> str:
    """RapidOCR（onnxruntime 后端）：不联网、不花钱、约 17s/页。"""
    global _rapid
    if _rapid is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as e:
            raise OcrUnavailable(
                "没装本地 OCR：pip install rapidocr_onnxruntime") from e
        _rapid = RapidOCR()
    import numpy as np
    from PIL import Image
    img = np.array(Image.open(io.BytesIO(png)).convert("RGB"))
    res, _ = _rapid(img)
    return "\n".join(line[1] for line in res) if res else ""


def ocr_cloud(png: bytes, model: str | None = None,
              paper_id: int | None = None) -> str:
    """qwen-vl-ocr：更准（精确率 +9~10pt），但要花钱（约 4850 token/页）。"""
    import urllib.request

    from . import budget, db
    if not (config.LLM_API_BASE and config.LLM_API_KEY):
        raise OcrUnavailable("未配置 LLM_API_BASE / LLM_API_KEY，云端 OCR 用不了")
    budget.check("OCR")
    m = model or CLOUD_MODEL
    b64 = base64.b64encode(png).decode("ascii")
    payload = {"model": m, "temperature": 0.0, "messages": [{"role": "user",
               "content": [{"type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"}},
                           {"type": "text", "text": _PROMPT}]}]}
    req = urllib.request.Request(
        config.LLM_API_BASE.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": "Bearer " + config.LLM_API_KEY,
                 "Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.loads(r.read().decode("utf-8"))
    txt = ((d.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    if isinstance(txt, list):               # 有的网关把 content 返成分段列表
        txt = "".join(p.get("text", "") for p in txt if isinstance(p, dict))
    u = d.get("usage") or {}
    try:                                    # OCR 也要进账本，否则闸门看不见它
        with db.conn() as c:
            db.add_llm_call(c, "ocr", paper_id, u.get("prompt_tokens", 0),
                            u.get("completion_tokens", 0), m,
                            latency_ms=round((time.time() - t0) * 1000, 1))
            c.commit()
    except Exception:                                   # noqa: BLE001
        pass                                # 记账失败不该把已经拿到的文本扔掉
    return txt


def ocr_pdf(pdf_path: str, max_pages: int = 50, mode: str | None = None,
            progress=None, paper_id: int | None = None) -> list[tuple[int, str]]:
    """整份 PDF 逐页 OCR，返回 `[(page_no, text)]`（空页会被丢掉）。

    **单页失败不中断**：扫描件里夹一张纯图的插页很常见，为它丢掉整份不值。
    """
    import pymupdf
    m = (mode or MODE).lower()
    if m not in ("local", "cloud"):
        raise OcrUnavailable(f"OCR 未启用（PAPERNEST_OCR={MODE}）")
    doc = pymupdf.open(pdf_path)
    n = min(doc.page_count, max_pages)
    doc.close()
    if m == "cloud":
        w = model_warning()
        if w and progress:
            progress(0.0, w)
    out: list[tuple[int, str]] = []
    for i in range(1, n + 1):
        if progress:
            progress(i / max(1, n), f"OCR 第 {i}/{n} 页（{m}）")
        try:
            png = render_page(pdf_path, i)
            txt = ocr_local(png) if m == "local" else ocr_cloud(png, paper_id=paper_id)
        except OcrUnavailable:
            raise                           # 引擎压根用不了：这是配置问题，要冒出去
        except Exception:                                # noqa: BLE001
            continue                        # 单页失败：跳过，别丢掉整份
        if txt.strip():
            out.append((i, f"{MARK}\n{txt.strip()}"))
    return out
