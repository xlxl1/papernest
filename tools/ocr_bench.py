# -*- coding: utf-8 -*-
"""OCR 实测对比：本地 RapidOCR vs 云端 qwen-vl-ocr。

## 为什么要合成扫描件

本来打算拿库里的扫描件评，**结果库里一个都没有**：26 篇 PDF、233 页，
`0 字` 和 `<50 字` 的页各 0 个，98.3% 的页超过 300 字。
（此前笔记里记的「1.pdf / 477.pdf 无文本层」经实测不成立。）

没有对象就没法评。所以这里**用真论文合成扫描件**：把有文本层的 PDF 按指定 DPI
渲染成位图、再拼成一个纯图片 PDF，原来的文本层就是**逐字的 ground truth**。

这比找两个真扫描件评更好，不是将就：真扫描件没有 ground truth，只能靠人眼抽查
「看着差不多」；合成的能算出确切的字符错误率。代价是合成件比真扫描件干净
（没有倾斜、噪点、装订阴影、印章），所以**这里的数字是上界**——
真扫描件只会更差。报数时必须带着这句话。

## 指标

- **CER**（字符错误率）：编辑距离 / ground truth 长度。OCR 的标准指标，
  但**在双栏论文上它基本不可用**：PDF 文本层按栏读、OCR 按视觉读，
  同一批字换个顺序，编辑距离就爆掉。实测 463 第 1 页 CER 0.661，
  可词面召回是 96.2%——0.661 里绝大部分是顺序差异，不是认错字。
  留着它是为了看**同一引擎跨页的相对变化**，不要拿它比引擎。
- **词面召回**：ground truth 的词有多少在 OCR 结果里还找得到。
  **这个才是决策依据**——OCR 结果是拿去做检索的（FTS + 向量），
  漏词直接变成检索不到，而标点错、顺序错基本无所谓。
- **词面精确率**：OCR 出的词有多少在原文里有。低了说明模型在**编字**——
  这是云端大模型 OCR 特有的风险（它会「顺手把话补完整」），本地 OCR 反而不会。
- **耗时 / token**：本地是电（实测约 18s/页），云端是钱，得放在一起看。

用法：

    python tools/ocr_bench.py --pdf data/pdf/463.pdf --pages 1,2,3 --dpi 200
    python tools/ocr_bench.py --pdf data/pdf/463.pdf --engines rapidocr   # 不花钱
"""
from __future__ import annotations

import argparse
import base64
import io
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from papernest import config, ocr  # noqa: E402

#: 云端 OCR 模型：跟生产同一个，见 papernest/ocr.py。
CLOUD_MODEL = ocr.CLOUD_MODEL

#: 合成扫描件的渲染 DPI。150 偏糊、300 文件大且慢；真扫描件常见 200~300。
DEFAULT_DPI = 200

_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    """比对前的归一化：折叠空白、统一常见全角标点。

    **不去标点、不转小写**：那会把 OCR 的真实错误洗掉。只折叠排版差异——
    PDF 抽取和 OCR 对换行/连字的处理天生不同，那不是识别错误。
    """
    s = (s or "").replace("­", "").replace("-\n", "")
    s = s.translate(str.maketrans("　（）：，。；！？－", " ():,.;!?-"))
    return _WS.sub(" ", s).strip()


def _tokens(s: str) -> list[str]:
    """检索意义上的「词」：英文词、数字，以及中文二元组。与 db.expand_terms 同思路。"""
    s = s.lower()
    out = re.findall(r"[a-z0-9]+", s)
    out += [s[i:i + 2] for i in range(len(s) - 1)
            if re.fullmatch(r"[一-鿿]{2}", s[i:i + 2])]
    return out


def cer(truth: str, got: str) -> float:
    """字符错误率 = 编辑距离 / len(truth)。用 O(min(n,m)) 空间的 DP。"""
    a, b = _norm(truth), _norm(got)
    if not a:
        return 0.0 if not b else 1.0
    if len(a) < len(b):
        a, b = b, a          # 让内层循环短一点
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / max(1, len(_norm(truth)))


def token_scores(truth: str, got: str) -> dict:
    """词面召回与精确率——**阅读顺序无关**，所以能拿来比引擎（CER 不能）。

    召回低 = 漏字，检索就找不到；精确率低 = **编字**，那是云端大模型 OCR
    特有的失败方式（它会顺手把话补完整），本地 OCR 反而不会。
    """
    ts = set(_tokens(_norm(truth)))
    gs = set(_tokens(_norm(got)))
    if not ts:
        return {"recall": 1.0, "precision": 1.0, "hit": 0,
                "n_truth": 0, "n_got": len(gs)}
    hit = len(ts & gs)
    return {"recall": hit / len(ts), "precision": (hit / len(gs)) if gs else 0.0,
            "hit": hit, "n_truth": len(ts), "n_got": len(gs)}


# ── 合成扫描件 ────────────────────────────────────────────────────────────

def render_pages(pdf_path: str, pages: list[int], dpi: int) -> list[tuple[int, bytes, str]]:
    """把指定页渲染成 PNG，同时取回该页的文本层作为 ground truth。

    返回 `[(page_no, png_bytes, ground_truth_text)]`。
    """
    import pymupdf
    doc = pymupdf.open(pdf_path)
    out = []
    try:
        for pno in pages:
            if pno < 1 or pno > len(doc):
                continue
            page = doc[pno - 1]
            truth = page.get_text("text")
            pix = page.get_pixmap(dpi=dpi)
            out.append((pno, pix.tobytes("png"), truth))
    finally:
        doc.close()
    return out


def write_image_only_pdf(shots, dest: Path) -> Path:
    """把渲染出的位图拼成一个**没有文本层**的 PDF——合成的「扫描件」。"""
    import pymupdf
    doc = pymupdf.open()
    try:
        for _, png, _ in shots:
            img = pymupdf.open("png", png)
            rect = img[0].rect
            page = doc.new_page(width=rect.width, height=rect.height)
            page.insert_image(rect, stream=png)
            img.close()
        dest.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(dest))
    finally:
        doc.close()
    return dest


# ── 两个引擎：**从 papernest.ocr 复用，不在这里重写** ──────────────────
#
# 本仓刚栽过一次：`llm.extract_json` 因为「新版加在文件开头、旧版没删」而同名
# 定义了两份，后定义的旧实现才生效、新版成了死代码。评测脚本和生产代码各写
# 一份 OCR 调用，是同一个坑的更隐蔽版本——评测出来的数字将不再代表生产行为。

def ocr_rapidocr(png: bytes) -> str:
    return ocr.ocr_local(png)


def ocr_cloud(png: bytes, model: str = "") -> tuple[str, dict]:
    """跑评测时**不记账到库里**（会污染真库的 llm_calls），所以自己发请求。

    但用的是 `papernest.ocr` 的同一段提示词与参数——那才是生产行为。
    """
    import json
    import urllib.request
    b64 = base64.b64encode(png).decode("ascii")
    payload = {"model": model or ocr.CLOUD_MODEL, "temperature": 0.0,
               "messages": [{"role": "user", "content": [
                   {"type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"}},
                   {"type": "text", "text": ocr._PROMPT}]}]}
    req = urllib.request.Request(
        config.LLM_API_BASE.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": "Bearer " + config.LLM_API_KEY,
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.loads(r.read().decode("utf-8"))
    txt = ((d.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    if isinstance(txt, list):
        txt = "".join(p.get("text", "") for p in txt if isinstance(p, dict))
    u = d.get("usage") or {}
    return txt, {"prompt_tokens": u.get("prompt_tokens") or 0,
                 "completion_tokens": u.get("completion_tokens") or 0}


# ── 主流程 ────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pdf", required=True, help="拿哪篇真论文合成扫描件")
    ap.add_argument("--pages", default="1,2,3", help="页码，逗号分隔")
    ap.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    ap.add_argument("--engines", default="rapidocr,cloud",
                    help="rapidocr / cloud，逗号分隔。只写 rapidocr 就一分钱不花。")
    ap.add_argument("--model", default=CLOUD_MODEL)
    ap.add_argument("--out", default="", help="把合成的扫描件 PDF 存到这里（可选）")
    args = ap.parse_args()

    sys.stdout.reconfigure(errors="replace")
    pages = [int(x) for x in args.pages.replace(",", " ").split()]
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]

    print(f"源 PDF : {args.pdf}")
    print(f"页码   : {pages}   渲染 DPI: {args.dpi}")
    shots = render_pages(args.pdf, pages, args.dpi)
    if not shots:
        print("没渲染出任何页"); return 1
    print(f"渲染完成：{len(shots)} 页，位图共 "
          f"{sum(len(p) for _, p, _ in shots) / 1024 / 1024:.1f} MB")

    if args.out:
        dest = write_image_only_pdf(shots, Path(args.out))
        import pymupdf
        d = pymupdf.open(str(dest))
        left = sum(len(d[i].get_text("text").strip()) for i in range(len(d)))
        d.close()
        print(f"合成扫描件：{dest}（文本层残留 {left} 字符——应当是 0）")

    rows = []
    for eng in engines:
        tot_tok = {"prompt_tokens": 0, "completion_tokens": 0}
        per = []
        t0 = time.time()
        for pno, png, truth in shots:
            try:
                if eng == "rapidocr":
                    got, u = ocr_rapidocr(png), {}
                else:
                    got, u = ocr_cloud(png, args.model)
                    for k in tot_tok:
                        tot_tok[k] += u.get(k, 0)
            except Exception as exc:                      # noqa: BLE001
                print(f"  [{eng}] 第 {pno} 页失败：{type(exc).__name__}: {exc}")
                continue
            sc = token_scores(truth, got)
            per.append({"page": pno, "cer": cer(truth, got), **sc,
                        "truth_chars": len(_norm(truth)), "got_chars": len(_norm(got)),
                        "text": got})
        dt = time.time() - t0
        if not per:
            continue
        rows.append({"engine": eng, "secs": dt, "tokens": tot_tok, "per": per})

    print("\n" + "=" * 74)
    print(f"{'引擎':<14}{'页':>3}{'词面召回':>10}{'精确率':>9}{'CER*':>8}"
          f"{'字符 真/出':>16}{'耗时':>9}{'token':>10}")
    for r in rows:
        n = len(r["per"])
        hit = sum(p["hit"] for p in r["per"])
        mrec = hit / max(1, sum(p["n_truth"] for p in r["per"]))
        mpre = hit / max(1, sum(p["n_got"] for p in r["per"]))
        mcer = sum(p["cer"] for p in r["per"]) / n
        tc = sum(p["truth_chars"] for p in r["per"])
        gc = sum(p["got_chars"] for p in r["per"])
        tk = r["tokens"].get("prompt_tokens", 0) + r["tokens"].get("completion_tokens", 0)
        print(f"{r['engine']:<14}{n:>3}{mrec:>10.1%}{mpre:>9.1%}{mcer:>8.3f}"
              f"{tc:>8,}/{gc:<7,}{r['secs']:>8.1f}s{tk:>10,}")
    print("=" * 78)
    print("* **看召回与精确率，不要看 CER**：双栏论文里 PDF 文本层按栏读、")
    print("  OCR 按视觉读，同一批字换个顺序编辑距离就爆掉——CER 里绝大部分是")
    print("  顺序差异，不是认错字（实测同一页 CER 0.66 而召回 96%）。")
    print("  召回低 = 漏字（检索找不到）；精确率低 = 编字（云端大模型特有的失败方式）。")
    print("注意：合成扫描件比真扫描件干净（无倾斜 / 噪点 / 装订阴影），")
    print("      所以这些数字是**上界**，真件只会更差。")

    for r in rows:
        worst = min(r["per"], key=lambda p: p["recall"])
        print(f"\n--- {r['engine']} 召回最低的一页（第 {worst['page']} 页，"
              f"召回 {worst['recall']:.1%}，精确率 {worst['precision']:.1%}）前 300 字 ---")
        print(worst["text"][:300])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
