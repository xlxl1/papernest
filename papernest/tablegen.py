"""方法对比表：从多篇文献的 L1 结构化卡片合成对比表，输出 Markdown 与 LaTeX（tabular）双格式。

写作痛点：相关工作章节需要「把 5 篇论文按维度对比」的表格——手工整理费时且易错。
数据全部来自库内已核验的结构化卡片，模型只做「组织」不做「编造」。
"""
import json

from . import config, db, llm

TABLE_SYSTEM = """你是学术写作助手。基于给定文献的结构化卡片，生成「相关工作方法对比表」。
输出 JSON 对象：
{"markdown": "Markdown 表格字符串，第一列为对比维度（问题设定/核心方法/关键结果/主要局限），之后每篇文献一列",
 "latex": "对应的 LaTeX 表格源码（\\begin{table*}...\\tabular，使用 booktabs 风格：\\toprule\\midrule\\bottomrule，列宽用 p{Xcm}）"}
硬性要求：表格内容只能来自给定卡片，不得编造；卡片中缺失的信息写「—」；只输出 JSON。"""


def compare(paper_ids: list[int], aspect: str = "") -> dict:
    if len(paper_ids) < 2:
        return {"error": "至少选择 2 篇文献"}
    if not llm.available():
        raise llm.LLMUnavailable("未配置 LLM")
    blocks = []
    with db.conn() as c:
        for i, pid in enumerate(paper_ids, 1):
            r = c.execute("SELECT * FROM papers WHERE id=?", (pid,)).fetchone()
            if not r:
                continue
            try:
                card = json.loads(r["card_json"] or "{}")
            except Exception:
                card = {}
            blocks.append(
                f"文献[{i}] {r['title']}（{r['venue'] or ''} {r['year'] or ''}）\n"
                f"  问题：{card.get('problem', '—')}\n  方法：{card.get('method', '—')}\n"
                f"  结果：{card.get('results', '—')}\n  局限：{card.get('limitations', '—')}")
    user = (f"对比维度侧重：{aspect or '问题设定、核心方法、关键结果、主要局限'}\n\n"
            "文献卡片：\n" + "\n\n".join(blocks))
    text = llm.chat(TABLE_SYSTEM, user, purpose="table", temperature=0.2,
                    model=config.heavy_model())
    try:
        data = llm.extract_json(text)
    except llm.LLMError:
        return {"markdown": text, "latex": "", "warn": "模型未按 JSON 输出，原文见 markdown"}
    return {"markdown": data.get("markdown", ""), "latex": data.get("latex", ""),
            "papers": paper_ids}
