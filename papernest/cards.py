"""L1 结构化卡片生成。

铁律：卡片只能基于「标题 + 摘要」（W2 的 L2 才允许全文），提示词里明令
不得编造摘要之外的信息；找不到依据的字段如实标注，不硬编。
无 key 时走 mock 摘要直取模式，字段全部显式标注，绝不假装是 LLM 分析。
"""
import json

from . import config, llm

CARD_SYSTEM = """你是学术文献分析助手。基于给定的论文标题与摘要，输出 JSON 卡片，
字段与要求：
- tldr: 一句话总结（不超过 50 字，中文）
- problem: 研究问题 / 动机（1-2 句）
- method: 方法要点（1-3 句）
- results: 主要结果，尽量保留摘要中的具体数字
- positioning: 该文献的核心定位与价值，基于摘要概括它实际做了什么、对谁有用（1-2 句）；
  若与我给的课题直接相关，可点明关联；不相关就不要提课题，绝不写「与课题无关」之类的话
- keywords: 3-6 个关键词（中文或英文原词）
硬性要求：所有字段只能依据标题与摘要，不得编造摘要之外的信息，不要推测摘要没写的局限；
摘要中没有依据的字段填「（摘要未提及）」。只输出 JSON，不要多余文字。"""


def make_card(paper: dict, topic: str) -> tuple[dict, str]:
    """返回 (card, model)。model 记录生成方式，mock 显式标注。"""
    if llm.available():
        user = (f"我的课题：{topic or config.RESEARCH_TOPIC}\n\n"
                f"标题：{paper['title']}\n\n摘要：{paper.get('abstract') or '（无摘要）'}")
        text = llm.chat(CARD_SYSTEM, user, purpose="card_l1",
                        paper_id=paper.get("_db_id"), temperature=0.3)
        try:
            card = llm.extract_json(text)
        except llm.LLMError:
            card = {"tldr": text[:80], "problem": "（卡片解析失败，原文见备注）",
                    "raw": text[:500]}
            return card, f"{config.LLM_MODEL}+parse-fallback"
        return card, config.LLM_MODEL
    return mock_card(paper), "mock-extractive"


def mock_card(paper: dict) -> dict:
    ab = (paper.get("abstract") or "").strip()
    first_sentence = ab.split("。")[0] if "。" in ab else ab[:120]
    tail = ab[len(first_sentence):].strip()[:200]
    return {
        "tldr": first_sentence or "（无摘要，仅元数据）",
        "problem": "（mock 模式：未接 LLM，本卡从摘要直取，配置 .env 后 cli.py recard 重刷）",
        "method": "（mock）",
        "results": tail or "（摘要未提及）",
        "positioning": "（mock）",
        "keywords": [],
        "_note": "mock-extractive：仅摘录摘要，非 LLM 分析",
    }


def recard_all(topic: str | None = None, limit: int | None = None,
               force_all: bool = False) -> int:
    """重刷卡片；force_all=True 时全量重刷（提示词变更后用）。"""
    from . import db
    done = 0
    where = "" if force_all else "WHERE card_model='mock-extractive'"
    with db.conn() as c:
        rows = c.execute(f"SELECT * FROM papers {where} ORDER BY id").fetchall()
    for row in rows:
        if limit and done >= limit:
            break
        paper = {"_db_id": row["id"], "title": row["title"],
                 "abstract": row["abstract"]}
        card, model = make_card(paper, topic or config.RESEARCH_TOPIC)
        with db.conn() as c:
            db.save_card(c, row["id"], card, model)
            c.commit()
        done += 1
    return done


def card_json(raw: str) -> dict:
    try:
        return json.loads(raw)
    except Exception:
        return {"raw": raw}
