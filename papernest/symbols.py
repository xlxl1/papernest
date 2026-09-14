"""符号与缩写大全：从 L2 全文抽取符号/缩写定义，LLM 归一化 LaTeX 源码。

用途：读文献时查「这个记号什么意思」，写作时复制 LaTeX 到 Overleaf；
MathType（Word 插件）支持「粘贴 TeX」，同一份 LaTeX 两边通用。
只抽论文中明确定义过的符号与首次展开的缩写，不臆造。
"""
import json
import re

from . import config, db, http, llm

SYM_SYSTEM = """你是学术论文符号抽取助手。从给定的论文文本（带页码）中抽取：
1. 明确定义过的数学符号：如 "where α denotes the step size"、"x ∈ R^n"、公式中的记号定义；
2. 首次展开的缩写：如 "channel state information (CSI)"。
输出 JSON 数组，每项：
{"kind": "symbol" 或 "abbrev",
 "sym": "符号原文（如 α、‖h‖₂、CSI）",
 "latex": "该符号的 LaTeX 源码（如 \\alpha、\\|\\mathbf{h}\\|_2），缩写此项留空字符串",
 "meaning": "含义，中文不超过 25 字",
 "page": 定义所在页码整数}
硬性要求：只抽文本中明确定义或首次展开的；不超过 30 条；文本中没有就输出 []。
只输出 JSON 数组，不要多余文字。"""


#: 数组解析已收口到 `llm.extract_json_array`——这里原来是它的平行实现，
#: 带着同样的两个洞：括号扫描不认字符串（而本模块返回的正是
#: `$\mathcal{T}_\mathrm{LM}$` 这种满是括号的 LaTeX）、解析失败静默返回 []。
extract_json_array = llm.extract_json_array


def extract_for_paper(paper_id: int) -> dict:
    """从已抽取全文的论文中抽符号。需要先 read（有 pages）。"""
    if not llm.available():
        raise llm.LLMUnavailable("未配置 LLM")
    with db.conn() as c:
        pages = {r["page_no"]: r["text"] for r in
                 c.execute("SELECT page_no,text FROM pages WHERE paper_id=? ORDER BY page_no",
                           (paper_id,)).fetchall()}
        row = c.execute("SELECT title FROM papers WHERE id=?", (paper_id,)).fetchone()
    if not pages:
        return {"error": "该论文还没有全文（先执行 L2 全文精读/抽取），才能抽取符号"}

    body = "\n\n".join(f"【第 {pno} 页】\n{txt[:2600]}" for pno, txt in sorted(pages.items()))
    text = llm.chat(SYM_SYSTEM, f"标题：{row['title']}\n\n{body[:50000]}",
                    purpose="symbols", paper_id=paper_id, temperature=0.1)
    try:
        raw_items = extract_json_array(text)
    except llm.LLMError as exc:
        # **解析不了就什么都不动**。下面是「先 DELETE 再 INSERT」，
        # 模型返回一次垃圾就会把用户已有的符号表清空后换成空的，
        # 而返回值只说 count=0——那是静默的破坏性失败。
        return {"paper_id": paper_id, "count": 0, "kept_existing": True,
                "error": f"模型输出解析失败，已保留原有符号：{exc}"}
    items = [x for x in raw_items if isinstance(x, dict) and x.get("sym")]
    if not items:
        return {"paper_id": paper_id, "count": 0, "kept_existing": True,
                "error": f"模型返回了 {len(raw_items)} 条但没有一条带 sym 字段，"
                         f"已保留原有符号（未覆盖）"}
    with db.conn() as c:
        c.execute("DELETE FROM symbols WHERE paper_id=?", (paper_id,))
        for x in items[:40]:
            c.execute("""INSERT INTO symbols(paper_id,kind,sym,latex,meaning,page)
                         VALUES(?,?,?,?,?,?)""",
                      (paper_id, x.get("kind") or "symbol", str(x.get("sym"))[:40],
                       str(x.get("latex") or "")[:120], str(x.get("meaning") or "")[:80],
                       int(x.get("page") or 0)))
        c.commit()
    return {"paper_id": paper_id, "count": len(items[:40])}


def _builtin(q: str = "") -> list[dict]:
    """内置符号库（写作常用全集）。q 匹配符号/中文名/LaTeX。"""
    from . import symbols_builtin
    ql = q.strip().lower()
    out = []
    for x in symbols_builtin.SYMBOLS:
        if ql and ql not in x["sym"].lower() and ql not in x["name"].lower()                 and ql not in x["latex"].lower() and ql not in x["cat"].lower():
            continue
        out.append({"kind": "builtin", "sym": x["sym"], "latex": x["latex"],
                    "name": x["name"], "cat": x["cat"], "meanings": []})
    return out


def library_symbols(q: str = "", kind: str = "", limit: int = 200) -> list[dict]:
    """符号库 = 内置常用符号全集（写作直接查）+ 论文抽取的领域符号（聚合多篇含义）。"""
    if kind != "abbrev":  # 内置表没有缩写
        items = _builtin(q)
    else:
        items = []
    sql = """SELECT s.*, p.title, p.year FROM symbols s JOIN papers p ON p.id=s.paper_id
             WHERE 1=1"""
    args: list = []
    if q:
        like = f"%{q}%"
        sql += " AND (s.sym LIKE ? OR s.meaning LIKE ? OR p.title LIKE ?)"
        args += [like, like, like]
    if kind in ("symbol", "abbrev"):
        sql += " AND s.kind=?"
        args.append(kind)
    sql += " ORDER BY s.sym, s.page LIMIT ?"
    args.append(limit)
    with db.conn() as c:
        rows = [dict(r) for r in c.execute(sql, args).fetchall()]
    agg: dict[str, dict] = {}
    for r in rows:
        key = f"{r['kind']}|{r['sym']}"
        e = agg.setdefault(key, {"kind": r["kind"], "sym": r["sym"], "latex": r["latex"],
                                 "meanings": []})
        e["meanings"].append({"meaning": r["meaning"], "paper": r["title"][:40],
                              "year": r["year"], "page": r["page"]})
    return items + list(agg.values())
