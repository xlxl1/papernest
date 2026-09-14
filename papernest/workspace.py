"""Persistence helpers for the PaperNest research workspace.

The original application treated papers as the only durable object.  A real
research workflow also needs a project, notes, drafts and explicit links from
claims in a draft back to papers.  This module keeps those operations small,
transactional and independent from FastAPI so they are easy to reuse in CLI,
background tasks and tests.
"""
from __future__ import annotations

import json
from typing import Any

from . import db


def _dict(row: Any) -> dict[str, Any] | None:
    return dict(row) if row else None


def _tags(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return _tags(parsed)
        except json.JSONDecodeError:
            return [x.strip() for x in value.split(",") if x.strip()]
    return []


def create_project(name: str, description: str = "", topic: str = "") -> dict[str, Any]:
    name = name.strip()
    if not name:
        raise ValueError("项目名称不能为空")
    db.init_db()
    with db.conn() as conn:
        cur = conn.execute(
            "INSERT INTO projects(name,description,topic) VALUES(?,?,?)",
            (name, description.strip(), topic.strip()),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM projects WHERE id=?", (cur.lastrowid,)).fetchone()
    return _dict(row) or {}


def list_projects() -> list[dict[str, Any]]:
    db.init_db()
    with db.conn() as conn:
        rows = conn.execute(
            """SELECT p.*, COUNT(DISTINCT n.id) AS note_count,
                      COUNT(DISTINCT d.id) AS document_count
               FROM projects p
               LEFT JOIN notes n ON n.project_id=p.id
               LEFT JOIN documents d ON d.project_id=p.id
               GROUP BY p.id ORDER BY p.updated_at DESC, p.id DESC"""
        ).fetchall()
    return [_dict(row) or {} for row in rows]


def get_project(project_id: int) -> dict[str, Any] | None:
    db.init_db()
    with db.conn() as conn:
        row = conn.execute(
            """SELECT p.*, COUNT(DISTINCT n.id) AS note_count,
                      COUNT(DISTINCT d.id) AS document_count
               FROM projects p
               LEFT JOIN notes n ON n.project_id=p.id
               LEFT JOIN documents d ON d.project_id=p.id
               WHERE p.id=? GROUP BY p.id""", (project_id,)
        ).fetchone()
    return _dict(row)


def update_project(project_id: int, *, name: str | None = None,
                   description: str | None = None,
                   topic: str | None = None) -> dict[str, Any] | None:
    current = get_project(project_id)
    if not current:
        return None
    new_name = (name.strip() if name is not None else current["name"]).strip()
    if not new_name:
        raise ValueError("项目名称不能为空")
    with db.conn() as conn:
        conn.execute(
            """UPDATE projects SET name=?, description=?, topic=?,
                      updated_at=datetime('now','localtime') WHERE id=?""",
            (new_name,
             description.strip() if description is not None else current["description"],
             topic.strip() if topic is not None else current["topic"], project_id),
        )
        conn.commit()
    return get_project(project_id)


def delete_project(project_id: int) -> bool:
    with db.conn() as conn:
        cur = conn.execute("DELETE FROM projects WHERE id=?", (project_id,))
        conn.commit()
    return cur.rowcount > 0


def _note_dict(row: Any) -> dict[str, Any]:
    out = _dict(row) or {}
    out["tags"] = _tags(out.pop("tags_json", "[]"))
    return out


def list_notes(project_id: int, paper_id: int | None = None) -> list[dict[str, Any]]:
    db.init_db()
    sql = "SELECT * FROM notes WHERE project_id=?"
    args: list[Any] = [project_id]
    if paper_id is not None:
        sql += " AND paper_id=?"
        args.append(paper_id)
    sql += " ORDER BY updated_at DESC, id DESC"
    with db.conn() as conn:
        rows = conn.execute(sql, args).fetchall()
    return [_note_dict(row) for row in rows]


def get_note(note_id: int) -> dict[str, Any] | None:
    with db.conn() as conn:
        row = conn.execute("SELECT * FROM notes WHERE id=?", (note_id,)).fetchone()
    return _note_dict(row) if row else None


def create_note(project_id: int, title: str, content: str = "",
                note_type: str = "insight", tags: list[str] | None = None,
                paper_id: int | None = None) -> dict[str, Any]:
    if not get_project(project_id):
        raise ValueError("项目不存在")
    title = title.strip() or "未命名笔记"
    with db.conn() as conn:
        cur = conn.execute(
            """INSERT INTO notes(project_id,paper_id,title,content,note_type,tags_json)
               VALUES(?,?,?,?,?,?)""",
            (project_id, paper_id, title, content, note_type or "insight",
             json.dumps(_tags(tags), ensure_ascii=False)),
        )
        conn.execute("UPDATE projects SET updated_at=datetime('now','localtime') WHERE id=?",
                     (project_id,))
        conn.commit()
        row = conn.execute("SELECT * FROM notes WHERE id=?", (cur.lastrowid,)).fetchone()
    return _note_dict(row)


def update_note(note_id: int, *, title: str | None = None, content: str | None = None,
                note_type: str | None = None, tags: list[str] | None = None,
                paper_id: int | None = None) -> dict[str, Any] | None:
    current = get_note(note_id)
    if not current:
        return None
    with db.conn() as conn:
        conn.execute(
            """UPDATE notes SET title=?, content=?, note_type=?, tags_json=?,
                      paper_id=?, updated_at=datetime('now','localtime') WHERE id=?""",
            (title.strip() if title is not None else current["title"],
             content if content is not None else current["content"],
             note_type if note_type is not None else current["note_type"],
             json.dumps(_tags(tags if tags is not None else current["tags"]),
                        ensure_ascii=False),
             paper_id if paper_id is not None else current["paper_id"], note_id),
        )
        conn.execute("UPDATE projects SET updated_at=datetime('now','localtime') WHERE id=?",
                     (current["project_id"],))
        conn.commit()
    return get_note(note_id)


def delete_note(note_id: int) -> bool:
    with db.conn() as conn:
        cur = conn.execute("DELETE FROM notes WHERE id=?", (note_id,))
        conn.commit()
    return cur.rowcount > 0


def _document_dict(row: Any) -> dict[str, Any]:
    out = _dict(row) or {}
    doc_id = out.get("id")
    with db.conn() as conn:
        sources = conn.execute(
            """SELECT ds.*, p.title, p.doi, p.arxiv_id
               FROM document_sources ds JOIN papers p ON p.id=ds.paper_id
               WHERE ds.document_id=? ORDER BY ds.paper_id, ds.page_no""", (doc_id,)
        ).fetchall()
    out["sources"] = [_dict(source) or {} for source in sources]
    return out


def list_documents(project_id: int) -> list[dict[str, Any]]:
    db.init_db()
    with db.conn() as conn:
        rows = conn.execute(
            "SELECT * FROM documents WHERE project_id=? ORDER BY updated_at DESC, id DESC",
            (project_id,),
        ).fetchall()
    return [_document_dict(row) for row in rows]


def get_document(document_id: int) -> dict[str, Any] | None:
    with db.conn() as conn:
        row = conn.execute("SELECT * FROM documents WHERE id=?", (document_id,)).fetchone()
    return _document_dict(row) if row else None


def create_document(project_id: int, title: str, content: str = "",
                    status: str = "draft") -> dict[str, Any]:
    if not get_project(project_id):
        raise ValueError("项目不存在")
    with db.conn() as conn:
        cur = conn.execute(
            "INSERT INTO documents(project_id,title,content,status) VALUES(?,?,?,?)",
            (project_id, title.strip() or "未命名文档", content, status or "draft"),
        )
        conn.execute("UPDATE projects SET updated_at=datetime('now','localtime') WHERE id=?",
                     (project_id,))
        conn.commit()
        row = conn.execute("SELECT * FROM documents WHERE id=?", (cur.lastrowid,)).fetchone()
    return _document_dict(row)


def update_document(document_id: int, *, title: str | None = None,
                    content: str | None = None, status: str | None = None) -> dict[str, Any] | None:
    current = get_document(document_id)
    if not current:
        return None
    with db.conn() as conn:
        conn.execute(
            """UPDATE documents SET title=?, content=?, status=?, version=version+1,
                      updated_at=datetime('now','localtime') WHERE id=?""",
            (title.strip() if title is not None else current["title"],
             content if content is not None else current["content"],
             status if status is not None else current["status"], document_id),
        )
        conn.execute("UPDATE projects SET updated_at=datetime('now','localtime') WHERE id=?",
                     (current["project_id"],))
        conn.commit()
    return get_document(document_id)


def delete_document(document_id: int) -> bool:
    with db.conn() as conn:
        cur = conn.execute("DELETE FROM documents WHERE id=?", (document_id,))
        conn.commit()
    return cur.rowcount > 0


def attach_source(document_id: int, paper_id: int, page_no: int | None = None,
                  quote: str = "", relation: str = "support") -> dict[str, Any]:
    if not get_document(document_id):
        raise ValueError("文档不存在")
    with db.conn() as conn:
        if not conn.execute("SELECT 1 FROM papers WHERE id=?", (paper_id,)).fetchone():
            raise ValueError("论文不存在")
        conn.execute(
            """INSERT INTO document_sources(document_id,paper_id,page_no,quote,relation)
               VALUES(?,?,?,?,?)
               -- 冲突目标必须用 COALESCE 那条表达式索引：复合主键里 page_no 可空，
               -- 而 UNIQUE 索引里 NULL 互不相等，按主键做 upsert 永远不触发
               -- （page_no 默认就是 None，同一条引用挂几次就存几行）。
               ON CONFLICT(document_id,paper_id,COALESCE(page_no,-1))
               DO UPDATE SET quote=excluded.quote, relation=excluded.relation""",
            (document_id, paper_id, page_no, quote, relation or "support"),
        )
        conn.commit()
    return get_document(document_id) or {}


def detach_source(document_id: int, paper_id: int, page_no: int | None = None) -> bool:
    with db.conn() as conn:
        cur = conn.execute(
            "DELETE FROM document_sources WHERE document_id=? AND paper_id=? "
            "AND page_no IS ?", (document_id, paper_id, page_no),
        )
        conn.commit()
    return cur.rowcount > 0

