"""Deterministic evaluation harness for the multi-agent writing pipeline.

The benchmark deliberately lives outside the application pipeline.  Every
topic/mode gets a fresh temporary SQLite database seeded with the same source
material; failures are recorded per item and do not stop the remaining run.
"""
from __future__ import annotations

import hashlib
import json
import re
import tempfile
import time
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

from . import config, db, embeddings, llm, pipeline, writing


RUNNER_VERSION = "p5-1"
_NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?%?")
_DEFAULT_DATASET_NAMES = ("writing_eval_set.json", "writing_benchmark.json",
                          "writing_eval.json")
_BENCHMARK_LOCK = threading.RLock()


def _numbers(text: str) -> list[str]:
    return _NUMBER_RE.findall(str(text or ""))


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _dataset_hash(dataset: dict) -> str:
    return hashlib.sha256(_canonical(dataset).encode("utf-8")).hexdigest()


def _default_path() -> Path:
    for name in _DEFAULT_DATASET_NAMES:
        p = Path(__file__).resolve().parent.parent / name
        if p.exists():
            return p
    raise FileNotFoundError("未找到 writing_eval_set.json（请提供 path）")


def load_writing_eval_set(path: str | Path | None = None) -> dict:
    """Load and validate the fixed P5 dataset.

    Quotes are checked literally against their source abstract.  Source and
    topic identifiers are normalized to stable strings while preserving the
    authored order, making the resulting hash reproducible.
    """
    p = Path(path) if path else _default_path()
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("评测集顶层必须是对象")
    if not isinstance(raw.get("topics"), list) or not isinstance(raw.get("sources"), list):
        raise ValueError("评测集必须包含 topics 和 sources 数组")
    if raw.get("schema_version", 1) != 1:
        raise ValueError(f"不支持的写作评测集版本：{raw.get('schema_version')!r}")
    if not raw["topics"] or not raw["sources"]:
        raise ValueError("写作评测集不能没有主题或来源")
    out = {"schema_version": raw.get("schema_version", 1),
           "name": str(raw.get("name") or p.stem),
           "description": str(raw.get("description") or ""),
           "topics": [], "sources": []}
    source_keys: set[str] = set()
    sources_by_key: dict[str, dict] = {}
    for i, src in enumerate(raw["sources"], 1):
        if not isinstance(src, dict):
            raise ValueError(f"source[{i}] 不是对象")
        key = str(src.get("source_key") or "").strip()
        if not key or key in source_keys:
            raise ValueError(f"source[{i}] source_key 无效或重复")
        title = str(src.get("title") or "").strip()
        abstract = str(src.get("abstract") or "")
        if not title or not abstract:
            raise ValueError(f"source {key} 缺 title/abstract")
        norm_key = str(src.get("norm_key") or "").strip()
        if not norm_key:
            raise ValueError(f"source {key} 缺 norm_key")
        clean = {"source_key": key, "norm_key": norm_key,
                 "title": title, "abstract": abstract,
                 "year": src.get("year"), "authors": src.get("authors") or []}
        source_keys.add(key); sources_by_key[key] = clean; out["sources"].append(clean)
    topic_ids: set[str] = set()
    claim_ids: set[str] = set()
    for i, topic in enumerate(raw["topics"], 1):
        if not isinstance(topic, dict):
            raise ValueError(f"topic[{i}] 不是对象")
        tid = str(topic.get("id") or "").strip()
        if not tid or tid in topic_ids:
            raise ValueError(f"topic[{i}] id 无效或重复")
        sections = topic.get("sections")
        if not isinstance(sections, list) or not sections:
            raise ValueError(f"topic {tid} 缺 sections")
        ns: list[dict] = []
        section_ids: set[str] = set()
        for j, sec in enumerate(sections, 1):
            if not isinstance(sec, dict):
                raise ValueError(f"topic {tid} section[{j}] 不是对象")
            sid = str(sec.get("id") or f"{tid}-section-{j}")
            if sid in section_ids:
                raise ValueError(f"topic {tid} section id 重复：{sid}")
            section_ids.add(sid)
            claims = sec.get("claims") or []
            if not isinstance(claims, list) or not claims:
                raise ValueError(f"section {sid} claims 必须是数组")
            nclaims: list[dict] = []
            for k, claim in enumerate(claims, 1):
                if not isinstance(claim, dict):
                    raise ValueError(f"claim {sid}/{k} 不是对象")
                text = str(claim.get("text") or claim.get("claim") or "").strip()
                sk = str(claim.get("source_key") or "").strip()
                quote = str(claim.get("quote") or "")
                if not text or sk not in sources_by_key or not quote:
                    raise ValueError(f"claim {sid}/{k} 缺 text/source_key/quote")
                if quote not in sources_by_key[sk]["abstract"]:
                    raise ValueError(f"claim {sid}/{k} quote 不在 source abstract 中")
                nums = claim.get("numbers")
                nums = [str(x) for x in nums] if isinstance(nums, list) else _numbers(quote)
                claim_id = str(claim.get("claim_id") or f"{sid}-claim-{k}")
                if claim_id in claim_ids:
                    raise ValueError(f"claim_id 重复：{claim_id}")
                if any(n not in quote for n in nums):
                    raise ValueError(f"claim {sid}/{k} numbers 不在 quote 中")
                claim_ids.add(claim_id)
                nclaims.append({"claim_id": claim_id,
                                "text": text, "source_key": sk, "quote": quote,
                                "numbers": nums})
            keys = [str(x) for x in (sec.get("source_keys") or [])]
            for c in nclaims:
                if c["source_key"] not in keys: keys.append(c["source_key"])
            if any(key not in sources_by_key for key in keys):
                raise ValueError(f"section {sid} source_keys 含未知来源")
            ns.append({"id": sid, "no": int(sec.get("no") or j),
                       "title": str(sec.get("title") or sid),
                       "points": [str(x) for x in (sec.get("points") or [])],
                       "claims": nclaims, "source_keys": keys})
        out["topics"].append({"id": tid, "query": str(topic.get("query") or topic.get("title") or tid),
                              "title": str(topic.get("title") or tid), "sections": ns})
        topic_ids.add(tid)
    out["dataset_hash"] = _dataset_hash(out)
    return out


@contextmanager
def _isolated_db():
    # config.DB_PATH/DATA_DIR and the db init caches are process globals.  A
    # benchmark must not race an HTTP worker or another benchmark while it
    # temporarily redirects them to the fixture database.
    with _BENCHMARK_LOCK:
        old_db, old_dir = config.DB_PATH, config.DATA_DIR
        old_init = getattr(db, "_initialized", set()).copy()
        old_wal = getattr(db, "_WAL_DONE", set()).copy()
        with tempfile.TemporaryDirectory(prefix="papernest_write_bench_") as td:
            config.DATA_DIR = Path(td) / "data"
            config.DB_PATH = config.DATA_DIR / "bench.sqlite"
            try:
                # Do not allow an in-process cache for the application's DB to
                # leak into the benchmark database.
                if hasattr(db, "_initialized"): db._initialized = set()
                if hasattr(db, "_WAL_DONE"): db._WAL_DONE = set()
                db.init_db(force=True)
                yield Path(td)
            finally:
                config.DB_PATH, config.DATA_DIR = old_db, old_dir
                if hasattr(db, "_initialized"): db._initialized = old_init
                if hasattr(db, "_WAL_DONE"): db._WAL_DONE = old_wal


def _seed_sources(sources: list[dict]) -> dict[str, int]:
    ids: dict[str, int] = {}
    with db.conn() as c:
        for src in sources:
            pid = db.insert_l0(c, {"norm_key": src["norm_key"], "title": src["title"],
                                   "abstract": src["abstract"], "year": src.get("year"),
                                   "venue": "benchmark", "authors": src.get("authors") or [],
                                   "source": "benchmark"})
            ids[src["source_key"]] = pid
            db.replace_chunks(c, pid, [{"text": src["abstract"], "section_path": "abstract",
                                       "start_page": None, "kind": "text", "block_type": "text"}])
        c.commit()
    return ids


def _ledger() -> dict[str, Any]:
    with db.conn() as c:
        rows = c.execute("SELECT prompt_tokens,completion_tokens,cost_usd,latency_ms FROM llm_calls").fetchall()
    if not rows:
        return {"llm_calls": 0, "prompt_tokens": None, "completion_tokens": None,
                "cost_usd": None, "latency_ms": None}
    ptok = sum((r[0] or 0) for r in rows)
    ctok = sum((r[1] or 0) for r in rows)
    # Some compatible endpoints omit usage.  llm.chat persists that omission
    # as zeroes; zero is therefore an unknown token count, not a measured one.
    return {"llm_calls": len(rows),
            "prompt_tokens": ptok if ptok else None,
            "completion_tokens": ctok if ctok else None,
            "cost_usd": (sum(r[2] for r in rows if r[2] is not None)
                          if any(r[2] is not None for r in rows) else None),
            "latency_ms": sum((r[3] or 0) for r in rows) if any(r[3] is not None for r in rows) else None}


def _claim_metrics(text: str, topic: dict, sources: list[dict], source_ids: dict[str, int],
                   bundles: Iterable[dict] = (), ref_keys: list[str] | None = None,
                   audited_major: int | None = None) -> tuple[dict, list[dict]]:
    """Compute per-topic metrics against the independent gold evaluator.

    The ``serial`` and ``multi`` modes hand a verified, claim-bound citation
    whitelist to the writer.  String-matching ``[n]`` against the gold
    ``source_key`` is therefore *not* a fair comparison — those modes may
    legitimately cite a paper that exists in the bundle but is not part of
    the gold claim set (the writer is allowed to ground additional context
    beyond the gold claims).  We expose two complementary metrics:

    * ``citation_precision``: fraction of references in the final text that
      trace to a verified, claim-bound paper (anything else is at best an
      unverified recall).  This is what the multi-agent contract promises.
    * ``citation_string_match``: legacy string-vs-source-key ratio, kept for
      compatibility with older reports and the offline baseline.

    The same logic lets the numeric-fidelity measure track whether the
    numbers actually appear in the body, *and* whether their associated
    claim is verified, instead of just substring-searching the text.
    """
    claims = [c for s in topic["sections"] for c in s.get("claims", [])]
    by_id = {c["source_key"]: c for c in claims}
    refs: dict[int, str] = {i: key for i, key in enumerate(ref_keys or [s["source_key"] for s in sources], 1)}
    used = [int(x) for x in re.findall(r"\[(\d+)\]", text or "")]
    cited_keys = [refs[n] for n in used if n in refs]
    cited_set = set(cited_keys)

    # Build the verified source-key set from the bundles (when available).
    # Only keys that actually appear as a paper_id in a verified piece of
    # evidence bound to at least one claim count.  This keeps the writer's
    # citation promise falsifiable instead of rewarding any ``[n]`` hit.
    verified_keys: set[str] = set()
    bound_keys: set[str] = set()
    bundle_has_evidence = False
    for bundle in bundles:
        if not isinstance(bundle, dict):
            continue
        evidence = bundle.get("evidence") or []
        claims_b = bundle.get("claims") or []
        if evidence:
            bundle_has_evidence = True
        by_evidence_id = {
            str(e.get("evidence_id")): e for e in evidence if isinstance(e, dict)
        }
        for claim in claims_b:
            if not isinstance(claim, dict):
                continue
            ids = claim.get("evidence_ids") or []
            if isinstance(ids, (str, int)):
                ids = [ids]
            for eid in ids:
                ev = by_evidence_id.get(str(eid))
                if not ev or not ev.get("verified"):
                    continue
                if str(ev.get("source_kind")) not in {"native_page", "imported_fulltext"}:
                    continue
                pid = ev.get("paper_id")
                if pid is None:
                    continue
                # paper_id is a database key; map back to the gold source_key
                # through the ``source_ids`` dict (built in _seed_sources).
                gold_key = next((k for k, v in (source_ids or {}).items() if int(v) == int(pid)), None)
                if gold_key:
                    verified_keys.add(gold_key)
                    bound_keys.add(gold_key)

    if cited_keys:
        if bundle_has_evidence:
            precision = (sum(1 for k in cited_keys if k in verified_keys) / len(cited_keys))
        else:
            precision = (sum(1 for k in cited_keys if k in by_id) / len(cited_keys))
        string_match = (sum(1 for k in cited_keys if k in by_id) / len(cited_keys))
    else:
        precision = 0.0
        string_match = 0.0

    # Numeric fidelity: only consider numbers tied to a claim whose source is
    # actually cited and verified.  Otherwise we reward model text that
    # happens to include the right digit anywhere in the body.
    fidelity_pairs: list[tuple[str, str]] = []
    for c in claims:
        if c["source_key"] not in cited_set:
            continue
        for n in (c.get("numbers") or _numbers(c["quote"])):
            fidelity_pairs.append((c["source_key"], n))
    if fidelity_pairs:
        fidelity_hits = sum(1 for _, n in fidelity_pairs if n in (text or ""))
        numeric_fidelity = fidelity_hits / len(fidelity_pairs)
    else:
        numeric_fidelity = 1.0  # no numbers to verify ⇒ vacuously true

    # Evidence coverage & unsupported rate: prefer bundle truth when
    # available, otherwise fall back to the gold presence check.
    verified_claims = 0
    bound_claim_count = 0
    for bundle in bundles:
        for claim in bundle.get("claims", []) or []:
            if not isinstance(claim, dict):
                continue
            ids = claim.get("evidence_ids") or []
            if isinstance(ids, (str, int)):
                ids = [ids]
            if ids:
                bound_claim_count += 1
            if claim.get("status") == "supported":
                verified_claims += 1

    if bundle_has_evidence:
        evidence_coverage = verified_claims / len(claims) if claims else 0.0
    else:
        gold_covered = sum(1 for c in claims if c["source_key"] in cited_set and c["quote"] in text)
        evidence_coverage = gold_covered / len(claims) if claims else 0.0

    gold_numbers = [n for c in claims for n in (c.get("numbers") or _numbers(c["quote"]))]
    if gold_numbers:
        referenced_numbers = [n for c in claims if c["source_key"] in cited_set
                              for n in (c.get("numbers") or _numbers(c["quote"]))]
        gaps = max(0, len(claims) - sum(1 for c in claims if c["source_key"] in cited_set))
    else:
        gaps = 0

    if audited_major is not None:
        gaps = max(gaps, int(audited_major))
    unsupported = gaps / max(len(claims), 1)
    return ({"evidence_coverage": round(evidence_coverage, 3),
             "citation_precision": round(precision, 3),
             "citation_string_match": round(string_match, 3),
             "numeric_fidelity": round(numeric_fidelity, 3),
             "unsupported_claim_rate": round(unsupported, 3),
             "verified_source_count": len(verified_keys)},
            [{"claim_id": c["claim_id"], "source_key": c["source_key"],
              "cited": c["source_key"] in cited_set,
              "verified_source": c["source_key"] in verified_keys,
              "quote_present": c["quote"] in (text or "")}
             for c in claims])


def _topic_sections(topic: dict, source_ids: dict[str, int]) -> list[dict]:
    out = []
    for sec in topic["sections"]:
        claims = sec.get("claims") or []
        out.append({"no": sec["no"], "title": sec["title"],
                    "points": sec.get("points") or [c["text"] for c in claims],
                    "claims": claims, "paper_ids": [source_ids[k] for k in sec.get("source_keys", []) if k in source_ids],
                    "words": 300})
    return out


def benchmark_topic(topic: dict, sources: list[dict], mode: str, offline: bool = True,
                    max_rewrites: int = 2,
                    budget_state: dict | None = None) -> dict:
    """Run one topic in one mode and return flat metrics plus ``metrics``.

    ``budget_state`` (optional) is a mutable dict shared across runs of the same
    benchmark invocation.  When set, the function records the wall-clock and
    early-exits with a budget error if the cumulative limit was breached
    before this topic started.
    """
    started = time.perf_counter()
    result: dict[str, Any] = {"mode": mode, "topic_id": topic.get("id"),
                              "offline": bool(offline), "error": None}
    try:
        if mode not in {"single", "serial", "multi"}: raise ValueError(f"未知模式: {mode}")
        if not offline and not llm.available():
            raise RuntimeError("在线写作评测需要可用的 LLM；未配置时请去掉 --online 使用离线基线")
        if not offline and not llm.available():
            raise llm.LLMUnavailable("online 评测需要已配置且可用的 LLM；请使用 offline=True 做模板链路验证")
        # Keep the direct function safe for callers that provide an in-memory
        # fixture instead of going through load_writing_eval_set.  The loader
        # performs the same check, but silently treating an empty quote as a
        # verified claim would make a benchmark look better than its data.
        source_by_key = {str(s.get("source_key")): s for s in sources}
        for sec in topic.get("sections") or []:
            for claim in sec.get("claims") or []:
                sk = str(claim.get("source_key") or "")
                quote = str(claim.get("quote") or "")
                if not sk or sk not in source_by_key:
                    raise ValueError(f"主张 {claim.get('claim_id', '')} 的 source_key 不存在")
                if not quote or quote not in str(source_by_key[sk].get("abstract") or ""):
                    raise ValueError(f"主张 {claim.get('claim_id', '')} 的 quote 未通过来源原文核验")
        patches = []
        if offline:
            from unittest import mock
            patches = [mock.patch.object(llm, "available", return_value=False),
                       mock.patch.object(embeddings, "available", return_value=False)]
            for p in patches: p.start()
        with _isolated_db():
            source_ids = _seed_sources(sources)
            claims = [c for s in topic["sections"] for c in s.get("claims", [])]
            if mode == "single":
                # Single-shot baseline: same source materials as the multi-pass
                # conditions receive (title, points, source abstracts) so the
                # comparison measures the *coordination* gain rather than
                # “got more context”.  Gold claim text and quotes are still
                # withheld — they belong to the independent evaluator.
                bullets = "\n".join(
                    f"第 {sec.get('no', i + 1)} 节 {sec.get('title') or ''}："
                    + "；".join(str(p) for p in (sec.get('points') or []))
                    for i, sec in enumerate(topic.get('sections') or []))
                sources_block = "\n".join(
                    f"[{i + 1}] {s.get('title') or s.get('source_key') or ''}：{s.get('abstract') or ''}"
                    for i, s in enumerate(sources))
                if not offline and llm.available():
                    text = llm.chat(
                        "你是单 Agent 学术写作者。事实必须能在指定来源中找到，"
                        "每条事实后用 [n] 引用，禁止编造数字与文献。",
                        f"课题：{topic.get('title') or topic.get('query') or ''}\n"
                        f"章节要点：\n{bullets}\n\n"
                        f"来源：\n{sources_block}",
                        purpose="writing_benchmark_single", temperature=0.3)
                else:
                    text = "\n\n".join(
                        f"## {sec.get('no', i + 1)}. {sec.get('title') or ''}\n"
                        + "\n".join(str(point) for point in (sec.get('points') or []))
                        for i, sec in enumerate(topic.get('sections') or []))
                bundles = []; ref_keys = [s["source_key"] for s in sources]; audited_major = 0
                rewrites = 0
            else:
                run_goal = topic.get("query") or topic.get("title") or topic["id"]
                if len(str(run_goal).strip()) < 4:
                    run_goal = topic.get("title") or f"benchmark {topic['id']}"
                run = pipeline.create_run(str(run_goal),
                                          target_words=3000, max_rewrites=max_rewrites)
                rid = run["id"]
                with db.conn() as c:
                    c.execute("UPDATE writing_runs SET title=?, glossary_json=? WHERE id=?",
                              (topic.get("title") or topic["id"], "{}", rid))
                    c.execute("INSERT INTO outlines(run_id,version,title,outline_json,warnings_json) VALUES(?,?,?,?,?)",
                              (rid, 1, topic.get("title") or topic["id"], json.dumps(_topic_sections(topic, source_ids), ensure_ascii=False), "[]"))
                    c.commit()
                if mode == "serial":
                    run = pipeline.get_run(rid); outline = pipeline.get_outline(rid)
                    for sec in outline["sections"]:
                        pipeline._write_one(run, sec, 1, force=True, evidence_bundle=None)
                    text, refs, infos = pipeline.assemble(rid)
                    bundles = [i.get("evidence_bundle") or {} for i in infos]
                else:
                    pipeline.stage_sections(rid)
                    text, refs, infos = pipeline.assemble(rid)
                    bundles = [i.get("evidence_bundle") or {} for i in infos]
                inverse = {pid: key for key, pid in source_ids.items()}
                ref_keys = [inverse.get(v.get("paper_id"), "") for _, v in sorted(refs.items())]
                audited_major = sum(int((i.get("audit") or {}).get("major_count") or 0) for i in infos)
                rewrites = sum(max(0, int(i.get("attempt") or 1) - 1) for i in infos)
            metrics, claim_rows = _claim_metrics(text, topic, sources, source_ids, bundles,
                                                 ref_keys=ref_keys, audited_major=audited_major)
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            metrics.update({"rewrite_count": rewrites, "latency_ms": elapsed_ms})
            ledger = _ledger()
            # The benchmark wall-clock latency remains available even when no
            # model call was made (offline runs have no ledger latency).
            ledger.pop("latency_ms", None)
            metrics.update(ledger)
            # ``citation_string_match`` and ``verified_source_count`` are
            # already present from _claim_metrics; ``citation_accuracy`` and
            # ``number_fidelity`` are kept as the legacy alias names.
            metrics["citation_accuracy"] = metrics["citation_precision"]
            metrics["number_fidelity"] = metrics["numeric_fidelity"]
            # ``unsupported_retention`` is no longer a degenerate 1-rate alias;
            # it now reports the fraction of pre-existing unsupported claims
            # (number=0 ⇒ N/A).  Until that number is well calibrated by the
            # new metrics, we report ``None`` to avoid misleading aggregations.
            metrics["unsupported_retention"] = None
            metrics["usage_known"] = bool(metrics.get("prompt_tokens") is not None
                                          or metrics.get("completion_tokens") is not None)
            result["model"] = config.heavy_model() if not offline else "offline-template"
            result["runner_version"] = RUNNER_VERSION
            result.update({"metrics": metrics, **metrics, "claims": claim_rows, "text": text})
        return result
    except Exception as exc:  # each benchmark item is isolated
        result["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        result["metrics"] = {"rewrite_count": 0, "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                              "llm_calls": 0, "prompt_tokens": None, "completion_tokens": None, "cost_usd": None}
        result.update(result["metrics"])
        return result
    finally:
        for p in locals().get("patches", []):
            try: p.stop()
            except RuntimeError: pass


def _mean(items: list[dict], key: str):
    vals = [i[key] for i in items if i.get("error") is None and isinstance(i.get(key), (int, float))]
    return round(sum(vals) / len(vals), 4) if vals else None


def run_benchmark(path: str | Path | None = None, modes: tuple[str, ...] = ("single", "serial", "multi"),
                  limit: int | None = None, offline: bool = True,
                  out_path: str | Path | None = None,
                  budget_guard: dict | None = None) -> dict:
    """Run all (mode, topic) combinations under an isolated database.

    ``budget_guard`` (optional) is a dict consumed by :func:`_enforce_budget`.
    When provided, the runner aborts the remaining items the moment any
    ceiling is exceeded and records the reason on every unfinished item.
    """
    modes = tuple(dict.fromkeys(str(mode).strip() for mode in modes if str(mode).strip()))
    unknown = sorted(set(modes) - {"single", "serial", "multi"})
    if unknown:
        raise ValueError(f"未知写作评测模式：{', '.join(unknown)}")
    if not modes:
        raise ValueError("至少指定一种写作评测模式")
    dataset = load_writing_eval_set(path)
    topics = dataset["topics"][:limit] if limit is not None else dataset["topics"]
    source_list = dataset["sources"]
    items: list[dict] = []
    budget_state = dict(budget_guard or {})
    budget_state.setdefault("max_total_cost_usd", None)
    budget_state.setdefault("max_llm_calls", None)
    budget_state.setdefault("max_total_tokens", None)
    budget_state.setdefault("max_wallclock_s", None)
    budget_state["started_at"] = time.perf_counter()
    budget_state["cost_usd"] = 0.0
    budget_state["llm_calls"] = 0
    budget_state["prompt_tokens"] = 0
    budget_state["completion_tokens"] = 0
    budget_state["stop_reason"] = None
    infeasible = _check_budget_guard_feasibility(budget_state)
    if infeasible:
        raise RuntimeError(f"预算护栏不可执行：{infeasible}")
    for mode in modes:
        for topic in topics:
            if budget_state.get("stop_reason"):
                break
            try:
                item = benchmark_topic(topic, source_list, mode, offline=offline,
                                       budget_state=budget_state)
                _update_budget_from_item(budget_state, item)
                if _enforce_budget(budget_state):
                    break
            except Exception as exc:  # isolate one bad topic/mode
                fail_metrics = {"evidence_coverage": None, "citation_precision": None,
                                 "numeric_fidelity": None, "unsupported_claim_rate": None,
                                 "citation_accuracy": None, "number_fidelity": None,
                                 "unsupported_retention": None, "rewrite_count": 0,
                                 "latency_ms": None, "llm_calls": 0,
                                 "prompt_tokens": None, "completion_tokens": None,
                                 "cost_usd": None}
                item = {"mode": mode, "topic_id": topic.get("id"),
                        "offline": bool(offline),
                        "model": config.heavy_model() if not offline else "offline-template",
                        "runner_version": RUNNER_VERSION,
                        "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                        "metrics": fail_metrics}
                item.update(fail_metrics)
            items.append(item)
    metric_names = ("evidence_coverage", "citation_precision", "citation_string_match",
                    "numeric_fidelity", "unsupported_claim_rate",
                    "citation_accuracy", "number_fidelity",
                    "rewrite_count", "latency_ms", "llm_calls",
                    "prompt_tokens", "completion_tokens", "cost_usd",
                    "verified_source_count")
    extra_keys = ("usage_known",)
    aggregate = {key: _mean(items, key) for key in metric_names}
    for key in extra_keys:
        vals = [i.get(key) for i in items if isinstance(i.get(key), bool)]
        if vals:
            aggregate[key] = round(sum(1 for v in vals if v) / len(vals), 3)
        else:
            aggregate[key] = None
    by_mode = {}
    for mode in modes:
        mode_items = [i for i in items if i.get("mode") == mode]
        by_mode[mode] = {key: _mean(mode_items, key) for key in metric_names}
        for key in extra_keys:
            vals = [i.get(key) for i in mode_items if isinstance(i.get(key), bool)]
            if vals:
                by_mode[mode][key] = round(sum(1 for v in vals if v) / len(vals), 3)
            else:
                by_mode[mode][key] = None
    result = {"dataset_hash": dataset.get("dataset_hash") or _dataset_hash(dataset), "runner_version": RUNNER_VERSION,
              "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "offline": bool(offline),
              "model": config.heavy_model() if not offline else "offline-template", "modes": list(modes),
              "items": items, "aggregate": aggregate, "by_mode": by_mode,
              "failure_count": sum(1 for i in items if i.get("error"))}
    if budget_state.get("stop_reason"):
        result["budget_stopped"] = budget_state["stop_reason"]
        result["budget_used"] = {k: budget_state.get(k) for k in
                                 ("cost_usd", "llm_calls", "prompt_tokens",
                                  "completion_tokens", "max_total_cost_usd",
                                  "max_llm_calls", "max_total_tokens")}
    if out_path:
        Path(out_path).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _check_budget_guard_feasibility(guard: dict) -> str | None:
    """Pre-flight check that a budget guard is enforceable.

    Returns ``None`` when the guard is safe to use, or a human-readable
    error message describing why the guard cannot be enforced (which should
    abort the run before any LLM call is made).
    """
    if not guard:
        return None
    if guard.get("max_total_cost_usd") is not None and not config.LLM_PRICES_JSON:
        return ("cost_usd 闸门不可用：未配置 LLM_PRICES_JSON；"
                "请配置计价表或去掉 --max-cost-usd")
    return None


def _update_budget_from_item(state: dict, item: dict) -> None:
    if not isinstance(item, dict):
        return
    metrics = item.get("metrics") or {}
    cost = metrics.get("cost_usd")
    calls = metrics.get("llm_calls")
    ptok = metrics.get("prompt_tokens")
    ctok = metrics.get("completion_tokens")
    if isinstance(cost, (int, float)):
        state["cost_usd"] = float(state.get("cost_usd") or 0) + float(cost)
    if isinstance(calls, (int, float)):
        state["llm_calls"] = int(state.get("llm_calls") or 0) + int(calls)
    if isinstance(ptok, (int, float)):
        state["prompt_tokens"] = int(state.get("prompt_tokens") or 0) + int(ptok)
    if isinstance(ctok, (int, float)):
        state["completion_tokens"] = int(state.get("completion_tokens") or 0) + int(ctok)


def _enforce_budget(state: dict) -> bool:
    """Return True if the budget guard demands an early stop."""
    now = time.perf_counter()
    if (state.get("max_total_cost_usd") is not None
            and state["cost_usd"] >= float(state["max_total_cost_usd"])):
        state["stop_reason"] = f"cost_usd ≥ {state['max_total_cost_usd']}"
        return True
    if (state.get("max_llm_calls") is not None
            and state["llm_calls"] >= int(state["max_llm_calls"])):
        state["stop_reason"] = f"llm_calls ≥ {state['max_llm_calls']}"
        return True
    total_tok = (state.get("prompt_tokens") or 0) + (state.get("completion_tokens") or 0)
    if (state.get("max_total_tokens") is not None
            and total_tok >= int(state["max_total_tokens"])):
        state["stop_reason"] = f"tokens ≥ {state['max_total_tokens']}"
        return True
    if state.get("max_wallclock_s") is not None:
        elapsed = now - state["started_at"]
        if elapsed >= float(state["max_wallclock_s"]):
            state["stop_reason"] = f"wallclock ≥ {state['max_wallclock_s']}s"
            return True
    return False


__all__ = ["load_writing_eval_set", "benchmark_topic", "run_benchmark",
           "BUDGET_KEYS"]


BUDGET_KEYS = ("max_total_cost_usd", "max_llm_calls",
               "max_total_tokens", "max_wallclock_s")
