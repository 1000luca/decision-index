import collections
import hashlib
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from decision_index.engines import NativeAbstention, Unsupported, load_engine, validate
from decision_index.suite.io import atomic_json, dumps, read_jsonl
from decision_index.vision.rows import ROWS_FILE


def stamp():
    return datetime.now(timezone.utc).isoformat()


def outcome(payload, r):
    if isinstance(r, NativeAbstention):
        return {"status": "abstained", "error": str(r)}
    if isinstance(r, Unsupported):
        return {"status": "unsupported", "error": str(r)}
    if isinstance(r, Exception):
        return {"status": "error", "error": str(r), "exception": type(r).__name__,
                "traceback": "".join(traceback.format_exception(type(r), r, r.__traceback__))}
    response, raw = r
    try:
        validate(payload["questions"], response)
    except Exception as e:
        return {"status": "error", "error": str(e), "exception": type(e).__name__}
    return {"status": "ok", "response": response, "raw_output": raw}


def run(engine_name, engine_options, root, out_dir, rows_path=None, workers=1, chunk=256, limit=None, compact=False, resume=True, log=print):
    root = Path(root).resolve()
    rows_path = Path(rows_path or root / ROWS_FILE)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def event(**kw):
        record = {"time": stamp(), **kw}
        log(dumps(record))
        atomic_json(out / "status.json", record, indent=None)

    t = time.perf_counter()
    event(event="loading", engine=engine_name)
    engine = load_engine(engine_name, image_root=str(root), **engine_options)
    atomic_json(out / "environment.json", {"engine": engine_name, "engine_options": engine_options, "model_source": engine.provenance, **engine.runtime(),
                                           "loaded_seconds": time.perf_counter() - t, "rows_path": str(rows_path),
                                           "rows_sha256": hashlib.sha256(rows_path.read_bytes()).hexdigest(), "latency": engine.latency})
    event(event="ready", engine=engine_name)
    results_path = out / "results.jsonl"
    previous = {}
    if resume and results_path.exists():
        for r in read_jsonl(results_path, complete_lines_only=True):
            previous[r["run_id"]] = r["status"]
        text = results_path.read_text(encoding="utf-8")
        if text and not text.endswith("\n"):
            results_path.write_text(text[: text.rfind("\n") + 1], encoding="utf-8")
    todo = [r for r in read_jsonl(rows_path) if previous.get(r["_evaluation"]["run_id"]) in (None, "error")]
    if limit:
        todo = todo[:limit]
    chunks = []
    for r in todo:
        if chunks and chunks[-1][0]["set"] == r["set"] and len(chunks[-1]) < chunk:
            chunks[-1].append(r)
        else:
            chunks.append([r])
    counts = collections.Counter()
    start = time.perf_counter()
    pool = ThreadPoolExecutor(max(1, workers))

    def call(p):
        s = time.perf_counter()
        try:
            r = engine(**p)
        except Exception as e:
            r = e
        return r, (time.perf_counter() - s) * 1000

    with results_path.open("a", encoding="utf-8") as logf:
        for rows in chunks:
            payloads = [{"state": r["state"], "questions": r["questions"], "images": r["images"], "prompt": r.get("prompt")} for r in rows]
            started = stamp()
            if hasattr(engine, "batch"):
                s = time.perf_counter()
                res = engine.batch(payloads)
                per = (time.perf_counter() - s) * 1000 / len(rows)
                timed = [(r, per) for r in res]
            else:
                timed = list(pool.map(call, payloads))
            for row, payload, (r, ms) in zip(rows, payloads, timed):
                result = {**row["_evaluation"], "started_utc": started, "engine": engine_name, **outcome(payload, r)}
                if compact:
                    result.pop("raw_output", None)
                else:
                    result["payload"] = payload
                result.update(completed_utc=stamp(), total_wall_ms=ms, model_request_wall_ms=ms)
                logf.write(dumps(result) + "\n")
                counts[result["status"]] += 1
            logf.flush()
            event(event="progress", engine=engine_name, completed=sum(counts.values()), total=len(todo), counts=dict(counts), elapsed_seconds=round(time.perf_counter() - start, 1))
            if counts["error"] >= 5 and counts["ok"] == 0:
                event(event="failed", engine=engine_name, counts=dict(counts), error="five failed requests without a success")
                raise RuntimeError("Five failed requests without a success; stop instead of filling the benchmark with runtime errors")
    pool.shutdown()
    engine.close()
    final = dict(event="complete", engine=engine_name, completed=sum(counts.values()), counts=dict(counts), elapsed_seconds=round(time.perf_counter() - start, 1))
    event(**final)
    return final
