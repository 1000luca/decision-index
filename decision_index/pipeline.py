import json
import os
from datetime import datetime, timezone
from pathlib import Path

from decision_index import constants as C
from decision_index import editions
from decision_index.scoring.index import index_entry, rnd, score_panel
from decision_index.scoring.report import benchmark_summary, load_results
from decision_index.suite.io import Suite, atomic_json


def score_run(suite, results_path, engine, out_dir, reference_results=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = load_results(results_path)
    reference = load_results(reference_results) if reference_results else None
    if editions.v2_family(suite.edition["id"]):
        return score_run_v02(suite, results, engine, out, reference)
    summary = benchmark_summary(suite, results, engine, reference)
    atomic_json(out / "benchmark-summary.json", summary)
    scored = score_panel(suite, results)
    index = index_entry(scored)
    atomic_json(out / "index.json", index)
    benchmarks = {}
    for b in summary["benchmarks"]:
        n = b["catalog_id"]
        entry = {k: (rnd(v) if isinstance(v, float) else v) for k, v in b.items() if k not in ("detail",)}
        entry["median_ms"] = rnd(b.get("median_ms"), 1)
        if n in C.HEADLINE and n in scored:
            h = index["benchmarks"][str(n)]
            entry.update(score=h["raw"], metric=C.HEADLINE_METRIC, tracks=h["tracks"], answered=round(h["coverage"] * b["requests"]) if b.get("requests") else b.get("answered"))
        entry["in_index"] = str(n) in index["benchmarks"] and index["benchmarks"][str(n)]["in_index"]
        benchmarks[str(n)] = entry
    completed = sum(summary["counts"].values())
    scores = {
        "engine": engine,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "suite": {"requests": C.SUITE_REQUESTS, "scoreable": C.SUITE_SCOREABLE_REQUESTS, "excluded": C.SUITE_EXCLUDED_REQUESTS, "benchmarks": C.SUITE_BENCHMARKS, "rows_sha256": C.SUITE_ROWS_GZ_SHA256},
        "completed": completed,
        "complete": completed >= C.SUITE_SCOREABLE_REQUESTS,
        "counts": summary["counts"],
        "latency_ms": {k: rnd(v, 1) for k, v in summary["successful_request_latency_ms"].items()},
        "decision_index": index["index"],
        "scores": index["scores"],
        "areas": index["areas"],
        "index_benchmarks": index["benchmarks"],
        "benchmarks": benchmarks,
        "frozen_panel": index["frozen_panel"],
        "formulas": index["formulas"],
        "panel_id": index["panel_id"],
    }
    atomic_json(out / "scores.json", scores)
    return scores


def score_run_v02(suite, results, engine, out, reference=None):
    import collections

    from decision_index.scoring import added, index02

    spec = index02.spec(suite.edition["id"])
    added_ids = {int(n) for n in spec["added"]}
    base, extra = [], collections.defaultdict(list)
    for r in suite.rows(apply_exclusions=True):
        n = r["_evaluation"]["catalog_id"]
        (extra[n] if n in added_ids else base).append(r)
    metrics = {int(n): (m["name"], m["key"]) for n, m in spec.get("metrics", {}).items()}
    summary = benchmark_summary(suite, results, engine, reference, rows=base, metrics=metrics)
    added_reports = {n: added.report(n, rows, results) for n, rows in sorted(extra.items())}
    for n, rep in added_reports.items():
        entry = {k: rep[k] for k in ("catalog_id", "dataset", "requests", "answered", "unsupported", "errors", "abstained", "pending", "metric", "score", "median_ms")}
        entry.update(scored_requests=rep["answered"], detail={"field_accuracy": rep["field_accuracy"], "scored_fields": rep["scored_fields"], "chance_on_rows": rep["chance"]})
        if reference:
            entry["reference_same_cases"] = added.report(n, [row for row in extra[n] if results.get(row["_evaluation"]["run_id"], {}).get("status") == "ok"], reference)["score"]
        summary["benchmarks"].append(entry)
    summary["edition"] = suite.edition["id"]
    atomic_json(out / "benchmark-summary.json", summary)
    index = index02.index_entry(suite, results, summary, added_reports, spec)
    atomic_json(out / "index.json", index)
    benchmarks = {}
    for b in summary["benchmarks"]:
        n = str(b["catalog_id"])
        entry = {k: (rnd(v) if isinstance(v, float) else v) for k, v in b.items() if k != "detail"}
        entry["median_ms"] = rnd(b.get("median_ms"), 1)
        if n in index["benchmarks"]:
            v = index["benchmarks"][n]
            entry.update(index_raw=v["raw"], index_skill=v["skill"], coverage=v["coverage"], chance=v["random"], in_index=v["in_index"])
            if v.get("tracks"):
                entry["tracks"] = v["tracks"]
            if int(n) in C.HEADLINE:
                entry.update(score=v["raw"], metric=C.HEADLINE_METRIC)
        else:
            entry["in_index"] = False
        benchmarks[n] = entry
    e = suite.edition
    completed = sum(1 for r in suite.rows(apply_exclusions=True) if r["_evaluation"]["run_id"] in results)
    counts = collections.Counter(results[r["_evaluation"]["run_id"]]["status"] for r in suite.rows(apply_exclusions=True) if r["_evaluation"]["run_id"] in results)
    expected = e["scoreable"] + e["added_requests"]
    scores = {
        "engine": engine,
        "edition": e["id"],
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "suite": {"edition": e["name"], "requests": e["requests"], "scoreable": e["scoreable"], "excluded": e["excluded"], "added_requests": e["added_requests"], "benchmarks": e["benchmarks"], "rows_sha256": e["rows_sha256"], "added_sha256": e["added_sha256"], "gsm8k_sha256": e.get("gsm8k_sha256")},
        "completed": completed,
        "complete": completed >= expected,
        "counts": dict(counts),
        "latency_ms": {k: rnd(v, 1) for k, v in summary["successful_request_latency_ms"].items()},
        "decision_index": index["index"],
        "raw_index": index["raw_index"],
        "scores": index["scores"],
        "areas": index["areas"],
        "index_benchmarks": index["benchmarks"],
        "benchmarks": benchmarks,
        "panel_id": index["panel_id"],
        "note": index["note"],
    }
    atomic_json(out / "scores.json", scores)
    return scores


def upload_run(repo_id, run_dir, path_in_repo="", private=True, token=None, compress=True):
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    import gzip
    import shutil

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    run_dir = Path(run_dir)
    staged = run_dir / "_upload"
    staged.mkdir(exist_ok=True)
    for name in ("benchmark-summary.json", "index.json", "scores.json", "vision-scores.json", "environment.json", "status.json"):
        if (run_dir / name).exists():
            shutil.copyfile(run_dir / name, staged / name)
    results = run_dir / "results.jsonl"
    if results.exists():
        if compress:
            with results.open("rb") as src, gzip.open(staged / "results.jsonl.gz", "wb", compresslevel=6) as dst:
                shutil.copyfileobj(src, dst)
        else:
            shutil.copyfile(results, staged / "results.jsonl")
    api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(staged), path_in_repo=path_in_repo or ".", commit_message=f"decision-index results {run_dir.name}")
    shutil.rmtree(staged)
    return f"https://huggingface.co/datasets/{repo_id}"
