import json
from pathlib import Path

from decision_index.vision import spec
from decision_index.vision.rows import read_rows


def skill(correct, chance):
    import numpy as np

    c, ch = np.asarray(correct, float), np.asarray(chance, float)
    m = ch.mean()
    return float((c.mean() - m) / (1 - m))


def combine(skills):
    import numpy as np

    names = [b["name"] for b in spec.BENCHMARKS if b["name"] in skills]
    W = np.array([spec.benchmark(n)["weight"] for n in names])
    P = np.array([min(1.0, max(0.0, skills[n])) for n in names])
    return 100 * float((W * P).sum() / W.sum())


def predictions(path):
    path = Path(path)
    preds = {}
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                rid = r.get("run_id") or r["id"]
                if "pred" in r:
                    preds[rid] = r["pred"]
                elif r.get("status") == "ok":
                    preds[rid] = r["response"]["answers"]["answer"]["choice"]
                else:
                    preds.setdefault(rid, None)
    return preds


def score(rows, preds):
    if isinstance(rows, (str, Path)):
        rows = read_rows(rows)
    by = {}
    for r in rows:
        by.setdefault(r["benchmark"], []).append(r)
    benchmarks, missing = {}, 0
    for b in spec.BENCHMARKS:
        rs = by.get(b["name"], [])
        if not rs:
            continue
        correct = [float(preds.get(r["id"]) == r["expected"]["answer"]) for r in rs]
        chance = [1.0 / len(r["questions"]["answer"]["criteria"]) for r in rs]
        missing += sum(r["id"] not in preds for r in rs)
        s = skill(correct, chance)
        benchmarks[b["name"]] = {"rows": len(rs), "answered": sum(preds.get(r["id"]) is not None for r in rs), "accuracy": sum(correct) / len(rs),
                                 "chance": sum(chance) / len(rs), "skill": s, "skill_clipped": min(1.0, max(0.0, s)), "weight": b["weight"]}
    index = combine({k: v["skill"] for k, v in benchmarks.items()}) if benchmarks else None
    complete = len(benchmarks) == len(spec.BENCHMARKS) and missing == 0
    return {"edition": spec.EDITION, "public_vision_index": index, "complete": complete,
            "missing_rows": missing, "benchmarks": benchmarks,
            "full_index": "0.5 x public + 0.5 x private; the private part is run by the maintainers and reported as an aggregate"}


def score_run(rows_path, results_path, out=None):
    result = score(rows_path, predictions(results_path))
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / "vision-scores.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
