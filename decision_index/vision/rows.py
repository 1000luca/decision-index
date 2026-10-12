import hashlib
import json
from pathlib import Path

from decision_index.vision import spec

ROWS_FILE = "vision-rows.jsonl"
OPTION_LETTERS = "ABCDEFGHIJKL"


def dumps(x):
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"))


def balance(rows, key):
    keep = [r for r in rows if r["subtask"] != "field_present"]
    groups = {}
    for r in rows:
        if r["subtask"] == "field_present":
            groups.setdefault(key(r), {}).setdefault(r["gold"], []).append(r)
    for g in groups.values():
        if len(g) < 2:
            continue
        m = min(len(v) for v in g.values())
        for v in g.values():
            keep += sorted(v, key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())[:m]
    return keep


BALANCE_KEY = {"full_cord": lambda r: r["id"].split(":")[-1], "full_funsd": lambda r: "all"}


def select(set_name, rows):
    b = spec.benchmark_of(set_name)
    if b.get("subtask"):
        rows = [r for r in rows if r["subtask"] == b["subtask"]]
    if b.get("balance_present"):
        rows = balance(rows, BALANCE_KEY[set_name])
    return rows


def load_items(root, set_name):
    with open(Path(root) / "full" / set_name / "items.jsonl", encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def request(it):
    k = len(it["options"])
    if it.get("vision_only"):
        instr = f"Answer the multiple-choice question shown in the image (options {OPTION_LETTERS[0]}-{OPTION_LETTERS[k - 1]})."
        crit = {OPTION_LETTERS[i]: OPTION_LETTERS[i] for i in range(k)}
    else:
        instr = it["question"]
        for j in range(1, 8):
            instr = instr.replace(f"<image {j}>", f"[image {j}]")
        crit = {OPTION_LETTERS[i]: str(o) for i, o in enumerate(it["options"])}
    return {"answer": {"type": "choice", "instructions": instr, "criteria": crit}}


def file_sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def to_row(root, it):
    b = spec.benchmark_of(it["bench"])
    questions = request(it)
    prompt = {"options_in_image": bool(it.get("vision_only")), "readout": spec.READOUT if b.get("readout") else None}
    image_sha = [file_sha256(Path(root) / p) for p in it["images"]]
    payload = {"state": {}, "questions": questions, "images": image_sha, "prompt": prompt}
    return {"id": it["id"], "benchmark": b["name"], "set": it["bench"], "subtask": it["subtask"], "images": it["images"],
            "state": {}, "questions": questions, "prompt": prompt, "expected": {"answer": it["gold"]},
            "_evaluation": {"run_id": it["id"], "benchmark": b["name"], "set": it["bench"],
                            "payload_sha256": hashlib.sha256(dumps(payload).encode()).hexdigest(), "image_sha256": image_sha}}


def write_rows(root, out=None):
    root = Path(root)
    out = Path(out or root / ROWS_FILE)
    rows = []
    for s in spec.SETS:
        rows += [to_row(root, it) for it in select(s, load_items(root, s))]
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(dumps(r) + "\n")
    return out, rows


def summarize(rows):
    by = {}
    for r in rows:
        by.setdefault(r["benchmark"], []).append(r["_evaluation"])
    return {name: {"rows": len(es), "run_ids_sha256": hashlib.sha256("".join(e["run_id"] + "\n" for e in sorted(es, key=lambda e: e["run_id"])).encode()).hexdigest(),
                   "payloads_sha256": hashlib.sha256("".join(f"{e['run_id']} {e['payload_sha256']}\n" for e in sorted(es, key=lambda e: e["run_id"])).encode()).hexdigest()}
            for name, es in by.items()}


def read_rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def verify(root):
    root = Path(root)
    pins = spec.pins()
    out = {"sets": {}, "benchmarks": {}}
    from decision_index.vision.build import images_digest

    for s in spec.SETS:
        p = root / "full" / s / "items.jsonl"
        if not p.exists():
            out["sets"][s] = {"missing": True}
            continue
        items = load_items(root, s)
        got = {"n": len(items), "items_sha256": file_sha256(p), "images_sha256": images_digest(root, items)}
        out["sets"][s] = {k: v == pins["sets"][s][k] for k, v in got.items()}
    rp = root / ROWS_FILE
    if rp.exists():
        got = summarize(read_rows(rp))
        for name, want in pins["benchmarks"].items():
            g = got.get(name, {})
            out["benchmarks"][name] = {k: g.get(k) == v for k, v in want.items()}
    out["ok"] = all(all(v.values()) for part in ("sets", "benchmarks") for v in out[part].values()) and bool(out["benchmarks"])
    return out
