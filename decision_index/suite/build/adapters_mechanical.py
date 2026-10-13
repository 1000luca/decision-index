import json
import re

from decision_index.suite.build.layout import sha256, write_jsonl


def acos_sources(layout):
    return sorted((layout.repos / "acos/data").glob("*/*_quad_*.tsv"))


def parse_acos_line(line):
    fields = line.rstrip("\n").split("\t")
    review = fields[0]
    pairs = set()
    for quad in fields[1:]:
        bits = quad.split()
        if len(bits) != 4:
            raise ValueError(f"malformed ACOS quadruple: {quad!r}")
        pairs.add((bits[1], bits[2]))
    return review, pairs


def build_acos(layout):
    sources = acos_sources(layout)
    parsed = {}
    counts = {}
    for path in sources:
        rows = [parse_acos_line(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        parsed[path] = rows
        counts[layout.rel(path)] = len(rows)
    by_domain = {}
    for path in sources:
        domain = path.parent.name
        by_domain.setdefault(domain, set()).update(pair[0] for _, pairs in parsed[path] for pair in pairs)
    by_domain = {domain: {(category, sentiment) for category in categories for sentiment in ("0", "1", "2")} for domain, categories in by_domain.items()}
    rows = []
    for path in sources:
        if not path.name.endswith("_test.tsv"):
            continue
        domain = path.parent.name
        inventories = [sorted(items)[start:start + 64] for start in range(0, len(by_domain[domain]), 64) for items in [by_domain[domain]]]
        for line_no, (review, gold_pairs) in enumerate(parsed[path], 1):
            for group_index, domain_inventory in enumerate(inventories, 1):
                questions = {}
                expected = {}
                for category, sentiment in domain_inventory:
                    sentiment_name = {"0": "negative", "1": "neutral", "2": "positive"}[sentiment]
                    key = f"{category}__sentiment_{sentiment_name}"
                    questions[key] = {"type": "choice", "instructions": ("Given the review below, decide whether it expresses at least one " f"aspect-category/sentiment pair ({category}, {sentiment_name}) anywhere " "in the review. Answer yes only when that exact category and " "sentiment pair is present; answer no otherwise. Multiple opposite " "sentiments are separate valid questions."), "criteria": {"yes": "The exact category/sentiment pair is present.", "no": "The exact category/sentiment pair is absent."}}
                    expected[key] = "yes" if (category, sentiment) in gold_pairs else "no"
                source_id = f"{domain}:{path.stem}:line-{line_no}:group-{group_index}"
                rows.append({"id": f"ACOS:category-sentiment:{source_id}", "family": "ACOS-category-sentiment", "split": "test", "state": {"review": review, "task": "ACOS fixed category/sentiment presence"}, "questions": questions, "expected": expected, "metadata": {"provenance": {"source": layout.rel(path), "sha256": sha256(path), "revision": "45d179a3dcc6a3dedd848d81b16f2552454805fe", "source_line": line_no}, "adaptation_scope": "Named category/sentiment presence projection; full ACOS tuple extraction is not claimed.", "category_inventory": domain_inventory, "inventory_group": group_index, "inventory_group_count": len(inventories), "group_id": f"{domain}:{path.stem}:line-{line_no}", "asserted_label_mapping": {"0": "negative", "1": "neutral", "2": "positive"}, "counts": {"source_reviews": len(parsed[path]), "questions_per_review": len(domain_inventory)}}})
    n = write_jsonl(layout.normalized / "ACOS-category-sentiment.jsonl", rows)
    return n, {"source_counts": counts, "inventory_by_domain": {k: sorted(v) for k, v in by_domain.items()}, "rows": n}


def build_finentity(layout):
    source = layout.repos / "finentity/data/FinEntity.json"
    docs = json.loads(source.read_text(encoding="utf-8"))
    conflicts = []
    for i, doc in enumerate(docs):
        spans = {}
        for ann in doc["annotations"]:
            key = (ann["start"], ann["end"], ann["value"])
            spans.setdefault(key, set()).add(ann["tag"])
        for key, labels in spans.items():
            if len(labels) > 1:
                conflicts.append({"document_index": i, "span": list(key), "labels": sorted(labels)})
    labels = ["Negative", "Neutral", "Positive"]
    rows = []
    for i, doc in enumerate(docs):
        questions = {}
        expected = {}
        for j, ann in enumerate(doc["annotations"]):
            if any(c["document_index"] == i and c["span"] == [ann["start"], ann["end"], ann["value"]] for c in conflicts):
                continue
            key = f"entity_{j:04d}_{ann['start']}_{ann['end']}"
            questions[key] = {"type": "choice", "instructions": ("Classify the sentiment toward the supplied financial entity span. " f"The entity span is given explicitly as {ann['start']}:{ann['end']} ({ann['value']}); " "do not extract or alter it. Choose exactly one of Negative, Neutral, or Positive."), "criteria": {label: label for label in labels}}
            expected[key] = ann["tag"]
        rows.append({"id": f"FinEntity:entity-given:{i:04d}", "family": "FinEntity-entity-given", "split": "evaluation-only", "state": {"document": doc["content"], "task": "FinEntity entity-given sentiment"}, "questions": questions, "expected": expected, "metadata": {"provenance": {"source": layout.rel(source), "sha256": sha256(source), "revision": "3b6cedc5485b669c2ed168f1d949f517636eb7b8", "document_index": i}, "adaptation_scope": "Entity-given sentiment classification; entity extraction is not claimed.", "asserted_label_mapping": {"Negative": "negative", "Neutral": "neutral", "Positive": "positive"}, "counts": {"source_documents": len(docs), "entities_in_document": len(doc["annotations"]), "entities_emitted": len(questions)}, "exclusions": [c for c in conflicts if c["document_index"] == i]}})
    n = write_jsonl(layout.normalized / "FinEntity-entity-given.jsonl", rows)
    return n, {"rows": n, "documents": len(docs), "entities": sum(len(d["annotations"]) for d in docs), "excluded_conflicting_spans": conflicts}


def build_sgd(layout):
    base = layout.repos / "sgd"
    schema = json.loads((base / "test/schema.json").read_text(encoding="utf-8"))
    schemas = {s["service_name"]: s for s in schema}
    files = sorted((base / "test").glob("dialogues_*.json"))
    rows = []
    for path in files:
        source_hash = sha256(path)
        for dialogue in json.loads(path.read_text(encoding="utf-8")):
            history = []
            for ti, turn in enumerate(dialogue["turns"]):
                history.append({"speaker": turn["speaker"], "utterance": turn["utterance"]})
                if turn["speaker"] != "USER":
                    continue
                for fi, frame in enumerate(turn.get("frames", [])):
                    service = frame.get("service")
                    if service not in schemas:
                        continue
                    sch = schemas[service]
                    intents = [x["name"] for x in sch.get("intents", [])] + ["NONE"]
                    gold = frame.get("state", {}).get("active_intent", "NONE")
                    if gold not in intents:
                        continue
                    prompt = "Using the dialogue history and service schema in state, choose the active intent " "for this service. Choose NONE when no service intent is active. Do not use future " "turns or hidden labels."
                    q = {"intent": {"type": "choice", "instructions": prompt, "criteria": {x: x for x in intents}}}
                    rows.append({"id": f"SGD-service-given-intent:test:{dialogue['dialogue_id']}:{ti}:{fi}", "family": "SGD-service-given-intent", "split": "test", "state": {"history": list(history), "service": service, "schema": sch, "task": "SGD current-service intent"}, "questions": q, "expected": {"intent": gold}, "metadata": {"provenance": {"source": layout.rel(path), "sha256": source_hash, "revision": "source snapshot 2026-09-18", "dialogue_id": dialogue["dialogue_id"], "turn_index": ti, "service": service}, "adaptation_scope": "Current-service intent classification; no system future turn or gold span is supplied.", "counts": {"service_intent_candidates": len(intents)}}})
    n = write_jsonl(layout.normalized / "SGD-service-given-intent.jsonl", rows)
    return n, {"rows": n, "source_dialogues": sum(len(json.loads(p.read_text())) for p in files), "files": len(files)}


def _tool_names(value):
    out = set()
    if isinstance(value, list):
        for call in value:
            if isinstance(call, dict):
                out.update(call.keys())
    elif isinstance(value, dict):
        out.update(value.keys())
    return {x for x in out if isinstance(x, str)}


def build_bfcl(layout):
    base = layout.repos / "bfcl/berkeley-function-call-leaderboard"
    categories = ["simple", "live_simple", "live_multiple"]
    rows = []
    exclusions = []
    for category in categories:
        data = base / "data" / f"BFCL_v3_{category}.json"
        goldp = base / "data/possible_answer" / f"BFCL_v3_{category}.json"
        if not data.exists() or not goldp.exists():
            continue
        gs = {x["id"]: x["ground_truth"] for x in (json.loads(l) for l in goldp.open())}
        data_hash = sha256(data)
        gold_hash = sha256(goldp)
        for raw in data.open():
            item = json.loads(raw)
            truth = gs.get(item["id"])
            if not truth or not isinstance(item.get("function"), list):
                exclusions.append({"id": item.get("id"), "reason": "missing function candidates or gold"})
                continue
            needed = _tool_names(truth)
            if not needed:
                exclusions.append({"id": item.get("id"), "reason": "empty gold tool set"})
                continue
            candidates = [f["name"] for f in item["function"] if isinstance(f, dict) and f.get("name")]
            if not candidates or not needed.issubset(set(candidates)):
                exclusions.append({"id": item.get("id"), "reason": "gold tool absent from published candidates", "gold_tools": sorted(needed)})
                continue
            instructions = "Given the complete user conversation and the published function schemas in state, mark each " "candidate tool yes if it should be invoked to answer the request, or no otherwise. " "This evaluates tool-name selection only; do not produce arguments or call ordering.\n\n"
            qs = {}
            ex = {}
            for name in candidates:
                key = "tool_" + re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")
                qs[key] = {"type": "choice", "instructions": instructions + f"\n\nCandidate tool: {name}", "criteria": {"yes": "Invoke this tool.", "no": "Do not invoke this tool."}}
                ex[key] = "yes" if name in needed else "no"
            rows.append({"id": f"BFCL-tool-selection:{category}:{item['id']}", "family": "BFCL-tool-selection", "split": "test", "state": {"conversation": item["question"], "functions": item["function"], "task": "BFCL tool-name selection"}, "questions": qs, "expected": ex, "metadata": {"provenance": {"source": layout.rel(data), "sha256": data_hash, "gold_source": layout.rel(goldp), "gold_sha256": gold_hash, "revision": "source snapshot 2026-09-18", "source_id": item["id"]}, "adaptation_scope": "Tool-name selection projection; arguments, ordering, and execution are not claimed.", "category": category, "candidate_tools": candidates, "selected_gold_tools": sorted(needed)}})
    n = write_jsonl(layout.normalized / "BFCL-tool-selection.jsonl", rows)
    return n, {"rows": n, "categories": categories, "exclusions": exclusions}


def acos(layout):
    return build_acos(layout)[1]


def finentity(layout):
    return build_finentity(layout)[1]


def sgd(layout):
    return build_sgd(layout)[1]


def bfcl(layout):
    return build_bfcl(layout)[1]


BUILDERS = {38: acos, 39: finentity, 10: sgd, 1: bfcl}
