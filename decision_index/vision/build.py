import ast
import hashlib
import io
import json
import os
import random
import re
import shutil
import time
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from decision_index.vision import spec
from decision_index.vision.convert import (CORD_PRESENT_Q, CORD_VALUE_FIELDS, CORD_VALUE_Q, LETTERS, NUM, SEED, Letters, arrange,
                                           cord_fields, draw_marks, fmt_dec, fmt_int, funsd_type, gold_ok, mc_item, norm,
                                           numeric_distractors, overlaps, parse_amount, pick_by_rank)

MAX_SIDE = 4096
VH = 1080
EXT = {"JPEG": "jpg", "MPO": "jpg", "PNG": "png", "WEBP": "webp", "GIF": "gif", "BMP": "bmp", "TIFF": "tif"}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


class Mismatch(RuntimeError):
    pass


class Ctx:
    def __init__(self, name, root, downloads, font=None):
        from PIL import Image

        Image.MAX_IMAGE_PIXELS = None
        self.name = name
        self.root = Path(root)
        self.dir = self.root / "full" / name
        self.img = self.dir / "img"
        self.dl = Path(downloads) / name
        self.font = font
        self.pin = spec.pins()["sets"][name]
        if self.img.exists():
            shutil.rmtree(self.img)
        self.img.mkdir(parents=True)
        self.shards = []
        self.saved = {}

    def check(self, file, digest):
        want = {s["file"]: s["sha256"] for s in self.pin["shards"]}.get(file)
        if want != digest:
            raise Mismatch(f"{self.name}: {file} sha256 {digest} != pinned {want}")
        self.shards.append({"file": file, "sha256": digest})

    def fetch(self, file, repo=None, revision=None):
        from huggingface_hub import hf_hub_download

        repo = repo or self.pin["repo"]
        revision = revision or self.pin["revision"]
        p = hf_hub_download(repo, file, repo_type="dataset", revision=revision, local_dir=self.dl)
        self.check(file, sha256_file(p))
        return p

    def save(self, data, stem):
        from PIL import Image

        if stem in self.saved:
            return self.saved[stem]
        im = Image.open(io.BytesIO(data))
        ext = EXT.get(im.format, "png")
        w, h = im.size
        if max(w, h) > MAX_SIDE:
            s = MAX_SIDE / max(w, h)
            im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
            if ext == "jpg":
                path = self.img / f"{stem}.jpg"
                im.convert("RGB").save(path, quality=95)
            else:
                path = self.img / f"{stem}.png"
                if im.mode not in ("RGB", "RGBA", "L", "LA"):
                    im = im.convert("RGBA" if "A" in im.mode or "transparency" in im.info else "RGB")
                im.save(path)
        else:
            path = self.img / f"{stem}.{ext}"
            path.write_bytes(data)
        rel = str(path.relative_to(self.root))
        self.saved[stem] = rel
        return rel

    def write(self, items):
        ids = [it["id"] for it in items]
        assert len(ids) == len(set(ids)), "duplicate ids"
        for it in items:
            assert it["gold"] in LETTERS[:len(it["options"])], it["id"]
        with open(self.dir / "items.jsonl", "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        got = {"n": len(items), "items_sha256": sha256_file(self.dir / "items.jsonl"), "images_sha256": images_digest(self.root, items)}
        meta = {"set": self.name, "source": f"{self.pin['repo']}@{self.pin['revision']}" if self.pin.get("revision") else self.pin["repo"],
                "shards": self.shards, **got, "built": time.strftime("%Y-%m-%d")}
        (self.dir / "source.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n")
        return {k: {"rebuilt": v, "pinned": self.pin[k], "match": v == self.pin[k]} for k, v in got.items()}


def images_digest(root, items):
    files = sorted({p for it in items for p in it["images"] + ([it["clip_image"]] if it.get("clip_image") else [])})
    return hashlib.sha256("".join(f"{f} {sha256_file(Path(root) / f)}\n" for f in files).encode()).hexdigest()


def iter_rows(path, columns=None):
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    for rg in range(pf.num_row_groups):
        for r in pf.read_row_group(rg, columns=columns).to_pylist():
            yield r


def item(bench, sid, images, question, options, gold, subtask, clip_texts=None, extra=None):
    return mc_item(bench, f"{bench}:{sid}", images, question, options, gold, subtask, clip_texts, extra)


def full_cvbench(c):
    keep = {"Count", "Relation", "Distance"}
    items = []
    for fn in ["test_2d.parquet", "test_3d.parquet"]:
        for r in iter_rows(c.fetch(fn)):
            if r["task"] not in keep:
                continue
            gold = r["answer"].strip().strip("()")
            if not gold_ok(gold, r["choices"]):
                continue
            img = c.save(r["image"]["bytes"], str(r["idx"]))
            items.append(item(c.name, r["idx"], [img], r["question"], r["choices"], gold, r["task"], extra={"source": r["source"]}))
    items.sort(key=lambda it: int(it["id"].split(":")[1]))
    return items


def full_blink(c):
    subs = ["Counting", "Relative_Depth", "Spatial_Relation", "Multi-view_Reasoning",
            "Semantic_Correspondence", "Visual_Correspondence", "Functional_Correspondence"]
    items = []
    for s in subs:
        for r in iter_rows(c.fetch(f"{s}/val-00000-of-00001.parquet")):
            gold = r["answer"].strip().strip("()")
            if not gold_ok(gold, r["choices"]):
                continue
            imgs = []
            for j in range(1, 5):
                v = r.get(f"image_{j}")
                if v and v.get("bytes"):
                    imgs.append(c.save(v["bytes"], f"{r['idx']}_{j}"))
            q = re.split(r"\n?Select from the following choices", r["prompt"])[0].strip()
            items.append(item(c.name, r["idx"], imgs, q, r["choices"], gold, s))
    return items


def parse_lettered(text, pat):
    parts = re.split(pat, text)
    if len(parts) < 3:
        return None, None
    stem = parts[0].strip()
    opts = {}
    for i in range(1, len(parts) - 1, 2):
        opts[parts[i]] = parts[i + 1].strip().rstrip(",").strip()
    letters = sorted(opts)
    if letters != list(LETTERS[:len(letters)]):
        return None, None
    return stem, [opts[l] for l in letters]


def full_realworldqa(c):
    items = []
    off = 0
    for k in range(2):
        for r in iter_rows(c.fetch(f"data/test-0000{k}-of-00002.parquet")):
            q, a = r["question"], r["answer"].strip()
            body = re.split(r"\n(Please answer|Answer with)", q)[0]
            stem, opts = parse_lettered(body, r"\n([A-F])\. ")
            if opts and len(opts) >= 2 and a in LETTERS[:len(opts)]:
                rec = (stem, opts, a, "mc")
            elif a.lower() in ("yes", "no"):
                rec = (body.strip(), ["Yes", "No"], "A" if a.lower() == "yes" else "B", "yesno")
            else:
                rec = None
            if rec:
                img = c.save(r["image"]["bytes"], str(off))
                items.append(item(c.name, off, [img], rec[0], rec[1], rec[2], rec[3]))
            off += 1
    return items


def full_rbench_m(c):
    items = []
    for r in iter_rows(c.fetch("rbench-m_en/test-00000-of-00001.parquet")):
        letters = [l for l in "ABCDEF" if r.get(l) is not None and str(r[l]).strip()]
        opts = [r[l] for l in letters]
        gold = r["answer"].strip()
        if letters != list(LETTERS[:len(letters)]) or not gold_ok(gold, opts):
            continue
        other = r[gold].strip().lower().startswith("other options are incorrect")
        img = c.save(r["image"]["bytes"], str(r["index"]))
        items.append(item(c.name, r["index"], [img], r["question"], opts, gold, "other_correct" if other else "content_correct"))
    return items


def full_mmmu_pro_vision(c):
    items = []
    off = 0
    seen = Counter()
    for k in range(4):
        for r in iter_rows(c.fetch(f"vision/test-0000{k}-of-00004.parquet")):
            opts = ast.literal_eval(r["options"]) if isinstance(r["options"], str) else r["options"]
            gold = r["answer"].strip()
            seen[r["id"]] += 1
            sid = r["id"] if seen[r["id"]] == 1 else f"{r['id']}#{seen[r['id']]}"
            if not gold_ok(gold, opts):
                off += 1
                continue
            img = c.save(r["image"]["bytes"], str(off))
            items.append(item(c.name, sid, [img], "", opts, gold, r["subject"], extra={"vision_only": True}))
            off += 1
    return items


def full_charxiv(c):
    import pyarrow.parquet as pq

    p = c.fetch("val.parquet")
    cols = [n for n in pq.ParquetFile(p).schema_arrow.names if n != "image"]
    meta = pq.read_table(p, columns=cols).to_pylist()
    cand = []
    for off, r in enumerate(meta):
        t = str(r["reasoning_a_type"])
        gold = str(r["reasoning_a"]).strip()
        pool = []
        for j in range(1, 5):
            a = str(r.get(f"descriptive_a{j}", ""))
            pool += [x.strip() for x in re.split(r",|;", a) if x.strip() and x.strip() not in ("Not Applicable",)]
        if t == "1":
            ds = [q for q in dict.fromkeys(pool) if q.lower() != gold.lower() and not NUM.match(q) and q.lower() not in ("yes", "no")]
            if len(ds) >= 3:
                cand.append((off, r, gold, ds, t))
        elif t in ("3", "4") and NUM.match(gold):
            cand.append((off, r, gold, pool, t))
    rng = random.Random(SEED + 4)
    imgs = pq.read_table(p, columns=["image"])
    items = []
    for off, r, gold, pool, t in cand:
        ds = rng.sample(pool, 3) if t == "1" else numeric_distractors(gold, pool, rng)
        opts = ds + [gold]
        rng.shuffle(opts)
        if len(opts) != 4 or len(set(opts)) != 4:
            continue
        img = c.save(imgs.slice(off, 1).to_pylist()[0]["image"]["bytes"], str(off))
        items.append(item(c.name, off, [img], r["reasoning_q"], opts, LETTERS[opts.index(gold)],
                          {"1": "text-in-chart", "3": "number-in-chart", "4": "number-in-general"}[t]))
    return items


def full_infovqa(c):
    import pyarrow.parquet as pq

    files = [c.fetch(f"InfographicVQA/validation-0000{k}-of-00004.parquet") for k in range(4)]
    meta = []
    for fi, f in enumerate(files):
        for ri, r in enumerate(pq.read_table(f, columns=["questionId", "question", "answers", "image_url"]).to_pylist()):
            meta.append(dict(fi=fi, ri=ri, **r))
    byimg = {}
    for m in meta:
        byimg.setdefault(m["image_url"], []).append(m)

    def isnum(a):
        return bool(re.search(r"\d", a))

    cands = []
    for m in meta:
        gold = m["answers"][0].strip().lower()
        golds = {norm(a) for a in m["answers"]}
        pool = []
        for o in byimg[m["image_url"]]:
            if o["questionId"] == m["questionId"]:
                continue
            a = o["answers"][0].strip().lower()
            if isnum(a) != isnum(gold) or norm(a) in golds or not norm(a):
                continue
            if any(norm(a) in g or g in norm(a) for g in golds if g):
                continue
            pool.append(a)
        pool = list(dict.fromkeys(pool))
        if len(pool) >= 3:
            cands.append((m, gold, pool))
    rng = random.Random(SEED + 12)
    rng.shuffle(cands)
    seen, pick = set(), []
    for cd in cands:
        if cd[0]["image_url"] in seen:
            continue
        seen.add(cd[0]["image_url"])
        pick.append(cd)
    tables = {}
    items = []
    for m, gold, pool in sorted(pick, key=lambda cd: (cd[0]["fi"], cd[0]["ri"])):
        if m["fi"] not in tables:
            tables[m["fi"]] = pq.read_table(files[m["fi"]], columns=["image"])
        img = c.save(tables[m["fi"]].slice(m["ri"], 1).to_pylist()[0]["image"]["bytes"], str(m["questionId"]))
        ds = rng.sample(pool, 3)
        opts = ds + [gold]
        rng.shuffle(opts)
        items.append(item(c.name, m["questionId"], [img], m["question"], opts, LETTERS[opts.index(gold)], "numeric" if isnum(gold) else "text"))
    return items


def full_winoground(c):
    ex = [json.loads(l) for l in open(c.fetch("data/examples.jsonl"))]
    zf = zipfile.ZipFile(c.fetch("data/images.zip"))
    rng = random.Random(SEED + 13)

    def img(name):
        return c.save(zf.read(f"images/{name}.png"), name)

    items = []
    for e in sorted(ex, key=lambda e: e["id"]):
        caps = [e["caption_0"], e["caption_1"]]
        ims = [img(e["image_0"]), img(e["image_1"])]
        for k in (0, 1):
            order = [0, 1]
            rng.shuffle(order)
            items.append(item(c.name, f"{e['id']}:text:{k}", [ims[k]], "Which caption describes this image?", [caps[j] for j in order],
                              LETTERS[order.index(k)], "text", extra={"group": e["id"], "tag": e["collapsed_tag"]}))
        for k in (0, 1):
            order = [0, 1]
            rng.shuffle(order)
            items.append(item(c.name, f"{e['id']}:image:{k}", [ims[j] for j in order], f"Which image matches this caption: \"{caps[k]}\"?",
                              ["the first image", "the second image"], LETTERS[order.index(k)], "image", clip_texts=[caps[k]],
                              extra={"group": e["id"], "tag": e["collapsed_tag"], "clip_image_choice": True}))
    return items


def full_cord(c):
    import pyarrow.parquet as pq

    shards = {"validation": "data/validation-00000-of-00001-cc3c5779fe22e8ca.parquet", "test": "data/test-00000-of-00001-9c204eb3f4e11791.parquet"}
    paths = {s: c.fetch(f) for s, f in shards.items()}
    recs = []
    for split, p in paths.items():
        for i, gt in enumerate(pq.read_table(p, columns=["ground_truth"]).column("ground_truth").to_pylist()):
            recs.append((f"{split}-{i}", split, i, json.loads(gt)["gt_parse"]))
    rng = random.Random(SEED + 101)
    rng.shuffle(recs)
    plan = []
    for key, split, i, gp in recs:
        f, amounts, menu = cord_fields(gp)
        vals = {}
        for a, k in amounts:
            vals.setdefault(a, set()).add(k)
        if len(vals) < 3:
            continue
        for fld in CORD_VALUE_FIELDS:
            if fld == "item_price":
                names = Counter(n.lower() for n, _ in menu)
                for j, (n, pr) in enumerate(menu):
                    if names[n.lower()] == 1 and 2 <= len(n) <= 40 and re.search(r"[A-Za-z]{2}", n):
                        plan.append((key, fld, j, n, pr, vals))
            else:
                g = parse_amount(f.get(fld))
                if g:
                    plan.append((key, fld, None, None, g, vals))
    n_elig = Counter(pl[1] for pl in plan)
    cap = {fld: -(-n_elig[fld] // 2) for fld in CORD_VALUE_FIELDS}
    letters = Letters(rng)
    rank_counts = {fld: Counter() for fld in CORD_VALUE_FIELDS}
    tables = {}

    def image(key, split, i):
        if split not in tables:
            tables[split] = pq.read_table(paths[split], columns=["image"])
        return c.save(tables[split].slice(i, 1).to_pylist()[0]["image"]["bytes"], key)

    where = {key: (split, i) for key, split, i, _ in recs}
    items = []
    for key, fld, j, nm, gold_v, vals in plan:
        sid = f"{key}:value:{fld}" + (f":{j}" if j is not None else "")
        q = CORD_VALUE_Q[fld].format(nm=nm) if fld == "item_price" else CORD_VALUE_Q[fld]
        pool = [(v, None) for v in vals if abs(v - gold_v) > 1e-9]
        dec = any(isinstance(v, float) and v != int(v) for v in list(vals) + [gold_v])
        fm = fmt_dec if dec else fmt_int
        ch = pick_by_rank(gold_v, pool, 3, rng, rank_counts[fld], cap[fld])
        if ch is None:
            continue
        pos = letters.pick(len(ch) + 1)
        dv = [x[0] for x in ch]
        rng.shuffle(dv)
        ov = dv[:pos] + [gold_v] + dv[pos:]
        opts = [fm(v) for v in ov]
        if len(set(opts)) != len(opts):
            continue
        items.append(item(c.name, sid, [image(key, *where[key])], q, opts, LETTERS[pos], "field_value", extra={"field": fld, "option_values": ov}))
    for key, split, i, gp in sorted(recs, key=lambda r: r[0]):
        f = cord_fields(gp)[0]
        for fld in CORD_PRESENT_Q:
            has = fld in f and parse_amount(f[fld] if not isinstance(f[fld], list) else f[fld][0]) is not None
            absent = fld not in f
            if not has and not absent:
                continue
            q = f"Does this receipt show {CORD_PRESENT_Q[fld]}?"
            items.append(item(c.name, f"{key}:present:{fld}", [image(key, split, i)], q, ["Yes", "No"], "A" if has else "B", "field_present",
                              clip_texts=[f"a receipt that shows {CORD_PRESENT_Q[fld]}", f"a receipt that does not show {CORD_PRESENT_Q[fld]}"],
                              extra={"field": fld}))
    items.sort(key=lambda x: x["id"])
    return items


def full_funsd(c):
    import httpx

    c.dl.mkdir(parents=True, exist_ok=True)
    zp = c.dl / "dataset.zip"
    if not zp.exists() or sha256_file(zp) != spec.FUNSD_SHA256:
        zp.write_bytes(httpx.get(spec.FUNSD_URL, timeout=300, follow_redirects=True).content)
    c.check("dataset.zip", sha256_file(zp))
    zf = zipfile.ZipFile(zp)
    base = "dataset/testing_data"
    forms = sorted(Path(n).stem for n in zf.namelist() if n.startswith(f"{base}/annotations/") and n.endswith(".json"))
    value_c, present_c = [], []
    for fid in forms:
        ents = json.loads(zf.read(f"{base}/annotations/{fid}.json"))["form"]
        byid = {e["id"]: e for e in ents}
        answers = [e for e in ents if e["label"] == "answer" and 1 <= len(e["text"].strip()) <= 60]
        qtexts = Counter(e["text"].strip().lower() for e in ents if e["label"] == "question")
        for e in ents:
            if e["label"] != "question":
                continue
            qt = e["text"].strip()
            if qtexts[qt.lower()] != 1:
                continue
            if not re.search(r"[A-Za-z]{2}", qt) or len(qt) > 50:
                continue
            linked = [byid[b] for a, b in e["linking"] if a == e["id"] and b in byid and byid[b]["label"] == "answer"]
            linked = [x for x in linked if x["text"].strip()]
            name = qt.strip(" :\"'").strip()
            if len(linked) == 1 and 1 <= len(linked[0]["text"].strip()) <= 60:
                gold = linked[0]["text"].strip()
                gt = funsd_type(gold)
                others = sorted({a["text"].strip() for a in answers if a["id"] != linked[0]["id"] and a["text"].strip().lower() != gold.lower()})
                same = [o for o in others if funsd_type(o) == gt]
                if len(same) >= 3:
                    value_c.append((fid, e, name, gold, same, gt))
                present_c.append(("yes", fid, name, e))
            elif not linked and not any(byid.get(b, {}).get("label") == "answer" for _, b in e["linking"]):
                x0, y0, x1, y1 = e["box"]
                near = [a for a in answers if (abs((a["box"][1] + a["box"][3]) / 2 - (y0 + y1) / 2) < 12 and 0 <= a["box"][0] - x1 < 250)
                        or (0 <= a["box"][1] - y1 < 25 and abs(a["box"][0] - x0) < 60)]
                if not near:
                    present_c.append(("no", fid, name, e))
    rng = random.Random(SEED + 103)
    letters = Letters(rng)

    def image(fid):
        return c.save(zf.read(f"{base}/images/{fid}.png"), fid)

    items = []
    for fid, e, name, gold, same, gt in value_c:
        ds = rng.sample(same, 3)
        pos = letters.pick(4)
        opts, gl = arrange(gold, ds, pos, rng)
        items.append(item(c.name, f"{fid}:value:{e['id']}", [image(fid)], f"On this form, what is filled in for the field \"{name}\"?",
                          opts, gl, "field_value", extra={"field": name, "answer_type": gt}))
    for lab, fid, name, e in present_c:
        items.append(item(c.name, f"{fid}:present:{e['id']}", [image(fid)], f"On this form, is the field \"{name}\" filled in?", ["Yes", "No"],
                          "A" if lab == "yes" else "B", "field_present",
                          clip_texts=[f"a form with the field {name} filled in", f"a form with the field {name} left blank"], extra={"field": name}))
    items.sort(key=lambda x: x["id"])
    return items


def full_hateful_memes(c):
    from huggingface_hub import HfApi, hf_hub_download

    repo, rev = c.pin["repo"], c.pin["revision"]
    have = {f for f in HfApi().list_repo_files(repo, repo_type="dataset", revision=rev) if f.startswith("img/")}
    rows = []
    for split in ["dev_seen", "test_seen"]:
        for r in (json.loads(l) for l in open(c.fetch(f"{split}.jsonl"))):
            if r["img"] in have:
                rows.append((split, r))
    with ThreadPoolExecutor(8) as ex:
        local = dict(zip([r["img"] for _, r in rows],
                         ex.map(lambda f: hf_hub_download(repo, f, repo_type="dataset", revision=rev, local_dir=c.dl), [r["img"] for _, r in rows])))
    digest = hashlib.sha256("".join(f"{f} {sha256_file(local[f])}\n" for f in sorted(local)).encode()).hexdigest()
    c.check(f"img/*.png ({len(local)} files)", digest)
    items = []
    for split, r in rows:
        img = c.save(Path(local[r["img"]]).read_bytes(), r["id"])
        items.append(item(c.name, r["id"], [img], "Is this meme hateful?", ["Yes", "No"], "A" if r["label"] == 1 else "B", "binary",
                          clip_texts=["a hateful meme", "a harmless meme"], extra={"meme_text": r["text"], "source_split": split}))
    return items


def m2w_box(cand):
    a = json.loads(json.loads(cand)["attributes"])
    r = a.get("bounding_box_rect")
    if not r:
        return None
    x, y, w, h = [float(v) for v in r.split(",")]
    return [x, y, x + w, y + h]


def m2w_convert(c, row, sid, rng):
    from PIL import Image

    if not row["pos_candidates"]:
        return None
    g = m2w_box(row["pos_candidates"][0])
    if not g or g[2] - g[0] < 4 or g[3] - g[1] < 4:
        return None
    im = Image.open(io.BytesIO(row["screenshot"]["bytes"])).convert("RGB")
    W, Hh = im.size
    vh = min(Hh, VH)
    cy = (g[1] + g[3]) / 2
    top = max(0, int(min(max(0, cy - vh / 2 + rng.uniform(-200, 200)), Hh - vh)))
    if not (g[1] >= top and g[3] <= top + vh and g[2] <= W):
        return None
    negs = []
    for cand in row["neg_candidates"]:
        b = m2w_box(cand)
        if not b:
            continue
        w, h = b[2] - b[0], b[3] - b[1]
        if w < 8 or h < 8 or w > 0.6 * W or h > 0.4 * vh:
            continue
        if b[1] < top or b[3] > top + vh or b[2] > W:
            continue
        if overlaps(b, g) or any(overlaps(b, x) for x in negs):
            continue
        negs.append(b)
    if len(negs) < 3:
        return None
    boxes = rng.sample(negs, 3) + [g]
    rng.shuffle(boxes)
    crop = im.crop((0, top, W, top + vh))
    px = [[b[0], b[1] - top, b[2], b[3] - top] for b in boxes]
    stem = sid.replace("/", "_")
    base = c.img / f"{stem}.jpg"
    marked = c.img / f"{stem}_marked.jpg"
    crop.save(base, quality=92)
    draw_marks(crop, px, c.font).save(marked, quality=92)
    op = json.loads(row["operation"])
    prev = row["action_reprs"][:int(row["target_action_index"])]
    hist = "\n".join(f"- {a}" for a in prev[-5:]) or "- (none)"
    q = (f"Task: {row['confirmed_task']}\nPrevious actions:\n{hist}\n"
         f"Which marked element should the next action ({op['op']}) be performed on?")
    return mc_item("full_mind2web", f"full_mind2web:{sid}", [str(marked.relative_to(c.root))], q,
                   [f"box {LETTERS[i]}" for i in range(4)], LETTERS[boxes.index(g)], row["domain"],
                   clip_texts=[row["confirmed_task"]] * 4,
                   extra={"clip_image": str(base.relative_to(c.root)), "clip_boxes": px, "op": op["op"],
                          "annotation_id": row["annotation_id"], "src_row": row["_off"], "viewport_top": top})


def full_mind2web(c):
    import pyarrow.parquet as pq

    cols = ["action_uid", "operation", "pos_candidates", "neg_candidates", "domain", "annotation_id",
            "confirmed_task", "screenshot", "action_reprs", "target_action_index"]
    items, seen = [], set()
    off = 0
    for shard in c.pin["shards"]:
        p = c.fetch(shard["file"])
        pf = pq.ParquetFile(p)
        for rg in range(pf.num_row_groups):
            for row in pf.read_row_group(rg, columns=cols).to_pylist():
                row["_off"] = off
                off += 1
                sid = row["action_uid"]
                assert sid not in seen
                seen.add(sid)
                it = m2w_convert(c, row, sid, random.Random(f"{SEED}:full_mind2web:{sid}"))
                if it:
                    items.append(it)
    return items


BUILDERS = {f.__name__: f for f in [full_cvbench, full_blink, full_realworldqa, full_charxiv, full_infovqa, full_mind2web, full_winoground,
                                    full_cord, full_funsd, full_hateful_memes, full_rbench_m, full_mmmu_pro_vision]}


def build(root, sets=None, downloads=None, font=None, keep_downloads=False, log=print):
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    root = Path(root)
    downloads = Path(downloads or root.parent / (root.name + "-downloads"))
    report = {}
    for name in sets or BUILDERS:
        t = time.time()
        c = Ctx(name, root, downloads, font)
        try:
            report[name] = c.write(BUILDERS[name](c))
        except Mismatch as e:
            report[name] = {"error": str(e)}
        if not keep_downloads:
            shutil.rmtree(c.dl, ignore_errors=True)
        log(json.dumps({"set": name, "seconds": round(time.time() - t, 1), **{k: v.get("match") if isinstance(v, dict) else v for k, v in report[name].items()}}), flush=True)
    return report
