import os
import re
from collections import Counter

LETTERS = "ABCDEFGHIJ"
SEED = 20261003
NUM = re.compile(r"^-?\d+(\.\d+)?%?$")
FONTS = ["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/System/Library/Fonts/Helvetica.ttc"]


def mc_item(bench, iid, images, question, options, gold, subtask, clip_texts=None, extra=None):
    it = {"id": iid, "bench": bench, "subtask": subtask, "images": images, "question": question,
          "options": options, "gold": gold, "clip_texts": clip_texts or options}
    if extra:
        it.update(extra)
    return it


def gold_ok(gold, options):
    return gold in LETTERS[:len(options)]


def norm(s):
    return re.sub(r"[^a-z0-9%.]", "", s.lower())


def numeric_distractors(gold, pool, rng):
    g = float(gold.rstrip("%"))
    pct = gold.endswith("%")
    dec = len(gold.rstrip("%").split(".")[1]) if "." in gold else 0
    cands = []
    for p in pool:
        p = p.strip()
        if NUM.match(p) and p != gold:
            cands.append(p)
    rng.shuffle(cands)
    out = []
    for c in cands:
        if c not in out and abs(float(c.rstrip("%")) - g) > 1e-9:
            out.append(c)
        if len(out) == 3:
            break
    if dec == 0 and abs(g) < 30:
        cands2 = [g + d for d in (1, -1, 2, -2, 3, -3, 4, 5, -4, 6)]
        if g >= 0:
            cands2 = [v for v in cands2 if v >= 0] + [g + 7, g + 8, g + 9]
    else:
        cands2 = [g * m for m in (0.5, 2, 1.5, 0.75, 1.25, 3, 0.25, 4, 0.9, 1.1)] + [g + 10 ** -dec * k for k in (1, -1, 2, -2, 5)]
    rng.shuffle(cands2)
    for v in cands2:
        if len(out) == 3:
            break
        s = f"{v:.{dec}f}" + ("%" if pct else "")
        if s != gold and s not in out:
            out.append(s)
    return out


def overlaps(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def font(size, path=None):
    from PIL import ImageFont

    for f in [path or os.environ.get("DECISION_INDEX_FONT")] + FONTS:
        if not f:
            continue
        try:
            return ImageFont.truetype(f, size)
        except Exception:
            pass
    return ImageFont.load_default()


def font_name(path=None):
    for f in [path or os.environ.get("DECISION_INDEX_FONT")] + FONTS:
        if f and os.path.exists(f):
            return f
    return "PIL default"


def draw_marks(im, boxes, font_path=None):
    from PIL import ImageDraw

    im = im.copy()
    d = ImageDraw.Draw(im)
    W, Hh = im.size
    lw = max(2, int(min(W, Hh) / 300))
    fs = max(14, int(min(W, Hh) / 40))
    f = font(fs, font_path)
    colors = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4"]
    for i, b in enumerate(boxes):
        c = colors[i % len(colors)]
        d.rectangle(b, outline=c, width=lw)
        tx, ty = b[0], max(0, b[1] - fs - 4)
        d.rectangle([tx, ty, tx + fs + 6, ty + fs + 4], fill=c)
        d.text((tx + 3, ty + 1), LETTERS[i], fill="white", font=f)
    return im


class Letters:
    def __init__(self, rng):
        self.rng = rng
        self.counts = Counter()

    def pick(self, n):
        valid = list(LETTERS[:n])
        lo = min(self.counts[l] for l in valid)
        c = [l for l in valid if self.counts[l] == lo]
        l = self.rng.choice(c)
        self.counts[l] += 1
        return LETTERS.index(l)


def arrange(gold, distractors, pos, rng):
    d = distractors[:]
    rng.shuffle(d)
    opts = d[:pos] + [gold] + d[pos:]
    return opts, LETTERS[pos]


def parse_amount(s):
    if s is None:
        return None
    t = str(s).strip()
    t = re.sub(r"(?i)^(rp\.?|idr|rm|@|x)\s*", "", t)
    t = re.sub(r"(?i)\s*(rp|sr|zr|z|s)$", "", t).strip()
    neg = t.startswith("-") or t.startswith("(")
    t = t.strip("-() ")
    if re.fullmatch(r"\d{1,3}([.,]\d{3})+", t):
        v = int(re.sub(r"\D", "", t))
    elif re.fullmatch(r"\d+", t):
        v = int(t)
    elif re.fullmatch(r"\d{1,3}(,\d{3})*\.\d{1,2}", t):
        v = float(t.replace(",", ""))
    elif re.fullmatch(r"\d+,\d{1,2}", t):
        v = float(t.replace(",", "."))
    else:
        return None
    if neg or v <= 0:
        return None
    return v


def fmt_int(v):
    return f"{int(round(v)):,}"


def fmt_dec(v):
    return f"{v:,.2f}"


def as_list(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def pick_by_rank(gold_v, pool, k, rng, counts=None, cap=None):
    above = [p for p in pool if p[0] > gold_v]
    below = [p for p in pool if p[0] < gold_v]
    k = min(k, len(above) + len(below))
    lo, hi = max(0, k - len(below)), min(k, len(above))
    feas = list(range(lo, hi + 1))
    if counts is None:
        na = rng.choice(feas)
    else:
        if cap is not None:
            feas = [n for n in feas if counts[n] < cap]
            if not feas:
                return None
        m = min(counts[n] for n in feas)
        na = rng.choice([n for n in feas if counts[n] == m])
        counts[na] += 1
    return rng.sample(above, na) + rng.sample(below, k - na)


CORD_VALUE_Q = {
    "total_price": "What is the total amount on this receipt?",
    "changeprice": "How much change was given back?",
    "tax_price": "What is the tax amount on this receipt?",
    "subtotal_price": "What is the subtotal on this receipt?",
    "item_price": "What is the price listed for the item \"{nm}\"?",
}
CORD_VALUE_FIELDS = ["total_price", "item_price", "changeprice", "tax_price", "subtotal_price"]
CORD_PRESENT_Q = {
    "tax_price": "a tax amount",
    "service_price": "a service charge",
    "discount_price": "a discount",
    "changeprice": "an amount of change given back",
    "creditcardprice": "a credit card payment",
    "cashprice": "a cash payment amount",
    "subtotal_price": "a subtotal",
}


def cord_fields(gp):
    sub = gp.get("sub_total") or {}
    tot = gp.get("total") or {}
    if isinstance(sub, list):
        sub = {k: v for d in sub for k, v in d.items()}
    if isinstance(tot, list):
        tot = {k: v for d in tot for k, v in d.items()}
    f = {}
    for k in ["subtotal_price", "tax_price", "service_price", "discount_price"]:
        if sub.get(k) is not None:
            f[k] = sub[k]
    for k in ["total_price", "cashprice", "changeprice", "creditcardprice", "emoneyprice"]:
        if tot.get(k) is not None:
            f[k] = tot[k]
    amounts = []
    for k, v in f.items():
        a = parse_amount(v if not isinstance(v, list) else v[0])
        if a:
            amounts.append((a, k))
    for v in as_list(sub.get("etc")):
        a = parse_amount(v)
        if a:
            amounts.append((a, "sub_etc"))
    items = []
    for m in as_list(gp.get("menu")):
        if not isinstance(m, dict):
            continue
        for key in ["price", "unitprice", "itemsubtotal"]:
            for v in as_list(m.get(key)):
                a = parse_amount(v)
                if a:
                    amounts.append((a, "menu_" + key))
        nm = m.get("nm")
        nm = nm if isinstance(nm, str) else (nm[0] if isinstance(nm, list) and nm and isinstance(nm[0], str) else None)
        pr = parse_amount(m.get("price") if not isinstance(m.get("price"), list) else m["price"][0])
        if nm and pr:
            items.append((nm.strip(), pr))
        for s in as_list(m.get("sub")):
            if isinstance(s, dict):
                for v in as_list(s.get("price")):
                    a = parse_amount(v)
                    if a:
                        amounts.append((a, "menu_sub_price"))
    return f, amounts, items


def funsd_type(t):
    if re.search(r"\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|(?i:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.? \d", t):
        return "date"
    d = sum(c.isdigit() for c in t)
    if d and d >= 0.5 * sum(c.isalnum() for c in t):
        return "number"
    return "text"
