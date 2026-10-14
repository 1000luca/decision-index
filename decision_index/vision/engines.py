import math
import os
import re
import threading
import time

from decision_index.engines.base import Engine, Unsupported
from decision_index.engines.http import CAPACITY_MARKERS
from decision_index.vision import images as img
from decision_index.vision.rows import OPTION_LETTERS


class ThreadClients:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.local = threading.local()
        self.lock = threading.Lock()
        self.all = []

    def get(self):
        c = getattr(self.local, "client", None)
        if c is None:
            import httpx

            c = self.local.client = httpx.Client(**self.kwargs)
            with self.lock:
                self.all.append(c)
        return c

    def close(self):
        with self.lock:
            for c in self.all:
                c.close()
            self.all.clear()
        self.local = threading.local()

MAX_PIXELS = img.MAX_PIXELS
TAIL = "Answer with the option's letter from the given choices directly."


def single_choice(questions):
    if len(questions) != 1:
        raise Unsupported("vision engines take one choice question per request")
    key, q = next(iter(questions.items()))
    if q["type"] != "choice":
        raise Unsupported("vision engines take choice questions only")
    keys = list(q["criteria"])
    if keys != list(OPTION_LETTERS[:len(keys)]):
        raise Unsupported("criteria keys must be the letters A, B, C, ...")
    return key, q, keys


def prompt_text(q, prompt=None):
    if (prompt or {}).get("options_in_image"):
        return f"{q['instructions']} {TAIL}"
    opts = "\n".join(f"{k}. {v}" for k, v in q["criteria"].items())
    return f"{q['instructions']}\n{opts}\n{TAIL}"


def answer(key, keys, choice, probs, model, usage=None):
    out = {"model": model, "answers": {key: {"type": "choice", "choice": choice, "probabilities": dict(zip(keys, probs))}}}
    if usage:
        out["usage"] = usage
    return out


def pixels(ref, root=None, max_pixels=None):
    from PIL import Image
    import io

    with Image.open(io.BytesIO(img.load(ref, root))) as im:
        return img.fit(im.width, im.height, max_pixels)[0] * img.fit(im.width, im.height, max_pixels)[1]


def resize_policy(max_pixels):
    if not max_pixels:
        return "none by the kit: original image bytes"
    return f"images over {max_pixels} pixels are downscaled to fit (aspect kept, LANCZOS); smaller images are sent as their original bytes"


def processor_budget(model, **kwargs):
    try:
        from transformers import AutoProcessor

        ip = getattr(AutoProcessor.from_pretrained(model, **kwargs), "image_processor", None)
    except Exception:
        return {}
    out = {}
    for k in ("max_pixels", "min_pixels", "size", "patch_size", "merge_size", "max_image_tiles", "max_num_patches", "image_seq_length"):
        v = getattr(ip, k, None)
        if v is not None:
            out[k] = v if isinstance(v, (int, float, str)) else dict(v) if hasattr(v, "items") else str(v)
    return out


def softmax(vals):
    top = max(vals)
    if top == -math.inf:
        return [1 / len(vals)] * len(vals)
    e = [math.exp(v - top) for v in vals]
    s = sum(e)
    return [x / s for x in e]


class VllmVisionChat(Engine):
    name = "vllm-vl"
    latency = "Batched llm.chat wall time per chunk divided by the rows in the chunk; excludes model loading."

    def __init__(self, model, max_model_len=32768, max_images=8, gpu_memory_utilization=0.9, seed=0, llm_options=None, mm_processor_kwargs=None, max_pixels=MAX_PIXELS, image_root=None, **options):
        super().__init__(**options)
        from vllm import LLM

        self.model = model
        self.image_root = image_root
        self.max_pixels = max_pixels
        llm_options = dict(llm_options or {})
        if mm_processor_kwargs:
            llm_options["mm_processor_kwargs"] = mm_processor_kwargs
        self.llm = LLM(model=model, max_model_len=max_model_len, limit_mm_per_prompt={"image": max_images}, gpu_memory_utilization=gpu_memory_utilization,
                       logprobs_mode="processed_logprobs", seed=seed, **llm_options)
        cfg = getattr(self.llm.llm_engine, "model_config", None)
        self.image_token_id = getattr(getattr(cfg, "hf_config", None), "image_token_id", None)
        tok = self.llm.get_tokenizer()
        self.letter_ids = [tok.convert_tokens_to_ids(l) for l in OPTION_LETTERS]
        import vllm

        self.provenance = {"kind": "vllm-chat", "model": model, "vllm": vllm.__version__, "max_model_len": max_model_len,
                           "max_pixels": max_pixels, "image_resize": resize_policy(max_pixels) + "; then the model's own processor",
                           "mm_processor_kwargs": mm_processor_kwargs or {}, "processor_image_budget": {**processor_budget(model, **(mm_processor_kwargs or {}))},
                           "policy": "one user turn (images, then question, lettered options and the letter instruction); one decoding step restricted to the option letters; the readout prefix is continued as an assistant turn where the row asks for it"}

    def conversation(self, payload):
        key, q, keys = single_choice(payload["questions"])
        content = [{"type": "image_pil", "image_pil": img.pil(p, self.image_root, self.max_pixels).convert("RGB")} for p in payload.get("images") or []]
        content.append({"type": "text", "text": prompt_text(q, payload.get("prompt"))})
        msgs = [{"role": "user", "content": content}]
        readout = (payload.get("prompt") or {}).get("readout")
        if readout:
            msgs.append({"role": "assistant", "content": readout})
        return msgs, key, keys, readout

    def read(self, o, key, keys):
        k = len(keys)
        lp = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
        vals = [float(lp[t].logprob) if t in lp else -math.inf for t in self.letter_ids[:k]]
        choice = keys[max(range(k), key=lambda i: vals[i])]
        raw = {"letter_logprobs": vals}
        if self.image_token_id is not None:
            raw["image_tokens"] = sum(t == self.image_token_id for t in o.prompt_token_ids)
        return answer(key, keys, choice, softmax(vals), self.model, {"input_tokens": len(o.prompt_token_ids)}), raw

    def batch(self, payloads):
        from vllm import SamplingParams

        out = [None] * len(payloads)
        groups = {}
        for i, p in enumerate(payloads):
            try:
                groups.setdefault((p.get("prompt") or {}).get("readout"), []).append((i, *self.conversation(p)))
            except Exception as e:
                out[i] = e
        for readout, members in groups.items():
            convs = [m[1] for m in members]
            sps = [SamplingParams(max_tokens=1, temperature=0.0, logprobs=len(m[3]), allowed_token_ids=self.letter_ids[:len(m[3])]) for m in members]
            kw = dict(add_generation_prompt=not readout, continue_final_message=bool(readout), use_tqdm=False)
            try:
                outs = self.llm.chat(convs, sps, **kw)
            except Exception:
                outs = []
                for cv, sp in zip(convs, sps):
                    try:
                        outs.append(self.llm.chat([cv], [sp], **kw)[0])
                    except Exception as e:
                        outs.append(e)
            for (i, _, key, keys, _), o in zip(members, outs):
                if isinstance(o, Exception):
                    out[i] = Unsupported(str(o)) if any(m in str(o) for m in CAPACITY_MARKERS) else o
                else:
                    out[i] = self.read(o, key, keys)
                    out[i][1]["image_pixels"] = [pixels(p, self.image_root, self.max_pixels) for p in payloads[i].get("images") or []]
        return out

    def __call__(self, state, questions, images=(), prompt=None):
        r = self.batch([{"state": state, "questions": questions, "images": images, "prompt": prompt}])[0]
        if isinstance(r, Exception):
            raise r
        return r

    def warmup(self):
        pass


class HttpDecide(Engine):
    name = "decide"
    latency = "HTTP request wall time against the configured decide endpoint; excludes server startup."

    def __init__(self, base_url=None, path=None, protocol="decide", model="default", token_env="DECISION_INDEX_API_KEY", timeout=900, retries=4,
                 image_encoding="data_url", thinking="off", extra=None, max_pixels=MAX_PIXELS, image_root=None, **options):
        super().__init__(**options)
        base_url = base_url or os.environ.get("DECISION_INDEX_BASE_URL")
        if not base_url:
            raise ValueError("HttpDecide needs base_url (or DECISION_INDEX_BASE_URL)")
        if protocol not in ("decide", "decisions"):
            raise ValueError("protocol is decide or decisions")
        headers = {}
        token = os.environ.get(token_env)
        if token:
            headers["Authorization"] = "Bearer " + token
        self.path = path or ("/v1/decide" if protocol == "decide" else "/v1/decisions")
        self.max_pixels = max_pixels
        self.protocol, self.model, self.retries, self.image_encoding, self.thinking, self.image_root = protocol, model, retries, image_encoding, thinking, image_root
        self.extra = extra or {}
        self.clients = ThreadClients(base_url=base_url, timeout=timeout, headers=headers)
        self.provenance = {"kind": "http-decide", "base_url": base_url, "path": self.path, "protocol": protocol, "model": model, "image_encoding": image_encoding,
                           "max_pixels": max_pixels, "image_resize": resize_policy(max_pixels) + "; the server's own image settings then apply",
                           "request_options": self.extra}

    def body(self, state, questions, images):
        key, q, keys = single_choice(questions)
        enc = [img.encode(p, self.image_encoding, self.image_root, self.max_pixels) for p in images or []]
        if self.protocol == "decide":
            b = {"kind": "choice", "state": [{"image": u} for u in enc] if enc else (state or {}), "question": q["instructions"],
                 "options": [q["criteria"][k] for k in keys]}
            if self.thinking is not None:
                b["thinking"] = self.thinking
        else:
            b = {"model": self.model, "state": state or {}, "questions": {key: q}}
            if enc:
                b["images"] = enc
        return {**b, **self.extra}, key, keys

    def parse(self, d, key, keys):
        if self.protocol == "decide":
            probs = [float(p) for p in d["probabilities"]][:len(keys)]
            choice = keys[d["choice_index"]]
            usage = {"input_tokens": (d.get("usage") or {}).get("prompt_tokens")}
        else:
            a = d["answers"][key]
            pr = a.get("probabilities") or {a["choice"]: 1.0}
            probs = [float(pr.get(k, 0.0)) for k in keys]
            choice = a["choice"]
            usage = {"input_tokens": (d.get("usage") or {}).get("input_tokens")}
        s = sum(probs)
        probs = [p / s for p in probs] if s > 0 else [float(k == choice) for k in keys]
        return answer(key, keys, choice, probs, self.model, usage)

    def __call__(self, state, questions, images=(), prompt=None):
        body, key, keys = self.body(state, questions, images)
        err = None
        for attempt in range(self.retries):
            r = None
            try:
                r = self.clients.get().post(self.path, json=body)
            except Exception as e:
                err = repr(e)[:300]
            if r is not None:
                if r.status_code == 200:
                    d = r.json()
                    d["image_pixels"] = [pixels(p, self.image_root, self.max_pixels) for p in images or []]
                    return self.parse(d, key, keys), d
                err = f"http {r.status_code}: {r.text[:300]}"
                if r.status_code in (400, 404, 413, 422):
                    if any(m in r.text for m in CAPACITY_MARKERS):
                        raise Unsupported(err)
                    break
            time.sleep(3 * (attempt + 1))
        raise RuntimeError(err)

    def close(self):
        self.clients.close()


def parse_short(text, k):
    t = (text or "").strip()
    m = re.match(r"^\(?([A-L])\b", t) or re.search(r"\b([A-L])\b", t)
    if m and OPTION_LETTERS.index(m.group(1)) < k:
        return m.group(1)
    return None


def parse_long(text, k):
    t = text or ""
    m = re.findall(r"(?:answer|option)\W{0,12}(?:is\W{0,4})?\(?([A-L])\b", t, re.I) or re.findall(r"\b([A-L])\b", t)
    if m and OPTION_LETTERS.index(m[-1].upper()) < k:
        return m[-1].upper()
    return None


class OpenAIVision(Engine):
    name = "openai-vlm"
    latency = "Wall time of the chat-completions calls for the row (including the long-answer retry when the short reply has no letter)."

    def __init__(self, model, base_url=None, token_env="DECISION_INDEX_API_KEY", max_pixels=MAX_PIXELS, max_tokens=16, long_max_tokens=1024,
                 timeout=180, retries=6, extra_body=None, image_root=None, **options):
        super().__init__(**options)
        base_url = base_url or os.environ.get("DECISION_INDEX_BASE_URL") or "https://router.huggingface.co/v1"
        token = os.environ.get(token_env)
        if not token and "huggingface.co" in base_url:
            from huggingface_hub import get_token

            token = get_token()
        self.model, self.max_pixels, self.max_tokens, self.long_max_tokens, self.retries, self.image_root = model, max_pixels, max_tokens, long_max_tokens, retries, image_root
        self.extra_body = extra_body or {}
        self.clients = ThreadClients(base_url=base_url.rstrip("/"), timeout=timeout, headers={"Authorization": "Bearer " + token} if token else {})
        self.provenance = {"kind": "openai-chat", "base_url": base_url, "model": model, "max_pixels": max_pixels, "image_resize": "every image re-encoded as JPEG quality 90" + (f", downscaled to fit {max_pixels} pixels when larger (aspect kept, bicubic)" if max_pixels else ""), "extra_body": self.extra_body,
                           "policy": "images re-encoded as JPEG (quality 90) under max_pixels, then the question with lettered options; temperature 0; the first option letter in the reply, else one longer reply and its last stated answer letter"}

    def complete(self, content, max_tokens):
        err = None
        for attempt in range(self.retries):
            try:
                r = self.clients.get().post("/chat/completions", json={"model": self.model, "messages": [{"role": "user", "content": content}],
                                                                "temperature": 0, "max_tokens": max_tokens, **self.extra_body})
                if r.status_code == 200:
                    d = r.json()
                    return d["choices"][0]["message"].get("content") or "", d
                err = f"http {r.status_code}: {r.text[:300]}"
                if r.status_code in (400, 413, 422) and any(m in r.text for m in CAPACITY_MARKERS):
                    raise Unsupported(err)
            except Unsupported:
                raise
            except Exception as e:
                err = repr(e)[:300]
            time.sleep(min(60, 2 ** attempt * 3))
        raise RuntimeError(err)

    def __call__(self, state, questions, images=(), prompt=None):
        key, q, keys = single_choice(questions)
        urls = [img.jpeg_data_url(p, self.image_root, self.max_pixels) for p in images or []]
        content = [{"type": "image_url", "image_url": {"url": u}} for u in urls]
        content.append({"type": "text", "text": prompt_text(q, prompt)})
        text, raw = self.complete(content, self.max_tokens)
        choice = parse_short(text, len(keys))
        if choice is None:
            text, raw = self.complete(content, self.long_max_tokens)
            choice = parse_long(text, len(keys))
        if choice is None:
            raise Unsupported("no option letter in the reply: " + text[-200:])
        usage = raw.get("usage") or {}
        return answer(key, keys, choice, [float(k == choice) for k in keys], raw.get("model") or self.model, {"input_tokens": usage.get("prompt_tokens")}), {"text": text, "model": raw.get("model"), "image_pixels": [pixels(u) for u in urls]}

    def warmup(self):
        pass

    def close(self):
        self.clients.close()
