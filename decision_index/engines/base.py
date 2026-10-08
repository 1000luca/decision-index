import json
import math
import random


class Unsupported(ValueError):
    pass


class NativeAbstention(Exception):
    def __init__(self, raw):
        self.raw = raw
        super().__init__("Native model selected its additional abstention outcome")


def text(x):
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, separators=(",", ":"))


def validate(questions, response):
    if set(response.get("answers", {})) != set(questions):
        raise ValueError("Question keys mismatch")
    for k, q in questions.items():
        a = response["answers"][k]
        if a.get("type") != q["type"]:
            raise ValueError("Type mismatch")
        if q["type"] == "choice":
            if a.get("choice") not in q["criteria"]:
                raise ValueError("Invalid chosen answer")
            p = a.get("probabilities", {})
            if set(p) != set(q["criteria"]) or any(not math.isfinite(v) or not 0 <= v <= 1 for v in p.values()) or abs(sum(p.values()) - 1) > 0.01:
                raise ValueError("Incomplete/invalid probability distribution")
        elif q["type"] == "noul":
            if not math.isfinite(a["noul"]) or not 0 <= a["noul"] <= 1:
                raise ValueError("Invalid noul probability")
        else:
            raise ValueError("Unsupported question type " + str(q["type"]))


class Engine:
    name = "engine"
    provenance = {}
    latency = "In-process request wall time including prompt construction; excludes model loading."

    def __init__(self, **options):
        self.options = options

    def __call__(self, state, questions):
        raise NotImplementedError

    def warmup(self):
        warm = {"warmup": {"type": "choice", "instructions": "Which color is named?", "criteria": {"red": "red", "blue": "blue"}}}
        for _ in range(2):
            try:
                self("The color is red.", warm)
            except NativeAbstention:
                pass

    def runtime(self):
        return {}

    def synchronize(self):
        pass

    def close(self):
        pass


class RandomEngine(Engine):
    name = "random"

    def __init__(self, seed=20260919, **options):
        super().__init__(**options)
        self.rng = random.Random(seed)
        self.provenance = {"kind": "baseline", "seed": seed, "policy": "uniform choice over the supplied options; a sanity check, not a model"}

    def __call__(self, state, questions, **inputs):
        answers = {}
        for k, q in questions.items():
            if q["type"] == "choice":
                keys = list(q["criteria"])
                choice = self.rng.choice(keys)
                answers[k] = {"type": "choice", "choice": choice, "probabilities": {x: 1 / len(keys) for x in keys}}
            elif q["type"] == "noul":
                answers[k] = {"type": "noul", "noul": 0.5}
            else:
                raise Unsupported("Unsupported question type " + q["type"])
        return {"model": "random", "answers": answers}, None
