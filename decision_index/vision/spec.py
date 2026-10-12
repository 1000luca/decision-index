import json
from functools import lru_cache
from pathlib import Path

EDITION = "0.3"
PINS = Path(__file__).resolve().parents[1] / "data" / "vision-0.3.json"
FUNSD_URL = "https://guillaumejaume.github.io/FUNSD/dataset.zip"
FUNSD_SHA256 = "c31735649e4f441bcbb4fd0f379574f7520b42286e80b01d80b445649d54761f"
WEIGHTS = {"public": 0.5, "private": 0.5}
READOUT = "The answer is ("

BENCHMARKS = [
    {"name": "CV-Bench", "sets": ["full_cvbench"], "weight": 1.0},
    {"name": "BLINK", "sets": ["full_blink"], "weight": 1.0},
    {"name": "RealWorldQA", "sets": ["full_realworldqa"], "weight": 1.0},
    {"name": "CharXiv", "sets": ["full_charxiv"], "weight": 1.0},
    {"name": "InfographicVQA", "sets": ["full_infovqa"], "weight": 1.0},
    {"name": "Mind2Web", "sets": ["full_mind2web"], "weight": 1.0},
    {"name": "Winoground", "sets": ["full_winoground"], "weight": 1.0, "subtask": "text"},
    {"name": "KIE (CORD+FUNSD)", "sets": ["full_cord", "full_funsd"], "weight": 1.0, "balance_present": True},
    {"name": "Moderation (Hateful Memes)", "sets": ["full_hateful_memes"], "weight": 1.0},
    {"name": "R-Bench-M", "sets": ["full_rbench_m"], "weight": 0.5, "readout": True},
    {"name": "MMMU-Pro vision", "sets": ["full_mmmu_pro_vision"], "weight": 0.25, "readout": True},
]
SETS = [s for b in BENCHMARKS for s in b["sets"]]


def benchmark(name):
    return next(b for b in BENCHMARKS if b["name"] == name)


def benchmark_of(set_name):
    return next(b for b in BENCHMARKS if set_name in b["sets"])


@lru_cache(maxsize=None)
def pins():
    return json.loads(PINS.read_text())
