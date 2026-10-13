import base64
import io
import json
from pathlib import Path

import pytest

from decision_index.vision import rows as vrows
from decision_index.vision import spec

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")

FIXTURES = Path(__file__).parent / "fixtures"


def png_bytes(color=(200, 10, 10), size=(8, 6)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def make_suite(tmp_path, n_per_bench=3):
    root = tmp_path / "vision"
    rows = []
    for b in spec.BENCHMARKS:
        s = b["sets"][0]
        (root / "full" / s / "img").mkdir(parents=True, exist_ok=True)
        for i in range(n_per_bench):
            p = root / "full" / s / "img" / f"{i}.png"
            p.write_bytes(png_bytes((i * 40, 10, 10)))
            it = {"id": f"{s}:{i}", "bench": s, "subtask": b.get("subtask") or "x", "images": [str(p.relative_to(root))],
                  "question": f"q{i} <image 1>", "options": ["one", "two", "three"], "gold": "A"}
            if b.get("readout"):
                it["vision_only"] = True
            rows.append(vrows.to_row(root, it))
    path = root / vrows.ROWS_FILE
    path.write_text("".join(vrows.dumps(r) + "\n" for r in rows))
    return root, rows


def test_pins_cover_every_public_benchmark():
    pins = spec.pins()
    assert set(pins["sets"]) == set(spec.SETS)
    assert set(pins["benchmarks"]) == {b["name"] for b in spec.BENCHMARKS}
    assert sum(v["rows"] for v in pins["benchmarks"].values()) == pins["rows"]["count"] == 12210
    for s in pins["sets"].values():
        assert s["shards"] and all(len(x["sha256"]) == 64 for x in s["shards"])
        assert s["revision"] is None or len(s["revision"]) == 40


def test_weights_and_public_split():
    w = {b["name"]: b["weight"] for b in spec.BENCHMARKS}
    assert len(w) == 11 and w["R-Bench-M"] == 0.5 and w["MMMU-Pro vision"] == 0.25
    assert sum(1 for v in w.values() if v == 1.0) == 9
    assert spec.WEIGHTS == {"public": 0.5, "private": 0.5}


def test_balance_keeps_equal_yes_no_per_group():
    rows = [{"id": f"r{i}:present:tax", "subtask": "field_present", "gold": "A" if i < 5 else "B"} for i in range(8)]
    rows += [{"id": "v1", "subtask": "field_value", "gold": "C"}]
    kept = vrows.balance(rows, lambda r: r["id"].split(":")[-1])
    present = [r for r in kept if r["subtask"] == "field_present"]
    assert len(present) == 6 and sum(r["gold"] == "A" for r in present) == 3
    assert kept[0]["id"] == "v1"


def test_winoground_keeps_caption_choice_only():
    rows = [{"id": "a", "subtask": "text"}, {"id": "b", "subtask": "image"}]
    assert [r["id"] for r in vrows.select("full_winoground", rows)] == ["a"]


def test_request_and_prompt_formats():
    from decision_index.vision.engines import prompt_text

    q = vrows.request({"question": "Which <image 1>?", "options": ["x", 2]})["answer"]
    assert q == {"type": "choice", "instructions": "Which [image 1]?", "criteria": {"A": "x", "B": "2"}}
    assert prompt_text(q) == "Which [image 1]?\nA. x\nB. 2\nAnswer with the option's letter from the given choices directly."
    v = vrows.request({"question": "", "options": list("abcd"), "vision_only": True})["answer"]
    assert v["criteria"] == {"A": "A", "B": "B", "C": "C", "D": "D"}
    assert prompt_text(v, {"options_in_image": True}) == "Answer the multiple-choice question shown in the image (options A-D). Answer with the option's letter from the given choices directly."


def test_index_reproduces_published_public_numbers():
    from decision_index.vision.score import combine

    fx = json.loads((FIXTURES / "vision-public-0.3.json").read_text())
    n = {k: v["rows"] for k, v in spec.pins()["benchmarks"].items()}
    for entrant in fx["entrants"].values():
        skills = {}
        for name, c in entrant["correct"].items():
            m = fx["chance"][name]
            skills[name] = (c / n[name] - m) / (1 - m)
            assert round(100 * skills[name], 2) == entrant["bench"][name]
        assert round(combine(skills), 2) == entrant["public"]


def test_score_skill_clip_and_weights(tmp_path):
    from decision_index.vision.score import score

    root, rows = make_suite(tmp_path)
    preds = {r["id"]: "A" for r in rows}
    s = score(rows, preds)
    assert s["complete"] and s["public_vision_index"] == pytest.approx(100.0)
    preds = {r["id"]: "B" for r in rows}
    s = score(rows, preds)
    assert s["public_vision_index"] == 0.0 and s["benchmarks"]["CV-Bench"]["skill"] == pytest.approx(-0.5)
    preds = {r["id"]: ("A" if r["benchmark"] == "MMMU-Pro vision" else "B") for r in rows}
    assert score(rows, preds)["public_vision_index"] == pytest.approx(100 * 0.25 / 9.75)
    partial = score(rows, {rows[0]["id"]: "A"})
    assert not partial["complete"] and partial["missing_rows"] == len(rows) - 1


def test_lab_style_prediction_directory(tmp_path):
    from decision_index.vision.score import predictions

    d = tmp_path / "preds"
    d.mkdir()
    (d / "full_cvbench.jsonl").write_text(json.dumps({"id": "full_cvbench:0", "pred": "B"}) + "\n")
    (tmp_path / "results.jsonl").write_text(json.dumps({"run_id": "x", "status": "ok", "response": {"answers": {"answer": {"choice": "C"}}}}) + "\n"
                                            + json.dumps({"run_id": "y", "status": "error"}) + "\n")
    assert predictions(d) == {"full_cvbench:0": "B"}
    assert predictions(tmp_path / "results.jsonl") == {"x": "C", "y": None}


def test_image_refs(tmp_path):
    from decision_index.vision import images

    raw = png_bytes()
    (tmp_path / "a.png").write_bytes(raw)
    url = images.data_url("a.png", tmp_path)
    assert url.startswith("data:image/png;base64,") and images.load(url) == raw
    assert images.load(base64.b64encode(raw).decode()) == raw
    assert images.load(str(tmp_path / "a.png")) == raw
    assert images.encode("a.png", "base64", tmp_path) == base64.b64encode(raw).decode()
    assert images.encode("a.png", "path", tmp_path) == str(tmp_path / "a.png")
    assert images.jpeg_data_url(raw, max_pixels=12).startswith("data:image/jpeg;base64,")
    with pytest.raises(FileNotFoundError):
        images.load("missing.png", tmp_path)


def mock_client(handler):
    import httpx

    from decision_index.vision.engines import ThreadClients

    return ThreadClients(base_url="http://test", transport=httpx.MockTransport(handler))


def test_decide_protocol(tmp_path):
    from decision_index.engines import validate
    from decision_index.vision.engines import HttpDecide

    (tmp_path / "a.png").write_bytes(png_bytes())
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        import httpx

        return httpx.Response(200, json={"choice_index": 1, "probabilities": [0.2, 0.8], "usage": {"prompt_tokens": 7}})

    e = HttpDecide(base_url="http://test", image_root=str(tmp_path))
    e.clients = mock_client(handler)
    q = {"answer": {"type": "choice", "instructions": "Pick", "criteria": {"A": "x", "B": "y"}}}
    resp, raw = e({}, q, images=["a.png"])
    validate(q, resp)
    assert seen["path"] == "/v1/decide" and seen["body"]["options"] == ["x", "y"] and seen["body"]["thinking"] == "off"
    assert seen["body"]["state"][0]["image"].startswith("data:image/png;base64,")
    assert resp["answers"]["answer"]["choice"] == "B" and resp["usage"]["input_tokens"] == 7
    assert raw["image_pixels"] == [48]


def test_decisions_protocol(tmp_path):
    from decision_index.vision.engines import HttpDecide

    (tmp_path / "a.png").write_bytes(png_bytes())
    seen = {}

    def handler(request):
        import httpx

        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"answers": {"answer": {"type": "choice", "choice": "A", "probabilities": {"A": 0.9, "B": 0.1}}}})

    e = HttpDecide(base_url="http://test", protocol="decisions", image_encoding="base64", image_root=str(tmp_path))
    e.clients = mock_client(handler)
    q = {"answer": {"type": "choice", "instructions": "Pick", "criteria": {"A": "x", "B": "y"}}}
    resp, _ = e({}, q, images=["a.png"])
    assert seen["path"] == "/v1/decisions" and seen["body"]["questions"] == q and not seen["body"]["images"][0].startswith("data:")
    assert resp["answers"]["answer"]["probabilities"] == {"A": 0.9, "B": 0.1}


def test_openai_vision_parses_letters(tmp_path):
    from decision_index.engines import Unsupported
    from decision_index.vision.engines import OpenAIVision, parse_long, parse_short

    assert parse_short("(B) because", 3) == "B" and parse_short("D", 3) is None
    assert parse_long("thinking... A is wrong, so the answer is (C).", 4) == "C"
    replies = iter(["I think", "The answer is B"])

    def handler(request):
        import httpx

        body = json.loads(request.content)
        assert body["messages"][0]["content"][0]["type"] == "image_url" and body["temperature"] == 0
        return httpx.Response(200, json={"model": "m", "choices": [{"message": {"content": next(replies)}}], "usage": {"prompt_tokens": 3}})

    e = OpenAIVision(model="m", base_url="http://test", token_env="UNSET_TOKEN_ENV", max_pixels=10000)
    e.clients = mock_client(handler)
    q = {"answer": {"type": "choice", "instructions": "Pick", "criteria": {"A": "x", "B": "y"}}}
    resp, raw = e({}, q, images=[png_bytes(size=(400, 300))])
    assert resp["answers"]["answer"]["choice"] == "B" and raw["image_pixels"][0] <= e.max_pixels
    replies = iter(["no idea", "still no idea"])
    with pytest.raises(Unsupported):
        e({}, q, images=[png_bytes()])


def test_runner_and_score_with_random_engine(tmp_path):
    from decision_index.vision.run import run
    from decision_index.vision.score import score_run

    root, rows = make_suite(tmp_path, n_per_bench=2)
    out = tmp_path / "run"
    final = run("random", {}, root, out, workers=2, chunk=3, log=lambda *_: None)
    assert final["counts"] == {"ok": len(rows)}
    res = [json.loads(l) for l in (out / "results.jsonl").read_text().splitlines()]
    assert {r["run_id"] for r in res} == {r["id"] for r in rows} and res[0]["payload"]["images"][0].startswith("full/")
    env = json.loads((out / "environment.json").read_text())
    assert env["effective_image_budget"]["max_image_pixels"] == 0
    s = score_run(root / vrows.ROWS_FILE, out / "results.jsonl", out)
    assert s["complete"] and (out / "vision-scores.json").exists()
    again = run("random", {}, root, out, log=lambda *_: None)
    assert again["completed"] == 0


def test_verify_flags_a_mismatching_suite(tmp_path):
    root, _ = make_suite(tmp_path, n_per_bench=1)
    report = vrows.verify(root)
    assert report["ok"] is False and report["sets"]["full_cord"] == {"missing": True}
    assert report["benchmarks"]["CV-Bench"]["rows"] is False


def test_cli_and_job_command():
    from decision_index.cli import build_parser
    from decision_index.hf_job import vision_command

    a = build_parser().parse_args(["suite", "rebuild", "--vision", "--work", "w", "--sets", "full_blink"])
    assert a.vision and a.sets == ["full_blink"]
    a = build_parser().parse_args(["vision", "run", "--engine", "decide", "--option", "base_url=http://x", "--dir", "d"])
    assert a.vision_command == "run" and a.workers == 8
    cmd = vision_command("me/results", "vllm-vl", {"model": "org/m"}, "r1")
    assert "suite rebuild --vision" in cmd and "vllm==0.30.0" in cmd and "--option model=org/m" in cmd and "[vision]" in cmd
    cmd = vision_command("me/results", "openai-vlm", {"model": "org/m"}, "r1", suite_dataset="me/vision-suite")
    assert "hf download me/vision-suite" in cmd and "vllm" not in cmd


class BudgetEngine:
    provenance = {}
    latency = ""

    def __init__(self, **options):
        pass

    def runtime(self):
        return {}

    def __call__(self, state, questions, images=(), prompt=None):
        keys = list(questions["answer"]["criteria"])
        return {"answers": {"answer": {"type": "choice", "choice": keys[0], "probabilities": {k: 1 / len(keys) for k in keys}}}}, {"image_pixels": [4096], "image_tokens": 7}

    def close(self):
        pass


def test_runner_records_effective_image_budget(tmp_path, monkeypatch):
    from decision_index.engines import REGISTRY
    from decision_index.vision.run import run

    monkeypatch.setitem(REGISTRY, "budget", f"{__name__}:BudgetEngine")

    root, rows = make_suite(tmp_path, n_per_bench=1)
    out = tmp_path / "run"
    final = run("budget", {}, root, out, limit=3, compact=True, log=lambda *_: None)
    assert final["effective_image_budget"] == {"max_image_pixels": 4096, "max_image_tokens": 7, "rows_with_image_tokens": 3}
    run("budget", {}, root, out, limit=2, log=lambda *_: None)
    env = json.loads((out / "environment.json").read_text())
    assert env["effective_image_budget"]["rows_with_image_tokens"] == 5


def test_images_are_capped_at_1_6_megapixels(tmp_path):
    from decision_index.vision import images
    from decision_index.vision.engines import HttpDecide, OpenAIVision, VllmVisionChat
    import inspect

    assert images.MAX_PIXELS == 1638400
    for cls in (VllmVisionChat, HttpDecide, OpenAIVision):
        assert inspect.signature(cls.__init__).parameters["max_pixels"].default == 1638400
    small = png_bytes(size=(640, 480))
    assert images.capped(small) == small
    big = png_bytes(size=(2000, 1500))
    out = Image.open(io.BytesIO(images.capped(big)))
    assert out.width * out.height <= 1638400 and abs(out.width / out.height - 4 / 3) < 0.01 and out.format == "PNG"
    assert images.load(images.data_url(big)) == big
    (tmp_path / "big.png").write_bytes(big)
    with pytest.raises(ValueError):
        images.encode("big.png", "path", tmp_path, max_pixels=1638400)


def test_decide_downscales_over_the_cap(tmp_path):
    from decision_index.vision.engines import HttpDecide

    (tmp_path / "big.png").write_bytes(png_bytes(size=(2000, 1500)))
    seen = {}

    def handler(request):
        import httpx

        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choice_index": 0, "probabilities": [0.6, 0.4]})

    e = HttpDecide(base_url="http://test", image_root=str(tmp_path))
    e.clients = mock_client(handler)
    q = {"answer": {"type": "choice", "instructions": "Pick", "criteria": {"A": "x", "B": "y"}}}
    _, raw = e({}, q, images=["big.png"])
    sent = Image.open(io.BytesIO(base64.b64decode(seen["body"]["state"][0]["image"].split(",", 1)[1])))
    assert sent.width * sent.height <= 1638400 and raw["image_pixels"] == [sent.width * sent.height]
