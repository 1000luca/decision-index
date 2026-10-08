import argparse
import json
import os
import sys
from pathlib import Path

from decision_index import constants as C
from decision_index import editions


def parse_value(v):
    try:
        return json.loads(v)
    except ValueError:
        return v


def engine_options(args):
    opts = {}
    if getattr(args, "model", None):
        opts["model"] = args.model
    for item in args.option or []:
        k, _, v = item.partition("=")
        opts[k] = parse_value(v)
    return opts


def suite_dir(args, attr="suite_dir"):
    return Path(getattr(args, attr, None) or editions.get(args.edition)["suite_dir"])


def ensure_suite(args):
    from decision_index.suite.download import download
    from decision_index.suite.io import Suite

    directory = suite_dir(args)
    if not all((directory / name).exists() for name in editions.files(args.edition) if name != editions.MANIFEST_FILE):
        print(json.dumps(download(directory, dataset=args.suite_dataset, edition=args.edition)), flush=True)
    return Suite(directory, args.edition)


def cmd_suite(args):
    from decision_index.suite.io import Suite

    if args.suite_command == "download":
        from decision_index.suite.download import download

        print(json.dumps(download(suite_dir(args, "dir"), dataset=args.dataset, revision=args.revision, verify=not args.no_verify, edition=args.edition), indent=2))
    elif args.suite_command == "verify":
        print(json.dumps(Suite(suite_dir(args, "dir"), args.edition).verify(strict=False), indent=2))
    elif args.suite_command == "import":
        from decision_index.suite.download import from_local

        print(json.dumps(from_local(suite_dir(args, "dir"), args.rows, args.exclusions, args.manifest, verify=not args.no_verify, edition=args.edition, added=args.added_rows, gsm8k=args.gsm8k_rows), indent=2))
    elif args.suite_command == "sample":
        from decision_index.suite.sample import stratified_sample

        print(json.dumps(stratified_sample(Suite(suite_dir(args, "dir"), args.edition), args.n, args.out, seed=args.seed, complete_groups=not args.single_requests), indent=2))
    elif args.suite_command == "rebuild" and args.vision:
        from decision_index.vision import build, rows

        root = Path(args.work) / "vision"
        report = build.build(root, sets=args.sets, font=args.font, keep_downloads=args.keep_downloads)
        rows.write_rows(root)
        print(json.dumps({"dir": str(root), "build": report, "verify": rows.verify(root)}, indent=2))
    elif args.suite_command == "rebuild":
        from decision_index.suite.build import rebuild

        print(json.dumps(rebuild.main(args.work, only=args.only, skip_download=args.skip_download, skip_normalize=args.skip_normalize, compare=args.compare, edition=args.edition), indent=2))
    else:
        raise SystemExit("unknown suite command")


def cmd_run(args):
    from decision_index.runner import run
    from decision_index.suite.io import sha256_file

    keep = None
    if args.rows:
        rows = Path(args.rows)
        corpus = None
    else:
        suite = ensure_suite(args)
        rows = suite.row_paths
        keep = suite.in_edition
        corpus = suite.verify(strict=not args.no_verify)["sha256"]
    out = Path(args.out or ("runs/" + args.engine))
    print(json.dumps(run(args.engine, engine_options(args), rows, out, limit=args.limit, compact=args.compact, resume=not args.fresh, corpus_sha256=corpus, keep=keep)))


def cmd_score(args):
    from decision_index.pipeline import score_run
    from decision_index.suite.io import Suite

    suite = Suite(suite_dir(args), args.edition)
    results = Path(args.results)
    out = Path(args.out or results.parent)
    scores = score_run(suite, results, args.engine or results.parent.name, out, reference_results=args.reference)
    print(json.dumps(brief(scores, out), indent=2))


def brief(scores, out=None):
    x = {"edition": scores.get("edition", "0.1"), "decision_index": scores["decision_index"], "scores": scores["scores"], "areas": [{k: a[k] for k in ("id", "skill", "raw", "coverage", "n")} for a in scores["areas"]], "completed": scores["completed"], "complete": scores["complete"]}
    if out is not None:
        x["out"] = str(out)
    return x


def cmd_pipeline(args):
    from decision_index.pipeline import score_run, upload_run
    from decision_index.runner import run

    suite = ensure_suite(args)
    corpus = suite.verify(strict=not args.no_verify)["sha256"]
    out = Path(args.out or ("runs/" + args.engine))
    rows = Path(args.rows) if args.rows else suite.row_paths
    run(args.engine, engine_options(args), rows, out, limit=args.limit, compact=args.compact, resume=not args.fresh, corpus_sha256=corpus, keep=None if args.rows else suite.in_edition)
    scores = score_run(suite, out / "results.jsonl", args.engine, out)
    print(json.dumps(brief(scores), indent=2))
    if args.upload:
        print(upload_run(args.upload, out, path_in_repo=args.upload_path, private=not args.public))


def cmd_hf_job(args):
    from decision_index.hf_job import submit

    engine_args = []
    if args.model:
        engine_args += ["--model", args.model]
    for o in args.option or []:
        engine_args += ["--option", o]
    extra = []
    if args.limit:
        extra += ["--limit", str(args.limit)]
    if args.compact:
        extra += ["--compact"]
    extra += ["--edition", editions.get(args.edition)["id"]]
    print(json.dumps(submit(args.results_repo, args.engine, engine_args, args.suite_dataset or editions.get(args.edition)["dataset"], args.run_name or args.engine, extra, flavor=args.flavor, image=args.image, timeout=args.timeout, git_url=args.git_url, namespace=args.namespace, private=not args.public, dry_run=args.dry_run), indent=2))


def cmd_vision(args):
    from decision_index.vision import rows
    from decision_index.vision.score import score_run

    root = Path(args.dir)
    if args.vision_command == "verify":
        print(json.dumps(rows.verify(root), indent=2))
        return
    if args.vision_command == "score":
        print(json.dumps(score_run(root / rows.ROWS_FILE, args.results, args.out), indent=2))
        return
    if args.vision_command == "hf-job":
        from decision_index.hf_job import submit_vision

        print(json.dumps(submit_vision(args.results_repo, args.engine, engine_options(args), args.run_name or args.engine, suite_dataset=args.suite_dataset,
                                       workers=args.workers, chunk=args.chunk, limit=args.limit, flavor=args.flavor, image=args.image, timeout=args.timeout,
                                       git_url=args.git_url, namespace=args.namespace, private=not args.public, dry_run=args.dry_run), indent=2))
        return
    from decision_index.vision.run import run

    if args.vision_command in ("run", "pipeline"):
        check = rows.verify(root)
        if not check["ok"] and not args.no_verify:
            raise SystemExit("the vision suite in " + str(root) + " does not match the pinned hashes (vision verify shows which part); pass --no-verify to run anyway")
    out = Path(args.out or ("runs/vision-" + args.engine))
    print(json.dumps(run(args.engine, engine_options(args), root, out, workers=args.workers, chunk=args.chunk, limit=args.limit, compact=args.compact, resume=not args.fresh)))
    if args.vision_command == "pipeline":
        print(json.dumps(score_run(root / rows.ROWS_FILE, out / "results.jsonl", out), indent=2))
        if args.upload:
            from decision_index.pipeline import upload_run

            print(upload_run(args.upload, out, path_in_repo=args.upload_path, private=not args.public))


def add_engine_args(p):
    p.add_argument("--engine", required=True, help="engine name (http, transformers, random) or module:Class")
    p.add_argument("--model", help="model id for the transformers engine")
    p.add_argument("--option", action="append", help="engine option key=value (JSON values accepted), repeatable")


def add_edition(p):
    p.add_argument("--edition", choices=sorted(editions.EDITIONS), help=f"suite edition (default: read from --suite-dir/--dir when given, else {editions.DEFAULT})")


def add_suite_args(p):
    add_edition(p)
    p.add_argument("--suite-dir", help="suite directory (default: suite-0.3 for 0.3, suite-0.2 for 0.2 and 0.2.1, suite for 0.1)")
    p.add_argument("--suite-dataset", help="Hub dataset holding your private copy of the suite")
    p.add_argument("--no-verify", action="store_true")


def build_parser():
    ap = argparse.ArgumentParser(prog="decision-index", description="Decision Index reproduction kit")
    sub = ap.add_subparsers(dest="command", required=True)

    s = sub.add_parser("suite", help="obtain, verify, sample or rebuild the frozen suite")
    ss = s.add_subparsers(dest="suite_command", required=True)
    d = ss.add_parser("download")
    add_edition(d)
    d.add_argument("--dir")
    d.add_argument("--dataset")
    d.add_argument("--revision")
    d.add_argument("--no-verify", action="store_true")
    v = ss.add_parser("verify")
    add_edition(v)
    v.add_argument("--dir")
    i = ss.add_parser("import")
    add_edition(i)
    i.add_argument("--dir")
    i.add_argument("--rows", required=True)
    i.add_argument("--added-rows", help="the new-benchmark rows file (0.2, 0.2.1 and 0.3)")
    i.add_argument("--gsm8k-rows", help="the rebuilt GSM8K rows file (0.3)")
    i.add_argument("--exclusions", help="default: the copy in hub/")
    i.add_argument("--manifest", help="default: the copy in hub/")
    i.add_argument("--no-verify", action="store_true")
    m = ss.add_parser("sample")
    add_edition(m)
    m.add_argument("--dir")
    m.add_argument("--n", type=int, default=100)
    m.add_argument("--out", default="sample-rows.jsonl.gz")
    m.add_argument("--seed", type=int, default=C.FREEZE_SEED)
    m.add_argument("--single-requests", action="store_true")
    r = ss.add_parser("rebuild")
    add_edition(r)
    r.add_argument("--work", default="work")
    r.add_argument("--only", type=int, nargs="*")
    r.add_argument("--skip-download", action="store_true")
    r.add_argument("--skip-normalize", action="store_true")
    r.add_argument("--compare", help="path to a reference selected-rows.jsonl(.gz) to compare run ids and payload hashes")
    r.add_argument("--vision", action="store_true", help="rebuild the public vision suite into <work>/vision instead of the text suite")
    r.add_argument("--sets", nargs="*", help="with --vision: only these sets (full_cvbench, full_blink, ...)")
    r.add_argument("--font", help="with --vision: TrueType font for the Mind2Web box labels (default: Arial Bold where installed)")
    r.add_argument("--keep-downloads", action="store_true")
    s.set_defaults(func=cmd_suite)

    p = sub.add_parser("run", help="run an engine over the suite with checkpoint/resume")
    add_engine_args(p)
    add_suite_args(p)
    p.add_argument("--rows", help="run over this rows file instead of the full suite")
    p.add_argument("--out")
    p.add_argument("--limit", type=int)
    p.add_argument("--compact", action="store_true", help="omit payload and raw_output from results rows")
    p.add_argument("--fresh", action="store_true", help="ignore an existing results.jsonl")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("score", help="score a results.jsonl and compute the Decision Index")
    add_edition(p)
    p.add_argument("--results", required=True)
    p.add_argument("--suite-dir")
    p.add_argument("--engine")
    p.add_argument("--out")
    p.add_argument("--reference", help="optional reference results.jsonl scored on the same complete cases")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("pipeline", help="suite -> run -> score -> index (-> upload) in one go")
    add_engine_args(p)
    add_suite_args(p)
    p.add_argument("--rows")
    p.add_argument("--out")
    p.add_argument("--limit", type=int)
    p.add_argument("--compact", action="store_true")
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--upload", help="Hub dataset repo to receive results and scores")
    p.add_argument("--upload-path", default="")
    p.add_argument("--public", action="store_true")
    p.set_defaults(func=cmd_pipeline)

    v = sub.add_parser("vision", help="run and score the public vision suite")
    vs = v.add_subparsers(dest="vision_command", required=True)
    for name in ("verify", "run", "score", "pipeline", "hf-job"):
        q = vs.add_parser(name)
        q.add_argument("--dir", default="work/vision", help="vision suite directory written by suite rebuild --vision")
        if name == "score":
            q.add_argument("--results", required=True)
            q.add_argument("--out")
        if name in ("run", "pipeline", "hf-job"):
            add_engine_args(q)
            q.add_argument("--workers", type=int, default=8, help="concurrent requests for engines without a batch call")
            q.add_argument("--chunk", type=int, default=256, help="rows per batch (vllm-vl) or per checkpoint")
            q.add_argument("--limit", type=int)
        if name in ("run", "pipeline"):
            q.add_argument("--out")
            q.add_argument("--compact", action="store_true")
            q.add_argument("--fresh", action="store_true")
            q.add_argument("--no-verify", action="store_true")
        if name == "pipeline":
            q.add_argument("--upload", help="Hub dataset repo to receive results and scores")
            q.add_argument("--upload-path", default="")
        if name in ("pipeline", "hf-job"):
            q.add_argument("--public", action="store_true")
        if name == "hf-job":
            q.add_argument("--results-repo", required=True)
            q.add_argument("--suite-dataset", help="private Hub dataset holding your rebuilt vision suite (default: rebuild inside the job)")
            q.add_argument("--run-name")
            q.add_argument("--flavor", default="rtx-pro-6000")
            q.add_argument("--image", default="pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime")
            q.add_argument("--timeout", default="24h")
            q.add_argument("--git-url")
            q.add_argument("--namespace")
            q.add_argument("--dry-run", action="store_true")
    v.set_defaults(func=cmd_vision)

    p = sub.add_parser("hf-job", help="submit one Hugging Face Job running the whole pipeline")
    add_engine_args(p)
    add_edition(p)
    p.add_argument("--results-repo", required=True, help="Hub dataset repo that receives code snapshot, results and scores")
    p.add_argument("--suite-dataset", help="Hub dataset holding your private copy of the suite")
    p.add_argument("--run-name")
    p.add_argument("--flavor", default="rtx-pro-6000")
    p.add_argument("--image", default="pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime")
    p.add_argument("--timeout", default="72h")
    p.add_argument("--git-url", help="install the kit from this git URL instead of uploading a source snapshot")
    p.add_argument("--namespace")
    p.add_argument("--limit", type=int)
    p.add_argument("--compact", action="store_true")
    p.add_argument("--public", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_hf_job)
    return ap


def resolve_edition(args):
    from decision_index.suite.io import detect_edition

    if not hasattr(args, "edition") or args.edition:
        return
    directory = getattr(args, "suite_dir", None) or getattr(args, "dir", None)
    if directory and Path(directory).exists() and getattr(args, "suite_command", None) not in ("download", "import"):
        found = detect_edition(directory)
        args.edition = editions.DEFAULT if editions.compatible(editions.DEFAULT, found) else found
    else:
        args.edition = editions.DEFAULT


def main(argv=None):
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    args = build_parser().parse_args(argv)
    resolve_edition(args)
    args.func(args)


if __name__ == "__main__":
    main()
