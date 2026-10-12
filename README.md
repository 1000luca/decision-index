# Decision Index

Reproduction kit for the **Decision Index**, a benchmark for *typed decision engines*: models that take a `state` and a set of typed `questions` (multiple-choice `choice` or yes/no `noul` primitives with explicit `criteria`) and return one answer per question with a probability for every supplied option. It lets anyone rebuild the frozen suite from public sources, run any engine through it with checkpoint/resume, score every benchmark with the leaderboard's own scorers, and compute the index, on a laptop or as one Hugging Face Job.

The live board is **Decision Index 0.3**, the default everywhere in this kit. Its main ranking is the **Full score**: 20% the public benchmarks in this kit, 50% private tests of the same skills and 30% private decision tasks from new domains. This kit runs and scores the public part, the **public index**: 37 benchmarks in five areas, each chance-corrected (0 = random guessing, 100 = perfect) and coverage-adjusted (unanswered = wrong). The private parts are run by the maintainers on submitted models (see [Submitting a model](#submitting-a-model-to-the-leaderboard)). Editions 0.2.1, 0.2 and 0.1 stay reproducible with `--edition 0.2.1`, `--edition 0.2` and `--edition 0.1` ([docs/edition-0.1.md](docs/edition-0.1.md)).

Not affiliated with TypeSafe AI.

## What changed in 0.3

On the board:

- The main ranking is the Full score: 20% public benchmarks, 50% private tests of the same skills and 30% private decision tasks from new domains. Before adding them up, each private part is put on the public scale using untuned stock models as the yardstick: a model that sits a given distance above the stock models' average on a private part gets the score that sits the same distance above their average on the public index. Most of the score comes from tests nobody can train on, so a public-only advantage counts for little.
- Models whose Full scores are less than 0.9 points apart share a rank: a gap that small is within what a fresh sample of questions could produce, 95% of the time.
- A Vision board ranks models that read images, on public vision benchmarks and private tests of the same skills, weighted equally. This kit rebuilds, runs and scores the public vision suite too: see [Vision](#vision).
- A Reasoning board is coming in 0.3.x, for models that can choose to think before answering the hard questions; thinking takes longer, so they get a board of their own.

In the public suite (what this kit runs):

- GSM8K is rebuilt: every wrong option is now the answer to a different GSM8K test problem of similar size, so the options alone no longer give the answer away ([issue #32](https://github.com/apolinario/decision-index/issues/32)). Method and checks: [docs/suite.md](docs/suite.md#edition-03).
- ForecastBench is retired from the index and from the run.
- WinoGrande now counts under Knowledge & Reasoning.
- Area weights stay at their 0.2.1 values: Knowledge & Reasoning 25.8%, Language Understanding 25.8%, Retrieval & Classification 20.0%, Tools & Automation 18.3%, Arts & Human Taste 10%. Gold ★ benchmarks still weigh 1.2 inside their area.

In the kit this means: 0.3 reads the 0.2.1 files unchanged plus one more, `gsm8k-rows.jsonl.gz` (2,638 requests), which `suite rebuild` builds from the pinned GSM8K test file. At read time the old GSM8K rows are replaced by the rebuilt ones and ForecastBench is skipped: 110,201 requests, 109,759 scoreable after the 442 exclusions, plus the same 30,419 for the seven benchmarks added in 0.2. The rebuilt GSM8K rows have new run ids, so resuming a complete 0.2.1 run directory under 0.3 runs only those 2,638 requests.

## What changed in 0.2.1

These are the site's own summary lines for the scoring:

- New area weights: Arts & Human Taste is fixed at 10%; the other four areas share the rest in proportion to the square root of their benchmark count: Tools & Automation 18.3%, Retrieval & Classification 20.0%, Knowledge & Reasoning 25.8%, Language Understanding 25.8%.
- Gold ★ benchmarks weigh 1.2 inside their area, the rest 1.0: MMLU-Pro, BBH, GPQA Diamond, HLE, ANLI, WinoGrande, HellaSwag, BANKING77, CLINC150, BRIGHT, BFCL, API-Bank, ForecastBench.
- SGD leaves the index until it is re-run with a fixed builder, and RouterBench leaves because its prompt gives away the best route. Both stay on the board.
- ACOS is now scored with F1 per review instead of all-or-nothing, and stays in the index.
- ToolRet and BRIGHT count only queries with at least one relevant candidate among the 32 (685 of 1,000 and 220 of 550); chance is recomputed on the same queries.
- RAGTruth chance is now always answering "hallucinated" (F1 0.518) instead of a fair coin (0.411).
- Home appliances drops 24 duplicate test rows and 48 rows identical to published dev rows (88 of 160 kept).

On the board, Surogate Rune 26B-A4B v3 and reflex 27B's full-coverage rerun replace their earlier runs, and Lavoir, JPT-0.8B and JPT-9B join.

In the kit this means: 0.2.1 reads the same files as 0.2 (same hashes, same `suite-0.2/`), so a complete 0.2 run is also a complete 0.2.1 run and `score --edition 0.2.1` rescores it. The answerable-query lists and the Home appliances cut ship in `decision_index/data/release-v2.1/` and are applied at read time, like the ACOS subset: 120,340 requests, 119,898 scoreable after the 442 exclusions, plus the same 30,419 for the seven benchmarks added in 0.2. `run` and `pipeline` under 0.2.1 skip the dropped rows.

## What changed in 0.2

These are the site's own summary lines:

- Scores are now chance-corrected: 0 means random guessing and 100 means perfect, so yes/no questions no longer give free points.
- The index now counts 40 benchmarks, up from 19: 7 brand-new ones (MMLU-Pro, BBH, When2Call, RAGTruth, HoVer, New Yorker caption matching, PhishNChips), plus 15 we already ran that now count, such as HLE, ANLI, WinoGrande, HellaSwag, BANKING77 and CLINC150.
- MMLU left the index: top models have nearly maxed it out, and MMLU-Pro takes its place. ARC-Easy, ARC-Challenge and SimpleBench are still shown but don't count.
- ToolRet and BRIGHT are run on a subset of their data that doesn't skew results but is faster and cheaper to run.
- 15 new models on the board, plus newer versions of 14 existing ones.

In the kit this means: the 0.2 suite is the 0.1 rows with ToolRet and BRIGHT cut to stratified subsets of 1,000 and 550 queries (121,057 requests after ACOS is cut to its 400-review subset at scoring time; 120,615 scoreable after the same 442 exclusions), plus 30,419 requests for the seven new benchmarks. The raw (not chance-corrected) index is still computed and reported as a secondary score.

## Quickstart (local)

The suite is not redistributed: several sources don't allow republishing (RAGTruth's Yelp and MS MARCO contexts, for one), so this repository ships the recipe and you build the files once. About 7 GB of downloads (the HoVer Wikipedia database alone is 2.2 GB) and 17 GB of working space; accept the [HLE](https://huggingface.co/datasets/cais/hle) terms on the Hub and be logged in first.

```sh
pip install -e ".[transformers,rebuild]"
export HF_HUB_DISABLE_XET=1

python -m decision_index suite rebuild --work work
python -m decision_index suite import \
    --rows work/artifacts/benchmark-suite/release-v2-rebuilt/selected-rows.jsonl.gz \
    --added-rows work/artifacts/benchmark-suite/release-v2-rebuilt/added-rows.jsonl.gz \
    --gsm8k-rows work/artifacts/benchmark-suite/release-v3-rebuilt/gsm8k-rows.jsonl.gz
```

`suite rebuild` rebuilds the 0.1 rows, cuts them to 0.2 with the published subset lists, builds the seven new benchmarks and then the 0.3 GSM8K rows; every source is pinned, the first two files come out byte-identical to the lab's, and the GSM8K rows match the board's on every run id, payload hash and answer. `suite import` stages them in `suite-0.3/` with the exclusion list and manifest from `hub/`, and refuses files whose hash does not match. Then:

```sh
python -m decision_index suite sample --n 100 --out sample-100.jsonl.gz
python -m decision_index run --engine transformers --model Qwen/Qwen2.5-0.5B-Instruct \
    --rows sample-100.jsonl.gz --out runs/qwen-0.5b-sample
python -m decision_index score --results runs/qwen-0.5b-sample/results.jsonl
```

The full suite in one command, scored at the end, entirely on your own machine (nothing is uploaded unless you pass `--upload`):

```sh
python -m decision_index pipeline --engine transformers --model Qwen/Qwen2.5-7B-Instruct --out runs/qwen-7b
```

For a model behind your own server that speaks the `/v1/systemone` wire format, use the `http` engine:

```sh
python -m decision_index pipeline --engine http --option base_url=http://127.0.0.1:8000 --option model=my-model --out runs/my-model
```

`run`/`pipeline` resume from an existing `results.jsonl`; rows whose status is `error` are retried, everything else is kept. A 0.2.1 run directory resumed under 0.3 runs only the 2,638 rebuilt GSM8K requests. A 0.2 `results.jsonl` scores under 0.2.1 as it is. A 0.1 `results.jsonl` can be rescored under 0.2 and 0.2.1 (their base rows are a subset of 0.1), but the seven new benchmarks still have to be run; resuming the same output directory runs only those.

Models whose median, mean or 80th-percentile latency is over 1,000 ms per request on one RTX PRO 6000 (measured by the maintainers, single process, one request at a time, the fastest path the model's code supports) are not added to the board: at that speed they are no longer Jev-like. The latency in your own `scores.json` is a guide, not that measurement.

## Quickstart (Hugging Face Jobs)

One job runs everything on a single RTX PRO 6000 (96 GB) and uploads `results.jsonl.gz`, `benchmark-summary.json`, `index.json`, `scores.json`, `environment.json` and `status.json` to a dataset repo you own. The job reads the suite from a Hub dataset, so first put your locally built copy in a **private** dataset under your account (keep it private: the sources' terms apply):

```sh
hf auth login
python scripts/prepare_hub_upload.py \
    --rows work/artifacts/benchmark-suite/release-v2-rebuilt/selected-rows.jsonl.gz \
    --added-rows work/artifacts/benchmark-suite/release-v2-rebuilt/added-rows.jsonl.gz \
    --gsm8k-rows work/artifacts/benchmark-suite/release-v3-rebuilt/gsm8k-rows.jsonl.gz --out hub-upload
hf repo create <you>/decision-index-suite-0.3 --repo-type dataset --private
hf upload <you>/decision-index-suite-0.3 hub-upload . --repo-type dataset

python -m decision_index hf-job --engine transformers --model Qwen/Qwen2.5-7B-Instruct \
    --suite-dataset <you>/decision-index-suite-0.3 \
    --results-repo <you>/decision-index-results --run-name qwen-7b
```

`hf-job` creates `<you>/decision-index-results` (private unless `--public`), uploads a snapshot of this package to `code/decision-index-src.tar.gz`, and submits `hf jobs run --flavor rtx-pro-6000 --timeout 72h pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime ...` with `HF_TOKEN` as a job secret. The job installs the snapshot, downloads the suite, runs the engine, scores, and uploads to `runs/<run-name>/`. `--git-url` installs from a git checkout instead, `--dry-run` prints the command, `--flavor`/`--image`/`--timeout` change the hardware, `--limit N` and `--compact` help while testing, `--edition 0.2.1`, `--edition 0.2` or `--edition 0.1` runs an earlier edition.

## Scoring and the index

`score` (and the end of `pipeline`) writes three files:

- `benchmark-summary.json`: the native metric of every benchmark on its complete, supported case groups (accuracy, macro-F1, nDCG@10, RouterBench quality, Brier, case-exact accuracy, tie-accepting accuracy; F1 on the hallucinated class for RAGTruth; PhishNChips on its `verdict` question only), with answered/unsupported/error counts.
- `index.json`: per benchmark `raw`, `skill`, `coverage` and chance level; the five areas; `index` (the Decision Index) and `raw_index`.
- `scores.json`: both combined, with completion flags. This is the file a submission points at.

`index` is the public index. The board's Full score also needs the private parts, which only the maintainers can run, so the kit does not compute it.

The 0.3 public index, exactly as the board computes it (`decision_index/scoring/index02.py`; panel, weights and chance levels in `decision_index/data/index-0.3.json`, and in `index-0.2.1.json` and `index-0.2.json` for the earlier editions):

1. **Per benchmark, coverage first.** Unanswered, unsupported, errored and abstained requests count as wrong: `raw = native score × answered / requests`. The benchmarks carried over from the 0.1 panel keep their track-level scoring (unanswered groups score zero inside the metric). ACOS is scored by F1 per review, averaged over reviews (case exact accuracy in 0.2).
2. **Chance correction.** `skill = clip((raw − chance) / (1 − chance), 0, 1)`. Chance is per track for GSM8K, per query (expected nDCG@10 of a random ranking, on the answerable queries) for ToolRet and BRIGHT, F1 of always answering "hallucinated" for RAGTruth (0.5177; a fair coin, 0.4113, in 0.2), per-review F1 of answering yes to every pair for ACOS (0.031), all fields of a case right by chance for BFCL, SATA-Bench and Home appliances (on the kept rows), F1 of random guessing for the other F1 benchmarks, and the mean of 1/options otherwise. In 0.2.1, **ForecastBench** entered against its baseline instead: `clip((0.25 − Brier) / 0.25) × coverage`, so always predicting 0.5 scored zero; it is retired in 0.3.
3. **Areas and index.** Inside an area, gold ★ benchmarks weigh 1.2 and the rest 1.0. The areas weigh Knowledge & Reasoning 25.8%, Language Understanding 25.8%, Retrieval & Classification 20.0%, Tools & Automation 18.3% and Arts & Human Taste 10% (in 0.2.1 these came from the square root of each area's benchmark count with Arts fixed at 10%; 0.3 keeps the same numbers, `index02.area_weights`); the index is `100 × the weighted mean of the five areas`, and `breadth_skill` uses the same weights in its geometric mean. `raw_index` is the same with `raw` in place of `skill`. MMLU, ARC-Easy, ARC-Challenge, SimpleBench, RouterBench and SGD are scored and shown but not counted. `index02.ranks` treats public-index scores within 0.25 points of the next one as tied, as the 0.2.1 board did; the 0.3 board's ties are set on the Full score (0.9 points).

0.2.1 differs from 0.3 in its GSM8K rows, in counting ForecastBench under Arts & Human Taste and in counting WinoGrande under Language Understanding. 0.2 differs from 0.2.1 in three places: RouterBench and SGD count, every area is the plain mean of its benchmarks, and the five areas weigh the same.

Parity: `tests/test_index03.py` checks the 0.3 public-index math against the published per-benchmark public results of 11 entrants, Jev included, and `score --edition 0.3` over a full lab run (Cloudflare clef, with its rebuilt-GSM8K results) reproduces its public index (61.71) and its GSM8K score. For 0.2.1, `score` over the lab's own results reproduces every one of the 67 entrants on the live 0.2.1 board, index, raw index, breadth, all five areas and all 38 per-benchmark values, and with `--edition 0.2` every one of the 64 entrants on the 0.2 board. `tests/test_index021.py` checks the 0.2.1 math against the published per-benchmark results of 13 entrants, from Rune 26B-A4B v3 (57.44) down to Lumma-Fev-0.1B (1.78), Jev (57.89) included; `tests/test_index02.py` does the same for 0.2.

## The suite

| File | Contents | sha256 |
|---|---|---|
| `selected-rows.jsonl.gz` | 124,971 requests: the 0.1 rows minus ToolRet/BRIGHT queries outside their subsets (excluded ToolRet/BRIGHT rows are carried over unchanged and never scored) | uncompressed `b2b56d6fb636837ca469e689087bdbf373dda8de7638aa2da6793e6eda0792d5` (the lab's gzip is `25aac5e890a54a3172c7a0c184b4cc8b9a43f10b6ee89bbad8da923be423c656`) |
| `added-rows.jsonl.gz` | 30,419 requests of the seven new benchmarks | uncompressed `7429f3c9cdddb772c1cfc42bb2a45e8516b0032152b746e6929f1c8b52f4ce89` |
| `gsm8k-rows.jsonl.gz` | 0.3 only: 2,638 rebuilt GSM8K requests that replace the GSM8K rows of `selected-rows.jsonl.gz` at read time | uncompressed `759334894858a37d6bd7bd07298eac8e55da236e31dad327ccd06a9979f4738a` |
| `excluded-questions.json` | 442 request ids dropped at scoring time for every engine (unchanged from 0.1) | `331df32d4b719c7db43214d0e5d85859d39c3b2eb7d0b3812214cce150155e81` |
| `manifest.json` | counts, subset rules, sources and licences (`hub/0.3/manifest.json`, `hub/0.2.1/manifest.json`, `hub/0.2/manifest.json`) | (any) |

The ToolRet/BRIGHT subset lists and the ACOS 400-review subset ship with the package (`decision_index/data/release-v2/`), and so do the 0.2.1 answerable-query lists and Home appliances cut (`decision_index/data/release-v2.1/`); their sha256 are pinned in `editions.py` and checked by `suite verify`. Rebuilt gzip files differ from the lab's in their header, so the kit verifies the uncompressed hash. Sources, revisions, sampling and licences per benchmark: [docs/suite.md](docs/suite.md). Row and result formats: [docs/format.md](docs/format.md).

## Benchmarks (0.2)

In 0.3, ForecastBench (48) is retired, WinoGrande (28) counts under Knowledge & Reasoning and GSM8K (30) uses the rebuilt rows (same 2,638 requests and chance levels); everything else in this table holds for 0.3 as it does for 0.2.1.

| # | Benchmark | Area | Metric | Chance | Requests | |
|---|---|---|---|---|---:|---|
| 1 | BFCL | Tools & Automation | case exact accuracy | 0.2592 | 1,694 | |
| 2 | ToolRet | Tools & Automation | nDCG@10 | per query (0.0937) | 1,000 | subset |
| 3 | API-Bank | Tools & Automation | accuracy | 0.0189 | 508 | |
| 4 | BANKING77 | Retrieval & Classification | macro-F1 | 0.0127 | 3,080 | |
| 5 | CLINC150+OOS | Retrieval & Classification | macro-F1 | 0.006 | 5,500 | |
| 6 | RouterBench | Tools & Automation | selected quality | per track (0.5735) | 10,000 | |
| 9 | Home appliances | Tools & Automation | case exact accuracy | 0 | 160 | |
| 10 | SGD | Retrieval & Classification | macro-F1 | 0.399 | 2,500 | |
| 11 | ContractNLI | Language Understanding | conservative macro-F1 | 0.3085 | 123 | |
| 12 | ANLI | Language Understanding | macro-F1 | 0.3324 | 3,200 | |
| 20 | BPoMP | Arts & Human Taste | accuracy | 0.5 | 5,000 | |
| 21 | Humicroedit | Arts & Human Taste | accuracy | 0.5 | 2,628 | |
| 22 | POP909 | Arts & Human Taste | song-macro accuracy | 0.0078 | 2,000 | |
| 23 | cfcolor | Arts & Human Taste | user-macro accuracy | 0.5 | 5,000 | |
| 24 | MMLU | shown, not counted | accuracy | 0.25 | 14,033 | |
| 25 | GPQA Diamond | Knowledge & Reasoning | accuracy | 0.25 | 196 | |
| 26 | ARC-Easy | shown, not counted | accuracy | 0.2502 | 2,376 | |
| 27 | ARC-Challenge | shown, not counted | accuracy | 0.2502 | 1,172 | |
| 28 | WinoGrande | Language Understanding | accuracy | 0.5 | 1,267 | |
| 29 | HellaSwag | Language Understanding | accuracy | 0.25 | 10,042 | |
| 30 | GSM8K | Knowledge & Reasoning | accuracy | per track (0.25, 0.1) | 2,638 | |
| 31 | ChessBench | Knowledge & Reasoning | best-move accuracy | 0.0819 | 5,000 | |
| 32 | MuSR | Knowledge & Reasoning | accuracy | 0.371 | 752 | |
| 33 | SATA-Bench | Knowledge & Reasoning | case exact accuracy | 0.0131 | 1,650 | |
| 34 | SimpleBench | shown, not counted | accuracy | 0.1667 | 10 | |
| 36 | BRIGHT | Retrieval & Classification | nDCG@10 | per query (0.0448) | 550 | subset |
| 37 | Amazon ESCI | Retrieval & Classification | conservative macro-F1 | 0.2027 | 5,000 | |
| 38 | ACOS | Language Understanding | case exact accuracy | 0 | 1,565 | subset |
| 39 | FinEntity | Language Understanding | macro-F1 | 0.3201 | 979 | |
| 40 | iSarcasmEval | Language Understanding | sarcasm F1, track A English | 0.2227 | 4,600 | |
| 41 | VAST | Language Understanding | conservative macro-F1 | 0.3333 | 3,006 | |
| 42 | NLI4CT | Language Understanding | macro-F1 | 0.486 | 5,500 | |
| 43 | CRUXEval | Knowledge & Reasoning | accuracy | 0.3697 | 570 | |
| 44 | CLadder | Knowledge & Reasoning | accuracy | 0.5 | 5,000 | |
| 45 | HLE | Knowledge & Reasoning | accuracy | 0.1641 | 501 | |
| 48 | ForecastBench | Arts & Human Taste | Brier, against the 0.25 baseline | 0.25 Brier | 10,139 | |
| 50 | Habermas Machine | Arts & Human Taste | group-preference accuracy | 0.311 | 1,676 | |
| 56 | PhishNChips | Retrieval & Classification | accuracy (verdict) | 0.5 | 2,000 | new |
| 57 | MMLU-Pro | Knowledge & Reasoning | accuracy | 0.1109 | 12,032 | new |
| 58 | BBH | Knowledge & Reasoning | accuracy | 0.3101 | 5,507 | new |
| 59 | RAGTruth | Language Understanding | F1 on hallucinated class | 0.4113 | 2,700 | new |
| 61 | HoVer | Retrieval & Classification | accuracy | 0.5 | 4,000 | new |
| 62 | When2Call | Tools & Automation | accuracy | 0.25 | 3,652 | new |
| 64 | New Yorker | Arts & Human Taste | accuracy | 0.2 | 528 | new |

Requests are after the 442 exclusions. In 0.2.1, RouterBench and SGD are shown but not counted; ToolRet, BRIGHT and Home appliances keep 685, 220 and 88 requests (chance 0.1341, 0.116 and 0); ACOS is scored by per-review F1 (chance 0.031); RAGTruth chance is 0.5177; and MMLU-Pro, BBH, GPQA Diamond, HLE, ANLI, WinoGrande, HellaSwag, BANKING77, CLINC150, BRIGHT, BFCL, API-Bank and ForecastBench are gold ★. The six interactive environments of the original panel (MiniWoB++, ScienceWorld, Boxoban, RTFM, Hanabi, Codenames) are still unrun and stay out.

## Rules

Encoded in the runner and engines, not left to the reader:

- **No truncation.** An engine that cannot fit a request raises `Unsupported`; the row is recorded as `unsupported` and counts as wrong. Nothing is cut to fit.
- **No option filtering.** Every option in `criteria` gets a probability or `validate` rejects the response.
- **No prompt tuning.** The reference engines use one fixed rendering for every benchmark.
- **Unanswered = wrong.** Unsupported, errored, abstained and pending requests score zero before chance correction. The per-benchmark report shows the native metric on answered cases separately so you can see the gap.
- **Linked cases stay whole.** Multi-request cases (ToolRet/BRIGHT chunks, RouterBench tracks, ACOS reviews) only count when every linked request succeeded.
- **Exclusions apply to everyone,** at scoring time; the frozen files are untouched.

## Engines

Subclass `decision_index.engines.Engine`, implement `__call__(state, questions) -> (response, raw)`, raise `Unsupported(reason)` for declared capacity limits, and pass `--engine my_module:MyEngine`. The kit ships two reference engines (details in [docs/engines.md](docs/engines.md)):

- `http`: a client for any `POST /v1/systemone` server (`--option base_url=...`, `--option model=...`, extra request fields via `--option extra='{...}'`, token from `DECISION_INDEX_API_KEY`).
- `transformers`: a stock causal LM scoring every option in one forward pass over a shared prompt.

The board's entrants were run with their authors' own inference code. Entrants whose authors publish a `/v1/systemone` server can be driven by `http` once that server is up; the others need their own harness. [docs/engines.md](docs/engines.md) lists which is which.

## Submitting a model to the leaderboard

Submit the **public 0.3 suite**; the maintainers run the private parts of the Full score themselves.

1. Run the full public 0.3 suite with this kit (`pipeline` or `hf-job`; a complete 0.2.1 run directory resumed under 0.3 only needs the 2,638 rebuilt GSM8K requests) and upload the run directory to a Hub dataset (`--upload <you>/<repo>`; the job does this for you).
2. Open a pull request adding a line to `submissions/README.md` (create it if needed) with the model name, the model repo on the Hub, the results dataset link (`runs/<name>/scores.json` must be present), the engine/commit used, hardware, and the exact inference settings you ran with. Runs must be complete (`scores.json` says `"complete": true`) and untouched: the results file is re-scored on review.
3. Mention any declared capacity limits; they show in `environment.json` and in the unsupported counts and are fine, as long as nothing was truncated.

The public suite is 20% of the Full score. After review, the maintainers run the submitted model themselves on the two private parts: private tests of the same skills as the public benchmarks (50%) and private decision tasks from new domains (30%). They run it exactly as you did, so the model has to be runnable by them: weights on the Hub (public, or access granted to the maintainers on request), the inference code you used (or a `/v1/systemone` server they can start), and its settings. The private sets are never published or shared, with submitters or anyone else, and they appear on the board only as each model's aggregate scores.

The maintainers measure latency themselves, single-process on one RTX PRO 6000 using the fastest path the model's code supports, on a private held-out sample. That same sample is used to validate submitted runs by comparing answers. Models whose median, mean or 80th-percentile latency there is over 1,000 ms per request are not added to the board: at that speed they are no longer Jev-like.

## Vision

The Vision board's Full score is 50% the **public vision index** and 50% private tests of the same skills. The kit covers the public half: 11 benchmarks, 12,210 multiple-choice rows over one to four images each (CV-Bench, BLINK, RealWorldQA, CharXiv, InfographicVQA, Mind2Web, Winoground, KIE on CORD and FUNSD, Hateful Memes at weight 1; R-Bench-M at 0.5 and MMMU-Pro vision at 0.25).

```bash
pip install -e '.[vision]'
decision-index suite rebuild --vision --work work        # pinned sources, about 5.7 GB of downloads
decision-index vision verify --dir work/vision
decision-index vision pipeline --dir work/vision --engine vllm-vl --model Qwen/Qwen3-VL-8B-Instruct --out runs/vision-qwen3-vl-8b
decision-index vision hf-job --engine vllm-vl --model Qwen/Qwen3-VL-8B-Instruct --results-repo <you>/decision-index-results
```

The suite is never distributed: the images and questions are fetched from the pinned upstream revisions at rebuild time and checked against the board's public rows (run ids and payload hashes). Engines: `vllm-vl` (the board's reference vLLM letter readout), `decide` (`/v1/decide` or `/v1/decisions` servers that take images) and `openai-vlm` (any OpenAI-compatible chat endpoint). The index is each benchmark's chance-corrected skill, clipped to [0, 1], in a weighted mean; rescoring the board's run files reproduces its published public numbers exactly.

To submit, run the whole public vision suite and open a pull request as for the text board, marked **vision**. Models run at native resolution capped at 1,638,400 pixels (1.6 MP) per image: lift any author-set cap below that, never go above it. Architectures with a fixed image budget use their largest setting at or below 1.6 MP (Gemma 4: 560 soft tokens, about 1.3 MP). Everything stays subject to the latency rule (median, mean and p80 at most 1,000 ms per request on one RTX PRO 6000). All three engines apply the cap by default (`max_pixels`), and every run records the effective maximum pixels and image tokens in `environment.json`. Details, sources, licences and the request format: [docs/vision.md](docs/vision.md).

## Layout

```
decision_index/
  editions.py           0.1, 0.2, 0.2.1 and 0.3: hashes, counts, subsets, retired and replaced benchmarks
  cli.py                suite | run | score | pipeline | hf-job | vision
  runner.py             checkpoint/resume loop, results.jsonl rows
  pipeline.py           score + index + upload
  hf_job.py             one-job submitter (rtx-pro-6000)
  engines/              Engine base, http, transformers, random
  vision/               public vision suite: pinned builders, rows, engines (vllm-vl, decide, openai-vlm), runner, scorer
  scoring/              metrics, per-benchmark report, 0.1 index, 0.2, 0.2.1 and 0.3 public index (index02), new-benchmark scorer (added)
  suite/                download, verify, sample, build/ (0.1 normalizers, 0.2 cut, new benchmarks, 0.3 GSM8K)
  data/                 panels, chance levels, benchmark catalog, release-v2 and release-v2.1 subset lists, vision-0.3.json pins
hub/                    exclusions and manifests to stage with the rows (hub/0.2, hub/0.2.1, hub/0.3)
scripts/prepare_hub_upload.py
docs/                   suite.md, format.md, engines.md, edition-0.1.md, vision.md
tests/                  metrics, index math (0.1, 0.2, 0.2.1 and 0.3), editions, report, runner, vision
```

## Licence notes

Code in this repository is MIT. The suite contains text from 43 upstream datasets plus one purpose-built benchmark, each under its own terms (listed per benchmark in [docs/suite.md](docs/suite.md)). Several are research-only or non-commercial (ANLI CC BY-NC 4.0; RAGTruth's MS MARCO contexts), some forbid redistribution (RAGTruth's Yelp contexts), some are share-alike (SGD, HoVer), GPQA asks that its questions not be posted in plain text, BBH carries the BIG-bench canary, and HLE is gated. Treat the suite as evaluation data under those terms; do not train on it and do not republish it.

The vision suite is built on your machine from its sources and is not in this repository. RealWorldQA is CC BY-ND 4.0, FUNSD is non-commercial, Winoground is research-only and gated, and the Hateful Memes licence forbids redistributing derivatives; the per-benchmark terms are in [docs/vision.md](docs/vision.md). Keep a rebuilt copy private.

Not affiliated with TypeSafe AI.
