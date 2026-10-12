# Vision (0.3)

The Vision board ranks models that read images. Its **Full score** is 50% the public vision index, which this kit rebuilds, runs and scores, and 50% private tests of the same skills, which the maintainers run themselves on submitted models.

## The public suite

Eleven public benchmarks, 12,210 rows. Every row is one multiple-choice question over one to four images, answered with a letter.

| Benchmark | Rows | Weight | Source (Hub dataset @ revision unless noted) | Split | Conversion | Licence |
|---|---:|---:|---|---|---|---|
| CV-Bench | 2,038 | 1 | nyu-visionx/CV-Bench @bc284db | test | native MC; Count, Relation, Distance (Depth left out: a blind model gets 0.80 on its 2-way items) | Apache-2.0 card; images ADE20K/COCO/Omni3D |
| BLINK | 961 | 1 | BLINK-Benchmark/BLINK @a3666eb | val | native MC over 1 to 4 images; Counting, Relative_Depth, Spatial_Relation, Multi-view_Reasoning, Semantic/Visual/Functional_Correspondence (Object_Localization left out) | Apache-2.0 card |
| RealWorldQA | 589 | 1 | xai-org/RealworldQA @17e7f75 | test | lettered options parsed from the prompt; yes/no answers become Yes/No; free-form answers dropped | CC BY-ND 4.0 |
| CharXiv | 532 | 1 | princeton-nlp/CharXiv @f441eb6 | validation, reasoning questions | 4-way MC; distractors from the same figure's descriptive answers, numeric fallbacks perturbed | CC BY-SA 4.0 questions; figures under their authors' licences |
| InfographicVQA | 244 | 1 | lmms-lab-encoder/DocVQA @539088e (InfographicVQA config) | validation | one question per infographic; 4-way MC with the answers to other questions on the same infographic as distractors | original RRC terms; mirror card Apache-2.0 |
| Mind2Web | 973 | 1 | osunlp/Multimodal-Mind2Web @1b4c6a8 | test_website | set-of-marks: a 1080px viewport around the target element, the target plus 3 real page elements drawn as boxes A to D | OpenRAIL card; Mind2Web data CC BY 4.0; screenshots third-party |
| Winoground | 800 | 1 | facebook/winoground @b400e17 (gated) | test | caption choice for each image of each pair | research-only licence agreement; images Getty Images |
| KIE (CORD+FUNSD) | 2,465 | 1 | naver-clova-ix/cord-v2 @7f0115a (validation + test); FUNSD official zip (sha256 c3173564…) testing_data | | receipt field values (MC from the other amounts on the receipt) and field-presence Yes/No questions; presence questions balanced to equal Yes and No per field | CORD CC BY 4.0; FUNSD non-commercial research and education only |
| Moderation (Hateful Memes) | 1,213 | 1 | neuralcatcher/hateful_memes @d201c48 (mirror of the Hateful Memes release) | dev_seen + test_seen | "Is this meme hateful?" Yes/No | Hateful Memes licence: research purpose, no redistribution of derivatives |
| R-Bench-M | 665 | 0.5 | R-Bench/R-Bench @c0e92a6 | rbench-m_en test | native MC, A to F | Apache-2.0 card |
| MMMU-Pro vision | 1,730 | 0.25 | MMMU/MMMU_Pro @563f3e8 | vision test | question and up to ten options are inside the image | Apache-2.0 card |

Full revisions, every source file's sha256, and the expected hashes of the built sets and rows are in `decision_index/data/vision-0.3.json`. The selection is deterministic: the conversions use fixed seeds and fixed orders, so there is no sampling step to reproduce.

R-Bench-M and MMMU-Pro vision are headroom sets with low weight. For them the reference engine reads the letter after the assistant has started its reply with `The answer is (`.

## Rebuild

The suite is not distributed. Several sources forbid it (RealWorldQA is no-derivatives, FUNSD non-commercial, Winoground research-only, Hateful Memes forbids redistributing derivatives), so you build it yourself from the pinned sources:

```bash
pip install -e '.[vision]'
hf auth login                        # Winoground is gated: accept its terms on the Hub first
decision-index suite rebuild --vision --work work
decision-index vision verify --dir work/vision
```

This downloads about 5.7 GB, one set at a time, deleting each set's downloads once it is built (`--keep-downloads` keeps them), and writes about 4.4 GB to `work/vision/`:

- `full/<set>/items.jsonl`, `full/<set>/img/` and `full/<set>/source.json` for the 12 source sets (KIE is CORD plus FUNSD);
- `vision-rows.jsonl`, the 12,210 public rows the engines run.

`--sets full_blink full_charxiv` rebuilds only some sets. Every downloaded file is checked against its pinned sha256 before it is used. The report, and `vision verify` afterwards, compare for each set the item count, the items file's sha256 and a digest of every image file, and for each benchmark the row count, the sorted run ids and the payload hashes (question, options and image sha256). All of these match the board's public rows.

Two things can make image bytes differ while the rows, questions and run ids still match:

- **Pillow.** Most images are the source bytes, unchanged. The few over 4096 px on the longer side (1 in BLINK, 56 in InfographicVQA, 2 in CORD) are resized and re-encoded, and Mind2Web's viewport crops and marked screenshots are JPEG-encoded. The pinned digests reproduce with Pillow 9.5.0 and 12.3.0.
- **The Mind2Web label font.** The letters on the marked boxes are drawn in Arial Bold (`/System/Library/Fonts/Supplemental/Arial Bold.ttf` on macOS). Pass `--font /path/to/Arial Bold.ttf` or set `DECISION_INDEX_FONT` elsewhere. Without it, Pillow's default font is used and `vision verify` flags `full_mind2web` images and the Mind2Web payloads.

For a Hugging Face Job, the simplest way to keep the bytes identical is to build once where they verify, push `work/vision` to a **private** dataset (`hf upload <you>/vision-suite work/vision . --repo-type dataset --private`) and pass `--suite-dataset <you>/vision-suite`.

## Request format

Each line of `vision-rows.jsonl`:

| Key | Meaning |
|---|---|
| `id`, `benchmark`, `set`, `subtask` | source item id (also the run id), board benchmark, source set, subtask |
| `images` | image paths relative to the suite directory, in the order the question refers to them |
| `state`, `questions` | `state` is `{}`; `questions` is `{"answer": {"type": "choice", "instructions": str, "criteria": {"A": option, "B": option, ...}}}` |
| `prompt` | `options_in_image` (true for MMMU-Pro vision, where `criteria` maps each letter to itself) and `readout` (`"The answer is ("` for R-Bench-M and MMMU-Pro vision, else null) |
| `expected` | `{"answer": letter}` |
| `_evaluation` | `run_id`, `benchmark`, `set`, `payload_sha256` (sha256 of `{"state", "questions", "images": [image sha256], "prompt"}`), `image_sha256` |

An engine is called as `engine(state, questions, images=[...], prompt={...})`. The vision runner passes `images` as paths under the suite directory; the engines accept any mix of paths, `data:` URLs, raw base64 strings and bytes (`decision_index.vision.images.load`), and send them on as data URLs, raw base64 or paths. The response is the usual `{"answers": {"answer": {"type": "choice", "choice": letter, "probabilities": {letter: p}}}}`.

## Engines

| Engine | What it drives | Main options |
|---|---|---|
| `vllm-vl` | vLLM in-process, the board's reference path for stock VLMs: images capped at 1.6 MP, one user turn (images, then the question, `A. option` lines and `Answer with the option's letter from the given choices directly.`), one decoding step restricted to the option letters, read from processed logprobs; the readout prefix is continued as an assistant turn. Rows go to `llm.chat` in batches of `--chunk` (256) per set. | `model`, `max_model_len` (32768), `max_images` (8), `gpu_memory_utilization` (0.9), `max_pixels` (1,638,400), `mm_processor_kwargs`, `llm_options` (passed to `vllm.LLM`) |
| `decide` | A `/v1/decide` server: `{"kind": "choice", "state": [{"image": data URL}, ...], "question", "options": [texts], "thinking": "off"}`, answered with `choice_index` and `probabilities`. With `protocol=decisions` it speaks `/v1/decisions` instead: `{"model", "state", "questions", "images": [...]}`. | `base_url`, `path`, `protocol` (`decide` or `decisions`), `model`, `image_encoding` (`data_url`, `base64`, `path`), `thinking` (`off`; null leaves it out), `max_pixels` (1,638,400), `extra` (merged into the body); token from `DECISION_INDEX_API_KEY` |
| `openai-vlm` | Any OpenAI-compatible `/chat/completions` endpoint (default `https://router.huggingface.co/v1` with your HF token). Images are re-encoded as JPEG and capped at `max_pixels`, temperature 0; the first option letter in a 16-token reply is the answer, else one 1,024-token reply is read for its last stated letter; a reply with no letter is recorded as `unsupported`. | `model`, `base_url`, `max_pixels` (1,638,400), `extra_body` (e.g. `{"chat_template_kwargs": {"enable_thinking": false}}`); token from `DECISION_INDEX_API_KEY` |

The `decide` and `openai-vlm` engines run `--workers` (8) requests at a time. Any engine can be your own class (`--engine my_module:MyEngine`): take `state, questions, images=(), prompt=None`, and optionally a `batch(payloads)` method returning one `(response, raw)` or exception per payload.

The board ran each entrant through its author's own inference code where there is one; these engines cover the common serving shapes.

## Image resolution

Models run at their **native resolution, capped at 1,638,400 pixels (1.6 MP) per image**. If the model's author caps image size below 1.6 MP (a `max_pixels` or similar setting in the processor config, a resize in the serving code), lift that cap up to 1.6 MP for the run; never go above it. Architectures with a fixed image budget by design use their largest setting at or below 1.6 MP; for example, Gemma 4 runs at 560 soft tokens, about 1.3 MP. Everything stays subject to the latency rule: median, mean and 80th-percentile latency at most 1,000 ms per request on one RTX PRO 6000. If the model breaks it at this resolution, it is not added.

All three engines apply the cap by default (`max_pixels`, default 1,638,400):

- `vllm-vl` and `decide` downscale any image over the cap to fit it (aspect ratio kept, LANCZOS, re-encoded as PNG or JPEG quality 95). Smaller images go out as their original bytes, to the model's own processor or to the server. Lift a processor cap below 1.6 MP with `--option mm_processor_kwargs='{"max_pixels": 1638400}'` for `vllm-vl` (the keys depend on the model), or in the server's own settings for `decide`. With `image_encoding=path` the server reads the files itself, so `decide` refuses images over the cap instead of resizing them.
- `openai-vlm` re-encodes every image as JPEG quality 90 and downscales it to fit the cap when larger.

`--option max_pixels=null` turns the kit's cap off. A board run doesn't do that.

Every run records the budget it actually used. In `environment.json`:

- `model_source.max_pixels` and `model_source.image_resize` hold the kit's cap and resize policy;
- for `vllm-vl`, `model_source.processor_image_budget` holds the processor's own image settings, with any `mm_processor_kwargs`;
- `effective_image_budget` holds `max_image_pixels`, the largest image sent in pixels, and for `vllm-vl` also `max_image_tokens`, the most image tokens in one request (counted from the prompt).

Each result row's `raw_output` has the row's `image_pixels` as sent, plus `image_tokens` where the engine can count them.

## Run and score

Locally:

```bash
pip install -e '.[vision]' 'vllm==0.30.0'
decision-index vision pipeline --dir work/vision --engine vllm-vl --model Qwen/Qwen3-VL-8B-Instruct --out runs/vision-qwen3-vl-8b
decision-index vision pipeline --dir work/vision --engine decide --option base_url=http://127.0.0.1:8000 --out runs/vision-my-server
decision-index vision pipeline --dir work/vision --engine openai-vlm --model Qwen/Qwen3.5-397B-A17B --out runs/vision-api
```

`vision run` and `vision score --results runs/<name>/results.jsonl` are the two halves. Runs checkpoint to `results.jsonl` and resume, retrying only rows that errored; `--limit N` runs the first N rows. `vision run` refuses a suite that fails `vision verify` unless you pass `--no-verify`.

As one Hugging Face Job (installs the kit, rebuilds the suite or downloads your private copy, runs, scores, uploads to `runs/<name>/` of the results repo):

```bash
decision-index vision hf-job --engine vllm-vl --model Qwen/Qwen3-VL-8B-Instruct --results-repo <you>/decision-index-results --suite-dataset <you>/vision-suite
```

`--dry-run` prints the job command; `--flavor`, `--image`, `--timeout` and `--git-url` work as for the text job. Jobs that rebuild the suite need an HF token that has accepted the Winoground terms.

## Scoring

Per benchmark, the skill is `(accuracy − chance) / (1 − chance)`, where chance is the mean of `1 / number of options` over the benchmark's rows; rows without an answer (unsupported, error, missing) count as wrong. Skills are clipped to [0, 1] and averaged with the weights above (nine benchmarks at 1, R-Bench-M 0.5, MMMU-Pro vision 0.25), times 100. That is the **public vision index**, written to `vision-scores.json` with each benchmark's rows, answered count, accuracy, chance and skill. Negative skills are shown as they are but count as 0 in the index.

The scorer reproduces the board: rescoring the board's own run files gives 72.78 for JEV-27B-VL and 69.80 for Solomon v1.1 (27B): the same floating-point values the board computed, and the same eleven benchmark values. `tests/fixtures/vision-public-0.3.json` keeps their per-benchmark correct counts so `tests/test_vision.py` checks the index math against the published numbers.

The **Full** vision score is `0.5 × public + 0.5 × private`.

## Submitting

1. Run the whole public vision suite with this kit (`vision pipeline` or `vision hf-job`) and upload the run directory to a Hub dataset. `vision-scores.json` must say `"complete": true`; the results file is rescored on review.
2. Open a pull request adding a line to `submissions/README.md` marked **vision**, with the model name and Hub repo, the results dataset link, the engine and its options, the kit commit, the hardware, any declared limits (for example a maximum number of images per request), and the image resolution you ran at: native up to 1.6 MP with every lower author-set cap lifted, or the model's largest fixed budget at or below 1.6 MP ([Image resolution](#image-resolution)). The run's `effective_image_budget` must agree.
3. The maintainers then run the private half themselves: private tests of the same skills as the public benchmarks. They run the model the way you did, so it must be runnable by them: weights on the Hub (public, or access granted on request), and the inference code or server you used with its settings. The private tests are never published or shared, with submitters or anyone else; the board shows only each model's aggregate private score.
