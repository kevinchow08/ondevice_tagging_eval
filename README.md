# 端侧打标签模型评测工具 / On-Device Tagging Model Evaluation Toolkit

[中文](#中文) | [English](#english)

---

## 中文

一套轻量级评测脚手架，用来评估**端侧（on-device）图片/文档打标签模型**的标签质量——例如跑在本地、通过 [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server` 之类工具起服务的中小尺寸多模态模型。

### 核心思路

开放词表打标签任务里，模型自己决定输出什么词，没有唯一"正确答案"。用固定标签表做字符串精确匹配，会误伤那些用同义词或更细/更粗粒度描述的正确输出。所以这里用两条互补的评测路径，而不是精确匹配：

1. **LLM 裁判评审**（[llm_judge.py](eval_pipeline/scripts/llm_judge.py)，推荐主线）：让一个更强的模型直接看原图/原文，同时看被测模型给出的标签，判断有没有瞎编（hallucination）、有没有漏打（missing），给一个 0~1 的综合准确度分，外加一句点评。不需要参考标签集。
2. **语义相似度打分**（[score_semantic.py](eval_pipeline/scripts/score_semantic.py)，可选辅助）：被测模型和参考模型各自独立打一遍标签，用 embedding 算 cosine 相似度统计 precision / recall。依赖较重（`sentence-transformers` + torch），且需要额外让参考模型也跑一遍打标签。

两条路可以只用其中一条，也可以都跑，[summarize.py](eval_pipeline/scripts/summarize.py) 会把跑出来的结果都汇总进 `results/summary_report.md`，没跑的部分会在报告里注明跳过，不会报错。

### 目录结构

```
.
├── eval_pipeline/
│   ├── core/                 # 共享库代码，被 scripts/ 下的脚本导入
│   │   ├── config.py         # 模型端点配置：被测模型(small) + 裁判/参考模型(reference)
│   │   ├── prompts.py        # 所有 prompt 集中在这里（中英双语），方便调整
│   │   └── client.py         # 公共工具：建 client、编码图片、并发执行、性能采集、宽松解析 JSON、失败重试
│   ├── scripts/               # 命令行入口，实际跑的脚本
│   │   ├── tag_images.py     # 给图片打标签
│   │   ├── tag_documents.py  # 给 PDF 文档打标签（先用 pdfplumber 抽文本）
│   │   ├── llm_judge.py      # LLM 裁判评审
│   │   ├── score_semantic.py # 语义相似度打分（可选）
│   │   └── summarize.py      # 汇总成 results/summary_report.md
│   ├── Makefile               # 常用命令的简写，见下面"快速开始"
│   ├── requirements.txt
│   ├── .env.example           # 环境变量模板，复制成 .env 填真实值
│   └── results/               # 跑出来的结果(jsonl/csv/md)，不入库，跑完自己本地看
└── tagging_test_samples/     # 测试样本，见该目录下的 README
```

所有 `scripts/*.py` 都要在 `eval_pipeline/` 目录下用 `python scripts/xxx.py` 的方式运行（不是 `cd scripts/` 再跑），脚本自己会把 `eval_pipeline/` 加进 `sys.path` 找到 `core` 包，不需要额外装包/配置。

### 快速开始

> 嫌下面每条命令参数太多？`eval_pipeline/` 下有个 `Makefile`，把常用参数组合收进了短命令，比如 `make tag-images`、`make judge-images LIMIT=20`、`make smoke`，跑 `make help` 看全部命令。两种方式等价，`make` 只是省得每次手打参数；要精细控制参数（比如 Windows 上没有 `make`），还是用下面这套完整命令。

1）装依赖

```bash
cd eval_pipeline
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

2）解压测试样本（图片/文档不入库，用 zip 存的，见 [tagging_test_samples/README.md](tagging_test_samples/README.md)）

```bash
cd tagging_test_samples
unzip images.zip
unzip documents.zip
cd ..
```

3）起被测的本地模型服务。以 `llama.cpp` 为例——**图片打标签要求模型带视觉能力**，起服务时必须带 `--mmproj`（多模态投影权重文件），否则一收到带图片的请求就会直接报错（常见现象：`502`）：

```bash
llama-server \
  -m <你的模型.gguf> \
  --mmproj <对应的mmproj.gguf> \
  --port 8080 \
  -c 16384 \
  -np 1 \
  --reasoning off \
  --temp 0.3 \
  --repeat-penalty 1.15
```

几个容易踩的坑，提前说明：

- `-np`（并发槽数）会把 `-c`（总 context）平分给每个槽。比如 `-c 65536 -np 16`，每个请求实际能用的 context 只有 4096，很容易在图片/长文档场景下不够用导致截断。脚本默认是顺序请求，`-np 1` 就够，把 `-c` 全部留给单个请求；**只有配合下面"并发跑"那节把客户端也改成并发发请求，调大 `-np` 才有意义**——单开 `-np` 不改客户端，多出来的槽位只是空转，不会变快。
- 会思考（reasoning）的模型建议用 `--reasoning off` 关掉思考模式：打标签是结构化 JSON 输出任务，不需要思考过程，开着思考容易把 `max_tokens` 预算耗在 `reasoning_content` 上，甚至偶尔陷入重复生成的死循环。
- **采样参数**：`llama-server` 默认 `--temp 0.80`、`--repeat-penalty 1.00`（=不惩罚重复）。打标签这类结构化抽取任务不需要高随机性，默认参数实测会出现"同一个标签复读好几遍直到把 token 预算耗光"的退化情况（比如标签列表变成 `["鸟","鸟","鸟",...]`，或者更极端的直接把 JSON 挤爆导致解析失败）。把 `--temp` 降到 0.2~0.4、`--repeat-penalty` 调到 1.1~1.2，能明显缓解。
- **结构化输出（JSON Schema）**：如果服务端支持 OpenAI 兼容的 `response_format: {"type": "json_schema", ...}`（`llama.cpp` 较新版本支持，用 `-j/--json-schema` 或请求里带 `response_format` 都行），`tag_images.py`/`tag_documents.py` 会自动对本地模型请求加上 schema 约束（`eval_pipeline/core/client.py` 里的 `TAGS_SCHEMA`/`DOCUMENT_TAGS_SCHEMA`），从语法层面保证输出一定是合法 JSON、标签数组长度受控，配合 `uniqueItems: true` 还能杜绝同一个标签被逐字重复。这个只在检测到本地地址（`localhost`/`127.0.0.1`）时启用，云端裁判模型的兼容性没验证过，不强行加。
- 如果系统配了 `HTTP_PROXY` / `HTTPS_PROXY` 环境变量，注意别让本地服务的请求被代理误伤（`core/client.py` 已经对 `localhost`/`127.0.0.1` 的 client 做了 `trust_env=False` 处理，绕开代理）。

4）配置模型端点：`eval_pipeline/core/config.py` 里 `small`（被测模型）的 `base_url`/`model` 一般不用改；`reference`（裁判/参考模型）的 `base_url`/`api_key`/`model` 从环境变量读取，不写在 `config.py` 里。把 `eval_pipeline/.env.example` 复制成 `eval_pipeline/.env`（已在 `.gitignore` 里，不会被提交），填入你自己的真实值：

```bash
cd eval_pipeline
cp .env.example .env
# 编辑 .env，填入 REFERENCE_API_KEY 等
```

5）先小规模跑通

```bash
python scripts/tag_images.py --profile small --limit 5
python scripts/tag_documents.py --profile small --limit 2
```

检查 `results/*.jsonl` 里每行 `error` 是不是 `null`，`tags` 有没有实际内容。

### 并发跑（样本量大、覆盖率跑得慢的时候再用）

默认所有脚本都是顺序请求（一条接一条），样本量不大时（几百条以内）够用。样本量上到几千条，可以开并发，但**服务端和客户端要同时改，缺一不可**：

1. 服务端 `-np` 调大，`-c` 要跟着等比例放大，保证单个请求的 context 不缩水（比如单槽 `-c 16384` 够用，开4个槽位并发就要 `-c 65536`）：

```bash
llama-server \
  -m <你的模型.gguf> \
  --mmproj <对应的mmproj.gguf> \
  --port 8080 \
  -c 65536 \
  -np 4 \
  --reasoning off \
  --temp 0.3 \
  --repeat-penalty 1.15
```

2. 客户端加 `--concurrency N`（`tag_images.py`/`tag_documents.py`/`llm_judge.py` 都支持，默认1即顺序执行，行为不变），`N` 跟服务端 `-np` 对齐：

```bash
python scripts/tag_images.py --profile small --concurrency 4
```

`llm_judge.py` 走的是付费云端API，并发数建议从小往大试（比如先2再4），避免撞到服务商的限速/并发上限导致大量请求失败重试，反而更慢更费钱。

6）跑全量 + 裁判评审 + 汇总

**裁判现在是"两两对比"模式**（candidate=被测的 small 模型 vs comparison=reference 模型独立打的标签，裁判直接看图/原文判断哪组更准，不是给 small 单独打一个绝对分——这样更可靠，参考 [MT-Bench 论文](https://arxiv.org/abs/2306.05685) 的结论）。所以 `llm_judge.py` 现在依赖 reference 模型也独立打过一遍标签，`tag_images.py --profile reference` 这步不能省：

裁判还会**可选地**用上 `images_manifest.csv`/`documents_manifest.csv` 里的 `reference_label_en`/`document_type_cn` 字段（如果有的话）——不是拿去做精确字符串匹配（细分物种名跟"只给大类"的要求会冲突），是作为"已知事实"喂给裁判帮它核对，能减少裁判和参考模型"两边共享同一个认知盲区"的风险（比如都把某种昆虫认错成另一类）。这个字段留空也完全能跑，裁判会退化成纯靠自己看图/看原文判断，不强制要求自建数据集也要维护这个标注。

```bash
python scripts/tag_images.py --profile small
python scripts/tag_documents.py --profile small
python scripts/tag_images.py --profile reference       # 裁判要用，付费API
python scripts/tag_documents.py --profile reference    # 裁判要用，付费API

# 裁判评审也要调用付费API，建议先抽查再决定要不要跑全量
python scripts/llm_judge.py --kind images --limit 20
python scripts/llm_judge.py --kind documents
python scripts/llm_judge.py --kind images     # 抽查结果OK了，再补跑剩下的

python scripts/summarize.py
```

7）看 `results/summary_report.md`

### 局限性（诚实说明，别拿这个当权威 benchmark）

- **裁判没做过人工校准**：裁判现在是两两对比（候选 vs 参考模型独立打的标签），比早期的绝对打分更可靠（[MT-Bench](https://arxiv.org/abs/2306.05685) 论文的结论），请求也把 `temperature` 锁到了 0（降低同一批内容重跑时判断漂移的概率）。但这些都只是减少"裁判自己不稳定"这类误差，**没有验证过裁判的判断本身准不准**——没有拿人工判断去校准过。MT-Bench 论文的建议是找 30~50 条样本人工标注、算裁判和人的一致率，这一步我们还没做。
- **样本量有限**：默认自带的测试图片是 ImageNet 风格的单目标自然图，文档样本只有 6 份——n 都不大，结论只能当方向性参考，别当成统计意义上的定论。
- **裁判模型自己也会犯错，"参考模型"也不是ground truth**：裁判和拿来对比的参考模型都可能犯同样的错（比如两边都把某种昆虫认错成同一个错误类别），纯两两对比看不出这种共享盲区，只有真正回看原图/原文才能发现——这也是为什么瞎编/漏打这两个指标是裁判直接核对原始内容得出的，不是靠对比算出来的。有 `reference_label_en`/`document_type_cn` 这个可选的弱参考字段时，裁判会额外用它核对，能缓解一部分这个问题，但不是每条样本都保证有这个字段，缓解不代表消除。
- **开放词表打标签这个场景，没有能直接照搬的现成评估基准**：图像描述里常用的 CHAIR/POPE 之类的成熟指标，本质上还是依赖每张图片预先标注好的固定物体列表做核对；我们这里没有固定标签集（图片/文档集合可以自由扩充），只是借用了 CHAIR 的计数方式（瞎编率算法），验证的判断来源换成了"更强模型直接看图裁决"，这个组合本身没有被哪篇论文直接验证过，需要靠上面说的人工校准去补上这一环。
- **依赖云端商业模型做基准的话可复现性打折扣**：商业模型的权重可能被服务商静默更新，同一个 model id 不同时间的表现未必一致，建议在报告里记录清楚用的是哪个模型快照/哪一天跑的。

**结论**：这是一个方法论合理的轻量评测脚手架，适合快速判断一个端侧模型的标签能力大致水平、找出系统性短板（比如"细分类目容易瞎编""文档结构化字段容易漏打"这类问题）；不适合当成严谨的排行榜工具来引用绝对分数，尤其是在人工校准这一步补上之前。

### License

按你实际情况填（MIT / Apache-2.0 等）。

---

## English

A lightweight evaluation scaffold for judging the tagging quality of **on-device image/document tagging models** — e.g. small-to-mid-size multimodal models served locally via tools like [llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server`.

### Core idea

Open-vocabulary tagging has no single "correct answer" — the model decides its own wording. Scoring against a fixed tag list with exact string matching unfairly penalizes correct outputs phrased with synonyms or a different granularity. Instead, this toolkit uses two complementary evaluation paths, not exact matching:

1. **LLM-as-judge** ([llm_judge.py](eval_pipeline/scripts/llm_judge.py), recommended primary path): a stronger model looks at the original image/text _and_ the candidate model's tags, and reports hallucinated tags, missing important tags, and an overall 0–1 accuracy score plus a short comment. No reference tag set required.
2. **Semantic similarity scoring** ([score_semantic.py](eval_pipeline/scripts/score_semantic.py), optional/auxiliary): the candidate model and a reference model each independently tag the same samples; tags are embedded and compared via cosine similarity to compute precision/recall. Heavier dependency (`sentence-transformers` + torch), and requires an extra tagging pass from the reference model.

Either path can be used alone, or both — [summarize.py](eval_pipeline/scripts/summarize.py) aggregates whatever result files exist into `results/summary_report.md`, noting any missing section as skipped rather than failing.

### Project layout

```
.
├── eval_pipeline/
│   ├── core/                 # shared library code, imported by scripts/
│   │   ├── config.py         # model endpoints: candidate model (small) + judge/reference model (reference)
│   │   ├── prompts.py        # all prompts live here (bilingual zh/en)
│   │   └── client.py         # shared helpers: client setup, image encoding, concurrency, perf capture, loose JSON parsing, retries
│   ├── scripts/               # CLI entrypoints, the scripts you actually run
│   │   ├── tag_images.py     # tag images
│   │   ├── tag_documents.py  # tag PDF documents (text extracted via pdfplumber)
│   │   ├── llm_judge.py      # LLM-as-judge evaluation
│   │   ├── score_semantic.py # semantic similarity scoring (optional)
│   │   └── summarize.py      # aggregates into results/summary_report.md
│   ├── Makefile               # shortcuts for common commands, see "Quickstart" below
│   ├── requirements.txt
│   ├── .env.example           # env var template, copy to .env and fill in real values
│   └── results/               # generated output (jsonl/csv/md), not checked into git
└── tagging_test_samples/     # test samples, see its own README
```

Every `scripts/*.py` is meant to be run from the `eval_pipeline/` directory as `python scripts/xxx.py` (not from inside `scripts/`) — each script adds `eval_pipeline/` to `sys.path` itself to find the `core` package, no install step or packaging needed.

### Quickstart

> Tired of typing out every flag? `eval_pipeline/` has a `Makefile` that bundles common flag combos into short commands — `make tag-images`, `make judge-images LIMIT=20`, `make smoke`, run `make help` for the full list. Both are equivalent; `make` just saves typing. For fine-grained flag control (or on Windows without `make`), use the full commands below.

1. Install dependencies

```bash
cd eval_pipeline
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

2. Unpack the test samples (images/documents aren't checked in directly, they're zipped — see [tagging_test_samples/README.md](tagging_test_samples/README.md))

```bash
cd tagging_test_samples
unzip images.zip
unzip documents.zip
cd ..
```

3. Start the candidate model's local server. Using `llama.cpp` as an example — **image tagging requires a vision-capable model**, so the server must be started with `--mmproj` (the multimodal projector weights), otherwise any request containing an image will fail server-side (commonly surfacing as `502`):

```bash
llama-server \
  -m <your-model.gguf> \
  --mmproj <matching-mmproj.gguf> \
  --port 8080 \
  -c 16384 \
  -np 1 \
  --reasoning off \
  --temp 0.3 \
  --repeat-penalty 1.15
```

A few gotchas worth knowing up front:

- `-np` (parallel slots) divides the total `-c` (context) evenly across slots. E.g. `-c 65536 -np 16` leaves only 4096 tokens per request — easy to blow through with an image or a long document, causing truncation. Scripts default to sequential requests, so `-np 1` is enough — give the full context to a single request. **Raising `-np` only pays off if you also switch the client to concurrent requests (see "Running concurrently" below)** — otherwise the extra slots just sit idle.
- For reasoning-capable models, `--reasoning off` is recommended: tagging is a structured JSON output task that doesn't need a chain of thought, and leaving it on can burn the `max_tokens` budget on `reasoning_content` — occasionally the model even gets stuck in a repetitive generation loop.
- **Sampling parameters**: `llama-server` defaults to `--temp 0.80` and `--repeat-penalty 1.00` (i.e. no repetition penalty at all). A structured extraction task like tagging doesn't need much randomness, and the defaults measurably produce degenerate output — the same tag repeated until the token budget runs out (e.g. a tag list collapsing into `["bird","bird","bird",...]`), or in worse cases blowing past the JSON entirely and failing to parse. Dropping `--temp` to 0.2–0.4 and raising `--repeat-penalty` to 1.1–1.2 clearly helps.
- **Structured output (JSON Schema)**: if the server supports the OpenAI-compatible `response_format: {"type": "json_schema", ...}` (recent `llama.cpp` builds do, via `-j/--json-schema` or a `response_format` in the request), `tag_images.py`/`tag_documents.py` automatically attach a schema (`TAGS_SCHEMA`/`DOCUMENT_TAGS_SCHEMA` in `eval_pipeline/core/client.py`) to requests sent to a local model. This grammar-constrains generation so the output is guaranteed valid JSON with a bounded tag array, and `uniqueItems: true` rules out the exact-duplicate-tag failure mode at the syntax level. It's only enabled when the endpoint is detected as local (`localhost`/`127.0.0.1`) — compatibility with cloud judge models hasn't been verified, so it's not forced there.
- If `HTTP_PROXY` / `HTTPS_PROXY` are set system-wide, make sure requests to your local server aren't accidentally routed through the proxy (`core/client.py` already forces `trust_env=False` for `localhost`/`127.0.0.1` clients to sidestep this).

4. Configure model endpoints: `small` (the candidate model) in `eval_pipeline/core/config.py` usually needs no changes. `reference` (the judge/reference model)'s `base_url`/`api_key`/`model` are read from environment variables instead of being written into `config.py`. Copy `eval_pipeline/.env.example` to `eval_pipeline/.env` (already in `.gitignore`, never committed) and fill in your real values:

```bash
cd eval_pipeline
cp .env.example .env
# edit .env, fill in REFERENCE_API_KEY etc.
```

5. Smoke-test on a small sample

```bash
python scripts/tag_images.py --profile small --limit 5
python scripts/tag_documents.py --profile small --limit 2
```

Check that every row in `results/*.jsonl` has `error: null` and non-empty `tags`.

### Running concurrently (once your sample count grows and sequential runs feel slow)

All scripts default to sequential requests (one at a time), which is fine for a few hundred samples. Once you're into the thousands, you can go concurrent — **both the server and the client need to change together**:

1. Raise `-np` on the server, and scale `-c` proportionally so a single request's context doesn't shrink (e.g. if one slot needs `-c 16384`, 4 concurrent slots need `-c 65536`):

```bash
llama-server \
  -m <your-model.gguf> \
  --mmproj <matching-mmproj.gguf> \
  --port 8080 \
  -c 65536 \
  -np 4 \
  --reasoning off \
  --temp 0.3 \
  --repeat-penalty 1.15
```

2. Add `--concurrency N` on the client side (`tag_images.py`/`tag_documents.py`/`llm_judge.py` all support it, defaulting to 1 = sequential, unchanged behavior), matching the server's `-np`:

```bash
python scripts/tag_images.py --profile small --concurrency 4
```

`llm_judge.py` hits a paid cloud API — ramp concurrency up gradually (try 2, then 4) rather than jumping straight to a high number, to avoid tripping the provider's rate/concurrency limits, which just causes retries and ends up slower and more expensive.

6. Run the full batch + judge + summary

**The judge now runs in pairwise mode** (candidate = the small/candidate model under test vs. comparison = the reference model's independently-generated tags; the judge looks directly at the image/text and decides which set is more accurate, rather than giving the candidate an absolute score — this is more reliable, per the [MT-Bench paper](https://arxiv.org/abs/2306.05685)'s findings). So `llm_judge.py` now depends on the reference model having tagged the same set independently — don't skip the `--profile reference` pass:

The judge also **optionally** uses the `reference_label_en`/`document_type_cn` field from `images_manifest.csv`/`documents_manifest.csv` when present — not for exact string matching (a fine-grained species name would conflict with the "coarse category only" requirement), but as a known fact fed to the judge to help it verify, which reduces the risk of the judge and the reference model sharing the same blind spot (e.g. both misidentifying the same insect the same wrong way). Leaving the field blank works fine too — the judge just falls back to judging purely from the image/text, so datasets you build yourself aren't required to maintain this label.

```bash
python scripts/tag_images.py --profile small
python scripts/tag_documents.py --profile small
python scripts/tag_images.py --profile reference       # needed for the judge, paid API
python scripts/tag_documents.py --profile reference    # needed for the judge, paid API

# Judging also calls a paid API — sample first before committing to the full run
python scripts/llm_judge.py --kind images --limit 20
python scripts/llm_judge.py --kind documents
python scripts/llm_judge.py --kind images     # once the sample looks good, run the rest

python scripts/summarize.py
```

7. Read `results/summary_report.md`

### Limitations (stated honestly — don't treat this as an authoritative benchmark)

- **The judge hasn't been calibrated against humans**: it now runs pairwise (candidate vs. the reference model's independently-generated tags) rather than absolute scoring, which is more reliable per the [MT-Bench paper](https://arxiv.org/abs/2306.05685)'s findings, and requests pin `temperature` to 0 (reducing drift when re-running the same batch). But both only reduce "the judge being unstable" — **there's still no check that the judge's actual judgment is correct**, since it hasn't been validated against human ratings. MT-Bench's own recommendation is a 30-50 example human-annotated calibration set to measure judge-human agreement; we haven't done that step yet.
- **Sample sizes are small**: the bundled test images are single-subject, ImageNet-style natural photos, and there are only 6 document samples — treat conclusions as directional, not statistically definitive.
- **The judge model can itself be wrong, and so can the "reference" model**: the judge and the reference model it compares against can share the same blind spot (e.g. both misclassify the same insect the same wrong way), which a pairwise comparison alone won't catch — only checking against the actual image/text catches that. That's why hallucination/missing-tag counts are derived from the judge directly checking the source content, not from the pairwise comparison itself. When the optional `reference_label_en`/`document_type_cn` field is present, the judge uses it as an extra check, which helps — but it's not guaranteed on every sample, so this mitigates the risk rather than eliminating it.
- **There's no off-the-shelf benchmark for truly open-vocabulary tagging**: established image-captioning hallucination metrics like CHAIR/POPE still rely on a fixed, pre-annotated object list per image; we have no fixed label set (the image/document collection is meant to be freely extensible), so we borrowed CHAIR's counting formula but swapped in "a stronger model judges directly from the image" as the source of truth — that specific combination hasn't been validated by any single published benchmark, which is exactly why the human-calibration step above matters.
- **Reproducibility is limited when the reference is a commercial cloud model**: providers can silently update model weights behind a stable model id, so results from the same id may drift over time — record which model snapshot and date you ran against.

**Bottom line**: this is a methodologically sound, lightweight evaluation scaffold — good for quickly gauging an on-device model's tagging ability and surfacing systematic weaknesses (e.g. "hallucinates on fine-grained categories", "under-extracts structured fields from documents"). It is not meant to be cited as a rigorous leaderboard tool with authoritative absolute scores, especially before the human-calibration step is done.

### License

Fill in what applies (MIT / Apache-2.0 / etc).
