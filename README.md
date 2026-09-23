# 端侧打标签模型评测工具 / On-Device Tagging Model Evaluation Toolkit

[中文](#中文) | [English](#english)

---

## 中文

一套轻量级评测脚手架，用来评估**端侧（on-device）图片打标签模型**的标签质量——例如跑在本地、通过 [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server` 之类工具起服务的中小尺寸多模态模型。

（这个工具最早也想覆盖文档打标签，但文档测试集样本量小、投入产出比不高，已经拿掉，先把图片这条路径做稳。）

### 核心思路

开放词表打标签任务里，模型自己决定输出什么词，没有唯一"正确答案"。用固定标签表做字符串精确匹配，会误伤那些用同义词或更细/更粗粒度描述的正确输出。所以评测基准不是字符串精确匹配，而是：

**人工核实过的闭集标签库**（`tagging_test_samples/ground_truth_images.csv`）+ **embedding 语义相似度比对**（[score_semantic.py](eval_pipeline/scripts/score_semantic.py)）：每张图片都提前由人工写好一份标准答案标签列表，被测模型跑出来的标签用 embedding cosine 相似度去对这份标准答案算 precision（打的标签有多少是真的）/主体标签覆盖率（最重要的主体大类标签打没打对）。

这套方法早期版本用的是"LLM 裁判直接看图评审"，实践下来发现这个方案有个根子问题：裁判本身是不是可信、判得准不准，需要额外一层人工校准去验证，而且裁判和被测模型如果共享同一个知识盲区（比如都不认识某种节肢动物），两两对比也发现不了。换成人工闭集标签库之后，比对本身是确定性计算（同一批数据重跑多少次结果都一样），不再需要"验证裁判靠不靠谱"这一环——人工核实的工作量没有消失，是从"每次跑完之后抽查裁判判得对不对"，挪到了"一次性把标准答案写对，之后随便重跑"。

[summarize.py](eval_pipeline/scripts/summarize.py) 把 `score_semantic.py` 跑出来的结果汇总进 `results/summary_report.md`，报告最上面给出一个分级判定（🟢可用/🟡有条件可用/🔴不建议直接使用）。

### 目录结构

```
.
├── eval_pipeline/
│   ├── core/                 # 共享库代码，被 scripts/ 下的脚本导入
│   │   ├── config.py         # 模型端点配置：被测模型(small)
│   │   ├── prompts.py        # 打标签用的 prompt（中英双语），方便调整
│   │   └── client.py         # 公共工具：建 client、编码图片、并发执行、性能采集、宽松解析 JSON、失败重试
│   ├── scripts/               # 命令行入口，实际跑的脚本
│   │   ├── tag_images.py     # 给图片打标签
│   │   ├── score_semantic.py # 语义相似度打分：对照人工闭集标签库算 precision/主体标签覆盖率
│   │   └── summarize.py      # 汇总成 results/summary_report.md（含分级判定）
│   ├── Makefile               # 常用命令的简写，见下面"快速开始"
│   ├── requirements.txt
│   ├── .env.example           # 环境变量模板，复制成 .env 填真实值
│   └── results/               # 跑出来的结果(jsonl/csv/md)，不入库，跑完自己本地看
└── tagging_test_samples/     # 测试样本 + 人工核实过的闭集标签库(ground_truth_images.csv)，见该目录下的 README
```

所有 `scripts/*.py` 都要在 `eval_pipeline/` 目录下用 `python scripts/xxx.py` 的方式运行（不是 `cd scripts/` 再跑），脚本自己会把 `eval_pipeline/` 加进 `sys.path` 找到 `core` 包，不需要额外装包/配置。

### 快速开始

> 嫌下面每条命令参数太多？`eval_pipeline/` 下有个 `Makefile`，把常用参数组合收进了短命令，比如 `make tag-images`、`make full`、`make smoke`，跑 `make help` 看全部命令。两种方式等价，`make` 只是省得每次手打参数；要精细控制参数（比如 Windows 上没有 `make`），还是用下面这套完整命令。

1）装依赖

```bash
cd eval_pipeline
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

2）解压测试样本（图片不入库，用 zip 存的，见 [tagging_test_samples/README.md](tagging_test_samples/README.md)）

```bash
cd tagging_test_samples
unzip images.zip
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

- `-np`（并发槽数）会把 `-c`（总 context）平分给每个槽。比如 `-c 65536 -np 16`，每个请求实际能用的 context 只有 4096，很容易在图片场景下不够用导致截断。脚本默认是顺序请求，`-np 1` 就够，把 `-c` 全部留给单个请求；**只有配合下面"并发跑"那节把客户端也改成并发发请求，调大 `-np` 才有意义**——单开 `-np` 不改客户端，多出来的槽位只是空转，不会变快。
- 会思考（reasoning）的模型建议用 `--reasoning off` 关掉思考模式：打标签是结构化 JSON 输出任务，不需要思考过程，开着思考容易把 `max_tokens` 预算耗在 `reasoning_content` 上，甚至偶尔陷入重复生成的死循环。
- **采样参数**：`llama-server` 默认 `--temp 0.80`、`--repeat-penalty 1.00`（=不惩罚重复）。打标签这类结构化抽取任务不需要高随机性，默认参数实测会出现"同一个标签复读好几遍直到把 token 预算耗光"的退化情况（比如标签列表变成 `["鸟","鸟","鸟",...]`，或者更极端的直接把 JSON 挤爆导致解析失败）。把 `--temp` 降到 0.2~0.4、`--repeat-penalty` 调到 1.1~1.2，能明显缓解。
- **结构化输出（JSON Schema）**：如果服务端支持 OpenAI 兼容的 `response_format: {"type": "json_schema", ...}`（`llama.cpp` 较新版本支持，用 `-j/--json-schema` 或请求里带 `response_format` 都行），`tag_images.py` 会自动对本地模型请求加上 schema 约束（`eval_pipeline/core/client.py` 里的 `TAGS_SCHEMA`），从语法层面保证输出一定是合法 JSON、标签数组长度受控，配合 `uniqueItems: true` 还能杜绝同一个标签被逐字重复。这个只在检测到本地地址（`localhost`/`127.0.0.1`）时启用，云端模型的兼容性没验证过，不强行加。
- 如果系统配了 `HTTP_PROXY` / `HTTPS_PROXY` 环境变量，注意别让本地服务的请求被代理误伤（`core/client.py` 已经对 `localhost`/`127.0.0.1` 的 client 做了 `trust_env=False` 处理，绕开代理）。

4）确认模型端点配置：`eval_pipeline/core/config.py` 里 `MODEL_PROFILES["small"]`（被测模型）的 `base_url`/`model` 一般不用改，除非你换了端口/引擎名。评测基准是 `tagging_test_samples/ground_truth_images.csv` 这份人工闭集标签库，不依赖任何云端模型，不需要配置额外的 API key。想改默认值的话复制 `.env.example` 成 `.env` 填真实值（`.env` 已在 `.gitignore` 里，不会被提交）：

```bash
cd eval_pipeline
cp .env.example .env
```

5）先小规模跑通

```bash
python scripts/tag_images.py --profile small --limit 5
```

检查 `results/images_tags_small.jsonl` 里每行 `error` 是不是 `null`，`tags` 有没有实际内容。

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

2. 客户端加 `--concurrency N`（`tag_images.py` 支持，默认1即顺序执行，行为不变），`N` 跟服务端 `-np` 对齐：

```bash
python scripts/tag_images.py --profile small --concurrency 4
```

6）跑全量 + 语义打分 + 汇总（全程本地免费，不需要调用任何付费云端API）

```bash
python scripts/tag_images.py --profile small
python scripts/score_semantic.py       # 对照 ground_truth_images.csv 算 precision/主体标签覆盖率
python scripts/summarize.py
```

（等价简写：`make full`）

7）看 `results/summary_report.md`——最上面"结论速览"给出一个分级判定（🟢可用/🟡有条件可用/🔴不建议直接使用，阈值在 `core/config.py` 的 `QUALITY_GATE` 里，按你的风险容忍度自己调）。报告里还会列出问题最多的几条样本，附上具体哪些标签没匹配上（`unmatched_candidate_tags`/`unmatched_reference_tags`）——这两列是留给你抽查用的：没匹配上不代表模型一定错了，也可能是标准答案本身当初没写全，抽查几条能分清楚。

### 中英文 prompt/输出

`--lang zh`（默认）/`--lang en` 切换 `tag_images.py` 用的 prompt 语言，两版规则内容一致，只是表达语言不同，包括都明确要求"不管图片里出现的文字/品牌是什么语言，标签本身都用 prompt 对应的语言输出"——避免模型被图片里的英文文字/品牌带偏，中途切换成英文回答（这个问题曾经真实出现过：200张图里有3张模型自己切成全英文标签，正好是图里带明显英文品牌的那几张）。标准答案库（`ground_truth_images.csv`）本身只维护了一份中文标签，不用因为测 `--lang en` 就分别维护两份——图片标签大多是颜色/动物/物体这类常见词，跨语言 embedding 相似度普遍在0.85以上（"狗" vs "dog" 能到0.975），混着比不构成实际问题，`score_semantic.py --lang` 只是用来找到对应语言跑出来的候选标签文件（`images_tags_small_en.jsonl` 这种），不影响比对基准。

### 扩充测试集时要做什么

`tagging_test_samples/images/` 加新样本很容易，麻烦的是标准答案：`ground_truth_images.csv` 里新增的文件也要补一行人工核实过的 `tags`（JSON 数组，5~15个标签，第一个固定写主体大类，规则跟打标签 prompt 一致——见 `core/prompts.py` 的 `IMAGE_TAGGING_PROMPT_ZH`）。这是这套方法唯一不能省的人工成本：换掉 LLM 裁判省下来的是"每次跑完还要验证裁判靠不靠谱"，标准答案本身还是得有人写——但只需要写一次，不需要每次评测重新验证。

### 局限性（诚实说明，别拿这个当权威 benchmark）

- **标准答案是人工一次性写的，不保证穷尽每一处细节**：写标签的时候很难保证把图片里所有能被合理提取的信息点都想到——这会导致模型说的某个东西明明是对的，但因为标准答案没写到，比对时被当成"没匹配上"。`semantic_scores_images.csv` 里的 `unmatched_candidate_tags` 列就是留给你处理这个问题的：定期抽查这一列，能分清楚是模型真错了，还是该给标准答案打个补丁。跟老版本的 LLM 裁判比，这个局限性更可控——标准答案是静态、可审查、可修的，不像裁判的判断那样是运行时黑盒。
- **分级判定用"主体标签覆盖率"而不是完整覆盖率**：标准答案的5~15个标签里，只有第一个（主体大类）被当成"必须覆盖"的硬指标，剩下颜色/数量/姿态这些次要标签只是参考展示——要求开放式生成任务把整份标准答案一个不漏全覆盖，对图片这种自由描述场景门槛偏高，容易把"漏了个次要细节"跟"完全没认出主体"混为一谈。
- **样本量有限**：默认自带的测试图片是 ImageNet 风格的单目标自然图（200张）——结论只能当方向性参考，别当成统计意义上的定论。
- **开放词表打标签这个场景，没有能直接照搬的现成评估基准**：图像描述里常用的 CHAIR/POPE 之类的成熟指标，思路上跟这里的方法接近（都依赖一份预先标注好的参考列表核对），但都是针对各自领域定制的，这里是借用了 CHAIR 的计数方式，没有哪篇论文直接验证过这个具体组合。
- **分级判定的阈值是主观设定的**：`QUALITY_GATE` 里 `green_halluc`/`yellow_halluc` 等阈值是一个中性起点，不是行业标准答案，需要结合你自己业务场景对错误的容忍度去调。

**结论**：这是一个方法论合理的轻量评测脚手架，适合快速判断一个端侧模型的图片打标签能力大致水平、找出系统性短板（比如"细分类目容易瞎编"这类问题）；不适合当成严谨的排行榜工具来引用绝对分数。

### License

按你实际情况填（MIT / Apache-2.0 等）。

---

## English

A lightweight evaluation scaffold for judging the tagging quality of **on-device image tagging models** — e.g. small-to-mid-size multimodal models served locally via tools like [llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server`.

(This toolkit originally also covered document tagging, but the document test set was small and iterating on it wasn't paying off — it's been dropped to keep the image path stable and focused.)

### Core idea

Open-vocabulary tagging has no single "correct answer" — the model decides its own wording. Scoring against a fixed tag list with exact string matching unfairly penalizes correct outputs phrased with synonyms or a different granularity. So the evaluation baseline isn't exact string matching, it's:

**A human-verified closed-set label library** (`tagging_test_samples/ground_truth_images.csv`) + **embedding-based semantic similarity scoring** ([score_semantic.py](eval_pipeline/scripts/score_semantic.py)): every image has a human-written answer key of what tags it should get, and the candidate model's output is matched against that answer key via cosine similarity to compute precision (how many of the tags it gave are actually right) and primary-tag coverage (did it get the most important thing — the main subject category — right).

An earlier version of this toolkit used an LLM-as-judge looking directly at the image, but that approach has a root problem: you need a separate human-calibration step to verify whether the judge's own judgment is trustworthy, and if the judge shares a blind spot with the model being tested (e.g. neither recognizes some arthropod), pairwise comparison alone won't catch it. Switching to a human-verified closed-set answer key makes the comparison itself a deterministic calculation (rerun the same data any number of times, same result) — the human effort doesn't disappear, it just moves from "spot-check whether the judge got it right, every run" to "get the answer key right once, then rerun freely."

[summarize.py](eval_pipeline/scripts/summarize.py) aggregates `score_semantic.py`'s output into `results/summary_report.md`, with a graded verdict (🟢 usable / 🟡 usable with conditions / 🔴 not recommended as-is) at the top.

### Project layout

```
.
├── eval_pipeline/
│   ├── core/                 # shared library code, imported by scripts/
│   │   ├── config.py         # model endpoints: candidate model (small)
│   │   ├── prompts.py        # tagging prompts (bilingual zh/en)
│   │   └── client.py         # shared helpers: client setup, image encoding, concurrency, perf capture, loose JSON parsing, retries
│   ├── scripts/               # CLI entrypoints, the scripts you actually run
│   │   ├── tag_images.py     # tag images
│   │   ├── score_semantic.py # semantic similarity scoring: precision/primary-tag coverage against the human-verified answer key
│   │   └── summarize.py      # aggregates into results/summary_report.md (incl. the graded verdict)
│   ├── Makefile               # shortcuts for common commands, see "Quickstart" below
│   ├── requirements.txt
│   ├── .env.example           # env var template, copy to .env and fill in real values
│   └── results/               # generated output (jsonl/csv/md), not checked into git
└── tagging_test_samples/     # test samples + the human-verified answer key (ground_truth_images.csv), see its own README
```

Every `scripts/*.py` is meant to be run from the `eval_pipeline/` directory as `python scripts/xxx.py` (not from inside `scripts/`) — each script adds `eval_pipeline/` to `sys.path` itself to find the `core` package, no install step or packaging needed.

### Quickstart

> Tired of typing out every flag? `eval_pipeline/` has a `Makefile` that bundles common flag combos into short commands — `make tag-images`, `make full`, `make smoke`, run `make help` for the full list. Both are equivalent; `make` just saves typing. For fine-grained flag control (or on Windows without `make`), use the full commands below.

1. Install dependencies

```bash
cd eval_pipeline
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

2. Unpack the test samples (images aren't checked in directly, they're zipped — see [tagging_test_samples/README.md](tagging_test_samples/README.md))

```bash
cd tagging_test_samples
unzip images.zip
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

- `-np` (parallel slots) divides the total `-c` (context) evenly across slots. E.g. `-c 65536 -np 16` leaves only 4096 tokens per request — easy to blow through with an image, causing truncation. Scripts default to sequential requests, so `-np 1` is enough — give the full context to a single request. **Raising `-np` only pays off if you also switch the client to concurrent requests (see "Running concurrently" below)** — otherwise the extra slots just sit idle.
- For reasoning-capable models, `--reasoning off` is recommended: tagging is a structured JSON output task that doesn't need a chain of thought, and leaving it on can burn the `max_tokens` budget on `reasoning_content` — occasionally the model even gets stuck in a repetitive generation loop.
- **Sampling parameters**: `llama-server` defaults to `--temp 0.80` and `--repeat-penalty 1.00` (i.e. no repetition penalty at all). A structured extraction task like tagging doesn't need much randomness, and the defaults measurably produce degenerate output — the same tag repeated until the token budget runs out (e.g. a tag list collapsing into `["bird","bird","bird",...]`), or in worse cases blowing past the JSON entirely and failing to parse. Dropping `--temp` to 0.2–0.4 and raising `--repeat-penalty` to 1.1–1.2 clearly helps.
- **Structured output (JSON Schema)**: if the server supports the OpenAI-compatible `response_format: {"type": "json_schema", ...}` (recent `llama.cpp` builds do, via `-j/--json-schema` or a `response_format` in the request), `tag_images.py` automatically attaches a schema (`TAGS_SCHEMA` in `eval_pipeline/core/client.py`) to requests sent to a local model. This grammar-constrains generation so the output is guaranteed valid JSON with a bounded tag array, and `uniqueItems: true` rules out the exact-duplicate-tag failure mode at the syntax level. It's only enabled when the endpoint is detected as local (`localhost`/`127.0.0.1`) — compatibility with cloud models hasn't been verified, so it's not forced there.
- If `HTTP_PROXY` / `HTTPS_PROXY` are set system-wide, make sure requests to your local server aren't accidentally routed through the proxy (`core/client.py` already forces `trust_env=False` for `localhost`/`127.0.0.1` clients to sidestep this).

4. Confirm the model endpoint config: `MODEL_PROFILES["small"]` (the candidate model) in `eval_pipeline/core/config.py` usually needs no changes, unless you've changed the port/engine name. The baseline is the human-verified answer key in `tagging_test_samples/ground_truth_images.csv`, which doesn't depend on any cloud model, so no extra API key is needed. To override the defaults, copy `.env.example` to `.env` and fill in real values (`.env` is already in `.gitignore`, never committed):

```bash
cd eval_pipeline
cp .env.example .env
```

5. Smoke-test on a small sample

```bash
python scripts/tag_images.py --profile small --limit 5
```

Check that every row in `results/images_tags_small.jsonl` has `error: null` and non-empty `tags`.

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

2. Add `--concurrency N` on the client side (`tag_images.py` supports it, defaulting to 1 = sequential, unchanged behavior), matching the server's `-np`:

```bash
python scripts/tag_images.py --profile small --concurrency 4
```

6. Run the full batch + semantic scoring + summary (entirely local and free — no paid cloud API calls needed)

```bash
python scripts/tag_images.py --profile small
python scripts/score_semantic.py       # precision/primary-tag coverage against ground_truth_images.csv
python scripts/summarize.py
```

(Equivalent shortcut: `make full`)

7. Read `results/summary_report.md` — the "Headline" section at the top gives a grade (🟢 usable / 🟡 usable with conditions / 🔴 not recommended as-is; thresholds live in `QUALITY_GATE` in `core/config.py`, tune them to your own risk tolerance). The report also lists the worst-scoring samples, with the specific unmatched tags (`unmatched_candidate_tags`/`unmatched_reference_tags`) — these two columns are there for you to spot-check: an unmatched tag doesn't necessarily mean the model was wrong, the answer key itself might just be incomplete; a quick look at a few rows tells you which.

### Chinese/English prompts and output

`--lang zh` (default) / `--lang en` switches which language `tag_images.py`'s prompt is written in — both versions have identical rules, just expressed in a different language, including an explicit instruction that tags must be output in the prompt's own language regardless of what language any text/branding visible in the image happens to be in — this avoids the model getting pulled off-language by English text/branding in the image itself (a real issue that showed up: 3 of 200 images spontaneously got fully English tags under the Chinese prompt, and those were exactly the images with prominent English branding). The answer key (`ground_truth_images.csv`) only maintains one Chinese tag set — there's no need for a separate English version just to test `--lang en`, since image tags are mostly common-vocabulary words (colors, animals, objects) where cross-lingual embedding similarity is reliably high (0.85+, e.g. "狗" vs "dog" scores 0.975); `score_semantic.py --lang` only picks which language's candidate-tag file to read (e.g. `images_tags_small_en.jsonl`), it doesn't change the matching baseline.

### What to do when you grow the test set

Adding new files to `tagging_test_samples/images/` is easy; the answer key is the part that takes work: new files also need a row added to `ground_truth_images.csv` with a human-verified `tags` list (JSON array, 5-15 tags, the first one always the main subject category, same rules as the tagging prompt — see `IMAGE_TAGGING_PROMPT_ZH` in `core/prompts.py`). This is the one piece of human cost this method can't avoid — what you save by dropping the LLM judge is having to re-verify the judge's reliability on every run; someone still has to write the answer key, just once instead of every time.

### Limitations (stated honestly — don't treat this as an authoritative benchmark)

- **The answer key is human-written once, and isn't guaranteed to be exhaustive**: it's hard to think of every reasonably-extractable detail in an image while writing tags for it — so a model can say something that's actually correct but get flagged as "unmatched" simply because the answer key didn't think to include it. The `unmatched_candidate_tags` column in `semantic_scores_images.csv` exists precisely for this: spot-check it periodically to tell whether the model is genuinely wrong or the answer key needs a patch. Compared to the old LLM-judge approach, this limitation is more manageable — the answer key is static, reviewable, and fixable, unlike a judge's runtime black-box decision.
- **Grading uses "primary-tag coverage", not full coverage**: of the 5-15 tags in each answer key, only the first one (the main subject category) is treated as a hard requirement — the rest (color/count/pose, etc.) are shown for reference only. Requiring an open-ended generation task to cover an entire answer key with nothing missed sets the bar too high for a free-form description task, and conflates "missed a minor detail" with "didn't recognize the subject at all."
- **Sample sizes are small**: the bundled test images are single-subject, ImageNet-style natural photos (200 of them) — treat conclusions as directional, not statistically definitive.
- **There's no off-the-shelf benchmark for truly open-vocabulary tagging**: established image-captioning hallucination metrics like CHAIR/POPE are conceptually close to this approach (both rely on a pre-annotated reference list to check against), but each is purpose-built for its own domain — this toolkit borrows CHAIR's counting formula, and that specific combination hasn't been validated by any published benchmark.
- **The grading thresholds are a subjective starting point**: `green_halluc`/`yellow_halluc` etc. in `QUALITY_GATE` are a neutral default, not an industry-standard answer — tune them to your own business tolerance for errors.

**Bottom line**: this is a methodologically sound, lightweight evaluation scaffold — good for quickly gauging an on-device model's image tagging ability and surfacing systematic weaknesses (e.g. "hallucinates on fine-grained categories"). It is not meant to be cited as a rigorous leaderboard tool with authoritative absolute scores.

### License

Fill in what applies (MIT / Apache-2.0 / etc).
