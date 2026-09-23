# 测试样本 / Test Samples

[中文](#中文) | [English](#english)

---

## 中文

这批文件是给 [eval_pipeline](../eval_pipeline) 用的**测试输入 + 人工核实过的标准答案**。按照 [根目录 README](../README.md) 里说的评测思路，开放词表打标签任务不该用固定标准答案做字符串精确匹配评分——`ground_truth_images.csv` 里的标签是人工写好、核实过的答案，但比对用的是 embedding 语义相似度，不是精确字符串匹配，用同义词或不同粒度描述的正确输出不会被误伤。

图片打包成 `images.zip`（不直接入库原始文件），使用前先解压：

```bash
unzip images.zip
```

### images/（200 张）

来源：ImageNet 每类抽 1 张的公开图片集（[github.com/EliSchwartz/imagenet-sample-images](https://github.com/EliSchwartz/imagenet-sample-images)），从原始 1000 类里按跨度均匀抽了 200 张，覆盖动物、交通工具、乐器、日用品、建筑等大类，避免只集中在某一个领域。

- 文件名格式：`<WordNet ID>_<英文类别名>.JPEG`，类别名已经编码在文件名里，比如 `n01530575_brambling.JPEG` 就是一只燕雀。
- `ground_truth_images.csv`：`filename` / `wordnet_id` / `reference_label_en`（ImageNet 原始单一类别名，弱参考） / `tags`（人工核实过的完整标准答案，JSON数组，5~15个标签，**第一个固定是主体大类**） / `confirmed` / `notes` 六列。`tags` 列是真正的评测基准，`reference_label_en` 只覆盖主体类别，覆盖不了颜色/数量/姿态这些其他维度。
- 已知问题：`n01608432_kite.JPEG` 这张图，`reference_label_en` 标的是"kite"（鸢，一种猛禽），但实际图片内容是玉兰花，没有任何鸟——这是原始图片集本身的图文不符，不是标注错误，`ground_truth_images.csv` 里 `tags` 列已经按实际图片内容标注，`reference_label_en` 这一列对这张图不可信，仅供你知晓。

用法：把这 200 张丢给被测模型跑一遍，输出标签用 `score_semantic.py` 跟 `ground_truth_images.csv` 的 `tags` 列做**语义相似度**比对（不要用精确字符串匹配）。

扩充这个数据集（新增图片）的时候，别忘了同步给 `ground_truth_images.csv` 补上新文件对应的人工标注行，不然新样本会被 `score_semantic.py` 直接跳过（找不到标准答案）。

### 建议的测试流程

1. 跑被测的端侧模型，拿到它自己的标签（`tag_images.py`）。
2. 用 `score_semantic.py` 对照 `ground_truth_images.csv` 算 precision/主体标签覆盖率。

---

## English

These files are **test inputs plus a human-verified answer key** for [eval_pipeline](../eval_pipeline). Per the methodology in the [root README](../README.md), open-vocabulary tagging shouldn't be scored by exact string match against a fixed answer list — the tags in `ground_truth_images.csv` are a human-written, human-verified answer key, but matching uses embedding-based semantic similarity, not exact string matching, so correct outputs phrased with synonyms or a different granularity aren't penalized.

Images ship as `images.zip` (raw files aren't checked in directly) — unzip before use:

```bash
unzip images.zip
```

### images/ (200 files)

Source: one sample image per class from the public [ImageNet sample image set](https://github.com/EliSchwartz/imagenet-sample-images), evenly sampled to 200 images out of the original 1000 classes, spanning animals, vehicles, instruments, everyday objects, and buildings so no single domain dominates.

- Filename format: `<WordNet ID>_<English class name>.JPEG` — the class name is encoded right in the filename, e.g. `n01530575_brambling.JPEG` is a brambling (a finch).
- `ground_truth_images.csv`: six columns — `filename` / `wordnet_id` / `reference_label_en` (the original single-class ImageNet label, a weak reference) / `tags` (the human-verified full answer key, a JSON array of 5-15 tags, **the first one always the main subject category**) / `confirmed` / `notes`. `tags` is the actual evaluation baseline; `reference_label_en` only covers the main subject's category, not color/count/pose/other dimensions.
- Known issue: for `n01608432_kite.JPEG`, `reference_label_en` says "kite" (the bird of prey), but the image actually shows magnolia flowers with no bird at all — this is a mismatch in the source image set itself, not a labeling error. `tags` in `ground_truth_images.csv` reflects the actual image content; `reference_label_en` isn't trustworthy for this one file, noted here for awareness.

Usage: run the candidate model over these 200 images and compare its output tags against the `tags` column in `ground_truth_images.csv` using `score_semantic.py`'s **semantic similarity** matching (not exact string match).

When you grow this dataset (add new images), remember to add a matching human-annotated row to `ground_truth_images.csv` — otherwise `score_semantic.py` will just skip the new files (no answer key found).

### Suggested evaluation flow

1. Run the candidate on-device model to get its own tags (`tag_images.py`).
2. Run `score_semantic.py` to compute precision/primary-tag coverage against `ground_truth_images.csv`.
