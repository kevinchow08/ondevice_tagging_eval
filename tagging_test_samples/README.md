# 测试样本 / Test Samples

[中文](#中文) | [English](#english)

---

## 中文

这批文件是给 [eval_pipeline](../eval_pipeline) 用的**测试输入 + 人工核实过的标准答案**。按照 [根目录 README](../README.md) 里说的评测思路，开放词表打标签任务不该用固定标准答案做字符串精确匹配评分——`ground_truth_images.csv`/`ground_truth_documents.csv` 里的标签是人工写好、核实过的答案，但比对用的是 embedding 语义相似度，不是精确字符串匹配，用同义词或不同粒度描述的正确输出不会被误伤。

图片和文档打包成 `images.zip` / `documents.zip`（不直接入库原始文件），使用前先解压：

```bash
unzip images.zip
unzip documents.zip
```

### images/（200 张）

来源：ImageNet 每类抽 1 张的公开图片集（[github.com/EliSchwartz/imagenet-sample-images](https://github.com/EliSchwartz/imagenet-sample-images)），从原始 1000 类里按跨度均匀抽了 200 张，覆盖动物、交通工具、乐器、日用品、建筑等大类，避免只集中在某一个领域。

- 文件名格式：`<WordNet ID>_<英文类别名>.JPEG`，类别名已经编码在文件名里，比如 `n01530575_brambling.JPEG` 就是一只燕雀。
- `ground_truth_images.csv`：`filename` / `wordnet_id` / `reference_label_en`（ImageNet 原始单一类别名，弱参考） / `tags`（人工核实过的完整标准答案，JSON数组，5~15个标签） / `confirmed` / `notes` 六列。`tags` 列是真正的评测基准，`reference_label_en` 只覆盖主体类别，覆盖不了颜色/数量/姿态这些其他维度。
- 已知问题：`n01608432_kite.JPEG` 这张图，`reference_label_en` 标的是"kite"（鸢，一种猛禽），但实际图片内容是玉兰花，没有任何鸟——这是原始图片集本身的图文不符，不是标注错误，`ground_truth_images.csv` 里 `tags` 列已经按实际图片内容标注，`reference_label_en` 这一列对这张图不可信，仅供你知晓。

用法：把这 200 张丢给被测模型跑一遍，输出标签用 `score_semantic.py` 跟 `ground_truth_images.csv` 的 `tags` 列做**语义相似度**比对（不要用精确字符串匹配）。

### documents/（18 份）

公开的"真实业务文档"数据集大多需要从 Hugging Face 或专门的数据托管站下载，考虑到环境限制，这 18 份是按常见文档类型手写的示例内容（中英文各半，公司名/人名均为虚构、延续同一套虚构的存储/半导体行业背景，专门用于测试；PDF 用 `reportlab` 生成，不是扫描件）：

| 文件 | 类型 |
|---|---|
| 01_增值税发票_invoice.pdf | 发票 |
| 02_resume_engineer.pdf | 简历（英文） |
| 03_合同节选_contract.pdf | 合同节选 |
| 04_meeting_minutes.pdf | 会议纪要（英文） |
| 05_产品规格书_spec_sheet.pdf | 产品规格书 |
| 06_news_article.pdf | 新闻报道（英文） |
| 07_采购订单_purchase_order.pdf | 采购订单 |
| 08_保密协议_nda.pdf | 保密协议/NDA（英文） |
| 09_银行对账单_bank_statement.pdf | 银行对账单 |
| 10_报销单_expense_report.pdf | 报销单 |
| 11_项目提案_project_proposal.pdf | 项目提案（英文） |
| 12_新闻通稿_press_release.pdf | 新闻通稿 |
| 13_质保条款_warranty_terms.pdf | 质保条款 |
| 14_测试报告_test_report.pdf | 测试报告（英文） |
| 15_录用offer_offer_letter.pdf | 录用通知（英文） |
| 16_装箱单_packing_list.pdf | 装箱单 |
| 17_绩效考核表_performance_review.pdf | 绩效考核表 |
| 18_专利摘要_patent_abstract.pdf | 专利摘要（英文） |

内容是真实句子、真实表格结构，不是随机字符堆砌，模型应该能从内容里读出"这是发票""这是合同"这类文档级标签，也能抽出其中的实体（公司名、金额、日期）。`ground_truth_documents.csv` 的标准答案是直接对着每份文档的原文写的（这批文档本身也是同一个人写的，内容确定，可信度高），列结构：`filename` / `document_type_cn` / `document_type_en` / `language` / `tags_zh` / `tags_en` / `confirmed` / `notes`。`tags_zh`/`tags_en` 是同一份标准答案的中英文两个版本（互为翻译），不是"文档原文语言配对应语言的答案、另一语言留空"——每份文档两个版本都有人工核实过的完整内容。这样设计是因为公司名/日期这类实体如果跨语言比对，embedding 相似度会明显下降（长复合短语尤其明显），`score_semantic.py --lang` 传哪个语言，就用哪一份，避免"模型其实答对了、只是用了另一种语言"被误判成漏打。`08_保密协议_nda.pdf` 是个分类边界情形——NDA 本质上也是一种合同，`document_type_cn` 标的是"保密协议"，不是"合同"，但两者不算互斥，供你在看结果时留意。

如果需要更大规模、更贴近真实业务场景的文档集，可以考虑：
1. 用浏览器手动去 Hugging Face 等站点下载现成的文档分类数据集（比如 RVL-CDIP）；
2. 自己准备一批脱敏后的真实文档（发票、合同扫描件等）。

扩充这两个数据集（新增图片/文档）的时候，别忘了同步给 `ground_truth_images.csv`/`ground_truth_documents.csv` 补上新文件对应的人工标注行，不然新样本会被 `score_semantic.py` 直接跳过（找不到标准答案）。

### 建议的测试流程

1. 跑被测的端侧模型，拿到它自己的标签（`tag_images.py`/`tag_documents.py`）。
2. 用 `score_semantic.py` 对照 `ground_truth_*.csv` 算 precision/recall。
3. 图片和文档分开看结果，不要混在一起算一个总分——这是模型两种完全不同的能力。

---

## English

These files are **test inputs plus a human-verified answer key** for [eval_pipeline](../eval_pipeline). Per the methodology in the [root README](../README.md), open-vocabulary tagging shouldn't be scored by exact string match against a fixed answer list — the tags in `ground_truth_images.csv`/`ground_truth_documents.csv` are a human-written, human-verified answer key, but matching uses embedding-based semantic similarity, not exact string matching, so correct outputs phrased with synonyms or a different granularity aren't penalized.

Images and documents ship as `images.zip` / `documents.zip` (raw files aren't checked in directly) — unzip before use:

```bash
unzip images.zip
unzip documents.zip
```

### images/ (200 files)

Source: one sample image per class from the public [ImageNet sample image set](https://github.com/EliSchwartz/imagenet-sample-images), evenly sampled to 200 images out of the original 1000 classes, spanning animals, vehicles, instruments, everyday objects, and buildings so no single domain dominates.

- Filename format: `<WordNet ID>_<English class name>.JPEG` — the class name is encoded right in the filename, e.g. `n01530575_brambling.JPEG` is a brambling (a finch).
- `ground_truth_images.csv`: six columns — `filename` / `wordnet_id` / `reference_label_en` (the original single-class ImageNet label, a weak reference) / `tags` (the human-verified full answer key, a JSON array of 5-15 tags) / `confirmed` / `notes`. `tags` is the actual evaluation baseline; `reference_label_en` only covers the main subject's category, not color/count/pose/other dimensions.
- Known issue: for `n01608432_kite.JPEG`, `reference_label_en` says "kite" (the bird of prey), but the image actually shows magnolia flowers with no bird at all — this is a mismatch in the source image set itself, not a labeling error. `tags` in `ground_truth_images.csv` reflects the actual image content; `reference_label_en` isn't trustworthy for this one file, noted here for awareness.

Usage: run the candidate model over these 200 images and compare its output tags against the `tags` column in `ground_truth_images.csv` using `score_semantic.py`'s **semantic similarity** matching (not exact string match).

### documents/ (18 files)

Public "real business document" datasets mostly require downloading from Hugging Face or a dedicated data hosting site; given environment constraints, these 18 are hand-written examples covering common document types (half Chinese, half English, all company/person names fictional and sharing one consistent fictional storage/semiconductor-industry backstory, purpose-built for testing; the PDFs are generated with `reportlab`, not scans):

| File | Type |
|---|---|
| 01_增值税发票_invoice.pdf | VAT invoice |
| 02_resume_engineer.pdf | Resume (English) |
| 03_合同节选_contract.pdf | Contract excerpt |
| 04_meeting_minutes.pdf | Meeting minutes (English) |
| 05_产品规格书_spec_sheet.pdf | Product spec sheet |
| 06_news_article.pdf | News article (English) |
| 07_采购订单_purchase_order.pdf | Purchase order |
| 08_保密协议_nda.pdf | Non-disclosure agreement (English) |
| 09_银行对账单_bank_statement.pdf | Bank statement |
| 10_报销单_expense_report.pdf | Expense report |
| 11_项目提案_project_proposal.pdf | Project proposal (English) |
| 12_新闻通稿_press_release.pdf | Press release |
| 13_质保条款_warranty_terms.pdf | Warranty terms |
| 14_测试报告_test_report.pdf | Test report (English) |
| 15_录用offer_offer_letter.pdf | Offer letter (English) |
| 16_装箱单_packing_list.pdf | Packing list |
| 17_绩效考核表_performance_review.pdf | Performance review |
| 18_专利摘要_patent_abstract.pdf | Patent abstract (English) |

Content is real sentences and real table structures, not random characters — a model should be able to read document-level tags ("this is an invoice", "this is a contract") from the content, as well as extract entities (company names, amounts, dates). The answer key in `ground_truth_documents.csv` was written directly against each document's source text (these documents were authored by one person, so the content is deterministic and the answer key is high-confidence); columns: `filename` / `document_type_cn` / `document_type_en` / `language` / `tags_zh` / `tags_en` / `confirmed` / `notes`. `tags_zh`/`tags_en` are two full translations of the same answer key, not "native-language answer filled in, the other left blank" — every document has a complete, human-verified answer key in both languages. This exists because cross-lingual embedding similarity for entities like company names and dates drops noticeably (especially for longer compound phrases); `score_semantic.py --lang` picks whichever column matches, so a model that got the right answer in a different language doesn't get scored as if it missed it. `08_保密协议_nda.pdf` sits on a category boundary — an NDA is technically a kind of contract — `document_type_cn` is labeled "保密协议" (NDA), not "合同" (contract), though the two aren't mutually exclusive; worth keeping in mind when reading results.

For a larger, more realistic document set, consider:
1. Manually downloading an existing document classification dataset (e.g. RVL-CDIP) from a site like Hugging Face via a browser;
2. Supplying your own de-identified real documents (invoices, scanned contracts, etc).

When you grow either dataset (add new images/documents), remember to add a matching human-annotated row to `ground_truth_images.csv`/`ground_truth_documents.csv` — otherwise `score_semantic.py` will just skip the new files (no answer key found).

### Suggested evaluation flow

1. Run the candidate on-device model to get its own tags (`tag_images.py`/`tag_documents.py`).
2. Run `score_semantic.py` to compute precision/recall against `ground_truth_*.csv`.
3. Look at images and documents separately — don't blend them into one overall score, since they exercise completely different model capabilities.
