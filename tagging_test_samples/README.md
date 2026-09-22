# 测试样本 / Test Samples

[中文](#中文) | [English](#english)

---

## 中文

这批文件是给 [eval_pipeline](../eval_pipeline) 用的**测试输入**，不是"数据集"——没有附带标准答案文件。按照 [根目录 README](../README.md) 里说的评测思路，开放词表打标签任务不该用固定标准答案做字符串精确匹配评分，`*_manifest.csv` 里的"参考标签"只是弱参考，方便你自己人工核对，不是用来做 exact-match 打分的 ground truth。

图片和文档打包成 `images.zip` / `documents.zip`（不直接入库原始文件），使用前先解压：

```bash
unzip images.zip
unzip documents.zip
```

### images/（200 张）

来源：ImageNet 每类抽 1 张的公开图片集（[github.com/EliSchwartz/imagenet-sample-images](https://github.com/EliSchwartz/imagenet-sample-images)），从原始 1000 类里按跨度均匀抽了 200 张，覆盖动物、交通工具、乐器、日用品、建筑等大类，避免只集中在某一个领域。

- 文件名格式：`<WordNet ID>_<英文类别名>.JPEG`，类别名已经编码在文件名里，比如 `n01530575_brambling.JPEG` 就是一只燕雀。
- `images_manifest.csv`：`filename` / `wordnet_id` / `reference_label_en` 三列，方便写脚本批量核对。

用法：把这 200 张丢给被测模型跑一遍，输出标签跟文件名里的类别做**语义相似度**比对（不要用精确字符串匹配），或者挑几十张人工过一遍眼，看有没有明显瞎标、漏标。

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

内容是真实句子、真实表格结构，不是随机字符堆砌，模型应该能从内容里读出"这是发票""这是合同"这类文档级标签，也能抽出其中的实体（公司名、金额、日期）。其中 `08_保密协议_nda.pdf` 故意留了个分类模糊的边界情形——NDA 本质上也是一种合同，模型判成"合同"不算错，这个可以用来观察裁判怎么处理这类模糊分类。

如果需要更大规模、更贴近真实业务场景的文档集，可以考虑：
1. 用浏览器手动去 Hugging Face 等站点下载现成的文档分类数据集（比如 RVL-CDIP）；
2. 自己准备一批脱敏后的真实文档（发票、合同扫描件等）。

### 建议的测试流程

1. 用一个更强的模型对这 218 个样本先跑一遍（或者直接用 [llm_judge.py](../eval_pipeline/scripts/llm_judge.py) 的裁判评审路径，不需要单独生成参考标签）。
2. 跑被测的端侧模型，拿到它自己的标签。
3. 两边对比：语义相似度看覆盖率，或者把"原始内容 + 端侧模型标签"丢给强模型当裁判，问它有没有瞎编、有没有漏标、准不准。
4. 图片和文档分开看结果，不要混在一起算一个总分——这是模型两种完全不同的能力。

---

## English

These files are **test inputs** for [eval_pipeline](../eval_pipeline), not a labeled dataset — there is no ground-truth answer key. Per the methodology in the [root README](../README.md), open-vocabulary tagging shouldn't be scored by exact string match against a fixed answer list; the "reference labels" in `*_manifest.csv` are a loose reference for manual spot-checking, not exact-match ground truth.

Images and documents ship as `images.zip` / `documents.zip` (raw files aren't checked in directly) — unzip before use:

```bash
unzip images.zip
unzip documents.zip
```

### images/ (200 files)

Source: one sample image per class from the public [ImageNet sample image set](https://github.com/EliSchwartz/imagenet-sample-images), evenly sampled to 200 images out of the original 1000 classes, spanning animals, vehicles, instruments, everyday objects, and buildings so no single domain dominates.

- Filename format: `<WordNet ID>_<English class name>.JPEG` — the class name is encoded right in the filename, e.g. `n01530575_brambling.JPEG` is a brambling (a finch).
- `images_manifest.csv`: three columns, `filename` / `wordnet_id` / `reference_label_en`, for scripting bulk checks.

Usage: run the candidate model over these 200 images and compare its output tags against the filename's class via **semantic similarity** (not exact string match), or manually eyeball a few dozen for obvious hallucination/omission.

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

Content is real sentences and real table structures, not random characters — a model should be able to read document-level tags ("this is an invoice", "this is a contract") from the content, as well as extract entities (company names, amounts, dates). `08_保密协议_nda.pdf` deliberately sits on a fuzzy category boundary — an NDA is technically a kind of contract, so a model calling it "contract" isn't really wrong; it's a useful case for observing how the judge handles ambiguous classification.

For a larger, more realistic document set, consider:
1. Manually downloading an existing document classification dataset (e.g. RVL-CDIP) from a site like Hugging Face via a browser;
2. Supplying your own de-identified real documents (invoices, scanned contracts, etc).

### Suggested evaluation flow

1. Run a stronger model over these 218 samples once (or just use the [llm_judge.py](../eval_pipeline/scripts/llm_judge.py) judge path directly, which doesn't need a separately-generated reference tag set).
2. Run the candidate on-device model to get its own tags.
3. Compare the two: check coverage via semantic similarity, or hand "original content + candidate tags" to the stronger model as a judge and ask whether it hallucinated, missed anything, or got it right.
4. Look at images and documents separately — don't blend them into one overall score, since they exercise completely different model capabilities.
