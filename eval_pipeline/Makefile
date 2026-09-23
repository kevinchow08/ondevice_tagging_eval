# 常用命令组合的简写。都在 eval_pipeline/ 目录下跑：make <target>
# 想覆盖参数就在后面加 VAR=值，比如：
#   make tag-images LIMIT=20
#   make tag-images PROFILE=reference CONCURRENCY=4
# 不传就用下面这些默认值。

PROFILE ?= small
LIMIT ?=
CONCURRENCY ?= 1
PROMPT_LANG ?= zh

LIMIT_FLAG = $(if $(LIMIT),--limit $(LIMIT))

.PHONY: setup unzip smoke tag-images tag-documents tag-images-reference tag-documents-reference \
        semantic-images semantic-documents summarize full clean help

help:
	@echo "常用命令："
	@echo "  make setup                装依赖（建venv + pip install）"
	@echo "  make unzip                解压 tagging_test_samples 里的图片/文档"
	@echo "  make smoke                小样本冒烟测试（5张图+2份文档，本地免费）"
	@echo "  make tag-images           被测(small)模型给图片打标签（默认全量，可加 LIMIT=N，本地免费）"
	@echo "  make tag-documents        被测(small)模型给文档打标签"
	@echo "  make semantic-images      语义相似度打分-图片，对照 ground_truth_images.csv（本地免费，前提：先跑过 tag-images）"
	@echo "  make semantic-documents   语义相似度打分-文档，对照 ground_truth_documents.csv（前提：先跑过 tag-documents）"
	@echo "  make summarize            汇总成 results/summary_report.md（分级判定放在报告最前面）"
	@echo "  make full                 本地全流程：打标签(图片+文档) + 语义打分 + 汇总，全程不需要调用付费云端API"
	@echo "  make tag-images-reference    可选：参考(reference)云端模型独立给图片打一遍标签（付费，仅用于你想额外对比"
	@echo "                                云端模型自己的输出时手动参考，不是评测主流程必需——评测基准是"
	@echo "                                tagging_test_samples/ground_truth_*.csv 这份人工闭集标签库，不是这个）"
	@echo "  make tag-documents-reference 同上，文档版"
	@echo "  make clean                清空 results/ 下生成的文件（tagging_test_samples/ground_truth_*.csv 不在这里，不会被清掉）"
	@echo ""
	@echo "可覆盖变量：PROFILE(默认small，tag-images/tag-documents可以传 PROFILE=reference 效果等价于上面两个专用命令)"
	@echo "           LIMIT(默认不限) CONCURRENCY(默认1) PROMPT_LANG(默认zh)"

setup:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

unzip:
	cd ../tagging_test_samples && unzip -n images.zip && unzip -n documents.zip

smoke:
	$(MAKE) tag-images LIMIT=5
	$(MAKE) tag-documents LIMIT=2

tag-images:
	python scripts/tag_images.py --profile $(PROFILE) $(LIMIT_FLAG) --concurrency $(CONCURRENCY) --lang $(PROMPT_LANG)

tag-documents:
	python scripts/tag_documents.py --profile $(PROFILE) $(LIMIT_FLAG) --concurrency $(CONCURRENCY) --lang $(PROMPT_LANG)

tag-images-reference:
	python scripts/tag_images.py --profile reference $(LIMIT_FLAG) --concurrency $(CONCURRENCY) --lang $(PROMPT_LANG)

tag-documents-reference:
	python scripts/tag_documents.py --profile reference $(LIMIT_FLAG) --concurrency $(CONCURRENCY) --lang $(PROMPT_LANG)

semantic-images:
	python scripts/score_semantic.py --kind images

semantic-documents:
	python scripts/score_semantic.py --kind documents

summarize:
	python scripts/summarize.py

full: tag-images tag-documents semantic-images semantic-documents summarize

clean:
	rm -f results/*.jsonl results/*.md results/*.csv
