# 常用命令组合的简写。都在 eval_pipeline/ 目录下跑：make <target>
# 想覆盖参数就在后面加 VAR=值，比如：
#   make tag-images LIMIT=20
#   make judge-images LIMIT=20 PROMPT_LANG=en
#   make tag-images PROFILE=reference CONCURRENCY=4
# 不传就用下面这些默认值。

PROFILE ?= small
LIMIT ?=
CONCURRENCY ?= 1
PROMPT_LANG ?= zh

LIMIT_FLAG = $(if $(LIMIT),--limit $(LIMIT))

.PHONY: setup unzip smoke tag-images tag-documents judge-images judge-documents \
        semantic-images semantic-documents summarize full clean help

help:
	@echo "常用命令："
	@echo "  make setup              装依赖（建venv + pip install）"
	@echo "  make unzip              解压 tagging_test_samples 里的图片/文档"
	@echo "  make smoke              小样本冒烟测试（5张图+2份文档，本地免费）"
	@echo "  make tag-images         本地模型给图片打标签（默认全量，可加 LIMIT=N）"
	@echo "  make tag-documents      本地模型给文档打标签"
	@echo "  make judge-images       LLM裁判评审图片（默认全量，建议先 LIMIT=20 抽查，付费）"
	@echo "  make judge-documents    LLM裁判评审文档（付费）"
	@echo "  make semantic-images    语义相似度打分-图片（需要先跑过 reference profile 的 tag-images）"
	@echo "  make semantic-documents 语义相似度打分-文档"
	@echo "  make summarize          汇总成 results/summary_report.md"
	@echo "  make full               本地打标签(图片+文档) + 汇总，不含付费裁判步骤"
	@echo "  make clean              清空 results/ 下生成的文件"
	@echo ""
	@echo "可覆盖变量：PROFILE(默认small) LIMIT(默认不限) CONCURRENCY(默认1) PROMPT_LANG(默认zh)"

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

judge-images:
	python scripts/llm_judge.py --kind images $(LIMIT_FLAG) --concurrency $(CONCURRENCY) --lang $(PROMPT_LANG)

judge-documents:
	python scripts/llm_judge.py --kind documents $(LIMIT_FLAG) --concurrency $(CONCURRENCY) --lang $(PROMPT_LANG)

semantic-images:
	python scripts/score_semantic.py --kind images

semantic-documents:
	python scripts/score_semantic.py --kind documents

summarize:
	python scripts/summarize.py

full: tag-images tag-documents summarize

clean:
	rm -f results/*.jsonl results/*.csv results/*.md
