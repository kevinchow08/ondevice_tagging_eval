# 常用命令组合的简写。都在 eval_pipeline/ 目录下跑：make <target>
# 想覆盖参数就在后面加 VAR=值，比如：
#   make tag-images LIMIT=20
#   make tag-images CONCURRENCY=4
# 不传就用下面这些默认值。

PROFILE ?= small
LIMIT ?=
CONCURRENCY ?= 1
PROMPT_LANG ?= zh

LIMIT_FLAG = $(if $(LIMIT),--limit $(LIMIT))

.PHONY: setup unzip smoke tag-images semantic-images summarize full clean help

help:
	@echo "常用命令："
	@echo "  make setup                装依赖（建venv + pip install）"
	@echo "  make unzip                解压 tagging_test_samples 里的图片"
	@echo "  make smoke                小样本冒烟测试（5张图，本地免费）"
	@echo "  make tag-images           被测(small)模型给图片打标签（默认全量，可加 LIMIT=N，本地免费）"
	@echo "  make semantic-images      语义相似度打分，对照 ground_truth_images.csv（本地免费，前提：先跑过 tag-images）"
	@echo "  make summarize            汇总成 results/summary_report.md（分级判定放在报告最前面）"
	@echo "  make full                 本地全流程：打标签 + 语义打分 + 汇总，全程不需要调用付费云端API"
	@echo "  make clean                清空 results/ 下生成的文件（tagging_test_samples/ground_truth_images.csv 不在这里，不会被清掉）"
	@echo ""
	@echo "可覆盖变量：PROFILE(默认small，core/config.py 的 MODEL_PROFILES 里目前只有这一个)"
	@echo "           LIMIT(默认不限) CONCURRENCY(默认1) PROMPT_LANG(默认zh)"

setup:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

unzip:
	cd ../tagging_test_samples && unzip -n images.zip

smoke:
	$(MAKE) tag-images LIMIT=5

tag-images:
	python scripts/tag_images.py --profile $(PROFILE) $(LIMIT_FLAG) --concurrency $(CONCURRENCY) --lang $(PROMPT_LANG)

semantic-images:
	python scripts/score_semantic.py --lang $(PROMPT_LANG)

summarize:
	python scripts/summarize.py

full: tag-images semantic-images summarize

clean:
	rm -f results/*.jsonl results/*.md results/*.csv
