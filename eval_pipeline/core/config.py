# -*- coding: utf-8 -*-
"""
统一配置：模型端点、文件路径、评测参数。

被测模型的 base_url、api_key 从环境变量读取，不写死在这个文件里，
这样 config.py 可以放心提交进 git，不会泄露 key。本地开发时把真实值放进
eval_pipeline/.env（已加入 .gitignore，不会被提交），参考 .env.example。
"""
import os

from dotenv import load_dotenv

load_dotenv()  # 加载同目录下的 .env；线上/CI 可以直接设真实环境变量，不依赖这个文件

MODEL_PROFILES = {
    # 被测的端侧模型：Qwen3.6-2B 或你自己的引擎，只要暴露 OpenAI 兼容的 /v1/chat/completions 接口即可
    # （llama.cpp server、vLLM、Ollama、LM Studio 默认都支持这个协议）
    "small": {
        "base_url": os.environ.get("SMALL_MODEL_BASE_URL", "http://localhost:8080"),
        "api_key": os.environ.get("SMALL_MODEL_API_KEY", "sk-no-key-required"),
        "model": os.environ.get("SMALL_MODEL_NAME", "qwen3.5-2b"),
    },
}

# 对应 tagging_test_samples 包解压后的路径，按需改成你实际存放的位置。
# ground_truth_images.csv 既是跑哪些文件的清单，也是人工核实过的标准答案（tags 列）——
# 不再需要单独的 manifest.csv + 裁判模型比对，这份文件本身就是评测的基准。
IMAGES_DIR = "../tagging_test_samples/images"
IMAGES_GROUND_TRUTH = "../tagging_test_samples/ground_truth_images.csv"

RESULTS_DIR = "results"

# 语义相似度打分阈值：模型标签和参考标签的 cosine similarity 超过这个值才算"命中"
SEMANTIC_MATCH_THRESHOLD = 0.55

REQUEST_TIMEOUT_SECONDS = 60
MAX_RETRIES = 2

# summarize.py"分级判定"用的阈值：瞎编率、漏打率（都是对着人工闭集标签库算出来的，
# 不再是 LLM 裁判的主观判断）哪个落在更差的档位，就用哪个定颜色（短板原则）。
# 这是写死但可调的规则，不是AI临场判断——具体数字要按你自己的业务风险容忍度定，
# 这里给的只是一个中性起点，不代表"标准答案"，改这两组数字就能调整分级的松紧。
QUALITY_GATE = {
    "green_halluc": 0.10, "green_missing": 0.30,
    "yellow_halluc": 0.25, "yellow_missing": 0.60,
}
