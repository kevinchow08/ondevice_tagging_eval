# -*- coding: utf-8 -*-
"""
统一配置：模型端点、文件路径、评测参数。
跑任何脚本之前，先把下面 MODEL_PROFILES 里两个 profile 的地址/模型名换成你自己的。
"""

MODEL_PROFILES = {
    # 被测的端侧模型：Qwen3.6-2B 或你自己的引擎，只要暴露 OpenAI 兼容的 /v1/chat/completions 接口即可
    # （llama.cpp server、vLLM、Ollama、LM Studio 默认都支持这个协议）
    "small": {
        "base_url": "http://localhost:8080",
        "api_key": "sk-no-key-required",   # 本地服务通常随便填一个非空字符串即可，不校验
        "model": "qwen3.5-2b",
    },
    # 用来生成参考标签、当裁判的更强模型：换成你能调用的任意更强模型
    # （云端 API，如 OpenAI/GPT-4o，或者你自己本地跑的更大 VL 模型）
    "reference": {
        "base_url": "xxx",
        "api_key": "xxx",
        "model": "qwen3.8-max",
        # qwen3.8-max 默认开思考模式，裁判任务只要结构化 JSON，关掉更快更省钱也更稳
        "extra_body": {"enable_thinking": False},
    },
}

# 对应 tagging_test_samples 包解压后的路径，按需改成你实际存放的位置
IMAGES_DIR = "../tagging_test_samples/images"
IMAGES_MANIFEST = "../tagging_test_samples/images_manifest.csv"
DOCS_DIR = "../tagging_test_samples/documents"
DOCS_MANIFEST = "../tagging_test_samples/documents_manifest.csv"

RESULTS_DIR = "results"

# 语义相似度打分阈值：模型标签和参考标签的 cosine similarity 超过这个值才算"命中"
SEMANTIC_MATCH_THRESHOLD = 0.55

REQUEST_TIMEOUT_SECONDS = 60
MAX_RETRIES = 2
