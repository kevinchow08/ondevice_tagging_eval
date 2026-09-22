# -*- coding: utf-8 -*-
"""
统一配置：模型端点、文件路径、评测参数。

云端/裁判模型的 base_url、api_key 从环境变量读取，不写死在这个文件里，
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
    # 用来生成参考标签、当裁判的更强模型：换成你能调用的任意更强模型
    # （云端 API，如 OpenAI/GPT-4o，或者你自己本地跑的更大 VL 模型）
    "reference": {
        "base_url": os.environ.get(
            "REFERENCE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ),
        "api_key": os.environ.get("REFERENCE_API_KEY", ""),
        "model": os.environ.get("REFERENCE_MODEL_NAME", "qwen3.8-max"),
        # qwen3.8-max 默认开思考模式，裁判任务只要结构化 JSON，关掉更快更省钱也更稳
        "extra_body": {"enable_thinking": False},
        # response_format(JSON Schema结构化输出)要不要对这个云端模型也用上，默认关闭。
        # 这个功能能不能生效完全取决于对面服务商的推理引擎支不支持，不是所有OpenAI兼容接口都支持，
        # 不支持的话请求可能直接报错，换了别的云端模型别想当然打开，自己测试确认过能用再改成 True。
        "supports_json_schema": False,
    },
}

if not MODEL_PROFILES["reference"]["api_key"]:
    import warnings

    warnings.warn(
        "REFERENCE_API_KEY 没设置，reference profile（裁判/参考模型）调用会失败。"
        "在 eval_pipeline/.env 里设置，参考 .env.example。"
    )

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

# summarize.py"分级判定"用的阈值：瞎编率(CHAIR_s)、漏打率哪个落在更差的档位，就用哪个定颜色
# （短板原则）。这是写死但可调的规则，不是AI临场判断——具体数字要按你自己的业务风险容忍度定，
# 这里给的只是一个中性起点，不代表"标准答案"，改这两组数字就能调整分级的松紧。
QUALITY_GATE = {
    "green_halluc": 0.10, "green_missing": 0.30,
    "yellow_halluc": 0.25, "yellow_missing": 0.60,
}

# 人工校准要满足这两个条件，QUALITY_GATE 算出来的颜色判定才算"可信"：
# 1. 填了判断的条数至少达到 CALIBRATION_MIN_N——填太少（比如1、2条）算出来的"一致率"本身
#    没有统计意义，不该拿来给判定背书；
# 2. 一致率至少达到 CALIBRATION_TRUST_THRESHOLD——低于这个，说明裁判本身系统性不可靠，
#    这时候瞎编率/漏打率这些数字很可能是裁判编出来的，不能直接采信，报告会明确标注不采信，
#    不会因为"反正算过校准了"就摆出一副confident的样子。
CALIBRATION_MIN_N = 10
CALIBRATION_TRUST_THRESHOLD = 0.7
