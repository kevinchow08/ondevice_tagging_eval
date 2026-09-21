# -*- coding: utf-8 -*-
"""各脚本共用的小工具：建client、读图片、宽松解析模型输出的JSON、失败重试。"""
import base64
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
from openai import OpenAI
from tqdm import tqdm

import config


def is_local(base_url: str) -> bool:
    return any(host in base_url for host in ("localhost", "127.0.0.1"))


def get_client(profile: dict) -> OpenAI:
    base_url = profile["base_url"]
    if is_local(base_url):
        # 本地服务不能被系统代理（HTTP_PROXY/HTTPS_PROXY）接管，否则请求会被转发到代理端口
        # 卡住不返回（表现为"跑不动/很慢"）。trust_env=False 让这个 client 完全忽略代理环境变量。
        http_client = httpx.Client(trust_env=False)
        return OpenAI(base_url=base_url, api_key=profile["api_key"], http_client=http_client)
    return OpenAI(base_url=base_url, api_key=profile["api_key"])


def json_schema_format(name: str, schema: dict) -> dict:
    """构造 OpenAI 兼容的 response_format：让服务端按 JSON Schema 语法约束生成（llama.cpp 支持），
    从语法层面保证输出一定是合法 JSON、数组长度受控、且（配合 uniqueItems）不会重复同一个标签，
    比生成完之后再用 parse_json_loose 宽松兜底更彻底。目前只对本地(小)模型用，云端裁判模型的
    response_format 兼容性没验证过，不确定支持就不强行加，保留原来的宽松解析兜底。"""
    return {"type": "json_schema", "json_schema": {"name": name, "schema": schema, "strict": True}}


# minItems 故意比 prompt 里说的"5~15个"下限低一点：有些图片/文档本来就没那么多能确定的信息，
# 卡死5会逼模型硬凑近义重复的标签（比如"鸟类"+"鸟类摄影"）来凑数，适度放宽下限。
TAGS_SCHEMA = {
    "type": "object",
    "properties": {
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 15,
            "uniqueItems": True,
        }
    },
    "required": ["tags"],
    "additionalProperties": False,
}

DOCUMENT_TAGS_SCHEMA = {
    "type": "object",
    "properties": {
        "document_type": {"type": "string"},
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 15,
            "uniqueItems": True,
        },
    },
    "required": ["document_type", "tags"],
    "additionalProperties": False,
}


def encode_image_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def parse_json_loose(text: str) -> dict:
    """模型有时会在 JSON 前后加废话，或用```json包裹，这里做宽松提取再解析。"""
    if text is None:
        raise ValueError("模型没有返回任何内容")
    text = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"没有在输出里找到 JSON，原始输出前200字符：{text[:200]!r}")
    return json.loads(match.group(0))


def run_concurrent(rows, worker, max_workers: int, desc: str):
    """对 rows 逐条跑 worker(row)->dict，按 rows 原始顺序返回结果列表。
    max_workers<=1 时退化成普通顺序执行，跟不传并发参数之前的行为完全一样。
    worker 自己要兜住所有异常（返回一个带 error 字段的 dict），这里不重新抛出，
    避免一个样本出错就打断整批（跟原来单线程 try/except 的语义保持一致）。
    openai 的 client 基于 httpx，多线程共享同一个 client 实例发请求是安全的。"""
    if max_workers <= 1:
        return [worker(row) for row in tqdm(rows, desc=desc)]

    results = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker, row): i for i, row in enumerate(rows)}
        for future in tqdm(as_completed(futures), total=len(rows), desc=desc):
            results[futures[future]] = future.result()
    return results


def call_with_retry(client: OpenAI, **kwargs):
    last_err = None
    for attempt in range(config.MAX_RETRIES + 1):
        try:
            return client.chat.completions.create(
                timeout=config.REQUEST_TIMEOUT_SECONDS, **kwargs
            )
        except Exception as e:  # noqa: BLE001 - 这里就是要兜住所有网络/接口错误
            last_err = e
            if attempt < config.MAX_RETRIES:
                time.sleep(1.5 * (attempt + 1))
    raise last_err
