# -*- coding: utf-8 -*-
"""各脚本共用的小工具：建client、读图片、宽松解析模型输出的JSON、失败重试。"""
import base64
import json
import re
import time

import httpx
from openai import OpenAI

import config


def get_client(profile: dict) -> OpenAI:
    base_url = profile["base_url"]
    is_local = any(host in base_url for host in ("localhost", "127.0.0.1"))
    if is_local:
        # 本地服务不能被系统代理（HTTP_PROXY/HTTPS_PROXY）接管，否则请求会被转发到代理端口
        # 卡住不返回（表现为"跑不动/很慢"）。trust_env=False 让这个 client 完全忽略代理环境变量。
        http_client = httpx.Client(trust_env=False)
        return OpenAI(base_url=base_url, api_key=profile["api_key"], http_client=http_client)
    return OpenAI(base_url=base_url, api_key=profile["api_key"])


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
