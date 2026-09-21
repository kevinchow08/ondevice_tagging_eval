# -*- coding: utf-8 -*-
"""
给 tagging_test_samples/documents 里的 PDF 跑打标签。
先用 pdfplumber 抽文本，再把文本喂给模型（如果你的模型是直接吃文档图片而不是文本，
把 extract_text() 换成"把PDF转成图片再走 tag_images.py 那套 image_url 流程"即可）。

用法：
  python tag_documents.py --profile small
  python tag_documents.py --profile reference
  python tag_documents.py --profile small --concurrency 4   # 并发跑，见README

输出：results/docs_tags_<profile>.jsonl
"""
import argparse
import csv
import json
import os

import pdfplumber

import config
from prompts import get_prompt
from tagger_core import (
    DOCUMENT_TAGS_SCHEMA,
    call_with_retry,
    get_client,
    is_local,
    json_schema_format,
    parse_json_loose,
    run_concurrent,
)

MAX_CHARS = 4000  # 文本太长会超模型上下文或拖慢速度，超过这个长度就截断


def extract_text(pdf_path: str) -> str:
    parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            parts.append(page.extract_text() or "")
    return "\n".join(parts)[:MAX_CHARS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=list(config.MODEL_PROFILES.keys()), required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--concurrency", type=int, default=1,
        help="并发请求数，默认1（顺序执行）。加大之前记得服务端 -np/-c 也要同步调大，见README",
    )
    ap.add_argument(
        "--lang", choices=["zh", "en"], default="zh", help="prompt 语言，默认中文",
    )
    args = ap.parse_args()

    profile = config.MODEL_PROFILES[args.profile]
    client = get_client(profile)
    prompt_template = get_prompt("document_tagging", args.lang)
    response_format = (
        json_schema_format("document_tags", DOCUMENT_TAGS_SCHEMA)
        if is_local(profile["base_url"])
        else None
    )

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    lang_suffix = "" if args.lang == "zh" else f"_{args.lang}"
    out_path = os.path.join(config.RESULTS_DIR, f"docs_tags_{args.profile}{lang_suffix}.jsonl")

    with open(config.DOCS_MANIFEST, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]

    def process_row(row):
        doc_path = os.path.join(config.DOCS_DIR, row["filename"])
        text = extract_text(doc_path)
        doc_type, tags, raw, error = "", [], "", None
        try:
            resp = call_with_retry(
                client,
                model=profile["model"],
                messages=[
                    {"role": "user", "content": prompt_template.format(document_text=text)}
                ],
                max_tokens=500,
                extra_body=profile.get("extra_body"),
                response_format=response_format,
            )
            raw = resp.choices[0].message.content
            parsed = parse_json_loose(raw)
            doc_type = parsed.get("document_type", "")
            tags = parsed.get("tags", [])
        except Exception as e:  # noqa: BLE001
            error = str(e)

        return {
            "filename": row["filename"],
            "reference_type_cn": row["document_type_cn"],
            "model_profile": args.profile,
            "document_type": doc_type,
            "tags": tags,
            "raw_response": raw,
            "error": error,
            "extracted_text_used": text,
        }

    results = run_concurrent(
        rows, process_row, args.concurrency, desc=f"tagging documents [{args.profile}]"
    )
    n_error = sum(1 for r in results if r["error"])

    with open(out_path, "w", encoding="utf-8") as out:
        for r in results:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"完成，共 {len(rows)} 条，{n_error} 条出错。写入 {out_path}")


if __name__ == "__main__":
    main()
