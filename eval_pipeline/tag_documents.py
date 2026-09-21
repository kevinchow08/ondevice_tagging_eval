# -*- coding: utf-8 -*-
"""
给 tagging_test_samples/documents 里的 PDF 跑打标签。
先用 pdfplumber 抽文本，再把文本喂给模型（如果你的模型是直接吃文档图片而不是文本，
把 extract_text() 换成"把PDF转成图片再走 tag_images.py 那套 image_url 流程"即可）。

用法：
  python tag_documents.py --profile small
  python tag_documents.py --profile reference

输出：results/docs_tags_<profile>.jsonl
"""
import argparse
import csv
import json
import os

import pdfplumber
from tqdm import tqdm

import config
from prompts import DOCUMENT_TAGGING_PROMPT
from tagger_core import call_with_retry, get_client, parse_json_loose

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
    args = ap.parse_args()

    profile = config.MODEL_PROFILES[args.profile]
    client = get_client(profile)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(config.RESULTS_DIR, f"docs_tags_{args.profile}.jsonl")

    with open(config.DOCS_MANIFEST, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]

    n_error = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for row in tqdm(rows, desc=f"tagging documents [{args.profile}]"):
            doc_path = os.path.join(config.DOCS_DIR, row["filename"])
            text = extract_text(doc_path)
            doc_type, tags, raw, error = "", [], "", None
            try:
                resp = call_with_retry(
                    client,
                    model=profile["model"],
                    messages=[
                        {"role": "user", "content": DOCUMENT_TAGGING_PROMPT.format(document_text=text)}
                    ],
                    max_tokens=500,
                    extra_body=profile.get("extra_body"),
                )
                raw = resp.choices[0].message.content
                parsed = parse_json_loose(raw)
                doc_type = parsed.get("document_type", "")
                tags = parsed.get("tags", [])
            except Exception as e:  # noqa: BLE001
                error = str(e)
                n_error += 1

            out.write(
                json.dumps(
                    {
                        "filename": row["filename"],
                        "reference_type_cn": row["document_type_cn"],
                        "model_profile": args.profile,
                        "document_type": doc_type,
                        "tags": tags,
                        "raw_response": raw,
                        "error": error,
                        "extracted_text_used": text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"完成，共 {len(rows)} 条，{n_error} 条出错。写入 {out_path}")


if __name__ == "__main__":
    main()
