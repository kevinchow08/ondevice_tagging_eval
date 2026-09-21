# -*- coding: utf-8 -*-
"""
LLM-as-judge：用 config.py 里 reference profile 的模型，直接看着原图/原文，
评审 small 模型打出来的标签有没有瞎编、有没有漏打、准不准。
不需要先跑 reference 的 tag_images.py/tag_documents.py，这条路是独立的。

用法（在 eval_pipeline/ 目录下运行）：
  python scripts/llm_judge.py --kind images --limit 30      # 先抽30张看看效果
  python scripts/llm_judge.py --kind documents
  python scripts/llm_judge.py --kind images --concurrency 4  # 并发跑，注意云端API可能有限速，先从小并发试起

输出：results/judge_images.jsonl 或 results/judge_documents.jsonl
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import config
from core.client import call_with_retry, encode_image_b64, get_client, parse_json_loose, run_concurrent
from core.prompts import get_prompt


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def judge_images(client, profile, limit, concurrency, lang):
    src_path = os.path.join(config.RESULTS_DIR, "images_tags_small.jsonl")
    if not os.path.exists(src_path):
        raise SystemExit(f"找不到 {src_path}，请先跑 python tag_images.py --profile small")
    rows = load_jsonl(src_path)
    if limit:
        rows = rows[:limit]
    judge_prompt_template = get_prompt("image_judge", lang)

    def process_row(row):
        img_path = os.path.join(config.IMAGES_DIR, row["filename"])
        verdict, error = {}, None
        try:
            b64 = encode_image_b64(img_path)
            prompt = judge_prompt_template.format(candidate_tags=row.get("tags", []))
            resp = call_with_retry(
                client,
                model=profile["model"],
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                            },
                        ],
                    }
                ],
                max_tokens=500,
                extra_body=profile.get("extra_body"),
            )
            verdict = parse_json_loose(resp.choices[0].message.content)
        except Exception as e:  # noqa: BLE001
            error = str(e)

        return {
            "filename": row["filename"], "candidate_tags": row.get("tags", []),
            "verdict": verdict, "error": error,
        }

    results = run_concurrent(rows, process_row, concurrency, desc="judging images")
    lang_suffix = "" if lang == "zh" else f"_{lang}"
    out_path = os.path.join(config.RESULTS_DIR, f"judge_images{lang_suffix}.jsonl")
    with open(out_path, "w", encoding="utf-8") as out:
        for r in results:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"完成，写入 {out_path}")


def judge_documents(client, profile, limit, concurrency, lang):
    src_path = os.path.join(config.RESULTS_DIR, "docs_tags_small.jsonl")
    if not os.path.exists(src_path):
        raise SystemExit(f"找不到 {src_path}，请先跑 python tag_documents.py --profile small")
    rows = load_jsonl(src_path)
    if limit:
        rows = rows[:limit]
    judge_prompt_template = get_prompt("document_judge", lang)

    def process_row(row):
        verdict, error = {}, None
        try:
            prompt = judge_prompt_template.format(
                document_text=row.get("extracted_text_used", ""),
                candidate_type=row.get("document_type", ""),
                candidate_tags=row.get("tags", []),
            )
            resp = call_with_retry(
                client, model=profile["model"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                extra_body=profile.get("extra_body"),
            )
            verdict = parse_json_loose(resp.choices[0].message.content)
        except Exception as e:  # noqa: BLE001
            error = str(e)

        return {
            "filename": row["filename"],
            "candidate_type": row.get("document_type", ""),
            "candidate_tags": row.get("tags", []),
            "verdict": verdict,
            "error": error,
        }

    results = run_concurrent(rows, process_row, concurrency, desc="judging documents")
    lang_suffix = "" if lang == "zh" else f"_{lang}"
    out_path = os.path.join(config.RESULTS_DIR, f"judge_documents{lang_suffix}.jsonl")
    with open(out_path, "w", encoding="utf-8") as out:
        for r in results:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"完成，写入 {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["images", "documents"], required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--concurrency", type=int, default=1,
        help="并发请求数，默认1（顺序执行）。这条路走的是付费云端API，加大并发前留意服务商的限速/并发上限",
    )
    ap.add_argument(
        "--lang", choices=["zh", "en"], default="zh", help="裁判 prompt 语言，默认中文",
    )
    args = ap.parse_args()

    profile = config.MODEL_PROFILES["reference"]
    client = get_client(profile)

    if args.kind == "images":
        judge_images(client, profile, args.limit, args.concurrency, args.lang)
    else:
        judge_documents(client, profile, args.limit, args.concurrency, args.lang)


if __name__ == "__main__":
    main()
