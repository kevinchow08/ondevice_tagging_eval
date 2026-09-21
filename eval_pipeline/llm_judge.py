# -*- coding: utf-8 -*-
"""
LLM-as-judge：用 config.py 里 reference profile 的模型，直接看着原图/原文，
评审 small 模型打出来的标签有没有瞎编、有没有漏打、准不准。
不需要先跑 reference 的 tag_images.py/tag_documents.py，这条路是独立的。

用法：
  python llm_judge.py --kind images --limit 30      # 先抽30张看看效果
  python llm_judge.py --kind documents

输出：results/judge_images.jsonl 或 results/judge_documents.jsonl
"""
import argparse
import json
import os

from tqdm import tqdm

import config
from prompts import DOCUMENT_JUDGE_PROMPT, IMAGE_JUDGE_PROMPT
from tagger_core import call_with_retry, encode_image_b64, get_client, parse_json_loose


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def judge_images(client, profile, limit):
    src_path = os.path.join(config.RESULTS_DIR, "images_tags_small.jsonl")
    if not os.path.exists(src_path):
        raise SystemExit(f"找不到 {src_path}，请先跑 python tag_images.py --profile small")
    rows = load_jsonl(src_path)
    if limit:
        rows = rows[:limit]

    out_path = os.path.join(config.RESULTS_DIR, "judge_images.jsonl")
    with open(out_path, "w", encoding="utf-8") as out:
        for row in tqdm(rows, desc="judging images"):
            img_path = os.path.join(config.IMAGES_DIR, row["filename"])
            verdict, error = {}, None
            try:
                b64 = encode_image_b64(img_path)
                prompt = IMAGE_JUDGE_PROMPT.format(candidate_tags=row.get("tags", []))
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

            out.write(
                json.dumps(
                    {"filename": row["filename"], "candidate_tags": row.get("tags", []),
                     "verdict": verdict, "error": error},
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"完成，写入 {out_path}")


def judge_documents(client, profile, limit):
    src_path = os.path.join(config.RESULTS_DIR, "docs_tags_small.jsonl")
    if not os.path.exists(src_path):
        raise SystemExit(f"找不到 {src_path}，请先跑 python tag_documents.py --profile small")
    rows = load_jsonl(src_path)
    if limit:
        rows = rows[:limit]

    out_path = os.path.join(config.RESULTS_DIR, "judge_documents.jsonl")
    with open(out_path, "w", encoding="utf-8") as out:
        for row in tqdm(rows, desc="judging documents"):
            verdict, error = {}, None
            try:
                prompt = DOCUMENT_JUDGE_PROMPT.format(
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

            out.write(
                json.dumps(
                    {
                        "filename": row["filename"],
                        "candidate_type": row.get("document_type", ""),
                        "candidate_tags": row.get("tags", []),
                        "verdict": verdict,
                        "error": error,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"完成，写入 {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["images", "documents"], required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    profile = config.MODEL_PROFILES["reference"]
    client = get_client(profile)

    if args.kind == "images":
        judge_images(client, profile, args.limit)
    else:
        judge_documents(client, profile, args.limit)


if __name__ == "__main__":
    main()
