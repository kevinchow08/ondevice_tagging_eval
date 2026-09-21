# -*- coding: utf-8 -*-
"""
给 tagging_test_samples/images 里的图片跑打标签。

用法：
  python tag_images.py --profile small                    # 跑你的端侧模型
  python tag_images.py --profile reference                 # 跑更强的模型，生成参考标签
  python tag_images.py --profile small --limit 20          # 先跑20张试跑一下流程通不通
  python tag_images.py --profile small --concurrency 4     # 并发跑（配合服务端 -np 一起调大，见README）

输出：results/images_tags_<profile>.jsonl，每行一个样本的结果。
"""
import argparse
import csv
import json
import os

import config
from prompts import IMAGE_TAGGING_PROMPT
from tagger_core import (
    TAGS_SCHEMA,
    call_with_retry,
    encode_image_b64,
    get_client,
    is_local,
    json_schema_format,
    parse_json_loose,
    run_concurrent,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=list(config.MODEL_PROFILES.keys()), required=True)
    ap.add_argument("--limit", type=int, default=None, help="只跑前N张，方便先小规模试跑")
    ap.add_argument(
        "--concurrency", type=int, default=1,
        help="并发请求数，默认1（顺序执行）。加大之前记得服务端 -np/-c 也要同步调大，见README",
    )
    args = ap.parse_args()

    profile = config.MODEL_PROFILES[args.profile]
    client = get_client(profile)
    response_format = (
        json_schema_format("image_tags", TAGS_SCHEMA) if is_local(profile["base_url"]) else None
    )

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(config.RESULTS_DIR, f"images_tags_{args.profile}.jsonl")

    with open(config.IMAGES_MANIFEST, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]

    def process_row(row):
        img_path = os.path.join(config.IMAGES_DIR, row["filename"])
        tags, raw, error = [], "", None
        try:
            b64 = encode_image_b64(img_path)
            resp = call_with_retry(
                client,
                model=profile["model"],
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": IMAGE_TAGGING_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                            },
                        ],
                    }
                ],
                max_tokens=500,
                extra_body=profile.get("extra_body"),
                response_format=response_format,
            )
            raw = resp.choices[0].message.content
            tags = parse_json_loose(raw).get("tags", [])
        except Exception as e:  # noqa: BLE001
            error = str(e)

        return {
            "filename": row["filename"],
            "reference_label_en": row["reference_label_en"],
            "model_profile": args.profile,
            "tags": tags,
            "raw_response": raw,
            "error": error,
        }

    results = run_concurrent(
        rows, process_row, args.concurrency, desc=f"tagging images [{args.profile}]"
    )
    n_error = sum(1 for r in results if r["error"])

    with open(out_path, "w", encoding="utf-8") as out:
        for r in results:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"完成，共 {len(rows)} 条，{n_error} 条出错。写入 {out_path}")


if __name__ == "__main__":
    main()
