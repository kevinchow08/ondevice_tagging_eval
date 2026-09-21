# -*- coding: utf-8 -*-
"""
给 tagging_test_samples/images 里的图片跑打标签。

用法：
  python tag_images.py --profile small           # 跑你的端侧模型
  python tag_images.py --profile reference        # 跑更强的模型，生成参考标签
  python tag_images.py --profile small --limit 20 # 先跑20张试跑一下流程通不通

输出：results/images_tags_<profile>.jsonl，每行一个样本的结果。
"""
import argparse
import csv
import json
import os

from tqdm import tqdm

import config
from prompts import IMAGE_TAGGING_PROMPT
from tagger_core import call_with_retry, encode_image_b64, get_client, parse_json_loose


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=list(config.MODEL_PROFILES.keys()), required=True)
    ap.add_argument("--limit", type=int, default=None, help="只跑前N张，方便先小规模试跑")
    args = ap.parse_args()

    profile = config.MODEL_PROFILES[args.profile]
    client = get_client(profile)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(config.RESULTS_DIR, f"images_tags_{args.profile}.jsonl")

    with open(config.IMAGES_MANIFEST, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]

    n_error = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for row in tqdm(rows, desc=f"tagging images [{args.profile}]"):
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
                )
                raw = resp.choices[0].message.content
                tags = parse_json_loose(raw).get("tags", [])
            except Exception as e:  # noqa: BLE001
                error = str(e)
                n_error += 1

            out.write(
                json.dumps(
                    {
                        "filename": row["filename"],
                        "reference_label_en": row["reference_label_en"],
                        "model_profile": args.profile,
                        "tags": tags,
                        "raw_response": raw,
                        "error": error,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"完成，共 {len(rows)} 条，{n_error} 条出错。写入 {out_path}")


if __name__ == "__main__":
    main()
