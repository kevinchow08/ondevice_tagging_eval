# -*- coding: utf-8 -*-
"""
LLM-as-judge：用 config.py 里 reference profile 的模型，直接看着原图/原文，
两两对比 small 模型（candidate）和 reference 模型自己独立打的标签（comparison），
判断哪组更准、A组（candidate）有没有瞎编/漏打。

用的是"两两对比"而不是"绝对打分"——这是照着 LLM-as-judge 的研究结论来的
（MT-Bench 论文：pairwise comparison 比 absolute scoring 更可靠），所以这一步
依赖 reference profile 也独立打过一遍标签，不能只跑过 small 就用。

前提：先跑过
  python scripts/tag_images.py --profile small
  python scripts/tag_images.py --profile reference   （或 make tag-images-reference）
（documents同理）

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
from core.prompts import get_prompt, ground_truth_hint


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_candidate_and_comparison(candidate_path, comparison_path, comparison_hint):
    if not os.path.exists(candidate_path):
        raise SystemExit(f"找不到 {candidate_path}，请先跑 python scripts/tag_images.py --profile small（或 tag_documents.py）")
    if not os.path.exists(comparison_path):
        raise SystemExit(
            f"找不到 {comparison_path}——现在裁判是两两对比模式，需要 reference 模型也独立打过一遍标签才能比。"
            f"请先跑：{comparison_hint}"
        )
    candidate_rows = load_jsonl(candidate_path)
    comparison = {r["filename"]: r for r in load_jsonl(comparison_path)}
    return candidate_rows, comparison


def judge_images(client, profile, limit, concurrency, lang):
    candidate_path = os.path.join(config.RESULTS_DIR, "images_tags_small.jsonl")
    comparison_path = os.path.join(config.RESULTS_DIR, "images_tags_reference.jsonl")
    rows, comparison = load_candidate_and_comparison(
        candidate_path, comparison_path, "python scripts/tag_images.py --profile reference"
    )
    if limit:
        rows = rows[:limit]
    judge_prompt_template = get_prompt("image_judge", lang)

    def process_row(row, comparison):
        img_path = os.path.join(config.IMAGES_DIR, row["filename"])
        comparison_tags = (comparison.get(row["filename"]) or {}).get("tags", [])
        verdict, error = {}, None
        try:
            b64 = encode_image_b64(img_path)
            prompt = judge_prompt_template.format(
                candidate_tags=row.get("tags", []),
                comparison_tags=comparison_tags,
                ground_truth_hint=ground_truth_hint(row.get("reference_label_en"), lang),
            )
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
                temperature=0,
                extra_body=profile.get("extra_body"),
            )
            verdict = parse_json_loose(resp.choices[0].message.content)
        except Exception as e:  # noqa: BLE001
            error = str(e)

        return {
            "filename": row["filename"],
            "candidate_tags": row.get("tags", []),
            "comparison_tags": comparison_tags,
            "verdict": verdict,
            "error": error,
        }

    results = run_concurrent(
        rows, lambda row: process_row(row, comparison), concurrency, desc="judging images"
    )
    lang_suffix = "" if lang == "zh" else f"_{lang}"
    out_path = os.path.join(config.RESULTS_DIR, f"judge_images{lang_suffix}.jsonl")
    with open(out_path, "w", encoding="utf-8") as out:
        for r in results:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"完成，写入 {out_path}")


def judge_documents(client, profile, limit, concurrency, lang):
    candidate_path = os.path.join(config.RESULTS_DIR, "docs_tags_small.jsonl")
    comparison_path = os.path.join(config.RESULTS_DIR, "docs_tags_reference.jsonl")
    rows, comparison = load_candidate_and_comparison(
        candidate_path, comparison_path, "python scripts/tag_documents.py --profile reference"
    )
    if limit:
        rows = rows[:limit]
    judge_prompt_template = get_prompt("document_judge", lang)

    def process_row(row, comparison):
        comp_row = comparison.get(row["filename"]) or {}
        comparison_type = comp_row.get("document_type", "")
        comparison_tags = comp_row.get("tags", [])
        verdict, error = {}, None
        try:
            prompt = judge_prompt_template.format(
                document_text=row.get("extracted_text_used", ""),
                candidate_type=row.get("document_type", ""),
                candidate_tags=row.get("tags", []),
                comparison_type=comparison_type,
                comparison_tags=comparison_tags,
                ground_truth_hint=ground_truth_hint(row.get("reference_type_cn"), lang),
            )
            resp = call_with_retry(
                client, model=profile["model"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0,
                extra_body=profile.get("extra_body"),
            )
            verdict = parse_json_loose(resp.choices[0].message.content)
        except Exception as e:  # noqa: BLE001
            error = str(e)

        return {
            "filename": row["filename"],
            "candidate_type": row.get("document_type", ""),
            "candidate_tags": row.get("tags", []),
            "comparison_type": comparison_type,
            "comparison_tags": comparison_tags,
            "verdict": verdict,
            "error": error,
        }

    results = run_concurrent(
        rows, lambda row: process_row(row, comparison), concurrency, desc="judging documents"
    )
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
