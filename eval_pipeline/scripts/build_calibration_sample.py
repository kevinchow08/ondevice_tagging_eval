# -*- coding: utf-8 -*-
"""
从裁判结果里分层抽样，导出一份人工核对用的 CSV——不是网页UI，就是让你能一条条对着原图/原文，
核实裁判判得准不准，最后算一个"人工-裁判一致率"，衡量现在这套裁判方法本身可不可信
（方法论参考 MT-Bench 论文：拿30~50条人工标注样本去校准 LLM 裁判）。

抽样策略：大部分从"候选(SLM)输给参考模型"的样本里抽（这类最多、最值得核实裁判有没有过严），
小部分从"赢/平"的样本里抽（防止裁判对好样本也有系统性误判，只看差的会漏掉这类问题）。
如果有效样本总数本来就不超过 --n，直接全部收进来，不做真正的抽样——现在文档有18份，
还没超过默认的 --n 30，就是这种情况；但这不是"文档"这个kind专属的特例，以后不管图片还是
文档，只要数据集还没大到超过 --n，都会自动全审，数据集变大了也会自动切换成抽样，不用改代码。

用法（在 eval_pipeline/ 目录下运行）：
  python scripts/build_calibration_sample.py --kind images --n 30
  python scripts/build_calibration_sample.py --kind documents

输出：results/calibration_images.csv / results/calibration_documents.csv
重新生成时，已经填过的 human_agree/human_notes 会按 filename 对齐保留下来，不会被清空。

填完之后（human_agree 列填"同意"/"不同意"/"不确定"）跑 python scripts/summarize.py，
报告最上面的"分级判定"就会自动带上人工校准的一致率。
"""
import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import config

FIELDNAMES = [
    "filename", "path_hint", "judge_winner", "judge_comment",
    "candidate_tags", "comparison_tags", "hallucinated_tags", "missing_important_tags",
    "human_agree", "human_notes",
]


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_existing_human_input(csv_path):
    """已经填过的人工判断，按 filename 存起来，重新生成抽样时不要把你填的东西清掉。"""
    existing = {}
    if os.path.exists(csv_path):
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if (row.get("human_agree") or "").strip() or (row.get("human_notes") or "").strip():
                    existing[row["filename"]] = {
                        "human_agree": row.get("human_agree", ""),
                        "human_notes": row.get("human_notes", ""),
                    }
    return existing


def stratified_sample(rows, n, seed, loss_ratio):
    scored = [r for r in rows if r.get("verdict") and r["verdict"].get("winner") in ("candidate", "comparison", "tie")]
    if len(scored) <= n:
        # 有效样本数不超过要抽的条数，没有"抽样"的必要，全部收进来就是了（小数据集时的自然行为，
        # 不需要特殊判断是图片还是文档——数据集小就全审，数据集大了这个分支自动就不会走进来了）。
        return scored

    losses = [r for r in scored if r["verdict"]["winner"] == "comparison"]
    others = [r for r in scored if r["verdict"]["winner"] != "comparison"]

    rng = random.Random(seed)
    n_loss = min(len(losses), round(n * loss_ratio))
    n_other = min(len(others), n - n_loss)
    picked = rng.sample(losses, n_loss) + rng.sample(others, n_other)
    rng.shuffle(picked)
    return picked


def build_row(r, path_hint):
    v = r.get("verdict") or {}
    return {
        "filename": r["filename"],
        "path_hint": path_hint,
        "judge_winner": v.get("winner", ""),
        "judge_comment": v.get("comment", ""),
        "candidate_tags": json.dumps(r.get("candidate_tags", []), ensure_ascii=False),
        "comparison_tags": json.dumps(r.get("comparison_tags", []), ensure_ascii=False),
        "hallucinated_tags": json.dumps(v.get("hallucinated_tags", []), ensure_ascii=False),
        "missing_important_tags": json.dumps(v.get("missing_important_tags", []), ensure_ascii=False),
        "human_agree": "",
        "human_notes": "",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["images", "documents"], required=True)
    ap.add_argument(
        "--n", type=int, default=30,
        help="抽样条数，默认30；有效样本总数不超过这个值就直接全审（现在文档只有6份就是这样），"
             "数据集以后扩大了会自动切换成真正的抽样",
    )
    ap.add_argument("--seed", type=int, default=42, help="抽样随机种子，固定下来方便复现")
    ap.add_argument(
        "--loss-ratio", type=float, default=0.8,
        help="抽样里\"候选输给参考模型\"这一类占的比例，默认0.8（8:2），剩下的从赢/平里抽做对照",
    )
    args = ap.parse_args()

    judge_path = os.path.join(config.RESULTS_DIR, f"judge_{args.kind}.jsonl")
    rows = load_jsonl(judge_path)
    if not rows:
        raise SystemExit(f"找不到 {judge_path}，请先跑 python scripts/llm_judge.py --kind {args.kind}")

    out_path = os.path.join(config.RESULTS_DIR, f"calibration_{args.kind}.csv")
    existing = load_existing_human_input(out_path)

    picked = stratified_sample(rows, args.n, args.seed, args.loss_ratio)

    src_dir = config.IMAGES_DIR if args.kind == "images" else config.DOCS_DIR
    out_rows = []
    for r in picked:
        row = build_row(r, os.path.join(src_dir, r["filename"]))
        if r["filename"] in existing:
            row.update(existing[r["filename"]])
        out_rows.append(row)

    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(out_rows)

    n_carried = sum(1 for r in out_rows if r["filename"] in existing)
    print(f"抽了 {len(out_rows)} 条（其中 {n_carried} 条沿用了你之前已经填过的判断），写入 {out_path}")
    print("打开这个 CSV，对着 path_hint 指的原图/原文，在 human_agree 列填 同意/不同意/不确定，"
          "human_notes 可选填备注；填完跑 python scripts/summarize.py 生成带校准结果的报告")


if __name__ == "__main__":
    main()
