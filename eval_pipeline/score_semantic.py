# -*- coding: utf-8 -*-
"""
语义相似度打分：拿 small 模型的标签跟 reference 模型的标签做embedding比对，
不做字符串精确匹配（开放词表模型不该被精确匹配惩罚，理由见 README）。

前提：先跑过
  python tag_images.py --profile small
  python tag_images.py --profile reference
（documents同理）

依赖较重：pip install sentence-transformers（会带一份 torch）。
如果不想装这个，可以跳过这一步，只用 llm_judge.py 那条路。

用法：
  python score_semantic.py --kind images
  python score_semantic.py --kind documents
输出：results/semantic_scores_<kind>.csv（逐条明细）+ 终端打印整体均值
"""
import argparse
import json
import os

import pandas as pd
from sentence_transformers import SentenceTransformer, util

import config

# 多语言轻量embedding模型，中英文标签混在一起也能比较
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["images", "documents"], required=True)
    args = ap.parse_args()

    prefix = "images_tags" if args.kind == "images" else "docs_tags"
    small_path = os.path.join(config.RESULTS_DIR, f"{prefix}_small.jsonl")
    ref_path = os.path.join(config.RESULTS_DIR, f"{prefix}_reference.jsonl")
    for p in (small_path, ref_path):
        if not os.path.exists(p):
            raise SystemExit(f"找不到 {p}，请先跑 tag_images.py/tag_documents.py 生成 small 和 reference 两份结果")

    small = {r["filename"]: r for r in load_jsonl(small_path)}
    ref = {r["filename"]: r for r in load_jsonl(ref_path)}

    print(f"加载 embedding 模型 {MODEL_NAME}（第一次跑会自动下载，可能要等一会）...")
    model = SentenceTransformer(MODEL_NAME)

    rows = []
    for filename, s in small.items():
        r = ref.get(filename)
        if not r:
            continue
        cand_tags = s.get("tags") or []
        ref_tags = r.get("tags") or []
        if not cand_tags or not ref_tags:
            rows.append(
                {"filename": filename, "precision": 0.0, "recall": 0.0,
                 "n_candidate": len(cand_tags), "n_reference": len(ref_tags)}
            )
            continue

        cand_emb = model.encode(cand_tags, convert_to_tensor=True)
        ref_emb = model.encode(ref_tags, convert_to_tensor=True)
        sim = util.cos_sim(cand_emb, ref_emb)  # [n_candidate, n_reference]

        # precision：模型打的每个标签，是否在参考标签里有相似的（>=阈值才算命中）
        precision = (sim.max(dim=1).values >= config.SEMANTIC_MATCH_THRESHOLD).float().mean().item()
        # recall：参考标签里的每一个，是否被模型的某个标签覆盖到了
        recall = (sim.max(dim=0).values >= config.SEMANTIC_MATCH_THRESHOLD).float().mean().item()

        rows.append(
            {
                "filename": filename,
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "n_candidate": len(cand_tags),
                "n_reference": len(ref_tags),
            }
        )

    df = pd.DataFrame(rows)
    out_csv = os.path.join(config.RESULTS_DIR, f"semantic_scores_{args.kind}.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"\n共 {len(df)} 条样本")
    print(f"平均 precision（模型标签里有效的比例）：{df['precision'].mean():.3f}")
    print(f"平均 recall（参考标签被覆盖到的比例）：{df['recall'].mean():.3f}")
    print(f"逐条明细写入 {out_csv}")


if __name__ == "__main__":
    main()
