# -*- coding: utf-8 -*-
"""
语义相似度打分：拿 small 模型的标签跟 reference 模型的标签做embedding比对，
不做字符串精确匹配（开放词表模型不该被精确匹配惩罚，理由见 README）。

前提：先跑过
  python scripts/tag_images.py --profile small
  python scripts/tag_images.py --profile reference
（documents同理）

依赖较重：pip install sentence-transformers（会带一份 torch）。
如果不想装这个，可以跳过这一步，只用 llm_judge.py 那条路。

用法（在 eval_pipeline/ 目录下运行）：
  python scripts/score_semantic.py --kind images
  python scripts/score_semantic.py --kind documents
输出：results/semantic_scores_<kind>.csv（逐条明细）+ 终端打印整体均值
"""
import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer, util

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import config

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
        # sim 是 [n_candidate, n_reference] 的矩阵：候选标签在行，参考标签在列，
        # sim[i][j] = 第i个候选标签 和 第j个参考标签 的余弦相似度(0~1，越大越像)。
        sim = util.cos_sim(cand_emb, ref_emb)  # [n_candidate, n_reference]

        # precision：站在"候选标签"这一边问——我打的每个标签，在参考标签里能不能找到相似的？
        # dim=1 是压掉"列"（参考标签）这个维度，对每一行（每个候选标签）取它跟所有参考标签里
        # 相似度最高的那个值，得到长度=n_candidate的向量。>=阈值就算这个候选标签"找到对应物"，
        # 再取均值 = 候选标签里有多少比例是"靠谱、参考里确实有类似东西"的，衡量"瞎编的多不多"。
        precision = (sim.max(dim=1).values >= config.SEMANTIC_MATCH_THRESHOLD).float().mean().item()
        # recall：反过来站在"参考标签"这一边问——参考模型觉得该打的每个标签，有没有被候选覆盖到？
        # dim=0 是压掉"行"（候选标签）这个维度，对每一列（每个参考标签）取它跟所有候选标签里
        # 相似度最高的那个值，得到长度=n_reference的向量。>=阈值就算这个参考标签"被覆盖到"，
        # 再取均值 = 该打的标签里有多少比例真被打出来了，衡量"漏打的多不多"。
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
