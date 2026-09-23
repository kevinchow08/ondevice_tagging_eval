# -*- coding: utf-8 -*-
"""
拿 small 模型给图片打的标签跟人工核实过的闭集标签库（tagging_test_samples/ground_truth_images.csv）
做 embedding 语义比对，不做字符串精确匹配（开放词表模型不该被精确匹配惩罚，理由见 README）。

这是评测的核心指标来源：不跟另一个模型的输出比，是跟人工写好、
人工确认过的标准答案比——precision/recall 因此是确定性计算，不依赖任何裁判模型的主观判断。

标准答案库里，每条 tags 的第一个标签固定是"主体大类"（抽查过200条里随机20条+前20条，
全部成立）。`primary_covered` 这一列就是单独看"这个最重要的主体标签，候选标签里有没有
覆盖到"，不要求剩下颜色/数量/姿态这些次要维度都覆盖到——要求模型把5~15个标准答案标签
一个不漏全覆盖，对开放式生成任务来说门槛偏高，分级判定用 primary_covered 的覆盖率定，
完整的 recall（对全部标准答案标签的覆盖率）仍然保留展示，作为更细粒度的参考，不参与定级。

--lang 要跟打标签时用的 --lang 保持一致（决定读哪个候选标签文件）。标准答案库本身只有
一份中文标签，不分语言——图片标签大多是颜色/动物/物体这类常见词，跨语言 embedding 相似度
普遍在0.85以上（"狗" vs "dog" 0.975 这个量级），混着比不构成实际问题。

依赖较重：pip install sentence-transformers（会带一份 torch）。

前提：先跑过 python scripts/tag_images.py --profile small

用法（在 eval_pipeline/ 目录下运行）：
  python scripts/score_semantic.py
  python scripts/score_semantic.py --lang en
输出：results/semantic_scores_images.csv（<lang>非zh时文件名带后缀，逐条明细，含未匹配上
的标签，方便你抽查是不是标准答案本身漏标了，还是模型真的瞎编/漏打）+ 终端打印整体均值
"""
import argparse
import csv
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


def load_ground_truth(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return {r["filename"]: json.loads(r["tags"]) for r in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--lang", choices=["zh", "en"], default="zh",
        help="要跟打标签时用的 --lang 一致，决定读哪份候选标签文件",
    )
    args = ap.parse_args()

    lang_suffix = "" if args.lang == "zh" else f"_{args.lang}"
    small_path = os.path.join(config.RESULTS_DIR, f"images_tags_small{lang_suffix}.jsonl")
    if not os.path.exists(small_path):
        raise SystemExit(f"找不到 {small_path}，请先跑 tag_images.py --profile small --lang {args.lang}")
    if not os.path.exists(config.IMAGES_GROUND_TRUTH):
        raise SystemExit(f"找不到 {config.IMAGES_GROUND_TRUTH}，闭集标签库缺失")

    small = {r["filename"]: r for r in load_jsonl(small_path)}
    ground_truth = load_ground_truth(config.IMAGES_GROUND_TRUTH)

    print(f"加载 embedding 模型 {MODEL_NAME}（第一次跑会自动下载，可能要等一会）...")
    model = SentenceTransformer(MODEL_NAME)

    rows = []
    for filename, s in small.items():
        ref_tags = ground_truth.get(filename)
        if ref_tags is None:
            continue
        cand_tags = s.get("tags") or []
        if not cand_tags or not ref_tags:
            rows.append(
                {"filename": filename, "precision": 0.0, "recall": 0.0, "primary_covered": False,
                 "n_candidate": len(cand_tags), "n_reference": len(ref_tags),
                 "unmatched_candidate_tags": json.dumps(cand_tags, ensure_ascii=False),
                 "unmatched_reference_tags": json.dumps(ref_tags, ensure_ascii=False)}
            )
            continue

        cand_emb = model.encode(cand_tags, convert_to_tensor=True)
        ref_emb = model.encode(ref_tags, convert_to_tensor=True)
        # sim 是 [n_candidate, n_reference] 的矩阵：候选标签在行，参考标签在列，
        # sim[i][j] = 第i个候选标签 和 第j个参考标签 的余弦相似度(0~1，越大越像)。
        sim = util.cos_sim(cand_emb, ref_emb)  # [n_candidate, n_reference]

        # precision：站在"候选标签"这一边问——我打的每个标签，在标准答案里能不能找到相似的？
        # dim=1 是压掉"列"（参考标签）这个维度，对每一行（每个候选标签）取它跟所有参考标签里
        # 相似度最高的那个值，得到长度=n_candidate的向量。>=阈值就算这个候选标签"找到对应物"，
        # 再取均值 = 候选标签里有多少比例是"标准答案里确实有类似东西"的，衡量"瞎编的多不多"。
        cand_max = sim.max(dim=1).values
        cand_hit = cand_max >= config.SEMANTIC_MATCH_THRESHOLD
        precision = cand_hit.float().mean().item()
        # recall：反过来站在"标准答案"这一边问——答案里该打的每个标签，有没有被候选覆盖到？
        # dim=0 是压掉"行"（候选标签）这个维度，对每一列（每个参考标签）取它跟所有候选标签里
        # 相似度最高的那个值，得到长度=n_reference的向量。>=阈值就算这个参考标签"被覆盖到"，
        # 再取均值 = 该打的标签里有多少比例真被打出来了，衡量"漏打的多不多"。
        ref_max = sim.max(dim=0).values
        ref_hit = ref_max >= config.SEMANTIC_MATCH_THRESHOLD
        recall = ref_hit.float().mean().item()

        # 没匹配上的标签单独列出来，不直接坐实成"瞎编"/"漏打"——标准答案是人一次性写的，
        # 不是每条都保证穷尽，候选标签没匹配上也可能是答案本身漏标了，这两列就是留给你
        # 抽查用的，看多了会发现哪些是模型真错、哪些是答案该补。
        unmatched_candidate = [cand_tags[i] for i in range(len(cand_tags)) if not cand_hit[i]]
        unmatched_reference = [ref_tags[j] for j in range(len(ref_tags)) if not ref_hit[j]]

        rows.append(
            {
                "filename": filename,
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                # ref_hit[0] = 标准答案第一个标签（约定俗成的"主体大类"）有没有被候选标签覆盖到。
                "primary_covered": bool(ref_hit[0].item()),
                "n_candidate": len(cand_tags),
                "n_reference": len(ref_tags),
                "unmatched_candidate_tags": json.dumps(unmatched_candidate, ensure_ascii=False),
                "unmatched_reference_tags": json.dumps(unmatched_reference, ensure_ascii=False),
            }
        )

    df = pd.DataFrame(rows)
    out_csv = os.path.join(config.RESULTS_DIR, f"semantic_scores_images{lang_suffix}.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"\n共 {len(df)} 条样本")
    print(f"平均 precision（模型标签里，标准答案能验证上的比例）：{df['precision'].mean():.3f}")
    print(f"平均 recall（标准答案里，被模型标签覆盖到的比例，仅参考）：{df['recall'].mean():.3f}")
    print(f"主体标签覆盖率（只看第一个标签有没有覆盖到，分级判定用这个）：{df['primary_covered'].mean():.3f}")
    print(f"逐条明细（含未匹配标签）写入 {out_csv}")


if __name__ == "__main__":
    main()
