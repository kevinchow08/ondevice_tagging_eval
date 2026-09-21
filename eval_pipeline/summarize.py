# -*- coding: utf-8 -*-
"""
把 judge_*.jsonl 和 semantic_scores_*.csv 汇总成一份可读的报告。
跑之前哪些文件存在就汇总哪些，缺了哪个就在报告里注明跳过，不会报错中断。

用法：python summarize.py
输出：results/summary_report.md
"""
import json
import os

import pandas as pd

import config


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def summarize_judge(rows, label):
    if not rows:
        return f"### {label}\n（没有找到 LLM 裁判评审结果，跳过。先跑 llm_judge.py）\n"

    scored = [r for r in rows if r.get("verdict") and r["verdict"].get("accuracy_score") is not None]
    avg_score = sum(r["verdict"]["accuracy_score"] for r in scored) / len(scored) if scored else None
    hallucinated_total = sum(len(r["verdict"].get("hallucinated_tags", [])) for r in rows if r.get("verdict"))
    missing_total = sum(len(r["verdict"].get("missing_important_tags", [])) for r in rows if r.get("verdict"))

    lines = [f"### {label}", f"- 样本数：{len(rows)}"]
    lines.append(
        f"- 平均准确度打分：{avg_score:.3f}" if avg_score is not None else "- 平均准确度打分：无有效数据"
    )
    lines.append(f"- 累计瞎编标签数：{hallucinated_total}")
    lines.append(f"- 累计漏打重要标签数：{missing_total}")

    worst = sorted(scored, key=lambda r: r["verdict"]["accuracy_score"])[:5]
    if worst:
        lines.append("- 得分最低的样本（优先看这些）：")
        for r in worst:
            lines.append(
                f"  - `{r['filename']}`：{r['verdict']['accuracy_score']} — "
                f"{r['verdict'].get('comment', '')}"
            )
    return "\n".join(lines) + "\n"


def summarize_semantic(csv_path, label):
    if not os.path.exists(csv_path):
        return f"### {label}\n（没有找到语义相似度打分结果，跳过。先跑 score_semantic.py）\n"
    df = pd.read_csv(csv_path)
    return (
        f"### {label}\n"
        f"- 样本数：{len(df)}\n"
        f"- 平均 precision（模型打出的标签里，语义上有效的比例）：{df['precision'].mean():.3f}\n"
        f"- 平均 recall（参考标签里，被模型标签覆盖到的比例）：{df['recall'].mean():.3f}\n"
    )


def main():
    parts = ["# 端侧打标签模型测评报告\n"]

    parts.append("## 一、LLM 裁判评审结果（推荐优先看这部分）\n")
    parts.append(summarize_judge(load_jsonl(os.path.join(config.RESULTS_DIR, "judge_images.jsonl")), "图片"))
    parts.append(summarize_judge(load_jsonl(os.path.join(config.RESULTS_DIR, "judge_documents.jsonl")), "文档"))

    parts.append("## 二、语义相似度打分结果（辅助参考）\n")
    parts.append(summarize_semantic(os.path.join(config.RESULTS_DIR, "semantic_scores_images.csv"), "图片"))
    parts.append(summarize_semantic(os.path.join(config.RESULTS_DIR, "semantic_scores_documents.csv"), "文档"))

    out_path = os.path.join(config.RESULTS_DIR, "summary_report.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print(f"报告已生成：{out_path}")


if __name__ == "__main__":
    main()
