# -*- coding: utf-8 -*-
"""
把 judge_*.jsonl / images_tags_small.jsonl / docs_tags_small.jsonl / semantic_scores_*.csv
汇总成一份可读的报告。跑之前哪些文件存在就汇总哪些，缺了哪个就在报告里注明跳过，不会报错中断。

用法（在 eval_pipeline/ 目录下运行）：python scripts/summarize.py
输出：results/summary_report.md
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import config


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def judge_stats(rows):
    """算出裁判结果的关键数字，返回 dict，None 表示没有有效数据。
    单独抽出来是因为"结论与建议"那节要复用这些数字，不想重新解析一遍 rows。"""
    scored = [r for r in rows if r.get("verdict") and r["verdict"].get("accuracy_score") is not None]
    n = len(scored)
    if not n:
        return None
    halluc_counts = [len(r["verdict"].get("hallucinated_tags", [])) for r in scored]
    missing_counts = [len(r["verdict"].get("missing_important_tags", [])) for r in scored]
    return {
        "n": n,
        "avg_score": sum(r["verdict"]["accuracy_score"] for r in scored) / n,
        "halluc_rate": sum(1 for c in halluc_counts if c > 0) / n,
        "missing_rate": sum(1 for c in missing_counts if c > 0) / n,
        "avg_halluc_per_sample": sum(halluc_counts) / n,
        "avg_missing_per_sample": sum(missing_counts) / n,
        "worst": sorted(scored, key=lambda r: r["verdict"]["accuracy_score"])[:5],
    }


def summarize_judge(rows, label):
    if not rows:
        return f"### {label}\n（没有找到 LLM 裁判评审结果，跳过。先跑 llm_judge.py）\n"

    stats = judge_stats(rows)
    lines = [f"### {label}", f"- 样本数：{len(rows)}"]
    if not stats:
        lines.append("- 平均准确度打分：无有效数据")
        return "\n".join(lines) + "\n"

    lines.append(f"- 平均准确度打分：{stats['avg_score']:.3f}")
    # accuracy_score 均值是唯一样本量无关、能跨批次比较的数字。瞎编/漏打用"受影响样本占比"+
    # "平均每条个数"，不展示累计总数——总数会随样本量线性增长，不同批次之间不可比，
    # 而且标签给得越少总数天然越好看，跟真实质量没有必然关系。
    lines.append(
        f"- 瞎编：{stats['halluc_rate']:.0%} 的样本至少有1个瞎编标签，"
        f"平均每条 {stats['avg_halluc_per_sample']:.2f} 个"
    )
    lines.append(
        f"- 漏打：{stats['missing_rate']:.0%} 的样本至少漏打1个重要标签，"
        f"平均每条 {stats['avg_missing_per_sample']:.2f} 个"
    )
    if stats["worst"]:
        lines.append("- 得分最低的样本（优先看这些）：")
        for r in stats["worst"]:
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


def _percentile(sorted_vals, p):
    if not sorted_vals:
        return None
    idx = min(len(sorted_vals) - 1, int(len(sorted_vals) * p))
    return sorted_vals[idx]


def summarize_perf(rows, label):
    """汇总 tag_images.py/tag_documents.py 存的 perf 字段：延迟、生成吞吐量。
    只对本地(端侧)模型有意义，衡量的是"这个模型实际部署在端上跑起来的资源/速度代价"，
    跟标签准不准是两件独立的事——一个模型标得很准但慢到不能用，也不算达标。"""
    perfs = [r["perf"] for r in rows if r.get("perf") and "latency_seconds" in r["perf"]]
    if not perfs:
        return f"### {label}\n（没有性能数据，用新版 tag_images.py/tag_documents.py 重新跑一遍即可采集到）\n"

    latencies = sorted(p["latency_seconds"] for p in perfs)
    tps_values = [p["tokens_per_second"] for p in perfs if p.get("tokens_per_second") is not None]

    lines = [f"### {label}", f"- 样本数：{len(perfs)}"]
    lines.append(
        f"- 单条延迟：平均 {sum(latencies)/len(latencies):.2f}s，"
        f"P50 {_percentile(latencies, 0.5):.2f}s，"
        f"P95 {_percentile(latencies, 0.95):.2f}s"
    )
    if tps_values:
        lines.append(f"- 平均生成吞吐：{sum(tps_values)/len(tps_values):.1f} tokens/s（仅本地 llama.cpp 等能拿到）")

    # prefill(理解输入) vs generation(生成输出) 耗时拆分 + token量，帮助定位瓶颈在哪一阶段。
    prompt_tok = [p["prompt_tokens"] for p in perfs if p.get("prompt_tokens") is not None]
    completion_tok = [p["completion_tokens"] for p in perfs if p.get("completion_tokens") is not None]
    prefill_ms = [p["prompt_eval_ms"] for p in perfs if p.get("prompt_eval_ms") is not None]
    gen_ms = [p["generation_ms"] for p in perfs if p.get("generation_ms") is not None]
    if prompt_tok and completion_tok:
        lines.append(
            f"- 平均输入/输出 token 数：{sum(prompt_tok)/len(prompt_tok):.0f} / "
            f"{sum(completion_tok)/len(completion_tok):.0f}"
        )
    if prefill_ms and gen_ms:
        avg_prefill = sum(prefill_ms) / len(prefill_ms)
        avg_gen = sum(gen_ms) / len(gen_ms)
        lines.append(
            f"- prefill(理解输入) vs generation(生成输出) 耗时：平均 {avg_prefill:.0f}ms / {avg_gen:.0f}ms"
            f"（占比 {avg_prefill/(avg_prefill+avg_gen):.0%} / {avg_gen/(avg_prefill+avg_gen):.0%}，"
            f"前者占大头说明瓶颈在输入太长/图片token太多，后者占大头说明瓶颈在生成本身）"
        )
    return "\n".join(lines) + "\n"


def build_conclusion(image_stats, doc_stats):
    """这节不是"能不能投产"的自动判定——评测样本量有限、裁判本身没做校准，
    工具没资格替你做这个业务决策。这里只是把已经算出来的数字翻译成可读的现状描述，
    加一个决策清单，帮你自己判断，不是代替你判断。"""
    lines = ["## 三、结论与建议（不是“能否投产”的自动判定，用于辅助你自己判断）\n"]
    lines.append(
        "**这份报告能告诉你“现状大概是什么样”，不能替你回答“能不能投产”——那个问题还依赖"
        "你的业务场景对各类错误的容忍度、下游有没有人工审核兜底、样本量是否足够覆盖你的真实场景，"
        "这些工具都不知道。以下是基于当前数据的客观现状 + 一个决策清单：**\n"
    )

    if image_stats:
        lines.append(
            f"- **图片**：{image_stats['n']} 个抽样样本，平均分 {image_stats['avg_score']:.2f}，"
            f"{image_stats['halluc_rate']:.0%} 的样本至少有1处瞎编（多为大类生物学/分类常识错误，"
            f"细分品类识别错误已通过 prompt 排除在评分之外）。"
        )
    if doc_stats:
        lines.append(
            f"- **文档**：{doc_stats['n']} 个样本，平均分 {doc_stats['avg_score']:.2f}，"
            f"瞎编率 {doc_stats['halluc_rate']:.0%}，但 {doc_stats['missing_rate']:.0%} 的样本"
            f"至少漏打1个重要字段/实体——文档这边的短板是漏打，不是瞎编。"
        )

    lines.append(
        "\n**投产前建议自查的清单：**\n"
        "1. 当前图片评审只抽了一部分样本，文档样本量更小（个位数）——正式上线前建议扩大到全量或"
        "更接近真实业务分布的数据集，再看这些比例是否稳定；\n"
        "2. 裁判本身没做多次重跑的一致性校验，单次结果有波动，重要决策前建议对同一批样本至少跑2次"
        "裁判，看分数/比例是否稳定；\n"
        "3. 看清楚失败模式是不是你业务场景真正关心的——比如图片“大类生物学分类错误”对“相册自动分类”"
        "场景可能无关痛痒，对“物种识别”类场景就是致命的；\n"
        "4. 结合下面的性能数据判断延迟能不能接受，标得准但跑不动同样不算达标。"
    )
    return "\n".join(lines) + "\n"


def main():
    parts = ["# 端侧打标签模型测评报告\n"]

    parts.append("## 一、LLM 裁判评审结果（推荐优先看这部分）\n")
    image_judge_rows = load_jsonl(os.path.join(config.RESULTS_DIR, "judge_images.jsonl"))
    doc_judge_rows = load_jsonl(os.path.join(config.RESULTS_DIR, "judge_documents.jsonl"))
    parts.append(summarize_judge(image_judge_rows, "图片"))
    parts.append(summarize_judge(doc_judge_rows, "文档"))

    parts.append("## 二、语义相似度打分结果（辅助参考）\n")
    parts.append(summarize_semantic(os.path.join(config.RESULTS_DIR, "semantic_scores_images.csv"), "图片"))
    parts.append(summarize_semantic(os.path.join(config.RESULTS_DIR, "semantic_scores_documents.csv"), "文档"))

    parts.append(build_conclusion(judge_stats(image_judge_rows), judge_stats(doc_judge_rows)))

    parts.append("## 四、端侧模型性能（延迟/吞吐，跟标签质量是两件独立的事）\n")
    parts.append(summarize_perf(load_jsonl(os.path.join(config.RESULTS_DIR, "images_tags_small.jsonl")), "图片"))
    parts.append(summarize_perf(load_jsonl(os.path.join(config.RESULTS_DIR, "docs_tags_small.jsonl")), "文档"))

    out_path = os.path.join(config.RESULTS_DIR, "summary_report.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print(f"报告已生成：{out_path}")


if __name__ == "__main__":
    main()
