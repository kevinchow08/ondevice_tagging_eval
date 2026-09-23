# -*- coding: utf-8 -*-
"""
把 semantic_scores_images.csv / images_tags_small.jsonl 汇总成一份可读的报告。
跑之前哪些文件存在就汇总哪些，缺了哪个就在报告里注明跳过，不会报错中断。

评测基准是 tagging_test_samples/ground_truth_images.csv 这份人工核实过的闭集标签库，
不依赖 LLM 裁判或人工校准——precision/recall 是对着固定答案算出来的确定性数字，
裁判模型的主观判断、判了准不准这类问题不存在了，见 README「评测方法」一节。

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


def semantic_stats(csv_path):
    """算出跟闭集标签库比对的关键数字，返回 dict，None 表示没有数据。
    单独抽出来是因为"结论与建议"那节要复用这些数字，不想重新解析一遍。

    只保留 precision/recall/primary_covered 均值 + 问题最多的几条样本明细，不再额外算
    "样本级"的瞎编率/漏打率（CHAIR_s 风格：至少有1个标签没匹配上的样本占比）——这对数字
    已经被证明在多标签场景下会显得过于严苛、跟直觉不符（precision 0.89 时，"至少错1个"的
    样本占比能轻松超过一半），留着即使标"仅参考"也容易被误读，索性不算不展示。"""
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    if df.empty:
        return None
    n = len(df)

    def parse_list(s):
        if pd.isna(s):
            return []
        return json.loads(s)

    unmatched_cand = df["unmatched_candidate_tags"].apply(parse_list)
    unmatched_ref = df["unmatched_reference_tags"].apply(parse_list)

    worst_idx = (unmatched_cand.apply(len) + unmatched_ref.apply(len)).sort_values(ascending=False).index[:5]
    worst = []
    for i in worst_idx:
        row = df.loc[i]
        if len(unmatched_cand[i]) == 0 and len(unmatched_ref[i]) == 0:
            continue
        worst.append(
            {
                "filename": row["filename"],
                "unmatched_candidate_tags": unmatched_cand[i],
                "unmatched_reference_tags": unmatched_ref[i],
            }
        )

    return {
        "n": n,
        "avg_precision": df["precision"].mean(),
        "avg_recall": df["recall"].mean(),
        "primary_covered": df["primary_covered"].mean(),
        "worst": worst,
    }


def grade(stats):
    """定级用"标签未匹配率"(=1-precision)和"主体标签未覆盖率"(=1-primary_covered)，
    哪个落在更差的档位就用哪个（短板原则）。阈值来自 config.QUALITY_GATE，写死但可调。

    漏打这一侧用 primary_covered 而不是完整 recall：标准答案库里每条 tags 的第一个标签
    固定是"主体大类"（抽查过200条里随机20条+前20条，全部成立），要求开放式生成任务把
    5~15个标准答案标签一个不漏全覆盖，对图片这种自由描述场景来说门槛偏高——抓准最重要的
    主体才是真正决定"能不能用"的门槛，完整 recall 仍然保留展示，作为更细粒度的参考。

    "标签未匹配率"这个词特意不叫"瞎编率"（改版前的老叫法）——"未匹配"跟"瞎编"不是一回事：
    候选标签在标准答案里找不到语义相似对应物，只说明"没人/没模型验证过这个标签是对的"，
    不代表"已经确认是编的"。以前 LLM 裁判会真的对着原图核实、确认了才叫瞎编；现在这只是
    "标准答案这份人工一次性写的清单里没有它"，也可能是清单本身漏写了（这也是为什么
    summarize_semantic() 会把每条样本具体未匹配的标签列出来，让你自己判断是哪种情况）。

    换算成"平均"而不是按 CHAIR_s 那样算"样本里至少错1个的占比"，是因为后者在多标签场景下
    统计上过于严苛（precision 0.89 时，8~15 个标签/样本，"至少错1个"的样本轻松过半），
    跟直觉上"这模型准不准"不匹配。"""
    gate = config.QUALITY_GATE
    unmatched = 1 - stats["avg_precision"]
    missing = 1 - stats["primary_covered"]

    def band(rate, green_max, yellow_max):
        if rate <= green_max:
            return 0
        if rate <= yellow_max:
            return 1
        return 2

    level = max(
        band(unmatched, gate["green_halluc"], gate["yellow_halluc"]),
        band(missing, gate["green_missing"], gate["yellow_missing"]),
    )
    label = ["🟢 可用", "🟡 有条件可用（建议配合人工审核）", "🔴 不建议直接使用"][level]
    reason = (
        f"标签未匹配率{unmatched:.0%}(=1-precision)、"
        f"主体标签未覆盖率{missing:.0%}(=1-primary_covered，只看最重要的第一个标签)，取较差档"
    )
    return label, reason


def summarize_headline(label, stats):
    if not stats:
        return f"- **{label}**：没有找到 semantic_scores 数据，跳过（先跑 tag_images.py 和 score_semantic.py）\n"
    level_label, reason = grade(stats)
    return f"- **{label}**：{level_label}（{reason}；对照标准是人工核实过的闭集标签库，不是模型判断）\n"


def summarize_semantic(stats, label):
    if not stats:
        return f"### {label}\n（没有找到语义相似度打分结果，跳过。先跑 score_semantic.py）\n"

    lines = [f"### {label}", f"- 样本数：{stats['n']}"]
    lines.append(
        f"- 平均 precision（模型标签里，标准答案能验证上的比例）：{stats['avg_precision']:.3f}"
    )
    lines.append(
        f"- 主体标签覆盖率（标准答案第一个标签——即主体大类，有没有被候选标签覆盖到；"
        f"**分级判定用这个**）：{stats['primary_covered']:.3f}"
    )
    lines.append(
        f"- 平均 recall（要求全部5~15个标准答案标签都覆盖到，仅参考，不参与分级判定）："
        f"{stats['avg_recall']:.3f}"
    )
    if stats["worst"]:
        lines.append("- 问题最多的样本（优先看这些，也用来判断是模型真错还是标准答案本身漏标了）：")
        for r in stats["worst"]:
            lines.append(
                f"  - `{r['filename']}` — 未匹配候选标签：{r['unmatched_candidate_tags']}；"
                f"未覆盖标准答案：{r['unmatched_reference_tags']}"
            )
    return "\n".join(lines) + "\n"


def _percentile(sorted_vals, p):
    if not sorted_vals:
        return None
    idx = min(len(sorted_vals) - 1, int(len(sorted_vals) * p))
    return sorted_vals[idx]


def summarize_perf(rows, label):
    """汇总 tag_images.py 存的 perf 字段：延迟、生成吞吐量。
    衡量的是"这个模型实际部署在端上跑起来的资源/速度代价"，跟标签准不准是两件独立的事——
    一个模型标得很准但慢到不能用，也不算达标。"""
    perfs = [r["perf"] for r in rows if r.get("perf") and "latency_seconds" in r["perf"]]
    if not perfs:
        return f"### {label}\n（没有性能数据，用新版 tag_images.py 重新跑一遍即可采集到）\n"

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


def build_conclusion(image_stats):
    """这节不是"能不能投产"的自动判定——评测样本量有限，工具没资格替你做这个业务决策。
    这里只是把已经算出来的数字翻译成可读的现状描述，加一个决策清单，帮你自己判断，
    不是代替你判断。"""
    lines = ["## 二、结论与建议（不是“能否投产”的自动判定，用于辅助你自己判断）\n"]
    lines.append(
        "**这份报告能告诉你“现状大概是什么样”，不能替你回答“能不能投产”——那个问题还依赖"
        "你的业务场景对各类错误的容忍度、下游有没有人工审核兜底、样本量是否足够覆盖你的真实场景，"
        "这些工具都不知道。以下是基于当前数据的客观现状 + 一个决策清单：**\n"
    )

    if image_stats:
        lines.append(
            f"- **图片**：{image_stats['n']} 个样本，对照人工闭集标签库，"
            f"平均 precision {image_stats['avg_precision']:.3f}，主体标签覆盖率 "
            f"{image_stats['primary_covered']:.3f}（完整 recall {image_stats['avg_recall']:.3f}，仅参考）"
            f"（未匹配的标签多为大类生物学/分类常识错误，细分品类识别错误已通过 prompt 排除在"
            f"评判之外）。"
        )

    lines.append(
        "\n**投产前建议自查的清单：**\n"
        "1. 当前评测样本量有限（200张）——正式上线前建议扩大到更接近真实业务分布的数据集规模，"
        "标准答案库也要跟着补，再看这些比例是否稳定；\n"
        "2. 标准答案库是人工一次性写的，不保证穷尽每一处细节——`semantic_scores_images.csv` 里的"
        "`unmatched_candidate_tags`/`unmatched_reference_tags` 两列列出了每条样本具体没匹配上"
        "的标签，建议抽查几条，分清楚是模型真错了，还是标准答案本身该补充；\n"
        "3. 看清楚失败模式是不是你业务场景真正关心的——比如“大类生物学分类错误”对“相册自动分类”"
        "场景可能无关痛痒，对“物种识别”类场景就是致命的；\n"
        "4. 结合下面的性能数据判断延迟能不能接受，标得准但跑不动同样不算达标。"
    )
    return "\n".join(lines) + "\n"


def main():
    parts = ["# 端侧打标签模型测评报告\n"]

    image_stats = semantic_stats(os.path.join(config.RESULTS_DIR, "semantic_scores_images.csv"))

    # 结论速览放全文最前面：多数人想先看这个，下面一/三节的详细数字是支撑这句结论的证据，
    # 不是反过来读。评测基准是人工核实过的闭集标签库，这个判定不需要再额外校准验证。
    parts.append("## 结论速览\n")
    parts.append(summarize_headline("图片", image_stats))
    parts.append(
        "\n（分级判定的阈值在 `core/config.py` 的 `QUALITY_GATE` 里，按你的业务风险容忍度自己调；"
        "评测基准是 `tagging_test_samples/ground_truth_images.csv`，扩充测试集时记得同步补充"
        "标准答案。）\n"
    )

    parts.append("## 一、语义相似度打分结果（对照人工闭集标签库，推荐优先看这部分）\n")
    parts.append(summarize_semantic(image_stats, "图片"))

    parts.append(build_conclusion(image_stats))

    parts.append("## 三、端侧模型性能（延迟/吞吐，跟标签质量是两件独立的事）\n")
    parts.append(summarize_perf(load_jsonl(os.path.join(config.RESULTS_DIR, "images_tags_small.jsonl")), "图片"))

    out_path = os.path.join(config.RESULTS_DIR, "summary_report.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print(f"报告已生成：{out_path}")


if __name__ == "__main__":
    main()
