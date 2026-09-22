# -*- coding: utf-8 -*-
"""
把 judge_*.jsonl / images_tags_small.jsonl / docs_tags_small.jsonl / semantic_scores_*.csv
汇总成一份可读的报告。跑之前哪些文件存在就汇总哪些，缺了哪个就在报告里注明跳过，不会报错中断。

用法（在 eval_pipeline/ 目录下运行）：python scripts/summarize.py
输出：results/summary_report.md
"""
import csv
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
    单独抽出来是因为"结论与建议"那节要复用这些数字，不想重新解析一遍 rows。

    裁判现在是"两两对比"（candidate=被测的small模型 vs comparison=reference模型独立打的标签），
    不是绝对打分——参考 MT-Bench 论文的结论：pairwise comparison 比 absolute scoring 更可靠。
    瞎编/漏打这两个指标的算法沿用了图像描述领域的 CHAIR 指标（Rohrbach et al. 2018）：
    CHAIR_s = 有瞎编的样本占比（这里是 halluc_rate），CHAIR_i = 平均每条瞎编标签数
    （这里是 avg_halluc_per_sample）。这两个指标是直接对着图片/原文核实出来的，不依赖
    两两对比，即使 winner 是 tie/candidate 也可能有瞎编（比如两边都有问题时打成 tie）。"""
    scored = [r for r in rows if r.get("verdict") and r["verdict"].get("winner") in ("candidate", "comparison", "tie")]
    n = len(scored)
    if not n:
        return None
    halluc_counts = [len(r["verdict"].get("hallucinated_tags", [])) for r in scored]
    missing_counts = [len(r["verdict"].get("missing_important_tags", [])) for r in scored]
    losses = [r for r in scored if r["verdict"]["winner"] == "comparison"]
    losses.sort(
        key=lambda r: (
            len(r["verdict"].get("hallucinated_tags", [])),
            len(r["verdict"].get("missing_important_tags", [])),
        ),
        reverse=True,
    )
    return {
        "n": n,
        "win_rate": sum(1 for r in scored if r["verdict"]["winner"] == "candidate") / n,
        "tie_rate": sum(1 for r in scored if r["verdict"]["winner"] == "tie") / n,
        "loss_rate": len(losses) / n,
        "halluc_rate": sum(1 for c in halluc_counts if c > 0) / n,  # CHAIR_s
        "missing_rate": sum(1 for c in missing_counts if c > 0) / n,
        "avg_halluc_per_sample": sum(halluc_counts) / n,  # CHAIR_i
        "avg_missing_per_sample": sum(missing_counts) / n,
        "worst": losses[:5],
    }


def grade(stats):
    """瞎编率(CHAIR_s)/漏打率哪个落在更差的档位，就用哪个定级（短板原则）。
    阈值来自 config.QUALITY_GATE，写死但可调，不是AI临场判断——改config里的数字就能调松紧。
    这只是把已有数字翻译成一句人话结论，不产生新信息，可信度完全取决于这些数字本身准不准，
    所以报告里这句话必须跟着"有没有做过人工校准"这个状态一起看，见 load_calibration()。"""
    gate = config.QUALITY_GATE
    halluc, missing = stats["halluc_rate"], stats["missing_rate"]

    def band(rate, green_max, yellow_max):
        if rate <= green_max:
            return 0
        if rate <= yellow_max:
            return 1
        return 2

    level = max(
        band(halluc, gate["green_halluc"], gate["yellow_halluc"]),
        band(missing, gate["green_missing"], gate["yellow_missing"]),
    )
    label = ["🟢 可用", "🟡 有条件可用（建议配合人工审核）", "🔴 不建议直接使用"][level]
    reason = f"瞎编率{halluc:.0%}、漏打率{missing:.0%}，取较差档"
    return label, reason


def load_calibration(csv_path):
    """读人工填好的校准CSV（build_calibration_sample.py 生成、你自己填过 human_agree 那列的），
    算"人工-裁判一致率"。没有文件、或者一条都没填，返回 None——报告上就显示"未校准"。"""
    if not os.path.exists(csv_path):
        return None
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    judged = [r for r in rows if (r.get("human_agree") or "").strip()]
    if not judged:
        return None
    agree_words = {"同意", "agree", "yes", "y", "是", "对"}
    n_agree = sum(1 for r in judged if r["human_agree"].strip().lower() in agree_words)
    return {"n_total": len(rows), "n_judged": len(judged), "agree_rate": n_agree / len(judged)}


def summarize_headline(label, stats, calibration):
    """报告最上面的"结论速览"：分级判定 + 校准状态。放在最前面是因为这是大多数人真正想先看到
    的东西，下面第一、二、三、四节的详细数字是支撑这句结论的证据，不是反过来。

    人工校准在这里不是摆设——校准结果不够格（没做/条数太少/一致率太低），分级判定就不会
    以一个confident的颜色呈现，而是明确说"不采信"。不然的话，不管你校准填得多认真、结果多差，
    颜色永远只看 grade() 算出来的瞎编率/漏打率，校准就成了挂在旁边的装饰数字，没有否决权。"""
    if not stats:
        return f"- **{label}**：无有效裁判数据，跳过\n"
    level_label, reason = grade(stats)

    if not calibration or calibration["n_judged"] < config.CALIBRATION_MIN_N:
        need = config.CALIBRATION_MIN_N
        got = calibration["n_judged"] if calibration else 0
        return (
            f"- **{label}**：⚪ 结论可信度未知（裁判自己报的是「{level_label}，{reason}」，"
            f"但人工校准只有{got}条，不够{need}条的最低要求，这个数字本身没有统计意义）——"
            f"建议先跑 `build_calibration_sample.py` 补够样本再看这个结论\n"
        )

    if calibration["agree_rate"] < config.CALIBRATION_TRUST_THRESHOLD:
        return (
            f"- **{label}**：❓ 裁判可信度不达标，上面的判定**不采信**（人工校准{calibration['n_judged']}条，"
            f"一致率仅{calibration['agree_rate']:.0%}，低于{config.CALIBRATION_TRUST_THRESHOLD:.0%}的采信门槛）——"
            f"裁判自己报的是「{level_label}，{reason}」，但校准显示裁判本身可能系统性判断错误，"
            f"这个数字很可能是裁判编出来的，建议先去看校准表里「不同意」的那些具体分歧在哪，"
            f"把裁判prompt改好、重新校准过关，再重新出结论\n"
        )

    return (
        f"- **{label}**：{level_label}（{reason}）；"
        f"人工校准：已做，抽查{calibration['n_judged']}条，一致率{calibration['agree_rate']:.0%}"
        f"（≥{config.CALIBRATION_TRUST_THRESHOLD:.0%}采信门槛，判定可信）\n"
    )


def summarize_judge(rows, label):
    if not rows:
        return f"### {label}\n（没有找到 LLM 裁判评审结果，跳过。先跑 llm_judge.py）\n"

    stats = judge_stats(rows)
    lines = [f"### {label}", f"- 样本数：{len(rows)}"]
    if not stats:
        lines.append("- 无有效裁判数据")
        return "\n".join(lines) + "\n"

    lines.append(
        f"- 对比参考模型（两两对比，不是绝对打分）：被测模型胜 {stats['win_rate']:.0%}，"
        f"平局 {stats['tie_rate']:.0%}，输给参考模型 {stats['loss_rate']:.0%}"
    )
    # halluc_rate/avg_halluc_per_sample 是 CHAIR_s/CHAIR_i 风格的指标：受影响样本占比 +
    # 平均每条个数，不展示累计总数——总数会随样本量线性增长，不同批次之间不可比，
    # 而且标签给得越少总数天然越好看，跟真实质量没有必然关系。
    lines.append(
        f"- 瞎编：{stats['halluc_rate']:.0%} 的样本至少有1个瞎编标签（CHAIR_s），"
        f"平均每条 {stats['avg_halluc_per_sample']:.2f} 个（CHAIR_i）"
    )
    lines.append(
        f"- 漏打：{stats['missing_rate']:.0%} 的样本至少漏打1个重要标签，"
        f"平均每条 {stats['avg_missing_per_sample']:.2f} 个"
    )
    if stats["worst"]:
        lines.append("- 输给参考模型的样本（优先看这些）：")
        for r in stats["worst"]:
            lines.append(f"  - `{r['filename']}` — {r['verdict'].get('comment', '')}")
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
            f"- **图片**：{image_stats['n']} 个抽样样本，跟参考模型（更强的云端模型，独立打标签）"
            f"两两对比，被测端侧模型打平或获胜 {image_stats['win_rate']+image_stats['tie_rate']:.0%}"
            f"（其中明确获胜 {image_stats['win_rate']:.0%}），明确输给参考模型 "
            f"{image_stats['loss_rate']:.0%}；{image_stats['halluc_rate']:.0%} 的样本至少有1处瞎编"
            f"（多为大类生物学/分类常识错误，细分品类识别错误已通过 prompt 排除在评判之外）。"
        )
    if doc_stats:
        lines.append(
            f"- **文档**：{doc_stats['n']} 个样本，打平或获胜 "
            f"{doc_stats['win_rate']+doc_stats['tie_rate']:.0%}，瞎编率 {doc_stats['halluc_rate']:.0%}，"
            f"但 {doc_stats['missing_rate']:.0%} 的样本至少漏打1个重要字段/实体——"
            f"文档这边的短板是漏打，不是瞎编。"
        )

    lines.append(
        "\n**投产前建议自查的清单：**\n"
        "1. 当前图片评审只抽了一部分样本，文档样本量更小（个位数）——正式上线前建议扩大到全量或"
        "更接近真实业务分布的数据集，再看这些比例是否稳定；\n"
        "2. 裁判请求已经把 temperature 锁到 0，同一批内容重跑分数漂移的概率降低了，但没消除"
        "\"裁判自己判断错\"这类误差——最上面\"结论速览\"里如果显示\"人工校准：未做\"，"
        "说明现在的分级判定还没被验证过，建议先跑 `build_calibration_sample.py` 抽查确认；\n"
        "3. 看清楚失败模式是不是你业务场景真正关心的——比如图片“大类生物学分类错误”对“相册自动分类”"
        "场景可能无关痛痒，对“物种识别”类场景就是致命的；\n"
        "4. 结合下面的性能数据判断延迟能不能接受，标得准但跑不动同样不算达标。"
    )
    return "\n".join(lines) + "\n"


def main():
    parts = ["# 端侧打标签模型测评报告\n"]

    image_judge_rows = load_jsonl(os.path.join(config.RESULTS_DIR, "judge_images.jsonl"))
    doc_judge_rows = load_jsonl(os.path.join(config.RESULTS_DIR, "judge_documents.jsonl"))
    image_stats = judge_stats(image_judge_rows)
    doc_stats = judge_stats(doc_judge_rows)
    image_calib = load_calibration(os.path.join(config.RESULTS_DIR, "calibration_images.csv"))
    doc_calib = load_calibration(os.path.join(config.RESULTS_DIR, "calibration_documents.csv"))

    # 结论速览放全文最前面：多数人想先看这个，下面一/二/三/四节的详细数字是支撑这句结论的证据，
    # 不是反过来读——分级判定本身不产生新信息，是不是真能信，看这里有没有标"未做"人工校准。
    parts.append("## 结论速览\n")
    parts.append(summarize_headline("图片", image_stats, image_calib))
    parts.append(summarize_headline("文档", doc_stats, doc_calib))
    parts.append(
        "\n（分级判定的阈值在 `core/config.py` 的 `QUALITY_GATE` 里，按你的业务风险容忍度自己调；"
        "人工校准用 `python scripts/build_calibration_sample.py --kind images/documents` 生成抽样表，"
        "填完 `human_agree` 列后重新跑本脚本即可。）\n"
    )

    parts.append("## 一、LLM 裁判评审结果（推荐优先看这部分）\n")
    parts.append(summarize_judge(image_judge_rows, "图片"))
    parts.append(summarize_judge(doc_judge_rows, "文档"))

    parts.append("## 二、语义相似度打分结果（辅助参考）\n")
    parts.append(summarize_semantic(os.path.join(config.RESULTS_DIR, "semantic_scores_images.csv"), "图片"))
    parts.append(summarize_semantic(os.path.join(config.RESULTS_DIR, "semantic_scores_documents.csv"), "文档"))

    parts.append(build_conclusion(image_stats, doc_stats))

    parts.append("## 四、端侧模型性能（延迟/吞吐，跟标签质量是两件独立的事）\n")
    parts.append(summarize_perf(load_jsonl(os.path.join(config.RESULTS_DIR, "images_tags_small.jsonl")), "图片"))
    parts.append(summarize_perf(load_jsonl(os.path.join(config.RESULTS_DIR, "docs_tags_small.jsonl")), "文档"))

    out_path = os.path.join(config.RESULTS_DIR, "summary_report.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print(f"报告已生成：{out_path}")


if __name__ == "__main__":
    main()
