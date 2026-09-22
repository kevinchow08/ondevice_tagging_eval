# -*- coding: utf-8 -*-
"""
所有发给模型的提示词，集中放这里方便你自己调整措辞。

中英文都有，用 get_prompt(name, lang) 取。lang 默认 "zh"，各脚本可以用 --lang en 切换成英文版，
测试"同样的规则用不同语言表达，模型表现会不会不一样"这类问题，也方便英文用户直接用。
"""

IMAGE_TAGGING_PROMPT_ZH = (
    "你是一个图像打标签助手。请仔细观察这张图片，输出你认为准确描述图片内容的标签。\n"
    "要求：\n"
    "1. 标签是名词或简短名词短语，不要输出完整句子；\n"
    "2. 只给大类标签，不要给出具体的物种/品种/型号/学名（比如给\"鸟\"\"蜥蜴\"\"蜘蛛\"\"花\"\"汽车\"，"
    "不需要判断具体是哪个物种/哪个型号）——大类判断准确就够用，不需要，也不要尝试展示专业知识；\n"
    "3. 除了主体大类，尽量补充其他容易确定、不需要专业知识就能看出来的信息，比如：颜色、数量、"
    "动作或姿态、所处环境/背景、材质、场景，这些和主体大类同样重要，不要漏掉；\n"
    "4. 只输出 5~15 个最相关的标签，不要堆砌；每个标签必须是不同的信息点。如果两个候选标签的核心"
    "含义重合（一个是另一个的近义词、更细的写法，或只是多加了个修饰词），只保留信息量更大的那一个，"
    "不要两个都写（比如\"鸟\"和\"鸟类\"、\"花\"和\"花朵\"、\"骨架\"和\"骨骼标本\"，这种每组只留一个）；\n"
    "5. 不确定的内容宁可粗略也不要瞎猜细节；\n"
    "6. 严格只输出如下 JSON，不要输出任何其他文字：\n"
    '{"tags": ["标签1", "标签2", ...]}'
)

IMAGE_TAGGING_PROMPT_EN = (
    "You are an image tagging assistant. Look at this image carefully and output tags that "
    "accurately describe its content.\n"
    "Requirements:\n"
    "1. Tags are nouns or short noun phrases, not full sentences;\n"
    "2. Only give general categories, not specific species/breed/model/scientific names (e.g. give "
    "\"bird\", \"lizard\", \"spider\", \"flower\", \"car\" — do not try to identify the exact species "
    "or exact model). Getting the general category right is enough; don't try to show off specialized "
    "knowledge;\n"
    "3. Besides the main subject's category, add other details that are easy to determine without "
    "specialized knowledge: color, count, action/pose, setting/background, material, scene. These "
    "matter just as much as the main category — don't skip them;\n"
    "4. Output 5-15 of the most relevant tags, don't pad the list. Every tag must be a distinct piece "
    "of information. If two candidate tags overlap in core meaning (one is a synonym, a more specific "
    "phrasing, or just the other with an extra modifier), keep only the one that carries more "
    "information, not both (e.g. \"bird\" and \"birds\", \"flower\" and \"blossom\", \"skeleton\" and "
    "\"bone specimen\" — keep only one from each pair);\n"
    "5. If you're not sure, stay general rather than guessing at detail;\n"
    "6. Output strictly the following JSON, nothing else:\n"
    '{"tags": ["tag1", "tag2", ...]}'
)

DOCUMENT_TAGGING_PROMPT_ZH = (
    "你是一个文档打标签助手。以下是一份文档的文本内容，请判断这份文档的类型，并提取关键标签。\n"
    "要求：\n"
    "1. document_type 是这份文档的类型（如：发票、合同、简历、会议纪要、产品规格书、新闻报道等），只给一个最贴切的；\n"
    "2. tags 是从文档中提取出的关键实体或主题词，尽量覆盖不同类别，不要漏掉容易被忽略的结构化信息，"
    "包括但不限于：公司名/人名、金额、日期、编号类字段（发票号/合同编号/税号等，如果文档里有的话）、"
    "职位/产品名，以及文档主题相关的专业术语或技术名词；5~15个；\n"
    "3. 数字、金额、日期这类信息要如实按文档原文摘录，不要自己简化、改写位数或四舍五入；\n"
    "4. 不确定的内容不要瞎猜；\n"
    "5. 严格只输出如下 JSON，不要输出任何其他文字：\n"
    '{{"document_type": "...", "tags": ["标签1", "标签2", ...]}}\n\n'
    "文档内容：\n{document_text}"
)

DOCUMENT_TAGGING_PROMPT_EN = (
    "You are a document tagging assistant. Below is the text content of a document. Determine the "
    "document's type and extract key tags.\n"
    "Requirements:\n"
    "1. document_type is this document's type (e.g. invoice, contract, resume, meeting minutes, "
    "product spec sheet, news article), give only the single best fit;\n"
    "2. tags are key entities or topic words extracted from the document, covering different "
    "categories as much as possible — don't skip structured information that's easy to overlook, "
    "including but not limited to: company/person names, amounts, dates, identifier-type fields "
    "(invoice number, contract number, tax ID, if present), job titles/product names, and domain-"
    "specific terminology relevant to the document's subject; 5-15 tags;\n"
    "3. Numbers, amounts and dates must be copied exactly as they appear in the source — don't "
    "simplify, round, or change the number of digits;\n"
    "4. Don't guess at anything you're unsure of;\n"
    "5. Output strictly the following JSON, nothing else:\n"
    '{{"document_type": "...", "tags": ["tag1", "tag2", ...]}}\n\n'
    "Document content:\n{document_text}"
)

IMAGE_JUDGE_PROMPT_ZH = (
    "你是标签质量评审员。请看这张图片，比较下面两组【各自独立打出来的】标签，判断哪一组更准确。\n"
    "两组标签是不同模型各自看图独立打的，互相不知道对方说了什么，请不要假设哪组一定更权威，"
    "完全依据你自己对图片内容的判断来评审。\n"
    "{ground_truth_hint}\n"
    "A组（candidate，待评审）：{candidate_tags}\n"
    "B组（comparison，对照组）：{comparison_tags}\n\n"
    "评审标准（重要，请严格遵守）：这个打标签任务只要求给出大类标签（比如\"鸟\"\"蜥蜴\"\"花\"），"
    "不要求、也不鼓励识别具体的物种/品种/型号/学名。所以：\n"
    "- 标签没有精确到具体物种/品种/型号，不算漏打，不要因为这个扣分；\n"
    "- 只有当标签连大类都判断错了（比如把蜘蛛叫成昆虫、把猫叫成狗），或者标签内容和图片明显不符，"
    "才算瞎编；\n"
    "- missing_important_tags 只填图片里明显存在、大类层面就能看出来但A组漏掉的内容"
    "（比如背景里明显的其他物体、颜色、动作这类，而不是更细的物种/型号）；\n"
    "- 如果一组标签堆砌了大量重复或近义反复、缺乏信息量（比如同一个意思用不同说法写了好几遍），"
    "这个要在判断哪组更好时算作扣分项。\n\n"
    "请你直接对着图片核实两组标签各自的问题，然后给出：\n"
    "1. hallucinated_tags：**A组**里编造的/大类判断错误的标签；\n"
    "2. missing_important_tags：图片里明显存在，但**A组**没提到的大类层面内容；\n"
    "3. winner：综合两组各自的瞎编情况、漏打情况、信息量，判断哪组整体更准确——"
    "\"candidate\"(A组更好)、\"comparison\"(B组更好)、\"tie\"(两组质量相当，都对或都有类似程度的问题)；\n"
    "4. comment：一句话说明你为什么这样判断。\n\n"
    "严格只输出以下 JSON，不要输出其他文字：\n"
    '{{"hallucinated_tags": ["A组里编造的、图片中并不存在的标签，或大类判断错误的标签"], '
    '"missing_important_tags": ["图片中明显存在但A组漏打的大类层面标签"], '
    '"winner": "candidate 或 comparison 或 tie", '
    '"comment": "一句话点评"}}'
)

IMAGE_JUDGE_PROMPT_EN = (
    "You are a tag quality reviewer. Look at this image and compare the two tag sets below, each "
    "produced independently by a different model. Neither model saw the other's output, so don't "
    "assume either one is automatically more authoritative — judge purely from what you see in the "
    "image.\n"
    "{ground_truth_hint}\n"
    "Set A (candidate, under review): {candidate_tags}\n"
    "Set B (comparison): {comparison_tags}\n\n"
    "Review criteria (important, follow strictly): this tagging task only asks for general-category "
    "tags (e.g. \"bird\", \"lizard\", \"flower\"), not specific species/breed/model/scientific names "
    "— that's neither required nor encouraged. So:\n"
    "- Not being precise down to species/breed/model does NOT count as missing — don't penalize for "
    "that;\n"
    "- Only count something as hallucinated if the general category itself is wrong (e.g. calling a "
    "spider an insect, a cat a dog), or the tag clearly doesn't match the image;\n"
    "- missing_important_tags should only list things clearly present in the image and identifiable "
    "at the general-category level that Set A left out (e.g. another obvious object in the "
    "background, a color, an action — not finer species/model detail);\n"
    "- If a set is padded with a lot of exact or near-synonym repetition that adds no real "
    "information, count that against it when deciding which set is better.\n\n"
    "Check both sets directly against the image, then give:\n"
    "1. hallucinated_tags: tags in **Set A** that are made up or get the general category wrong;\n"
    "2. missing_important_tags: general-category-level things clearly present in the image that "
    "**Set A** left out;\n"
    "3. winner: weighing hallucination, omission and informativeness for both sets, which is overall "
    "more accurate — \"candidate\" (A is better), \"comparison\" (B is better), or \"tie\" (comparable "
    "quality, whether both are good or both have similar-degree issues);\n"
    "4. comment: one sentence explaining your reasoning.\n\n"
    "Output strictly the following JSON, nothing else:\n"
    '{{"hallucinated_tags": ["tags in Set A that are made up and aren\'t in the image, or where the '
    'general category itself is wrong"], '
    '"missing_important_tags": ["general-category-level tags clearly present in the image but Set A left out"], '
    '"winner": "candidate or comparison or tie", '
    '"comment": "one-sentence assessment"}}'
)

DOCUMENT_JUDGE_PROMPT_ZH = (
    "你是标签质量评审员。下面是一份文档的原文，以及两组【各自独立生成的】文档类型和标签。\n"
    "两组是不同模型各自看原文独立给出的，互相不知道对方说了什么，请不要假设哪组一定更权威，"
    "完全依据原文内容自己判断。\n"
    "{ground_truth_hint}\n"
    "文档原文：\n{document_text}\n\n"
    "A组（candidate，待评审）document_type：{candidate_type}，tags：{candidate_tags}\n"
    "B组（comparison，对照组）document_type：{comparison_type}，tags：{comparison_tags}\n\n"
    "请你直接对着原文核实两组各自的问题，然后给出：\n"
    "1. type_correct：A组的 document_type 判断得对不对（true/false）；\n"
    "2. hallucinated_tags：**A组**里编造的、原文中并不存在的标签（数字/金额/日期这类要核对是否跟"
    "原文完全一致，改写、四舍五入、抄错位数都算瞎编）；\n"
    "3. missing_important_tags：原文里明显存在，但**A组**没提到的重要标签；\n"
    "4. winner：综合两组各自的瞎编情况、漏打情况、document_type是否判断对，判断哪组整体更准确——"
    "\"candidate\"(A组更好)、\"comparison\"(B组更好)、\"tie\"(两组质量相当)；\n"
    "5. comment：一句话说明你为什么这样判断。\n\n"
    "严格只输出以下 JSON，不要输出其他文字：\n"
    '{{"type_correct": true或false, '
    '"hallucinated_tags": ["A组里编造的、原文中并不存在的标签"], '
    '"missing_important_tags": ["原文中明显存在但A组漏打的重要标签"], '
    '"winner": "candidate 或 comparison 或 tie", '
    '"comment": "一句话点评"}}'
)

DOCUMENT_JUDGE_PROMPT_EN = (
    "You are a tag quality reviewer. Below is a document's original text, along with two "
    "independently-generated sets of document type and tags. Neither model saw the other's output, "
    "so don't assume either one is automatically more authoritative — judge purely from the "
    "original text.\n"
    "{ground_truth_hint}\n"
    "Document text:\n{document_text}\n\n"
    "Set A (candidate, under review) document_type: {candidate_type}, tags: {candidate_tags}\n"
    "Set B (comparison) document_type: {comparison_type}, tags: {comparison_tags}\n\n"
    "Check both sets directly against the original text, then give:\n"
    "1. type_correct: is Set A's document_type correct (true/false);\n"
    "2. hallucinated_tags: tags in **Set A** that are made up and aren't in the original text "
    "(for numbers/amounts/dates, check they exactly match the source — rewording, rounding, or a "
    "wrong digit all count as hallucinated);\n"
    "3. missing_important_tags: important things clearly present in the original text that "
    "**Set A** left out;\n"
    "4. winner: weighing hallucination, omission, and whether document_type was correct for both "
    "sets, which is overall more accurate — \"candidate\" (A is better), \"comparison\" (B is "
    "better), or \"tie\" (comparable quality);\n"
    "5. comment: one sentence explaining your reasoning.\n\n"
    "Output strictly the following JSON, nothing else:\n"
    '{{"type_correct": true or false, '
    '"hallucinated_tags": ["tags in Set A that are made up and aren\'t in the original text"], '
    '"missing_important_tags": ["important tags clearly present in the original text but Set A left out"], '
    '"winner": "candidate or comparison or tie", '
    '"comment": "one-sentence assessment"}}'
)

PROMPTS = {
    "image_tagging": {"zh": IMAGE_TAGGING_PROMPT_ZH, "en": IMAGE_TAGGING_PROMPT_EN},
    "document_tagging": {"zh": DOCUMENT_TAGGING_PROMPT_ZH, "en": DOCUMENT_TAGGING_PROMPT_EN},
    "image_judge": {"zh": IMAGE_JUDGE_PROMPT_ZH, "en": IMAGE_JUDGE_PROMPT_EN},
    "document_judge": {"zh": DOCUMENT_JUDGE_PROMPT_ZH, "en": DOCUMENT_JUDGE_PROMPT_EN},
}


def get_prompt(name: str, lang: str = "zh") -> str:
    return PROMPTS[name][lang]


def ground_truth_hint(label, lang: str = "zh") -> str:
    """给裁判 prompt 里 {ground_truth_hint} 占位符用的那句话。label 是 manifest 里预先标好的
    弱参考（图片的 reference_label_en / 文档的 document_type_cn），可选——传空/None 就返回空字符串，
    裁判退化成纯靠自己看图/看原文判断，不强制要求每条样本都有这个标注。
    故意强调"不代表标准的标签表达方式，只是帮你核对事实"，不要让裁判把这当成必须原样出现在标签里的
    精确匹配目标——那样会误伤"只给大类、不给细分物种"这种我们主动要求的正确行为。"""
    if not label:
        return ""
    if lang == "zh":
        return (
            f"供参考（不代表标准的标签表达方式，只是帮你核对关键事实，不要求标签里必须原样出现这个词）："
            f"已知的标准分类/类型是「{label}」。\n"
        )
    return (
        f"For reference (not the required tag wording, just a fact to help you verify — the tags "
        f'don\'t need to literally contain this word): the known ground-truth category/type is "{label}".\n'
    )


# 保留旧的模块级常量名，指向中文版，避免破坏还在用旧名字导入的代码。
IMAGE_TAGGING_PROMPT = IMAGE_TAGGING_PROMPT_ZH
DOCUMENT_TAGGING_PROMPT = DOCUMENT_TAGGING_PROMPT_ZH
IMAGE_JUDGE_PROMPT = IMAGE_JUDGE_PROMPT_ZH
DOCUMENT_JUDGE_PROMPT = DOCUMENT_JUDGE_PROMPT_ZH
