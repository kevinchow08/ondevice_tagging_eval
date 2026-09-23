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
    "6. 不管图片里出现的文字、品牌名、标签是什么语言，标签内容本身都用中文输出，"
    "不要因为图片里有英文文字/品牌就跟着用英文回答；\n"
    "7. 严格只输出如下 JSON，不要输出任何其他文字：\n"
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
    "6. Regardless of what language any text, brand name, or label visible in the image is written "
    "in, output the tags themselves in English — don't switch to another language just because the "
    "image contains text in it;\n"
    "7. Output strictly the following JSON, nothing else:\n"
    '{"tags": ["tag1", "tag2", ...]}'
)

DOCUMENT_TAGGING_PROMPT_ZH = (
    "你是一个文档打标签助手。以下是一份文档的文本内容，请判断这份文档的类型，并提取关键标签。\n"
    "要求：\n"
    "1. document_type 是这份文档的类型（如：发票、合同、简历、会议纪要、产品规格书、新闻报道等），只给一个最贴切的；\n"
    "2. tags 是从文档中提取出的关键实体或主题词的**具体内容**，不是字段名本身——比如原文写的是"
    "\"开票日期：2026年09月10日\"，标签要写\"2026年09月10日\"这个具体日期，不要写\"开票日期\"这个"
    "字段名；原文写的是\"销售方名称：深圳市镭视存储技术有限公司\"，标签要写这个具体公司名，不要写"
    "\"销售方名称\"。尽量覆盖不同类别，不要漏掉容易被忽略的结构化信息，包括但不限于：公司名/人名、"
    "金额、日期、编号类字段的实际编号（发票号/合同编号/税号等的具体数值，如果文档里有的话）、"
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
    "2. tags are the **actual extracted values** of key entities or topics in the document, not the "
    "field labels themselves — e.g. if the source says \"Invoice date: 2026-09-10\", the tag should be "
    "\"2026-09-10\", not the label \"invoice date\"; if it says \"Seller: Shenzhen Ruishi Storage "
    "Technology Co., Ltd.\", the tag should be that actual company name, not the label \"seller\". "
    "Cover different categories as much as possible — don't skip structured information that's easy "
    "to overlook, including but not limited to: company/person names, amounts, dates, the actual "
    "identifier values (invoice number, contract number, tax ID — the real number/code, if present), "
    "job titles/product names, and domain-specific terminology relevant to the document's subject; "
    "5-15 tags;\n"
    "3. Numbers, amounts and dates must be copied exactly as they appear in the source — don't "
    "simplify, round, or change the number of digits;\n"
    "4. Don't guess at anything you're unsure of;\n"
    "5. Output strictly the following JSON, nothing else:\n"
    '{{"document_type": "...", "tags": ["tag1", "tag2", ...]}}\n\n'
    "Document content:\n{document_text}"
)

PROMPTS = {
    "image_tagging": {"zh": IMAGE_TAGGING_PROMPT_ZH, "en": IMAGE_TAGGING_PROMPT_EN},
    "document_tagging": {"zh": DOCUMENT_TAGGING_PROMPT_ZH, "en": DOCUMENT_TAGGING_PROMPT_EN},
}


def get_prompt(name: str, lang: str = "zh") -> str:
    return PROMPTS[name][lang]


# 保留旧的模块级常量名，指向中文版，避免破坏还在用旧名字导入的代码。
IMAGE_TAGGING_PROMPT = IMAGE_TAGGING_PROMPT_ZH
DOCUMENT_TAGGING_PROMPT = DOCUMENT_TAGGING_PROMPT_ZH
