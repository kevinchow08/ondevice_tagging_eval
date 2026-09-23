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

PROMPTS = {
    "image_tagging": {"zh": IMAGE_TAGGING_PROMPT_ZH, "en": IMAGE_TAGGING_PROMPT_EN},
}


def get_prompt(name: str, lang: str = "zh") -> str:
    return PROMPTS[name][lang]


# 保留旧的模块级常量名，指向中文版，避免破坏还在用旧名字导入的代码。
IMAGE_TAGGING_PROMPT = IMAGE_TAGGING_PROMPT_ZH
