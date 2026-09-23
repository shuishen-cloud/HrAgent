"""多模态 LLM 封装：候选人回答文字 + 摄像头帧 → 结构化 JSON 结果。

设计上把「纯逻辑」和「网络调用」分开：
  build_messages / parse_reply  —— 纯函数，可直接单测
  chat                          —— 真正发请求，通过 _get_client() 这个接缝打桩

因此单元测试不需要安装 openai，也不需要 API key。
"""
from __future__ import annotations

import base64
import json
import re

from . import config

# 面试官人设 + 输出契约。改这里就能调整面试风格与判定标准。
DEFAULT_SYSTEM_PROMPT = """你是一位专业、友好的中文面试官，正在通过视频面试一位候选人。

你会同时收到两样东西：
1. 候选人本轮回答的**文字转写**（由语音识别得到，可能有错别字，请合理理解）
2. 本轮回答期间从候选人摄像头采集的**若干张画面帧**（按时间顺序）

你的任务：
- 评估候选人本轮回答的质量
- **观察画面**判断候选人的状态：是否专注看着屏幕、是否频繁转头/离开、画面里是否出现多个人（疑似他人代答）、座位是否空着（候选人离开）
- 决定下一个问题：可以顺着回答追问，也可以换下一题
- 判断面试是否可以结束了

输出要求：**只输出一个 JSON 对象**，不要任何解释性文字、不要 markdown 代码块。格式如下：

{
  "evaluation": {"score": 0到10的整数, "comment": "对本次回答的简短点评"},
  "attention": {"focused": true或false, "note": "对画面观察的简短说明，如：候选人正对镜头，专注"},
  "cheating_suspected": true或false,
  "next_question": "你要问的下一个问题（中文，口语化，像真人面试官说话）",
  "should_end": true或false
}

判定参考：
- 画面里只有候选人一人、正对镜头 → focused=true, cheating_suspected=false
- 候选人频繁转头、长时间不看屏幕、离开画面 → focused=false
- 画面里出现两个人及以上、或明显在看别的屏幕/资料 → cheating_suspected=true
- 注意：短时间低头思考是正常的，不要误判为走神
"""


# 有简历时追加到系统提示后面，让面试官围绕简历提问而不是念题库
RESUME_INSTRUCTIONS = """本次面试有候选人简历，内容如下：

<简历>
{resume}
</简历>

提问要求：
- **围绕这份简历提问**，优先深挖里面写到的项目、技术选型、职责和成果
- 简历里含糊、可疑或值得追问的地方，要顺着问下去（比如只写了「负责优化」却没说
  优化了什么、效果如何）
- 简历里没写但岗位关心的能力，也要覆盖到
- 不要问简历里已经写得很清楚的事实性问题（如「你是哪个学校的」）
- 每次只问一个问题，口语化，像真人面试官说话
"""

# 只在「没有简历」时用的开场话术，有简历时开场问题由 LLM 生成
OPENING_SYSTEM_PROMPT = """你是一位专业、友好的中文面试官，马上要开始一场视频面试。

你会收到候选人的简历。请基于简历内容，说一段**开场白并抛出第一个问题**：
- 先简短寒暄（一句话即可），然后直接问
- 第一个问题要针对简历里最值得深挖的点（通常是最近的项目或最核心的经历）
- 口语化、自然，像真人面试官开口说话，不要写成书面语
- 只输出这段话本身，不要任何前缀、引号或解释
"""

# 简历模式下 STT 失败时，让 LLM 就当前问题生成一段「候选人视角」的回答
SAMPLE_ANSWER_SYSTEM_PROMPT = """你在帮一位候选人准备面试。请针对面试官的问题，
写一段**候选人视角的口语化回答**，用于演示。

要求：
- 直接说答案，不要「好的，我认为」这类铺垫
- 控制在 80 字以内，口语化，像真人在说话
- 内容要合理可信，不要空话套话
- 只输出回答本身，不要任何前缀或解释
"""


def _image_part(frame: bytes, mime: str = "image/jpeg") -> dict:
    """把一帧图片字节包装成 OpenAI 兼容的 image_url 结构。"""
    b64 = base64.b64encode(frame).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def build_messages(
    history: list[dict],
    answer: str,
    frames: list[bytes],
    system: str = DEFAULT_SYSTEM_PROMPT,
) -> list[dict]:
    """拼装发给 LLM 的消息列表。

    history: 之前的对话，形如 [{"role": "assistant", "content": "..."}]
    answer:  候选人本轮回答的文字
    frames:  本轮采集到的画面帧（原始图片字节）
    """
    messages: list[dict] = [{"role": "system", "content": system}]
    messages.extend(history)

    content: list[dict] = [
        {"type": "text", "text": f"候选人本轮的回答（语音转写）：\n{answer or '（候选人没有说话）'}"}
    ]
    if frames:
        content.append({"type": "text", "text": f"以下是本轮回答期间采集的 {len(frames)} 张画面帧："})
        content.extend(_image_part(f) for f in frames)
    else:
        content.append({"type": "text", "text": "（本轮没有采集到画面帧）"})

    messages.append({"role": "user", "content": content})
    return messages


_SAFE_QUESTION = "抱歉，我没太听清。能就刚才的问题再展开说说吗？"


def _fallback_reply(raw: str) -> dict:
    """LLM 没按格式返回时的兜底，保证面试不中断。

    **不要把原文当成下一个问题。** 模型被 max_tokens 截断时，返回的往往是
    半截的 ```json {...} —— 那样这段乱码会被 TTS 念给候选人听，还会写进报告。
    改成一句安全的追问，原文另存到 _raw 供排查。
    """
    return {
        "evaluation": {"score": None, "comment": "（本轮未获得结构化点评）"},
        "attention": {"focused": True, "note": ""},
        "cheating_suspected": False,
        "next_question": _SAFE_QUESTION,
        "should_end": False,
        "_parsed": False,
        "_raw": (raw or "")[:500],
    }


def _extract_json(raw: str) -> dict | None:
    """从模型输出里抠出 JSON 对象：容忍 ```json 代码块和前后多余文字。"""
    if not raw:
        return None
    text = raw.strip()

    # 去掉 markdown 代码块围栏
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()

    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        # 退而求其次：取第一个 { 到最后一个 } 之间的内容
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            obj = json.loads(text[start : end + 1])
        except (json.JSONDecodeError, ValueError):
            return None

    return obj if isinstance(obj, dict) else None


_TRUE_WORDS = {"true", "yes", "y", "1", "是", "真", "有", "对"}
# 注意：这里**不能**包含空串。空串属于「没给出值」，应该走 default，
# 而不是被当成「认得出的 False」—— 否则模型对 focused 输出 "" 时
# 会被记成走神，正好违反 _as_bool 声明的「不轻易给候选人扣帽子」。
_FALSE_WORDS = {"false", "no", "n", "0", "否", "假", "无", "不"}


def _as_bool(value, default: bool) -> bool:
    """把 LLM 可能给出的各种写法转成 bool。

    **不能直接用 bool()**：字符串 "false" 是非空字符串，bool("false") 是 True。
    模型偶尔会把布尔值写成字符串，那样「没有作弊」会被读成「作弊」——
    在招聘场景里这是最不能出错的方向。
    认不出来时返回 default，而不是瞎猜。
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        word = value.strip().lower()
        if word in _TRUE_WORDS:
            return True
        if word in _FALSE_WORDS:
            return False
    return default


def _as_dict_or_text(value) -> tuple[dict, str]:
    """模型有时把本该是对象的字段直接写成一句话。

    返回 (字典, 那句话)。这样既不会崩，也不丢掉它说的内容。
    """
    if isinstance(value, dict):
        return value, ""
    if isinstance(value, str):
        return {}, value.strip()
    return {}, ""


def _as_score(value) -> int | None:
    """分数容错：接受 7 / 7.0 / "7" / "7.5分" 等写法，认不出返回 None。

    结果**夹在 0-10**：报告用的是 8/6 两档阈值，若换了模型或端点后对方按
    百分制给分（85），平均分会变成 85，结论直接失真成「建议通过」。
    """
    if isinstance(value, bool):          # bool 是 int 的子类，得先挡掉
        return None
    if isinstance(value, int):
        number = value
    elif isinstance(value, float):
        number = int(round(value))
    elif isinstance(value, str):
        found = re.search(r"-?\d+(?:\.\d+)?", value)
        if not found:
            return None
        number = int(round(float(found.group())))
    else:
        return None
    return max(0, min(10, number))


def _normalize(obj: dict) -> dict:
    """补齐缺失字段、纠正类型，保证下游拿到稳定的结构。

    任何字段类型异常都不能让它抛异常 —— 这是 parse_reply 的对外承诺。
    """
    evaluation, evaluation_text = _as_dict_or_text(obj.get("evaluation"))
    attention, attention_text = _as_dict_or_text(obj.get("attention"))

    return {
        "evaluation": {
            "score": _as_score(evaluation.get("score")),
            "comment": str(evaluation.get("comment") or evaluation_text or ""),
        },
        "attention": {
            # 认不出来时偏向「专注」：不轻易给候选人扣走神的帽子
            "focused": _as_bool(attention.get("focused"), default=True),
            "note": str(attention.get("note") or attention_text or ""),
        },
        # 同理，偏向「没作弊」：宁可漏报也不误伤
        "cheating_suspected": _as_bool(obj.get("cheating_suspected"), default=False),
        "next_question": str(obj.get("next_question") or "").strip() or "请继续说说你的想法。",
        "should_end": _as_bool(obj.get("should_end"), default=False),
        "_parsed": True,
    }


def parse_reply(raw: str) -> dict:
    """解析模型回复 → 结构化 dict。解析失败走兜底，绝不抛异常。"""
    obj = _extract_json(raw)
    if obj is None:
        return _fallback_reply(raw)
    return _normalize(obj)


def _get_client():
    """建立 LLM 客户端（接缝：单测里打桩这个函数即可）。"""
    from openai import OpenAI

    kwargs = {"api_key": config.LLM_API_KEY, "timeout": config.LLM_TIMEOUT}
    if config.LLM_BASE_URL:
        kwargs["base_url"] = config.LLM_BASE_URL
    return OpenAI(**kwargs)


def system_prompt_for(resume: str = "") -> str:
    """有简历就把简历并进系统提示，让面试官围绕它提问。

    注意：简历会进入**每一轮**请求，输入 token 会明显变大（一份 4000 字简历
    约合 2000+ token）。所以 config.RESUME_MAX_CHARS 做了截断。
    """
    resume = (resume or "").strip()
    if not resume:
        return DEFAULT_SYSTEM_PROMPT
    return DEFAULT_SYSTEM_PROMPT + "\n\n" + RESUME_INSTRUCTIONS.format(resume=resume)


def _plain_completion(system: str, user: str, temperature: float = 0.8) -> str:
    """要一段纯文本（不是 JSON）时的调用。失败抛异常，由上层兜底。"""
    client = _get_client()
    resp = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()


def opening_question(resume: str) -> str:
    """基于简历生成开场白 + 第一个问题。"""
    return _plain_completion(OPENING_SYSTEM_PROMPT, f"候选人简历：\n\n{resume}")


def sample_answer(question: str) -> str:
    """就某个问题生成一段候选人视角的回答（STT 失败时的降级素材）。"""
    return _plain_completion(SAMPLE_ANSWER_SYSTEM_PROMPT, f"面试官的问题是：{question}")


def chat(
    history: list[dict],
    answer: str,
    frames: list[bytes],
    system: str = DEFAULT_SYSTEM_PROMPT,
) -> dict:
    """面试官的核心一次调用：看画面 + 读回答 → 结构化结果。"""
    messages = build_messages(history, answer, frames, system)
    client = _get_client()

    kwargs = {
        "model": config.LLM_MODEL,
        "messages": messages,
        "temperature": 0.7,
    }
    if config.LLM_JSON_MODE:
        kwargs["response_format"] = {"type": "json_object"}

    resp = client.chat.completions.create(**kwargs)
    raw = resp.choices[0].message.content or ""
    return parse_reply(raw)
