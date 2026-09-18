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


def _fallback_reply(raw: str) -> dict:
    """LLM 没按格式返回时的兜底：把原文当成下一个问题，保证面试不中断。"""
    text = (raw or "").strip()
    return {
        "evaluation": {"score": None, "comment": "（本轮未获得结构化点评）"},
        "attention": {"focused": True, "note": ""},
        "cheating_suspected": False,
        "next_question": text or "请继续说说你的想法。",
        "should_end": False,
        "_parsed": False,
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


def _normalize(obj: dict) -> dict:
    """补齐缺失字段、纠正类型，保证下游拿到稳定的结构。"""
    evaluation = obj.get("evaluation") or {}
    attention = obj.get("attention") or {}

    score = evaluation.get("score")
    if isinstance(score, str) and score.strip().isdigit():
        score = int(score)
    if not isinstance(score, int):
        score = None

    return {
        "evaluation": {
            "score": score,
            "comment": str(evaluation.get("comment") or ""),
        },
        "attention": {
            "focused": bool(attention.get("focused", True)),
            "note": str(attention.get("note") or ""),
        },
        "cheating_suspected": bool(obj.get("cheating_suspected", False)),
        "next_question": str(obj.get("next_question") or "").strip() or "请继续说说你的想法。",
        "should_end": bool(obj.get("should_end", False)),
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
