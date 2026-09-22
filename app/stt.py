"""语音识别（听）：一段录音字节 → 文字。

faster-whisper 是重依赖且首次运行要下模型，所以 import 和模型加载都放在函数内部、
并做进程级缓存 —— 这样单测可以只打桩 _get_model()，不必真的装它。
"""
from __future__ import annotations

import io
import os

from . import config

_model = None


def _build_model(local_files_only: bool):
    from faster_whisper import WhisperModel

    return WhisperModel(
        config.STT_MODEL,
        device=config.STT_DEVICE,
        compute_type=config.STT_COMPUTE_TYPE,
        cpu_threads=config.STT_CPU_THREADS,
        local_files_only=local_files_only,
    )


def _get_model():
    """懒加载并缓存 Whisper 模型（接缝：单测里打桩这个函数即可）。

    加载策略：**先试纯本地，本地没缓存才联网走镜像**。

    这两个坑根子是同一件事：**huggingface_hub 在首次 import 时就把
    HF_HUB_OFFLINE / HF_ENDPOINT / HF_HUB_DISABLE_XET 读成模块常量**
    （constants.py），之后再改 os.environ 一律不生效。

    1. **不能用 HF_HUB_OFFLINE 表达「先离线」。** 只要 `from faster_whisper
       import` 之前它已经存在，常量就被锁成 True —— 之后再把环境变量删掉也
       回不去，回退联网必然以 OfflineModeIsEnabled 告终。于是「先离线」实际
       变成了「永远离线」，**新机器永远下不了模型**。所以离线改用
       `local_files_only` 参数（faster-whisper 原生支持，正是为这个场景设计的）。
    2. **镜像地址必须在 import 之前设好**，否则不生效，会直连 huggingface.co ——
       国内那是「丢包挂起」不是「快速失败」，进程会一直卡住不动，比报错更难查。
    """
    global _model
    if _model is not None:
        return _model

    # 必须赶在 `from faster_whisper import ...` 之前，理由见上面第 2 条。
    # 用 setdefault：用户显式设过就尊重用户的值。
    os.environ.setdefault("HF_ENDPOINT", config.HF_ENDPOINT)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # xet 存储后端在国内也不通

    try:
        # 纯本地读缓存：秒开、不碰网络（huggingface_hub 即使已缓存也会联网查更新）
        _model = _build_model(local_files_only=True)
    except Exception:
        # 本地确实没缓存，这时才联网；上面设的镜像地址此时已生效
        _model = _build_model(local_files_only=False)

    return _model


# Whisper 偶尔吐出半角/异体标点，统一成中文全角，避免下游看着别扭
_PUNCT_MAP = str.maketrans({
    "﹐": "，", "﹑": "、", "﹒": "。", "․": "。",
    "﹔": "；", "﹕": "：", "﹖": "？", "﹗": "！",
    ",": "，", "?": "？", "!": "！", ";": "；",
})


def _normalize(text: str) -> str:
    return text.translate(_PUNCT_MAP)


# Whisper 在「无效音频」（极短、底噪、切在半句话上）上不会老实返回空，
# 而是**幻觉出固定话术**。最常见的就是把 initial_prompt 原样吐回来，
# 也有这些社区里广为人知的幻听句。
# 只收「组合起来才成立」的完整话术。像「订阅」「转发」这种**单词不能要** ——
# 候选人正常回答里完全可能说「我订阅了一个技术专栏」，误杀真回答比漏判更糟。
_HALLUCINATION_PHRASES = (
    "请用简体中文", "以下是普通话",
    "谢谢观看", "谢谢大家观看",
    "请不吝点赞", "点赞订阅", "订阅转发",
    "字幕由", "字幕志愿者", "字幕组",
    "amara.org", "明镜与点点",
)


def _looks_like_hallucination(text: str) -> bool:
    """判断这段转写是不是「模型在没话说时编出来的」。

    这类输出**不是空串**，所以只判断 `if not text` 是拦不住的 ——
    它会冒充成正常回答进入报告，比空串更危险。
    """
    stripped = text.strip()
    if not stripped:
        return True

    # 把 initial_prompt 原样或大段吐回来 —— 最强的幻觉信号
    prompt = config.STT_INITIAL_PROMPT.strip()
    if prompt:
        core = prompt.rstrip("。").strip()
        if stripped in prompt or (core and core in stripped):
            return True

    return any(phrase in stripped for phrase in _HALLUCINATION_PHRASES)


def transcribe(audio_bytes: bytes, language: str | None = None) -> str:
    """把整段录音转成文字。返回去空白后的纯文本。

    **不可信的输入一律返回空字符串**（而不是垃圾文字），上层据此降级。
    三种情况会返回空：
      - 没录到东西 / 全是静音
      - 音频太短（短音频会让 Whisper 幻觉出提示词）
      - 输出命中幻觉话术（见 _looks_like_hallucination）
    """
    if not audio_bytes:
        return ""

    model = _get_model()
    segments, info = model.transcribe(
        io.BytesIO(audio_bytes),
        language=language or config.STT_LANGUAGE,
        initial_prompt=config.STT_INITIAL_PROMPT,
        vad_filter=True,  # 模型内部的静音过滤，去空白段，不是我们做切段
        beam_size=5,
    )
    text = _normalize("".join(seg.text for seg in segments).strip())

    # 太短就别信：实测 0.3 秒音频会吐回 "请用简体中文转写。"
    if getattr(info, "duration", 0) < config.STT_MIN_SECONDS:
        return ""
    if _looks_like_hallucination(text):
        return ""
    return text
