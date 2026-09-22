"""语音识别（听）：一段录音字节 → 文字。

faster-whisper 是重依赖且首次运行要下模型，所以 import 和模型加载都放在函数内部、
并做进程级缓存 —— 这样单测可以只打桩 _get_model()，不必真的装它。
"""
from __future__ import annotations

import io
import os

from . import config

_model = None


def _build_model():
    from faster_whisper import WhisperModel

    return WhisperModel(
        config.STT_MODEL,
        device=config.STT_DEVICE,
        compute_type=config.STT_COMPUTE_TYPE,
        cpu_threads=config.STT_CPU_THREADS,
    )


def _get_model():
    """懒加载并缓存 Whisper 模型（接缝：单测里打桩这个函数即可）。

    加载策略：**先试离线**。
    huggingface_hub 即使模型已缓存，仍会联网检查有没有新版本；国内连不上
    huggingface.co，而且那是「挂起」不是「快速失败」，进程会一直卡住不动
    （实测：跑到第一轮 STT 就停住，不报错也不退出）。走离线模式直接用本地缓存，
    秒开且不碰网络。

    本地确实没有缓存时才回退到联网下载，并指向镜像站。
    """
    global _model
    if _model is not None:
        return _model

    # 环境变量只在这段加载期间生效，出来就还原，避免污染进程里的其他 HF 调用
    prev_offline = os.environ.get("HF_HUB_OFFLINE")
    prev_endpoint = os.environ.get("HF_ENDPOINT")
    try:
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            _model = _build_model()
        except Exception:
            # 离线加载失败 = 本地确实没缓存，这时才联网，并指向镜像站
            os.environ.pop("HF_HUB_OFFLINE", None)
            os.environ.setdefault("HF_ENDPOINT", config.HF_ENDPOINT)
            _model = _build_model()
    finally:
        _restore_env("HF_HUB_OFFLINE", prev_offline)
        _restore_env("HF_ENDPOINT", prev_endpoint)

    return _model


def _restore_env(key: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value


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
