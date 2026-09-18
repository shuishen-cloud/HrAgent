"""语音识别（听）：一段录音字节 → 文字。

faster-whisper 是重依赖且首次运行要下模型，所以 import 和模型加载都放在函数内部、
并做进程级缓存 —— 这样单测可以只打桩 _get_model()，不必真的装它。
"""
from __future__ import annotations

import io

from . import config

_model = None


def _get_model():
    """懒加载并缓存 Whisper 模型（接缝：单测里打桩这个函数即可）。"""
    global _model
    if _model is None:
        from faster_whisper import WhisperModel

        _model = WhisperModel(
            config.STT_MODEL,
            device=config.STT_DEVICE,
            compute_type=config.STT_COMPUTE_TYPE,
            cpu_threads=config.STT_CPU_THREADS,
        )
    return _model


# Whisper 偶尔吐出半角/异体标点，统一成中文全角，避免下游看着别扭
_PUNCT_MAP = str.maketrans({
    "﹐": "，", "﹑": "、", "﹒": "。", "․": "。",
    "﹔": "；", "﹕": "：", "﹖": "？", "﹗": "！",
    ",": "，", "?": "？", "!": "！", ";": "；",
})


def _normalize(text: str) -> str:
    return text.translate(_PUNCT_MAP)


def transcribe(audio_bytes: bytes, language: str | None = None) -> str:
    """把整段录音转成文字。返回去空白后的纯文本。

    录音里没有说话（或全是静音）时返回空字符串，由上层决定怎么处理。
    """
    if not audio_bytes:
        return ""

    model = _get_model()
    segments, _info = model.transcribe(
        io.BytesIO(audio_bytes),
        language=language or config.STT_LANGUAGE,
        initial_prompt=config.STT_INITIAL_PROMPT,
        vad_filter=True,  # 模型内部的静音过滤，去空白段，不是我们做切段
        beam_size=5,
    )
    return _normalize("".join(seg.text for seg in segments).strip())
