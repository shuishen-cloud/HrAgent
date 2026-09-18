"""语音合成（说）：面试官的文字 → 音频字节。

edge-tts 是异步库，这里用 asyncio.run 包成同步接口，
让 run_turn 保持纯同步、调用方（FastAPI 的 sync 路由 / 测试）都简单。
"""
from __future__ import annotations

from . import config


async def _stream_audio(text: str, voice: str) -> bytes:
    """调用 edge-tts 流式合成，把音频分片拼成完整字节。"""
    import edge_tts

    communicate = edge_tts.Communicate(text, voice)
    buf = bytearray()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.extend(chunk["data"])
    return bytes(buf)


def synthesize(text: str, voice: str | None = None) -> bytes:
    """把文字合成为 MP3 音频字节。文字为空则返回空字节。"""
    if not text or not text.strip():
        return b""

    import asyncio

    return asyncio.run(_stream_audio(text, voice or config.TTS_VOICE))
