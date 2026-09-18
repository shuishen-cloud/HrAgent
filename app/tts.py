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
    """把文字合成为 MP3 音频字节。文字为空则返回空字节。

    **必须带超时**：edge-tts 连上服务端后不再回数据（限流/网络抖动）时，
    `asyncio.run` 会永不返回。而调用方是在持有全局锁的情况下调它的，
    一次挂起就会让整个服务（含 /api/status）永久无响应，且没有任何报错。
    """
    if not text or not text.strip():
        return b""

    import asyncio

    async def _run() -> bytes:
        return await asyncio.wait_for(
            _stream_audio(text, voice or config.TTS_VOICE),
            timeout=config.TTS_TIMEOUT,
        )

    try:
        return asyncio.run(_run())
    except asyncio.TimeoutError as exc:
        raise RuntimeError(
            f"语音合成超时（{config.TTS_TIMEOUT:.0f} 秒无响应）"
        ) from exc
