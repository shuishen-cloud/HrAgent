"""FastAPI 入口：把面试引擎包成 HTTP 接口，给浏览器前端调用。

数据来源与引擎解耦：前端把「录音文件 + 画面帧」POST 上来，
后端只管调 interview.run_turn。所以前端无论是用真实摄像头采集，
还是像现在这样发模拟图 + 模拟 mp3，后端一行都不用改。

启动：
    .venv/bin/uvicorn app.main:app --reload
然后浏览器打开 http://localhost:8000
"""
from __future__ import annotations

import base64
import logging
import threading
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import config, interview, resume as resume_parser

logger = logging.getLogger("hragent")

app = FastAPI(title="HrAgent 多模态面试官")


class Session:
    """单场面试的会话状态。

    这是个单人 demo，全局只维护一场面试。真要多用户并发，
    把这里换成按 session_id 存 dict 即可，引擎本身是无状态的入参/出参。
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()   # 引擎跑得慢，防止两次请求交叉改同一份 state
        self.state = interview.InterviewState()
        self.started = False
        # 候选人简历。**故意不随 reset 清空** —— 重开一场面试
        # 还是同一位候选人，简历不该让用户重传一遍
        self.resume = ""
        self.resume_name = ""

    def reset(self) -> None:
        with self.lock:
            self.state = interview.InterviewState()
            self.started = False


session = Session()


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _fail(exc: Exception, what: str) -> HTTPException:
    """把内部异常转成可读的 HTTP 错误。

    只回异常**类型**，不回原始消息 —— 上游报错体里可能带请求头、
    内网地址等内容（自建 base_url 的网关尤其常见），不该原样送到浏览器。
    完整信息写服务端日志，排查照样有据可查。
    """
    logger.exception("%s失败", what)
    return HTTPException(status_code=502, detail=f"{what}失败：{exc.__class__.__name__}")


# ---------------------------------------------------------------- 页面

# 样本资源直接静态挂载，前端在模拟模式下从这里取图和音频
app.mount("/fixtures", StaticFiles(directory=str(config.FIXTURES_DIR)), name="fixtures")
app.mount("/static", StaticFiles(directory=str(config.STATIC_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(config.STATIC_DIR / "index.html")


# ---------------------------------------------------------------- 接口

@app.get("/api/fixtures")
def list_fixtures() -> dict:
    """列出可用的模拟素材，给前端下拉框用。"""
    def names(sub: str, exts: tuple[str, ...]) -> list[str]:
        d = config.FIXTURES_DIR / sub
        if not d.exists():
            return []
        return sorted(p.name for p in d.iterdir() if p.suffix.lower() in exts)

    return {
        "frames": names("frames", (".jpg", ".jpeg", ".png")),
        "audio": names("audio", (".mp3", ".wav", ".m4a", ".webm")),
    }


@app.get("/api/status")
def status() -> dict:
    with session.lock:
        return {
            "started": session.started,
            "turns": len(session.state.records),
            "finished": session.state.finished,
            "max_turns": session.state.max_turns,
            "resume_loaded": bool(session.resume),
            "resume_chars": len(session.resume),
        }


# ---------------------------------------------------------------- 简历

@app.post("/api/resume")
def upload_resume(
    file: Annotated[UploadFile, File(description="简历文件（pdf/docx/txt/md）")],
) -> dict:
    """上传并解析简历。解析失败返回 400 + 可直接展示给用户的原因。"""
    file.file.seek(0)
    data = file.file.read()
    try:
        text = resume_parser.extract_text(file.filename or "", data)
    except resume_parser.ResumeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    with session.lock:
        session.resume = text
        session.resume_name = file.filename or "简历"

    return {
        "loaded": True,
        "filename": session.resume_name,
        "chars": len(text),
        "preview": resume_parser.summarize(text, 160),
    }


@app.get("/api/resume")
def get_resume() -> dict:
    with session.lock:
        text = session.resume
    return {
        "loaded": bool(text),
        "filename": session.resume_name if text else "",
        "chars": len(text),
        "preview": resume_parser.summarize(text, 160) if text else "",
    }


@app.delete("/api/resume")
def clear_resume() -> dict:
    """清掉简历，回到固定题库模式。"""
    with session.lock:
        session.resume = ""
        session.resume_name = ""
    return {"loaded": False, "filename": "", "chars": 0, "preview": ""}


@app.post("/api/start")
def start() -> dict:
    """开始面试：重置会话，返回开场问题及其语音。

    这里也要兜异常：开场语音合成失败（首次跑、被墙、edge-tts 限流）时，
    不兜就是裸 500 + 非 JSON 响应，前端只能显示 "500 Internal Server Error"，
    用户完全看不出是 TTS 的问题。
    """
    session.reset()
    try:
        with session.lock:
            # 有简历就在 start 内部由 LLM 按简历生成开场问题
            question, audio = interview.start(session.state, session.resume)
            session.started = True
    except Exception as exc:
        raise _fail(exc, "开始面试") from exc
    return {
        "question": question,
        "audio": _b64(audio),
        "resume_used": bool(session.resume),
    }


@app.post("/api/turn")
def turn(
    audio: Annotated[UploadFile, File(description="候选人的回答录音")],
    frames: Annotated[list[UploadFile], File(description="回答期间的画面帧")] = [],
) -> dict:
    """收一轮回答：录音 + 画面帧 → 点评 + 下一个问题。

    **必须是同步 def**：FastAPI 会把同步端点丢进线程池执行，那里没有运行中的事件循环。
    若写成 async def，run_turn 会跑在事件循环上，而 tts.synthesize 内部用的
    asyncio.run() 在有循环的线程里会直接抛
    "asyncio.run() cannot be called from a running event loop"。
    所以这里用 audio.file.read() 同步读，而不是 await audio.read()。
    """
    if not session.started:
        raise HTTPException(status_code=409, detail="面试还没开始，先调用 /api/start")
    if session.state.finished:
        # 前端也会禁用按钮，但不能只靠前端 —— 直接调 API 同样得挡住，
        # 否则会在已结束的面试上继续记分，报告轮数超出上限
        raise HTTPException(status_code=409, detail="面试已经结束，需要重新调用 /api/start")

    # 显式回到开头再读：不依赖 UploadFile 内部文件指针的位置
    audio.file.seek(0)
    audio_bytes = audio.file.read()
    frame_bytes = []
    for f in frames:
        f.file.seek(0)
        frame_bytes.append(f.file.read())

    # run_turn 是同步且耗时的（STT + LLM + TTS），放到线程里跑，别卡住事件循环
    with session.lock:
        try:
            result = interview.run_turn(session.state, audio_bytes, frame_bytes)
        except Exception as exc:
            raise _fail(exc, "处理") from exc

    return {
        "answer": result.answer,
        "score": result.score,
        "comment": result.comment,
        "focused": result.focused,
        "attention_note": result.attention_note,
        "cheating": result.cheating,
        "question": result.question,
        "audio": _b64(result.audio),
        "finished": result.finished,
        # 本轮 LLM 是否按格式返回。False = 走了兜底（问题可能不是真问题），
        # 前端据此给个提示，别让人以为模型正常答了
        "parsed": result.parsed,
        # 本轮回答来自题库预设而非真实语音转写（长按太短 / 没收到声音时）
        "fallback": result.fallback,
    }


@app.get("/api/report")
def report() -> dict:
    """当前面试的汇总结果。"""
    with session.lock:
        if not session.state.records:
            raise HTTPException(status_code=409, detail="还没有任何问答记录")
        data = interview.build_report(session.state)
        data["markdown"] = interview.render_markdown(session.state, data)
    return data
