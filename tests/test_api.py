"""阶段 C 接口测试：FastAPI 路由层。引擎全部打桩，不联网、不花钱。

验的是「HTTP 这层」：参数怎么收、错误怎么报、状态怎么维护。
真正的面试逻辑在 test_pipeline.py 里验。
"""
from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from app import interview, main


@pytest.fixture
def client(monkeypatch):
    """每个用例都从干净的会话开始。"""
    main.session.reset()
    # reset() 故意不清简历（重开一场面试还是同一位候选人），
    # 所以测试之间必须显式清掉，否则简历会泄漏到后续用例
    main.session.resume = ""
    monkeypatch.setattr(interview.tts, "synthesize", lambda text, voice=None: b"MP3")
    # start() 在「有简历」时会调 LLM 生成开场问题，这里必须打桩 ——
    # 否则相关用例会真的走网络（表现：整个测试文件从 0.3s 变成 5s）
    monkeypatch.setattr(interview.llm, "opening_question", lambda resume: "请自我介绍")
    monkeypatch.setattr(interview.llm, "sample_answer", lambda q: "这是示例回答")

    def fake_start(state, resume=""):
        state.resume = resume or ""
        state.current_question = "请自我介绍"
        state.history = [{"role": "assistant", "content": "请自我介绍"}]
        state.finished = False
        return "请自我介绍"          # 只回文字，语音走独立的 /api/tts

    def fake_run_turn(state, audio_bytes=b"", frames=None):
        frames = frames or []
        state.records.append(
            interview.TurnRecord(
                index=len(state.records) + 1, question="请自我介绍", answer="我是张三",
                score=8, comment="不错", focused=True, attention_note="正对镜头",
                cheating=False, frames=len(frames), next_question="下一题？",
            )
        )
        state.finished = len(state.records) >= state.max_turns
        return interview.TurnResult(
            answer="我是张三", score=8, comment="不错", focused=True,
            attention_note="正对镜头", cheating=False,
            question="下一题？", finished=state.finished,
        )

    monkeypatch.setattr(interview, "start", fake_start)
    monkeypatch.setattr(interview, "run_turn", fake_run_turn)
    # build_report / render_markdown 用真的，它们是纯计算
    return TestClient(main.app)


def _answer(client, frames=()):
    files = [("audio", ("a.mp3", io.BytesIO(b"RIFFfake"), "audio/mpeg"))]
    files += [("frames", (n, io.BytesIO(b"JPEGdata"), "image/jpeg")) for n in frames]
    return client.post("/api/turn", files=files)


# ---------------------------------------------------------------- 静态与素材

def test_首页能打开(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "HrAgent" in res.text


def test_列出模拟素材(client):
    data = client.get("/api/fixtures").json()
    assert "focused_01.jpg" in data["frames"]
    assert "answer_01.mp3" in data["audio"]


def test_模拟素材能直接下载(client):
    res = client.get("/fixtures/frames/focused_01.jpg")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("image/")


def test_素材路径穿越被挡住(client):
    """StaticFiles 不该让 ../ 穿出去读到项目外的文件。"""
    res = client.get("/fixtures/frames/..%2f..%2f..%2fREADME.md")
    assert res.status_code in (404, 400)


# ---------------------------------------------------------------- 状态

def test_初始状态是未开始(client):
    data = client.get("/api/status").json()
    assert data["started"] is False
    assert data["turns"] == 0


def test_没开始就发回答会被拒绝(client):
    res = _answer(client)
    assert res.status_code == 409
    assert "开始" in res.json()["detail"]


def test_开始后返回问题和语音(client):
    res = client.post("/api/start")
    assert res.status_code == 200
    data = res.json()
    assert data["question"] == "请自我介绍"
    # 语音不在这里返回 —— 前端拿到文字后异步请求 /api/tts
    assert "audio" not in data
    assert client.get("/api/status").json()["started"] is True


def test_重复开始会重置会话(client):
    client.post("/api/start")
    _answer(client)
    assert client.get("/api/status").json()["turns"] == 1

    client.post("/api/start")                 # 重开
    assert client.get("/api/status").json()["turns"] == 0


# ---------------------------------------------------------------- 一轮

def test_发送回答返回完整结果(client):
    client.post("/api/start")
    res = _answer(client, frames=["focused_01.jpg"])

    assert res.status_code == 200
    data = res.json()
    assert data["answer"] == "我是张三"
    assert data["score"] == 8
    assert data["focused"] is True
    assert data["cheating"] is False
    assert data["question"] == "下一题？"
    assert data["finished"] is False
    assert "audio" not in data


def test_画面帧会传给引擎(client, monkeypatch):
    client.post("/api/start")
    seen = {}
    original = interview.run_turn

    def spy(state, audio_bytes=b"", frames=None):
        seen["frames"] = frames
        seen["audio"] = audio_bytes
        return original(state, audio_bytes, frames)

    monkeypatch.setattr(interview, "run_turn", spy)
    _answer(client, frames=["focused_01.jpg", "multiple_01.jpg"])

    assert len(seen["frames"]) == 2
    assert seen["frames"][0] == b"JPEGdata"
    assert seen["audio"] == b"RIFFfake"


def test_没有画面帧也能发(client):
    client.post("/api/start")
    assert _answer(client).status_code == 200


def test_引擎报错时返回可读信息而不是500堆栈(client, monkeypatch):
    client.post("/api/start")

    def boom(state, audio_bytes=b"", frames=None):
        raise RuntimeError("模型超时；内网地址 10.0.0.7，Authorization: Bearer sk-xxx")

    monkeypatch.setattr(interview, "run_turn", boom)
    res = _answer(client)

    assert res.status_code == 502
    detail = res.json()["detail"]
    assert "RuntimeError" in detail          # 回异常类型，方便定位
    # 但原始消息不能回给浏览器：上游报错体常带请求头、内网地址等
    assert "10.0.0.7" not in detail
    assert "sk-xxx" not in detail
    assert "Authorization" not in detail


def test_开始接口出错也返回可读信息(client, monkeypatch):
    """开场语音合成失败时不能是裸 500 —— 前端只能显示 "500 Internal Server Error"，
    用户根本看不出是 TTS 的问题。"""
    def boom(state, resume=""):
        raise RuntimeError("edge-tts 连接被重置")

    monkeypatch.setattr(interview, "start", boom)
    res = client.post("/api/start")

    assert res.status_code == 502
    assert "RuntimeError" in res.json()["detail"]
    assert client.get("/api/status").json()["started"] is False


def test_跑满轮数后自动结束(client, monkeypatch):
    monkeypatch.setattr(interview.config, "MAX_TURNS", 1)
    client.post("/api/start")
    data = _answer(client).json()

    assert data["finished"] is True
    assert client.get("/api/status").json()["finished"] is True


def test_结束后不能再提交(client, monkeypatch):
    """前端会禁用按钮，但不能只靠前端 —— 直接调 API 也得挡住，
    否则会在已结束的面试上继续记分，报告轮数超出上限。"""
    monkeypatch.setattr(interview.config, "MAX_TURNS", 1)
    client.post("/api/start")
    assert _answer(client).json()["finished"] is True

    res = _answer(client)
    assert res.status_code == 409
    assert "结束" in res.json()["detail"]
    # 记录没有被继续追加
    assert client.get("/api/status").json()["turns"] == 1


# ---------------------------------------------------------------- 报告

def test_没有记录时不给报告(client):
    client.post("/api/start")
    res = client.get("/api/report")
    assert res.status_code == 409


# ---------------------------------------------------------------- 简历

# 必须够长：短于 config.RESUME_MIN_CHARS 会被解析器判为「没解析出有效文字」
# （那个阈值是用来识别扫描件的，不是用来卡正常简历的）
RESUME_BODY = """张伟
求职意向：后端开发工程师    电话：138-0000-0000
教育经历：太原工业学院 软件工程 本科 2019-2023
项目经历：西瓜甜度机器视觉检测 2022.03-2022.10
  负责图像预处理与模型推理，用 OpenCV 去噪与轮廓提取，轻量 CNN 预测甜度
技能：Python / OpenCV / MySQL / Git
"""


def test_没上传简历时状态显示未加载(client):
    data = client.get("/api/status").json()
    assert data["resume_loaded"] is False
    assert data["resume_chars"] == 0


def test_上传简历后返回预览(client):
    res = client.post(
        "/api/resume",
        files={"file": ("简历.txt", io.BytesIO(RESUME_BODY.encode("utf-8")), "text/plain")},
    )

    assert res.status_code == 200
    data = res.json()
    assert data["filename"] == "简历.txt"
    assert data["chars"] > 20
    assert "张伟" in data["preview"]

    assert client.get("/api/status").json()["resume_loaded"] is True


def test_简历解析失败返回可读的_400(client):
    res = client.post(
        "/api/resume",
        files={"file": ("照片.png", io.BytesIO(b"\x89PNG\r\n\x1a\n" + b"x" * 50), "image/png")},
    )

    assert res.status_code == 400
    assert "不支持" in res.json()["detail"]


def test_可以清掉简历回到题库模式(client):
    client.post(
        "/api/resume",
        files={"file": ("a.txt", io.BytesIO(RESUME_BODY.encode("utf-8")), "text/plain")},
    )
    assert client.get("/api/status").json()["resume_loaded"] is True

    assert client.delete("/api/resume").json()["loaded"] is False
    assert client.get("/api/status").json()["resume_loaded"] is False


def test_开始面试会把简历传给引擎(client, monkeypatch):
    """验证简历确实流到了 interview.start，而不是只存着没用。"""
    seen = {}
    original = interview.start

    def spy(state, resume=""):
        seen["resume"] = resume
        return original(state, resume)

    monkeypatch.setattr(interview, "start", spy)
    client.post("/api/resume",
                files={"file": ("a.txt", io.BytesIO(RESUME_BODY.encode("utf-8")), "text/plain")})
    client.post("/api/start")

    assert "张伟" in seen["resume"]


def test_重开面试不会把简历清掉(client):
    """简历属于候选人，不属于某一场面试，重开不该让用户重传。"""
    client.post("/api/resume",
                files={"file": ("a.txt", io.BytesIO(RESUME_BODY.encode("utf-8")), "text/plain")})
    client.post("/api/start")
    client.post("/api/start")          # 重开

    assert client.get("/api/status").json()["resume_loaded"] is True


def test_报告包含汇总和_markdown(client):
    client.post("/api/start")
    _answer(client, frames=["focused_01.jpg"])

    data = client.get("/api/report").json()
    assert data["turns"] == 1
    assert data["average_score"] == 8.0
    assert data["records"][0]["answer"] == "我是张三"
    assert data["markdown"].startswith("# 面试报告")


# ---------------------------------------------------------------- 同步/异步回归

@pytest.fixture
def real_engine_client(monkeypatch):
    """只打桩最底层的叶子依赖，run_turn 与 tts.synthesize 都走真实代码路径。

    为什么需要这个夹具：上面那个 client 夹具把 run_turn 整个打桩了，于是
    **端点自身的同步/异步性质从没被覆盖到**。结果漏掉一个真实 bug：
    /api/turn 写成 async def 时，tts.synthesize 里的 asyncio.run() 会因为
    「已有运行中的事件循环」而抛异常，接口直接 502。

    这里故意不打桩 tts.synthesize —— 让它真的执行 asyncio.run()。
    """
    main.session.reset()
    main.session.resume = ""     # 同上：简历不在 reset 范围内，得显式清

    async def _fake_stream(text, voice):
        return b"MP3"          # 内部桩掉，避免真的联网调 edge-tts

    monkeypatch.setattr(interview.tts, "_stream_audio", _fake_stream)
    monkeypatch.setattr(interview.stt, "transcribe", lambda b, language=None: "转写结果")
    monkeypatch.setattr(
        interview.llm, "chat",
        lambda h, a, f, system=None: {
            "evaluation": {"score": 8, "comment": "点评"},
            "attention": {"focused": True, "note": ""},
            "cheating_suspected": False,
            "next_question": "下一题？",
            "should_end": False,
            "_parsed": True,
        },
    )
    return TestClient(main.app)


def test_一轮接口走真实引擎不报错(real_engine_client):
    """run_turn 走真实代码路径（只桩最底层叶子依赖）。"""
    client = real_engine_client
    client.post("/api/start")
    res = _answer(client, frames=["focused_01.jpg"])

    assert res.status_code == 200, res.json()
    assert res.json()["answer"] == "转写结果"


def test_tts_接口能真的执行_asyncio_run(real_engine_client):
    """回归：TTS 内部的 asyncio.run 不能在事件循环线程里跑。

    端点必须是**同步 def**（FastAPI 丢线程池，那里没有运行中的循环）。
    写成 async def 就会抛
    'asyncio.run() cannot be called from a running event loop'。
    """
    res = real_engine_client.post("/api/tts", json={"text": "你好"})

    assert res.status_code == 200, res.json()
    assert res.json()["audio"]            # base64 非空，说明真的合成了


def test_tts_接口拒绝空文本(real_engine_client):
    res = real_engine_client.post("/api/tts", json={"text": "   "})
    assert res.status_code == 400


def test_tts_接口拒绝超长文本(real_engine_client, monkeypatch):
    monkeypatch.setattr(interview.config, "TTS_MAX_CHARS", 10)
    res = real_engine_client.post("/api/tts", json={"text": "很长的文字" * 20})
    assert res.status_code == 400
    assert "过长" in res.json()["detail"]
