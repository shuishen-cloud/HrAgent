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
    monkeypatch.setattr(interview.tts, "synthesize", lambda text, voice=None: b"MP3")

    def fake_start(state):
        state.current_question = "请自我介绍"
        state.history = [{"role": "assistant", "content": "请自我介绍"}]
        state.finished = False
        return "请自我介绍", b"MP3"

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
            question="下一题？", audio=b"MP3", finished=state.finished,
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
    assert data["audio"]                      # base64 非空
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
    assert data["audio"]
    assert data["finished"] is False


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
        raise RuntimeError("模型超时")

    monkeypatch.setattr(interview, "run_turn", boom)
    res = _answer(client)

    assert res.status_code == 502
    assert "模型超时" in res.json()["detail"]


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


def test_接口里能真的执行_asyncio_run(real_engine_client):
    """回归：async 端点会让 asyncio.run 抛
    'cannot be called from a running event loop'，导致整轮 502。"""
    client = real_engine_client
    client.post("/api/start")
    res = _answer(client, frames=["focused_01.jpg"])

    assert res.status_code == 200, res.json()
    assert res.json()["answer"] == "转写结果"
    assert res.json()["audio"]            # TTS 真的产出了音频


def test_开场接口也走真实_tts(real_engine_client):
    res = real_engine_client.post("/api/start")
    assert res.status_code == 200
    assert res.json()["audio"]
