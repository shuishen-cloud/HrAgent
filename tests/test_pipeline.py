"""阶段 B 集成测试：用 fixture 跑完整场面试，全程不碰硬件、不联网、不花钱。

三个模块都在这里打桩，但**桩是有意义的**：
- 假 LLM 会检查收到的画面帧字节，只有真的传进来才判作弊
  → 这样能验证「帧确实从 run_turn 流到了 LLM」，而不只是「流程没报错」
"""
from __future__ import annotations

import pytest

from app import config, interview, llm, stt, tts

FRAMES_DIR = config.FIXTURES_DIR / "frames"
AUDIO_DIR = config.FIXTURES_DIR / "audio"


def _frame(name: str) -> bytes:
    return (FRAMES_DIR / name).read_bytes()


# ---------------------------------------------------------------- 打桩

@pytest.fixture(autouse=True)
def pipeline(monkeypatch):
    """装好三个桩，并记录 LLM 实际收到了什么。

    autouse：本文件所有用例都不该碰网络。漏装桩会让 edge-tts / whisper 真的被调用，
    测试会变慢且依赖联网 —— 靠耗时能看出来（正常应 < 1s）。
    """
    seen: dict = {"frames": [], "answers": [], "questions": []}

    def fake_transcribe(audio_bytes, language=None):
        return f"这是第 {len(audio_bytes)} 字节的转写"

    cheating_frame = _frame("multiple_01.jpg")

    def fake_chat(history, answer, frames, system=None):
        seen["frames"].append(list(frames))
        seen["answers"].append(answer)
        # 只有真的收到了那张「多人」画面才判作弊
        cheating = any(f == cheating_frame for f in frames)
        seen["questions"].append(history[-1]["content"] if history else "")
        return {
            "evaluation": {"score": 7, "comment": "点评"},
            "attention": {"focused": not cheating, "note": "画面说明"},
            "cheating_suspected": cheating,
            "next_question": f"追问 {len(seen['answers'])}？",
            "should_end": False,
            "_parsed": True,
        }

    monkeypatch.setattr(stt, "transcribe", fake_transcribe)
    monkeypatch.setattr(llm, "chat", fake_chat)
    monkeypatch.setattr(tts, "synthesize", lambda text, voice=None: b"AUDIO")

    return seen


def _run(state, monkeypatch, frames=("focused_01.jpg",), audio=b"x" * 100):
    return interview.run_turn(state, audio, [_frame(f) for f in frames])


# ---------------------------------------------------------------- 开场

def test_start_返回开场问题并合成语音(pipeline):
    state = interview.InterviewState()
    question, audio = interview.start(state)

    assert "自我介绍" in question
    assert audio == b"AUDIO"
    # 开场问题要进历史，否则 LLM 不知道面试官问过什么
    assert state.history == [{"role": "assistant", "content": question}]
    assert state.current_question == question


# ---------------------------------------------------------------- 一轮

def test_run_turn_完整走一遍(pipeline):
    state = interview.InterviewState(max_turns=5)
    interview.start(state)

    result = interview.run_turn(state, b"x" * 100, [_frame("focused_01.jpg")])

    assert result.answer == "这是第 100 字节的转写"   # 听
    assert result.score == 7                          # 想
    assert result.question == "追问 1？"               # 说
    assert result.audio == b"AUDIO"
    assert result.finished is False
    assert len(state.records) == 1
    assert state.records[0].question == state.history[0]["content"]


def test_画面帧真的传到了_llm(pipeline):
    state = interview.InterviewState()
    interview.start(state)
    frame = _frame("focused_01.jpg")

    interview.run_turn(state, b"audio", [frame])

    assert pipeline["frames"][0] == [frame]   # 字节原样送达


def test_历史只带文字不带图片(pipeline):
    """历史里塞图片会让 token 随轮数膨胀，所以只保留文字。"""
    state = interview.InterviewState(max_turns=3)
    interview.start(state)
    for i in range(3):
        interview.run_turn(state, b"a" * (i + 1), [_frame("focused_01.jpg")])

    for msg in state.history:
        assert isinstance(msg["content"], str)


def test_多轮之后历史累积正确(pipeline):
    state = interview.InterviewState(max_turns=4)
    interview.start(state)
    for _ in range(3):
        interview.run_turn(state, b"audio", [_frame("focused_01.jpg")])

    # 开场 1 条 + 每轮 2 条（用户回答 + 面试官追问）
    assert len(state.history) == 1 + 3 * 2
    assert len(state.records) == 3


def test_没有语音时标记为没有说话(pipeline):
    state = interview.InterviewState()
    interview.start(state)

    result = interview.run_turn(state, b"", [])   # 空录音

    assert result.answer == ""
    assert pipeline["answers"][0] == ""            # 空串照样送进 LLM
    assert state.history[1]["content"] == "（候选人没有说话）"


# ---------------------------------------------------------------- 判定与结束

def test_多人画面触发疑似作弊(pipeline):
    state = interview.InterviewState()
    interview.start(state)

    result = interview.run_turn(state, b"audio", [_frame("multiple_01.jpg")])

    assert result.cheating is True
    assert result.focused is False


def test_达到最大轮数自动结束(pipeline):
    state = interview.InterviewState(max_turns=2)
    interview.start(state)

    assert interview.run_turn(state, b"a", [_frame("focused_01.jpg")]).finished is False
    assert interview.run_turn(state, b"a", [_frame("focused_01.jpg")]).finished is True
    assert state.finished is True


def test_llm_说要结束就结束(pipeline, monkeypatch):
    state = interview.InterviewState(max_turns=99)
    interview.start(state)

    def ending_chat(history, answer, frames, system=None):
        return {
            "evaluation": {"score": 5, "comment": ""},
            "attention": {"focused": True, "note": ""},
            "cheating_suspected": False,
            "next_question": "下一个问题",
            "should_end": True,
            "_parsed": True,
        }

    monkeypatch.setattr(llm, "chat", ending_chat)
    assert interview.run_turn(state, b"a", []).finished is True


def test_结束时播报的是结束语不是问题(pipeline):
    state = interview.InterviewState(max_turns=1)
    interview.start(state)

    result = interview.run_turn(state, b"a", [_frame("focused_01.jpg")])

    assert result.finished is True
    assert "自我介绍" not in result.question   # 不该再问下一个问题
    assert result.question                        # 有结束语


# ---------------------------------------------------------------- 报告

def _full_interview(monkeypatch) -> interview.InterviewState:
    """跑一场三轮面试，其中第三轮带「多人」画面。"""
    state = interview.InterviewState(max_turns=3)
    interview.start(state)
    interview.run_turn(state, b"a" * 10, [_frame("focused_01.jpg")])
    interview.run_turn(state, b"a" * 20, [_frame("focused_02.jpg")])
    interview.run_turn(state, b"a" * 30, [_frame("multiple_01.jpg")])
    return state


def test_报告结构完整(pipeline):
    state = _full_interview(None)
    report = interview.build_report(state)

    assert report["turns"] == 3
    assert report["average_score"] == 7.0
    assert report["cheating_turns"] == [3]        # 第三轮
    assert report["distracted_turns"] == [3]      # 多人轮 focused=False
    assert len(report["records"]) == 3
    assert report["recommendation"]


def test_报告_markdown_可渲染(pipeline):
    state = _full_interview(None)
    md = interview.render_markdown(state)

    assert md.startswith("# 面试报告")
    assert "第 3 轮" in md
    assert "疑似作弊" in md
    assert "逐轮明细" in md


def test_报告在无有效评分时不崩():
    state = interview.InterviewState(max_turns=1)
    interview.start(state)
    # 绕过 LLM，直接构造一条无分数的记录
    state.records.append(
        interview.TurnRecord(
            index=1, question="q", answer="a", score=None, comment="",
            focused=True, attention_note="", cheating=False, frames=0, next_question="n",
        )
    )
    report = interview.build_report(state)

    assert report["average_score"] is None
    assert "无法评估" in report["recommendation"]


def test_作弊时结论要求人工复核():
    state = interview.InterviewState(max_turns=1)
    interview.start(state)
    state.records.append(
        interview.TurnRecord(
            index=7, question="q", answer="a", score=9, comment="",
            focused=False, attention_note="", cheating=True, frames=1, next_question="n",
        )
    )
    assert "复核" in interview.build_report(state)["recommendation"]


@pytest.mark.parametrize(
    "avg,expected",
    [(9.0, "建议通过"), (7.0, "待定"), (4.0, "建议不通过")],
)
def test_结论分档(avg, expected):
    assert expected in interview._recommend(avg, [], [])
