"""面试引擎：一场面试的状态机 + 报告生成。

对外只有三个动作，都是同步的：
    start(state)                     —— 开场，返回第一个问题
    run_turn(state, audio, frames)   —— 收一轮回答，返回点评 + 下一个问题
    build_report(state)              —— 面试结束，产出报告

设计要点：
- 引擎只接收字节，不关心数据来自文件还是真实麦克风（见 README「自动测试优先」）
- 送给 LLM 的历史**只保留文字**，不带历史图片 —— 否则 token 会随轮数线性膨胀。
  画面只对当前轮生效，这对「候选人此刻是否专注」的判断已经够用。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime

from . import config, llm, stt, tts


@dataclass
class TurnRecord:
    """一轮问答的完整记录，报告就是把它渲染出来。"""

    index: int
    question: str          # 本轮开始时面试官问的问题
    answer: str            # 候选人回答（STT 转写）
    score: int | None
    comment: str
    focused: bool
    attention_note: str
    cheating: bool
    frames: int            # 本轮用了多少张画面帧
    next_question: str     # 本轮结束时面试官问的下一题


@dataclass
class TurnResult:
    """一轮的返回值，前端拿它来播报和展示。"""

    answer: str
    score: int | None
    comment: str
    focused: bool
    attention_note: str
    cheating: bool
    question: str          # 下一个问题（要播报的内容）
    audio: bytes           # 上面那句话的 TTS 音频
    finished: bool


@dataclass
class InterviewState:
    history: list[dict] = field(default_factory=list)   # 给 LLM 的对话历史（纯文字）
    records: list[TurnRecord] = field(default_factory=list)
    max_turns: int = 0
    current_question: str = ""
    finished: bool = False

    def __post_init__(self) -> None:
        if not self.max_turns:
            self.max_turns = config.MAX_TURNS


def load_questions() -> dict:
    """读题库。文件缺失或损坏时给一份最小兜底，不让面试开不了场。"""
    path = config.DATA_DIR / "questions.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "opening": "你好，请先做个自我介绍。",
            "questions": [],
            "closing": "今天先到这里，谢谢。",
        }


def start(state: InterviewState) -> tuple[str, bytes]:
    """开场：给出第一个问题。返回 (问题文本, TTS 音频)。"""
    bank = load_questions()
    state.current_question = bank.get("opening") or "你好，请先做个自我介绍。"
    state.history = [{"role": "assistant", "content": state.current_question}]
    state.finished = False
    return state.current_question, tts.synthesize(state.current_question)


def run_turn(state: InterviewState, audio_bytes: bytes = b"", frames: list[bytes] | None = None) -> TurnResult:
    """收一轮回答：听 → 看 + 想 → 说。"""
    frames = frames or []

    # 1. 听：整段录音转文字
    answer = stt.transcribe(audio_bytes) if audio_bytes else ""

    # 2. 看 + 想：文字 + 画面帧一起给多模态 LLM
    reply = llm.chat(state.history, answer, frames)

    # 3. 记录这一轮
    record = TurnRecord(
        index=len(state.records) + 1,
        question=state.current_question,
        answer=answer,
        score=reply["evaluation"]["score"],
        comment=reply["evaluation"]["comment"],
        focused=reply["attention"]["focused"],
        attention_note=reply["attention"]["note"],
        cheating=reply["cheating_suspected"],
        frames=len(frames),
        next_question=reply["next_question"],
    )
    state.records.append(record)

    # 4. 更新历史（只留文字，不带图片）
    state.history.append({"role": "user", "content": answer or "（候选人没有说话）"})
    state.history.append({"role": "assistant", "content": reply["next_question"]})

    state.finished = bool(reply["should_end"]) or len(state.records) >= state.max_turns
    state.current_question = reply["next_question"]

    # 5. 说：把下一个问题合成语音
    spoken = reply["next_question"] if not state.finished else load_questions().get("closing", "")
    return TurnResult(
        answer=answer,
        score=record.score,
        comment=record.comment,
        focused=record.focused,
        attention_note=record.attention_note,
        cheating=record.cheating,
        question=spoken,
        audio=tts.synthesize(spoken),
        finished=state.finished,
    )


# ---------------------------------------------------------------- 报告

def build_report(state: InterviewState) -> dict:
    """把整场面试汇总成结构化结果（不调 LLM，纯本地计算）。"""
    records = state.records
    scored = [r.score for r in records if isinstance(r.score, int)]
    avg = round(sum(scored) / len(scored), 1) if scored else None

    distracted = [r.index for r in records if not r.focused]
    cheating = [r.index for r in records if r.cheating]

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "turns": len(records),
        "average_score": avg,
        "scores": [r.score for r in records],
        "distracted_turns": distracted,
        "cheating_turns": cheating,
        "recommendation": _recommend(avg, cheating, records),
        "records": [
            {
                "index": r.index,
                "question": r.question,
                "answer": r.answer,
                "score": r.score,
                "comment": r.comment,
                "focused": r.focused,
                "attention_note": r.attention_note,
                "cheating": r.cheating,
                "frames": r.frames,
            }
            for r in records
        ],
    }


def _recommend(avg: float | None, cheating: list[int], records: list[TurnRecord]) -> str:
    """给一个粗略的结论。**这是规则算出来的，不是 LLM 的判断**，只作参考。"""
    if cheating:
        return f"建议复核：第 {'、'.join(map(str, cheating))} 轮画面出现疑似作弊，需人工确认"
    if avg is None:
        return "无法评估：本轮没有拿到有效评分"
    if avg >= 8:
        return "建议通过：回答质量整体较好"
    if avg >= 6:
        return "待定：回答基本合格，建议结合岗位要求人工复核"
    return "建议不通过：回答质量偏低"


def render_markdown(state: InterviewState, report: dict | None = None, summary: str = "") -> str:
    """把报告渲染成可读的 Markdown。"""
    report = report or build_report(state)
    lines = [
        "# 面试报告",
        "",
        f"- 生成时间：{report['generated_at']}",
        f"- 问答轮数：{report['turns']}",
        f"- 平均得分：{report['average_score'] if report['average_score'] is not None else '—'}",
        f"- 走神轮次：{report['distracted_turns'] or '无'}",
        f"- 疑似作弊：{report['cheating_turns'] or '无'}",
        f"- **结论**：{report['recommendation']}",
        "",
    ]

    if summary:
        lines += ["## 总体评价", "", summary, ""]

    lines += ["## 逐轮明细", ""]
    for r in report["records"]:
        lines += [
            f"### 第 {r['index']} 轮　得分 {r['score'] if r['score'] is not None else '—'}",
            "",
            f"- **问题**：{r['question']}",
            f"- **回答**：{r['answer'] or '（未识别到语音）'}",
            f"- **点评**：{r['comment']}",
            f"- **画面**：{'专注' if r['focused'] else '⚠️ 走神'}　{r['attention_note']}",
        ]
        if r["cheating"]:
            lines.append("- **⚠️ 疑似作弊**：画面中出现多人或异常情况")
        lines += [f"- 本轮采集帧数：{r['frames']}", ""]

    return "\n".join(lines)
