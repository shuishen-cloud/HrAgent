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
    fallback: bool = False  # 回答是否来自题库预设（而非真实的语音转写）


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
    parsed: bool = True    # 本轮 LLM 是否按约定格式返回；False = 走了兜底
    fallback: bool = False  # 回答是否来自题库预设（而非真实的语音转写）


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


_FALLBACK_BANK = {
    "opening": "你好，请先做个自我介绍。",
    "questions": [],
    "closing": "今天先到这里，谢谢。",
}


def load_questions() -> dict:
    """读题库。文件缺失、损坏、或**顶层类型不对**时都走兜底，不让面试开不了场。

    光 catch 异常不够：`json.loads('["a","b"]')` 是合法的，返回 list，
    后面 `bank.get(...)` 就会 AttributeError。所以还得校验顶层是 dict。
    """
    path = config.DATA_DIR / "questions.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return dict(_FALLBACK_BANK)
    return data if isinstance(data, dict) else dict(_FALLBACK_BANK)


def _closing() -> str:
    """结束语。题库里没写 closing 时不能返回空串 ——
    空串会被合成为空音频，候选人在静音中结束面试，且界面上看不出任何异常。"""
    text = load_questions().get("closing")
    return text if isinstance(text, str) and text.strip() else _FALLBACK_BANK["closing"]


def preset_answer(turn_index: int) -> str:
    """取题库里第 turn_index 题（1 起）的预设答案，没有就返回空串。

    用途是降级：候选人没说出有效内容时（长按太短、麦克风没收到声音），
    用预置的参考答案把这一轮补上，演示流程不至于断在这里。
    """
    questions = load_questions().get("questions")
    if not isinstance(questions, list):
        return ""
    if not (1 <= turn_index <= len(questions)):
        return ""
    item = questions[turn_index - 1]
    if not isinstance(item, dict):
        return ""
    answer = item.get("answer")
    return answer.strip() if isinstance(answer, str) else ""


def start(state: InterviewState) -> tuple[str, bytes]:
    """开场：给出第一个问题。返回 (问题文本, TTS 音频)。"""
    question = load_questions().get("opening")
    if not isinstance(question, str) or not question.strip():
        question = _FALLBACK_BANK["opening"]

    # 同 run_turn：先做最容易失败的一步（TTS 要联网），成功了再改 state，
    # 免得合成失败后 state 停在半路、用户反复点开始却只看到 500
    audio = tts.synthesize(question)

    state.current_question = question
    state.history = [{"role": "assistant", "content": question}]
    state.finished = False
    return question, audio


def run_turn(state: InterviewState, audio_bytes: bytes = b"", frames: list[bytes] | None = None) -> TurnResult:
    """收一轮回答：听 → 看 + 想 → 说。"""
    frames = frames or []

    # 1. 听：整段录音转文字
    answer = stt.transcribe(audio_bytes) if audio_bytes else ""

    # 2. 降级：没转出有效内容就用题库预设答案顶上。
    #    「长按太短」和「没说话」的结果都是转出空串，物理上等价，一并处理。
    #    按轮次对应题目：第 N 轮 → questions[N-1].answer
    index = len(state.records) + 1
    fallback = False
    if not answer.strip():
        preset = preset_answer(index)
        if preset:
            answer, fallback = preset, True

    # 3. 看 + 想：文字 + 画面帧一起给多模态 LLM
    reply = llm.chat(state.history, answer, frames)

    # 4. 先算出这一轮的结果，但**先别写进 state**
    finished = bool(reply["should_end"]) or index >= state.max_turns
    next_question = reply["next_question"]
    spoken = next_question if not finished else _closing()

    # 5. 说：合成语音。这一步要联网，是最容易失败的一环，
    #    所以放在提交状态之前 —— 失败时 state 保持原样，用户重试仍是同一轮；
    #    否则这一轮已经记进 records，重试会变成两轮，
    #    而且报告里那轮的「问题」是从没被念出来过的那句，问与答对不上。
    audio = tts.synthesize(spoken)

    # 6. 全部成功，提交状态
    record = TurnRecord(
        index=index,
        question=state.current_question,
        answer=answer,
        score=reply["evaluation"]["score"],
        comment=reply["evaluation"]["comment"],
        focused=reply["attention"]["focused"],
        attention_note=reply["attention"]["note"],
        cheating=reply["cheating_suspected"],
        frames=len(frames),
        next_question=next_question,
        fallback=fallback,
    )
    state.records.append(record)
    # 历史只留文字，不带图片
    state.history.append({"role": "user", "content": answer or "（候选人没有说话）"})
    state.history.append({"role": "assistant", "content": next_question})
    state.finished = finished
    state.current_question = next_question

    return TurnResult(
        answer=answer,
        score=record.score,
        comment=record.comment,
        focused=record.focused,
        attention_note=record.attention_note,
        cheating=record.cheating,
        question=spoken,
        audio=audio,
        finished=finished,
        parsed=reply.get("_parsed", True),
        fallback=fallback,
    )


# ---------------------------------------------------------------- 报告

def build_report(state: InterviewState) -> dict:
    """把整场面试汇总成结构化结果（不调 LLM，纯本地计算）。"""
    records = state.records
    scored = [r.score for r in records if isinstance(r.score, int)]
    avg = round(sum(scored) / len(scored), 1) if scored else None

    distracted = [r.index for r in records if not r.focused]
    cheating = [r.index for r in records if r.cheating]

    verdict, recommendation = _recommend(avg, cheating)

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "turns": len(records),
        "average_score": avg,
        "scores": [r.score for r in records],
        "distracted_turns": distracted,
        "cheating_turns": cheating,
        # verdict 是给前端着色的枚举，避免前端拿中文串做关键词匹配
        "verdict": verdict,
        "recommendation": recommendation,
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
                "fallback": r.fallback,
            }
            for r in records
        ],
    }


def _recommend(avg: float | None, cheating: list[int]) -> tuple[str, str]:
    """给一个粗略的结论，返回 (verdict 枚举, 中文结论)。

    **这是规则算出来的，不是 LLM 的判断**，只作参考。
    verdict 供前端着色：review / fail / hold / pass / unknown
    """
    if cheating:
        turns = "、".join(map(str, cheating))
        return "review", f"建议复核：第 {turns} 轮画面出现疑似作弊，需人工确认"
    if avg is None:
        return "unknown", "无法评估：本轮没有拿到有效评分"
    if avg >= 8:
        return "pass", "建议通过：回答质量整体较好"
    if avg >= 6:
        return "hold", "待定：回答基本合格，建议结合岗位要求人工复核"
    return "fail", "建议不通过：回答质量偏低"


def render_markdown(state: InterviewState, report: dict | None = None, summary: str = "") -> str:
    """把报告渲染成可读的 Markdown。

    **故意保持简短**：逐轮只留一行（得分 + 一句点评 + 异常标记），
    不展开问题与回答全文 —— 那些在 build_report 的 records 里都有，
    要看详情去查结构化数据，不要在报告里堆成一大篇。
    """
    report = report or build_report(state)
    done = report["average_score"]
    lines = [
        "# 面试报告",
        "",
        f"- 平均得分：{done if done is not None else '—'}",
        f"- 走神轮次：{report['distracted_turns'] or '无'}",
        f"- 疑似作弊：{report['cheating_turns'] or '无'}",
        f"- **结论**：{report['recommendation']}",
        "",
    ]

    if summary:
        lines += ["## 总体评价", "", summary, ""]

    lines += ["## 逐轮明细", ""]
    for r in report["records"]:
        score = r["score"] if r["score"] is not None else "—"
        parts = [f"第 {r['index']} 轮", f"{score} 分"]
        if r["comment"]:
            parts.append(r["comment"].strip())
        parts.append("专注" if r["focused"] else "⚠️ 走神")
        if r["cheating"]:
            parts.append("⚠️ 疑似作弊")
        if r.get("fallback"):
            parts.append("预设答案")
        lines.append("- " + " · ".join(parts))

    return "\n".join(lines) + "\n"
