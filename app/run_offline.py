"""离线跑一整场面试，产出报告。用 fixture 样本，不需要麦克风和摄像头。

用法：
    python -m app.run_offline              # 真实跑：真 STT + 真 LLM + 真 TTS
    python -m app.run_offline --mock       # 不联网不花钱，只验证流程串得通
    python -m app.run_offline --turns 2    # 只跑 2 轮
    python -m app.run_offline --no-tts     # 跳过语音合成，快一些

产出：
    reports/interview_<时间戳>.md    面试报告
    reports/audio/<时间戳>/*.mp3     面试官每句话的语音（可实际听）
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime

from . import config, interview, stt, llm, tts

# 默认脚本：(样本语音, [画面帧])。第 3 轮故意配「画面里多个人」来触发作弊判定。
SCRIPT = [
    ("answer_01.mp3", ["focused_01.jpg"]),
    ("answer_02.mp3", ["focused_02.jpg"]),
    ("answer_03.mp3", ["multiple_01.jpg"]),
]
FRAME_CYCLE = ["focused_01.jpg", "focused_02.jpg", "distracted_01.jpg"]


# ---------------------------------------------------------------- mock 模式

class _MockReply:
    """假 LLM：给出结构合理、随轮次变化的回复，只为把流程跑通。"""

    def __init__(self) -> None:
        self.n = 0

    def __call__(self, history, answer, frames, system=None):
        self.n += 1
        has_frames = bool(frames)
        return {
            "evaluation": {
                "score": [8, 7, 6, 7, 8][self.n % 5],
                "comment": f"[mock] 第 {self.n} 轮点评：回答基本切题。",
            },
            "attention": {
                "focused": has_frames,
                "note": "[mock] 收到 %d 帧" % len(frames),
            },
            "cheating_suspected": False,
            "next_question": f"[mock] 第 {self.n} 个追问？",
            "should_end": False,
            "_parsed": True,
        }


def _install_mocks() -> None:
    stt.transcribe = lambda b, language=None: f"[mock] 这是第 {len(b)} 字节录音的转写结果"
    llm.chat = _MockReply()
    tts.synthesize = lambda text, voice=None: b"MOCK-MP3" if text.strip() else b""


# ---------------------------------------------------------------- 主流程

def _load_audio(name: str) -> bytes:
    return (config.FIXTURES_DIR / "audio" / name).read_bytes()


def _load_frames(names: list[str]) -> list[bytes]:
    return [(config.FIXTURES_DIR / "frames" / n).read_bytes() for n in names]


def main() -> int:
    ap = argparse.ArgumentParser(description="离线跑一场面试并出报告")
    ap.add_argument("--mock", action="store_true", help="不联网、不花钱，只验证流程")
    ap.add_argument("--turns", type=int, default=3, help="跑几轮（默认 3）")
    ap.add_argument("--no-tts", action="store_true", help="跳过语音合成")
    args = ap.parse_args()

    if args.mock:
        _install_mocks()
        print("【mock 模式】不调真实 STT / LLM / TTS\n")
    else:
        print(f"LLM：{config.describe_llm()}")
        if not config.LLM_API_KEY:
            print("❌ 未配置 LLM_API_KEY，无法真实运行。用 --mock 可先验证流程。", file=sys.stderr)
            return 2
        print()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_dir = config.REPORTS_DIR / "audio" / stamp
    if args.no_tts:
        # 真正把合成换掉，而不是「合成完再丢掉」。否则 --no-tts 依然会去联网调
        # edge-tts，离线时照样在第一步就中断 —— 而它本意正是「不依赖网络、快一点」。
        tts.synthesize = lambda text, voice=None: b""
    else:
        audio_dir.mkdir(parents=True, exist_ok=True)

    def _speak(text: str) -> bytes:
        """题的语音要**单独**合成 —— 引擎不再内嵌 TTS 了。

        TTS 从主链路拆走的理由见 app/interview.py：它要联网、要 1-2 秒，
        留在里面会让用户白等这段时间才看得到文字。离线跑这里只是把音频存盘，
        所以补上这一步。
        """
        if args.no_tts or not text or not text.strip():
            return b""
        return tts.synthesize(text)

    state = interview.InterviewState()
    question = interview.start(state)
    audio = _speak(question)
    if audio:
        (audio_dir / "00_opening.mp3").write_bytes(audio)

    print(f"面试官：{question}\n")

    for i in range(args.turns):
        src_audio, src_frames = SCRIPT[i % len(SCRIPT)]
        if i >= len(SCRIPT):  # 超出预设脚本就循环取帧
            src_frames = [FRAME_CYCLE[i % len(FRAME_CYCLE)]]

        print(f"--- 第 {i + 1} 轮 ---")
        print(f"候选人：（录音 {src_audio}，画面 {', '.join(src_frames)}）")

        try:
            result = interview.run_turn(
                state, _load_audio(src_audio), _load_frames(src_frames)
            )
        except Exception as exc:
            print(f"❌ 第 {i + 1} 轮失败：{exc.__class__.__name__}: {exc}", file=sys.stderr)
            return 1

        print(f"　　转写：{result.answer}")
        print(f"　　得分：{result.score}　点评：{result.comment}")
        flag = "专注" if result.focused else "⚠️ 走神"
        print(f"　　画面：{flag}　{result.attention_note}")
        if result.cheating:
            print("　　⚠️ 疑似作弊：画面中出现多人或异常")
        print(f"面试官：{result.question}\n")

        audio = _speak(result.question)
        if audio:
            (audio_dir / f"{i + 1:02d}_question.mp3").write_bytes(audio)

        if result.finished:
            print(f"（面试在第 {i + 1} 轮结束）\n")
            break

    # ---- 出报告 ----
    report = interview.build_report(state)
    md = interview.render_markdown(state, report)

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    md_path = config.REPORTS_DIR / f"interview_{stamp}.md"
    md_path.write_text(md, encoding="utf-8")

    print("=" * 62)
    print(f"平均得分：{report['average_score']}")
    print(f"走神轮次：{report['distracted_turns'] or '无'}")
    print(f"疑似作弊：{report['cheating_turns'] or '无'}")
    print(f"结论　　：{report['recommendation']}")
    print("=" * 62)
    print(f"\n报告已写入：{md_path}")
    if not args.no_tts:
        print(f"面试官语音：{audio_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
