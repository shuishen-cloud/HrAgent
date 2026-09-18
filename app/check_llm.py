"""多模态 LLM 冒烟测试：真实调用，验证「看画面」的能力（会消耗少量额度）。

用法：
    python -m app.check_llm

做法：拿**同一段回答 + 同一个问题**，分别配不同的画面帧，
对比模型给出的 attention / cheating_suspected 是否随画面变化，并自动判定通过与否。
如果判定不随画面变化，说明它没在「看」——prompt 或模型选型要调整。

注意：fixtures 里的画面帧是程序画的占位图。**占位图判不准不代表真人也判不准**，
反过来也一样。要得到可信结论，请把 tests/fixtures/frames/ 换成真实照片再跑。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from . import config, llm

ANSWER = "我之前做过后端开发，主要负责订单系统的接口设计，用 Python 和 FastAPI 写过不少服务。"


@dataclass
class Case:
    frame: str
    desc: str
    # 期望判定；None = 该项不检查（属于主观判断，只展示不判分）
    focused: bool | None
    cheating: bool | None


CASES = [
    Case("focused_01.jpg", "正对镜头", True, False),
    Case("focused_02.jpg", "正对镜头", True, False),
    # 实测结论：占位图验不了「姿态类」判断。程序画的圆头+肩膀，无论怎么改五官，
    # 模型都读作「一个人坐在镜头前」。结构性线索（几个人、有没有人）能验，
    # 细粒度姿态（头转没转、低没低头）需要真实照片。
    # → 换成真实照片后，把下面这行的 focused 改回 False 即可生效。
    Case("distracted_01.jpg", "转头走神（占位图无法验证）", None, None),
    Case("empty_01.jpg", "空座位（人不在）", False, None),
    Case("multiple_01.jpg", "画面里多个人", False, True),
]


def _verdict(ok: bool | None) -> str:
    return "—" if ok is None else ("✅" if ok else "❌")


def main() -> int:
    print(f"LLM 配置：{config.describe_llm()}")

    if not config.LLM_API_KEY:
        print(
            "\n❌ 没有配置 LLM_API_KEY。\n"
            "   复制 .env.example 为 .env 并填入密钥：cp .env.example .env\n",
            file=sys.stderr,
        )
        return 2

    frames_dir = config.FIXTURES_DIR / "frames"
    if not frames_dir.exists() or not any(frames_dir.iterdir()):
        print("❌ 没有画面帧样本，先跑：python tests/fixtures/make_fixtures.py", file=sys.stderr)
        return 2

    rows: list[tuple[Case, dict]] = []
    failures = 0

    for case in CASES:
        path: Path = frames_dir / case.frame
        if not path.exists():
            print(f"跳过 {case.frame}（文件不存在）")
            continue

        print(f"\n{'=' * 62}\n画面：{case.frame}    期望：{case.desc}\n{'=' * 62}")
        try:
            result = llm.chat([], ANSWER, [path.read_bytes()])
        except Exception as exc:
            failures += 1
            print(f"❌ 调用失败：{exc.__class__.__name__}: {exc}", file=sys.stderr)
            if "response_format" in str(exc):
                print("   提示：该端点不支持 JSON 模式，在 .env 里加 LLM_JSON_MODE=0", file=sys.stderr)
            continue

        if not result["_parsed"]:
            print("⚠️  模型没按 JSON 格式返回（已走兜底）")

        ev, at = result["evaluation"], result["attention"]
        print(f"  评分    ：{ev['score']}")
        print(f"  点评    ：{ev['comment']}")
        print(f"  专注    ：{at['focused']}   （{at['note']}）")
        print(f"  疑似作弊：{result['cheating_suspected']}")
        print(f"  下一问  ：{result['next_question']}")

        rows.append((case, result))

    if failures:
        print(f"\n有 {failures} 个用例调用失败。")
        return 1

    # ---- 汇总 ----
    print(f"\n{'=' * 62}\n汇总\n{'=' * 62}")
    print(f"{'画面':<18}{'期望':<16}{'专注':<8}{'疑似作弊':<10}")
    misses = 0
    for case, res in rows:
        at, ch = res["attention"], res["cheating_suspected"]
        ok_f = None if case.focused is None else (at["focused"] == case.focused)
        ok_c = None if case.cheating is None else (ch == case.cheating)
        misses += sum(1 for o in (ok_f, ok_c) if o is False)
        print(
            f"{case.frame:<18}{case.desc:<16}"
            f"{_verdict(ok_f):<8}{_verdict(ok_c):<10}"
        )

    verdicts = {res["attention"]["focused"] for _, res in rows}
    print()
    if len(verdicts) == 1:
        print(f"❌ 所有画面的 focused 判定都一样（{verdicts.pop()}）——**模型很可能没在看画面**。")
        return 1
    if misses:
        print(f"⚠️  有 {misses} 项与期望不符。先确认是否只是占位图太抽象所致，再考虑调 prompt。")
        return 1

    skipped = [c.frame for c, _ in rows if c.focused is None]
    if skipped:
        print(f"（{'、'.join(skipped)} 未判定：当前是程序画的占位图，姿态类判断验不了）")
    print("✅ 全部符合期望，视觉判定确实随画面变化。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
