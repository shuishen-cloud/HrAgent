"""多模态 LLM 冒烟测试：真实调用一次，验证「看画面」的能力（会消耗少量额度）。

用法：
    python -m app.check_llm

做法：拿**同一段回答 + 同一个问题**，分别配上不同的画面帧，
看模型给出的 attention / cheating_suspected 判定是否随画面变化。
如果四种画面判定结果全都一样，说明它没在「看」，prompt 或模型选型需要调整。

注意：fixtures 里的画面帧是程序画的占位图，只能验证「链路通不通」。
要验证「判得准不准」，请把 tests/fixtures/frames/ 换成真实照片再跑。
"""
from __future__ import annotations

import sys

from . import config, llm

ANSWER = "我之前做过后端开发，主要负责订单系统的接口设计，用 Python 和 FastAPI 写过不少服务。"

# 画面 → 期望的判定，用来人工对照
CASES = [
    ("focused_01.jpg", "应该是：专注，无作弊"),
    ("focused_02.jpg", "应该是：专注，无作弊"),
    ("distracted_01.jpg", "应该是：走神（转头不看镜头）"),
    ("empty_01.jpg", "应该是：座位空着，人不在"),
    ("multiple_01.jpg", "应该是：画面里多个人 → 疑似作弊"),
]


def _load(name: str) -> bytes:
    return (config.FIXTURES_DIR / "frames" / name).read_bytes()


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

    failures = 0
    for name, expect in CASES:
        path = frames_dir / name
        if not path.exists():
            print(f"跳过 {name}（文件不存在）")
            continue

        print(f"\n{'=' * 62}\n画面：{name}    {expect}\n{'=' * 62}")
        try:
            result = llm.chat([], ANSWER, [_load(name)])
        except Exception as exc:
            failures += 1
            print(f"❌ 调用失败：{exc.__class__.__name__}: {exc}", file=sys.stderr)
            if "response_format" in str(exc):
                print("   提示：该端点不支持 JSON 模式，在 .env 里加 LLM_JSON_MODE=0", file=sys.stderr)
            continue

        if not result["_parsed"]:
            failures += 1
            print("⚠️  模型没按 JSON 格式返回（已走兜底）")

        ev, at = result["evaluation"], result["attention"]
        print(f"  评分    ：{ev['score']}")
        print(f"  点评    ：{ev['comment']}")
        print(f"  专注    ：{at['focused']}   （{at['note']}）")
        print(f"  疑似作弊：{result['cheating_suspected']}")
        print(f"  下一问  ：{result['next_question']}")
        print(f"  结束    ：{result['should_end']}")

    print(f"\n{'=' * 62}")
    if failures:
        print(f"有 {failures} 个用例调用失败，见上面的报错。")
        return 1
    print("全部调用成功。请人工对照每条的期望判定，确认模型确实在看画面。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
