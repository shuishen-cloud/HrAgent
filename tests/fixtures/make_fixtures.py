"""一次性生成测试样本（fixtures）。已存在则跳过，加 --force 覆盖。

用法：
    python tests/fixtures/make_fixtures.py

生成内容：
    frames/*.jpg  —— 合成占位图：正对镜头 / 转头走神 / 空座位 / 多人。
                     **这些是程序画的占位图，只为让自动测试跑起来。**
                     要真正验证视觉判定效果，请用真实照片替换它们。
    audio/*.mp3   —— 用 edge-tts 合成的中文「候选人回答」样本语音（需要联网）。
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FRAMES_DIR = HERE / "frames"
AUDIO_DIR = HERE / "audio"

W, H = 640, 480
BG = (235, 238, 242)
SKIN = (226, 190, 160)
SHIRT = (90, 120, 180)
DARK = (70, 75, 85)
CHAIR = (150, 150, 155)


def _draw_chair(img) -> None:
    """空座位：只有一把椅子，没有人。"""
    from PIL import ImageDraw

    d = ImageDraw.Draw(img)
    d.rounded_rectangle([W // 2 - 150, H - 260, W // 2 + 150, H - 40], radius=24, fill=CHAIR)
    d.rectangle([W // 2 - 170, H - 40, W // 2 - 140, H - 10], fill=DARK)
    d.rectangle([W // 2 + 140, H - 40, W // 2 + 170, H - 10], fill=DARK)


def _draw_person(img, cx: int, head_r: int, facing: str = "front") -> None:
    """在图上画一个简笔人物。facing='front' 正对，'side' 侧头。"""
    from PIL import ImageDraw

    d = ImageDraw.Draw(img)

    # 肩膀 / 身体
    d.rounded_rectangle(
        [cx - head_r * 2, H - 210, cx + head_r * 2, H - 20],
        radius=60,
        fill=SHIRT,
    )
    # 头
    d.ellipse([cx - head_r, H - 210 - head_r * 2, cx + head_r, H - 210], fill=SKIN)

    head_cy = H - 210 - head_r  # 头部圆心

    if facing == "side":
        # 侧脸朝右：鼻梁凸出 + 只露一只眼 + 后脑勺的耳朵。
        # 关键是让「头本身转了」，而不只是眼睛偏移 —— 否则人看也认为它是朝前的。
        d.polygon(
            [
                (cx + head_r * 0.75, head_cy - head_r * 0.15),
                (cx + head_r * 1.45, head_cy + head_r * 0.22),
                (cx + head_r * 0.72, head_cy + head_r * 0.5),
            ],
            fill=SKIN,
        )
        # 唯一可见的那只眼，靠右
        d.ellipse(
            [cx + head_r * 0.3, head_cy - head_r * 0.3,
             cx + head_r * 0.58, head_cy - head_r * 0.06],
            fill=DARK,
        )
        # 后脑的头发（先画，别盖住耳朵）
        d.chord(
            [cx - head_r, head_cy - head_r, cx + head_r, head_cy + head_r],
            100, 260, fill=DARK,
        )
        # 耳朵，靠左（后脑侧）
        d.ellipse(
            [cx - head_r * 0.95, head_cy - head_r * 0.05,
             cx - head_r * 0.6, head_cy + head_r * 0.35],
            fill=SKIN, outline=DARK, width=3,
        )
    else:
        # 正对：两眼居中 + 微笑
        d.ellipse([cx - head_r * 0.5, H - 210 - head_r * 1.4,
                   cx - head_r * 0.2, H - 210 - head_r * 1.15], fill=DARK)
        d.ellipse([cx + head_r * 0.2, H - 210 - head_r * 1.4,
                   cx + head_r * 0.5, H - 210 - head_r * 1.15], fill=DARK)
        d.arc([cx - head_r * 0.45, H - 210 - head_r * 0.95,
               cx + head_r * 0.45, H - 210 - head_r * 0.4], 0, 180, fill=DARK, width=5)


def make_frames(force: bool = False) -> list[Path]:
    from PIL import Image

    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    specs = {
        "focused_01.jpg": lambda im: _draw_person(im, W // 2, 70, "front"),
        "focused_02.jpg": lambda im: _draw_person(im, W // 2 - 30, 65, "front"),
        "distracted_01.jpg": lambda im: _draw_person(im, W // 2, 70, "side"),
        "empty_01.jpg": lambda im: _draw_chair(im),
        "multiple_01.jpg": lambda im: (
            _draw_person(im, W // 3, 55, "front"),
            _draw_person(im, W - W // 3, 55, "front"),
        ),
    }

    written = []
    for name, draw in specs.items():
        path = FRAMES_DIR / name
        if path.exists() and not force:
            written.append(path)
            continue
        img = Image.new("RGB", (W, H), BG)
        draw(img)
        img.save(path, quality=88)
        written.append(path)
    return written


SAMPLE_ANSWERS = {
    "answer_01.mp3": "我之前在一家公司做过后端开发，主要负责订单系统的接口设计，"
                     "用 Python 和 FastAPI 写过不少服务，也做过一些性能优化的工作。",
    "answer_02.mp3": "遇到过一个线上接口超时的问题，我先看了监控和日志，"
                     "定位到是数据库查询没有走索引，加了索引之后响应时间从两秒降到了两百毫秒。",
    "answer_03.mp3": "我平时会看一些技术博客和开源项目的源码，最近在学分布式相关的内容，"
                     "比如消息队列和一致性协议，也会自己写一些小项目练手。",
}


async def _synth(text: str) -> bytes:
    import edge_tts

    communicate = edge_tts.Communicate(text, "zh-CN-XiaoxiaoNeural")
    buf = bytearray()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.extend(chunk["data"])
    return bytes(buf)


def make_audio(force: bool = False) -> list[Path]:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for name, text in SAMPLE_ANSWERS.items():
        path = AUDIO_DIR / name
        if path.exists() and not force:
            written.append(path)
            continue
        try:
            path.write_bytes(asyncio.run(_synth(text)))
        except Exception as exc:  # 没网 / 没装 edge-tts 时不该让整个脚本失败
            print(f"  [跳过] {name}：合成失败（{exc.__class__.__name__}: {exc}）", file=sys.stderr)
            continue
        written.append(path)
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description="生成测试样本")
    ap.add_argument("--force", action="store_true", help="已存在也重新生成")
    args = ap.parse_args()

    try:
        frames = make_frames(args.force)
        print(f"画面帧 {len(frames)} 张 → {FRAMES_DIR}")
    except ImportError:
        print("未安装 Pillow，跳过画面帧生成（pip install pillow）", file=sys.stderr)

    audio = make_audio(args.force)
    print(f"样本语音 {len(audio)} 段 → {AUDIO_DIR}")

    if not audio:
        print("提示：样本语音生成失败，需要联网。集成测试会退化为使用合成静音。", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
