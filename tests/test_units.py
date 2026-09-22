"""阶段 A 单测：stt / llm / tts 三个模块。

原则：不装重依赖、不联网、不需要 API key —— 全部通过模块里预留的接缝打桩。
    llm  → 打桩 llm._get_client
    stt  → 打桩 stt._get_model
    tts  → 往 sys.modules 塞一个假的 edge_tts
"""
from __future__ import annotations

import base64
import json
import os
import sys
import types

import pytest

from app import llm, stt, tts

# ---------------------------------------------------------------- llm: 拼消息


def test_build_messages_结构正确():
    msgs = llm.build_messages(
        history=[{"role": "assistant", "content": "请自我介绍"}],
        answer="我叫张三",
        frames=[b"\xff\xd8fake-jpeg"],
    )

    assert msgs[0]["role"] == "system"
    assert msgs[1] == {"role": "assistant", "content": "请自我介绍"}
    assert msgs[2]["role"] == "user"

    content = msgs[2]["content"]
    assert isinstance(content, list)
    # 第一段是文字，且带上了候选人的回答
    assert content[0]["type"] == "text"
    assert "我叫张三" in content[0]["text"]
    # 最后一段是图片，base64 编码正确
    image_part = content[-1]
    assert image_part["type"] == "image_url"
    assert image_part["image_url"]["url"].startswith("data:image/jpeg;base64,")
    b64 = image_part["image_url"]["url"].split(",", 1)[1]
    assert base64.b64decode(b64) == b"\xff\xd8fake-jpeg"


def test_build_messages_无画面也能工作():
    msgs = llm.build_messages([], "回答", [])
    content = msgs[-1]["content"]
    assert all(part["type"] == "text" for part in content)
    assert "没有采集到画面帧" in content[-1]["text"]


def test_build_messages_空回答标记为没有说话():
    msgs = llm.build_messages([], "", [])
    assert "候选人没有说话" in msgs[-1]["content"][0]["text"]


# ---------------------------------------------------------------- llm: 解析回复

GOOD_REPLY = {
    "evaluation": {"score": 8, "comment": "表达清晰"},
    "attention": {"focused": True, "note": "正对镜头"},
    "cheating_suspected": False,
    "next_question": "能具体讲讲吗？",
    "should_end": False,
}


def test_parse_reply_标准_json():
    got = llm.parse_reply(json.dumps(GOOD_REPLY, ensure_ascii=False))
    assert got["evaluation"]["score"] == 8
    assert got["next_question"] == "能具体讲讲吗？"
    assert got["cheating_suspected"] is False
    assert got["_parsed"] is True


def test_parse_reply_容忍_markdown_围栏():
    raw = f"```json\n{json.dumps(GOOD_REPLY, ensure_ascii=False)}\n```"
    assert llm.parse_reply(raw)["evaluation"]["score"] == 8


def test_parse_reply_容忍前后多余文字():
    raw = f"好的，这是我的判断：{json.dumps(GOOD_REPLY, ensure_ascii=False)} 希望有帮助"
    assert llm.parse_reply(raw)["next_question"] == "能具体讲讲吗？"


def test_parse_reply_补全缺失字段():
    got = llm.parse_reply('{"next_question": "下一题"}')
    assert got["next_question"] == "下一题"
    assert got["evaluation"]["score"] is None      # 缺失 → None
    assert got["attention"]["focused"] is True    # 缺失 → 默认专注
    assert got["cheating_suspected"] is False


def test_parse_reply_字符串分数转成整数():
    got = llm.parse_reply('{"evaluation": {"score": "7"}}')
    assert got["evaluation"]["score"] == 7


def test_parse_reply_彻底解析失败时不把原文当问题():
    """模型被 max_tokens 截断时返回的常是半截的 ```json {...}。

    那样的乱码会被 TTS **念给候选人听**，还会写进报告的问题字段。
    所以兜底要换成一句安全追问，原文另存 _raw 供排查。
    """
    got = llm.parse_reply("```json\n{\"evaluation\": {\"score\": 7")
    assert got["_parsed"] is False
    assert "```" not in got["next_question"]
    assert "evaluation" not in got["next_question"]
    assert got["next_question"]          # 是一句能念出口的话
    assert got["should_end"] is False    # 不因为解析失败就结束面试
    assert "_raw" in got                 # 原文没丢，供排查


def test_parse_reply_空字符串也不崩():
    got = llm.parse_reply("")
    assert got["next_question"]  # 有默认问题
    assert got["_parsed"] is False


# ---------------------------------------------------------------- 类型纠错
# 这一组是回归测试。模型偶尔会把布尔/数字写成字符串，
# 直接用 bool() 会把 "false" 读成 True —— 在招聘场景里就是
# 「没作弊」被判成「作弊」，是最不能出错的方向。

def test_字符串_false_不能被当成_true():
    got = llm.parse_reply('{"cheating_suspected": "false", "next_question": "x"}')
    assert got["cheating_suspected"] is False


def test_字符串_true_能识别():
    assert llm.parse_reply('{"cheating_suspected": "true"}')["cheating_suspected"] is True


def test_should_end_字符串_false_不会提前结束面试():
    assert llm.parse_reply('{"should_end": "false"}')["should_end"] is False


def test_走神字段字符串_false_不会被算成走神():
    got = llm.parse_reply('{"attention": {"focused": "false"}, "next_question": "x"}')
    assert got["attention"]["focused"] is False


def test_认不出的布尔值走安全默认():
    """认不出来时宁可漏报，也不能误伤候选人。"""
    got = llm.parse_reply('{"cheating_suspected": "说不清", "attention": {"focused": "?"}}')
    assert got["cheating_suspected"] is False   # 不说人作弊
    assert got["attention"]["focused"] is True  # 不说人走神


def test_布尔值写法汇总():
    for v in (False, "false", "False", "no", 0, "否", "无", None):
        assert llm.parse_reply(json.dumps({"cheating_suspected": v}))["cheating_suspected"] is False, v
    for v in (True, "true", "True", "yes", 1, "是"):
        assert llm.parse_reply(json.dumps({"cheating_suspected": v}))["cheating_suspected"] is True, v


@pytest.mark.parametrize(
    "raw,expected",
    [(7, 7), ("7", 7), (7.0, 7), ("7.5", 8), ("8分", 8), ("给7分", 7), (None, None), (True, None)],
)
def test_分数容错(raw, expected):
    got = llm.parse_reply(json.dumps({"evaluation": {"score": raw}}))
    assert got["evaluation"]["score"] == expected


def test_attention_是字符串时不崩溃():
    """曾经这里会抛 AttributeError，整轮面试直接 502。"""
    got = llm.parse_reply('{"attention": "候选人正对镜头", "next_question": "x"}')
    assert got["attention"]["focused"] is True
    assert got["attention"]["note"] == "候选人正对镜头"   # 内容不丢


@pytest.mark.parametrize("bad", [123, None, ["a"], 3.14])
def test_字段类型完全不对也不抛异常(bad):
    got = llm.parse_reply(json.dumps({"attention": bad, "evaluation": bad, "next_question": "x"}))
    assert got["_parsed"] is True
    assert isinstance(got["attention"]["focused"], bool)
    assert got["next_question"]


# ---------------------------------------------------------------- llm: 调用

class _FakeCompletions:
    def __init__(self, content: str, recorder: dict):
        self._content = content
        self._recorder = recorder

    def create(self, **kwargs):
        self._recorder.update(kwargs)
        msg = types.SimpleNamespace(content=self._content)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


class _FakeClient:
    def __init__(self, content: str, recorder: dict):
        self.chat = types.SimpleNamespace(
            completions=_FakeCompletions(content, recorder)
        )


def test_chat_走通并回传结构化结果(monkeypatch):
    recorder: dict = {}
    payload = json.dumps(GOOD_REPLY, ensure_ascii=False)
    monkeypatch.setattr(llm, "_get_client", lambda: _FakeClient(payload, recorder))

    got = llm.chat([], "我叫张三", [b"frame"])

    assert got["evaluation"]["score"] == 8
    # 验证请求参数确实带上了图片
    sent = recorder["messages"][-1]["content"]
    assert any(p["type"] == "image_url" for p in sent)
    assert recorder["temperature"] == 0.7


def test_chat_可关闭_json_mode(monkeypatch):
    recorder: dict = {}
    monkeypatch.setattr(llm, "_get_client", lambda: _FakeClient("{}", recorder))

    monkeypatch.setattr(llm.config, "LLM_JSON_MODE", False)
    llm.chat([], "回答", [])
    assert "response_format" not in recorder

    monkeypatch.setattr(llm.config, "LLM_JSON_MODE", True)
    llm.chat([], "回答", [])
    assert recorder["response_format"] == {"type": "json_object"}


# ---------------------------------------------------------------- stt

class _FakeSegment:
    def __init__(self, text: str):
        self.text = text


class _FakeWhisperModel:
    def __init__(self, texts, recorder: dict, duration: float = 10.0):
        self._texts = texts
        self._recorder = recorder
        self._duration = duration

    def transcribe(self, audio, **kwargs):
        self._recorder["audio"] = audio
        self._recorder["kwargs"] = kwargs
        # info.duration 是判断「音频太短」的依据，假模型也得给
        info = types.SimpleNamespace(language="zh", duration=self._duration)
        return [_FakeSegment(t) for t in self._texts], info


def test_transcribe_拼接分段并去空白(monkeypatch):
    recorder: dict = {}
    fake = _FakeWhisperModel([" 你好，", "我是张三。 "], recorder)
    monkeypatch.setattr(stt, "_get_model", lambda: fake)

    assert stt.transcribe(b"fake-audio-bytes") == "你好，我是张三。"
    assert recorder["kwargs"]["language"] == "zh"
    assert recorder["kwargs"]["vad_filter"] is True


def test_transcribe_空输入直接返回空串(monkeypatch):
    def _boom():
        raise AssertionError("空输入不应该去加载模型")

    monkeypatch.setattr(stt, "_get_model", _boom)
    assert stt.transcribe(b"") == ""


# ---------------------------------------------------------------- 幻觉防护
# Whisper 在无效音频（极短、底噪、切在半句话上）上不会返回空，
# 而是幻觉出固定话术 —— 最常见的是把 initial_prompt 原样吐回来。
# 实测：0.3 秒音频 → "请用简体中文转写。"
# 这类输出不是空串，只判断 `if not text` 拦不住，会冒充成正常回答进报告。

def test_把提示词吐回来算幻觉():
    assert stt._looks_like_hallucination("请用简体中文转写。") is True
    assert stt._looks_like_hallucination("以下是普通话的面试对话，请用简体中文转写。") is True


def test_常见幻听句算幻觉():
    for phrase in ["谢谢观看", "请不吝点赞订阅", "字幕由 XXX 提供", "amara.org"]:
        assert stt._looks_like_hallucination(phrase) is True, phrase


def test_空串算幻觉():
    assert stt._looks_like_hallucination("") is True
    assert stt._looks_like_hallucination("   ") is True


@pytest.mark.parametrize(
    "text",
    [
        "我之前在一家公司做过后端开发",
        "请用一下你说的那个技术",          # 含「请用」但整体是真话
        "我今天订阅了一个技术专栏",        # 含「订阅」但是正常内容
    ],
)
def test_正常回答不算幻觉(text):
    assert stt._looks_like_hallucination(text) is False, text


def test_音频太短直接返回空(monkeypatch):
    """0.3 秒的音频不该送进结果 —— 实测会吐回提示词。"""
    fake = _FakeWhisperModel(["请用简体中文转写。"], {}, duration=0.3)
    monkeypatch.setattr(stt, "_get_model", lambda: fake)

    assert stt.transcribe(b"short") == ""


def test_音频够长且内容正常就正常返回(monkeypatch):
    fake = _FakeWhisperModel(["我叫张三"], {}, duration=5.0)
    monkeypatch.setattr(stt, "_get_model", lambda: fake)

    assert stt.transcribe(b"ok") == "我叫张三"


def test_幻觉输出被清成空_让上层走降级(monkeypatch):
    """关键：短音频幻觉出的非空垃圾必须变成空串，
    否则 run_turn 里「空串才降级」的判断不会触发。"""
    fake = _FakeWhisperModel(["请用简体中文转写。"], {}, duration=8.0)
    monkeypatch.setattr(stt, "_get_model", lambda: fake)

    assert stt.transcribe(b"x") == ""


def test_标点归一化成全角():
    assert stt._normalize("你好﹐我是张三.") == "你好，我是张三."
    assert stt._normalize("真的吗﹖太好了﹗") == "真的吗？太好了！"
    assert stt._normalize("a,b;c") == "a，b；c"


def test_transcribe_结果已归一化(monkeypatch):
    fake = _FakeWhisperModel(["你好﹐", "我是张三"], {})
    monkeypatch.setattr(stt, "_get_model", lambda: fake)
    assert stt.transcribe(b"x") == "你好，我是张三"


def test_加载模型优先走本地缓存(monkeypatch):
    """已缓存时必须纯本地加载，否则会卡在联网检查上（实测会无限挂起）。"""
    seen = {}

    def fake_build(local_files_only):
        seen["local_files_only"] = local_files_only
        return "MODEL"

    monkeypatch.setattr(stt, "_build_model", fake_build)
    monkeypatch.setattr(stt, "_model", None)

    assert stt._get_model() == "MODEL"
    assert seen["local_files_only"] is True


def test_本地无缓存时回退到联网下载(monkeypatch):
    """本地加载失败 = 没缓存，此时才该联网（local_files_only=False）。

    这条同时锁住那个 bug：**不能用 HF_HUB_OFFLINE 表达「先离线」**。
    huggingface_hub 在 import 时把该常量读死，之后删环境变量也回退不了，
    于是「先离线」变成「永远离线」—— 回退这一步照样被挡，新机器永远下不了模型。
    所以断言「全程不碰这个变量」。
    """
    calls = []

    def fake_build(local_files_only):
        calls.append(local_files_only)
        if local_files_only:                     # 第一次：纯本地，没缓存
            raise RuntimeError("not found in cache")
        return "DOWNLOADED"

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(stt, "_build_model", fake_build)
    monkeypatch.setattr(stt, "_model", None)

    assert stt._get_model() == "DOWNLOADED"
    assert calls == [True, False]                # 先本地，再联网
    assert "HF_HUB_OFFLINE" not in os.environ    # 全程不碰它


def test_镜像地址赶在_import_之前设好(monkeypatch):
    """HF_ENDPOINT 必须在首次 import huggingface_hub 之前设好，否则不生效。

    它是 import 时读取的常量，等 import 完再设就晚了 —— 会直连
    huggingface.co（国内丢包挂起，进程无声卡死），而不是走镜像。
    """
    seen = {}

    def fake_build(local_files_only):
        seen["endpoint"] = os.environ.get("HF_ENDPOINT")
        seen["xet"] = os.environ.get("HF_HUB_DISABLE_XET")
        return "MODEL"

    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
    monkeypatch.setattr(stt, "_build_model", fake_build)
    monkeypatch.setattr(stt, "_model", None)

    stt._get_model()

    assert seen["endpoint"] == stt.config.HF_ENDPOINT  # 走镜像
    assert seen["xet"] == "1"                          # 关掉国内不通的 xet


def test_用户自己设的镜像地址不被覆盖(monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", "https://my-mirror.example")
    monkeypatch.setattr(stt, "_build_model", lambda local_files_only: "MODEL")
    monkeypatch.setattr(stt, "_model", None)

    stt._get_model()

    assert os.environ["HF_ENDPOINT"] == "https://my-mirror.example"


def test_两次加载都失败时异常照样抛出(monkeypatch):
    """别把异常吞掉 —— 上层靠它报错，吞了就变成「静默没模型」。"""

    def always_fail(local_files_only):
        raise RuntimeError("boom")

    monkeypatch.setattr(stt, "_build_model", always_fail)
    monkeypatch.setattr(stt, "_model", None)

    with pytest.raises(RuntimeError):
        stt._get_model()


def test_transcribe_模型只加载一次(monkeypatch):
    """_get_model 内部的缓存逻辑：连续调用应复用同一个实例。"""
    calls = []

    class _Loader:
        def __init__(self, *a, **kw):
            calls.append(1)

    fake_module = types.SimpleNamespace(WhisperModel=_Loader)
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)
    monkeypatch.setattr(stt, "_model", None)

    first = stt._get_model()
    second = stt._get_model()

    assert first is second
    assert len(calls) == 1


# ---------------------------------------------------------------- tts

def _install_fake_edge_tts(monkeypatch, chunks, recorder: dict):
    class _FakeCommunicate:
        def __init__(self, text, voice):
            recorder["text"] = text
            recorder["voice"] = voice

        async def stream(self):
            for c in chunks:
                yield c

    monkeypatch.setitem(sys.modules, "edge_tts", types.SimpleNamespace(Communicate=_FakeCommunicate))


def test_synthesize_只拼接音频分片(monkeypatch):
    recorder: dict = {}
    _install_fake_edge_tts(
        monkeypatch,
        [
            {"type": "audio", "data": b"AAA"},
            {"type": "WordBoundary", "data": b"SHOULD-BE-IGNORED"},
            {"type": "audio", "data": b"BBB"},
        ],
        recorder,
    )

    assert tts.synthesize("你好") == b"AAABBB"
    assert recorder["text"] == "你好"


def test_synthesize_空文字不调用合成(monkeypatch):
    recorder: dict = {}
    _install_fake_edge_tts(monkeypatch, [], recorder)

    assert tts.synthesize("") == b""
    assert tts.synthesize("   ") == b""
    assert "text" not in recorder  # 压根没发起合成


# ---------------------------------------------------------------- 分数与布尔边界

@pytest.mark.parametrize("raw,expected", [(85, 10), (-3, 0), (100, 10), (0, 0), (10, 10)])
def test_分数夹在_0_到_10(raw, expected):
    """报告用 8/6 两档阈值。换模型/端点后对方若按百分制给分，
    平均分会变成 85，结论直接失真成「建议通过」。"""
    got = llm.parse_reply(json.dumps({"evaluation": {"score": raw}}))
    assert got["evaluation"]["score"] == expected


@pytest.mark.parametrize("blank", ["", "   "])
def test_空字符串不该被当成走神(blank):
    """空串是「没给出值」，该走 default（偏向专注）。
    若把它当「认得出的 False」，模型输出空值时候选人就被记成走神 ——
    正是 _as_bool 声明要避免的误伤方向。"""
    got = llm.parse_reply(json.dumps({"attention": {"focused": blank}}))
    assert got["attention"]["focused"] is True


def test_明确表示否定的词仍然生效():
    for word in ["false", "no", "否", "无", "不", "0"]:
        got = llm.parse_reply(json.dumps({"attention": {"focused": word}}))
        assert got["attention"]["focused"] is False, word
