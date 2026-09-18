# HrAgent — 多模态 AI 面试官

一个能**看**（摄像头画面）和**听**（语音回答）的 AI 面试官智能体：主动提问、追问、评估候选人，最终产出面试报告。

## 技术栈

| 层 | 选择 | 说明 |
|---|---|---|
| 后端 | FastAPI | 轻量，一条命令起服务 |
| 前端 | 原生 HTML + JS 单页 | `getUserMedia` 拿摄像头 + 麦克风，无需框架 |
| 语音识别 STT | faster-whisper | 本地免费、中英文，整段转文字 |
| 多模态 LLM | **Qwen-VL-Max**（阿里通义，走 DashScope 兼容模式） | 文本 + 图片输入、中文强、国内直连 |
| 语音合成 TTS | edge-tts | 免费、中文音色自然、无需 key |
| 测试 | pytest | 单测 + 集成 + dry-run |
| 编排 | 纯 Python 状态机 | 不引入 LangChain / VAD |

## 架构

**关键设计**：音频不直接进 LLM。

```
录音整段 → faster-whisper 转文字 → 文字 + 摄像头帧 一起进多模态 LLM → 回复文字用 edge-tts 合成语音
```

### 数据流（点击录制 / 点击发送）

```
浏览器（Windows）
  1. AI 提问：后端返回 文字 + TTS 音频 → 前端播放
  2. 候选人点「开始回答」→ MediaRecorder 录音，期间 canvas 每 ~2s 抽一帧
  3. 点「发送」→ 录音文件 + 摄像头帧（base64）一起 POST 到后端
  4. 后端：STT 转文字 → 文字 + 帧 进 LLM → 返回 评估/追问 + TTS 音频
  5. 回到 1 循环；面试结束 → 生成报告
```

### 核心：一个纯函数，测试与线上共用

面试引擎不关心数据从哪来，只接收字节：

```python
# interview.py —— 唯一核心入口
def run_turn(audio_bytes: bytes, frames: list[bytes], state) -> TurnResult:
    text = stt.transcribe(audio_bytes)              # 听
    reply = llm.chat(state.history, text, frames)   # 看 + 想（结构化 JSON）
    audio = tts.synthesize(reply["question"])       # 说
    return TurnResult(text=reply, audio=audio, ...)
```

- **自动测试**：把 fixture 文件内容当作 `audio_bytes` / `frames` 喂进来
- **线上**：把上传的录音文件、抽到的帧当作同样的参数喂进来

核心引擎、测试、报告逻辑在接硬件前就写死并验证，接硬件时**零改动**。

## 部署环境

服务跑在 **WSL 的 Arch Linux**；摄像头 / 麦克风由 Windows 浏览器通过 `getUserMedia` 采集，服务端只做处理。
WSL2 会自动把 `localhost` 转发到 Windows，浏览器直接访问 `http://localhost:8000` 即可。

## 自动测试优先

接真实麦克风 / 摄像头之前，先用样本数据自动跑通全流程。

**准备 fixtures**（一次性）：
- `tests/fixtures/audio/`：几段语音 WAV（edge-tts 生成或人工录几段）
- `tests/fixtures/frames/`：几类画面帧 —— 正对镜头 / 转头走神 / 空座位 / 多人

**执行路径**（按序跑完，全程不碰硬件）：

1. **单元测试** `pytest tests/test_units.py`
   - `stt.py`：喂 WAV → 断言文字
   - `llm.py`：喂「帧 + 问题」→ 断言结构化 JSON（评估 / 追问 / 走神标记），外部 API mock
   - `tts.py`：喂文字 → 断言音频字节

2. **集成测试** `pytest tests/test_pipeline.py`
   - 用 fixture 音频 + 帧喂 `run_turn`，跑完一场完整模拟面试
   - 断言：状态机流转正确、产出报告、走神帧被正确标记

3. **端到端 dry-run** `python -m app.run_offline`
   - 用 fixture 跑完整面试，报告落到 `reports/`，人工可读检查质量

## 项目结构

```
HrAgent/
├── app/
│   ├── main.py          # FastAPI 入口 + /api/turn 路由
│   ├── stt.py           # faster-whisper 封装
│   ├── llm.py           # 多模态 LLM 封装（供应商可一行切换）
│   ├── tts.py           # edge-tts 封装
│   ├── interview.py     # run_turn + 状态机 + prompt
│   ├── run_offline.py   # 离线 dry-run CLI
│   └── config.py        # API key / 模型名等配置
├── static/
│   └── index.html       # 单页前端
├── tests/
│   ├── fixtures/
│   │   ├── audio/       # 样本语音 WAV
│   │   └── frames/      # 正对 / 走神 / 空座 / 多人 样本帧
│   ├── test_units.py    # 模块单测（mock 外部 API）
│   └── test_pipeline.py # 集成测试（跑完整场面试）
├── data/
│   └── questions.json   # 题库
├── reports/             # dry-run 产出的面试报告
├── requirements.txt
├── README.md            # 本文件
└── 工作管理.md           # 任务清单 + 进度跟踪
```

## 分阶段实施

### 阶段 A：模块 + 单测（纯逻辑，可离线验证）
1. 搭骨架：`requirements.txt`、`config.py`、目录结构
2. `llm.py`、`stt.py`、`tts.py` 三个模块
3. `tests/fixtures/` 准备样本音频 + 样本帧
4. `tests/test_units.py` 单测全绿（外部 API mock）

### 阶段 B：面试引擎 + 集成测试（跑通完整面试，仍无硬件）
5. `interview.py`：`run_turn` + 状态机 + prompt（「你是面试官，观察画面判断走神 / 作弊」）
6. `tests/test_pipeline.py` 用 fixtures 跑完整场面试，产出报告
7. `app/run_offline.py`：dry-run CLI，报告落盘

### 阶段 C：接真实硬件（最后做）
8. `main.py` FastAPI：`POST /api/turn`（收录音文件 + 帧，调 `run_turn`，返回文字 + TTS 音频）
9. 前端 `static/index.html`：摄像头抽帧 + 点击录音 / 发送 + 播放音频
10. 真实环境验证

## 运行

```bash
# 安装依赖
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt

# 生成测试样本（画面帧 + 样本语音，一次性）
.venv/bin/python tests/fixtures/make_fixtures.py

# 跑测试（无需硬件、无需联网、无需 API key）
.venv/bin/python -m pytest

# 离线 dry-run
.venv/bin/python -m app.run_offline

# 启动服务（阶段 C）
.venv/bin/uvicorn app.main:app --reload
```

浏览器打开 `http://localhost:8000`，授权摄像头 / 麦克风 → 点录音 → 点发送 → 看到文字回复 + 听到语音追问。

## 环境与已知问题（本机实测）

### 1. 加载 Whisper 模型会卡死在联网检查上（已在代码里修掉）

**症状**：跑到第一轮 STT 就停住不动 —— 不报错、不退出，看着像死循环。

**原因**：模型即使已经缓存到本地，`huggingface_hub` 加载时**仍会联网检查有没有新版本**。
国内连不上 `huggingface.co`，而且是「丢包挂起」不是「快速失败」，于是进程一直等。

**已修**：`stt.py` 加载模型时**先走离线模式**（只用本地缓存，秒开不碰网络）；
只有本地确实没缓存、离线加载失败时，才回退到联网下载，并自动指向 `HF_ENDPOINT`
（默认 `https://hf-mirror.com`）。环境变量只在加载期间生效，用完还原。

**所以你不需要手动 export 任何东西。** 首次下载模型才需要网络。

> 如果哪天缓存被清了又下不动，可以手动 `export HF_ENDPOINT=https://hf-mirror.com
> HF_HUB_DISABLE_XET=1`（xet 存储后端在国内也不通）再跑一次。

### 2. ctranslate2 多线程会导致进程崩溃（已规避）

本机（WSL2 / Arch）上 faster-whisper 多线程推理会 `malloc(): unsorted double linked list corrupted` 直接 SIGABRT。
**已默认 `STT_CPU_THREADS=1` 规避**，稳定但慢。换机器可试着调大。

### 3. `small` 模型中文识别有错字

实测样本（`small` + int8）：`Python→Pathong`、`日志→日制`、`索引→索隐`、`响应→想应`、`源码→原码`。

- 已在 `stt.py` 加 `initial_prompt` 引导词，解决了「输出繁体中文」的问题
- 错字属模型能力问题，**架构上已容忍**：`llm.py` 的系统提示里明确告诉 LLM「转写可能有错别字，请合理理解」
- 若要更好的中文精度，换 SenseVoice / FunASR（见下方「可替换点」）

### 4. LLM 密钥配置（本项目用 Qwen-VL-Max）

```bash
cp .env.example .env
# 然后编辑 .env，填入 DashScope 密钥
```

`.env` 内容（`app/config.py` 会自动加载，已加进 `.gitignore` 不会外泄）：

```
LLM_API_KEY=sk-你的DashScope密钥
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen-vl-max
```

配好后跑冒烟测试，真实调用一次验证「看画面」的能力（会消耗少量额度）：

```bash
.venv/bin/python -m app.check_llm
```

它会拿**同一段回答**分别配不同画面，自动对比 `focused` / `cheating_suspected` 是否随画面变化，
并打出通过/失败汇总。判定不随画面变化 = 模型没在「看」。

> 完整测试说明：`pytest` 不需要 key（全部打桩）；只有 `check_llm` 会真实调 LLM。

### 实测结论（2026-09-18，Qwen-VL-Max）

链路全部打通：端点连通 ✅、鉴权 ✅、`response_format` JSON 模式 ✅、解析 ✅。

视觉判定：

| 画面 | 结果 | 说明 |
|---|---|---|
| 正对镜头 ×2 | ✅ | 判为专注 |
| 画面里多个人 | ✅ | 判为疑似作弊 |
| 空座位 | ✅ | 判为人不在 |
| 转头走神 | ⚠️ **验不了** | 见下 |

**重要限制：程序画的占位图只能验证「结构性线索」，验不了「姿态类判断」。**

- 「几个人」「有没有人」是结构性线索，抽象色块也保留得住 → 判得对
- 「头转没转」「低没低头」是细粒度姿态，需要真实人像先验 → 实测无论怎么画，
  模型都读作「一个人坐在镜头前」。做过多轮对照：侧脸（鼻梁/耳朵/后脑头发齐全）、
  低头（只见头顶）、后脑勺（无五官）—— 全部判为 `focused=True`

**所以要验证走神检测，必须换成真实照片**：把 `tests/fixtures/frames/` 里的文件用
同名真实照片替换（用自己摄像头拍几张即可），再把 `app/check_llm.py` 里
`distracted_01.jpg` 那行的 `focused` 从 `None` 改回 `False` 就生效。

## 关键依赖

```
fastapi, uvicorn[standard], faster-whisper, edge-tts,
openai(或 dashscope), python-multipart, pytest, pytest-asyncio
```

## 可替换点（按需换，不影响架构）

- **LLM**：无 OpenAI 访问 → 换 `llm.py` 里的 Qwen-VL-Max / GLM-4V（国内通义 / 智谱）
- **STT**：追求中文精度 → 换 FunASR / 阿里 SenseVoice
- **TTS**：追求更强音色 → 换 CosyVoice / OpenAI TTS
- **前端**：想快速出 demo → 用 Gradio 替代手写 HTML
