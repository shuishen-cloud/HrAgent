# HrAgent — 多模态 AI 面试官

一个能**看**（摄像头画面）和**听**（语音回答）的 AI 面试官：主动提问、追问、评估候选人，最终产出面试报告。

上传候选人简历后，面试官会**围绕简历出题**并深挖细节；不上传则走内置固定题库。

## 技术栈

| 层 | 选择 | 说明 |
|---|---|---|
| 后端 | FastAPI | 轻量，一条命令起服务 |
| 前端 | 原生 HTML + JS 单页 | `getUserMedia` 拿摄像头 + 麦克风，无需框架、无构建步骤 |
| 简历解析 | pypdfium2 / python-docx | PDF（PDFium 引擎）、Word、txt、md |
| 语音识别 STT | faster-whisper | 本地免费、中英文，整段转文字 |
| 多模态 LLM | **Qwen-VL-Max**（阿里通义，走 DashScope 兼容模式） | 文本 + 图片输入、中文强、国内直连 |
| 语音合成 TTS | edge-tts | 免费、中文音色自然、无需 key |
| 测试 | pytest | 143 个用例，全部离线可跑 |
| 编排 | 纯 Python 状态机 | 不引入 LangChain / VAD |

## 架构

**两个关键设计**：

1. **音频不直接进 LLM** —— 先转文字，再和画面帧一起进多模态模型
2. **TTS 不在主链路上** —— 接口只回文字，前端显示后再异步取语音

```
简历文件 ──解析──► 简历文本 ──并入系统提示──► 面试官围绕简历提问
                                                  │
录音整段 ──STT──► 回答文字 ──┐                    │
                            ├──► 多模态 LLM ──► 点评 + 追问（文字）
摄像头帧 ────────────────────┘                    │
                                                  ▼
                                    前端先显示文字，再异步请求 /api/tts 播报
```

### 一轮的完整流程

```
浏览器                          后端
  │  点击「开始回答」
  │  ├─ MediaRecorder 录音
  │  └─ canvas 每 2s 从 <video> 抽一帧
  │  再点一下结束
  │
  ├── POST /api/turn（录音 + 帧）──►  STT 转文字
  │                                  文字 + 帧 → Qwen-VL-Max
  │                                  返回：点评 / 是否走神 / 疑似作弊 / 下一题
  ◄── 只回文字 ─────────────────────┘
  │
  ├─ 立刻把文字显示在对话区
  └── POST /api/tts（下一题文字）──►  edge-tts 合成
      ◄── 音频 base64 ─────────────┘  → 播放
```

**为什么把 TTS 拆出去**：合成要联网、要 1~2 秒。塞在 `/api/turn` 里，用户得白等这段时间
才看得到文字。拆开后**看到文字从 ~11.5s 降到 6.05s**（语音随后 1.4s 到达）。
副作用是 `run_turn` 里不再有联网操作，「TTS 失败导致状态半改」那一类问题彻底消失。

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/` | 前端页面 |
| `GET` | `/api/status` | 面试进度、简历是否已加载 |
| `POST` | `/api/resume` | 上传简历（multipart）→ 解析成文本 |
| `GET` | `/api/resume` | 当前简历状态与预览 |
| `DELETE` | `/api/resume` | 移除简历，回到固定题库 |
| `POST` | `/api/start` | 开始面试 → 开场问题（**只有文字**） |
| `POST` | `/api/tts` | 文字 → 语音（前端异步调） |
| `POST` | `/api/turn` | 录音 + 帧 → 点评 / 追问（**只有文字**） |
| `GET` | `/api/report` | 面试报告（结构化 + markdown） |
| `GET` | `/api/fixtures` | 列出生模拟素材（模拟模式用） |

## 核心：引擎只接收字节

面试引擎不关心数据从哪来：

```python
# app/interview.py —— 唯一核心入口
def run_turn(state, audio_bytes: bytes = b"", frames: list[bytes] | None = None) -> TurnResult:
    answer = stt.transcribe(audio_bytes)                 # 听
    reply = llm.chat(state.history, answer, frames,      # 看 + 想
                     system=llm.system_prompt_for(state.resume))
    ...                                                  # 提交状态
```

- **自动测试**：把 fixture 文件内容当作 `audio_bytes` / `frames` 喂进来
- **线上**：把上传的录音、抽到的帧当作同样的参数喂进来

同一个入口既跑测试也跑线上，所以接真实摄像头时引擎零改动。

## 项目结构

```
HrAgent/
├── app/
│   ├── main.py          # FastAPI 入口 + 全部路由
│   ├── interview.py     # 引擎：start / run_turn / 报告
│   ├── llm.py           # 多模态 LLM 封装（含 JSON 解析容错）
│   ├── stt.py           # faster-whisper 封装（含幻觉防护）
│   ├── tts.py           # edge-tts 封装（带超时）
│   ├── resume.py        # 简历解析：PDF / docx / txt / md
│   ├── config.py        # 全部配置（环境变量 + .env）
│   ├── run_offline.py   # 离线 dry-run CLI
│   └── check_llm.py     # 视觉能力冒烟测试（真实调 LLM）
├── static/index.html    # 单页前端
├── tests/
│   ├── test_units.py     # 模块单测（stt / llm / tts）
│   ├── test_pipeline.py  # 引擎集成测试
│   ├── test_api.py       # 接口测试
│   ├── test_resume.py    # 简历解析测试
│   └── fixtures/         # 样本帧 + 样本语音
├── data/questions.json  # 题库（含预设答案，供降级用）
├── scripts/curl_demo.sh # 纯 curl 走完整流程
├── reports/             # 产出的报告（不入库）
├── requirements.txt
├── README.md
└── 工作管理.md           # 任务清单 + 进度 + 决策记录
```

## 运行

```bash
# 安装依赖
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt

# 生成测试样本（画面帧 + 样本语音，一次性）
.venv/bin/python tests/fixtures/make_fixtures.py

# 跑测试（无需硬件、无需联网、无需 API key）
.venv/bin/python -m pytest

# 离线 dry-run（--mock 完全不联网）
.venv/bin/python -m app.run_offline --mock

# 启动服务
.venv/bin/uvicorn app.main:app --reload
```

浏览器打开 `http://localhost:8000`：上传简历（可选）→ 开始面试 → 点一下开始回答、
再点一下结束 → 看到文字点评、听到语音追问。

**纯 curl 验证**（不需要浏览器）：

```bash
scripts/curl_demo.sh --turns 2
```

## 环境与已知问题（本机实测）

### 1. 加载 Whisper 模型：必须用 `local_files_only`，不能用 `HF_HUB_OFFLINE`

**症状**：新机器永远下不了模型 —— 即使联网正常。

**原因**：`huggingface_hub` 在**首次 import 时**就把 `HF_HUB_OFFLINE` / `HF_ENDPOINT`
读成模块常量，之后再改 `os.environ` 一律不生效。所以「先设 `HF_HUB_OFFLINE=1` 试试离线、
失败了再放开」这种写法是**死代码**：常量一旦锁成 True 就回不去。

**已修**：离线改用 faster-whisper 原生的 `local_files_only` 参数（正是为这个场景设计的）；
镜像地址在 import **之前**用 `setdefault` 设好。实测空缓存目录下 18.6 秒成功下载。

### 2. ctranslate2：默认线程数会崩，但 4 线程又快又稳

**症状**：`malloc(): unsorted double linked list corrupted` 直接 SIGABRT。

**实测澄清**：崩的是**默认线程数**（= CPU 核数；本机 32 核 → 崩），固定 1 线程只是躲开了它。
benchmark（2 秒样本，small 模型）：

| 线程数 | 耗时 |
|---|---|
| 1 | 3.22s |
| **4** | **1.29s** |
| 8 | 1.28s |
| 16 | 1.21s |

**4 线程已吃满收益（约 2.5x）**，再往上没有额外好处。默认值现为 `STT_CPU_THREADS=4`。
真实 12 秒音频：4.81s → 1.97s。

### 3. Whisper 在无效音频上会「幻觉」，不是返回空

**症状**：说了一句话，转写出来却是 `请用简体中文转写。` —— 那正是 `initial_prompt` 的原文。

**原因**：音频太短或无效时（实测 **0.3 秒**），Whisper 不会老实返回空，而是**幻觉出固定话术**。
这类输出**不是空串**，所以只判断 `if not text` 拦不住，它会冒充成正常回答进报告。

**已修**（两层，都在 `stt.py`）：
- `STT_MIN_SECONDS`（默认 0.8 秒）—— 音频太短直接不采信
- `_looks_like_hallucination()` —— 命中提示词回吐或已知幻听话术则清成空串

两者都返回空串，于是降级逻辑自然生效。

**注意**：幻觉特征只收「组合起来才成立」的完整话术（`请不吝点赞`、`字幕由` 等）。
`订阅`/`转发` 这种**单词不能当特征** —— 候选人完全可能说「我订阅了一个技术专栏」，
**误杀真回答比漏判更糟**。

### 4. 简历解析：PDF 库必须用 pypdfium2，不能用 pypdf

**症状**：上传一个**不是 PDF** 的文件（改错扩展名、下载损坏）时，
pypdf 会**间歇性段错误**把整个进程打掉 —— 崩的是**服务**，不只是这一次请求。

**实测数据**：非 PDF 样本 8/8 崩，正常 PDF 约 10% 崩（连跑 20 次崩 2 次）。

**已修**：换 pypdfium2（PDFium，Chrome 的 PDF 引擎）。同样输入只抛可捕获的 `PdfiumError`。
稳定性实测 **0/15 段错误**。

**教训**：第三方解析库在**对抗性输入**下不能假设「只会抛异常」。
测试里要放一个「伪装成 PDF 的文本」用例，只测正常文件根本发现不了。

### 5. `small` 模型中文识别有错字

实测：`Python→Pathong`、`日志→日制`、`索引→索隐`、`响应→想应`、`源码→原码`。

- 已加 `initial_prompt` 引导词，解决了「输出繁体中文」的问题
- 错字属模型能力问题，**架构上已容忍**：系统提示里明确告诉 LLM
  「转写可能有错别字，请合理理解」—— 实测 LLM 会把「索隐」正确读成「索引」并顺着追问
- 若要更好的中文精度，换 SenseVoice / FunASR（见下方「可替换点」）

### 6. LLM 密钥配置（本项目用 Qwen-VL-Max）

```bash
cp .env.example .env
# 编辑 .env，填入 DashScope 密钥
```

```
LLM_API_KEY=sk-你的DashScope密钥
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen-vl-max
```

> `.env` 已在 `.gitignore` 里，不会外泄。**别把密钥贴进聊天或提交到仓库。**

配好后跑视觉冒烟测试（真实调 LLM，会消耗少量额度）：

```bash
.venv/bin/python -m app.check_llm
```

它拿**同一段回答**分别配不同画面，自动对比 `focused` / `cheating_suspected` 是否随画面变化。
判定不随画面变化 = 模型没在「看」。**零用例执行时会返回 2 而不是假报成功。**

### 视觉能力的实测边界（2026-09-18，Qwen-VL-Max）

| 画面 | 结果 |
|---|---|
| 正对镜头 ×2 | ✅ 判为专注 |
| 画面里多个人 | ✅ 判为疑似作弊 |
| 空座位 | ✅ 判为人不在 |
| 转头走神 | ⚠️ **占位图验不了** |

**程序画的占位图只能验证「结构性线索」，验不了「姿态类判断」**：
「几个人」「有没有人」抽象后仍保留，判得对；「头转没转」需要真实人像先验，
实测侧脸/低头/后脑勺全被判为 `focused=True`。

**要验证走神检测必须换真实照片**：用同名真实照片替换 `tests/fixtures/frames/` 里的文件，
再把 `app/check_llm.py` 里 `distracted_01.jpg` 那行的 `focused` 从 `None` 改回 `False`。

## 关键依赖

```
fastapi, uvicorn[standard], python-multipart,
faster-whisper, edge-tts, openai,
pypdfium2, python-docx,
pytest, pillow
```

## 可替换点（按需换，不影响架构）

- **LLM**：换 `LLM_BASE_URL` / `LLM_MODEL` 环境变量即可（OpenAI / 通义 / 智谱都试过同一套代码）
- **STT**：追求中文精度 → 换 FunASR / 阿里 SenseVoice
- **TTS**：追求更强音色 → 换 CosyVoice / OpenAI TTS
- **前端**：想快速出 demo → 用 Gradio 替代手写 HTML（但音频流仍建议手写）
