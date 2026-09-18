"""集中配置：代码里只放默认值，全部可用环境变量覆盖。

切换 LLM 供应商不用改代码，改环境变量即可：
  OpenAI 官方： LLM_API_KEY=sk-xxx
  国内通义千问： LLM_API_KEY=sk-xxx  LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
                 LLM_MODEL=qwen-vl-max
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# ---- 目录 ----
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "static"
REPORTS_DIR = BASE_DIR / "reports"
FIXTURES_DIR = BASE_DIR / "tests" / "fixtures"

# ---- 多模态 LLM（看 + 想）----
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")  # 留空 = OpenAI 官方
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))
# 强制 JSON 输出。部分国内兼容端点不支持该参数，不支持时设为 0
LLM_JSON_MODE = os.getenv("LLM_JSON_MODE", "1") == "1"

# ---- 语音识别 STT（听）----
STT_MODEL = os.getenv("STT_MODEL", "small")  # tiny / base / small / medium
STT_LANGUAGE = os.getenv("STT_LANGUAGE", "zh")
STT_DEVICE = os.getenv("STT_DEVICE", "cpu")
STT_COMPUTE_TYPE = os.getenv("STT_COMPUTE_TYPE", "int8")
# 引导词：显著减少「输出繁体中文」的问题，也让标点更自然
STT_INITIAL_PROMPT = os.getenv("STT_INITIAL_PROMPT", "以下是普通话的面试对话，请用简体中文转写。")
# ctranslate2 在本机 CPU 上多线程会内存崩溃（corrupted double-linked list），
# 固定单线程换稳定。机器上跑得动的话可以调大，例如 STT_CPU_THREADS=4
STT_CPU_THREADS = int(os.getenv("STT_CPU_THREADS", "1"))

# ---- 语音合成 TTS（说）----
TTS_VOICE = os.getenv("TTS_VOICE", "zh-CN-XiaoxiaoNeural")

# ---- 面试流程 ----
MAX_TURNS = int(os.getenv("MAX_TURNS", "5"))          # 最多几轮问答
FRAME_INTERVAL_SEC = float(os.getenv("FRAME_INTERVAL_SEC", "2.0"))  # 前端抽帧间隔
