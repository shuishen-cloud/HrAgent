"""集中配置：代码里只放默认值，全部可用环境变量覆盖。

支持项目根目录下的 .env 文件（见 .env.example），省得每次 export。
已存在的环境变量优先于 .env，方便临时覆盖。

切换 LLM 供应商不用改代码，改环境变量即可：
  OpenAI 官方： LLM_API_KEY=sk-xxx
  阿里通义千问： LLM_API_KEY=sk-xxx
                 LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
                 LLM_MODEL=qwen-vl-max
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """极简 .env 加载器（十来行，不值得为它加个依赖）。

    只认 KEY=VALUE，支持 # 注释和引号；已存在的环境变量不被覆盖。
    """
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(BASE_DIR / ".env")

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
# 强制 JSON 输出。部分兼容端点不支持该参数，报错时设为 0
LLM_JSON_MODE = os.getenv("LLM_JSON_MODE", "1") == "1"

# ---- 语音识别 STT（听）----
STT_MODEL = os.getenv("STT_MODEL", "small")  # tiny / base / small / medium
STT_LANGUAGE = os.getenv("STT_LANGUAGE", "zh")
STT_DEVICE = os.getenv("STT_DEVICE", "cpu")
STT_COMPUTE_TYPE = os.getenv("STT_COMPUTE_TYPE", "int8")
# 引导词：显著减少「输出繁体中文」的问题，也让标点更自然
STT_INITIAL_PROMPT = os.getenv("STT_INITIAL_PROMPT", "以下是普通话的面试对话，请用简体中文转写。")
# 短于这个秒数的录音不送进模型 —— 音频太短时 Whisper 会把 initial_prompt
# 原样吐回来（实测 0.3 秒音频 → "请用简体中文转写。"），比空串更隐蔽
STT_MIN_SECONDS = float(os.getenv("STT_MIN_SECONDS", "0.8"))
# ctranslate2 在本机 CPU 上多线程会内存崩溃（corrupted double-linked list），
# 固定单线程换稳定。机器上跑得动的话可以调大，例如 STT_CPU_THREADS=4
STT_CPU_THREADS = int(os.getenv("STT_CPU_THREADS", "1"))

# ---- HuggingFace（只影响首次下载 Whisper 模型）----
# 国内直连 huggingface.co 不通，且是「挂起」不是「快速失败」，
# 会导致加载模型时无限等待。代码里已默认优先走本地缓存（离线模式），
# 只有本地没有缓存、需要下载时才会用到下面这个镜像地址。
HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")

# ---- 语音合成 TTS（说）----
TTS_VOICE = os.getenv("TTS_VOICE", "zh-CN-XiaoxiaoNeural")
# 合成超时。没有它的话，edge-tts 挂起会让持锁的整轮请求永不返回
TTS_TIMEOUT = float(os.getenv("TTS_TIMEOUT", "30"))

# ---- 面试流程 ----
MAX_TURNS = int(os.getenv("MAX_TURNS", "5"))          # 最多几轮问答
FRAME_INTERVAL_SEC = float(os.getenv("FRAME_INTERVAL_SEC", "2.0"))  # 前端抽帧间隔


def describe_llm() -> str:
    """给日志/冒烟测试用的一句话描述，注意别把 key 打出来。"""
    where = LLM_BASE_URL or "OpenAI 官方"
    key = "已配置" if LLM_API_KEY else "**未配置**"
    return f"model={LLM_MODEL}  endpoint={where}  key={key}"
