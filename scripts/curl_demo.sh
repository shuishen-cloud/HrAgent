#!/usr/bin/env bash
# 用 curl 走一遍完整面试，验证 HTTP 接口。不需要浏览器、不需要摄像头和麦克风。
#
# 用法：
#   scripts/curl_demo.sh                     # 3 轮：专注 / 专注 / 多人
#   scripts/curl_demo.sh --turns 2           # 只跑 2 轮
#   scripts/curl_demo.sh --frames multiple_01.jpg focused_01.jpg
#   scripts/curl_demo.sh --audio answer_02.mp3 --frames focused_02.jpg
#   scripts/curl_demo.sh --host 127.0.0.1:8000
#
# 前提：服务已在跑（.venv/bin/uvicorn app.main:app）
set -uo pipefail

HOST="${HOST:-127.0.0.1:8000}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
FIX="$ROOT/tests/fixtures"

TURNS=3
AUDIO=()
FRAMES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)   HOST="$2"; shift 2 ;;
    --turns)  TURNS="$2"; shift 2 ;;
    --audio)  shift; while [[ $# -gt 0 && "$1" != --* ]]; do AUDIO+=("$1"); shift; done ;;
    --frames) shift; while [[ $# -gt 0 && "$1" != --* ]]; do FRAMES+=("$1"); shift; done ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "未知参数：$1" >&2; exit 2 ;;
  esac
done

# 默认脚本：第 3 轮故意用「画面里多个人」来触发作弊判定
[[ ${#AUDIO[@]}  -eq 0 ]] && AUDIO=(answer_01.mp3 answer_02.mp3 answer_03.mp3)
[[ ${#FRAMES[@]} -eq 0 ]] && FRAMES=(focused_01.jpg focused_02.jpg multiple_01.jpg)

# 把 JSON 里的 audio 字段（一大坨 base64）压成摘要，其余原样打印
fmt() {
  "$PY" -c '
import json,sys
try:
    d = json.load(sys.stdin)
except Exception:
    print(sys.stdin.read()[:400]); raise SystemExit(1)
if isinstance(d, dict) and isinstance(d.get("audio"), str):
    d["audio"] = "<%d 字节 base64>" % len(d["audio"])
print(json.dumps(d, ensure_ascii=False, indent=2))
'
}

die() { echo "❌ $*" >&2; exit 1; }

# --- 前置检查 ---------------------------------------------------------------
curl -sf -o /dev/null "http://$HOST/api/status" \
  || die "连不上 http://$HOST —— 先启动服务：.venv/bin/uvicorn app.main:app"

echo "服务：http://$HOST"
echo

# --- 开始面试 ---------------------------------------------------------------
echo "── POST /api/start ─────────────────────────────"
curl -s -X POST "http://$HOST/api/start" | fmt || die "开始面试失败"

# --- 逐轮 --------------------------------------------------------------
for ((i = 0; i < TURNS; i++)); do
  a="${AUDIO[$((i % ${#AUDIO[@]}))]}"
  f="${FRAMES[$((i % ${#FRAMES[@]}))]}"
  [[ -f "$FIX/audio/$a"  ]] || die "找不到 $FIX/audio/$a"
  [[ -f "$FIX/frames/$f" ]] || die "找不到 $FIX/frames/$f"

  echo
  echo "── 第 $((i + 1)) 轮：$a + $f ─────────────────"
  resp=$(curl -s -X POST "http://$HOST/api/turn" \
    -F "audio=@$FIX/audio/$a" \
    -F "frames=@$FIX/frames/$f") || die "请求失败"
  echo "$resp" | fmt || die "返回的不是 JSON：$resp"

  # 结束后再发应当被拒（后端守卫）
  if [[ "$(echo "$resp" | "$PY" -c 'import json,sys;print(json.load(sys.stdin).get("finished"))')" == "True" ]]; then
    echo
    echo "（面试已结束，验证守卫：再发一轮应当被拒）"
    curl -s -o /dev/null -w "  结束后再提交 → HTTP %{http_code}（期望 409）\n" \
      -X POST "http://$HOST/api/turn" \
      -F "audio=@$FIX/audio/$a" -F "frames=@$FIX/frames/$f"
    break
  fi
done

# --- 报告 -------------------------------------------------------------------
echo
echo "── GET /api/report ────────────────────────────"
curl -s "http://$HOST/api/report" | "$PY" -c '
import json,sys
d = json.load(sys.stdin)
print("轮数      :", d["turns"])
print("平均分    :", d["average_score"])
print("走神轮次  :", d["distracted_turns"] or "无")
print("疑似作弊  :", d["cheating_turns"] or "无")
print("结论      :", d["recommendation"])
print()
print("--- markdown 报告 ---")
print(d["markdown"])
' || die "取报告失败"
