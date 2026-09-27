FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 默认走腾讯云 pip 镜像（国内服务器快几十倍）；本地/海外构建可 --build-arg 覆盖
ARG PIP_INDEX_URL=https://mirrors.cloud.tencent.com/pypi/simple
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i ${PIP_INDEX_URL}

COPY . .

# 非 root 运行；HOME 指到 /home/appuser，Whisper 模型缓存从这里挂持久卷
# .cache 目录在镜像里预建并归 appuser 所有 —— 命名卷首次挂载会继承这个属主
RUN useradd --uid 10001 --create-home appuser \
    && mkdir -p /home/appuser/.cache \
    && chown -R appuser:appuser /app /home/appuser/.cache
USER appuser
ENV HOME=/home/appuser

EXPOSE 8000

# 部署.md 硬性要求：单 worker（面试会话状态在进程内全局单例）
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--proxy-headers", "--forwarded-allow-ips", "*"]
