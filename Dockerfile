FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN useradd --system --uid 10001 --create-home --shell /usr/sbin/nologin papernest

COPY requirements.txt .

# Chroma 后端已停用——项目定的向量库是 Milvus，默认**不装** chromadb，
# 镜像因此省下约 163MB（那棵依赖树里还有用不到的 onnxruntime / kubernetes）。
# 要跑 Chroma 那部分测试再打开：docker compose build --build-arg WITH_CHROMA=1
# 注：chromadb 与 tokenizers 都发 abi3 wheel（cp39-abi3 / cp310-abi3），
# 在 python:3.12-slim 上直接装二进制，不会现场编译，构建不会因此变慢。
ARG WITH_CHROMA=0
RUN set -eux; \
    if [ "$WITH_CHROMA" = "0" ]; then \
        grep -v '^chromadb' requirements.txt > /tmp/req.txt; \
    else \
        cp requirements.txt /tmp/req.txt; \
    fi; \
    pip install --no-cache-dir -r /tmp/req.txt; \
    rm -f /tmp/req.txt

COPY --chown=papernest:papernest papernest ./papernest
COPY --chown=papernest:papernest web ./web
COPY --chown=papernest:papernest cli.py verify_sources.py eval_set.json ./

# 数据全部落在 /app/data，挂卷即持久化：
#   papernest.db（论文/全文/FTS/向量原始 BLOB）· pdf/ · chroma/（Chroma 派生索引）· 评测报告
# Chroma 只是派生索引——真相来源始终是 papernest.db 的 vectors 表，
# 所以 chroma/ 丢了也能用 `cli.py vec rebuild` 重建，不需要单独备份。
EXPOSE 8765

# 不以 root 运行应用；挂载的数据目录由部署环境负责授予该 UID 写权限。
RUN mkdir -p /app/data \
    && chown -R papernest:papernest /app
VOLUME ["/app/data"]
USER papernest

# 只验证进程和静态入口可达，不触发外部 embedding/LLM 请求。
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/', timeout=3)"

# 容器内必然绑 0.0.0.0。如实声明给应用，它会在「暴露且无口令」时拒绝启动
# （逃生口：PAPERNEST_ALLOW_NO_AUTH=1）。
ENV PAPERNEST_BIND_HOST=0.0.0.0
CMD ["uvicorn", "papernest.api:app", "--host", "0.0.0.0", "--port", "8765"]
