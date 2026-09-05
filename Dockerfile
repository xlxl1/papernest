FROM python:3.12-slim

WORKDIR /app

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

COPY papernest ./papernest
COPY web ./web
COPY cli.py verify_sources.py eval_set.json ./

# 数据全部落在 /app/data，挂卷即持久化：
#   papernest.db（论文/全文/FTS/向量原始 BLOB）· pdf/ · chroma/（Chroma 派生索引）· 评测报告
# Chroma 只是派生索引——真相来源始终是 papernest.db 的 vectors 表，
# 所以 chroma/ 丢了也能用 `cli.py vec rebuild` 重建，不需要单独备份。
VOLUME ["/app/data"]
EXPOSE 8765

# 容器内必然绑 0.0.0.0。如实声明给应用，它会在「暴露且无口令」时拒绝启动
# （逃生口：PAPERNEST_ALLOW_NO_AUTH=1）。
ENV PAPERNEST_BIND_HOST=0.0.0.0
CMD ["uvicorn", "papernest.api:app", "--host", "0.0.0.0", "--port", "8765"]
