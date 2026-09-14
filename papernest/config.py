import os
import re
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "papernest.db"


def _strip_env_value(v: str) -> str:
    """`.env` 的值：剥掉行内注释与外层引号。

    原来是原样 `v.strip()`，于是 `KEY=value  # 注释` 会把整串注释一起塞进
    `os.environ`。这不是假想：项目自己的 `.env.example` 就有这种写法
    （`PAPERNEST_RERANK=off          # api | llm | bm25 | off…`），而 README 明确让人
    `copy .env.example .env`——「照文档做」这条路径本身就会产出被污染的值，且全程静默。

    口径与通行的 dotenv 一致，两条都必要：
    · 带引号的值**整体取引号内的内容**、不再剥注释（值里本来就可能有 ` #`，
      比如口令；引号就是「我要字面量」的显式声明）；
    · 不带引号时，`#` 只有**前面是空白**才算注释起点——URL 片段
      （`https://x/y#frag`）里的 `#` 不该被当注释。
    """
    v = v.strip()
    if len(v) >= 2 and v[0] in "\"'" and v[-1] == v[0]:
        return v[1:-1]
    return re.split(r"\s#", v, maxsplit=1)[0].strip()


def _env_float(default: float, *names: str) -> float:
    """读环境变量里的浮点数；**非法值降级到默认，不在 import 期把整个包炸掉**。

    原来是模块级裸 `float(os.environ.get(...))`：`PAPERNEST_CHUNK_WEIGHT=abc` 会在
    import 阶段抛 ValueError，把 papernest 连同 uvicorn 与 cli.py 一起拖死——
    一个排序权重配错，整个服务起不来。而且原来的 `A or B` 链让「设成空串」
    直接跳过 A 去拿 B 的默认值，看着像生效了其实没有。
    """
    for n in names:
        raw = os.environ.get(n)
        if raw is None or not raw.strip():
            continue
        try:
            return float(raw)
        except ValueError:
            warnings.warn(f"{n}={raw!r} 不是合法的数，已按默认值 {default} 处理",
                          RuntimeWarning, stacklevel=2)
            return default
    return default


def _env_int(default: int, name: str, minimum: int = 1) -> int:
    """同上，整数版；额外夹一个下界（0 或负数会让批大小/并发数变成死循环或崩溃）。"""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        warnings.warn(f"{name}={raw!r} 不是合法的整数，已按默认值 {default} 处理",
                      RuntimeWarning, stacklevel=2)
        return default


def _load_env():
    """极简 .env 读取：不引依赖；已存在的环境变量优先。"""
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), _strip_env_value(v))


_load_env()

# 数据源
S2_API_KEY = os.environ.get("S2_API_KEY", "")
# OpenAlex 礼貌池：填了邮箱能进更快的队列。不填就不带该参数（走匿名池，仍可用），
# 不在代码里硬编码任何人的私人邮箱——那会随仓库一起公开出去。
OPENALEX_MAILTO = os.environ.get("OPENALEX_MAILTO", "")

# LLM（OpenAI 兼容；不配则 L1 卡片走 mock 摘要直取模式）
LLM_API_BASE = os.environ.get("LLM_API_BASE", "")
# key 兜底链：LLM_API_KEY → DASHSCOPE_API_KEY（阿里云百炼，环境变量已有则不落盘）
LLM_API_KEY = os.environ.get("LLM_API_KEY") or os.environ.get("DASHSCOPE_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "")
# 重档模型（L2 全文精读用），不填则与 LLM_MODEL 相同——分级用模
LLM_MODEL_HEAVY = os.environ.get("LLM_MODEL_HEAVY", "")
# 向量模型（RAG 检索与证据句匹配）；不填则降级 FTS/词面重叠
EMBED_MODEL = os.environ.get("EMBED_MODEL", "")
# 视觉模型（插图理解用）。**默认为空 = 关掉这个功能**：它按图片像素计费，
# 悄悄给一个默认值等于替用户决定花钱。
# 实测可选：qwen3-vl-plus（1168 token/图）、qwen3.5-omni-plus（809 token/图，更省
# 且在「数曲线条数」上比另外两个准）、qwen3-vl-flash / qwen3.5-omni-flash（更便宜，
# 但 flash 那两个都犯过「说五条列四条」的数量错误）。
VISION_MODEL = os.environ.get("PAPERNEST_VISION_MODEL", "")

# embedding 可以指向与 chat **不同的**端点。实测踩到：常见的中转站只转发
# /chat/completions，/embeddings 与 /rerank 都是 404——而 embedding 又必须有，
# 否则语义检索整条链路是哑的。留两个口子分别覆盖 base 与 key，都不填就沿用 LLM_*。
EMBED_API_BASE = os.environ.get("EMBED_API_BASE") or LLM_API_BASE
EMBED_API_KEY = (os.environ.get("EMBED_API_KEY")
                 or os.environ.get("DASHSCOPE_API_KEY")
                 or LLM_API_KEY)


def heavy_model() -> str:
    return LLM_MODEL_HEAVY or LLM_MODEL

# 课题描述：卡片里「与我课题的关系」的上下文
RESEARCH_TOPIC = os.environ.get("RESEARCH_TOPIC", "大模型 Agent 的构建与评测")

# 出网代理：OpenAlex / arXiv 在部分网络下被阻断，开启 Clash 等代理后在此填
# 例如 http://127.0.0.1:7897。不填则直连（Semantic Scholar 可直连）。
PAPERNEST_PROXY = os.environ.get("PAPERNEST_PROXY", "")

# 计价表（USD / 1M tokens）：JSON，键为模型 ID 或模型 ID 前缀，值 [prompt, completion]。
# 未匹配到价格的模型 cost 记 NULL（如实不编造）。示例见 .env.example 的 LLM_PRICES_JSON。
LLM_PRICES_JSON = os.environ.get("LLM_PRICES_JSON", "")


def model_price(model: str) -> tuple[float, float] | None:
    """按「精确 ID → 前缀」顺序匹配计价表，返回 (prompt, completion) 每百万 tokens 美元价。"""
    if not (LLM_PRICES_JSON and model):
        return None
    try:
        import json
        table = json.loads(LLM_PRICES_JSON)
    except ValueError:
        return None
    if model in table:
        price = table[model]
    else:
        price = next((v for k, v in table.items()
                      if model.lower().startswith(k.lower())), None)
    if not price or len(price) < 2:
        return None
    return float(price[0]), float(price[1])

HTTP_TIMEOUT = 30
