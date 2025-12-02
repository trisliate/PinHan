# PinHan 拼音输入法引擎

轻量级智能拼音输入法引擎，采用 **词典召回 + SLM 重排序** 架构，专为嵌入式/MCU 设备优化。

## 特性

- **精简架构**: 词典为主，SLM 辅助重排序（仅在有上下文时使用）
- **低延迟**: 平均 10-15ms，缓存命中 < 1ms
- **模糊纠错**: 支持常见拼音错误（如 `shengme` → 什么，`zh/z` 模糊音）
- **上下文感知**: SLM 利用上下文消歧（如 `我做` + `shi` → 事）
- **标点透传**: 拼音中可直接输入标点（如 `nihao!!haha` → 你好!!哈哈）
- **轻量模型**: SLM Lite 仅 ~1M 参数，适合嵌入式部署

## 架构说明

```
用户输入 (nihao) 
    ↓
[分词器] nihao → ["ni", "hao"]
    ↓
[纠错器] 模糊音扩展 (可选)
    ↓
[词典召回] 查 char_dict / word_dict → 候选列表
    ↓
[SLM 重排序] (仅当有上下文时) → 最终排序
    ↓
返回 Top-K 候选
```

### 核心模块

| 模块 | 功能 | 文件 |
|------|------|------|
| **Engine** | 主引擎，协调各模块 | `engine/core.py` |
| **Segmenter** | 拼音切分（动态规划） | `engine/segmenter.py` |
| **Corrector** | 拼音纠错（模糊音+编辑距离） | `engine/corrector.py` |
| **Dictionary** | 字典查询服务 | `engine/dictionary.py` |
| **SLM** | 语义语言模型（重排序） | `slm/model.py` |
| **Logging** | 日志记录 | `engine/logging.py` |

## 性能指标

| 场景 | 延迟 | 说明 |
|------|------|------|
| 缓存命中 | < 1ms | LRU 缓存 2000 条 |
| 无上下文 | 5-10ms | 纯词典查询 |
| 有上下文 | 10-20ms | 词典 + SLM 重排 |
| 段落连续输入 | ~10ms/词 | 实测 Top-1 93.5%, Top-3 100% |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 构建词典

```bash
python scripts/build_dict.py
```

### 3. 准备训练数据 (可选)

```bash
python scripts/build_training_data.py --xml zhwiki-latest-pages-articles.xml --max-samples 50000
```

### 4. 训练 SLM Lite（可选，已预置）

```bash
python slm/train_lite.py --epochs 20 --batch-size 128 --max-samples 50000
```

模型配置: 2 层 Transformer, 128 维, 4 头, 5000 词表, ~1M 参数

### 5. 启动 API 服务

```bash
python -m api.server
```

- API 文档: http://localhost:8000/docs
- 健康检查: http://localhost:8000/health

### 6. 调用示例

```bash
# 简单查询
curl "http://localhost:8000/ime/simple?pinyin=nihao"

# 带上下文
curl -X POST http://localhost:8000/ime \
  -H "Content-Type: application/json" \
  -d '{"pinyin": "shi", "context": "我做", "top_k": 5}'
```

```python
# Python 调用
from engine import create_engine_v3

engine = create_engine_v3()
result = engine.process("nihao", context="")
print([c.text for c in result.candidates])  # ['你好', '拟好', ...]
```

## 目录结构

```
PinHan/
├── api/                   # API 服务
│   └── server.py
├── data/                  # 数据目录
│   ├── dicts/             # 生成的字典文件
│   ├── patches/           # 高优先级词频修正
│   └── sources/           # 外部数据源 (SUBTLEX-CH, cedict)
├── engine/                # 核心引擎（所有核心模块）
│   ├── __init__.py        # 统一导出
│   ├── core.py            # 主引擎逻辑
│   ├── config.py          # 配置类
│   ├── dictionary.py      # 字典查询
│   ├── corrector.py       # 拼音纠错
│   ├── segmenter.py       # 拼音切分
│   ├── generator.py       # 候选生成
│   ├── cache.py           # LRU 缓存
│   └── logging.py         # 日志配置
├── scripts/               # 构建脚本
│   ├── build_dict.py
│   └── build_training_data.py
├── slm/                   # 语言模型
│   ├── model.py
│   └── train_lite.py
├── checkpoints/           # 模型检查点
├── logs/                  # 日志输出
├── tests/                 # 测试
└── requirements.txt
```

## 测试

```bash
# 运行所有测试
pytest tests/ -v

# 单独测试
python tests/test_story.py       # 段落+标点透传测试
python tests/test_local.py       # 本地引擎基础测试
```

## 技术栈

- **Python 3.11** + PyTorch 2.x + CUDA
- **FastAPI** + Uvicorn（API 服务）
- **Transformer Decoder**（SLM 语言模型）
- **SUBTLEX-CH** + **CC-CEDICT**（词频和拼音映射）
- **orjson**（高性能 JSON）

## API 接口

### POST /ime

主接口，返回完整信息。

**请求体:**
```json
{
  "pinyin": "nihao",
  "context": "今天天气",
  "top_k": 10
}
```

**响应:**
```json
{
  "raw_pinyin": "nihao",
  "segmented_pinyin": ["ni", "hao"],
  "candidates": [
    {"text": "你好", "score": 0.95, "source": "dict"},
    {"text": "拟好", "score": 0.82, "source": "dict"}
  ],
  "metadata": {"elapsed_ms": 12.5, "cached": false}
}
```

### GET /ime/simple

简单接口，只返回候选文本。

```
GET /ime/simple?pinyin=nihao&top_k=5
```

### GET /stats

获取引擎统计信息（缓存命中率、SLM 调用率等）。

## 日志系统

项目内置分模块日志，自动输出到 `logs/` 目录：

| 日志文件 | 内容 |
|----------|------|
| `pinhan.api.log` | API 请求/响应日志 |
| `pinhan.engine.log` | 引擎处理日志 |
| `pinhan.train.log` | 模型训练日志 |
| `*_error.log` | 错误单独记录 |

```python
# 使用日志
from engine import get_api_logger, get_engine_logger

logger = get_api_logger()
logger.info("请求处理完成")
```

## 版本历史

- **v3.0** - 精简架构，移除 P2H，仅保留词典 + SLM Lite，标点透传
- **v2.0** - 分层策略 (P2H + SLM)
- **v1.0** - 初版

## License

MIT
