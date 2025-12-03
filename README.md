# PinHan 拼音输入法引擎

轻量级智能拼音输入法引擎，纯词典架构，专为嵌入式/MCU 设备优化。

## 特性

- **纯词典架构**: 无深度学习依赖，轻量高效
- **低延迟**: 平均 5-10ms，缓存命中 < 1ms
- **模糊纠错**: 支持常见拼音错误（如 `shengme` → 什么，`zh/z` 模糊音）
- **标点透传**: 拼音中可直接输入标点（如 `nihao!!haha` → 你好!!哈哈）
- **易部署**: 支持 pip 安装、Docker 部署

## 快速开始

### 方式 1: Docker 部署（推荐）

```bash
docker run -d -p 3000:3000 ghcr.io/trisliate/pinhan
```

### 方式 2: pip 安装

```bash
pip install pinhan

# 启动 API 服务
pinhan-server

# 或命令行查询
pinhan query nihao
```

### 方式 3: 源码安装

```bash
git clone https://github.com/trisliate/PinHan.git
cd PinHan
pip install -e .
pinhan-server
```

## API 接口

服务启动后访问 http://localhost:3000/docs 查看完整 API 文档。

### POST /ime

主接口，返回完整信息。

```bash
curl -X POST http://localhost:3000/ime \
  -H "Content-Type: application/json" \
  -d '{"pinyin": "nihao", "top_k": 5}'
```

响应:
```json
{
  "raw_pinyin": "nihao",
  "segmented_pinyin": ["ni", "hao"],
  "candidates": [
    {"text": "你好", "score": 0.1, "source": "dict"},
    {"text": "呢好", "score": 0.01, "source": "dict"}
  ],
  "metadata": {"elapsed_ms": 5.2, "cached": false}
}
```

### GET /ime/simple

简单接口，只返回候选文本。

```bash
curl "http://localhost:3000/ime/simple?pinyin=nihao&top_k=5"
```

### GET /health

健康检查接口。

### GET /stats

获取引擎统计信息（缓存命中率等）。

## 命令行工具

```bash
# 查看版本
pinhan version

# 查询拼音
pinhan query nihao
pinhan query shi -c "我做" -k 10

# 启动服务
pinhan server --host 0.0.0.0 --port 3000
```

## Python 调用

```python
from pinhan import create_engine_v3

engine = create_engine_v3()

# 简单查询
result = engine.process("nihao")
print([c.text for c in result.candidates])  # ['你好', '呢好', ...]

# 带上下文
result = engine.process("shi", context="我做")
print(result.candidates[0].text)  # '事'
```

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
返回 Top-K 候选
```

### 核心模块

| 模块 | 功能 | 文件 |
|------|------|------|
| **Engine** | 主引擎，协调各模块 | `pinhan/engine/core.py` |
| **Segmenter** | 拼音切分（动态规划） | `pinhan/engine/segmenter.py` |
| **Corrector** | 拼音纠错（模糊音+编辑距离） | `pinhan/engine/corrector.py` |
| **Dictionary** | 字典查询服务 | `pinhan/engine/dictionary.py` |
| **Generator** | 候选生成 | `pinhan/engine/generator.py` |

## 目录结构

```
PinHan/
├── pinhan/                # Python 包
│   ├── __init__.py
│   ├── cli.py             # 命令行工具
│   ├── api/               # API 服务
│   │   └── server.py
│   ├── engine/            # 核心引擎
│   │   ├── core.py
│   │   ├── config.py
│   │   ├── dictionary.py
│   │   ├── corrector.py
│   │   ├── segmenter.py
│   │   ├── generator.py
│   │   ├── cache.py
│   │   └── logging.py
│   └── data/              # 词典数据
│       └── dicts/
├── pyproject.toml         # 打包配置
├── Dockerfile             # Docker 构建
└── README.md
```

## 性能指标

| 场景 | 延迟 | 说明 |
|------|------|------|
| 缓存命中 | < 1ms | LRU 缓存 2000 条 |
| 无缓存 | 5-10ms | 纯词典查询 |

## 技术栈

- **Python 3.9+**
- **FastAPI** + Uvicorn（API 服务）
- **orjson**（高性能 JSON）
- **pypinyin**（拼音处理）

## License

MIT
