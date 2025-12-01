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
| **Engine** | 主引擎，协调各模块 | `engine.py` |
| **Segmenter** | 拼音切分（动态规划） | `segmenter/segmenter.py` |
| **Corrector** | 拼音纠错（模糊音+编辑距离） | `corrector/corrector.py` |
| **DictService** | 词典查询服务 | `dicts/service.py` |
| **SLM** | 语义语言模型（重排序） | `slm/model.py` |

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

### 2. 构建词典（可选，已预置）

```bash
# 从 CC-CEDICT + jieba 构建词典
python preprocess/build_dict.py
```

### 3. 训练 SLM Lite（可选，已预置）

```bash
# 需要先准备训练数据
python preprocess/build_training_data.py --xml zhwiki-latest-pages-articles.xml --max-samples 50000

# 训练模型
python slm/train_lite.py --epochs 20 --batch-size 128 --max-samples 50000
```

模型配置: 2 层 Transformer, 128 维, 4 头, 5000 词表, ~1M 参数

### 4. 启动 API 服务

```bash
python -m api.server
```

- API 文档: http://localhost:8000/docs
- 健康检查: http://localhost:8000/health

### 5. 调用示例

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
├── engine.py              # IME 引擎主入口
├── api/
│   └── server.py          # FastAPI 服务
├── corrector/
│   └── corrector.py       # 拼音纠错（模糊音、编辑距离）
├── segmenter/
│   └── segmenter.py       # 拼音切分（DP 最优切分）
├── slm/
│   ├── model.py           # SLM 模型定义
│   └── train_lite.py      # SLM Lite 训练脚本
├── dicts/
│   ├── service.py         # 词典服务
│   ├── char_dict.json     # 单字字典 (拼音→汉字)
│   ├── word_dict.json     # 词组字典 (拼音序列→词组)
│   ├── char_freq.json     # 字频表
│   └── word_freq.json     # 词频表
├── preprocess/
│   ├── build_dict.py      # 词典构建脚本
│   └── build_training_data.py  # 训练数据生成
├── checkpoints/
│   └── slm_lite/          # SLM Lite 模型文件
├── tests/
│   ├── test_story.py      # 段落输入+标点测试
│   └── ...
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
- **CC-CEDICT** + **jieba**（词典和词频）
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

## 版本历史

- **v3.0** - 精简架构，移除 P2H，仅保留词典 + SLM Lite，标点透传
- **v2.0** - 分层策略 (P2H + SLM)
- **v1.0** - 初版

## License

MIT
