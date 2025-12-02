# PinHan 拼音输入法引擎

轻量级智能拼音输入法引擎，采用 **词典召回 + SLM 重排序** 架构，专为嵌入式/MCU 设备优化。

## 特性

- **精简架构**: 词典为主，SLM 辅助重排序（仅在有上下文时使用）
- **低延迟**: 平均 10-15ms，缓存命中 < 1ms
- **模糊纠错**: 支持常见拼音错误（如 `shengme` → 什么，`zh/z` 模糊音）
- **上下文感知**: SLM 利用上下文消歧（如 `我做` + `shi` → 事）
- **标点透传**: 拼音中可直接输入标点（如 `nihao!!haha` → 你好!!哈哈）
- **轻量模型**: SLM Lite 仅 ~1M 参数，适合嵌入式部署

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载数据源

需要手动下载以下数据源到 `data/sources/` 目录：

| 数据源 | 说明 | 下载地址 |
|--------|------|----------|
| SUBTLEX-CH | 电影字幕词频 (核心) | [官网下载](http://crr.ugent.be/programs-data/subtitle-frequencies/subtlex-ch) |
| CC-CEDICT | 拼音映射 | [下载 cedict.txt.gz](https://www.mdbg.net/chinese/dictionary?page=cc-cedict) |

下载后放置:
```
data/sources/
├── SUBTLEX-CH/
│   └── SUBTLEX-CH-WF          # 词频文件
└── cedict.txt.gz               # 拼音映射
```

### 3. 构建基础词典

```bash
python scripts/build_dict.py
```

这会生成:
- `data/dicts/word_dict.json` - 词组字典
- `data/dicts/char_dict.json` - 单字字典
- `data/dicts/word_freq.json` - 词频表
- `data/dicts/char_freq.json` - 字频表

### 4. 下载扩展词库 (可选但推荐)

自动下载并整合第三方词库（短语、人名等）：

```bash
python scripts/download_vocab.py
```

这会下载:
- 短语拼音词库 (~9MB) - [mozillazg/phrase-pinyin-data](https://github.com/mozillazg/phrase-pinyin-data)
- 中文人名语料库 (~12MB) - [wainshine/Chinese-Names-Corpus](https://github.com/wainshine/Chinese-Names-Corpus)

### 5. 下载/训练 SLM 模型

**方式 A: 使用预训练模型 (推荐)**

从 Release 页面下载预训练的量化模型，放到 `model/slm_lite/` 目录：

```
model/slm_lite/
├── best.pt      # 最佳模型
├── last.pt      # 最新模型
└── vocab.json   # 词表
```

**方式 B: 自己训练**

1. 下载维基百科语料 ([zhwiki dump](https://dumps.wikimedia.org/zhwiki/latest/))
2. 生成训练数据:
   ```bash
   python scripts/build_training_data.py --xml zhwiki-latest-pages-articles.xml --max-samples 50000
   ```
3. 训练模型:
   ```bash
   python slm/train_lite.py --epochs 20 --batch-size 128
   ```

### 6. 启动 API 服务

```bash
python -m api.server
```

- API 文档: http://localhost:8000/docs
- 健康检查: http://localhost:8000/health

## 快速初始化脚本

如果你已准备好数据源，可以用以下命令一键初始化：

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 构建词典
python scripts/build_dict.py

# 3. 下载扩展词库
python scripts/download_vocab.py

# 4. 启动服务
python -m api.server
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

## 目录结构

```
PinHan/
├── api/                   # API 服务
│   └── server.py
├── data/                  # 数据目录
│   ├── dicts/             # 生成的字典文件
│   ├── patches/           # 高优先级词频修正
│   ├── sources/           # 外部数据源
│   │   ├── SUBTLEX-CH/    # 词频数据
│   │   ├── cedict.txt.gz  # 拼音映射
│   │   ├── phrase_pinyin.txt    # (自动下载)
│   │   └── chinese_names.txt    # (自动下载)
│   └── training/          # 训练数据
├── engine/                # 核心引擎
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
│   ├── build_dict.py      # 构建词典
│   ├── build_training_data.py  # 生成训练数据
│   ├── download_vocab.py  # 下载扩展词库
│   └── wiki_parser.py     # Wiki 解析器
├── slm/                   # 语言模型
│   ├── model.py           # 模型定义
│   └── train_lite.py      # 训练脚本
├── model/                 # 模型文件
│   └── slm_lite/          # SLM Lite 模型
├── logs/                  # 日志输出
├── tests/                 # 测试
│   ├── config.py          # 测试配置
│   ├── datasets.py        # 测试数据集
│   └── test_comprehensive.py  # 综合测试
└── requirements.txt
```

## 性能指标

| 场景 | 延迟 | 说明 |
|------|------|------|
| 缓存命中 | < 1ms | LRU 缓存 2000 条 |
| 无上下文 | 5-10ms | 纯词典查询 |
| 有上下文 | 10-20ms | 词典 + SLM 重排 |
| 段落连续输入 | ~10ms/词 | Top-1 93%+, Top-3 100% |

## API 接口

### POST /ime

主接口，返回完整信息。

```bash
curl -X POST http://localhost:8000/ime \
  -H "Content-Type: application/json" \
  -d '{"pinyin": "shi", "context": "我做", "top_k": 5}'
```

响应:
```json
{
  "raw_pinyin": "shi",
  "segmented_pinyin": ["shi"],
  "candidates": [
    {"text": "事", "score": 0.95, "source": "dict"},
    {"text": "是", "score": 0.82, "source": "dict"}
  ],
  "metadata": {"elapsed_ms": 12.5, "cached": false}
}
```

### GET /ime/simple

简单接口，只返回候选文本。

```bash
curl "http://localhost:8000/ime/simple?pinyin=nihao&top_k=5"
```

### GET /stats

获取引擎统计信息（缓存命中率、SLM 调用率等）。

## 测试

```bash
# 运行综合测试
python tests/test_comprehensive.py

# 冒烟测试 (快速)
python tests/test_comprehensive.py --level smoke

# 完整测试 + 保存报告
python tests/test_comprehensive.py --level full --save
```

## Python 调用示例

```python
from engine import create_engine_v3

engine = create_engine_v3()

# 简单查询
result = engine.process("nihao")
print([c.text for c in result.candidates])  # ['你好', '拟好', ...]

# 带上下文
result = engine.process("shi", context="我做")
print(result.candidates[0].text)  # '事'
```

## 日志系统

日志自动输出到 `logs/` 目录：

| 日志文件 | 内容 |
|----------|------|
| `pinhan.api.log` | API 请求/响应日志 |
| `pinhan.engine.log` | 引擎处理日志 |
| `pinhan.train.log` | 模型训练日志 |

## 技术栈

- **Python 3.11** + PyTorch 2.x + CUDA
- **FastAPI** + Uvicorn（API 服务）
- **Transformer Decoder**（SLM 语言模型）
- **SUBTLEX-CH** + **CC-CEDICT**（词频和拼音映射）
- **orjson**（高性能 JSON）

## License

MIT
