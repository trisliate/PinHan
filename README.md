# PinHan - 轻量级智能拼音输入法引擎

![GitHub license](https://img.shields.io/github/license/trisliate/pinhan) 
![Python Version](https://img.shields.io/badge/python-3.9+-blue) 
![Status](https://img.shields.io/badge/status-stable-green)

> **纯词典架构**的拼音输入法，无深度学习，专为嵌入式/MCU设备和轻量化部署优化

## ✨ 核心特性

- **纯词典设计**：无 PyTorch、无 ONNX、无神经网络，仅依赖高质量词表
- **低延迟响应**：毫秒级响应时间（缓存命中 <1ms），支持实时输入
- **轻量级部署**：Docker 镜像 < 200MB，支持 ARM/MCU 嵌入式设备
- **灵活的词库融合**：支持多来源词表优先级设置（SUBTLEX-CH、jieba、自定义扩展）
- **模糊音纠错**：支持声母/韵母模糊音、键盘相邻键纠错、编辑距离纠错
- **智能拼音切分**：自动处理连续拼音切分歧义（"xian" → "西安" vs "先"）
- **标点符号透传**：支持中英文标点直通输出
- **多接口支持**：RESTful API + Python 库调用 + 命令行工具

---

## 🏗️ 架构设计

### 纯词典架构（为什么移除了SLM？）

PinHan v3 放弃了复杂的序列级模型（SLM），采用**纯词典 + 规则**的设计：

```
输入 "nihao" 
  │
  ├─ [拼音切分] 连续拼音 → 单个拼音
  │    nihao → [ni, hao]
  │
  ├─ [拼音纠错] 验证/纠正不合法拼音
  │    [ni, hao] → [ni, hao] ✓
  │
  ├─ [词典召回] 查询字典生成候选
  │    词组: "你好" (freq=999)
  │    单字: "你" (freq=900) + "好" (freq=850)
  │
  ├─ [Beam Search] 动态规划排序
  │    Score = Freq + RankBonus + LengthBonus
  │
  └─ [输出] Top-K 候选
       ["你好", "你号", "泥好", ...]
```

**为什么是纯词典而不是SLM？**

| 特性 | 纯词典 | SLM |
|------|--------|-----|
| 可解释性 | ✅ 每个候选都有明确来源 | ❌ 黑盒，难以调试 |
| 训练成本 | ✅ 无，直接使用高质量词表 | ❌ 需要大规模语料库+GPU |
| 部署灵活 | ✅ 任何设备（包括MCU） | ❌ 需要足够内存和计算力 |
| 可定制性 | ✅ 添加热词即时生效 | ❌ 需要重训练 |
| 性能稳定 | ✅ 无OOM、无模型异常 | ❌ 易出现模型问题 |
| 实时更新 | ✅ 更新词库快速生效 | ❌ 需要重新部署 |

### 词库融合设计（三层优先级）

采用**三层优先级融合**策略，不同来源的词表通过权重覆盖：

| 优先级 | 来源 | 权重 | 特点 |
|------|------|------|------|
| 🔴 高 | **SUBTLEX-CH** | 50 | 电影/电视剧字幕，口语频率最真实 |
| 🟡 中 | **自定义扩展** | 40 | 热词、品牌词、行业术语、方言词 |
| 🟢 低 | **第三方词库** | 30 | jieba、pkuseg、THUOCL 等通用词表 |
| ⚪ 基础 | **CC-CEDICT** | 10 | 拼音映射和冷启动备用 |

**融合示例**：
```
权重 50: 你好:999   (SUBTLEX-CH)
权重 40: 你好:888   (extensions) ← 被上层覆盖，不使用
权重 30: 你好:500   (sources)    ← 被上层覆盖，不使用
权重 10: 你好:100   (CEDICT)     ← 被上层覆盖，不使用
→ 最终使用: 你好:999 (SUBTLEX-CH)
```

---

## 📦 项目结构

```
PinHan/
├── data/                          # 🔑 所有数据处理和存储
│   ├── dicts/                     # ✅ 编译后的词典（JSON格式）
│   │   ├── char_dict.json         # 拼音→字 映射
│   │   ├── word_dict.json         # 拼音→词 映射
│   │   ├── char_freq.json         # 字→频率
│   │   ├── word_freq.json         # 词→频率
│   │   └── pinyin_table.txt       # 合法拼音表
│   ├── extensions/                # 自定义热词/扩展词库
│   │   ├── README.md
│   │   └── hotwords.txt           # 示例：品牌词、热词等
│   └── sources/                   # 第三方词库源
│       ├── SUBTLEX-CH/            # 电影字幕词表（内置）
│       ├── cedict.txt.gz          # CC-CEDICT拼音字典
│       └── （其他第三方词库）
│
├── scripts/
│   └── build_dict.py              # 词库构建脚本（融合多源→data/dicts/）
│
├── pinhan/                         # Python 包
│   ├── engine/                    # 核心引擎模块
│   │   ├── core.py                # IME 主引擎
│   │   ├── dictionary.py          # 词典查询服务
│   │   ├── corrector.py           # 拼音纠错器
│   │   ├── segmenter.py           # 拼音切分器
│   │   ├── generator.py           # 候选生成器
│   │   ├── cache.py               # LRU 缓存
│   │   ├── config.py              # 配置类
│   │   ├── logging.py             # 日志
│   │   └── __init__.py
│   ├── api/                       # FastAPI 服务
│   │   ├── server.py
│   │   └── __init__.py
│   ├── cli.py                     # 命令行工具
│   └── __init__.py
│
├── pyproject.toml                 # Python 包配置
├── Dockerfile                     # Docker 构建
├── README.md
└── .gitignore
```

**关键点**：
- 👉 词典数据存储在**根目录 `data/dicts/`** （不在包内）
- 📝 扩展词库在 **`data/extensions/`**（自动扫描）
- 🔄 构建脚本自动输出到 `data/dicts/`
- 📦 PyPI 安装时自动包含 `data/dicts/` 中的词典

---

## 🚀 快速开始

### 方式 1：Docker（推荐）

```bash
# 拉取镜像
docker pull ghcr.io/trisliate/pinhan:latest

# 运行容器
docker run -d -p 3000:3000 ghcr.io/trisliate/pinhan:latest

# 测试 API
curl -X POST http://localhost:3000/ime \
  -H "Content-Type: application/json" \
  -d '{"pinyin": "nihao", "context": "", "top_k": 10}'

# 查看 API 文档
open http://localhost:3000/docs
```

### 方式 2：本地源码（开发）

```bash
# 克隆仓库
git clone https://github.com/trisliate/pinhan.git
cd pinhan

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装开发依赖
pip install -e ".[dev]"

# 重建词典（可选）
python scripts/build_dict.py

# 运行 API 服务
pinhan-server --host 0.0.0.0 --port 3000
```

### 方式 3：Python 库

```bash
pip install pinhan
```

```python
from pinhan import IMEEngineV3, EngineConfig

# 初始化引擎
config = EngineConfig(top_k=10, use_fuzzy=True, cache_size=2000)
engine = IMEEngineV3(config)

# 处理拼音输入
result = engine.process("nihao", context="")

# 访问候选结果
for i, candidate in enumerate(result.candidates[:5]):
    print(f"{i+1}. {candidate.text}: {candidate.score:.3f} ({candidate.source})")

# 输出：
# 1. 你好: 0.999 (dict)
# 2. 你号: 0.342 (beam)
# 3. 泥好: 0.098 (char)
```

### 方式 4：命令行工具

```bash
# 查看版本
pinhan version

# 简单查询
pinhan query nihao

# 指定上下文和候选数
pinhan query shi -c "我做" -k 10

# 启动 API 服务
pinhan server --host 0.0.0.0 --port 3000 --log-level info
```

---

## 🔌 REST API 接口

### POST /ime - 主接口

完整的拼音处理接口，返回分词结果、候选和元数据。

```bash
curl -X POST http://localhost:3000/ime \
  -H "Content-Type: application/json" \
  -d '{
    "pinyin": "nihao",
    "context": "你好",
    "top_k": 10
  }'
```

**请求体**：
```json
{
  "pinyin": "nihao",           // 必填：拼音输入
  "context": "你好",            // 可选：上文（已确认的文本）
  "top_k": 10                  // 可选：返回候选数量 (1-50, 默认10)
}
```

**响应**：
```json
{
  "raw_pinyin": "nihao",
  "segmented_pinyin": ["ni", "hao"],
  "candidates": [
    {
      "text": "你好",
      "score": 0.999,
      "source": "dict"
    },
    {
      "text": "你号",
      "score": 0.342,
      "source": "beam"
    },
    {
      "text": "泥好",
      "score": 0.098,
      "source": "char"
    }
  ],
  "metadata": {
    "elapsed_ms": 1.23,
    "cache_rate": 0.85
  }
}
```

### GET /ime/simple - 快速查询

仅返回候选文本列表，适合简单场景。

```bash
curl "http://localhost:3000/ime/simple?pinyin=nihao&top_k=5"
```

**响应**：
```json
{
  "pinyin": "nihao",
  "candidates": ["你好", "你号", "泥好", "呢好", "你郝"]
}
```

### GET /health - 健康检查

```bash
curl http://localhost:3000/health
```

### GET /stats - 引擎统计

```bash
curl http://localhost:3000/stats
```

---

## 📚 词库定制指南

### 添加热词

编辑 **`data/extensions/hotwords.txt`**（在项目根目录，不是包内），格式为 `词语<空格>频率`：

```text
你好 100
拜拜 80
谢谢 90
确定 85
开始 75
```

然后重建词典：

```bash
python scripts/build_dict.py
```

词典会自动更新到 `data/dicts/` 并即时生效。

### 集成第三方词库

**步骤**：

1. **获取词库源文件**（如 THUOCL、jieba、SogouDict）
2. **转换为标准格式**：`词语<空格>频率`（一行一条）
3. **放入** `data/sources/` 目录（任意子目录）
4. **运行** `python scripts/build_dict.py` 自动融合

**示例**：

```bash
# 添加 jieba 词库
mkdir -p data/sources/jieba
cp jieba_dict.txt data/sources/jieba/

# 添加 THUOCL 词库
mkdir -p data/sources/thuocl
cp thuocl_*.txt data/sources/thuocl/

# 重建词典（自动扫描所有源）
python scripts/build_dict.py

# 输出日志：
# ✓ 加载 SUBTLEX-CH: 10000 条词
# ✓ 加载 extensions/: 100 条词  
# ✓ 加载 sources/: 15000 条词
# ✓ 融合优先级...
# ✓ 生成 JSON 词典...
# ✓ 完成！词典已保存: data/dicts/
```

### 推荐的第三方词库

| 词库 | 来源 | 链接 | 特点 |
|------|------|------|------|
| **SUBTLEX-CH** | 电影/电视字幕 | http://subtlex.org/ | 口语频率最真实（已内置）|
| **THUOCL** | 清华大学 | https://github.com/thunlp/THUOCL | 30个专业领域 |
| **jieba** | 自然语言处理 | https://github.com/fxsjy/jieba | 通用高频词 |
| **pkuseg** | 北京大学 | https://github.com/lancopku/pkuseg-python | 跨领域分词 |
| **SogouDict** | 搜狗 | https://pinyin.sogou.com/dict | 互联网热词 |

### 词库优先级说明

融合时采用**权重覆盖**策略（高优先级覆盖低优先级）：

```
权重 50: 你好:999   (SUBTLEX-CH) ✓ 采用
权重 40: 你好:888   (extensions) ← 被上层覆盖
权重 30: 你好:500   (sources)    ← 被上层覆盖
权重 10: 你好:100   (CEDICT)     ← 被上层覆盖

最终结果: 你好:999 (来自 SUBTLEX-CH)
```

通过提高 `data/extensions/` 中的频率可以优先选择热词：

```text
# data/extensions/hotwords.txt
新产品 200     # 权重40，可能会覆盖源中较低的词频
明星名字 180
```

## ⚙️ 配置说明

### Python 库配置

```python
from pinhan import EngineConfig, IMEEngineV3

config = EngineConfig(
    top_k=10,           # 返回候选数量 (默认10)
    use_fuzzy=True,     # 启用模糊音 (默认True)
    cache_size=2000,    # LRU 缓存大小 (默认2000)
)

engine = IMEEngineV3(config)
```

### API 服务配置

通过环境变量配置：

```bash
export HOST=0.0.0.0
export PORT=3000
export LOG_LEVEL=info

pinhan-server
```

---

## 📊 性能指标

| 场景 | 响应时间 | 缓存命中率 |
|-----|---------|---------|
| 缓存命中 | < 0.5ms | - |
| 单字拼音 | 0.5-2ms | 99% |
| 词组拼音 | 2-5ms | 95% |
| 冷启动 | 5-10ms | 0% |
| 连续输入 | 0.3-1ms | 98% |

---

## 🔧 故障排查

### 词库相关问题

```bash
# 重建词库
python scripts/build_dict.py

# 验证文件
ls -la pinhan/data/dicts/
```

### API 启动问题

```bash
# 检查依赖
pip list | grep -E "fastapi|uvicorn"

# 查看详细日志
LOG_LEVEL=debug pinhan-server
```

---

## 🤝 贡献

欢迎 Issue 和 Pull Request！

---

## 📄 许可证

MIT License

---

**最后更新**：2025-12-03 | **版本**：0.1.0
