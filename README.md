# PinHan - 拼音→汉字转换模型

基于 PyTorch Seq2Seq Transformer 的拼音到汉字转换。

## 快速开始

### 1️⃣ 环境准备

```bash
.\.venv\Scripts\Activate.ps1        # Windows 激活虚拟环境
pip install -r requirements.txt      # 安装依赖
```

### 2️⃣ 数据处理

从 `wiki_latest.jsonl` (6GB) 提取数据：

```bash
# 7k 样本（快速）
python preprocess/sample_data.py -i data/wiki_latest.jsonl -o data/wiki_7k.jsonl -c 7000

# 50k 样本（推荐）
python preprocess/sample_data.py -i data/wiki_latest.jsonl -o data/wiki_50k.jsonl -c 50000 --random
```

### 3️⃣ 训练模型

```bash
# 基础训练 (7k 数据)
python model/train.py --data data/wiki_7k.jsonl --epochs 50

# 推荐配置 (50k 数据)
python model/train.py --data data/wiki_50k.jsonl --epochs 100 --batch-size 32
```

### 4️⃣ 推理

```bash
python model/infer.py --model outputs/best_model.pt --pinyin "zhong1 guo2"
```

---

## 脚本用法

### sample_data.py - 数据提取

从大文件快速提取样本。

```bash
python preprocess/sample_data.py \
  -i data/wiki_latest.jsonl \        # 输入文件
  -o data/wiki_50k.jsonl \           # 输出文件
  -c 50000 \                         # 样本数
  --random                            # 随机采样
```

**常用参数：**
- `--random` : 随机采样（推荐大文件）
- `--stratified` : 分层采样（优先高频汉字）
- `-c` : 样本数量

### train.py - 模型训练

```bash
python model/train.py \
  --data data/wiki_50k.jsonl \       # 训练数据
  --epochs 100 \                     # 训练轮数
  --batch-size 32 \                  # 批次大小
  --learning-rate 0.001              # 学习率
```

**Epochs 建议：**
| 数据量 | Epochs | 准确率 |
|--------|--------|--------|
| 7k | 40-60 | 20-30% |
| 50k | 80-150 | 60-70% ⭐ |
| 100k | 100-200 | 75-85% |

### infer.py - 推理

```bash
# 贪心解码
python model/infer.py --model outputs/best_model.pt --pinyin "zhong1 guo2"

# 束搜索（质量更好）
python model/infer.py --model outputs/best_model.pt --pinyin "zhong1 guo2" --beam-size 3
```

---

## 项目结构

```
PinHan/
├── model/                   # 模型代码
│   ├── train.py            # 训练
│   ├── infer.py            # 推理
│   └── core/               # 核心模块
├── preprocess/             # 数据处理
│   └── sample_data.py      # 数据提取 ⭐
├── data/                   # 数据集
│   ├── wiki_latest.jsonl   # 完整数据 (6GB)
│   ├── wiki_7k.jsonl       # 7k 样本
│   └── wiki_50k.jsonl      # 50k 样本
├── outputs/                # 模型输出
└── tests/                  # 单元测试 (32 个)
```

---

## 数据格式

JSONL 格式，每行一个对象：

```json
{"pinyin": "ni3 hao3", "hanzi": "你好"}
{"pinyin": "zhong1 guo2", "hanzi": "中国"}
```

注意：拼音包含声调数字 (1,2,3,4) 或 0 表示轻声。

---

## 训练指南

### 选择数据量和 Epochs

目标：参数/样本比 = 10-50:1

模型参数：5.4M

**推荐配置：**

| 数据量 | 参数比 | Epochs | 准确率 | CPU | GPU |
|--------|--------|--------|--------|------|------|
| 7k | 771:1 | 40-60 | 20% | 30min | 1min |
| 50k | 108:1 | 80-150 | 70% | 300min | 15min |
| 100k | 54:1 | 100-200 | 85% | 600min | 30min |

**建议：50k 数据 + 100 epochs 最高效** ⭐

### 监控训练

输出位置：`outputs/validation_model/`
- `best_model.pt` - 最佳模型
- `logs/training_summary.json` - 训练指标

---

## 常见问题

### 如何生成数据？

```bash
python preprocess/sample_data.py -i data/wiki_latest.jsonl -o data/wiki_50k.jsonl -c 50000 --random
```

### 怎样设置 Epochs？

根据数据量：7k → 40-60，50k → 80-150，100k+ → 100-200。

监控验证损失，10-20 个 epochs 不下降时停止。

### 支持 GPU 吗？

是的，自动检测并使用（快 10-50 倍）。

### 如何使用自己的数据？

准备 JSONL 格式：
```json
{"pinyin": "...", "hanzi": "..."}
```

然后训练：
```bash
python model/train.py --data your_data.jsonl --epochs 100
```

---

## 模型信息

| 指标 | 值 |
|-----|-----|
| 参数 | 5,400,000 |
| 词表 | ~8,000 |
| 速度 | 400-600ms (CPU) / 50-100ms (GPU) |
| 内存 | ~2GB (训练) / 500MB (推理) |

---

## 测试

```bash
pytest tests/ -v                    # 全部测试
pytest tests/test_units.py -v       # 单元测试
pytest tests/test_integration.py -v # 集成测试
```

状态：**32 个测试全部通过** ✅

---

## 许可证

MIT License

