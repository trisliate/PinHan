# PinHan — 拼音到汉字 Seq2Seq Transformer

**一句话：** 基于 PyTorch Seq2Seq Transformer 的拼音→汉字转换模型，支持完整训练、推理、检查点管理。

---

## � 文档导航

根据你的需求选择对应文档：

| 我想... | 查看文档 | 时间 |
|--------|--------|------|
| 🚀 立即运行代码 | [QUICK_START.md](QUICK_START.md) | 3分钟 |
| 📚 了解所有参数和工作流 | [REMADE.md](REMADE.md) | 15分钟 |
| 🔧 优化性能或调试 | [TECH_REPORT.md](TECH_REPORT.md) | 20分钟 |

---

## ⚡ 30 秒快速开始

```bash
# 激活环境
.\.venv\Scripts\Activate.ps1

# 训练（5k 数据，40 epochs）
python model/train_pinhan.py --data data/5k.jsonl --epochs 40 --batch-size 32 --save-dir outputs/5k_model

# 推理
python model/infer_pinhan.py --model outputs/5k_model/best_model.pt --pinyin "zhong1 guo2 ren2"
# 输出: 中国人
```

---

## 📊 当前状态

| 指标 | 值 |
|------|-----|
| Best Loss | 0.2027 (epoch 37) |
| 训练时间 | 1414.5s (40 epochs, CPU) |
| 推理速度 | 400-600ms (贪心解码) |
| 模型大小 | ~5.4M 参数 |
| 磁盘占用 | 20MB (100 epochs，优化后) |

---

## ✨ 核心特性

✅ **开箱即用** — 完整的数据加载、训练、推理工作流

✅ **智能检查点** — 自动清理旧检查点，节省 96% 磁盘空间

✅ **代码标准化** — 全部使用 orjson，二进制 I/O 优化

✅ **CPU 优化** — 8 核 CPU 上 40 epochs 仅需 23.5 分钟

✅ **完整命令行接口** — 所有参数都支持自定义

---

## � 项目结构简览

```
PinHan/
├── model/                           ⭐ 核心模型与训练代码
│   ├── seq2seq_transformer.py       模型架构定义（Transformer encoder-decoder）
│   ├── train_pinhan.py              ⭐ 主训练脚本（完整训练流程）
│   ├── train_utils.py               训练辅助工具（设备选择、模型保存等）
│   ├── train_improvements.py        模型改进工具（最优模型追踪、检查点）
│   ├── checkpoint_manager.py        ⭐ 检查点管理器（智能保存、恢复训练）
│   ├── infer_pinhan.py              ⭐ 推理脚本（模型预测）✅ 已修复命令行参数
│   ├── evaluate.py                  评估脚本（模型性能评估）
│   └── __pycache__/                 Python 缓存目录（可忽略）
│
├── preprocess/                      数据预处理工具
│   ├── pinyin_utils.py              拼音处理工具（标准化、验证）
│   ├── extract_and_clean.py         数据提取与清洗
│   ├── split_dataset.py             数据集分割
│   ├── split_shards.py              大型数据集分片处理
│   ├── __init__.py                  包初始化文件
│   └── __pycache__/                 Python 缓存目录（可忽略）
│
├── tests/                           ⭐ 测试套件
│   ├── run_tests.py                 ⭐ 主测试运行器（26项测试）
│   ├── test_units.py                单元测试（17项）
│   ├── test_integration.py          集成测试（4项）
│   ├── test_performance.py          性能测试（5项）
│   ├── test_report.json             最近的测试报告
│   ├── TEST_GUIDE.md                测试指南
│   └── __pycache__/                 Python 缓存目录（可忽略）
│
├── data/                            📊 训练数据
│   ├── 10k.jsonl                    10,000 条数据集（完整）
│   ├── 5k.jsonl                     ✅ 5,000 条数据集（当前训练结果）
│   └── test_mini.jsonl              迷你数据集（100 条，快速验证）
│
├── outputs/                         📁 训练输出目录
│   ├── 5k_model/                    ⭐ 5k 数据集训练结果（当前生产模型）
│   │   ├── best_model.pt            最优模型权重（推荐用于推理）
│   │   ├── model.pt                 完整模型（含元数据）
│   │   ├── checkpoint_epoch37.pt    epoch 37 检查点（最优 epoch）
│   │   ├── checkpoint_epoch38.pt    epoch 38 检查点
│   │   ├── checkpoint_epoch40.pt    最新检查点
│   │   ├── src_vocab.json           源语言词表（拼音）
│   │   ├── tgt_vocab.json           目标语言词表（汉字）
│   │   └── logs/
│   │       ├── config.json          训练配置快照
│   │       ├── training_summary.json ⭐ 详细训练日志（loss 曲线、统计）
│   │       └── metrics.json         训练指标
│   └── pinhan_model/                其他训练结果（如有）
│
├── README.md                        📖 项目概览与快速开始
├── REMADE.md                        📋 本文件（完整使用指南）
├── QUICK_REFERENCE.md               ⚡ 命令速查表
├── PROJECT_STRUCTURE.md             📁 详细项目结构说明
├── EXECUTE_NOW.md                   🚀 执行清单
├── requirements.txt                 📦 Python 依赖列表
├── gpu_fix.py                       🔧 GPU 诊断脚本
└── wiki_latest.jsonl / zhwiki...   原始数据文件（可选）
```

---

## 🎯 下一步优先级

| 优先级 | 任务 | 预期效果 |
|--------|------|--------|
| � 高 | 数据扩展（5k → 100k+） | 准确率 +20-30% |
| 🟡 中 | 模型优化（更深/更宽） | 性能 +10-15% |
| 🟢 低 | 多音字消歧（上下文） | 多音字准确率 +15% |

---

## 💡 关键参数

**训练：** `--data` (必需), `--epochs` (50), `--batch-size` (32), `--lr` (1e-4), `--resume`

**推理：** `--model` (必需), `--pinyin` (必需), `--beam-size` (1), `--device` (auto)

**⚠️ 重要：** 拼音必须包含声调数字（1-4），如 `ni3 hao3` 而非 `ni hao`

---

## 📞 快速导航

- **代码出错？** → 查看 [REMADE.md](REMADE.md) 的故障排除部分
- **不知道怎么用？** → 查看 [QUICK_START.md](QUICK_START.md)
- **想优化性能？** → 查看 [TECH_REPORT.md](TECH_REPORT.md)
- **查看训练日志？** → `outputs/5k_model/logs/training_summary.json`

---

**最后更新：** 2025-11-01 | **版本：** 1.1

👉 **从这里开始：** [QUICK_START.md](QUICK_START.md) (3分钟快速上手)

**作用：** 完整的训练管道，包括数据加载、模型初始化、优化循环、检查点管理和日志记录。这是项目的核心训练入口。

**使用方法：**
```bash
python model/train_pinhan.py [OPTIONS]
```

**完整参数列表：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--data` | str | 必需 | 训练数据路径 (JSONL 格式)。例：`data/5k.jsonl` |
| `--save-dir` | str | `outputs/default` | 输出目录，保存模型、检查点和日志 |
| `--epochs` | int | 50 | 训练轮数 |
| `--batch-size` | int | 32 | 每批数据量 |
| `--lr` | float | 1e-4 | 学习率（Adam 优化器） |
| `--warmup-steps` | int | 0 | 预热步数（学习率预热）|
| `--weight-decay` | float | 0.0 | L2 正则化系数 |
| `--resume` | flag | 无 | 从最近检查点恢复训练（不指定 epoch） |
| `--seed` | int | 42 | 随机种子（可复现性） |
| `--log-every` | int | 10 | 每 N 个批次打印一次日志 |
| `--save-every` | int | 1 | 每 N 个 epoch 保存一次检查点 |

**使用示例：**

```bash
# 1. 基础训练（最简单）
python model/train_pinhan.py --data data/5k.jsonl --save-dir outputs/my_run

# 2. 完整参数示例（推荐）
python model/train_pinhan.py \
    --data data/5k.jsonl \
    --save-dir outputs/5k_v1 \
    --epochs 40 \
    --batch-size 32 \
    --lr 1e-4 \
    --seed 42

# 3. 小规模快速测试（验证代码）
python model/train_pinhan.py \
    --data data/test_mini.jsonl \
    --epochs 2 \
    --batch-size 16 \
    --save-dir outputs/debug

# 4. 恢复中断的训练（从最后检查点）
python model/train_pinhan.py \
    --data data/5k.jsonl \
    --epochs 100 \
    --save-dir outputs/5k_v1 \
    --resume

# 5. 长期训练（大数据集）
python model/train_pinhan.py \
    --data data/10k.jsonl \
    --epochs 100 \
    --batch-size 64 \
    --lr 5e-5 \
    --save-dir outputs/10k_model
```

**数据格式 (JSONL)：**

每行为一个 JSON 对象，包含 `pinyin` 和 `hanzi` 字段：

```json
{"pinyin": "ni3 hao3", "hanzi": "你好"}
{"pinyin": "xie4 xie4", "hanzi": "谢谢"}
{"pinyin": "zhong1 guo2 ren2", "hanzi": "中国人"}
```

**拼音格式说明：**
- 空格分隔多个字的拼音
- 必须包含声调数字 (1-4)
- 格式：`pinyin1 pinyin2 pinyin3` (例：`zhong1 guo2 ren2`)
- ❌ 错误格式：`zhong guo ren`（缺少声调）
- ✅ 正确格式：`zhong1 guo2 ren2`

**训练日志示例：**

```
2025-11-01 22:07:41,130 INFO: Epoch 3 [  46/157] loss=5.0672 grad_norm=3.0573 time=0.18s ETA=22s
```

**日志字段解释：**

| 字段 | 例值 | 说明 |
|------|------|------|
| `Epoch N` | 3 | 当前轮数 |
| `[M/Total]` | [46/157] | 当前批 / 该 epoch 总批数 |
| `loss` | 5.0672 | 当前批的交叉熵损失值 |
| `grad_norm` | 3.0573 | 模型梯度的全局 L2 范数 |
| `time` | 0.18s | 处理该批耗时（秒） |
| `ETA` | 22s | 完成当前 epoch 的估计剩余时间 |

**关键指标解释：**

- **grad_norm**：梯度范数，反映梯度大小
  - 健康范围：1.0–5.0（根据模型而异）
  - 过大（>10）：梯度爆炸 → 减小学习率或启用梯度裁剪
  - 过小（~0）：梯度消失 → 增大学习率或检查反向传播

- **ETA**：估计完成时间，基于最近批次的平均速度计算
  - 用途：监控训练进度；如果 ETA 突然增大，可能有 I/O 瓶颈或系统争用

- **loss**：交叉熵损失，越小越好
  - 初期快速下降（epoch 1–20）
  - 中期缓慢改善（epoch 20–40）
  - 后期趋于平台（可考虑早停）

**设备选择：**

脚本自动选择最佳设备：
```
1. 检测 NVIDIA CUDA → 使用 GPU
2. 无 CUDA → 使用 CPU（自动设置线程数为系统核心数）
```

对于 AMD Ryzen 9 6900H（无独立显卡）：使用 CPU 训练（8 核，≈32.5 秒/epoch）

---

### ⭐ **2. model/infer_pinhan.py** — 推理脚本

**作用：** 使用训练好的模型进行在线拼音→汉字转换预测。

**使用方法：**
```bash
python model/infer_pinhan.py [OPTIONS]
```

**参数列表：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | str | 必需 | 模型权重路径。例：`outputs/5k_model/best_model.pt` |
| `--pinyin` | str | 必需 | 输入拼音序列（空格分隔，包含声调）。例：`"zhong1 guo2 ren2"` |
| `--beam-size` | int | 1 | 束搜索大小（1 = 贪心解码；>1 = 束搜索） |
| `--device` | str | `auto` | 设备选择：`cpu`、`cuda` 或 `auto`（自动选择） |

**使用示例：**

```bash
# 1. 基础推理（贪心解码）
python model/infer_pinhan.py \
    --model outputs/5k_model/best_model.pt \
    --pinyin "zhong1 guo2 ren2"
# 输出: 中国人

# 2. 其他例子
python model/infer_pinhan.py \
    --model outputs/5k_model/best_model.pt \
    --pinyin "zi4 you2 ren2"
# 输出: 自由人

# 3. 束搜索（返回多个候选）
python model/infer_pinhan.py \
    --model outputs/5k_model/best_model.pt \
    --pinyin "wo3 de5 ming2 zi4" \
    --beam-size 3

# 4. 指定设备（强制 CPU）
python model/infer_pinhan.py \
    --model outputs/5k_model/best_model.pt \
    --pinyin "ni3 hao3" \
    --device cpu

# 5. 在 GPU 上推理（如果可用）
python model/infer_pinhan.py \
    --model outputs/5k_model/best_model.pt \
    --pinyin "ni3 hao3" \
    --device cuda
```

**预期输出格式：**

```
加载模型和词表...
✅ 模型加载成功

输入拼音: zhong1 guo2 ren2
使用贪心解码...
预测汉字: 中国人
推理耗时: 390.50ms
```

**推理效果说明：**

当前模型（5k.jsonl, 40 epochs）的推理效果：
- ✅ **准确率较高**的字：常见词汇、高频字（如"中、国、人"）
- ⚠️ **可能有误**的字：低频字、多音字、有特殊语境的字
- 📈 **改进方向**：使用更大数据集（10k, 100k+）重新训练，可显著提高准确率和语境理解能力

**示例推理结果：**

| 拼音 | 预测汉字 | 备注 |
|------|---------|------|
| `zhong1 guo2 ren2` | 中国人 | ✅ 正确 |
| `zi4 you2 ren2` | 自由人 | ✅ 正确 |
| `ni3 hao3` | 拟好 | ⚠️ 应为"你好"（数据不足） |
| `hao3 peng2 you3` | 好彭有 | ⚠️ 应为"好朋友"（数据不足） |

---

### **3. model/checkpoint_manager.py** — 检查点管理器

**作用：** 智能地保存和管理训练检查点，仅保留最近 3 个 + 1 个最优模型，节省磁盘空间（96% 空间节省）。

**功能说明：**

- **自动清理**：仅保留最近 3 个检查点 + 1 个最优检查点，其余自动删除
- **磁盘优化**：100 epochs 仅占用 ~20 MB（vs. 480 MB 未优化）
- **恢复训练**：支持从任意检查点恢复
- **元数据保存**：自动保存训练配置和性能指标

---

### **4. model/train_utils.py** — 训练辅助工具

**作用：** 提供模型保存/加载、设备检测、词表管理等工具函数。由 `train_pinhan.py` 内部调用。

**核心函数：**

| 函数 | 用途 |
|------|------|
| `get_device()` | 智能设备选择（CUDA → CPU）；返回 torch.device 对象 |
| `save_model_complete(...)` | 保存模型及元数据（含元信息，便于版本管理） |
| `load_model_with_metadata(...)` | 加载模型及元数据 |

**（通常不直接使用，由 train_pinhan.py 内部调用）**

---

### **5. model/seq2seq_transformer.py** — 模型架构

**作用：** 定义 Seq2Seq Transformer 模型架构（编码器-解码器）。

**模型参数：**

| 参数 | 值 | 说明 |
|------|-----|------|
| `d_model` | 256 | 隐藏维度 |
| `nhead` | 4 | 注意力头数 |
| `num_encoder_layers` | 3 | 编码器层数 |
| `num_decoder_layers` | 3 | 解码器层数 |
| `dim_feedforward` | 1024 | FFN 隐藏维度 |
| `dropout` | 0.1 | dropout 率 |
| `总参数量` | ~5.4M | 可训练参数 |

**（通常不直接使用，由 train_pinhan.py 内部调用）**

---

### **6. tests/run_tests.py** — 测试运行器

**作用：** 运行全套测试（26 项）并生成测试报告。

**使用方法：**

```bash
python tests/run_tests.py
```

**输出示例：**

```
======================================================================
🧪 PinHan 测试套件
======================================================================
运行 26 项测试...

✅ test_vocab_creation .......................... [PASS]
✅ test_data_loading ........................... [PASS]

======================================================================
📊 测试总结
======================================================================
成功: 24
失败: 0
错误: 2
成功率: 92.3%

✅ 测试报告已保存到: tests/test_report.json
```

**测试包含：**
- 单元测试（17 项）：Vocab、数据加载、模型前向等
- 集成测试（4 项）：完整训练流程
- 性能测试（5 项）：训练速度、内存占用等

---

## 🚀 完整工作流示例

### **场景 1：从零开始快速验证（5 分钟）**

```bash
# 1. 激活环境
.\.venv\Scripts\Activate.ps1

# 2. 安装依赖
pip install -r requirements.txt

# 3. 快速训练（mini 数据集，2 epochs）
python model/train_pinhan.py \
    --data data/test_mini.jsonl \
    --epochs 2 \
    --batch-size 16 \
    --save-dir outputs/quick_test

# 4. 推理测试
python model/infer_pinhan.py \
    --model outputs/quick_test/best_model.pt \
    --pinyin "zhong1 guo2 ren2"

# 5. 运行测试
python tests/run_tests.py
```

---

### **场景 2：完整训练流程（30 分钟）**

```bash
# 1. 标准训练
python model/train_pinhan.py \
    --data data/5k.jsonl \
    --epochs 40 \
    --batch-size 32 \
    --lr 1e-4 \
    --save-dir outputs/5k_model

# 2. 检查训练结果
cat outputs/5k_model/logs/training_summary.json

# 3. 批量推理测试
python model/infer_pinhan.py --model outputs/5k_model/best_model.pt --pinyin "zhong1 guo2 ren2"
python model/infer_pinhan.py --model outputs/5k_model/best_model.pt --pinyin "zi4 you2 ren2"
python model/infer_pinhan.py --model outputs/5k_model/best_model.pt --pinyin "ni3 hao3"
```

---

### **场景 3：恢复中断的训练**

```bash
# 训练被中断（仅训练到 epoch 20）
# 恢复从 epoch 20 继续训练到 epoch 100
python model/train_pinhan.py \
    --data data/5k.jsonl \
    --epochs 100 \
    --save-dir outputs/5k_model \
    --resume
```

---

## 📊 数据格式详解

### **输入数据格式 (JSONL)**

每行为一个 JSON 对象，包含 `pinyin` 和 `hanzi` 字段：

```json
{"pinyin": "zhong1 guo2 ren2", "hanzi": "中国人"}
{"pinyin": "ni3 hao3", "hanzi": "你好"}
{"pinyin": "zi4 you2 ren2", "hanzi": "自由人"}
```

**重要说明：**
- 拼音必须用空格分隔
- 拼音必须包含声调数字 (1-4)，例如 `ni3`、`hao3`、`zhong1`
- ❌ 不能用 `ni hao`（缺少声调）
- ✅ 应使用 `ni3 hao3`

---

## 🛠️ 常见问题 & 故障排除

| 问题 | 症状 | 解决方案 |
|------|------|--------|
| 推理时找不到词表 | `错误：词表文件不存在` | 确保模型路径正确，词表文件在同级目录（`src_vocab.json`, `tgt_vocab.json`） |
| 推理拼音格式错误 | 模型加载成功但输出乱码 | 检查拼音格式：必须空格分隔且包含声调，如 `ni3 hao3` 而非 `ni hao` |
| 梯度爆炸 | `grad_norm` 很大 (>10)，loss 上升 | 减小学习率或启用梯度裁剪 |
| 训练很慢 | ETA 很大，CPU/GPU 利用率低 | 检查数据 I/O、减少日志频率或增大 batch size |
| 内存不足 (OOM) | 程序崩溃 | 减小 `--batch-size` 或 `--epochs` |
| 模型推理效果差 | 预测汉字不准确 | 数据不足（当前 5k 样本受限）→ 用更大数据集重新训练 |

---

## 📈 性能基准（参考）

### 训练性能 (5k.jsonl, 40 epochs)

| 指标 | 值 |
|------|-----|
| **Best Loss** | 0.2027 |
| **Best Epoch** | 37 |
| **Total Time** | 1414.5 s (~23.5 min) |
| **Time/Epoch** | ~35.4 s |
| **Device** | CPU (R9-6900H, 8 cores) |
| **Batch Size** | 32 |
| **Memory** | ~2 GB |

### 推理性能（贪心解码）

| 指标 | 值 |
|------|-----|
| **单句推理时间** | 380–600 ms |
| **设备** | CPU (R9-6900H) |
| **句长** | 3–4 字 |

### 模型大小

| 文件 | 大小 |
|------|------|
| `best_model.pt` | ~18 MB |
| 100 epochs 检查点 | ~20 MB (vs. 480 MB 未优化，节省 96%) |

---

## 📝 后续改进方向

### 1. **数据扩展** ⭐ 优先级最高
   - 当前数据集：5k 样本
   - 建议方向：10k–100k+ 样本（维基百科、字典数据）
   - 预期效果：准确率显著提升，多音字处理更好

### 2. **模型架构优化**
   - 增加层数/隐藏维度
   - 加入 attention 可视化
   - 支持批量推理

### 3. **训练优化**
   - 启用梯度裁剪
   - 学习率调度（warm-up + decay）
   - 集束搜索优化

### 4. **语境模型**
   - 加入上下文信息
   - 支持多音字消歧
   - 词级别而非字级别

---

## 📦 依赖说明

### requirements.txt 包含：

```
torch>=2.0              # PyTorch 核心库
transformers>=4.30      # NLP 工具（可选）
datasets                # 数据集工具
accelerate              # 分布式训练加速
sentencepiece           # 分词工具
numpy                   # 数值计算
safetensors             # 模型序列化
orjson                  # ⭐ 快速 JSON 序列化
pypinyin                # 拼音处理
bitarray                # 位数组（可选）
mmh3                    # 哈希函数（可选）
```

---

## ✅ 快速检查清单

- [ ] 已安装依赖：`pip install -r requirements.txt`
- [ ] 数据文件存在：`ls data/5k.jsonl`
- [ ] 首次运行时进行测试：`python tests/run_tests.py`
- [ ] 训练前备份重要数据
- [ ] 磁盘空间充足（100 epochs 需 ~20 MB）

---

**最后更新：** 2025-11-01

**版本：** 1.1 (推理脚本修复 + 推理示例)

**主要改动：**
- ✅ 修复 `infer_pinhan.py` 命令行参数支持
- ✅ 添加详细推理示例和预期输出
- ✅ 补充拼音格式说明和常见错误
- ✅ 添加后续改进方向与数据扩展建议
