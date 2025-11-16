# 3 分钟快速开始

**目标：** 在 3 分钟内运行训练或推理。

---

## 🔧 Step 1: 环境准备（1 分钟）

```bash
# 激活虚拟环境（Windows）
.\.venv\Scripts\Activate.ps1

# 安装依赖
pip install -r requirements.txt
```

**检查：** 运行 `python -c "import torch; print(torch.__version__)"`，应该看到版本号。

---

## 📊 Step 2: 准备数据（理解格式很关键！）

### 数据格式（JSONL）
每行一个 JSON 对象，包含 `pinyin` 和 `hanzi`：

```json
{"pinyin": "ni3 hao3", "hanzi": "你好"}
{"pinyin": "zhong1 guo2", "hanzi": "中国"}
{"pinyin": "zi4 you2", "hanzi": "自由"}
```

**⚠️ 重要：** 拼音必须包含声调数字（1,2,3,4）！

### 可用数据集
- `data/5k.jsonl` ✅ 当前推荐（5000 对）
- `data/10k.jsonl` 完整版（10000 对）
- `data/test_mini.jsonl` 测试（500 对）

---

## 🚂 Step 3: 训练模型（1 分钟命令）

### 最简命令（使用默认参数）
```bash
python model/train_pinhan.py --data data/5k.jsonl
```

### 完整命令（自定义参数）
```bash
python model/train_pinhan.py \
    --data data/5k.jsonl \
    --epochs 40 \
    --batch-size 32 \
    --lr 0.0001 \
    --save-dir outputs/5k_model
```

**预期输出：**
```
Epoch 1/40: loss=5.2345, lr=0.0001
Epoch 2/40: loss=4.8901, lr=0.0001
...
Epoch 40/40: loss=0.2027, lr=0.0001
Training completed! Best loss: 0.2027 (epoch 37)
Model saved to: outputs/5k_model/best_model.pt
```

**关键参数：**
- `--epochs` (default: 50) — 训练轮数
- `--batch-size` (default: 32) — 每批样本数
- `--lr` (default: 1e-4) — 学习率
- `--save-dir` (default: outputs/5k_model) — 输出目录

---

## 🔮 Step 4: 推理（验证模型）

### 最简命令
```bash
python model/infer_pinhan.py \
    --model outputs/5k_model/best_model.pt \
    --pinyin "zhong1 guo2 ren2"
```

**期望输出：**
```
Input:  中国人
Predicted: 中国人
```

### 其他推理模式
```bash
# 使用束搜索（beam size=3）
python model/infer_pinhan.py \
    --model outputs/5k_model/best_model.pt \
    --pinyin "ni3 hao3" \
    --beam-size 3

# 指定设备
python model/infer_pinhan.py \
    --model outputs/5k_model/best_model.pt \
    --pinyin "za4 me5" \
    --device cpu
```

---

## 🧪 Step 5: 测试（可选，5 秒）

```bash
python tests/run_tests.py
```

输出示例：
```
Running 26 tests...
✅ 17 Unit Tests passed
✅ 4 Integration Tests passed
✅ 5 Performance Tests passed
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
All tests passed! ✅
```

---

## 📁 输出结构

训练完成后，`outputs/5k_model/` 中包含：

```
outputs/5k_model/
├── best_model.pt           ⭐ 最优模型（用于推理）
├── checkpoint_epoch7.pt    检查点（可用于恢复）
├── checkpoint_epoch8.pt
├── checkpoint_epoch9.pt
├── model.pt                最后一个 epoch 的模型
├── src_vocab.json          拼音词表
├── tgt_vocab.json          汉字词表
└── logs/
    ├── config.json         训练配置
    └── training_summary.json 性能指标
```

---

## 🎯 常见命令速查表

| 场景 | 命令 |
|------|------|
| 快速训练 5 分钟 | `python model/train_pinhan.py --data data/5k.jsonl --epochs 5` |
| 标准训练（40 epochs） | `python model/train_pinhan.py --data data/5k.jsonl --epochs 40` |
| 恢复中断的训练 | `python model/train_pinhan.py --data data/5k.jsonl --resume` |
| 有未提交更改时强制训练 | `python model/train_pinhan.py --data data/5k.jsonl --force` |
| 简单推理 | `python model/infer_pinhan.py --model outputs/5k_model/best_model.pt --pinyin "ni3 hao3"` |
| 束搜索推理 | `python model/infer_pinhan.py --model outputs/5k_model/best_model.pt --pinyin "ni3 hao3" --beam-size 3` |
| 运行测试 | `python tests/run_tests.py` |
| 检查设置 | `python tests/run_tests.py` |

---

## ⚠️ 常见问题（4 个）

### ❌ "No module named 'torch'"
```bash
# 解决方案：重新安装依赖
pip install -r requirements.txt
```

### ❌ "拼音格式错误"
```bash
# ❌ 错误示例（缺少声调）
python model/infer_pinhan.py --model outputs/5k_model/best_model.pt --pinyin "ni hao"

# ✅ 正确示例（包含声调）
python model/infer_pinhan.py --model outputs/5k_model/best_model.pt --pinyin "ni3 hao3"
```

### ❌ "模型文件不存在"
```bash
# 检查模型是否存在
ls outputs/5k_model/best_model.pt  # 应该存在

# 如果不存在，重新训练
python model/train_pinhan.py --data data/5k.jsonl
```

### ⚠️ "Uncommitted changes detected"
```bash
# 说明：训练前检测到未提交的代码更改
# 这是为了确保训练的可复现性

# 解决方案1：提交您的更改
git add .
git commit -m "Your commit message"

# 解决方案2：使用 --force 标志跳过检查（不推荐）
python model/train_pinhan.py --data data/5k.jsonl --force
```

---

## 📚 更多信息

- **详细参数说明？** → 查看 [REMADE.md](REMADE.md)
- **技术细节？** → 查看 [TECH_REPORT.md](TECH_REPORT.md)
- **项目导航？** → 查看 [README.md](README.md)

---

**预计运行时间：**
- 训练 5k 数据 40 epochs: ~25 分钟 (CPU)
- 单次推理: ~400-600ms (CPU)
- 完整测试套件: ~2-3 分钟

**下一步：** 训练后查看 `outputs/5k_model/logs/training_summary.json` 验证性能指标。
