# PinHan Seq2Seq Transformer - 拼音到汉字转换

本项目使用 PyTorch Transformer 实现拼音到汉字的端到端转换模型。

## 快速开始

### 1. 环境设置

```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux/Mac:
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

数据格式：`data/10k.jsonl` 中每行为 JSON，需包含 `pinyin`（源）和 `hanzi`（目标）字段。

```json
{"pinyin": "ni3 hao3", "hanzi": "你好"}
{"pinyin": "xie4 xie4", "hanzi": "谢谢"}
```

### 3. 小规模验证训练

在投入大型 GPU 前，先用小规模数据验证效果：

```bash
python quick_small_train.py --sample-size 1500 --epochs 50
```

输出：`training_analysis.json`（包含损失曲线和收敛估计）

### 4. 完整模型训练

```bash
python model/train_pinhan.py \
    --data data/10k.jsonl \
    --save-dir outputs/pinhan_model \
    --epochs 100 \
    --batch-size 32 \
    --lr 1e-4
```

### 5. 模型推理

```bash
python model/infer_pinhan.py \
    --model outputs/pinhan_model/checkpoint_epoch50.pt \
    --pinyin "ni3 hao3"
```

### 6. 项目验证

在提交代码前运行验证脚本：

```bash
python verify_project.py
```

## 项目结构

```
model/
  ├── seq2seq_transformer.py   # 模型架构
  ├── train_pinhan.py          # 生产训练脚本
  ├── train_utils.py           # 训练工具函数
  ├── infer_pinhan.py          # 推理脚本
  └── evaluate.py              # 评估脚本

preprocess/
  ├── pinyin_utils.py          # 拼音处理工具
  ├── extract_and_clean.py     # 数据清洗
  └── split_dataset.py         # 数据分割

tests/
  ├── test_units.py            # 单元测试（17项）
  ├── test_integration.py      # 集成测试（4项）
  ├── test_performance.py      # 性能测试（5项）
  └── run_tests.py             # 测试运行器

data/
  ├── 10k.jsonl                # 验证数据集
  └── test_mini.jsonl          # 测试数据集

outputs/
  └── pinhan_model/            # 模型检查点和词表
```

## 测试

运行完整测试套件（26 个测试）：

```bash
python tests/run_tests.py
```

## 模型参数

- **架构**：Seq2Seq Transformer（编码器-解码器）
- **隐藏维度**：d_model=256
- **注意力头数**：nhead=4
- **层数**：3 层编码器 + 3 层解码器
- **总参数**：~5.4M

## 部署

详见以下文档：
- `PRODUCTION_READINESS.md` - 生产部署检查清单
- `GPU_RENTAL_GUIDE.md` - GPU 租赁选择指南（RTX 3090 推荐）
- `CLOUD_DEPLOYMENT_GUIDE.md` - 云平台部署指南
- `QUICK_EXECUTION_GUIDE.md` - 快速执行指南

## 许可证

MIT License
