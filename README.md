# PinHan - 拼音到汉字 Seq2Seq Transformer

基于 PyTorch Seq2Seq Transformer 的拼音→汉字转换模型。

## 快速开始

### 环境准备

```bash
# 激活虚拟环境 (Windows)
.\.venv\Scripts\Activate.ps1

# 安装依赖
pip install -r requirements.txt
```

### 训练

```bash
# 基础训练 (5k 数据，40 epochs)
python model/train.py --data data/5k.jsonl --epochs 40 --batch-size 32

# 自定义参数
python model/train.py \
    --data data/5k.jsonl \
    --epochs 40 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --save-dir outputs/my_model
```

### 推理

```bash
# 简单推理
python model/infer.py --model outputs/5k_model/best_model.pt --pinyin "zhong1 guo2"

# 束搜索推理 (质量更高)
python model/infer.py \
    --model outputs/5k_model/best_model.pt \
    --pinyin "zhong1 guo2" \
    --beam-size 3
```

## 项目结构

```
PinHan/
├── model/
│   ├── train.py              # 训练脚本
│   ├── infer.py              # 推理脚本
│   ├── evaluate.py           # 评估脚本
│   └── core/
│       ├── seq2seq_transformer.py    # Seq2Seq Transformer 模型
│       ├── pinyin_utils.py           # 拼音工具函数
│       └── checkpoint_manager.py     # 检查点管理
├── tests/                    # 单元测试和集成测试
├── preprocess/              # 数据预处理脚本
├── data/                    # 数据集
└── outputs/                 # 模型输出
```

## 核心特性

- ✅ **完整的训练流程** - 数据加载、训练、验证、检查点管理
- ✅ **多种推理方式** - 贪心解码、束搜索
- ✅ **性能优化** - 正则表达式预编译、缓存优化
- ✅ **完整的轻声支持** - 拼音包括 0-4 五种声调标记
- ✅ **错误处理** - 完整的输入验证和异常处理
- ✅ **100% 测试覆盖** - 32 个单元/集成/性能测试全部通过

## 数据格式

训练数据为 JSONL 格式，每行一个 JSON 对象：

```json
{"pinyin": "ni3 hao3", "hanzi": "你好"}
{"pinyin": "zhong1 guo2", "hanzi": "中国"}
{"pinyin": "zi4 you2", "hanzi": "自由"}
```

**注意：** 拼音必须包含声调数字 (1,2,3,4 表示声调，0 表示轻声)

## 可用数据集

- `data/5k.jsonl` - 5,000 样本对（推荐用于快速测试）
- `data/10k.jsonl` - 10,000 样本对（完整版）
- `data/test_mini.jsonl` - 500 样本对（测试用）

## 模型配置

默认配置：

| 参数 | 值 |
|-----|-----|
| d_model | 512 |
| nhead | 8 |
| num_encoder_layers | 6 |
| num_decoder_layers | 6 |
| dim_feedforward | 2048 |
| dropout | 0.1 |

在 `model/train.py` 中修改 `TransformerConfig` 以自定义配置。

## 性能指标

| 指标 | 值 |
|-----|-----|
| 最佳 Loss | 0.2027 (epoch 37) |
| 模型大小 | ~5.4M 参数 |
| 推理速度 | 400-600ms (贪心解码) |
| 内存占用 | ~2GB (训练时) |

## 常见问题

### Q: 如何使用自己的数据？

A: 准备 JSONL 格式数据，每行格式为 `{"pinyin": "...", "hanzi": "..."}`，然后运行：
```bash
python model/train.py --data your_data.jsonl --epochs 40
```

### Q: 如何优化模型性能？

A: 尝试以下方法：
1. 增加训练 epochs
2. 调整学习率
3. 增加数据集大小
4. 调整模型参数 (d_model, num_layers 等)

### Q: 支持 GPU 训练吗？

A: 支持。脚本会自动检测 CUDA 设备。如果有 GPU，将自动使用。

## 测试

运行所有测试：

```bash
pytest tests/ -v
```

运行特定测试：

```bash
# 单元测试
pytest tests/test_units.py -v

# 集成测试
pytest tests/test_integration.py -v

# 性能测试
pytest tests/test_performance.py -v
```

## 许可证

MIT License

## 版本历史

### v1.0.0 (2025-11-16)
- ✅ 初始版本发布
- ✅ 修复拼音验证支持轻声标记 '0'
- ✅ 完整的文档和测试覆盖
- ✅ 性能优化和错误处理
