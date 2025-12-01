# IME-SLM 拼音输入法引擎

基于 Transformer 的智能拼音输入法引擎，采用分层策略优化响应速度。

## 架构

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI 服务层                        │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│                   IME Engine v2                         │
│  ┌────────────────────────────────────────────────────┐ │
│  │ Level 1: 热词缓存 (LRU)           < 1ms            │ │
│  ├────────────────────────────────────────────────────┤ │
│  │ Level 2: 词典快速路径 (≤6字符)    < 10ms           │ │
│  ├────────────────────────────────────────────────────┤ │
│  │ Level 3: 模型推理 (长句)          < 300ms          │ │
│  └────────────────────────────────────────────────────┘ │
└───────────────────────┬─────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
  ┌──────────┐   ┌──────────┐   ┌──────────┐
  │ Corrector│   │   P2H    │   │   SLM    │
  │ 拼音纠错 │   │  主模型  │   │ 语义重排 │
  └──────────┘   └──────────┘   └──────────┘
```

## 特性

- **分层策略**: 短输入走词典快速路径，长输入走模型推理
- **LRU 缓存**: 热词缓存，加速比 400x+
- **拼音纠错**: 支持常见拼音错误纠正
- **智能切分**: 连续拼音自动切分
- **语义重排**: SLM 模型进行候选重排序

## 性能指标

| 场景 | 延迟 |
|------|------|
| Level 1 (缓存命中) | < 0.01ms |
| Level 2 (短输入词典) | < 1ms |
| Level 3 (长输入模型) | < 300ms |
| 缓存命中率 | 97%+ (压测) |

## 安装

```bash
pip install -r requirements.txt
```

## 训练模型

### P2H 模型 (Pinyin-to-Hanzi)

```bash
# 验证训练 (RTX 3060 6GB)
python p2h/train.py --epochs 15 --batch-size 32 --max-samples 50000 --amp --patience 5

# 完整训练 (RTX 4070Ti 16GB)
python p2h/train.py --epochs 25 --batch-size 128 --amp --patience 8
```

### SLM 模型 (语义重排)

```bash
# 验证训练
python slm/train.py --epochs 15 --batch-size 64 --max-samples 50000 --amp --patience 5

# 完整训练
python slm/train.py --epochs 20 --batch-size 256 --amp --patience 8
```

## 启动服务

```bash
python -m api.server
# 或
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

API 文档: http://localhost:8000/docs

## API 使用

### POST /ime

```bash
curl -X POST http://localhost:8000/ime \
  -H "Content-Type: application/json" \
  -d '{"pinyin": "jintiantianqihenhao", "top_k": 5}'
```

响应:
```json
{
  "raw_pinyin": "jintiantianqihenhao",
  "corrected_pinyin": "jin tian tian qi hen hao",
  "segmented_pinyin": ["jin", "tian", "tian", "qi", "hen", "hao"],
  "candidates": [
    {"text": "今天天气很好", "score": 0.85, "source": "p2h_model"},
    ...
  ],
  "metadata": {
    "level": 3,
    "elapsed_ms": 265.4,
    "cache_hit_rate": 0.95
  }
}
```

### GET /ime/simple

```bash
curl "http://localhost:8000/ime/simple?pinyin=nihao&top_k=5"
```

### GET /stats

```bash
curl http://localhost:8000/stats
```

## 目录结构

```
PinHan/
├── engine.py          # v1 引擎
├── engine_v2.py       # v2 引擎 (分层策略)
├── config.py          # 配置
├── api/
│   └── server.py      # FastAPI 服务
├── corrector/         # 拼音纠错模块
├── segmenter/         # 拼音切分模块
├── p2h/               # P2H 主模型
│   ├── model.py
│   └── train.py
├── slm/               # SLM 语义模型
│   ├── model.py
│   └── train.py
├── dicts/             # 词典数据
│   ├── char_dict.json
│   ├── word_dict.json
│   └── bigram.json
├── checkpoints/       # 模型检查点
│   ├── p2h/
│   └── slm/
└── tests/             # 测试
```

## 测试

```bash
# v2 引擎测试
python tests/test_v2.py

# 集成测试
python tests/test_integration.py
```

## 技术栈

- Python 3.11
- PyTorch 2.5.1 + CUDA 12.x
- FastAPI + Uvicorn
- Transformer (Encoder-Decoder for P2H, Causal LM for SLM)
