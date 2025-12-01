# PinHan 拼音输入法引擎

轻量级智能拼音输入法引擎，采用词典召回 + SLM 重排序架构，专为嵌入式设备优化。

## 特性

- **精简架构**: 词典为主，SLM 辅助重排序（仅在有上下文时使用）
- **低延迟**: 平均 15ms，缓存命中 < 1ms
- **模糊纠错**: 支持常见拼音错误（如 shengme -> 什么）
- **上下文感知**: SLM 利用上下文消歧（如 做+shi -> 事）
- **轻量模型**: SLM Lite 仅 ~1M 参数，适合嵌入式部署

## 性能指标

| 场景 | 延迟 |
|------|------|
| 缓存命中 | < 1ms |
| 无上下文 (纯词典) | 5-10ms |
| 有上下文 (SLM 重排) | 10-20ms |
| 段落连续输入平均 | ~15ms/词 |

## 安装

pip install -r requirements.txt

## 训练 SLM Lite

python slm/train_lite.py --epochs 20 --batch-size 128 --max-samples 50000

配置: 2 层, 128 维, 4 头, 5000 词表

## 启动服务

python -m api.server

API 文档: http://localhost:8000/docs

## 目录结构

- engine.py - IME 引擎 v3 (词典 + SLM Lite)
- api/server.py - FastAPI 服务
- corrector/ - 拼音纠错模块
- segmenter/ - 拼音切分模块
- slm/ - SLM 语义模型
- dicts/ - 词典数据
- checkpoints/slm_lite/ - 模型文件
- tests/ - 测试脚本

## 测试

python tests/test_local.py       # 本地引擎测试
python tests/test_context.py     # 上下文消歧测试
python tests/test_paragraph.py   # 段落输入模拟

## 技术栈

- Python 3.11 + PyTorch 2.x + CUDA
- FastAPI + Uvicorn
- Transformer (Causal LM)

## 版本历史

- v3.0 - 精简架构，移除 P2H，仅保留词典 + SLM Lite
- v2.0 - 分层策略 (P2H + SLM)
- v1.0 - 初版
