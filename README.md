# PinHan - 拼音到汉字转换

基于词典匹配 + 神经网络的高精度拼音输入法引擎。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 构建字典

```bash
python preprocess/build_dict.py
```

这会从 CC-CEDICT 下载并构建：
- `word_dict.json` - 词语字典（103,031 条）
- `char_dict.json` - 单字字典（1,734 个拼音）
- `char_freq.json` - 字频表（20,992 字）

### 3. 测试转换

```bash
python -m model.infer --pinyin "ni3 hao3, jin1 tian1 tian1 qi4 hen3 hao3!"
```

输出：
```
你好,今天天气很好!
```

---

## 使用方法

### 推理

```bash
# 基本用法
python -m model.infer -p "wo3 ai4 zhong1 guo2"
# 输出: 我爱中国

# 带标点
python -m model.infer -p "ni3 hao3 ma5?"
# 输出: 你好么?

# 完整句子
python -m model.infer -p "jin1 tian1 wo3 men5 yi1 qi3 chu1 qu4 wan2"
# 输出: 今天我们一起出去完
```

### 提取语料（可选）

如果你有维基百科 XML 文件：

```bash
python preprocess/extract_corpus.py -i data/zhwiki.xml -o data/corpus.jsonl --max 100000
```

---

## 项目结构

```
PinHan/
├── model/
│   ├── infer.py              # 推理脚本
│   └── core/
│       └── pinyin_dict.py    # 字典类
├── preprocess/
│   ├── build_dict.py         # 构建字典（CC-CEDICT）
│   └── extract_corpus.py     # 提取语料（维基百科）
├── dicts/
│   ├── word_dict.json        # 词语字典
│   ├── char_dict.json        # 单字字典
│   └── char_freq.json        # 字频表
├── tests/
│   └── test_dict.py          # 测试
└── requirements.txt
```

---

## 技术架构

### 三层解码策略

```
输入: ni3 hao3 ma5
      ↓
[第1层] 词语匹配: "ni3 hao3" → "你好"
      ↓
[第2层] 单字回退: "ma5" → 轻声回退 → "么"
      ↓
[第3层] 神经网络: (待实现) 上下文消歧
      ↓
输出: 你好么
```

### 字典来源

| 字典 | 来源 | 数量 |
|------|------|------|
| 词语 | CC-CEDICT | 103,031 条 |
| 单字 | CC-CEDICT + 字频排序 | 1,734 拼音 |
| 字频 | 现代汉语常用字表 | 20,992 字 |

### 特性

- ✅ 词语优先匹配（最大正向匹配）
- ✅ 常用字优先（按字频排序）
- ✅ 轻声支持（ma5 自动回退到 ma1/ma2/ma3/ma4）
- ✅ 标点保留
- 🔄 神经网络消歧（开发中）

---

## 测试

```bash
python -m pytest tests/ -v
```

---

## 拼音格式

使用数字声调：

| 声调 | 格式 | 示例 |
|------|------|------|
| 一声 | 1 | ma1 (妈) |
| 二声 | 2 | ma2 (麻) |
| 三声 | 3 | ma3 (马) |
| 四声 | 4 | ma4 (骂) |
| 轻声 | 5 | ma5 (么/吗) |

---

## 许可证

MIT License

