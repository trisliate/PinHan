# 目录结构说明

## 整体架构

```
PinHan/
├── data/                          # 🔑 数据层（词库源 + 编译输出）
│   ├── dicts/                     # ✅ 编译后的词典（运行时加载）
│   ├── extensions/                # 用户扩展词库（优先级40）
│   └── sources/                   # 第三方词库（优先级30）
│
├── scripts/                       # 脚本层
│   └── build_dict.py              # 词库构建脚本
│
├── pinhan/                        # 🔥 应用层（Python包）
│   ├── engine/                    # 核心引擎
│   ├── api/                       # REST API
│   ├── cli.py                     # 命令行工具
│   └── __init__.py
│
└── pyproject.toml                 # 打包配置（包含data/dicts/）
```

## 关键设计

### ✅ 词典数据位置

**使用外部 `data/` 而不是包内**：

| 位置 | 用途 | 说明 |
|------|------|------|
| `data/dicts/` | ✅ 运行时加载 | 编译后的JSON词典，程序启动时从此加载 |
| `data/extensions/` | 📝 用户编辑 | 热词、品牌词等，支持热更新 |
| `data/sources/` | 📚 词库源 | SUBTLEX-CH、THUOCL等第三方词库 |

**工作流**：

```
data/sources/ (SUBTLEX-CH, jieba, 等)
    ↓
    + data/extensions/ (hotwords.txt)
    ↓
scripts/build_dict.py (融合)
    ↓
data/dicts/ (char_dict.json, word_dict.json, ...)
    ↓
pinhan/engine/__init__ (启动时加载)
```

### 🔄 运行时路径查找

**核心引擎自动查找词典**：

```python
# pinhan/engine/core.py
def __init__(self, config=None, dicts_dir=None):
    if dicts_dir is None:
        # 自动查找项目根目录下的 data/dicts
        pkg_dir = os.path.dirname(os.path.dirname(__file__))  # pinhan/
        root_dir = os.path.dirname(pkg_dir)                   # 项目根
        dicts_dir = os.path.join(root_dir, 'data', 'dicts')
    
    self.dicts_dir = dicts_dir
```

**工作场景**：

1. **本地开发**：自动找到 `../../../data/dicts/`（项目根）
2. **pip 安装**：词典包含在 `site-packages/pinhan/data/dicts/`
3. **Docker**：COPY 时包含 `data/dicts/`

### 📦 打包配置

**pyproject.toml**:

```toml
[tool.hatch.build.targets.wheel]
packages = ["pinhan"]
include = [
    "data/dicts/*.json",
    "data/dicts/*.txt",
]

[tool.hatch.build.targets.sdist]
include = [
    "pinhan/",
    "data/dicts/",
    "scripts/",
]
```

**结果**：
- ✅ wheel 包含 `data/dicts/` 中的词典
- ✅ source 包含 `scripts/build_dict.py`（用户可重新构建）
- ✅ 无需包含 `data/sources/`（太大）

### 🐳 Docker 配置

**Dockerfile**:

```dockerfile
COPY data/dicts/ /app/pinhan/data/dicts/
# 或
COPY data/dicts/ /app/data/dicts/
```

## 目录职责划分

| 目录 | 职责 | 修改频率 | 备注 |
|------|------|---------|------|
| `data/` | 数据处理 | 每次构建 | 包含词库源和编译输出 |
| `data/dicts/` | 词典存储 | 重建词库时 | JSON格式，运行时加载 |
| `data/extensions/` | 热词管理 | 频繁 | 用户添加的词库 |
| `data/sources/` | 词库源 | 很少 | 第三方词库（需手动下载） |
| `scripts/` | 构建脚本 | 很少 | 词库融合逻辑 |
| `pinhan/` | 应用代码 | 常常 | 引擎、API、CLI |

## 迁移清单（已完成）

- ✅ `pinhan/data/dicts/` → `data/dicts/`（移动词典文件）
- ✅ `core.py` 更新路径查找逻辑
- ✅ `__init__.py` 更新工厂函数签名
- ✅ `api/server.py` 移除硬编码路径
- ✅ `pyproject.toml` 配置 include
- ✅ `.gitignore` 更新规则
- ✅ `build_dict.py` 注释说明
- ✅ `README.md` 更新文档
- ✅ 验证功能正常

## 最终目录树

```
PinHan/
├── data/
│   ├── dicts/                    # ✅ 核心词典
│   │   ├── char_dict.json
│   │   ├── word_dict.json
│   │   ├── char_freq.json
│   │   ├── word_freq.json
│   │   └── pinyin_table.txt
│   ├── extensions/
│   │   ├── README.md
│   │   └── hotwords.txt
│   └── sources/
│       ├── SUBTLEX-CH/
│       └── cedict.txt.gz
│
├── scripts/
│   └── build_dict.py
│
├── pinhan/
│   ├── engine/
│   │   ├── core.py               # ✅ 自动查找 ../../../data/dicts/
│   │   ├── dictionary.py
│   │   ├── corrector.py
│   │   ├── segmenter.py
│   │   ├── generator.py
│   │   ├── cache.py
│   │   ├── config.py
│   │   ├── logging.py
│   │   └── __init__.py
│   ├── api/
│   │   ├── server.py
│   │   └── __init__.py
│   ├── cli.py
│   └── __init__.py
│
├── pyproject.toml               # ✅ include data/dicts/
├── Dockerfile
├── README.md
└── .gitignore                   # ✅ 保留 data/dicts/, data/extensions/
```

## 使用指南

**添加热词**：

```bash
# 编辑 data/extensions/hotwords.txt
echo "新产品 100" >> data/extensions/hotwords.txt

# 重新构建
python scripts/build_dict.py

# 程序自动加载新词典
```

**集成第三方词库**：

```bash
# 将转换后的词库放入 data/sources/
cp my_vocab.txt data/sources/

# 重新构建
python scripts/build_dict.py
```

**验证词典加载**：

```python
from pinhan import IMEEngineV3

engine = IMEEngineV3()
print(engine.dicts_dir)  # 输出词典目录路径
```

