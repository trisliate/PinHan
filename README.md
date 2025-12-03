# PinHan - 轻量级智能拼音输入法引擎

![GitHub license](https://img.shields.io/github/license/trisliate/pinhan) 
![Python Version](https://img.shields.io/badge/python-3.9+-blue) 
![Status](https://img.shields.io/badge/status-stable-green)

> **纯词典架构**的拼音输入法，无深度学习，专为嵌入式/MCU 设备和轻量化部署优化

## 🚀 快速开始

### 最快启动（3 秒）

**Windows 用户：** 双击 `run.bat`  
**Linux/Mac 用户：** 执行 `bash run.sh`  

**或用命令：**
```powershell
pip install -e .
python api/server.py
# 打开浏览器访问 http://localhost:8000/docs
```

---

## 📋 三种启动方式

### 方式 1：开发模式（推荐开发者）

```powershell
pip install -e .
uvicorn api.server:app --reload  # 代码改动自动重启
```

访问：http://localhost:8000/docs

### 方式 2：Wheel 分发（推荐分享给其他开发者）

```powershell
# 第一步：构建（已完成）
python -m build --wheel
# 生成：dist/pinhan-0.1.0-py3-none-any.whl (30 KB)

# 第二步：他人安装
pip install pinhan-0.1.0-py3-none-any.whl
python -c "from api.server import app; import uvicorn; uvicorn.run(app, port=8000)"
```

### 方式 3：Docker（推荐生产环境）

```powershell
docker build -t pinhan:latest .
docker run -p 8000:8000 pinhan:latest
```

---

## ✨ 核心特性

- **纯词典设计** - 无深度学习，无神经网络，仅依赖高质量词表
- **低延迟** - <10ms 响应，缓存命中 <1ms
- **轻量级** - Docker 镜像 <200MB，支持嵌入式设备
- **灵活词库** - 支持多来源词表融合（SUBTLEX-CH、jieba、自定义扩展）
- **词库规模** - 325,507 词条（+176% 扩充）
- **模糊音纠错** - 支持声母/韵母模糊音和键盘纠错
- **RESTful API** - 完整的 HTTP 接口和交互式文档

---

## 🏗️ 为什么是纯词典架构？

| 特性 | 纯词典 | 神经网络模型 |
|------|--------|----------|
| 可解释性 | ✅ 清晰 | ❌ 黑盒 |
| 训练成本 | ✅ 无 | ❌ 需要 GPU |
| 部署灵活 | ✅ 任何设备 | ❌ 需要足够内存 |
| 可定制性 | ✅ 添加热词即时生效 | ❌ 需要重训练 |
| 实时更新 | ✅ 快速 | ❌ 需要重新部署 |

---

## 📦 项目结构

```
pinhan/
├── api/               # FastAPI 应用
├── engine/            # 核心引擎
│   ├── core.py       # 主引擎
│   ├── dictionary.py # 词典管理
│   ├── segmenter.py  # 拼音切分
│   ├── corrector.py  # 错误纠正
│   └── generator.py  # 候选生成
├── scripts/          # 工具脚本
│   ├── build_dict.py # 构建词典
│   └── download_vocab.py # 下载词库
└── data/             # 数据文件
    └── dicts/        # 编译后的词典
```

---

## 🎯 常用命令

### 环境管理
```powershell
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境（Windows）
.\.venv\Scripts\Activate.ps1

# 激活虚拟环境（Linux/Mac）
source .venv/bin/activate
```

### 安装和运行
```powershell
# 开发模式安装
pip install -e .

# 启动 API 服务
python api/server.py

# 自动重启模式（推荐开发）
uvicorn api.server:app --reload

# 指定端口
uvicorn api.server:app --port 9000
```

### 打包
```powershell
# 构建 Wheel 包
python -m build --wheel

# 安装 Wheel 包
pip install dist/pinhan-0.1.0-py3-none-any.whl
```

### Docker
```powershell
# 构建镜像
docker build -t pinhan:latest .

# 运行容器
docker run -d -p 8000:8000 pinhan:latest

# 查看日志
docker logs <container_id>
```

### 词库管理
```powershell
# 下载第三方词库
python scripts/download_vocab.py

# 构建词典
python scripts/build_dict.py
```

---

## 📊 词库统计

- **词条数量：** 325,507（包含 SUBTLEX-CH、jieba、THUOCL、Sogou）
- **词典大小：** 33 MB
- **支持拼音：** 420 种组合
- **响应时间：** <10ms（缓存命中 <1ms）

---

## 🔗 API 使用示例

### Python 直接调用
```python
from pinhan.engine import create_engine_v3

engine = create_engine_v3(None)
result = engine.process('zhongguoren')

print(result.candidates[0].text)  # 输出：中国人

# 获取所有候选
for c in result.candidates[:3]:
    print(f"{c.text}: {c.score:.4f}")
```

### REST API 调用
```bash
# 简单查询
curl "http://localhost:8000/api/convert?pinyin=nihao"

# 指定返回数量
curl "http://localhost:8000/api/convert?pinyin=zhongguoren&top_k=5"
```

### 浏览器
打开 http://localhost:8000/docs 查看交互式文档

---

## 🆘 故障排查

### 问题：ModuleNotFoundError

```powershell
# 确认虚拟环境已激活（前缀应该是 (.venv)）
.\.venv\Scripts\Activate.ps1

# 重新安装
pip install -e .
```

### 问题：虚拟环境无法激活

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1
```

### 问题：端口被占用

```powershell
# 更换端口
python api/server.py --port 8001

# 或杀死占用进程（Windows）
Get-NetTCPConnection -LocalPort 8000 | Stop-Process -Force
```

### 问题：找不到词典文件

```powershell
# 构建词典
python scripts/build_dict.py
```

### 问题：Docker 镜像构建失败

```powershell
# 清理旧镜像后重试
docker system prune -a
docker build -t pinhan:latest .
```

---

## 📚 项目配置文件

| 文件 | 用途 |
|------|------|
| `pyproject.toml` | Python 项目配置和依赖 |
| `requirements.txt` | 依赖列表 |
| `Dockerfile` | Docker 镜像定义 |
| `.gitignore` | Git 忽略规则 |
| `run.bat` | Windows 启动脚本 |
| `run.sh` | Linux/Mac 启动脚本 |

---

## 🎓 工作流示例

### 本地开发
```powershell
pip install -e .
uvicorn api.server:app --reload
# 修改代码自动重启
```

### 分享给他人
```powershell
python -m build --wheel
# 发送 dist/pinhan-0.1.0-py3-none-any.whl (30 KB)
```

### 生产部署
```powershell
docker build -t pinhan:latest .
docker run -d -p 8000:8000 pinhan:latest
```

### 发布到 PyPI
```powershell
pip install twine
twine upload dist/*
```

---

## 📝 词库来源

项目采用**三层优先级融合**策略：

| 优先级 | 来源 | 特点 |
|------|------|------|
| 🔴 高 | SUBTLEX-CH | 电影/电视字幕，口语频率最真实 |
| 🟡 中 | 自定义扩展 | 热词、品牌词、行业术语 |
| 🟢 低 | jieba/THUOCL/Sogou | 通用词表和领域词库 |
| ⚪ 基础 | CC-CEDICT | 拼音映射和冷启动 |

---

## 🌟 项目特点

✅ **无依赖** - 不依赖 PyTorch、TensorFlow 等大型框架  
✅ **快速启动** - 3 秒内启动服务  
✅ **易于部署** - 支持 Docker、Wheel、本地部署  
✅ **轻量级** - Docker 镜像 <200MB  
✅ **可扩展** - 支持自定义词库和扩展  
✅ **生产就绪** - 完整的 API 文档和错误处理  

---

## 📖 更多信息

- **项目地址：** https://github.com/trisliate/pinhan
- **问题反馈：** GitHub Issues
- **许可证：** MIT

---

## 🚀 现在就开始

1. **快速启动**：`pip install -e .` 然后 `python api/server.py`
2. **查看文档**：打开 http://localhost:8000/docs
3. **测试 API**：http://localhost:8000/api/convert?pinyin=nihao
4. **分享给他人**：`python -m build --wheel` 然后发送 `.whl` 文件

---

**祝你使用愉快！🎉**

**最后更新：** 2025-12-03  
**版本：** 0.1.0  
**维护者：** PinHan Team
