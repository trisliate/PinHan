#!/bin/bash

# PinHan 启动脚本 (Linux/macOS)
# 快速启动拼音输入法引擎 API

echo ""
echo "========================================"
echo "   PinHan - 拼音输入法引擎启动脚本"
echo "========================================"
echo ""

# 检查 Python 是否安装
if ! command -v python3 &> /dev/null; then
    echo "[错误] 未检测到 Python 3！请先安装 Python 3.8+"
    echo "macOS: brew install python3"
    echo "Ubuntu: sudo apt-get install python3 python3-pip"
    exit 1
fi

# 检查虚拟环境
if [ ! -d ".venv" ]; then
    echo "[正在创建虚拟环境...]"
    python3 -m venv .venv
    if [ $? -ne 0 ]; then
        echo "[错误] 虚拟环境创建失败！"
        exit 1
    fi
fi

# 激活虚拟环境
echo "[激活虚拟环境...]"
source .venv/bin/activate

# 检查依赖是否已安装
python -c "import fastapi" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "[正在安装依赖（仅需一次）...]"
    pip install -e . -q
    if [ $? -ne 0 ]; then
        echo "[错误] 依赖安装失败！"
        exit 1
    fi
fi

# 检查词典是否存在
if [ ! -f "data/dicts/word_dict.json" ]; then
    echo "[错误] 词典文件不存在！"
    echo "请先运行: python scripts/build_dict.py"
    exit 1
fi

echo ""
echo "✓ 环境检查完毕"
echo ""
echo "[启动 PinHan API 服务...]"
echo ""
echo "📍 访问地址:"
echo "   - API 文档: http://localhost:8000/docs"
echo "   - ReDoc: http://localhost:8000/redoc"
echo "   - API 地址: http://localhost:8000"
echo ""
echo "💡 快速测试:"
echo "   curl http://localhost:8000/api/convert?pinyin=nihao"
echo ""
echo "[按 Ctrl+C 停止服务]"
echo ""

python api/server.py
