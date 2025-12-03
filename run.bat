@echo off
REM PinHan 启动脚本 (Windows)
REM 快速启动拼音输入法引擎 API

echo.
echo ========================================
echo    PinHan - 拼音输入法引擎启动脚本
echo ========================================
echo.

REM 检查 Python 是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到 Python！请先安装 Python 3.8+
    echo 下载: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM 检查虚拟环境
if not exist ".venv" (
    echo [正在创建虚拟环境...]
    python -m venv .venv
    if errorlevel 1 (
        echo [错误] 虚拟环境创建失败！
        pause
        exit /b 1
    )
)

REM 激活虚拟环境
echo [激活虚拟环境...]
call .venv\Scripts\activate.bat

REM 检查依赖是否已安装
python -c "import fastapi" >nul 2>&1
if errorlevel 1 (
    echo [正在安装依赖（仅需一次）...]
    pip install -e . -q
    if errorlevel 1 (
        echo [错误] 依赖安装失败！
        pause
        exit /b 1
    )
)

REM 检查词典是否存在
if not exist "data\dicts\word_dict.json" (
    echo [错误] 词典文件不存在！
    echo 请先运行: python scripts/build_dict.py
    pause
    exit /b 1
)

echo.
echo ✓ 环境检查完毕
echo.
echo [启动 PinHan API 服务...]
echo.
echo 📍 访问地址:
echo    - API 文档: http://localhost:8000/docs
echo    - ReDoc: http://localhost:8000/redoc
echo    - API 地址: http://localhost:8000
echo.
echo 💡 快速测试:
echo    curl http://localhost:8000/api/convert?pinyin=nihao
echo.
echo [Ctrl+C 停止服务]
echo.

python api/server.py

pause
