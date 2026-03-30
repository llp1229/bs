@echo off
echo ======================================
echo    古建筑健康监测系统启动脚本
echo ======================================
echo.

REM 检查是否安装了 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到 Python，请先安装 Python 3.8+
    pause
    exit /b
)

REM 激活虚拟环境（如果使用了 venv）
if exist ".venv\Scripts\activate.bat" (
    echo [信息] 激活虚拟环境...
    call .venv\Scripts\activate.bat
) else (
    echo [信息] 未找到虚拟环境，使用全局 Python 环境
)

REM 安装依赖（如果 requirements.txt 存在）
if exist "requirements.txt" (
    echo [信息] 检查并安装依赖...
    pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
) else (
    echo [信息] 未找到 requirements.txt，跳过依赖安装
)

REM 启动 Streamlit 网页
echo [信息] 启动网页系统...
streamlit run main.py --server.port 8501 --server.headless false

echo.
echo [信息] 系统已退出
pause