@echo off
chcp 65001 >nul
title AI养护咨询服务器
cd /d %~dp0

echo.
echo ==========================================
echo   山西古建筑健康监测系统 - AI 助手
echo ==========================================
echo.

:: 检查Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到Python，请先安装Python 3.8+
    pause
    exit /b 1
)

:: 检查flask
python -c "import flask" >nul 2>&1
if errorlevel 1 (
    echo [安装] 正在安装 Flask...
    pip install flask flask-cors -q
)

:: 检查flask_cors
python -c "import flask_cors" >nul 2>&1
if errorlevel 1 (
    pip install flask-cors -q
)

echo.
echo [启动] 正在启动AI服务器...
echo.
echo 访问地址:
echo   AI对话页面: http://localhost:5188/ai.html
echo   综合大屏:   http://localhost:5188/
echo.
echo 按 Ctrl+C 停止服务
echo ==========================================
echo.

python ai_server.py

pause
