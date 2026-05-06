@echo off
chcp 65001 >nul
title 古建监测大屏 - 服务启动

echo ============================================
echo   古建监测大屏 v5 - 服务启动
echo ============================================
echo.

cd /d D:\bs\sxgjz

echo [1/2] 启动 AI 服务 (端口 5188)...
start "AI服务" cmd /c "python ai_server.py"
timeout /t 3 /nobreak >nul
echo         已启动

echo [2/2] 启动 HTTP 静态服务器 (端口 8080)...
start "HTTP服务器" cmd /c "python serve.py"
timeout /t 2 /nobreak >nul
echo         已启动

echo.
echo ============================================
echo   ✅ 全部服务已启动
echo.
echo   浏览器打开：http://localhost:8080/古建监测大屏_v5.html
echo ============================================
echo.
pause
