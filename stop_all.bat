@echo off
chcp 65001 >nul
title 山西古建筑监测系统 - 停止服务

echo 正在停止所有服务...

:: 停止 AI 服务 (端口 5188)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :5188 ^| findstr LISTENING') do (
    echo 停止 AI 服务 (PID: %%a)...
    taskkill /f /pid %%a 2>nul
)

:: 停止 Streamlit (端口 8501)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8501 ^| findstr LISTENING') do (
    echo 停止管理后台 (PID: %%a)...
    taskkill /f /pid %%a 2>nul
)

echo 全部服务已停止.
pause
