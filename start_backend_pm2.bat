@echo off
chcp 65001 >nul 2>&1
cd /d D:\bs\sxgjz\backend
npx pm2 list >nul 2>&1
if errorlevel 1 (npx pm2 resurrect)
npx pm2 list
echo.
npx pm2 logs
