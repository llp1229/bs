#!/bin/bash
# ====================================================
# 古建监测系统 — 阿里云 ECS 后端部署脚本
# 在 ECS 上运行: bash deploy.sh
# ====================================================

set -e
echo "========================================="
echo "  山西古建监测 后端部署脚本"
echo "========================================="

PROJECT_DIR="/var/www/sxgjz-backend"
BACKEND_PORT=3000

# 1. 安装 Node.js 16+ (如未安装)
echo "[1/5] Checking Node.js..."
if ! command -v node &> /dev/null; then
    echo "  Installing Node.js 18..."
    curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
    sudo apt-get install -y nodejs
fi
echo "  Node.js: $(node -v)"
echo "  npm:     $(npm -v)"

# 2. 创建项目目录
echo "[2/5] Setting up project..."
sudo mkdir -p $PROJECT_DIR
sudo cp -r ./* $PROJECT_DIR/
sudo chown -R $USER:$USER $PROJECT_DIR

# 3. 安装依赖
echo "[3/5] Installing npm dependencies..."
cd $PROJECT_DIR
npm install --production

# 4. 配置环境变量
echo "[4/5] Configuring environment..."
cat > $PROJECT_DIR/.env << 'ENVEOF'
PORT=3000
# 高德API Key (https://console.amap.com/dev/key/app)
AMAP_KEY=YOUR_AMAP_KEY_HERE
# 通义千问 Key (复用ai_server.py的)
DASHSCOPE_KEY=sk-abf81210d5bc4443b041f4ed25bfbe9d
# 算力包配置
ECS_BUDGET_REMAIN=750
ECS_BUDGET_TOTAL=850
ECS_COMPUTE_REMAIN=200
ECS_COMPUTE_TOTAL=200
ECS_TRAFFIC_USED=1.0
ECS_TRAFFIC_TOTAL=20
ENVEOF
echo "  .env created — EDIT WITH YOUR AMAP_KEY!"

# 5. systemd 服务
echo "[5/5] Creating systemd service..."
sudo tee /etc/systemd/system/sxgjz-backend.service > /dev/null << SVCEOF
[Unit]
Description=Shanxi Ancient Building Monitoring Backend
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_DIR
Environment=NODE_ENV=production
Environment=PORT=$BACKEND_PORT
ExecStart=/usr/bin/node server.js
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
SVCEOF

sudo systemctl daemon-reload
sudo systemctl enable sxgjz-backend
sudo systemctl start sxgjz-backend

sleep 2
if sudo systemctl is-active --quiet sxgjz-backend; then
    echo ""
    echo "========================================="
    echo "  DEPLOY SUCCESS!"
    echo "========================================="
    echo "  Service: sudo systemctl status sxgjz-backend"
    echo "  Logs:    sudo journalctl -u sxgjz-backend -f"
    echo "  API:     http://localhost:$BACKEND_PORT/api/health"
    echo "  Dashboard: http://localhost:$BACKEND_PORT/dashboard"
else
    echo ""
    echo " SERVICE FAILED — check logs:"
    echo "  sudo journalctl -u sxgjz-backend -n 30"
fi
