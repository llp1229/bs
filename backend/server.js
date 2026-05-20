const express = require('express');
const path = require('path');
const cron = require('node-cron');

const app = express();
const PORT = process.env.PORT || 3000;

// 项目根目录（大屏前端文件所在）
const PROJECT_ROOT = path.join(__dirname, '..');

// 静态文件：管理后台 public/
app.use(express.static(path.join(__dirname, 'public')));
// 静态文件：大屏前端（modules/、zy/ 等）
app.use(express.static(PROJECT_ROOT));
app.use(express.json());

// ========== 路由 ==========
app.use('/api/amap',   require('./routes/amap'));
app.use('/api/qwen',   require('./routes/qwen'));
app.use('/api/server', require('./routes/server-stats'));
app.use('/api/weather',require('./routes/weather'));
app.use('/api/amap', require('./routes/weather-amap'));

// 健康检查
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', uptime: process.uptime(), ts: new Date().toISOString() });
});

// /dashboard → 管理后台
app.get('/dashboard', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'dashboard.html'));
});

// / → 古建监测大屏
app.get('/', (req, res) => {
  res.sendFile(path.join(PROJECT_ROOT, '古建监测大屏_v5.html'));
});

// ========== 每日零点重置 ==========
const { resetDailyCounters } = require('./lib/counters');
resetDailyCounters();
cron.schedule('0 0 * * *', () => {
  console.log('[CRON] Resetting daily counters at midnight');
  resetDailyCounters();
});

app.listen(PORT, '0.0.0.0', () => {
  console.log('Backend running: http://localhost:' + PORT);
  console.log('Dashboard:       http://localhost:' + PORT + '/dashboard');
});
