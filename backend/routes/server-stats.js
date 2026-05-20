const express = require('express');
const router = express.Router();
const os = require('os');
const { exec } = require('child_process');
const util = require('util');
const execPromise = util.promisify(exec);

// ========== 阿里云算力包配置 ==========
// 阿里云 ECS 实例信息（手动填写或从环境变量读取）
const ECS_INFO = {
  instanceId: process.env.ECS_INSTANCE_ID || 'i-0x1234567890abcdef',
  publicIP: process.env.ECS_PUBLIC_IP || '47.121.201.141',
  // 算力包/流量包：阿里云真实数据（2026-05-10 22:24 更新）
  budget: {
    remaining: parseFloat(process.env.ECS_BUDGET_REMAIN || '783.42'),
    total: parseFloat(process.env.ECS_BUDGET_TOTAL || '850'),
    unit: '\u5143',
    note: '\u963f\u91cc\u4e91\u771f\u5b9e\u6570\u636e',
    hourlyRate: '1.03\u5143/\u5c0f\u65f6'
  },
  computePack: {
    remaining: parseFloat(process.env.ECS_COMPUTE_REMAIN || '850'),
    total: parseFloat(process.env.ECS_COMPUTE_TOTAL || '850'),
    unit: '\u5143',
    expireDate: '2025-03-18',
    note: '\u963f\u91cc\u4e91\u771f\u5b9e\u6570\u636e'
  },
  traffic: {
    used: parseFloat(process.env.ECS_TRAFFIC_USED || '1.0'),
    total: parseFloat(process.env.ECS_TRAFFIC_TOTAL || '20.0'),
    unit: 'GB',
    expireDate: '2025-05-18',
    note: '\u963f\u91cc\u4e91\u771f\u5b9e\u6570\u636e'
  }
};
// ========== 配置区 ==========

function getCpuUsage() {
  return new Promise((resolve) => {
    const cpus = os.cpus();
    let totalIdle = 0, totalTick = 0;
    cpus.forEach(cpu => {
      for (const type in cpu.times) totalTick += cpu.times[type];
      totalIdle += cpu.times.idle;
    });
    const idle = totalIdle / cpus.length;
    const total = totalTick / cpus.length;
    const usage = total > 0 ? Math.round((1 - idle / total) * 10000) / 100 : 0;
    resolve(usage);
  });
}

function getMemUsage() {
  const total = os.totalmem();
  const free = os.freemem();
  return Math.round(((total - free) / total) * 10000) / 100;
}

function getDiskUsage() {
  return new Promise((resolve) => {
    if (process.platform === 'win32') {
      // Windows: PowerShell 获取 C: 盘使用率
      exec('powershell -NoProfile -Command "(Get-PSDrive C).Used; (Get-PSDrive C).Free"', (err, stdout) => {
        if (err || !stdout.trim()) return resolve(null);
        const lines = stdout.trim().split(/\r?\n/).filter(Boolean);
        const used = parseFloat(lines[0]);
        const free = parseFloat(lines[1]);
        if (isNaN(used) || isNaN(free) || (used + free) === 0) return resolve(null);
        resolve(Math.round(used / (used + free) * 10000) / 100);
      });
    } else {
      exec("df -h / | tail -1 | awk '{print $5}'", (err, stdout) => {
        if (err || !stdout.trim()) return resolve(null);
        resolve(parseInt(stdout.trim().replace('%', '')));
      });
    }
  });
}

function getLoadAvg() {
  try { return os.loadavg().map(v => Math.round(v * 100) / 100); }
  catch { return [0, 0, 0]; }
}

// GET /api/server/status
router.get('/status', async (req, res) => {
  const [cpu, mem, disk, load] = await Promise.all([
    getCpuUsage(),
    getMemUsage(),
    getDiskUsage(),
    getLoadAvg()
  ]);

  const cpuCores = os.cpus().length;
  const totalMem = Math.round(os.totalmem() / 1024 / 1024 / 1024 * 100) / 100;
  const freeMem = Math.round(os.freemem() / 1024 / 1024 / 1024 * 100) / 100;

  res.json({
    cpu: { usage: cpu, cores: cpuCores, load: load },
    memory: { usage: mem, total: totalMem, free: freeMem, unit: 'GB' },
    disk: { usage: disk, mount: '/', unit: '%' },
    uptime: Math.round(os.uptime()),
    hostname: os.hostname(),
    platform: os.platform() + ' ' + os.release(),
    publicIP: ECS_INFO.publicIP,
    status: 'running',
    budget: ECS_INFO.budget,
    computePack: ECS_INFO.computePack,
    traffic: ECS_INFO.traffic,
    ts: new Date().toISOString()
  });
});

module.exports = router;
