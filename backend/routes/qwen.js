const express = require('express');
const router = express.Router();
const axios = require('axios');
const { incQwen, addTokens, getToday, getMonth } = require('../lib/counters');

// ========== 配置区 ==========
const DASHSCOPE_KEY = process.env.DASHSCOPE_KEY || 'sk-abf81210d5bc4443b041f4ed25bfbe9d'; // 复用ai_server.py的key
const MODEL = 'qwen-turbo';
// ========== 配置区 ==========

const MONTH_LIMIT = 1000000; // qwen-turbo 每月100万Token免费额度

// GET /api/qwen/usage
router.get('/usage', (req, res) => {
  const today = getToday();
  const month = getMonth();
  const todayTokens = today.tokens.total || 0;
  const monthTokens = month.monthTokens || 0;
  const remaining = Math.max(0, MONTH_LIMIT - monthTokens);

  res.json({
    todayCalls: today.qwen.qwen || 0,
    todayTokens: todayTokens,
    monthlyCalls: month.monthCalls || 0,
    monthlyTokens: monthTokens,
    monthlyLimit: MONTH_LIMIT,
    remainingTokens: remaining,
    remainingPercent: Math.round((remaining / MONTH_LIMIT) * 10000) / 100,
    lastCallTime: today.qwen.lastQwen || null
  });
});

// POST /api/qwen/chat
router.post('/chat', async (req, res) => {
  const { messages } = req.body;
  if (!messages || !Array.isArray(messages)) {
    return res.status(400).json({ error: 'messages is required' });
  }

  try {
    const r = await axios.post(
      'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions',
      { model: MODEL, messages },
      {
        headers: {
          'Authorization': `Bearer ${DASHSCOPE_KEY}`,
          'Content-Type': 'application/json'
        },
        timeout: 15000
      }
    );

    incQwen();
    const tokens = r.data.usage?.total_tokens || 0;
    if (tokens > 0) addTokens(tokens);

    res.json(r.data);
  } catch (err) {
    const msg = err.response?.data || err.message;
    res.status(500).json({ error: msg });
  }
});

module.exports = router;
