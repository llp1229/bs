const express = require('express');
const router = express.Router();
const axios = require('axios');
const { incAmap } = require('../lib/counters');

// 山西省11个地级市
const SHANXI_CITIES = [
  { name: '\u592a\u539f', adcode: '140100', lat: 37.87, lon: 112.55 },
  { name: '\u5927\u540c', adcode: '140200', lat: 40.08, lon: 113.30 },
  { name: '\u9633\u6cc9', adcode: '140300', lat: 37.86, lon: 113.58 },
  { name: '\u957f\u6cbb', adcode: '140400', lat: 36.20, lon: 113.12 },
  { name: '\u664b\u57ce', adcode: '140500', lat: 35.49, lon: 112.85 },
  { name: '\u6714\u5dde', adcode: '140600', lat: 39.33, lon: 112.43 },
  { name: '\u5ffb\u5dde', adcode: '140900', lat: 38.42, lon: 112.73 },
  { name: '\u5415\u6881', adcode: '141100', lat: 37.52, lon: 111.14 },
  { name: '\u664b\u4e2d', adcode: '140700', lat: 37.68, lon: 112.75 },
  { name: '\u4e34\u6c7e', adcode: '141000', lat: 36.08, lon: 111.52 },
  { name: '\u8fd0\u57ce', adcode: '140800', lat: 35.02, lon: 111.01 }
];

// 使用 Open-Meteo（免费，无需Key）获取天气
// 逐个请求（避免并发限流），最多重试1次
function delay(ms) { return new Promise(r => setTimeout(r, ms)); }

async function getCityWeather(city) {
  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      const url = `https://api.open-meteo.com/v1/forecast?latitude=${city.lat}&longitude=${city.lon}&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m&daily=weather_code,temperature_2m_max,temperature_2m_min&timezone=Asia%2FShanghai&forecast_days=1`;
      const r = await axios.get(url, { timeout: 8000 });
      const cur = r.data.current || {};
      return {
        city: city.name,
        adcode: city.adcode,
        temp: cur.temperature_2m ?? '--',
        humidity: cur.relative_humidity_2m ?? '--',
        wind: cur.wind_speed_10m ?? '--',
        weatherCode: cur.weather_code ?? 0,
        weather: weatherCodeToDesc(cur.weather_code ?? 0),
        updateTime: new Date().toISOString()
      };
    } catch {
      if (attempt === 0) await delay(500); // 重试前等待500ms
    }
  }
  return { city: city.name, adcode: city.adcode, temp: '--', humidity: '--', wind: '--', weatherCode: 0, weather: '--', updateTime: new Date().toISOString() };
}

function weatherCodeToDesc(code) {
  const map = {
    0: '\u6674', 1: '\u5c11\u4e91', 2: '\u591a\u4e91', 3: '\u9634',
    45: '\u972d', 48: '\u8f75\u973e',
    51: '\u8f7b\u7f0e', 53: '\u4e2d\u7f0e', 55: '\u91cd\u7f0e',
    61: '\u5c0f\u96e8', 63: '\u4e2d\u96e8', 65: '\u5927\u96e8',
    71: '\u5c0f\u96ea', 73: '\u4e2d\u96ea', 75: '\u5927\u96ea',
    80: '\u5c0f\u6674\u8f6c\u96e8', 81: '\u4e2d\u6674\u8f6c\u96e8', 82: '\u5927\u6674\u8f6c\u96e8',
    95: '\u95f9\u79bd\u6620',
    96: '\u5c0f\u96ea\u5e26\u96ea\u7b94', 99: '\u5927\u96ea\u5e26\u96ea\u7b94'
  };
  return map[code] || '\u672a\u77e5';
}

// GET /api/weather/shanxi-pie
// 逐个请求避免 Open-Meteo 免费API并发限流
router.get('/shanxi-pie', async (req, res) => {
  const results = [];
  for (const c of SHANXI_CITIES) {
    results.push(await getCityWeather(c));
    await delay(200); // 每个请求间隔200ms
  }

  // 聚合饼图数据
  const pie = {};
  results.forEach(r => {
    const w = r.weather;
    if (w && w !== '--') pie[w] = (pie[w] || 0) + 1;
  });

  const weatherPie = Object.entries(pie).map(([name, value]) => ({ name, value }));

  res.json({
    cities: results,
    weatherPie,
    updateTime: new Date().toISOString()
  });
});

// GET /api/weather/:city
router.get('/:city', async (req, res) => {
  const name = decodeURIComponent(req.params.city);
  const city = SHANXI_CITIES.find(c => c.name === name);
  if (!city) return res.status(404).json({ error: '\u57ce\u5e02\u672a\u627e\u5230' });
  const data = await getCityWeather(city);
  res.json(data);
});

module.exports = router;
