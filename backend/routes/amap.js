const express = require('express');
const router = express.Router();
const axios = require('axios');
const { incAmap, getToday } = require('../lib/counters');

// ========== 配置区 ==========
const AMAP_KEY = process.env.AMAP_KEY || '29316b5952b698bdfb9940dfbeec76dc';
const AMAP_DAILY_LIMIT = 5000;
// ========== 配置区 ==========

// 山西11市 + 下辖区县
const SHANXI_CITIES = {
  '太原': { adcode: '140100', counties: ['小店区','迎泽区','杏花岭区','尖草坪区','万柏林区','晋源区','清徐县','阳曲县','娄烦县','古交市'] },
  '大同': { adcode: '140200', counties: ['平城区','云冈区','新荣区','云州区','阳高县','天镇县','广灵县','灵丘县','浑源县','左云县'] },
  '阳泉': { adcode: '140300', counties: ['城区','矿区','郊区','平定县','盂县'] },
  '长治': { adcode: '140400', counties: ['潞州区','上党区','屯留区','潞城区','襄垣县','平顺县','黎城县','壶关县','长子县','武乡县','沁县','沁源县'] },
  '晋城': { adcode: '140500', counties: ['城区','沁水县','阳城县','陵川县','泽州县','高平市'] },
  '朔州': { adcode: '140600', counties: ['朔城区','平鲁区','山阴县','应县','右玉县','怀仁市'] },
  '晋中': { adcode: '140700', counties: ['榆次区','太谷区','榆社县','左权县','和顺县','昔阳县','寿阳县','祁县','平遥县','灵石县','介休市'] },
  '运城': { adcode: '140800', counties: ['盐湖区','临猗县','万荣县','闻喜县','稷山县','新绛县','绛县','垣曲县','夏县','平陆县','芮城县','永济市','河津市'] },
  '忻州': { adcode: '140900', counties: ['忻府区','定襄县','五台县','代县','繁峙县','宁武县','静乐县','神池县','五寨县','岢岚县','河曲县','保德县','偏关县','原平市'] },
  '临汾': { adcode: '141000', counties: ['尧都区','曲沃县','翼城县','襄汾县','洪洞县','古县','安泽县','浮山县','吉县','乡宁县','大宁县','隰县','永和县','蒲县','汾西县','侯马市','霍州市'] },
  '吕梁': { adcode: '141100', counties: ['离石区','文水县','交城县','兴县','临县','柳林县','石楼县','岚县','方山县','中阳县','交口县','孝义市','汾阳市'] }
};

// 风力等级 → km/h 近似映射
const WIND_POWER_KMH = { '1':5, '2':10, '3':20, '4':30, '5':40, '6':50, '7':60, '8':75, '9':90 };

function windPowerToKmh(power) {
  if (!power) return 5;
  const nums = power.replace(/[^0-9]/g, '');
  if (nums.length >= 2) {
    return WIND_POWER_KMH[nums[1]] || 10;
  }
  return WIND_POWER_KMH[nums[0]] || 10;
}

function weatherEmoji(code) {
  const map = { '晴':'☀️','少云':'🌤️','多云':'⛅','阴':'☁️','小雨':'🌧️','中雨':'🌧️','大雨':'🌧️','暴雨':'⛈️','雷阵雨':'⛈️','小雪':'🌨️','中雪':'🌨️','大雪':'🌨️','雾':'🌫️','霾':'🌫️','沙尘':'💨' };
  return map[code] || '🌤️';
}

function weatherCodeDesc(code) {
  const map = {0:'晴',1:'少云',2:'多云',3:'阴',45:'雾',48:'雾凇',51:'小雨',53:'中雨',55:'大雨',61:'小雨',63:'中雨',65:'大雨',71:'小雪',73:'中雪',75:'大雪',80:'阵雨',81:'中阵雨',82:'大阵雨',95:'雷阵雨',96:'冰雹',99:'冰雹'};
  return map[code] || '晴';
}

function openMeteoEmoji(code) {
  if ([0].includes(code)) return '☀️';
  if ([1,2].includes(code)) return '🌤️';
  if ([3].includes(code)) return '⛅';
  if ([45,48].includes(code)) return '🌫️';
  if ([51,53,55,61,63,65,80,81,82].includes(code)) return '🌧️';
  if ([71,73,75].includes(code)) return '🌨️';
  if ([95,96,99].includes(code)) return '⛈️';
  return '🌤️';
}

// ========== GET /api/amap/usage ==========
router.get('/usage', (req, res) => {
  const today = getToday();
  const used = today.amap.amap || 0;
  res.json({
    todayCalls: used,
    remainingCalls: Math.max(0, AMAP_DAILY_LIMIT - used),
    limit: AMAP_DAILY_LIMIT,
    usedPercent: Math.round((used / AMAP_DAILY_LIMIT) * 100),
    lastCallTime: today.amap.lastAmap || null,
    keyStatus: 'configured',
    msg: '✅ 高德API已配置'
  });
});

// ========== POST /api/amap/weather (legacy) ==========
router.post('/weather', async (req, res) => {
  const { city, adcode } = req.body;
  const code = adcode || city;
  try {
    incAmap();
    const r = await axios.get('https://restapi.amap.com/v3/weather/weatherInfo', {
      params: { city: code, key: AMAP_KEY, extensions: 'all' },
      timeout: 5000
    });
    res.json(r.data);
  } catch (err) {
    res.status(500).json({ error: err.message, code: 'AMAP_ERROR' });
  }
});

// ========== GET /api/amap/weather/shanxi ==========
// 查询山西11市实时天气 → 返回 { cities: { 太原: { counties, forecast }, ... } }
router.get('/weather/shanxi', async (req, res) => {
  try {
    const entries = Object.entries(SHANXI_CITIES);
    const results = {};

    // 串行查询11个城市 (避免并发限流)
    for (const [cityName, info] of entries) {
      try {
        incAmap();
        const r = await axios.get('https://restapi.amap.com/v3/weather/weatherInfo', {
          params: { city: info.adcode, key: AMAP_KEY, extensions: 'base' },
          timeout: 5000
        });
        if (r.data && r.data.lives && r.data.lives[0]) {
          const live = r.data.lives[0];
          const temp = parseFloat(live.temperature_float || live.temperature) || 20;
          const hum = parseInt(live.humidity_float || live.humidity) || 30;
          const wind = windPowerToKmh(live.windpower);
          const weather = live.weather || '--';
          const risk = calcRisk(temp, hum, wind);

          // 为每个区县生成数据项
          const counties = info.counties.map(name => ({
            name, temp, hum, wind, risk, desc: weather, aqi: 50
          }));

          // 用 Open-Meteo 获取7天预报 (免费, 无配额)
          let forecast = null;
          try {
            forecast = await fetchOpenMeteoForecast(info.adcode);
          } catch(e) { /* ignore */ }

          results[cityName] = { counties, forecast: forecast || [] };
        }
      } catch (e) {
        console.warn(`[AMap] weather/shanxi failed for ${cityName}: ${e.message}`);
      }
      // 200ms 间隔防止限流
      await new Promise(r => setTimeout(r, 200));
    }

    res.json({ cities: results, updateTime: new Date().toISOString() });
  } catch (e) {
    console.error('[AMap] weather/shanxi error:', e);
    res.status(500).json({ error: e.message });
  }
});

// ========== GET /api/amap/weather/county?lat=xx&lon=xx ==========
// 用 Open-Meteo 获取区县级实时天气 + 7天预报 (免费, 不消耗高德额度)
// ===== GET /api/amap/weather/county?lat=xx&lon=xx&cityAdcode=xxxx =====
// 高德天气 API：先用逆地理编码获取区域码，再用天气接口（并发调用 base + all）
router.get('/weather/county', async (req, res) => {
  try {
    const { lat, lon, cityAdcode } = req.query;
    if (!lat || !lon) {
      return res.status(400).json({ error: 'lat 和 lon 参数必填' });
    }

    // 确定 adcode：优先用传入的城市码（来自 COUNTY_POINTS），否则用逆地理编码
    let adcode = cityAdcode || null;

    if (!adcode) {
      // 高德逆地理编码：经纬度 → adcode
      const regeoUrl = `https://restapi.amap.com/v3/geocode/regeo?key=${AMAP_KEY}&location=${lon},${lat}&extensions=base&output=json`;
      const regeoResp = await axios.get(regeoUrl, { timeout: 6000 });
      if (regeoResp.data?.status !== '1') {
        return res.status(502).json({ error: '逆地理编码失败', detail: regeoResp.data?.info || '' });
      }
      adcode = regeoResp.data?.regeocode?.addressComponent?.district?.[0]?.adcode
             || regeoResp.data?.regeocode?.addressComponent?.citycode;

      if (!adcode) {
        return res.status(502).json({ error: '无法获取区域码，请传入 cityAdcode 参数' });
      }
    }

    // 并发拉取：实时天气 + 7天预报
    const [baseResp, forecastResp] = await Promise.all([
      axios.get(`https://restapi.amap.com/v3/weather/weatherInfo?key=${AMAP_KEY}&city=${adcode}&extensions=base`, { timeout: 6000 }),
      axios.get(`https://restapi.amap.com/v3/weather/weatherInfo?key=${AMAP_KEY}&city=${adcode}&extensions=all`, { timeout: 6000 })
    ]);

    const live = baseResp.data?.lives?.[0];
    if (!live) {
      return res.status(502).json({ error: '高德实时天气无数据' });
    }

    const temp = Math.round(parseFloat(live.temperature_float || live.temperature || 20));
    const hum = parseInt(live.humidity_float || live.humidity || 50);
    const wind = windPowerToKmh(live.windpower || '≤3');
    const desc = live.weather || '--';

    const result = {
      temp, hum, wind, desc,
      risk: calcRisk(temp, hum, wind),
      reporttime: live.reporttime,
      province: live.province || '',
      city: live.city || ''
    };

    // 解析7天预报
    const casts = forecastResp.data?.forecasts?.[0]?.casts;
    if (casts && casts.length > 0) {
      result.forecast = casts.map(c => ({
        d: c.date || '',
        i: amapWeatherEmoji(c.dayweather || c.day_weather || ''),
        h: Math.round(parseFloat(c.daytemp_float || c.daytemp || 20)),
        l: Math.round(parseFloat(c.nighttemp_float || c.nighttemp || 10)),
        w: windPowerToKmh(c.daypower || c.day_power || '≤3')
      }));
    }

    res.json(result);
  } catch (e) {
    console.error('[AMap County] Error:', e.message || e);
    res.status(500).json({ error: '高德县级天气请求失败: ' + (e.message || '') });
  }
});

// 高德天气文字 → emoji
function amapWeatherEmoji(w) {
  if (!w) return '⛅';
  const w2 = String(w).trim();
  const map = {
    '晴': '\u2600\uFE0F', '多云': '\u26C5', '阴': '\u2601\uFE0F',
    '小雨': '\uD83C\uDF26\uFE0F', '中雨': '\uD83C\uDF26\uFE0F', '大雨': '\u26C8\uFE0F',
    '暴雨': '\u26C8\uFE0F', '雷阵雨': '\u26C8\uFE0F',
    '小雪': '\u2744\uFE0F', '中雪': '\u2744\uFE0F', '大雪': '\u2744\uFE0F',
    '雾': '\uD83C\uDF2B\uFE0F', '霾': '\uD83C\uDF2B\uFE0F',
    '扬沙': '\uD83C\uDF2B\uFE0F', '沙尘': '\uD83C\uDF2B\uFE0F',
    '阵雨': '\uD83C\uDF26\uFE0F', '阵雪': '\u2744\uFE0F'
  };
  return map[w2] || '\u26C1';
}

// Open-Meteo 风力等级 → km/h 数值（保持向后兼容，其他路由还在用）
function windPowerToKmh(power) {
  if (!power) return 5;
  const s = String(power).trim().replace('\u2264', '').replace('<', '');
  const n = parseInt(s);
  if (!isNaN(n)) return n;
  if (s.includes('\u5FAE\u98CE') || s === '微风') return 2;
  if (s.includes('\u5C0F\u4E3A') || s === '3-4') return 8;
  if (s.includes('\u4E3A') || s === '4-5') return 12;
  if (s.includes('\u5C3C') || s === '6-7') return 18;
  if (s.includes('\u5F3A') || s === '7-8') return 22;
  if (s.includes('\u5927') || s === '8-9') return 28;
  return 5;
}
// ========== 辅助函数 ==========
function calcRisk(temp, hum, wind) {
  let tScore = 0;
  if (temp < -5) tScore = 35;
  else if (temp < 0) tScore = 30;
  else if (temp < 5) tScore = 22;
  else if (temp > 35) tScore = 28;
  else if (temp > 30) tScore = 18;
  else tScore = Math.round(Math.abs(temp - 22) * 0.6);

  let hScore = 0;
  if (hum > 85) hScore = 35;
  else if (hum > 70) hScore = 25;
  else if (hum > 55) hScore = 15;
  else if (hum < 20) hScore = 12;

  let wScore = 0;
  if (wind > 40) wScore = 28;
  else if (wind > 25) wScore = 18;
  else if (wind > 15) wScore = 10;
  else if (wind > 8) wScore = 5;

  const total = tScore + hScore + wScore;
  return total >= 70 ? '高' : total >= 40 ? '中' : '低';
}

// Open-Meteo 城市预报 (通过城市 adcode → 城市中心经纬度)
const CITY_COORDS = {
  '140100': [37.87, 112.55],
  '140200': [40.09, 113.30],
  '140300': [37.86, 113.58],
  '140400': [36.20, 113.12],
  '140500': [35.49, 112.85],
  '140600': [39.33, 112.43],
  '140700': [37.69, 112.75],
  '140800': [35.03, 111.00],
  '140900': [38.42, 112.73],
  '141000': [36.09, 111.52],
  '141100': [37.52, 111.14]
};

async function fetchOpenMeteoForecast(adcode) {
  const coords = CITY_COORDS[adcode];
  if (!coords) return [];
  const [lat, lon] = coords;
  const url = 'https://api.open-meteo.com/v1/forecast'
    + '?latitude=' + lat + '&longitude=' + lon
    + '&daily=temperature_2m_max,temperature_2m_min,weather_code,wind_speed_10m_max'
    + '&timezone=Asia%2FShanghai&forecast_days=4';
  const r = await axios.get(url, { timeout: 8000 });
  const j = r.data;
  if (!j.daily || !j.daily.time) return [];
  const fc = [];
  for (let i = 0; i < Math.min(4, j.daily.time.length); i++) {
    fc.push({
      d: '',
      i: openMeteoEmoji(j.daily.weather_code[i]),
      h: Math.round(j.daily.temperature_2m_max[i]),
      l: Math.round(j.daily.temperature_2m_min[i]),
      w: Math.round(j.daily.wind_speed_10m_max ? (j.daily.wind_speed_10m_max[i] || 0) : 0)
    });
  }
  return fc;
}

module.exports = router;
