const express = require('express');
const router = express.Router();
const axios = require('axios');
const { incAmap } = require('../lib/counters');

const AMAP_KEY = process.env.AMAP_KEY || '29316b5952b698bdfb9940dfbeec76dc';

// 山西11地级市
const SHANXI_CITIES = [
  { name: '太原', adcode: '140100', lat: 37.87, lon: 112.55,
    counties: ['小店区','迎泽区','杏花岭区','尖草坪区','万柏林区','晋源区','清徐县','阳曲县','娄烦县','古交市'] },
  { name: '大同', adcode: '140200', lat: 40.08, lon: 113.30,
    counties: ['平城区','云冈区','新荣区','阳高县','天镇县','广灵县','灵丘县','浑源县','左云县','云州区'] },
  { name: '阳泉', adcode: '140300', lat: 37.86, lon: 113.58,
    counties: ['城区','矿区','郊区','平定县','盂县'] },
  { name: '长治', adcode: '140400', lat: 36.20, lon: 113.12,
    counties: ['潞州区','上党区','屯留区','潞城区','襄垣县','平顺县','黎城县','壶关县','长子县','武乡县','沁县','沁源县'] },
  { name: '晋城', adcode: '140500', lat: 35.49, lon: 112.85,
    counties: ['城区','沁水县','阳城县','陵川县','泽州县','高平市'] },
  { name: '朔州', adcode: '140600', lat: 39.33, lon: 112.43,
    counties: ['朔城区','平鲁区','山阴县','应县','右玉县','怀仁市'] },
  { name: '忻州', adcode: '140900', lat: 38.42, lon: 112.73,
    counties: ['忻府区','定襄县','五台县','代县','繁峙县','宁武县','静乐县','神池县','五寨县','岢岚县','河曲县','保德县','偏关县','原平市'] },
  { name: '吕梁', adcode: '141100', lat: 37.52, lon: 111.14,
    counties: ['离石区','文水县','交城县','兴县','临县','柳林县','石楼县','岚县','方山县','中阳县','交口县','孝义市','汾阳市'] },
  { name: '晋中', adcode: '140700', lat: 37.68, lon: 112.75,
    counties: ['榆次区','太谷区','榆社县','左权县','和顺县','昔阳县','寿阳县','祁县','平遥县','灵石县','介休市'] },
  { name: '临汾', adcode: '141000', lat: 36.08, lon: 111.52,
    counties: ['尧都区','曲沃县','翼城县','襄汾县','洪洞县','古县','安泽县','浮山县','吉县','乡宁县','大宁县','隰县','永和县','蒲县','汾西县','侯马市','霍州市'] },
  { name: '运城', adcode: '140800', lat: 35.02, lon: 111.01,
    counties: ['盐湖区','临猗县','万荣县','闻喜县','稷山县','新绛县','绛县','垣曲县','夏县','平陆县','芮城县','永济市','河津市'] }
];

// wind power -> km/h (取中间值)
function windToKmh(power) {
  if (!power) return 10;
  var p = String(power).replace(/[^0-9]/g,'');
  var n = parseInt(p) || 3;
  var map = {0:2, 1:5, 2:10, 3:16, 4:24, 5:34, 6:44, 7:54, 8:64, 9:74, 10:84, 11:94, 12:104};
  return map[n] || 16;
}

// weather desc -> emoji
function weatherEmojiAMap(w) {
  if (!w) return '☀️';
  if (w.indexOf('晴')>-1) return '☀️';
  if (w.indexOf('云')>-1) return '⛅';
  if (w.indexOf('阴')>-1) return '☁️';
  if (w.indexOf('雾')>-1 || w.indexOf('霾')>-1) return '🌫️';
  if (w.indexOf('雨')>-1) return w.indexOf('大')>-1?'🌧️':'🌧️';
  if (w.indexOf('雪')>-1) return '🌨️';
  return '🌤️';
}

function riskLabel(temp, hum, wind) {
  var tS=0; if(temp<-5)tS=35;else if(temp<0)tS=30;else if(temp<5)tS=22;else if(temp>35)tS=28;else if(temp>30)tS=18;else tS=Math.round(Math.abs(temp-22)*0.6);
  var hS=0; if(hum>85)hS=35;else if(hum>70)hS=25;else if(hum>55)hS=15;else if(hum<20)hS=12;
  var wS=0; if(wind>40)wS=28;else if(wind>25)wS=18;else if(wind>15)wS=10;else if(wind>8)wS=5;
  var t=tS+hS+wS; return t>=70?'高':t>=40?'中':'低';
}

// GET /api/amap/weather/shanxi
// 返回所有11城市实时+预报（供大屏 fetchRealWeather 使用）
router.get('/shanxi', async (req, res) => {
  var result = {};
  for (var ci = 0; ci < SHANXI_CITIES.length; ci++) {
    var city = SHANXI_CITIES[ci];
    var cityData = { live: null, forecast: [], counties: [] };

    try {
      // live
      var lr = await axios.get('https://restapi.amap.com/v3/weather/weatherInfo', {
        params: { city: city.adcode, key: AMAP_KEY, extensions: 'base' }, timeout: 6000
      });
      incAmap();
      var live = (lr.data && lr.data.lives && lr.data.lives[0]) ? lr.data.lives[0] : null;

      if (live) {
        var t = parseFloat(live.temperature) || 20;
        var h = parseFloat(live.humidity) || 50;
        var w = windToKmh(live.windpower);
        cityData.live = { temp: Math.round(t), humidity: h, wind: w, weather: live.weather || '晴' };

        // generate county-level data
        city.counties.forEach(function(cn, idx) {
          var v = (idx % 3) - 1;
          var ct = Math.round(t + v);
          var ch = Math.min(95, Math.max(10, h + v * 2));
          var cw = Math.round(w + v * 2);
          cityData.counties.push({
            name: cn, temp: ct, hum: ch, wind: cw,
            risk: riskLabel(ct, ch, cw),
            desc: live.weather, aqi: 50
          });
        });
      }
    } catch(e) { console.warn('[AMap live] ' + city.name + ' fail: ' + (e.message||'')); }

    try {
      // forecast
      var fr = await axios.get('https://restapi.amap.com/v3/weather/weatherInfo', {
        params: { city: city.adcode, key: AMAP_KEY, extensions: 'all' }, timeout: 6000
      });
      incAmap();
      var casts = (fr.data && fr.data.forecasts && fr.data.forecasts[0] && fr.data.forecasts[0].casts)
        ? fr.data.forecasts[0].casts : [];

      casts.forEach(function(c) {
        cityData.forecast.push({
          d: c.date || '',
          i: weatherEmojiAMap(c.dayweather),
          h: parseInt(c.daytemp) || 20,
          l: parseInt(c.nighttemp) || 10,
          w: windToKmh(c.daypower)
        });
      });
    } catch(e) { console.warn('[AMap forecast] ' + city.name + ' fail: ' + (e.message||'')); }

    result[city.name] = cityData;
    // stagger requests
    if (ci < SHANXI_CITIES.length - 1) await new Promise(r => setTimeout(r, 150));
  }

  res.json({ cities: result, timestamp: new Date().toISOString() });
});

// GET /api/amap/weather/county?lat=LAT&lon=LON
// 通过高德逆地理编码获取adcode，再获取天气
router.get('/county', async (req, res) => {
  var lat = parseFloat(req.query.lat);
  var lon = parseFloat(req.query.lon);
  if (isNaN(lat) || isNaN(lon)) return res.status(400).json({ error: '需要 lat 和 lon' });

  try {
    // reverse geocode
    var rg = await axios.get('https://restapi.amap.com/v3/geocode/regeo', {
      params: { location: lon.toFixed(6) + ',' + lat.toFixed(6), key: AMAP_KEY, output: 'json' }, timeout: 5000
    });
    incAmap();

    var adcode = '';
    if (rg.data && rg.data.regeocode && rg.data.regeocode.addressComponent) {
      var ac = rg.data.regeocode.addressComponent;
      adcode = ac.adcode || '';
    }

    if (!adcode) return res.status(500).json({ error: 'geocode fail' });

    // live weather
    var lr = await axios.get('https://restapi.amap.com/v3/weather/weatherInfo', {
      params: { city: adcode, key: AMAP_KEY, extensions: 'base' }, timeout: 5000
    });
    incAmap();
    var live = (lr.data && lr.data.lives && lr.data.lives[0]) ? lr.data.lives[0] : null;
    if (!live) return res.status(500).json({ error: 'no live data' });

    var t = Math.round(parseFloat(live.temperature) || 20);
    var h = parseFloat(live.humidity) || 50;
    var w = windToKmh(live.windpower);
    var risk = riskLabel(t, h, w);

    // forecast
    var forecast = [];
    try {
      var fr = await axios.get('https://restapi.amap.com/v3/weather/weatherInfo', {
        params: { city: adcode, key: AMAP_KEY, extensions: 'all' }, timeout: 5000
      });
      incAmap();
      var casts = (fr.data && fr.data.forecasts && fr.data.forecasts[0] && fr.data.forecasts[0].casts)
        ? fr.data.forecasts[0].casts : [];
      casts.forEach(function(c) {
        forecast.push({
          d: c.date || '', i: weatherEmojiAMap(c.dayweather),
          h: parseInt(c.daytemp) || 20, l: parseInt(c.nighttemp) || 10,
          w: windToKmh(c.daypower)
        });
      });
    } catch(e) {}

    res.json({
      temp: t, hum: h, wind: w, risk: risk, desc: live.weather,
      forecast: forecast, timestamp: new Date().toISOString()
    });
  } catch(e) {
    res.status(500).json({ error: e.message || 'fail' });
  }
});

module.exports = router;
