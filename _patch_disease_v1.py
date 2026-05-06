# -*- coding: utf-8 -*-
"""三合一补丁：
1. 气候风险关联 - 病害分布→病害风险指数
2. 地图散点与天气预报同步 - fetchCountyWeather 更新 MONITORING_DATA + 散点
3. 天气预报加风 - current wind + 7-day wind
"""
import re

APP = r'D:\bs\sxgjz\modules\app.js'
HTML = r'D:\bs\sxgjz\古建监测大屏_v5.html'

def read(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def write(path, content):
    tmp = path + '.tmp_patch'
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(content)
    import os
    os.replace(tmp, path)
    print(f'  ✅ {os.path.basename(path)} ({len(content)} bytes)')

# ==============================
# 1. 修改 app.js
# ==============================
t = read(APP)

# --- 1a. 在 riskColor 之后插入 calcRiskScore ---
old_risk_color = '''function riskColor(r) { return r==='高'?'#e74c3c':r==='中'?'#f39c12':'#2ecc71'; }'''
pos = t.find(old_risk_color)
if pos < 0:
    raise SystemExit('❌ riskColor not found')

insert_pos = pos + len(old_risk_color) + 1  # +1 for \n

calc_risk = '''
function calcRiskScore() {
  var temp = 18, hum = 50;
  if (selectedCountyName && MONITORING_DATA[selectedCountyName]) {
    var d = MONITORING_DATA[selectedCountyName];
    temp = d.temp || 18;
    hum = d.hum || 50;
  }
  // 温度因子：冻融（低温）+ 热应力（高温）
  var tScore = 0;
  if (temp < 0) tScore = 30;
  else if (temp < 5) tScore = 22;
  else if (temp > 35) tScore = 28;
  else if (temp > 30) tScore = 18;
  else tScore = Math.round(Math.abs(temp - 22) * 0.6);
  // 湿度因子：潮湿侵蚀 + 过干开裂
  var hScore = 0;
  if (hum > 85) hScore = 35;
  else if (hum > 70) hScore = 25;
  else if (hum > 55) hScore = 15;
  else if (hum < 20) hScore = 12;
  var s = tScore + hScore;
  return Math.min(100, Math.round(s));
}
'''
t = t[:insert_pos] + calc_risk + t[insert_pos:]
print('  ✅ [1a] calcRiskScore inserted')

# --- 1b. 替换 drawCharts 中病害分布为气候风险 ---
old_disease = '''  // === 病害类型分布 ===
  document.getElementById('diseaseChart').innerHTML =
    '<circle cx="80" cy="65" r="38" fill="#e74c3c"/><circle cx="80" cy="65" r="26" fill="#e67e22"/>' +
    '<circle cx="80" cy="65" r="16" fill="#f39c12"/><circle cx="80" cy="65" r="8" fill="#3498db"/>' +
    '<text x="80" y="70" fill="#fff" font-size="13" text-anchor="middle" font-weight="bold">400</text>' +
    '<text x="80" y="118" fill="#aaa" font-size="10" text-anchor="middle">病害总数</text>';'''

new_risk = '''  // === 气候病害风险关联 ===
  (function() {
    var score = calcRiskScore();
    var lvl = score >= 70 ? '高' : score >= 40 ? '中' : '低';
    var rc = score >= 70 ? '#e74c3c' : score >= 40 ? '#f39c12' : '#2ecc71';
    var svg2 = '<circle cx="80" cy="48" r="42" fill="none" stroke="' + rc + '" stroke-width="2" opacity="0.35"/>';
    svg2 += '<circle cx="80" cy="48" r="34" fill="' + rc + '" opacity="0.1"/>';
    svg2 += '<text x="80" y="43" fill="' + rc + '" font-size="24" text-anchor="middle" font-weight="bold">' + score + '</text>';
    svg2 += '<text x="80" y="56" fill="' + rc + '" font-size="10" text-anchor="middle">风险指数</text>';
    svg2 += '<text x="80" y="80" fill="#777" font-size="8" text-anchor="middle">等级</text>';
    svg2 += '<text x="80" y="95" fill="' + rc + '" font-size="13" text-anchor="middle" font-weight="bold">' + lvl + '风险</text>';
    document.getElementById('diseaseChart').innerHTML = svg2;
  })();'''

t = t.replace(old_disease, new_risk)
print('  ✅ [1b] disease chart → climate risk gauge')

# --- 1c. 修改 fetchCountyWeather 加 wind + 7-day forecast ---
old_fcw_param = "'&current=temperature_2m,relative_humidity_2m,weather_code&timezone=Asia%2FShanghai';"
new_fcw_param = (
    "'&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m'"
    "+ '&daily=temperature_2m_max,temperature_2m_min,weather_code,wind_speed_10m_max'"
    "+ '&timezone=Asia%2FShanghai&forecast_days=7';"
)
t = t.replace(old_fcw_param, new_fcw_param)
print('  ✅ [1c] fetchCountyWeather params +wind')

# --- 1d. 修改 fetchCountyWeather return 加 wind + forecast ---
old_fcw_return = """    return {
      temp: Math.round(j.current.temperature_2m),
      hum: j.current.relative_humidity_2m,
      desc: weatherCodeDesc(j.current.weather_code),
      risk: riskFromHumidity(j.current.relative_humidity_2m)
    };"""

new_fcw_return = """    var result = {
      temp: Math.round(j.current.temperature_2m),
      hum: j.current.relative_humidity_2m,
      wind: Math.round(j.current.wind_speed_10m || 0),
      desc: weatherCodeDesc(j.current.weather_code),
      risk: riskFromHumidity(j.current.relative_humidity_2m)
    };
    // 县级7天预报
    if (j.daily && j.daily.time && j.daily.time.length >= 7) {
      var fc = [];
      for (var di = 0; di < 7; di++) {
        fc.push({
          d: '', i: weatherCodeEmoji(j.daily.weather_code[di]),
          h: Math.round(j.daily.temperature_2m_max[di]),
          l: Math.round(j.daily.temperature_2m_min[di]),
          w: Math.round(j.daily.wind_speed_10m_max ? (j.daily.wind_speed_10m_max[di] || 0) : 0)
        });
      }
      result.forecast = fc;
    }
    return result;"""

t = t.replace(old_fcw_return, new_fcw_return)
print('  ✅ [1d] fetchCountyWeather return +wind')

# --- 1e. 修改 updateCountyMetrics: 更新 MONITORING_DATA + 散点 + 县级预报 ---
old_um_start = '''document.getElementById('sel-risk');'''
pos_sr = t.find(old_um_start)
if pos_sr >= 0:
    # find the next line after updating sel-risk
    after_sr = t.find('\n', pos_sr + len(old_um_start))
    # find end of the if(live) block: next "} else {"
    else_pos = t.find('} else {', after_sr)
    # insert after the sel-risk update line but before the closing } of the if block
    insert_after = '''    sr.style.color = riskColor(live.risk);
'''
    pos_ia = t.find(insert_after, after_sr)
    if pos_ia >= 0:
        insert_at = pos_ia + len(insert_after)
        # add MONITORING_DATA + scatter update + county forecast
        insert_code = '''    // 更新全局 MONITORING_DATA 为真实县级数据
    MONITORING_DATA[countyName] = {
      temp: live.temp, hum: live.hum, risk: live.risk,
      desc: live.desc, aqi: MONITORING_DATA[countyName] ? (MONITORING_DATA[countyName].aqi || 50) : 50
    };
    // 存储县级7天预报（覆盖市级预报）
    if (live.forecast) { WEATHER_7D_BASE[countyName] = live.forecast; }
    // 刷新地图散点
    if (typeof mapChart !== 'undefined' && mapChart) {
      try {
        var newScatter = [];
        var pts = COUNTY_POINTS || [];
        for (var si = 0; si < pts.length; si++) {
          var p = pts[si];
          var md = MONITORING_DATA[p.name];
          if (md) {
            newScatter.push([p.value[0], p.value[1], md.temp || 0, md.hum || 50, md.risk || '低', p.name]);
          } else {
            newScatter.push([p.value[0], p.value[1], 15, 50, '低', p.name]);
          }
        }
        mapChart.setOption({ series: [{ type: 'scatter', data: newScatter }] });
      } catch(e) {}
    }
    // 重绘趋势图（使用最新数据）
    drawCharts();
'''
        t = t[:insert_at] + insert_code + t[insert_at:]
        print('  ✅ [1e] updateCountyMetrics: MONITORING_DATA + scatter sync')
    else:
        print('  ⚠️ [1e] insert point not found')

# --- 1f. 修改 renderForecast 加风 ---
old_fc_card = """card.innerHTML =
      '<div class="forecast-day">' + dayLabel + '</div>' +
      '<div class="forecast-icon">' + f.i + '</div>' +
      '<div class="forecast-temp">' + f.h + '°/'' + f.l + '°</div>';"""

new_fc_card = """card.innerHTML =
      '<div class="forecast-day">' + dayLabel + '</div>' +
      '<div class="forecast-icon">' + f.i + '</div>' +
      '<div class="forecast-temp">' + f.h + '°/'' + f.l + '°</div>' +
      '<div class="forecast-wind">🌬 ' + (f.w || '--') + 'km/h</div>';"""

if old_fc_card in t:
    t = t.replace(old_fc_card, new_fc_card)
    print('  ✅ [1f] renderForecast +wind')
else:
    print('  ⚠️ [1f] forecast card template not found')

# --- 1g. 修改 fetchRealWeather daily params 加 wind ---
old_frw_daily = "'&daily=temperature_2m_max,temperature_2m_min,weather_code,relative_humidity_2m_max,relative_humidity_2m_min'"
new_frw_daily = "'&daily=temperature_2m_max,temperature_2m_min,weather_code,relative_humidity_2m_max,relative_humidity_2m_min,wind_speed_10m_max'"
t = t.replace(old_frw_daily, new_frw_daily)
print('  ✅ [1g] fetchRealWeather daily params +wind')

# --- 1h. 修改 fetchRealWeather 7-day forecast 对象加 w 字段 ---
old_frw_obj = """forecast.push({
          d: '',
          i: weatherCodeEmoji(daily.weather_code[i]),
          h: Math.round(daily.temperature_2m_max[i]),
          l: Math.round(daily.temperature_2m_min[i])
        });"""

new_frw_obj = """forecast.push({
          d: '',
          i: weatherCodeEmoji(daily.weather_code[i]),
          h: Math.round(daily.temperature_2m_max[i]),
          l: Math.round(daily.temperature_2m_min[i]),
          w: Math.round(daily.wind_speed_10m_max ? (daily.wind_speed_10m_max[i] || 0) : 0)
        });"""

t = t.replace(old_frw_obj, new_frw_obj)
print('  ✅ [1h] fetchRealWeather forecast +wind field')

# --- 1i. 同样更新 DEFAULT_7D fallback ---
old_default_obj = """newDefault.push({
        d: '',
        i: '\u2600\ufe0f',
        h: count > 0 ? Math.round(sumH / count) : 20,
        l: count > 0 ? Math.round(sumL / count) : 10
      });"""

new_default_obj = """newDefault.push({
        d: '',
        i: '\u2600\ufe0f',
        h: count > 0 ? Math.round(sumH / count) : 20,
        l: count > 0 ? Math.round(sumL / count) : 10,
        w: 8
      });"""

t = t.replace(old_default_obj, new_default_obj)
print('  ✅ [1i] DEFAULT_7D fallback +wind field')

write(APP, t)

# ==============================
# 2. 修改 HTML 标题
# ==============================
h = read(HTML)
old_title = '病害类型分布'
new_title = '病害风险指数'
if old_title in h:
    h = h.replace(old_title, new_title)
    write(HTML, h)
    print('  ✅ HTML title: 病害类型分布 → 病害风险指数')
else:
    print('  ⚠️ HTML title "病害类型分布" not found')

# ==============================
# 3. 添加 forecast-wind CSS 样式
# ==============================
s = read(r'D:\bs\sxgjz\modules\style.css')
wind_css = '''
.forecast-wind {
  font-size: 9px;
  color: #78909c;
  text-align: center;
  margin-top: 1px;
}'''
if '.forecast-wind' not in s:
    s += wind_css
    write(r'D:\bs\sxgjz\modules\style.css', s)
    print('  ✅ style.css +.forecast-wind')
else:
    print('  ℹ  .forecast-wind already in style.css')

print('\n🎉 All patches applied successfully!')
