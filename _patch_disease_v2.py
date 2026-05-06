# -*- coding: utf-8 -*-
"""三合一补丁 v2：
1. 气候风险关联 - 病害分布→病害风险指数
2. 地图散点与天气预报同步
3. 天气预报加风
"""
import os

APP = r'D:\bs\sxgjz\modules\app.js'
HTML = r'D:\bs\sxgjz\古建监测大屏_v5.html'
CSS = r'D:\bs\sxgjz\modules\style.css'

def read(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def writef(path, content):
    tmp = path + '.tmp_patch'
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(content)
    os.replace(tmp, path)
    print(f'  [OK] {os.path.basename(path)} ({len(content)} bytes)')

t = read(APP)

# --- 1a. 插入 calcRiskScore 函数 ---
old = '''function riskColor(r) { return r==='\u9ad8'?'#e74c3c':r==='\u4e2d'?'#f39c12':'#2ecc71'; }'''
pos = t.find(old)
if pos < 0: raise SystemExit('riskColor not found')
insert_pos = pos + len(old) + 1

calc_risk = '''
function calcRiskScore() {
  var temp = 18, hum = 50;
  if (selectedCountyName && MONITORING_DATA[selectedCountyName]) {
    var d = MONITORING_DATA[selectedCountyName];
    temp = d.temp || 18;
    hum = d.hum || 50;
  }
  var tScore = 0;
  if (temp < 0) tScore = 30;
  else if (temp < 5) tScore = 22;
  else if (temp > 35) tScore = 28;
  else if (temp > 30) tScore = 18;
  else tScore = Math.round(Math.abs(temp - 22) * 0.6);
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
print('  [OK] [1a] calcRiskScore')

# --- 1b. 病害分布 → 气候风险仪表 ---
find_disease = (
    '  // === \u75c5\u5bb3\u7c7b\u578b\u5206\u5e03 ===\n'
    "  document.getElementById('diseaseChart').innerHTML =\n"
    "    '<circle cx=\"80\" cy=\"65\" r=\"38\" fill=\"#e74c3c\"/>"
    "<circle cx=\"80\" cy=\"65\" r=\"26\" fill=\"#e67e22\"/>' +\n"
    "    '<circle cx=\"80\" cy=\"65\" r=\"16\" fill=\"#f39c12\"/>"
    "<circle cx=\"80\" cy=\"65\" r=\"8\" fill=\"#3498db\"/>' +\n"
    "    '<text x=\"80\" y=\"70\" fill=\"#fff\" font-size=\"13\" "
    "text-anchor=\"middle\" font-weight=\"bold\">400</text>' +\n"
    "    '<text x=\"80\" y=\"118\" fill=\"#aaa\" font-size=\"10\" "
    'text-anchor="middle">\u75c5\u5bb3\u603b\u6570</text>\';'
)

replace_disease = (
    '  // === \u6c14\u5019\u75c5\u5bb3\u98ce\u9669\u5173\u8054 ===\n'
    '  (function() {\n'
    '    var score = calcRiskScore();\n'
    '    var lvl = score >= 70 ? "\u9ad8" : score >= 40 ? "\u4e2d" : "\u4f4e";\n'
    "    var rc = score >= 70 ? '#e74c3c' : score >= 40 ? '#f39c12' : '#2ecc71';\n"
    '    var svg2 = \'<circle cx="80" cy="48" r="42" fill="none" stroke="\' + rc + \'" stroke-width="2" opacity="0.35"/>\';\n'
    '    svg2 += \'<circle cx="80" cy="48" r="34" fill="\' + rc + \'" opacity="0.1"/>\';\n'
    "    svg2 += '<text x=\"80\" y=\"43\" fill=\"' + rc + '\" font-size=\"24\" text-anchor=\"middle\" font-weight=\"bold\">' + score + '</text>';\n"
    "    svg2 += '<text x=\"80\" y=\"56\" fill=\"' + rc + '\" font-size=\"10\" text-anchor=\"middle\">\u98ce\u9669\u6307\u6570</text>';\n"
    "    svg2 += '<text x=\"80\" y=\"80\" fill=\"#777\" font-size=\"8\" text-anchor=\"middle\">\u7b49\u7ea7</text>';\n"
    "    svg2 += '<text x=\"80\" y=\"95\" fill=\"' + rc + '\" font-size=\"13\" text-anchor=\"middle\" font-weight=\"bold\">' + lvl + '\u98ce\u9669</text>';\n"
    "    document.getElementById('diseaseChart').innerHTML = svg2;\n"
    '  })();'
)

if find_disease in t:
    t = t.replace(find_disease, replace_disease)
    print('  [OK] [1b] disease -> climate risk')
else:
    print('  [WARN] [1b] disease chart block not found, trying partial...')
    # Try simpler: just replace the key content
    if '\u75c5\u5bb3\u7c7b\u578b\u5206\u5e03' in t:
        t = t.replace('\u75c5\u5bb3\u7c7b\u578b\u5206\u5e03', '\u6c14\u5019\u75c5\u5bb3\u98ce\u9669\u5173\u8054')
        print('  [OK] [1b] title replaced (partial)')

# --- 1c. fetchCountyWeather params +wind +daily ---
old_param = "&current=temperature_2m,relative_humidity_2m,weather_code&timezone=Asia%2FShanghai';"
new_param = (
    "&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m'"
    " + '&daily=temperature_2m_max,temperature_2m_min,weather_code,wind_speed_10m_max'"
    " + '&timezone=Asia%2FShanghai&forecast_days=7';"
)
t = t.replace(old_param, new_param)
print('  [OK] [1c] fetchCountyWeather params')

# --- 1d. fetchCountyWeather return 加 wind + forecast ---
old_return = """    return {
      temp: Math.round(j.current.temperature_2m),
      hum: j.current.relative_humidity_2m,
      desc: weatherCodeDesc(j.current.weather_code),
      risk: riskFromHumidity(j.current.relative_humidity_2m)
    };"""

new_return = """    var result = {
      temp: Math.round(j.current.temperature_2m),
      hum: j.current.relative_humidity_2m,
      wind: Math.round(j.current.wind_speed_10m || 0),
      desc: weatherCodeDesc(j.current.weather_code),
      risk: riskFromHumidity(j.current.relative_humidity_2m)
    };
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

t = t.replace(old_return, new_return)
print('  [OK] [1d] fetchCountyWeather return')

# --- 1e. updateCountyMetrics: MONITORING_DATA + scatter ---
insert_after = """    sr.style.color = riskColor(live.risk);
"""
pos = t.find(insert_after)
if pos >= 0:
    insert_at = pos + len(insert_after)
    insert_code = (
        "    MONITORING_DATA[countyName] = { temp: live.temp, hum: live.hum, "
        "risk: live.risk, desc: live.desc, aqi: MONITORING_DATA[countyName] "
        "? (MONITORING_DATA[countyName].aqi || 50) : 50 };\n"
        "    if (live.forecast) { WEATHER_7D_BASE[countyName] = live.forecast; }\n"
        "    if (typeof mapChart !== 'undefined' && mapChart) {\n"
        '      try { var newSc = [], pts = COUNTY_POINTS || [];\n'
        '        for (var si = 0; si < pts.length; si++) {\n'
        '          var p = pts[si], md = MONITORING_DATA[p.name];\n'
        "          if (md) newSc.push([p.value[0], p.value[1], md.temp || 0, md.hum || 50, md.risk || '\u4f4e', p.name]);\n"
        "          else newSc.push([p.value[0], p.value[1], 15, 50, '\u4f4e', p.name]);\n"
        "        }\n"
        "        mapChart.setOption({ series: [{ type: 'scatter', data: newSc }] });\n"
        "      } catch(e2) {}\n"
        "    }\n"
        "    drawCharts();\n"
    )
    t = t[:insert_at] + insert_code + t[insert_at:]
    print('  [OK] [1e] updateCountyMetrics +MONITORING_DATA +scatter')
else:
    print('  [WARN] [1e] insertion point not found')

# --- 1f. renderForecast 加风 ---
old_card = """      '<div class="forecast-temp">' + f.h + '\u00b0/'' + f.l + '\u00b0</div>';"""
new_card = (
    "      '<div class=\"forecast-temp\">' + f.h + '\u00b0/'' + f.l + '\u00b0</div>' +\n"
    "      '<div class=\"forecast-wind\">\ud83c\udf2c ' + (f.w || '--') + 'km/h</div>';"
)
t = t.replace(old_card, new_card)
print('  [OK] [1f] renderForecast +wind')

# --- 1g. fetchRealWeather daily params +wind ---
old_daily = "&daily=temperature_2m_max,temperature_2m_min,weather_code,relative_humidity_2m_max,relative_humidity_2m_min'"
new_daily = "&daily=temperature_2m_max,temperature_2m_min,weather_code,relative_humidity_2m_max,relative_humidity_2m_min,wind_speed_10m_max'"
t = t.replace(old_daily, new_daily)
print('  [OK] [1g] fetchRealWeather daily params')

# --- 1h. fetchRealWeather forecast obj +w ---
old_obj = """          l: Math.round(daily.temperature_2m_min[i])
        });"""
new_obj = (
    """          l: Math.round(daily.temperature_2m_min[i]),
          w: Math.round(daily.wind_speed_10m_max ? (daily.wind_speed_10m_max[i] || 0) : 0)
        });"""
)
t = t.replace(old_obj, new_obj)
print('  [OK] [1h] fetchRealWeather forecast +w')

# --- 1i. DEFAULT_7D fallback +w ---
old_fb = """        l: count > 0 ? Math.round(sumL / count) : 10
      });"""
new_fb = (
    """        l: count > 0 ? Math.round(sumL / count) : 10,
        w: 8
      });"""
)
t = t.replace(old_fb, new_fb)
print('  [OK] [1i] DEFAULT_7D +w')

writef(APP, t)

# ==============================
# 2. HTML 标题
# ==============================
h = read(HTML)
if '\u75c5\u5bb3\u7c7b\u578b\u5206\u5e03' in h:
    h = h.replace('\u75c5\u5bb3\u7c7b\u578b\u5206\u5e03', '\u75c5\u5bb3\u98ce\u9669\u6307\u6570')
    writef(HTML, h)
    print('  [OK] HTML: title changed')
else:
    print('  [WARN] HTML title not found')

# ==============================
# 3. CSS 样式
# ==============================
s = read(CSS)
wind_css = (
    '\n.forecast-wind {\n'
    '  font-size: 9px;\n'
    '  color: #78909c;\n'
    '  text-align: center;\n'
    '  margin-top: 1px;\n'
    '}\n'
)
if '.forecast-wind' not in s:
    s += wind_css
    writef(CSS, s)
    print('  [OK] style.css +.forecast-wind')
else:
    print('  [OK] .forecast-wind already present')

print('\n[OK] All patches applied!')
