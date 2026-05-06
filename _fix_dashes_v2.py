# -*- coding: utf-8 -*-
"""三合一修复：
1. 温度/湿度横杠 → fetchRealWeather URL 去掉无效 humidity daily 参数
2. 散点不过滤 → updateCountyMetrics 不直接 set scatter，改为调 drawMap()
3. 卡片模板语法 → 修复 renderForecast 的 card.innerHTML 换行问题
"""
path = r'D:\bs\sxgjz\modules\app.js'
with open(path, 'r', encoding='utf-8') as f:
    t = f.read()

changes = 0

# === Fix 1: remove invalid daily params from fetchRealWeather ===
old_url = "+ '&daily=temperature_2m_max,temperature_2m_min,weather_code,relative_humidity_2m_max,relative_humidity_2m_min,wind_speed_10m_max'"
new_url = "+ '&daily=temperature_2m_max,temperature_2m_min,weather_code,wind_speed_10m_max'"
if old_url in t:
    t = t.replace(old_url, new_url)
    changes += 1
    print("Fix1: removed invalid humidity daily params from fetchRealWeather")
else:
    print("Fix1 FAILED: old_url not found")

# === Fix 2: updateCountyMetrics scatter overwrite → drawMap() ===
old_scatter = "    if (typeof mapChart !== 'undefined' && mapChart) {\n      try { var newSc = [], pts = COUNTY_POINTS || [];\n        for (var si = 0; si < pts.length; si++) {\n          var p = pts[si], md = MONITORING_DATA[p.name];\n          if (md) newSc.push([p.value[0], p.value[1], md.temp || 0, md.hum || 50, md.risk || '\u4f4e', p.name]);\n          else newSc.push([p.value[0], p.value[1], 15, 50, '\u4f4e', p.name]);\n        }\n        mapChart.setOption({ series: [{ type: 'scatter', data: newSc }] });\n      } catch(e2) {}\n    }"
new_scatter = "    if (typeof drawMap === 'function') { drawMap(); }"

if old_scatter in t:
    t = t.replace(old_scatter, new_scatter)
    changes += 1
    print("Fix2: replaced scatter overwrite with drawMap() call")
else:
    import re
    pattern_bad = r"if \(typeof mapChart !== 'undefined' \&\& mapChart\).*?catch\(e2\) \{\}"
    m = re.search(pattern_bad, t, re.DOTALL)
    if m:
        old_text = m.group()
        print(f"Fix2: found similar block, length={len(old_text)}, replacing...")
        t = t.replace(old_text, new_scatter)
        changes += 1
        print("Fix2: replaced scatter overwrite with drawMap() call (regex fallback)")
    else:
        print("Fix2 FAILED")

# === Fix 3: card template syntax ===
# The stray ; after forecast-temp line breaks the wind assignment
idx = t.find("forecast-temp")
if idx > 0:
    snippet = t[idx:idx+250]
    # Find the pattern: ...</div>'; +\n      '<div class="forecast-wind"
    bad = "'</div>'; +"
    good = "'</div>' +"
    if bad in snippet:
        snippet = snippet.replace(bad, good)
        t = t[:idx] + snippet + t[idx+len(snippet):]
        changes += 1
        print("Fix3: removed stray ';' in card template")
    else:
        print("Fix3: bad pattern not found in snippet")
else:
    print("Fix3 FAILED")

print(f"\nTotal changes: {changes}")

# Write
tmp = path + '.tmp'
with open(tmp, 'w', encoding='utf-8') as f:
    f.write(t)
import os
os.replace(tmp, path)

# Verify
with open(path, 'r', encoding='utf-8') as f:
    t2 = f.read()

print(f"\n=== Verification ===")
print(f"File: {len(t2)} bytes")
ok1 = "relative_humidity_2m_max" not in t2
print(f"[{'OK' if ok1 else 'FAIL'}] No invalid humidity params")
ok2 = "typeof drawMap" in t2
print(f"[{'OK' if ok2 else 'FAIL'}] drawMap in scatter fix")
# Check card template
idx3 = t2.find("forecast-temp")
if idx3 > 0:
    snip = t2[idx3:idx3+200]
    ok3 = "'</div>' +" in snip and not "'</div>';" in snip
    print(f"[{'OK' if ok3 else 'FAIL'}] Card template syntax fixed")
