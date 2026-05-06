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
# Find the fetchRealWeather URL line with the bad params
old_url = (
    "+ '&daily=temperature_2m_max,temperature_2m_min,weather_code,relative_humidity_2m_max,relative_humidity_2m_min,wind_speed_10m_max'"
)
new_url = (
    "+ '&daily=temperature_2m_max,temperature_2m_min,weather_code,wind_speed_10m_max'"
)
if old_url in t:
    t = t.replace(old_url, new_url)
    changes += 1
    print("Fix1: removed invalid humidity daily params from fetchRealWeather")
else:
    print("Fix1 FAILED: old_url not found")

# === Fix 2: updateCountyMetrics scatter overwrite → drawMap() ===
# Find the block that does mapChart.setOption({series: [{...}]})
old_scatter = """    if (typeof mapChart !== 'undefined' && mapChart) {
      try { var newSc = [], pts = COUNTY_POINTS || [];
        for (var si = 0; si < pts.length; si++) {
          var p = pts[si], md = MONITORING_DATA[p.name];
          if (md) newSc.push([p.value[0], p.value[1], md.temp || 0, md.hum || 50, md.risk || '\\u4f4e', p.name]);
          else newSc.push([p.value[0], p.value[1], 15, 50, '\\u4f4e', p.name]);
        }
        mapChart.setOption({ series: [{ type: 'scatter', data: newSc }] });
      } catch(e2) {}
    }"""

new_scatter = """    if (typeof drawMap === 'function') { drawMap(); }"""

if old_scatter in t:
    t = t.replace(old_scatter, new_scatter)
    changes += 1
    print("Fix2: replaced scatter overwrite with drawMap() call")
else:
    # Try searching with different unicode
    import re
    pattern = r"if \(typeof mapChart !== 'undefined' \&\& mapChart\).*?catch\(e2\) \{\}"
    m = re.search(pattern, t, re.DOTALL)
    if m:
        print(f"Fix2 found similar block at offset {m.start()}, length {len(m.group())}")
        # Print snippet for manual fix
        print(f"  Snippet: {repr(m.group()[:200])}")
    print("Fix2 FAILED: scatter overwrite block not found")
    print("  (Likely the Unicode \\u4f4e mismatch)")

# === Fix 3: card template syntax ===
old_card = "'<div class=\"forecast-temp\">' + f.h + '\\u00b0/' + f.l + '\\u00b0</div>'; +"
# The file might use ℃ not °, try both
for pattern_try in [
    "'<div class=\"forecast-temp\">' + f.h + '°/' + f.l + '°</div>'; +",
]:
    if pattern_try in t:
        new_card = "'<div class=\"forecast-temp\">' + f.h + '°/' + f.l + '°</div>' +"
        t = t.replace(pattern_try, new_card)
        changes += 1
        print(f"Fix3: removed stray ';' in card template (pattern 1)")
        break
else:
    # Try case-insensitive search for the middle part
    idx = t.find("forecast-temp")
    if idx > 0:
        snippet = t[idx:idx+300]
        import re
        # Find the exact line ending
        m = re.search(r"(<div class=\"forecast-temp.*?)</div>';\s+\+", snippet)
        if m:
            old = m.group(0)
            # remove the ; before +
            new_one = old.replace("';", "'")
            new_one = new_one.replace('; +', "' +")
            t = t[:idx] + snippet.replace(old, new_one) + t[idx+len(snippet):]
            changes += 1
            print("Fix3: removed stray ';' in card template (regex fallback)")
        else:
            print("Fix3 FAILED: card template not matched by regex")

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
print(f"No invalid humidity param: {'relative_humidity_2m_max' not in t2}")
print(f"drawMap in scatter fix: {'drawMap()' in t2 and 'typeof drawMap' in t2}")

# Check card template for stray ';'
idx2 = t2.find('card.innerHTML')
snippet2 = t2[idx2:idx2+200]
print(f"Card template OK: {'temp\"> +' in snippet2 and not ''</div>\'; +' in snippet2}")
