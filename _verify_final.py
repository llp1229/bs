# -*- coding: utf-8 -*-
"""Final verification of all 3 features"""
t = open(r'D:\bs\sxgjz\modules\app.js', 'r', encoding='utf-8').read()
h = open(r'D:\bs\sxgjz\古建监测大屏_v5.html', 'r', encoding='utf-8').read()
s = open(r'D:\bs\sxgjz\modules\style.css', 'r', encoding='utf-8').read()

checks = [
    # 风四 - 气候风险关联
    ("calcRiskScore exists", "calcRiskScore" in t, "app.js"),
    ("climate risk gauge code", u"\u6c14\u5019\u75c5\u5bb3\u98ce\u9669\u5173\u8054" in t, "app.js"),
    ("old disease removed", u"\u75c5\u5bb3\u7c7b\u578b\u5206\u5e03" not in t, "app.js"),
    ("HTML title changed", u"\u75c5\u5bb3\u98ce\u9669\u6307\u6570" in h, "HTML"),

    # 地图散点与天气预报匹配
    ("MONITORING_DATA sync", "MONITORING_DATA[countyName] = {" in t, "app.js"),
    ("scatter update code", "mapChart.setOption({ series:" in t, "app.js"),
    ("county forecast storage", "WEATHER_7D_BASE[countyName]" in t, "app.js"),
    ("fetchCountyWeather uses county coord", "COUNTY_POINTS.find" in t, "app.js"),

    # 风
    ("wind in fcw params", "wind_speed_10m" in t, "app.js"),
    ("wind in fcw return", "wind:" in t, "app.js"),
    ("wind in renderForecast", "forecast-wind" in t, "app.js"),
    ("wind in frw daily params", "wind_speed_10m_max" in t, "app.js"),
    ("CSS forecast-wind", ".forecast-wind" in s, "style.css"),
]

print("=" * 55)
print(f"{'CHECK':<42} {'PASS':>5} {'FILE':>8}")
print("=" * 55)
all_ok = True
for name, result, src in checks:
    mark = "OK" if result else "FAIL"
    if not result:
        all_ok = False
    print(f"{name:<42} {mark:>5} {src:>8}")
print("=" * 55)

if all_ok:
    print("\nAll 13 checks passed!")
else:
    print("\nSome checks FAILED")
