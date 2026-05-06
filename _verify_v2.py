# -*- coding: utf-8 -*-
t = open(r'D:\bs\sxgjz\modules\app.js', 'r', encoding='utf-8').read()
print('1. calcRiskScore:', 'calcRiskScore' in t)
print('2. climate risk gauge:', u'\u6c14\u5019\u75c5\u5bb3\u98ce\u9669\u5173\u8054' in t)
print('3. disease removed:', u'\u75c5\u5bb3\u7c7b\u578b\u5206\u5e03' not in t)
print('4. wind in params:', 'wind_speed_10m' in t)
print('5. wind in return:', 'live.wind' in t)
print('6. MONITORING_DATA update:', 'MONITORING_DATA[countyName] = {' in t)
print('7. scatter update:', 'mapChart.setOption' in t)
print('8. forecast-wind:', 'forecast-wind' in t)
print('9. daily wind:', 'wind_speed_10m_max' in t)

i = t.find(u'\u6c14\u5019\u75c5\u5bb3\u98ce\u9669\u5173\u8054')
if i >= 0:
    print()
    print('=== Climate risk section ===')
    print(t[i:i+400])
else:
    print()
    print('WARNING: climate risk section not found in app.js')

# Also check HTML title
h = open(r'D:\bs\sxgjz\古建监测大屏_v5.html', 'r', encoding='utf-8').read()
print()
print('10. HTML title OK:', u'\u75c5\u5bb3\u98ce\u9669\u6307\u6570' in h)
print('11. HTML old title removed:', u'\u75c5\u5bb3\u7c7b\u578b\u5206\u5e03' not in h)
