# -*- coding: utf-8 -*-
"""Fix renderForecast card template to show wind"""
path = r'D:\bs\sxgjz\modules\app.js'
with open(path, 'r', encoding='utf-8') as f:
    t = f.read()

# Find the exact card template line
i = t.find("card.innerHTML")
if i < 0:
    raise SystemExit("card.innerHTML not found")

chunk = t[i:i+300]

# Find the line containing forecast-temp and figure out exact end
lines = chunk.split('\n')
card_start = None
old_line = None
for j, line in enumerate(lines):
    if "card.innerHTML" in line:
        card_start = j
    if "forecast-temp" in line and "f.h" in line:
        old_line = line
        break

if old_line is None:
    raise SystemExit("forecast-temp line not found")

print(f"Old line: {old_line.strip()}")

# The new line replaces the last part after the temperature
# old_line ends with something like: '</div>';
# We need to add a new line after it
new_card_line = old_line + " +\n      '<div class=\"forecast-wind\">\\ud83c\\udf2c ' + (f.w || '--') + 'km/h</div>';"

t = t.replace(old_line, new_card_line)

# Atomic write
tmp = path + '.tmp'
with open(tmp, 'w', encoding='utf-8') as f:
    f.write(t)
import os
os.replace(tmp, path)

# Verify
with open(path, 'r', encoding='utf-8') as f:
    t2 = f.read()
print('wind in template:', 'forecast-wind' in t2)
print(f'Done. File: {len(t2)} bytes')
