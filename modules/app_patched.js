// ===== 状态 =====
let mapChart = null;
let currentCityData = null;  // 当前选中的城市对象
let currentForecastCity = null; // 当前预报所属城市
let selectedCountyName = null;

// ===== 工具 =====
function riskColor(r) { return r==='高'?'#e74c3c':r==='中'?'#f39c12':'#2ecc71'; }

function calcRiskScore() {
  var temp = 18, hum = 50, wnd = 5;
  if (selectedCountyName && MONITORING_DATA[selectedCountyName]) {
    var d = MONITORING_DATA[selectedCountyName];
    temp = d.temp || 18;
    hum = d.hum || 50;
    wnd = d.wind || 5;
  }
  var tScore = 0;
  if (temp < -5) tScore = 35;
  else if (temp < 0) tScore = 30;
  else if (temp < 5) tScore = 22;
  else if (temp > 35) tScore = 28;
  else if (temp > 30) tScore = 18;
  else tScore = Math.round(Math.abs(temp - 22) * 0.6);
  var hScore = 0;
  if (hum > 85) hScore = 35;
  else if (hum > 70) hScore = 25;
  else if (hum > 55) hScore = 15;
  else if (hum < 20) hScore = 12;
  var wScore = 0;
  if (wnd > 40) wScore = 28;
  else if (wnd > 25) wScore = 18;
  else if (wnd > 15) wScore = 10;
  else if (wnd > 8) wScore = 5;
  var s = tScore + hScore + wScore;
  return Math.min(100, Math.round(s));
}

function diseaseImpact(temp, hum, wind) {
  var items = [];
  // Temperature analysis
  if (temp < 0) items.push({ label: '冻融循环', color: '#e74c3c', score: 90 });
  else if (temp < 5) items.push({ label: '低温脆化', color: '#e74c3c', score: 70 });
  else if (temp < 10) items.push({ label: '低温收缩', color: '#f39c12', score: 50 });
  else if (temp > 35) items.push({ label: '极端热胀', color: '#e74c3c', score: 95 });
  else if (temp > 30) items.push({ label: '高温应力', color: '#e74c3c', score: 75 });
  else if (temp > 25) items.push({ label: '温升膨胀', color: '#f39c12', score: 40 });
  else if (temp >= 15) items.push({ label: '温度宜人', color: '#2ecc71', score: 15 });
  else items.push({ label: '偏低温', color: '#f39c12', score: 30 });
  // Humidity
  if (hum > 85) items.push({ label: '极度受潮', color: '#e74c3c', score: 95 });
  else if (hum > 70) items.push({ label: '潮湿侵蚀', color: '#e74c3c', score: 75 });
  else if (hum > 55) items.push({ label: '湿度偏高', color: '#f39c12', score: 45 });
  else if (hum < 20) items.push({ label: '过度干燥', color: '#e74c3c', score: 70 });
  else if (hum < 35) items.push({ label: '偏干燥', color: '#f39c12', score: 35 });
  else items.push({ label: '湿度宜人', color: '#2ecc71', score: 15 });
  // Wind
  if (wind > 39) items.push({ label: '强风破坏', color: '#e74c3c', score: 85 });
  else if (wind > 20) items.push({ label: '表面侵蚀', color: '#f39c12', score: 50 });
  else if (wind > 10) items.push({ label: '轻微影响', color: '#2ecc71', score: 20 });
  else items.push({ label: '无影响', color: '#2ecc71', score: 5 });
  return items;
}

function predictDiseases(temp, hum, wind) {
  var d = [];
  // Frost + moisture = spalling
  if (temp < 5 && hum > 55) { d.push('冻融剥落'); d.push('砂浆酥化'); }
  // Heat + moisture = mold
  if (temp > 25 && hum > 65) { d.push('霹菌滋生'); d.push('盐析泛碱'); }
  // Extreme humidity
  if (hum > 80) { d.push('木质腐朽'); d.push('生物侵害'); }
  // Extreme cold
  if (temp < 0) d.push('低温脆裂');
  // Extreme heat
  if (temp > 32) d.push('表面龟裂');
  // Dry
  if (hum < 25) { d.push('干缩裂缝'); d.push('涂层剥落'); }
  // Wind erosion
  if (wind > 25) d.push('风蚀磨损');
  // Normal
  if (d.length === 0) d.push('轻微风化');
  // Deduplicate
  var seen = {}, uniq = [];
  for (var i = 0; i < d.length; i++) { if (!seen[d[i]]) { seen[d[i]] = 1; uniq.push(d[i]); } }
  return uniq;
}

function tempColor(t) { return t>=22?'#e74c3c':t>=17?'#f39c12':'#3498db'; }
function aqiInfo(a) { return a<=50?'优':a<=100?'良':a<=150?'轻度':a<=200?'中度':'重度'; }

// ===== 预报卡片渲染 =====
// 核心：统一调用 renderForecast(cityName) 更新预报区
function renderForecast(cityName) {
  const data = (cityName && WEATHER_7D_BASE[cityName]) ? WEATHER_7D_BASE[cityName] : DEFAULT_7D;
  currentForecastCity = cityName;

  // 更新标题
  document.getElementById('forecastCityName').textContent = (selectedCountyName ? selectedCountyName + ' · ' : '') + (cityName || '山西省');

  const container = document.getElementById('forecastCards');
  container.innerHTML = '';
  // syncBothPanels(); // CSS stretch代替

  data.forEach((f, i) => {
    const card = document.createElement('div');
    card.className = 'forecast-card' + (i === 0 ? ' active' : '');
    // 动态计算星期几
      const dayNames = ['周日','周一','周二','周三','周四','周五','周六'];
      const now = new Date();
      const day = new Date(now);
      day.setDate(now.getDate() + i);
      const month = day.getMonth() + 1;
      const dd = day.getDate();
      const dateStr = month + '/' + dd;
      const dayLabel = (i === 0 ? '今天' : dayNames[day.getDay()]) + ' ' + dateStr;
      card.innerHTML =
      '<div class="forecast-day">' + dayLabel + '</div>' +
      '<div class="forecast-icon">' + f.i + '</div>' +
      '<div class="forecast-temp">' + f.h + '°/' + f.l + '°</div>' +
      '<div class="forecast-wind">\ud83c\udf2c ' + (f.w || '--') + 'km/h</div>';

    card.onclick = function() {
      document.querySelectorAll('.forecast-card').forEach(c => c.classList.remove('active'));
      this.classList.add('active');
      // 选中的那一天更新顶栏温度
      document.getElementById('m-temp').textContent = f.h + '°C';
      // 同时更新湿度趋势（简化处理：用该城市平均湿度）
      const monD = MONITORING_DATA[selectedCountyName];
      if (monD) {
        document.getElementById('m-hum').textContent = monD.hum + '%';
      }
    };

    container.appendChild(card);
  });

  // 渲染完后，默认高亮"今天"并用今天温度更新顶栏
  if (data[0]) {
    document.getElementById('m-temp').textContent = data[0].h + '°C';
  }

  // 更新趋势图
  drawCharts();
}

// ===== 地图点击 → 更新所有UI =====
function onCountyClick(name, cityName) {
  selectedCountyName = name;
  const d = MONITORING_DATA[name] || {};

  // 1. 更新顶部指标卡
  document.getElementById('m-region').textContent = name + ' · ' + cityName;
  // Show loading state while fetching real county weather
  document.getElementById('m-temp').textContent = '...';
  document.getElementById('m-hum').textContent = '...';
  document.getElementById('m-risk').textContent = '...';
  updateCountyMetrics(name, cityName);

  // 2. 更新右侧详情面板
  document.getElementById('sel-name').textContent = name + ' · ' + cityName;
  document.getElementById('sel-temp').textContent = (d.temp || '--') + '°C';
  document.getElementById('sel-hum').textContent = (d.hum || '--') + '%';
  document.getElementById('sel-desc').textContent = d.desc || '--';
  const aqiEl = document.getElementById('sel-aqi');
  aqiEl.textContent = (d.aqi || '--') + ' (' + aqiInfo(d.aqi) + ')';
  const selRisk = document.getElementById('sel-risk');
  selRisk.textContent = d.risk || '--';
  selRisk.style.color = riskColor(d.risk);
  document.getElementById('selectedInfo').classList.add('show');

  // 显示区县信息后重绘地图，消除空白
  setTimeout(() => mapChart.resize(), 50);

  // 3. ★ 关键：更新7天预报为该区县所属城市的天气
  renderForecast(cityName);
}



// ===== county-level real weather (fetch on click) =====
async function fetchCountyWeather(countyName) {
  var pt = COUNTY_POINTS.find(function(p) { return p.name === countyName; });
  if (!pt) return null;
  var lat = pt.value[1], lon = pt.value[0];
  try {
    var ctrl = new AbortController();
    var tid = setTimeout(function() { ctrl.abort(); }, 8000);
    var r = await fetch('/api/amap/weather/county?lat=' + lat + '&lon=' + lon, { signal: ctrl.signal });
    clearTimeout(tid);
    if (!r.ok) return null;
    var j = await r.json();
    if (!j || !j.temp) return null;
    j.risk = j.risk || (function() {
      var _t=j.temp, _h=j.hum, _w=j.wind||0;
      var tS=0; if(_t<-5)tS=35;else if(_t<0)tS=30;else if(_t<5)tS=22;else if(_t>35)tS=28;else if(_t>30)tS=18;else tS=Math.round(Math.abs(_t-22)*0.6);
      var hS=0; if(_h>85)hS=35;else if(_h>70)hS=25;else if(_h>55)hS=15;else if(_h<20)hS=12;
      var wS=0; if(_w>40)wS=28;else if(_w>25)wS=18;else if(_w>15)wS=10;else if(_w>8)wS=5;
      var total=tS+hS+wS; return total>=70?'\u9ad8':total>=40?'\u4e2d':'\u4f4e';
    })();
    return j;
  } catch(e) {
    console.warn('[Weather] county fetch failed:', countyName, e.message||e);
    return null;
  }
}

async function updateCountyMetrics(countyName, cityName) {
  var live = await fetchCountyWeather(countyName);
  if (live) {
    document.getElementById('m-temp').textContent = live.temp + '°C';
    document.getElementById('m-hum').textContent = live.hum + '%';
    var riskEl = document.getElementById('m-risk');
    riskEl.textContent = live.risk;
    riskEl.style.color = riskColor(live.risk);
    document.getElementById('sel-temp').textContent = live.temp + '°C';
    document.getElementById('sel-hum').textContent = live.hum + '%';
    document.getElementById('sel-desc').textContent = live.desc;
    var sr = document.getElementById('sel-risk');
    sr.textContent = live.risk;
    sr.style.color = riskColor(live.risk);
    MONITORING_DATA[countyName] = { temp: live.temp, hum: live.hum, wind: live.wind, risk: live.risk, desc: live.desc, aqi: MONITORING_DATA[countyName] ? (MONITORING_DATA[countyName].aqi || 50) : 50 };
    if (live.forecast) { WEATHER_7D_BASE[countyName] = live.forecast; }
    if (typeof drawMap === 'function') { drawMap(); }
    drawCharts();
  } else {
    // Open-Meteo failed, fall back to MONITORING_DATA
    var d = MONITORING_DATA[countyName] || {};
    document.getElementById('m-temp').textContent = (d.temp || '--') + '°C';
    document.getElementById('m-hum').textContent = (d.hum || '--') + '%';
    var riskEl2 = document.getElementById('m-risk');
    riskEl2.textContent = d.risk || '--';
    riskEl2.style.color = riskColor(d.risk);
  }
}
// ===== 城市下拉切换 =====
function initMap() {

// 左面板同步到右面板
function syncBothPanels(){var lp=document.getElementById("leftPanel");var rp=document.getElementById("rightPanel");if(lp&&rp){rp.style.height=lp.offsetHeight+"px";}}// syncBothPanels(); // CSS stretch代替setTimeout(function(){var lp=document.getElementById("leftPanel");if(lp&&window.ResizeObserver){new ResizeObserver(syncBothPanels).observe(lp);}},500);
  mapChart = echarts.init(document.getElementById('mapChart'));
// 只阻止双击缩放，不阻止click事件
mapChart.getZr().on('dblclick',e=>e.stopPropagation());
  echarts.registerMap('shanxi', SHANXI_GEO);

  // 填充城市下拉
  const sel = document.getElementById('citySelect');
  CITY_LIST.forEach(c => {
    const opt = document.createElement('option');
    opt.value = c.name;
    opt.textContent = c.name + ' (' + c.counties.length + '区县)';
    sel.appendChild(opt);
  });

  sel.onchange = function() {
    const cityName = this.value;
    const countySel = document.getElementById('countySelect');
    if (!cityName) {
      countySel.innerHTML = '<option value="">— 选择区县 —</option>';
      resetMap();
      return;
    }
    currentCityData = CITY_LIST.find(x => x.name === cityName) || null;

    // 填充区县下拉
    countySel.innerHTML = '<option value="">— 选择区县 —</option>';
    if (currentCityData) {
      currentCityData.counties.forEach(name => {
        const opt = document.createElement('option');
        opt.value = name;
        opt.textContent = name;
        countySel.appendChild(opt);
      });
    }

    if (currentCityData) {
      const first = currentCityData.counties[0];
      if (first) onCountyClick(first, cityName);
    }
    drawMap();
  };

  // 区县下拉选择
  const countySel = document.getElementById('countySelect');
  countySel.addEventListener('change', function() {
    const countyName = this.value;
    const cityName = sel.value;
    if (!countyName || !cityName) return;
    onCountyClick(countyName, cityName);
  });

  drawMap();
}

function drawMap() {
  if (!mapChart) return;

  // === 注册地图区域点击（在setOption之前，防止被覆盖） ===
  mapChart.off('click');

  // 区县区域点击 → 放大到地级市
  mapChart.on('click', function(params) {
    console.log('Click params:', params.componentType, params.name, params.seriesType);
    if (params.componentType !== 'geo') return;
    const clickedName = params.name;
    if (!clickedName) return;

    const countyPoint = COUNTY_POINTS.find(p => p.name === clickedName);
    if (!countyPoint || !countyPoint.cityName) return;

    const cityData = CITY_LIST.find(c => c.name === countyPoint.cityName);
    if (!cityData) return;

    currentCityData = cityData;
    const sel = document.getElementById('citySelect');
    if (sel) sel.value = cityData.name;
    // 同步区县下拉
    var coSel2 = document.getElementById('countySelect');
    if (coSel2) coSel2.value = clickedName;
    onCountyClick(clickedName, cityData.name);
    drawMap();
  });

  // 散点点击 → 显示该县天气
  mapChart.on('click', function(params) {
    if (params.seriesType !== 'scatter') return;
    const name = params.name;
    const cityName = params.value[params.value.length - 1];
    // 同步下拉框
    var sel3 = document.getElementById('citySelect');
    if (sel3) sel3.value = cityName;
    var coSel3 = document.getElementById('countySelect');
    if (coSel3) coSel3.value = name;
    onCountyClick(name, cityName);
  });

  // === 渲染地图 ===
  let displayPoints = COUNTY_POINTS;
  let zoom = 1.1;
  let center = [112.5, 37.5];

  if (currentCityData) {
    displayPoints = COUNTY_POINTS.filter(p => p.cityName === currentCityData.name);
    zoom = 2.2;
    center = currentCityData.center || [112.5, 37.5];
  }

  const scatterData = displayPoints.map(p => {
    const d = MONITORING_DATA[p.name] || { temp: 16, hum: 50, risk: '低' };
    return { name: p.name, value: [...p.value, d.temp, d.hum, d.risk, p.cityName] };
  });

  const showLabels = !!currentCityData;

  mapChart.setOption({
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'item',
      formatter: function(p) {
        if (p.seriesType === 'scatter') {
          const d = MONITORING_DATA[p.name] || {};
          return '<b style="color:#00f0ff;font-size:13px">' + p.name + '</b><br/>' +
                 '<span style="color:#f39c12">🌡 ' + (d.temp||'--') + '°C</span> &nbsp;' +
                 '<span style="color:#3498db">💧 ' + (d.hum||'--') + '%</span><br/>' +
                 '<span style="color:#bbb">🌤 ' + (d.desc||'') + '</span><br/>' +
                 '<span style="color:#e74c3c">⚠ ' + (d.risk||'--') + '</span>';
        }
        return '<b>' + p.name + '</b>';
      }
    },
    geo: {
      map: 'shanxi',
      roam: false,
      zoom: zoom,
      center: center,
      scaleLimit: { min: 0.8, max: 15 },
      itemStyle: {
        areaColor: 'rgba(10,80,180,0.12)',
        borderColor: 'rgba(0,180,216,0.5)',
        borderWidth: 1.5
      },
      emphasis: {
        itemStyle: { areaColor: 'rgba(0,180,216,0.35)' },
        label: { show: false }
      }
    },
    series: [{
      type: 'scatter',
      coordinateSystem: 'geo',
      data: scatterData,
      symbol: 'circle',
      symbolSize: showLabels ? 14 : function(p) {
        const t = p[2]; return t>=22?12:t>=18?9:t>=15?7:6;
      },
      label: {
        show: showLabels,
        formatter: '{b}',
        position: 'right',
        color: '#fff', fontSize: 10,
        backgroundColor: 'rgba(0,0,0,0.55)',
        padding: [2,5], borderRadius: 3, distance: 5
      },
      itemStyle: {
        color: function(p) { return tempColor(p.value[2]); },
        shadowBlur: 8, shadowColor: 'rgba(255,215,0,0.3)'
      },
      emphasis: {
        scale: showLabels ? 1.3 : 1.8,
        itemStyle: { shadowBlur: 25, shadowColor: '#00f0ff' }
      }
    }]
  });
}

function resetMap() {
  currentCityData = null;
  selectedCountyName = null;
  document.getElementById('citySelect').value = '';
  var coSel = document.getElementById('countySelect');
  if (coSel) coSel.innerHTML = '<option value="">— 选择区县 —</option>';
  document.getElementById('selectedInfo').classList.remove('show');
  document.getElementById('m-region').textContent = '山西省';
  document.getElementById('m-temp').textContent = '17°C';
  document.getElementById('m-hum').textContent = '48%';
  document.getElementById('m-risk').textContent = '中';
  document.getElementById('m-risk').style.color = '#f39c12';
  renderForecast(null);
  drawMap();
}

// ===== 时钟 =====
setInterval(() => {
  document.getElementById('clock').textContent = new Date().toLocaleTimeString('zh-CN', {hour12:false});
}, 1000);

// ===== 图表 =====
function drawCharts() {
  // === 温度/湿度趋势图（带刻度和动态日期） ===
  const now = new Date();
  const dayNames = ['周日','周一','周二','周三','周四','周五','周六'];
  // 获取当前城市的7天预报数据
  var forecastData = null;
  if (currentForecastCity && WEATHER_7D_BASE[currentForecastCity]) {
    forecastData = WEATHER_7D_BASE[currentForecastCity];
  }
  // 从MONITORING_DATA获取湿度数据
  var humBase = 50;
  if (selectedCountyName && MONITORING_DATA[selectedCountyName]) {
    humBase = MONITORING_DATA[selectedCountyName].hum || 50;
  }

  var temps = [], hums = [], labels = [];
  for (var i = 0; i < 7; i++) {
    var day = new Date(now);
    day.setDate(now.getDate() + i);
    var m = day.getMonth() + 1, d = day.getDate();
    labels.push(i === 0 ? '今天' + m + '/' + d : dayNames[day.getDay()] + m + '/' + d);
    if (forecastData && forecastData[i]) {
      temps.push(forecastData[i].h);
      hums.push(forecastData[i].l + Math.floor((forecastData[i].h - forecastData[i].l) * 0.6));
    } else {
      temps.push(15 + Math.floor(Math.random() * 8));
      hums.push(humBase + Math.floor(Math.random() * 10) - 5);
    }
  }

  // SVG参数: viewBox 0 0 400 150, 绘图区 x:45-375, y:10-110
  var svgX = 45, svgXEnd = 375, svgY = 10, svgYEnd = 110;
  var w = svgXEnd - svgX, h2 = svgYEnd - svgY;
  var tempMax = Math.max.apply(null, temps), tempMin = Math.min.apply(null, temps);
  var tempRange = Math.max(tempMax - tempMin, 1);
  var humMax = Math.max.apply(null, hums), humMin = Math.min.apply(null, hums);
  var humRange = Math.max(humMax - humMin, 1);

  function toX(i) { return svgX + (i / 6) * w; }
  function tempY(v) { return svgYEnd - ((v - tempMin) / tempRange) * h2 * 0.85 - h2 * 0.05; }
  function humY(v) { return svgYEnd - ((v - humMin) / humRange) * h2 * 0.85 - h2 * 0.05; }

  var svg = '';

  // 水平网格线 + Y轴刻度（温度）
  for (var g = 0; g <= 4; g++) {
    var gy = svgY + g * (h2 / 4);
    var val = Math.round(tempMax - g * (tempRange / 4));
    svg += '<line x1="' + svgX + '" y1="' + gy + '" x2="' + svgXEnd + '" y2="' + gy + '" stroke="rgba(255,255,255,0.08)" stroke-width="0.5"/>';
    svg += '<text x="' + (svgX - 5) + '" y="' + (gy + 3) + '" fill="#e74c3c" font-size="8" text-anchor="end">' + val + '°</text>';
  }

  // 温度折线 + 圆点
  var tempPoints = '';
  for (var i = 0; i < 7; i++) {
    var tx = toX(i), ty = tempY(temps[i]);
    tempPoints += tx + ',' + ty + ' ';
    svg += '<circle cx="' + tx + '" cy="' + ty + '" r="3" fill="#e74c3c"/>';
    svg += '<text x="' + tx + '" y="' + (ty - 6) + '" fill="#e74c3c" font-size="8" text-anchor="middle">' + temps[i] + '°</text>';
  }
  svg += '<polyline points="' + tempPoints + '" fill="none" stroke="#e74c3c" stroke-width="2"/>';

  // 湿度折线 + 圆点
  var humPoints = '';
  for (var i = 0; i < 7; i++) {
    var hx = toX(i), hy = humY(hums[i]);
    humPoints += hx + ',' + hy + ' ';
    svg += '<circle cx="' + hx + '" cy="' + hy + '" r="3" fill="#3498db"/>';
    svg += '<text x="' + hx + '" y="' + (hy + 12) + '" fill="#3498db" font-size="8" text-anchor="middle">' + hums[i] + '%</text>';
  }
  svg += '<polyline points="' + humPoints + '" fill="none" stroke="#3498db" stroke-width="2"/>';

  // X轴日期标签
  for (var i = 0; i < 7; i++) {
    var lx = toX(i);
    var lbl = labels[i];
    // 交替上下显示避免重叠
    var ly = i % 2 === 0 ? 128 : 140;
    svg += '<text x="' + lx + '" y="' + ly + '" fill="#888" font-size="8" text-anchor="middle">' + lbl + '</text>';
  }

  // 图例
  svg += '<line x1="' + (svgX + 60) + '" y1="6" x2="' + (svgX + 80) + '" y2="6" stroke="#e74c3c" stroke-width="2"/>';
  svg += '<text x="' + (svgX + 83) + '" y="9" fill="#e74c3c" font-size="9">温度(°C)</text>';
  svg += '<line x1="' + (svgX + 145) + '" y1="6" x2="' + (svgX + 165) + '" y2="6" stroke="#3498db" stroke-width="2"/>';
  svg += '<text x="' + (svgX + 168) + '" y="9" fill="#3498db" font-size="9">湿度(%)</text>';

  document.getElementById('trendChart').innerHTML = svg;

  // === 气候病害风险关联 ===
  (function() {
    var score = calcRiskScore();
    var temp = 18, hum = 50, wnd = 5;
    if (selectedCountyName && MONITORING_DATA[selectedCountyName]) {
      var d = MONITORING_DATA[selectedCountyName];
      temp = d.temp || 18; hum = d.hum || 50; wnd = d.wind || 5;
    }
    var lvl = score >= 70 ? "\u9ad8" : score >= 40 ? "\u4e2d" : "\u4f4e";
    var rc = score >= 70 ? '#e74c3c' : score >= 40 ? '#f39c12' : '#2ecc71';

    // Wind Beaufort scale
    function beaufort(ws) {
      if (ws < 1) return '0\u7ea7';
      if (ws < 6) return '1\u7ea7';
      if (ws < 12) return '2\u7ea7';
      if (ws < 20) return '3\u7ea7';
      if (ws < 29) return '4\u7ea7';
      if (ws < 39) return '5\u7ea7';
      if (ws < 50) return '6\u7ea7';
      if (ws < 62) return '7\u7ea7';
      return '8+\u7ea7';
    }

    var impacts = diseaseImpact(temp, hum, wnd);
    var diseases = predictDiseases(temp, hum, wnd);

    var svg2 = '';

    // --- LEFT: Score gauge ---
    // Outer glow ring
    svg2 += '<circle cx="70" cy="48" r="45" fill="none" stroke="' + rc + '" stroke-width="1.5" opacity="0.15"/>';
    // Arc (semi-circle top)
    svg2 += '<path d="M 25 48 A 45 45 0 0 1 115 48" fill="none" stroke="' + rc + '" stroke-width="2.5" opacity="0.3" stroke-linecap="round"/>';
    // Inner filled circle
    svg2 += '<circle cx="70" cy="48" r="34" fill="' + rc + '" opacity="0.08"/>';
    // Score number
    svg2 += '<text x="70" y="44" fill="' + rc + '" font-size="26" text-anchor="middle" font-weight="bold">' + score + '</text>';
    // Label
    svg2 += '<text x="70" y="58" fill="#888" font-size="8" text-anchor="middle">\u75c5\u5bb3\u98ce\u9669\u6307\u6570</text>';
    // Level badge below gauge
    svg2 += '<rect x="45" y="68" width="50" height="16" rx="8" fill="' + rc + '" opacity="0.15"/>';
    svg2 += '<text x="70" y="80" fill="' + rc + '" font-size="10" text-anchor="middle" font-weight="bold">' + lvl + '\u98ce\u9669</text>';

    // --- DIVIDER ---
    svg2 += '<line x1="135" y1="10" x2="135" y2="115" stroke="rgba(255,255,255,0.06)" stroke-width="1"/>';

    // --- RIGHT: Environmental analysis ---
    var rx = 148, rw = 242; // right content area

    // Title
    svg2 += '<text x="' + rx + '" y="16" fill="#00f0ff" font-size="10" font-weight="bold">\ud83c\udf21\ufe0f \u73af\u5883\u56e0\u7d20\u5206\u6790</text>';

    // Factor rows
    function drawFactor(y, icon, name, val, unit, color, impact, barPct) {
      var s = '';
      var bx = rx + 62, bw = 110, bh = 6;
      s += '<text x="' + rx + '" y="' + (y+6) + '" font-size="12">' + icon + '</text>';
      s += '<text x="' + (rx+16) + '" y="' + (y+6) + '" fill="#bbb" font-size="8">' + name + '</text>';
      s += '<text x="' + (rx+42) + '" y="' + (y+6) + '" fill="' + color + '" font-size="10" font-weight="bold">' + val + unit + '</text>';
      // Bar background
      s += '<rect x="' + bx + '" y="' + (y+1) + '" width="' + bw + '" height="' + bh + '" rx="3" fill="rgba(255,255,255,0.05)"/>';
      // Bar fill
      var fw = Math.min(bw, Math.max(4, bw * barPct / 100));
      s += '<rect x="' + bx + '" y="' + (y+1) + '" width="' + fw + '" height="' + bh + '" rx="3" fill="' + color + '" opacity="0.7"/>';
      // Impact text
      var ic = barPct >= 70 ? '#e74c3c' : barPct >= 40 ? '#f39c12' : '#2ecc71';
      s += '<text x="' + (bx + bw + 8) + '" y="' + (y+6) + '" fill="' + ic + '" font-size="7">' + impact + '</text>';
      return s;
    }

    // Temperature factor
    var tImpact = impacts[0] ? impacts[0].label : '\u6b63\u5e38';
    var tScore = impacts[0] ? impacts[0].score : 20;
    svg2 += drawFactor(27, '\ud83c\udf21\ufe0f', '\u6e29\u5ea6', temp, '\u00b0C', '#e74c3c', tImpact, tScore);

    // Humidity factor
    var hImpact = impacts[1] ? impacts[1].label : '\u6b63\u5e38';
    var hScore = impacts[1] ? impacts[1].score : 20;
    svg2 += drawFactor(44, '\ud83d\udca7', '\u6e7f\u5ea6', hum, '%', '#3498db', hImpact, hScore);

    // Wind factor
    var wImpact = impacts[2] ? impacts[2].label : '\u6b63\u5e38';
    var wScore = impacts[2] ? impacts[2].score : 10;
    svg2 += drawFactor(61, '\ud83c\udf2c\ufe0f', '\u98ce\u529b', beaufort(wnd), '', '#2ecc71', wImpact, wScore);

    // Separator
    svg2 += '<line x1="' + rx + '" y1="78" x2="' + (rx + rw) + '" y2="78" stroke="rgba(255,255,255,0.07)" stroke-width="0.5"/>';

    // Disease prediction header
    var dlColor = score >= 70 ? '#e74c3c' : score >= 40 ? '#f39c12' : '#2ecc71';
    svg2 += '<text x="' + rx + '" y="93" fill="' + dlColor + '" font-size="8" font-weight="bold">\u26a0\ufe0f \u53ef\u80fd\u8bf1\u53d1\u75c5\u5bb3:</text>';

    // Disease tags
    var tagX = rx;
    for (var di = 0; di < diseases.length; di++) {
      if (tagX > rx + rw - 10) { tagX = rx; /* wrap not needed for 130 height, just truncate */ }
      var tc = score >= 70 ? 'rgba(231,76,60,0.12)' : score >= 40 ? 'rgba(243,156,18,0.1)' : 'rgba(46,204,113,0.1)';
      var ttc = score >= 70 ? '#e74c3c' : score >= 40 ? '#f39c12' : '#2ecc71';
      var tw = diseases[di].length * 9 + 16;
      svg2 += '<rect x="' + tagX + '" y="100" width="' + tw + '" height="20" rx="10" fill="' + tc + '" stroke="' + ttc + '" stroke-width="0.5" stroke-opacity="0.3"/>';
      svg2 += '<text x="' + (tagX + tw/2) + '" y="113" fill="' + ttc + '" font-size="9" text-anchor="middle">' + diseases[di] + '</text>';
      tagX += tw + 6;
    }

    document.getElementById('diseaseChart').innerHTML = svg2;
  })();
}

// ===== AI =====
function addChat(txt) {
  const el = document.createElement('div');
  el.className = 'chat-msg ai';
  el.innerHTML = txt;
  document.getElementById('chatHistory').appendChild(el);
  document.getElementById('chatHistory').scrollTop = 99999;
}
function askAI(q) {
  document.getElementById('aiQuestion').value = q;
  sendMsg();
}
async function sendMsg() {
  const q = document.getElementById('aiQuestion').value.trim();
  if (!q) return;
  const btn = document.getElementById('sendBtn');
  const hist = document.getElementById('chatHistory');
  hist.innerHTML += '<div class="chat-msg user">' + q.replace(/</g,'&lt;').replace(/>/g,'&gt;') + '</div>';
  document.getElementById('aiQuestion').value = '';
  btn.disabled = true; btn.textContent = '思考中…';

  const THINKING_HTML = '<div class="chat-msg ai" id="thinkingMsg">🤔 正在思考…</div>';
  hist.innerHTML += THINKING_HTML;
  hist.scrollTop = 99999;

  try {

    // 注入天气上下文：将当前选中区县的实时温湿度风力发给AI
    let weatherCtx = '';
    const loc = selectedCountyName || (currentCityData ? currentCityData.name : null);
    if (loc && MONITORING_DATA[loc]) {
      const w = MONITORING_DATA[loc];
      weatherCtx = '[当前' + loc + '天气：温度' + w.temp + '°C，湿度' + w.hum + '%，风力' + w.wind + '级] ';
    }
    const r = await fetch('/api/qwen/chat', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({messages: [{role:'system',content:'你是山西古建筑养护专家，精通木结构、砖石结构、彩绘等保护技术，回答简洁专业'},{role:'user',content: weatherCtx + q}]})
    });
    const d = await r.json();
    const answer = d.choices?.[0]?.message?.content || '无响应';
    const tokens = d.usage?.total_tokens || 0;
    const thinkEl = document.getElementById('thinkingMsg');
    if (thinkEl) thinkEl.remove();
    addChat(answer.replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\n/g,'<br/>') + (tokens ? '<br/><span style="font-size:10px;color:#666;">消耗 '+tokens+' Token</span>' : ''));
  } catch(e) {
    const thinkEl = document.getElementById('thinkingMsg');
    if (thinkEl) thinkEl.remove();
    addChat('❌ 连接失败，请确保后端已启动：<br/><span style="color:#aaa">cd backend && node server.js</span>');
  }
  btn.disabled = false; btn.textContent = '发送';
}

// ===== 摄像头 =====
function startWebcam() {
  var video = document.getElementById('camVideo');
  var btn = document.getElementById('camBtnFallback');
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    if (video) { video.style.display = 'block'; }
    if (btn) { btn.style.display = 'none'; }
    return;
  }
  // 显示 摄像头启动中
  if (btn) { btn.innerHTML = '<span class="cam-icon">' + '⏳' + '</span><span class="cam-label">' + '启动中...' + '</span>'; }
  navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' } }).then(function(stream) {
    if (video) {
      video.srcObject = stream;
      video.style.display = 'block';
      video.setAttribute('playsinline', '');
      // 关键：MediaStream 替换 src 后需要手动 play
      video.play().then(function() {
        console.log('[Webcam] playing');
        startYoloInference(video);
      }).catch(function(e) {
        console.warn('[Webcam] play failed:', e);
      });
    }
    if (btn) {
      btn.innerHTML = '<span class="cam-icon">' + '📹' + '</span><span class="cam-label">' + '录制中' + '</span>';
      btn.style.background = 'rgba(0,255,0,0.3)';
      btn.onclick = function() {
        // 点击停止摄像头，恢复演示视频
        if (video && video.srcObject) {
          var tracks = video.srcObject.getTracks();
          tracks.forEach(function(t) { t.stop(); });
          video.srcObject = null;
          video.load();
          stopYoloInference();
        }
        btn.innerHTML = '<span class="cam-icon">' + '📷' + '</span><span class="cam-label">' + '摄像头' + '</span>';
        btn.style.background = '';
        btn.onclick = startWebcam;
      };
    }
    // 黑屏检测：3秒后用canvas检查是否全黑
    setTimeout(function() {
      if (!video || !video.srcObject) return;
      try {
        var canvas = document.createElement('canvas');
        canvas.width = 1; canvas.height = 1;
        var ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, 1, 1);
        var pixel = ctx.getImageData(0, 0, 1, 1).data;
        // 如果RGB都 10，判定为黑屏
        if (pixel[0] < 10 && pixel[1] < 10 && pixel[2] < 10) {
          console.warn('[Webcam] 黑屏检测->恢复演示视频');
          stopYoloInference();
          var tracks = video.srcObject.getTracks();
          tracks.forEach(function(t) { t.stop(); });
          video.srcObject = null;
          video.load();
          stopYoloInference();
          if (btn) {
            btn.innerHTML = '<span class="cam-icon">' + '⚠️' + '</span><span class="cam-label">' + '摄像头黑屏' + '</span>';
            btn.style.background = 'rgba(255,0,0,0.3)';
            btn.onclick = startWebcam;
          }
        }
      } catch(e) { /* canvas失败 */ }
    }, 3000);
  }).catch(function() {
    // 摄像头被拒或无权限
    if (video) { video.style.display = 'block'; }
    if (btn) {
      btn.innerHTML = '<span class="cam-icon">' + '🛑' + '</span><span class="cam-label">' + '无权限' + '</span>';
      btn.style.background = 'rgba(255,0,0,0.3)';
    }
  });
}




// ===== YOLO 实时病害检测 =====
var yoloSession = null;
var yoloRetryCount = 0;
var yoloMaxRetries = 2;
var yoloRunning = false;
var yoloInterval = null;

// COCO 80 class names (YOLO default output order) + our 2 custom classes
var YOLO_CLASS_NAMES = ['crack', 'spall'];

async function startYoloInference(video) {
  var statusEl = document.getElementById('yoloStatus');
  if (statusEl) {
    statusEl.style.display = 'block';
    statusEl.textContent = '⏳ YOLO 模型加载中...';
    statusEl.className = 'yolo-status loading';
  }

  if (!window.ort) {
    if (statusEl) {
      statusEl.textContent = '❌ ONNX Runtime 未加载';
      statusEl.className = 'yolo-status error';
    }
    console.error('[YOLO] ort not available');
    return;
  }

  try {
    
    // Configure WASM path for local files
    if (typeof ort !== 'undefined' && ort.env && ort.env.wasm) {
      console.log('[YOLO] WASM path set to: modules/');
    }
    // Load model (cached by browser after first load)
    console.log('[YOLO] Creating session from: modules/model/best.onnx'); console.log('[YOLO] WASM config:', ort.env.wasm); yoloSession = await ort.InferenceSession.create('modules/model/best.onnx', {
      graphOptimizationLevel: 'all'
    });
    console.log('[YOLO] Model loaded, input:', yoloSession.inputNames, 'output:', yoloSession.outputNames);

    if (statusEl) {
      statusEl.textContent = '🔍 YOLO 检测中';
      statusEl.className = 'yolo-status';
    }

    yoloRunning = true;
    // Run inference every 800ms
    yoloInterval = setInterval(function() {
      if (!yoloRunning || !yoloSession) return;
      runYoloDetection(video);
    }, 800);

    // First detection immediately
    setTimeout(function() { runYoloDetection(video); }, 500);

  } catch(e) {
    console.error('[YOLO] Load failed:', e);
    var errMsg = e && e.message ? e.message : String(e);
    yoloRetryCount++;
    if (yoloRetryCount <= yoloMaxRetries) {
      if (statusEl) { statusEl.textContent = '⏳ 重试中... (' + yoloRetryCount + '/' + yoloMaxRetries + ') [' + errMsg.substring(0,60) + ']'; }
      await new Promise(function(r) { setTimeout(r, 2000); });
      return startYoloInference(video);
    }
    if (statusEl) {
      statusEl.textContent = '❌ 失败: ' + errMsg.substring(0,120);
      statusEl.className = 'yolo-status error';
    }
  }
}

function stopYoloInference() {
  yoloRunning = false;
  if (yoloInterval) {
    clearInterval(yoloInterval);
    yoloInterval = null;
  }
  var canvas = document.getElementById('yoloCanvas');
  if (canvas) {
    var ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }
  var statusEl = document.getElementById('yoloStatus');
  if (statusEl) { statusEl.style.display = 'none'; }
}

// NMS: remove overlapping boxes
function nmsBoxes(boxes, iouThreshold) {
  boxes.sort(function(a, b) { return b.conf - a.conf; });
  var keep = [];
  for (var i = 0; i < boxes.length; i++) {
    var suppress = false;
    for (var j = 0; j < keep.length; j++) {
      var b1 = boxes[i], b2 = keep[j];
      var xi1 = Math.max(b1.x1, b2.x1), yi1 = Math.max(b1.y1, b2.y1);
      var xi2 = Math.min(b1.x2, b2.x2), yi2 = Math.min(b1.y2, b2.y2);
      var inter = Math.max(0, xi2 - xi1) * Math.max(0, yi2 - yi1);
      var union = (b1.x2 - b1.x1) * (b1.y2 - b1.y1) + (b2.x2 - b2.x1) * (b2.y2 - b2.y1) - inter;
      if (inter / union > iouThreshold) { suppress = true; break; }
    }
    if (!suppress) keep.push(boxes[i]);
  }
  return keep;
}

function runYoloDetection(video) {
  if (video.readyState < 2) return; // not enough data

  var canvas = document.getElementById('yoloCanvas');
  if (!canvas) return;

  // Match canvas size to video display size
  var rect = video.getBoundingClientRect();
  if (rect.width > 0 && rect.height > 0) {
    canvas.width = rect.width;
    canvas.height = rect.height;
  }

  var ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Draw video frame to offscreen canvas for preprocessing
  var offscreen = document.createElement('canvas');
  offscreen.width = 640;
  offscreen.height = 640;
  var offCtx = offscreen.getContext('2d');
  // Letterbox: scale and center
  var vw = video.videoWidth || 640;
  var vh = video.videoHeight || 480;
  var scale = Math.min(640 / vw, 640 / vh);
  var sw = vw * scale;
  var sh = vh * scale;
  var dx = (640 - sw) / 2;
  var dy = (640 - sh) / 2;
  offCtx.fillStyle = '#808080';
  offCtx.fillRect(0, 0, 640, 640);
  offCtx.drawImage(video, dx, dy, sw, sh);

  // Get image data and normalize to [0,1]
  var imgData = offCtx.getImageData(0, 0, 640, 640);
  var pixels = imgData.data; // RGBA
  var input = new Float32Array(3 * 640 * 640);
  for (var i = 0; i < 640 * 640; i++) {
    input[i] = pixels[i * 4] / 255.0;           // R
    input[640*640 + i] = pixels[i * 4 + 1] / 255.0; // G
    input[2*640*640 + i] = pixels[i * 4 + 2] / 255.0; // B
  }

  // Create tensor [1,3,640,640]
  var tensor = new ort.Tensor('float32', input, [1, 3, 640, 640]);

  // Run inference
  yoloSession.run({ 'images': tensor }).then(function(results) {
    var outputName = yoloSession.outputNames[0];
    var output = results[outputName];
    var data = output.data;
    var dims = output.dims;
    var numPreds = dims[2];
    var predSize = dims[1];

    var scaleX = canvas.width / 640;
    var scaleY = canvas.height / 640;
    var boxScaleX = scaleX * scale;
    var boxScaleY = scaleY * scale;

    // === Collect detections (don't draw yet) ===
    var detections = [];
    for (var i = 0; i < numPreds; i++) {
      var base = i * predSize;
      var objConf = data[base + 4];
      if (objConf < 0.85) continue;

      var maxCls = 0, maxClsConf = 0;
      for (var c = 0; c < (predSize - 5); c++) {
        var clsConf = data[base + 5 + c];
        if (clsConf > maxClsConf) { maxClsConf = clsConf; maxCls = c; }
      }
      var conf = objConf * maxClsConf;
      if (conf < 0.85) continue;

      var cx = data[base], cy = data[base + 1];
      var w = data[base + 2], h = data[base + 3];

      var x1 = (cx - w/2 - dx/scale) * boxScaleX;
      var y1 = (cy - h/2 - dy/scale) * boxScaleY;
      var x2 = (cx + w/2 - dx/scale) * boxScaleX;
      var y2 = (cy + h/2 - dy/scale) * boxScaleY;

      x1 = Math.max(0, Math.min(canvas.width, x1));
      y1 = Math.max(0, Math.min(canvas.height, y1));
      x2 = Math.max(0, Math.min(canvas.width, x2));
      y2 = Math.max(0, Math.min(canvas.height, y2));

      if (x2 - x1 < 5 || y2 - y1 < 5) continue;

      detections.push({ x1: x1, y1: y1, x2: x2, y2: y2, conf: conf, maxCls: maxCls });
    }

    // Run NMS to deduplicate
    detections = nmsBoxes(detections, 0.45);

    var count = detections.length;
    var statusEl = document.getElementById('yoloStatus');

    // Sanity check: too many = not a building surface
    if (count > 30) {
      if (statusEl) {
        statusEl.textContent = '⚠️ 非建筑表面 (' + count + ' 个误报已屏蔽) — 请将摄像头对准墙壁';
        statusEl.style.background = 'rgba(200,100,0,0.85)';
      }
      return;
    }

    // === Draw filtered detections ===
    ctx.strokeStyle = '#00ff00';
    ctx.font = '12px sans-serif';
    ctx.lineWidth = 2;
    for (var d = 0; d < count; d++) {
      var det = detections[d];
      if (det.maxCls === 0) { ctx.strokeStyle = '#ff4444'; ctx.fillStyle = '#ff4444'; }
      else { ctx.strokeStyle = '#ffaa00'; ctx.fillStyle = '#ffaa00'; }

      ctx.strokeRect(det.x1, det.y1, det.x2 - det.x1, det.y2 - det.y1);

      var label = (det.maxCls < YOLO_CLASS_NAMES.length ? YOLO_CLASS_NAMES[det.maxCls] : 'cls' + det.maxCls);
      label += ' ' + (det.conf * 100).toFixed(0) + '%';
      var textY = det.y1 > 18 ? det.y1 - 4 : det.y1 + 14;
      ctx.fillStyle = det.maxCls === 0 ? '#ff4444' : '#ffaa00';
      ctx.fillText(label, det.x1 + 2, textY);
    }

    if (statusEl) {
      if (count > 0) {
        statusEl.textContent = '⚠️ 检测到 ' + count + ' 处病害';
        statusEl.style.background = 'rgba(255,60,0,0.85)';
      } else {
        statusEl.textContent = '✅ 无病害';
        statusEl.style.background = 'rgba(0,200,100,0.75)';
      }
    }
  }).catch(function(e) {
    console.error('[YOLO] Inference error:', e);
  });
}

// ===== Open-Meteo 实时天气 =====
function weatherCodeEmoji(code) {
  if (code <= 0) return '\u2600\ufe0f';       // Clear
  if (code <= 3) return '\u26c5';              // Partly cloudy
  if (code <= 48) return '\ud83c\udf2b\ufe0f'; // Fog
  if (code <= 57) return '\ud83c\udf27\ufe0f'; // Drizzle
  if (code <= 67) return '\ud83c\udf27\ufe0f'; // Rain
  if (code <= 77) return '\ud83c\udf28\ufe0f'; // Snow
  if (code <= 82) return '\ud83c\udf27\ufe0f'; // Rain showers
  return '\u26c8\ufe0f';                       // Thunderstorm
}

function weatherCodeDesc(code) {
  if (code <= 0) return '\u6674';
  if (code <= 3) return '\u591a\u4e91';
  if (code <= 48) return '\u96fe';
  if (code <= 57) return '\u5c0f\u96e8';
  if (code <= 67) return '\u96e8';
  if (code <= 77) return '\u96ea';
  if (code <= 82) return '\u9635\u96e8';
  return '\u66b4\u96e8';
}

function riskFromHumidity(hum) {
  if (hum > 70) return '\u9ad8';
  if (hum > 50) return '\u4e2d';
  return '\u4f4e';
}

async function fetchRealWeather() {
  try {
    var r = await fetch('/api/amap/weather/shanxi');
    if (!r.ok) { console.warn('[Weather] fetchRealWeather HTTP ' + r.status); return; }
    var j = await r.json();
    if (!j || !j.cities) return;

    Object.keys(j.cities).forEach(function(cityName) {
      var cd = j.cities[cityName];
      if (!cd || !cd.counties) return;
      cd.counties.forEach(function(co) {
        MONITORING_DATA[co.name] = {
          temp: co.temp, hum: co.hum, wind: co.wind,
          risk: co.risk, desc: co.desc, aqi: co.aqi || 50
        };
      });
      if (cd.forecast && cd.forecast.length > 0) {
        WEATHER_7D_BASE[cityName] = cd.forecast;
      }
    });

    // Build DEFAULT_7D from all city forecasts
    var allFc = [];
    Object.keys(WEATHER_7D_BASE).forEach(function(k) {
      var f = WEATHER_7D_BASE[k]; if (f && f.length > 0) allFc.push(f);
    });
    if (allFc.length > 0) {
      var newDefault = [];
      for (var di = 0; di < 4; di++) {
        var sH = 0, sL = 0, cnt = 0;
        allFc.forEach(function(f) { if (f[di]) { sH += f[di].h; sL += f[di].l; cnt++; } });
        newDefault.push({ d: '', i: '☀️', h: cnt > 0 ? Math.round(sH / cnt) : 20, l: cnt > 0 ? Math.round(sL / cnt) : 10, w: 8 });
      }
      DEFAULT_7D = newDefault;
    }

    // Update top metrics
    var temps = [], hums = [];
    for (var k in MONITORING_DATA) {
      if (MONITORING_DATA.hasOwnProperty(k)) {
        var md = MONITORING_DATA[k];
        if (md.temp !== undefined) temps.push(md.temp);
        if (md.hum !== undefined) hums.push(md.hum);
      }
    }
    if (temps.length > 0) {
      document.getElementById('m-temp').textContent = Math.round(temps.reduce(function(a, b) { return a + b; }) / temps.length) + '°C';
      document.getElementById('m-hum').textContent = Math.round(hums.reduce(function(a, b) { return a + b; }) / hums.length) + '%';
      var rH = 0, rM = 0, rL = 0;
      for (var rk in MONITORING_DATA) {
        if (MONITORING_DATA.hasOwnProperty(rk)) {
          var rd = MONITORING_DATA[rk].risk;
          if (rd === '高') rH++;
          else if (rd === '中') rM++;
          else rL++;
        }
      }
      var mre = document.getElementById('m-risk');
      if (rH > rM && rH > rL) { mre.textContent = '高'; mre.style.color = '#e74c3c'; }
      else if (rM > rL) { mre.textContent = '中'; mre.style.color = '#f39c12'; }
      else { mre.textContent = '低'; mre.style.color = '#2ecc71'; }
    }
    console.log('✅ AMap实时天气已加载');
  } catch (e) { console.warn('[Weather] fetchRealWeather error:', e.message || e); }
}

// ===== 初始化 =====
document.addEventListener('DOMContentLoaded', async function() {
  await fetchRealWeather();
  initMap();

  renderForecast(null);  // 默认显示全省天气
  drawCharts();

  // 每30分钟静默刷新天气，保持 MONITORING_DATA 实时
  setInterval(function() { fetchRealWeather(); }, 1800000);
});
