# -*- coding: utf-8 -*-
"""
山西古建监测系统 - Streamlit 管理后台
功能：总控面板 | 病害检测 | 数据管理 | 气象数据 | 报告生成 | 系统设置
"""
import streamlit as st
import pandas as pd
import numpy as np
import os, json, sqlite3, time, shutil
from datetime import datetime, timedelta
from pathlib import Path
from PIL import Image
import requests
import plotly.express as px
import plotly.graph_objects as go

# ── 路径配置 ──
BASE_DIR = Path(__file__).parent if '__file__' in dir() else Path.cwd()
DATA_DIR = BASE_DIR / 'data'
WEATHER_DIR = DATA_DIR / 'weather'
MODEL_DIR = BASE_DIR / 'model'
DB_PATH = BASE_DIR / 'sites.db'
DISEASE_DIR = DATA_DIR / 'disease_dataset'
UPLOAD_DIR = DATA_DIR / 'uploads'
REPORT_DIR = BASE_DIR / 'reports'
WEATHER_JSON = BASE_DIR / 'data' / 'weather' / 'realtime_weather.json'
ENV_CSV = DATA_DIR / 'environment_data.csv'

for d in [UPLOAD_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── 页面配置 ──
st.set_page_config(
    page_title="古建监测管理后台",
    page_icon="🏯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── 自定义 CSS ──
st.markdown("""
<style>
    .main-header { font-size: 2rem; font-weight: bold; color: #1a5276; margin-bottom: 0; }
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px; padding: 18px; color: white;
        text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .stat-card.green { background: linear-gradient(135deg, #11998e, #38ef7d); }
    .stat-card.orange { background: linear-gradient(135deg, #f093fb, #f5576c); }
    .stat-card.blue { background: linear-gradient(135deg, #4facfe, #00f2fe); }
    .stat-number { font-size: 2.4rem; font-weight: bold; }
    .stat-label { font-size: 0.9rem; opacity: 0.9; margin-top: 4px; }
    .alarm-row { padding: 10px 14px; border-radius: 8px; margin: 4px 0; }
    .alarm-high { background: #ffe0e0; border-left: 4px solid #e74c3c; }
    .alarm-mid { background: #fff3cd; border-left: 4px solid #f39c12; }
    .alarm-low { background: #e0f0ff; border-left: 4px solid #3498db; }
    .stButton>button { border-radius: 8px; font-weight: 500; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════

def get_db():
    """获取数据库连接"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """初始化数据库表"""
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS buildings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            city TEXT, county TEXT,
            lat REAL, lng REAL,
            type TEXT, era TEXT,
            status TEXT DEFAULT '正常',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            building_id INTEGER,
            image_path TEXT,
            disease_type TEXT,
            confidence REAL,
            severity TEXT,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (building_id) REFERENCES buildings(id)
        );
        CREATE TABLE IF NOT EXISTS alarms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            building_id INTEGER,
            alarm_type TEXT,
            level TEXT,
            message TEXT,
            resolved INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (building_id) REFERENCES buildings(id)
        );
        CREATE TABLE IF NOT EXISTS config (
            key TEXT PRIMARY KEY,
            value TEXT
        );
    """)
    conn.commit()
    # 插入默认配置
    defaults = [
        ('confidence_threshold', '0.5'),
        ('alarm_email', ''),
        ('auto_report', 'false'),
        ('model_name', 'yolov8n.pt'),
    ]
    for k, v in defaults:
        try:
            conn.execute("INSERT OR IGNORE INTO config VALUES (?,?)", (k, v))
        except:
            pass
    conn.commit()
    conn.close()


def get_config(key, default=''):
    """读取配置"""
    try:
        conn = get_db()
        row = conn.execute("SELECT value FROM config WHERE key=?", (key,)).fetchone()
        conn.close()
        return row['value'] if row else default
    except:
        return default


def set_config(key, value):
    """写入配置"""
    conn = get_db()
    conn.execute("INSERT OR REPLACE INTO config VALUES (?,?)", (key, str(value)))
    conn.commit()
    conn.close()


def load_weather_data():
    """加载气象数据"""
    data = {'stations': [], 'summary': {}}
    try:
        if WEATHER_JSON.exists():
            with open(WEATHER_JSON, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                for name, info in raw.items():
                    if isinstance(info, dict):
                        cur = info.get('current', {})
                        data['stations'].append({
                            'name': name,
                            'temp': cur.get('temp', 0),
                            'humidity': cur.get('humidity', 0),
                            'wind': cur.get('wind_speed', 0),
                            'pressure': cur.get('pressure', 0),
                            'condition': cur.get('weather', '未知')
                        })
            if data['stations']:
                temps = [s['temp'] for s in data['stations']]
                hums = [s['humidity'] for s in data['stations']]
                data['summary'] = {
                    'avg_temp': round(sum(temps)/len(temps), 1),
                    'max_temp': max(temps),
                    'min_temp': min(temps),
                    'avg_humid': round(sum(hums)/len(hums), 1),
                    'stations': len(data['stations'])
                }
    except Exception as e:
        data['error'] = str(e)
    return data


# 山西11个地级市坐标
SHANXI_CITIES = {
    '太原': (37.87, 112.55), '大同': (40.09, 113.30), '阳泉': (37.86, 113.58),
    '长治': (36.20, 113.12), '晋城': (35.49, 112.85), '朔州': (39.33, 112.43),
    '晋中': (37.69, 112.75), '运城': (35.03, 111.00), '忻州': (38.42, 112.73),
    '临汾': (36.09, 111.52), '吕梁': (37.52, 111.14),
}

WMO_MAP = {
    0: '☀️ 晴', 1: '🌤 大部晴', 2: '⛅ 多云', 3: '☁️ 阴',
    45: '🌫 雾', 48: '🌫 雾凇', 51: '🌧 小雨', 53: '🌧 中雨', 55: '🌧 大雨',
    61: '🌧 阵雨', 63: '🌧 中阵雨', 65: '🌧 大阵雨',
    71: '❄️ 小雪', 73: '❄️ 中雪', 75: '❄️ 大雪',
    80: '🌦 雷阵雨', 81: '🌦 中雷雨', 82: '🌦 强雷雨',
    95: '⛈ 雷暴', 96: '⛈ 雷暴+冰雹', 99: '⛈ 强雷暴',
}

@st.cache_data(ttl=600)
def load_weather_live():
    """从 Open-Meteo 拉取实时气象数据（11个地级市，缓存10分钟）"""
    data = {'stations': [], 'summary': {}, 'source': 'Open-Meteo (实时)'}
    lats = ','.join(str(c[0]) for c in SHANXI_CITIES.values())
    lons = ','.join(str(c[1]) for c in SHANXI_CITIES.values())
    url = (
        'https://api.open-meteo.com/v1/forecast'
        '?latitude=' + lats + '&longitude=' + lons +
        '&current=temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code,pressure_msl'
    )
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        results = resp.json()  # 批量查询返回数组 [{...}, {...}, ...]
        for i, (city, _) in enumerate(SHANXI_CITIES.items()):
            if i < len(results):
                cur = results[i].get('current', {})
                code = int(cur.get('weather_code', 0))
                data['stations'].append({
                    'name': city,
                    'temp': cur.get('temperature_2m', 0),
                    'humidity': cur.get('relative_humidity_2m', 0),
                    'wind': cur.get('wind_speed_10m', 0),
                    'pressure': cur.get('pressure_msl', 0),
                    'condition': WMO_MAP.get(code, f'代码{code}'),
                })
        if data['stations']:
            temps = [s['temp'] for s in data['stations']]
            hums = [s['humidity'] for s in data['stations']]
            data['summary'] = {
                'avg_temp': round(sum(temps)/len(temps), 1),
                'max_temp': max(temps),
                'min_temp': min(temps),
                'avg_humid': round(sum(hums)/len(hums), 1),
                'stations': len(data['stations']),
            }
    except Exception as e:
        data['error'] = str(e)
    return data


def get_stats():
    """获取统计数据"""
    conn = get_db()
    buildings = conn.execute("SELECT COUNT(*) as c FROM buildings").fetchone()['c']
    detections = conn.execute("SELECT COUNT(*) as c FROM detections").fetchone()['c']
    alarms = conn.execute("SELECT COUNT(*) as c FROM alarms WHERE resolved=0").fetchone()['c']
    recent = conn.execute("""
        SELECT d.*, b.name as building_name
        FROM detections d LEFT JOIN buildings b ON d.building_id=b.id
        ORDER BY d.created_at DESC LIMIT 10
    """).fetchall()
    alarm_list = conn.execute("""
        SELECT a.*, b.name as building_name
        FROM alarms a LEFT JOIN buildings b ON a.building_id=b.id
        WHERE a.resolved=0 ORDER BY a.created_at DESC LIMIT 20
    """).fetchall()
    conn.close()

    # 病害分布
    disease_dist = {}
    try:
        if DISEASE_DIR.exists():
            # 英文→中文映射，排除 train/val 等非病害目录
            DISEASE_CN = {'crack': '裂缝', 'spall': '剥落', 'all_diseases': '全部病害'}
            SKIP_DIRS = {'train', 'val', 'images', 'labels'}
            for d in DISEASE_DIR.iterdir():
                if d.is_dir() and d.name not in SKIP_DIRS:
                    count = sum(1 for _ in d.rglob('*.jpg')) + sum(1 for _ in d.rglob('*.png'))
                    if count > 0:
                        disease_dist[DISEASE_CN.get(d.name, d.name)] = count
    except:
        pass

    return {
        'buildings': buildings,
        'detections': detections,
        'active_alarms': alarms,
        'recent': recent,
        'alarm_list': alarm_list,
        'disease_dist': disease_dist
    }


# ═══════════════════════════════════════════════
# 页面：总控面板
# ═══════════════════════════════════════════════

def page_dashboard():
    st.markdown('<p class="main-header">📊 总控面板</p>', unsafe_allow_html=True)
    st.caption(f"最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    stats = get_stats()
    weather = load_weather_live()

    # ── 统计卡片 ──
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'''<div class="stat-card blue">
            <div class="stat-number">{stats['buildings']}</div>
            <div class="stat-label">🏯 监测古建</div></div>''', unsafe_allow_html=True)
    with c2:
        st.markdown(f'''<div class="stat-card green">
            <div class="stat-number">{stats['detections']}</div>
            <div class="stat-label">🔍 检测记录</div></div>''', unsafe_allow_html=True)
    with c3:
        st.markdown(f'''<div class="stat-card orange">
            <div class="stat-number">{stats['active_alarms']}</div>
            <div class="stat-label">🚨 活跃报警</div></div>''', unsafe_allow_html=True)
    with c4:
        temp = weather['summary'].get('avg_temp', '--')
        st.markdown(f'''<div class="stat-card">
            <div class="stat-number">{temp}°C</div>
            <div class="stat-label">🌡️ 全省均温 ({weather['summary'].get('stations',0)}站)</div></div>''',
            unsafe_allow_html=True)

    # ── 下半部分 ──
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("🚨 活跃报警")
        if stats['alarm_list']:
            for a in stats['alarm_list']:
                level_class = {'高': 'alarm-high', '中': 'alarm-mid', '低': 'alarm-low'}
                cls = level_class.get(a['level'], 'alarm-low')
                st.markdown(f'''
                <div class="alarm-row {cls}">
                    <strong>[{a["level"]}]</strong> {a.get("building_name","未知古建")} — {a["message"]}
                    <span style="float:right;font-size:12px;color:#888">{a["created_at"][:16]}</span>
                </div>''', unsafe_allow_html=True)
        else:
            st.success("✅ 当前无活跃报警")

    with col_right:
        st.subheader("📋 最近检测")
        if stats['recent']:
            for r in stats['recent']:
                st.markdown(f'''
                <div style="padding:8px 12px;border-radius:8px;background:#f8f9fa;margin:4px 0;
                            border-left:3px solid #3498db;">
                    <strong>{r.get("building_name","未知")}</strong> — {r["disease_type"]}
                    <span style="float:right;color:#888;font-size:12px">{r["created_at"][:16]}</span>
                </div>''', unsafe_allow_html=True)
        else:
            st.info("暂无检测记录")

    # ── 病害分布 ──
    st.subheader("📈 病害数据分布")
    if stats['disease_dist']:
        df_dist = pd.DataFrame(list(stats['disease_dist'].items()), columns=['类别', '数量'])
        fig = px.bar(df_dist, x='类别', y='数量', text='数量',
                     color_discrete_sequence=['#3498db'],
                     title='各类型病害图片数量')
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("暂无病害数据统计")

    # ── 气象概览 ──
    if weather['stations']:
        st.subheader("🌤️ 全省气象概览")
        df_w = pd.DataFrame(weather['stations'])

        # 上半部：温度分布
        df_sorted_temp = df_w.sort_values('temp', ascending=False)
        fig_t = px.bar(df_sorted_temp, x='name', y='temp',
                      color='temp', color_continuous_scale='thermal',
                      title='全省各市温度分布（°C）',
                      height=350)
        fig_t.update_layout(xaxis_tickangle=-45, xaxis_title='城市', yaxis_title='温度 (°C)')
        st.plotly_chart(fig_t, width='stretch')

        # 下半部：湿度分布
        df_sorted_hum = df_w.sort_values('humidity', ascending=False)
        fig_h = px.bar(df_sorted_hum, x='name', y='humidity',
                      color='humidity', color_continuous_scale='blues',
                      title='全省各市湿度分布（%）',
                      height=350)
        fig_h.update_layout(xaxis_tickangle=-45, xaxis_title='城市', yaxis_title='湿度 (%)')
        st.plotly_chart(fig_h, width='stretch')


# ═══════════════════════════════════════════════
# 页面：病害检测
# ═══════════════════════════════════════════════

def page_detection():
    st.markdown('<p class="main-header">📷 YOLO 病害检测</p>', unsafe_allow_html=True)

    # ── 模型选择 ──
    models = []
    if MODEL_DIR.exists():
        models = [f.name for f in MODEL_DIR.glob('*.pt')]

    col1, col2, col3 = st.columns(3)
    with col1:
        model_name = st.selectbox("检测模型", models, index=0 if models else None)
    with col2:
        conf = st.slider("置信度阈值", 0.1, 1.0, float(get_config('confidence_threshold', '0.5')), 0.05)
    with col3:
        severity = st.selectbox("病害严重度", ['轻微', '中等', '严重'])

    # ── 图片上传 ──
    uploaded = st.file_uploader("上传古建图片", type=['jpg', 'jpeg', 'png', 'bmp'],
                                accept_multiple_files=True)

    if uploaded and model_name:
        with st.spinner(f"正在使用 {model_name} 进行检测..."):
            try:
                from ultralytics import YOLO
                model_path = MODEL_DIR / model_name
                model = YOLO(str(model_path))

                for img_file in uploaded:
                    col_a, col_b = st.columns(2)

                    # 保存临时文件
                    tmp_path = UPLOAD_DIR / img_file.name
                    tmp_path.write_bytes(img_file.read())

                    # YOLO推理
                    results = model(str(tmp_path), conf=conf)

                    with col_a:
                        st.image(str(tmp_path), caption=f"原始: {img_file.name}",
                                width='stretch')

                    with col_b:
                        # 渲染结果
                        for r in results:
                            annotated = r.plot()
                            # 保存标注图
                            out_path = UPLOAD_DIR / f"detected_{img_file.name}"
                            Image.fromarray(annotated[..., ::-1]).save(str(out_path))
                            st.image(annotated, caption=f"检测结果: {img_file.name}",
                                    width='stretch')

                            # 显示检测信息
                            boxes = r.boxes
                            if boxes is not None and len(boxes) > 0:
                                st.success(f"🔍 检测到 {len(boxes)} 处病害")
                                names = r.names if hasattr(r, 'names') else {}
                                for i, box in enumerate(boxes):
                                    cls_id = int(box.cls[0]) if hasattr(box, 'cls') else 0
                                    cls_name = names.get(cls_id, f'类别{cls_id}')
                                    conf_val = float(box.conf[0]) if hasattr(box, 'conf') else 0
                                    st.markdown(f"- **{cls_name}** | 置信度: {conf_val:.2%}")

                                # 保存到数据库
                                conn = get_db()
                                conn.execute("""
                                    INSERT INTO detections (building_id, image_path, disease_type,
                                        confidence, severity, notes)
                                    VALUES (NULL, ?, ?, ?, ?, ?)
                                """, (str(out_path), '裂缝/剥落', conf, severity,
                                      f'模型: {model_name}'))
                                conn.commit()
                                conn.close()
                            else:
                                st.info("未检测到病害")

            except ImportError:
                st.error("❌ ultralytics 未安装。请运行: pip install ultralytics")
            except Exception as e:
                st.error(f"❌ 检测失败: {e}")
    elif uploaded and not models:
        st.warning("⚠️ model/ 目录下未找到 .pt 模型文件，请先放置 YOLO 模型")

    # ── 检测历史 ──
    with st.expander("📜 检测历史记录", expanded=False):
        try:
            conn = get_db()
            records = conn.execute("""
                SELECT d.*, b.name as building_name
                FROM detections d LEFT JOIN buildings b ON d.building_id=b.id
                ORDER BY d.created_at DESC LIMIT 50
            """).fetchall()
            conn.close()
            if records:
                df = pd.DataFrame([dict(r) for r in records])
                st.dataframe(df, width='stretch', hide_index=True)
            else:
                st.info("暂无记录")
        except Exception as e:
            st.info(f"暂无历史记录 ({e})")


# ═══════════════════════════════════════════════
# 页面：数据管理
# ═══════════════════════════════════════════════

def page_data():
    st.markdown('<p class="main-header">📁 数据管理</p>', unsafe_allow_html=True)

    tabs = st.tabs(["🏯 古建管理", "📷 病害图片库", "📊 环境数据", "📋 检测记录"])

    # ── Tab 1: 古建管理 ──
    with tabs[0]:
        col_a, col_b = st.columns([3, 2])
        with col_a:
            conn = get_db()
            buildings = conn.execute("SELECT * FROM buildings ORDER BY id DESC").fetchall()
            if buildings:
                df_b = pd.DataFrame([dict(b) for b in buildings])
                st.dataframe(df_b, width='stretch', hide_index=True,
                            column_config={'id': 'ID', 'name': '名称', 'city': '城市',
                                          'county': '区县', 'type': '类型', 'era': '年代'})
            else:
                st.info("暂无古建数据，请在右侧添加")
            conn.close()
        with col_b:
            st.subheader("➕ 添加古建")
            with st.form("add_building"):
                name = st.text_input("古建名称*")
                city = st.text_input("城市")
                county = st.text_input("区县")
                c1, c2 = st.columns(2)
                with c1:
                    btype = st.selectbox("类型", ['木结构', '砖石结构', '混合结构', '石窟', '其他'])
                with c2:
                    era = st.selectbox("年代", ['唐', '宋/辽/金', '元', '明', '清', '民国', '近现代'])
                submitted = st.form_submit_button("✅ 添加")
                if submitted and name:
                    conn = get_db()
                    conn.execute("INSERT INTO buildings (name, city, county, type, era) VALUES (?,?,?,?,?)",
                               (name, city or '山西省', county, btype, era))
                    conn.commit()
                    conn.close()
                    st.success(f"已添加: {name}")
                    st.rerun()

    # ── Tab 2: 病害图片库 ──
    with tabs[1]:
        disease_types = ['crack', 'spall']
        sel_type = st.selectbox("病害类型", disease_types)
        img_path = DISEASE_DIR / sel_type
        if img_path.exists():
            # 遍历train/val
            images = list(img_path.rglob('*.jpg')) + list(img_path.rglob('*.png'))
            images = sorted(images)[:100]
            if images:
                st.caption(f"共 {len(images)} 张 ({sel_type})")
                cols = st.columns(4)
                for i, img in enumerate(images[:20]):
                    with cols[i % 4]:
                        st.image(str(img), caption=img.name, width='stretch')
            else:
                st.info(f"({sel_type}) 目录下无图片")
        else:
            st.info(f"({sel_type}) 目录不存在")

    # ── Tab 3: 环境数据 ──
    with tabs[2]:
        st.subheader("📊 环境监测数据")
        if ENV_CSV.exists():
            try:
                df_env = pd.read_csv(ENV_CSV, encoding='utf-8')
                st.dataframe(df_env, width='stretch', hide_index=True)
                # 折线图
                if len(df_env.columns) >= 2:
                    numeric = df_env.select_dtypes(include=[np.number])
                    if not numeric.empty:
                        st.line_chart(numeric, height=300)
            except Exception as e:
                st.error(f"读取失败: {e}")
        else:
            st.info("暂无环境数据 CSV")
        st.file_uploader("上传环境数据 (CSV)", type=['csv'], key="upload_env")

    # ── Tab 4: 检测记录 ──
    with tabs[3]:
        st.subheader("🔍 检测记录搜索")
        try:
            conn = get_db()
            total = conn.execute("SELECT COUNT(*) as c FROM detections").fetchone()['c']
            st.caption(f"共 {total} 条记录")

            records = conn.execute("""
                SELECT d.*, b.name as building_name
                FROM detections d LEFT JOIN buildings b ON d.building_id=b.id
                ORDER BY d.created_at DESC LIMIT 100
            """).fetchall()
            conn.close()
            if records:
                df_r = pd.DataFrame([dict(r) for r in records])
                st.dataframe(df_r, width='stretch', hide_index=True)
        except Exception as e:
            st.info(f"暂无检测记录 ({e})")


# ═══════════════════════════════════════════════
# 页面：气象数据
# ═══════════════════════════════════════════════

def page_weather():
    st.markdown('<p class="main-header">🌤️ 气象数据</p>', unsafe_allow_html=True)

    weather = load_weather_live()

    if weather.get('error'):
        st.warning(f"数据加载提示: {weather['error']}")

    stations = weather.get('stations', [])

    if stations:
        # 概览卡片
        s = weather['summary']
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📡 气象站", f"{s['stations']} 个")
        c2.metric("🌡️ 均温", f"{s['avg_temp']}°C")
        c3.metric("🔺 最高温", f"{s['max_temp']}°C")
        c4.metric("💧 均湿度", f"{s['avg_humid']}%")

        # 站点表格
        df = pd.DataFrame(stations)
        if not df.empty:
            # 排序筛选
            sort_col = st.selectbox("排序", df.select_dtypes(include=[np.number]).columns.tolist(),
                                   index=0)
            df_sorted = df.sort_values(sort_col, ascending=False)
            st.dataframe(df_sorted, width='stretch', hide_index=True)

            # 对比图
            st.subheader("📈 站点数据对比")
            selected = st.multiselect("选择站点对比", df_sorted['name'].tolist(),
                                     default=df_sorted['name'].tolist()[:5])
            if selected:
                df_sel = df_sorted[df_sorted['name'].isin(selected)]
                tab1, tab2 = st.tabs(["温度对比", "湿度对比"])
                with tab1:
                    fig = px.bar(df_sel, x='name', y='temp', color='name',
                                title='站点温度对比 (°C)')
                    st.plotly_chart(fig, width='stretch')
                with tab2:
                    fig = px.bar(df_sel, x='name', y='humidity', color='name',
                                title='站点湿度对比 (%)',
                                color_discrete_sequence=px.colors.sequential.Blues)
                    st.plotly_chart(fig, width='stretch')
    else:
        st.info("📭 暂无气象站数据。请确保 data/weather/realtime_weather.json 存在")

    # 上传
    st.file_uploader("上传气象数据 JSON", type=['json'], key="upload_weather")


# ═══════════════════════════════════════════════
# 页面：报告生成
# ═══════════════════════════════════════════════

def page_report():
    st.markdown('<p class="main-header">📝 报告生成</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        report_type = st.selectbox("报告类型", ['病害检测报告', '环境数据报告', '综合巡检报告'])
    with col2:
        export_format = st.selectbox("导出格式", ['Excel (.xlsx)', 'CSV (.csv)'])

    st.divider()

    # 预览
    st.subheader("📋 数据预览")

    with st.spinner("加载数据..."):
        conn = get_db()
        buildings = pd.DataFrame([dict(r) for r in
            conn.execute("SELECT * FROM buildings ORDER BY id DESC LIMIT 200").fetchall()])
        detections = pd.DataFrame([dict(r) for r in
            conn.execute("""
                SELECT d.*, b.name as building_name
                FROM detections d LEFT JOIN buildings b ON d.building_id=b.id
                ORDER BY d.created_at DESC LIMIT 200
            """).fetchall()])
        alarms = pd.DataFrame([dict(r) for r in
            conn.execute("SELECT * FROM alarms ORDER BY created_at DESC LIMIT 200").fetchall()])
        conn.close()

    tab_b, tab_d, tab_a = st.tabs(["古建列表", "检测记录", "报警记录"])

    with tab_b:
        if not buildings.empty:
            st.dataframe(buildings, width='stretch', hide_index=True)
            st.caption(f"共 {len(buildings)} 条")
        else:
            st.info("暂无古建数据")
    with tab_d:
        if not detections.empty:
            st.dataframe(detections, width='stretch', hide_index=True)
            st.caption(f"共 {len(detections)} 条")
        else:
            st.info("暂无检测记录")
    with tab_a:
        if not alarms.empty:
            st.dataframe(alarms, width='stretch', hide_index=True)
            st.caption(f"共 {len(alarms)} 条")
        else:
            st.info("暂无报警记录")

    st.divider()

    # 导出按钮
    if st.button("🚀 生成并下载报告", width='stretch', type="primary"):
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{report_type}_{now}"

        if export_format.startswith('Excel'):
            filename += '.xlsx'
            filepath = REPORT_DIR / filename
            try:
                with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                    if not buildings.empty:
                        buildings.to_excel(writer, sheet_name='古建列表', index=False)
                    if not detections.empty:
                        detections.to_excel(writer, sheet_name='检测记录', index=False)
                    if not alarms.empty:
                        alarms.to_excel(writer, sheet_name='报警记录', index=False)
                    # 汇总
                    summary = pd.DataFrame({
                        '项目': ['报告类型', '生成时间', '古建总数', '检测总数', '活跃报警'],
                        '值': [report_type, datetime.now().strftime('%Y-%m-%d %H:%M'),
                              len(buildings), len(detections), len(alarms[alarms.get('resolved',0)==0])]
                    })
                    summary.to_excel(writer, sheet_name='汇总', index=False)

                with open(filepath, 'rb') as f:
                    st.download_button("📥 下载 Excel", f.read(), filename,
                                      mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                st.success(f"✅ 已生成: {filename}")
            except Exception as e:
                st.error(f"导出失败: {e}")

        elif export_format.startswith('CSV'):
            filename += '.zip'
            try:
                import zipfile, io
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
                    if not buildings.empty:
                        zf.writestr('古建列表.csv', buildings.to_csv(index=False, encoding='utf-8-sig'))
                    if not detections.empty:
                        zf.writestr('检测记录.csv', detections.to_csv(index=False, encoding='utf-8-sig'))
                    if not alarms.empty:
                        zf.writestr('报警记录.csv', alarms.to_csv(index=False, encoding='utf-8-sig'))
                buf.seek(0)
                st.download_button("📥 下载 CSV (ZIP)", buf.read(), filename, mime="application/zip")
                st.success(f"✅ 已打包: {filename}")
            except Exception as e:
                st.error(f"导出失败: {e}")


# ═══════════════════════════════════════════════
# 页面：系统设置
# ═══════════════════════════════════════════════

def page_settings():
    st.markdown('<p class="main-header">⚙️ 系统设置</p>', unsafe_allow_html=True)

    s1, s2, s3 = st.tabs(["🤖 AI 知识库", "📊 检测参数", "🔧 系统配置"])

    # ── AI 知识库 ──
    with s1:
        st.subheader("📚 古建筑养护知识库")
        st.caption("编辑 ai_server.py 中的 KNOWLEDGE_BASE，下次重启AI服务生效")

        kb_items = {
            "裂缝": "古建裂缝修补：1)表面裂缝→环氧树脂注浆 2)结构裂缝→专业评估加固 3)温度裂缝→控制温湿度",
            "潮湿": "防潮处理：1)改善通风 2)控制湿度40-60% 3)使用防霉剂 4)检查防水层",
            "冬季": "冬季防冻：1)水管排空 2)木构防潮 3)石质防冻融 4)室内温控",
            "防虫": "木构防虫：1)定期检查 2)防虫剂处理 3)保持干燥 4)更换受损构件",
            "剥落": "砖墙剥落：1)清理剥落部位 2)原配比材料修补 3)防水处理 4)定期检查"
        }

        for i, (title, content) in enumerate(kb_items.items()):
            with st.expander(f"📖 {title}", expanded=False):
                new_content = st.text_area(f"编辑 - {title}", content, height=120,
                                          key=f"kb_{i}")
                if new_content != content:
                    st.caption("💡 修改需手动同步到 ai_server.py 的 KNOWLEDGE_BASE 字典")
        st.info("💡 知识库编辑完成后，需重启 ai_server.py 使更改生效")

    # ── 检测参数 ──
    with s2:
        st.subheader("🔧 YOLO 检测参数")
        conf_thresh = st.slider("默认置信度阈值", 0.1, 0.95,
                                float(get_config('confidence_threshold', '0.5')), 0.05)
        if st.button("💾 保存阈值"):
            set_config('confidence_threshold', str(conf_thresh))
            st.success(f"已保存: {conf_thresh}")

        default_model = st.selectbox("默认模型",
            ["yolov8n.pt", "yolov8m.pt", "yolo11n.pt"],
            index=["yolov8n.pt", "yolov8m.pt", "yolo11n.pt"].index(
                get_config('model_name', 'yolov8n.pt')))
        if st.button("💾 保存模型"):
            set_config('model_name', default_model)
            st.success(f"已保存: {default_model}")

    # ── 系统配置 ──
    with s3:
        st.subheader("🔧 系统信息")
        info = {
            '项目路径': str(BASE_DIR),
            '数据库': str(DB_PATH),
            '模型目录': str(MODEL_DIR),
            '数据目录': str(DATA_DIR),
            'AI服务端口': '5188',
            'Flask后端': str(BASE_DIR / 'ai_server.py'),
            'Streamlit后台': str(BASE_DIR / 'admin.py'),
        }
        for k, v in info.items():
            exists = Path(v).exists() if '/' not in v and '\\' not in v else True
            icon = '✅' if exists else '❌'
            st.text(f"{icon}  {k}: {v}")

        st.divider()
        if st.button("🔄 重新初始化数据库", type="secondary"):
            init_db()
            st.success("数据库已初始化")


# ═══════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════

def main():
    # 侧边栏
    with st.sidebar:
        st.markdown("## 🏯 古建监测后台")
        st.markdown("---")
        menu = st.radio(
            "导航菜单",
            ["📊 总控面板", "📷 病害检测", "📁 数据管理",
             "🌤️ 气象数据", "📝 报告生成", "⚙️ 系统设置"],
            label_visibility="collapsed"
        )
        st.markdown("---")
        st.caption(f"项目: {BASE_DIR.name}")
        st.caption("© 2026 山西古建养护")

    # 初始化数据库
    try:
        init_db()
    except:
        st.sidebar.warning("⚠️ 数据库初始化异常")

    # 路由
    pages = {
        "📊 总控面板": page_dashboard,
        "📷 病害检测": page_detection,
        "📁 数据管理": page_data,
        "🌤️ 气象数据": page_weather,
        "📝 报告生成": page_report,
        "⚙️ 系统设置": page_settings,
    }

    pages.get(menu, page_dashboard)()


if __name__ == "__main__":
    main()
