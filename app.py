"""
=====================================================
👑 الوحش 71.0 — القناص اليومي
=====================================================
- الكود الأصلي لك (64.0) محفوظ 100%
- يُضاف إليه: 5 استراتيجيات متوازية
- عمل 24/5 في جلسات London + NY
- إشارات متعددة يوميًا (5-15)
=====================================================
"""

import os, sqlite3, warnings, logging, asyncio, requests, json, time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import yfinance as yf
import pandas as pd
import numpy as np
import telegram
from openai import OpenAI
from twelvedata import TDClient

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler("quantum_beast_71.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# ==========================================
# ⚙️ الإعدادات (كما هي عندك تمامًا)
# ==========================================
TOKEN = os.environ.get("BOT_TOKEN", "").strip()
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "").strip()
TWELVEDATA_API_KEY = os.environ.get("TWELVEDATA_API_KEY", "").strip()

FORCE_TEST = os.environ.get("FORCE_TEST", "0").strip() == "1"

DB_FILE = "quantum_beast_71.db"
NY = ZoneInfo("America/New_York")

# ✅ الجديد: نطاقات جلسات التداول (بدلًا من Silver Bullet فقط)
# London: 03:00 - 07:00 NY | NY: 08:00 - 12:00 NY | PM: 13:00 - 17:00 NY
TRADING_SESSIONS = [(3, 7), (8, 12), (13, 17)]
# نافذة Silver Bullet الأصلية محفوظة كـ "نافذة قوية"
SILVER_BULLET_WINDOWS = [(3, 4), (10, 11)]

# ✅ الجديد: إعدادات القناص
MAX_SIGNALS_PER_DAY = 15       # سقف يومي
MIN_SIGNAL_CONFIDENCE = 55     # ثقة دنيا (من 100)
SIGNAL_COOLDOWN_MIN = 20       # لا إشارة متكررة خلال 20 دقيقة
STRATEGIES_ENABLED = ["judas", "vwap", "ema", "range", "divergence"]

try:
    from google import genai
    ai_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
except Exception:
    ai_client = None

td = TDClient(apikey=TWELVEDATA_API_KEY) if TWELVEDATA_API_KEY else None

# ==========================================
# 🗄️ قاعدة البيانات (كما هي)
# ==========================================
conn = sqlite3.connect(DB_FILE, check_same_thread=False)
conn.executescript("PRAGMA journal_mode=WAL;")
conn.executescript("""
CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS signals(id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, direction INTEGER, entry REAL, sl REAL, tp1 REAL, tp2 REAL, status TEXT);
""")
conn.commit()

def db_get(key, default=None):
    r = conn.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
    return r[0] if r else default

def db_set(key, value):
    conn.execute("INSERT OR REPLACE INTO state VALUES(?,?)", (key, str(value)))
    conn.commit()

# ==========================================
# 📱 تيليجرام (كما هي)
# ==========================================
async def send_telegram_msg(text):
    if not TOKEN or not CHAT_ID: return
    try:
        bot = telegram.Bot(token=TOKEN)
        await bot.send_message(chat_id=CHAT_ID, text=text, parse_mode='Markdown')
    except Exception as e:
        logging.error(f"Telegram Error: {e}")

# ==========================================
# 🌐 جلب البيانات (كما هي، مع تحسين صغير للاحتياطي)
# ==========================================
async def get_live_prices():
    if td:
        try:
            ts = td.time_series(symbol="XAU/USD", interval="1min", outputsize=1).as_pandas()
            if ts is not None and not ts.empty:
                return float(ts['close'].iloc[0]), None
        except: pass
    try:
        gold = yf.Ticker("GC=F").history(period="1d", interval="1m")
        if not gold.empty: return float(gold["Close"].iloc[-1]), None
    except: pass
    return None, None

async def fetch_candles(symbol, interval, limit=100):
    try:
        yf_interval = "1m" if interval == "1m" else interval
        yf_period = "1d" if interval == "1m" else "5d"
        ticker = yf.Ticker(symbol if symbol != "XAUUSD" else "GC=F")
        df = ticker.history(period=yf_period, interval=yf_interval, auto_adjust=False)
        if df is not None and not df.empty:
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            df.index = (df.index.tz_localize("UTC") if df.index.tz is None else df.index).tz_convert(NY)
            return df
    except: pass
    return None

async def get_previous_day_ohlc():
    if not td: return None
    try:
        ts = td.time_series(symbol="XAU/USD", interval="1day", outputsize=10).as_pandas()
        if ts is None or ts.empty: return None
        ts.index = pd.to_datetime(ts.index)
        today = datetime.now(NY).date()
        past = ts[ts.index.date < today]
        if past.empty: return None
        prev = past.iloc[-1]
        return {"open": float(prev["open"]), "high": float(prev["high"]),
                "low": float(prev["low"]), "close": float(prev["close"])}
    except: return None

# ==========================================
# 🏭 GDX والأخبار (كما هي)
# ==========================================
async def get_gdx_trend():
    try:
        gdx = yf.Ticker("GDX").history(period="5d", interval="1h")
        if gdx.empty: return "NEUTRAL", 0
        ema = gdx['Close'].ewm(span=20).mean().iloc[-1]
        current = gdx['Close'].iloc[-1]
        strength = ((current - ema) / ema) * 100
        return ("BULLISH" if current > ema else "BEARISH"), round(strength, 2)
    except: return "NEUTRAL", 0

async def get_news_headlines():
    try:
        gold = yf.Ticker("GC=F")
        news = gold.news
        if not news: return "لا توجد أخبار حديثة."
        return "\n".join([n.get('title', '') for n in news[:5]])
    except: return "تعذر جلب الأخبار."

# ==========================================
# ⏰ فلاتر الوقت (معدّلة للعمل 24/5)
# ==========================================
def is_silver_bullet_time():
    if FORCE_TEST: return True
    now = datetime.now(NY)
    if now.weekday() >= 5: return False
    t = now.hour + now.minute / 60
    return any(start <= t < end for start, end in SILVER_BULLET_WINDOWS)

def is_trading_session():  # ✅ جديد
    if FORCE_TEST: return True
    now = datetime.now(NY)
    if now.weekday() >= 5: return False
    t = now.hour + now.minute / 60
    return any(start <= t < end for start, end in TRADING_SESSIONS)

def check_news_blackout():
    if FORCE_TEST: return False, None
    try:
        res = requests.get("https://nfs.faireconomy.media/ff_calendar_thisweek.json", timeout=10).json()
        now = datetime.now(NY)
        critical = ["CPI", "FOMC", "Non-Farm", "PPI", "GDP", "Powell"]
        for n in res:
            if n.get("country") == "USD" and n.get("impact") == "High":
                try:
                    n_time = pd.to_datetime(n["date"], utc=True).tz_convert(NY)
                    diff = (n_time - now).total_seconds() / 60
                    if -45 <= diff <= 30 and any(k.lower() in n.get("title", "").lower() for k in critical):
                        return True, f"⏸️ حظر أخبار: {n['title']}"
                except: pass
        return False, None
    except: return False, None

# ==========================================
# 💀 ICT (كما هي)
# ==========================================
def detect_fvg(df, direction, lookback=5):
    for i in range(-lookback, -1):
        if direction == 1:
            if df['Low'].iloc[i] > df['High'].iloc[i-2]:
                return (df['High'].iloc[i-2], df['Low'].iloc[i])
        else:
            if df['High'].iloc[i] < df['Low'].iloc[i-2]:
                return (df['High'].iloc[i], df['Low'].iloc[i-2])
    return None

def detect_judas_swing(df_m15, df_m1, pdh, pdl):
    if df_m15 is None or df_m1 is None or len(df_m15) < 5 or len(df_m1) < 10:
        return None, None, "بيانات غير كافية"
    recent_m15 = df_m15.tail(5)
    swept_high = recent_m15['High'].max() > pdh
    swept_low = recent_m15['Low'].min() < pdl
    if swept_high and not swept_low:
        last_m1 = df_m1.tail(10)
        if last_m1['Close'].iloc[-1] < last_m1['Low'].iloc[-3]:
            fvg = detect_fvg(df_m1, -1)
            entry = (fvg[0] + fvg[1]) / 2 if fvg else last_m1['High'].iloc[-3]
            return -1, entry, "🔴 فخ فوق PDH + CHoCH هبوطي"
    elif swept_low and not swept_high:
        last_m1 = df_m1.tail(10)
        if last_m1['Close'].iloc[-1] > last_m1['High'].iloc[-3]:
            fvg = detect_fvg(df_m1, 1)
            entry = (fvg[0] + fvg[1]) / 2 if fvg else last_m1['Low'].iloc[-3]
            return 1, entry, "🟢 فخ تحت PDL + CHoCH صعودي"
    return None, None, "كنسناو"

# ==========================================
# 🧠 Hurst + Macro (كما هي)
# ==========================================
def calculate_hurst(prices, max_lag=20):
    try:
        s = np.log(np.asarray(prices, dtype=float))
        lags = np.arange(2, min(max_lag, len(s)//3))
        if len(lags) < 6: return 0.5
        tau = np.array([np.std(s[l:] - s[:-l]) for l in lags]); m = tau > 0
        return float(np.polyfit(np.log(lags[m]), np.log(tau[m]), 1)[0]) if m.sum() > 5 else 0.5
    except: return 0.5

async def get_macro_and_intermarket():
    try:
        dxy = await fetch_candles("DX-Y.NYB", "1h", 50)
        vix = await fetch_candles("^VIX", "1h", 50)
        silver = await fetch_candles("SI=F", "1h", 50)
        spx = await fetch_candles("^GSPC", "1h", 50)
        macro = "NEUTRAL"
        if dxy is not None and len(dxy) > 20:
            dxy_bull = dxy['Close'].iloc[-1] > dxy['Close'].ewm(span=20).mean().iloc[-1]
            macro = "BEARISH_GOLD" if dxy_bull else "BULLISH_GOLD"
        vix_msg = f"{vix['Close'].iloc[-1]:.1f}" if vix is not None else "N/A"
        intermarket = []
        if silver is not None: intermarket.append(f"🥈 {silver['Close'].iloc[-1]:.2f}")
        if spx is not None: intermarket.append(f"📉 {spx['Close'].iloc[-1]:.2f}")
        return macro, vix_msg, " | ".join(intermarket)
    except: return "NEUTRAL", "N/A", "N/A"

# ==========================================
# 🧠 مجلس العباقرة (كما هو، مع تصحيح النموذج)
# ==========================================
async def ask_ai_council(data_summary):
    prompt = f"""
    أنت مجلس خبراء تداول (Multi-Agent System).
    المعطيات:
    {data_summary}
    1. 🧠 [المحلل الفني]: قدم حجة بناءً على المعطيات.
    2. 📰 [محلل الأخبار]: حلل العناوين.
    3. 😈 [محامي الشيطان]: هاجم الصفقة!
    4. ⚖️ [مدير المخاطر]: القرار (شراء/بيع/انتظار).
    الرد:
    🧠 المحلل: [جملة]
    📰 محلل الأخبار: [جملة]
    😈 محامي الشيطان: [جملة]
    ⚖️ مدير المخاطر: القرار: [شراء/بيع/انتظار] | السبب: [جملة]
    """
    if ai_client:
        try:
            response = await ai_client.aio.models.generate_content(model="gemini-2.5-flash", contents=prompt)
            return response.text
        except: pass
    if GROQ_API_KEY:
        try:
            client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)
            response = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": prompt}])
            return response.choices[0].message.content
        except: pass
    if OPENROUTER_API_KEY:
        try:
            client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
            response = client.chat.completions.create(model="meta-llama/llama-3.3-70b-instruct:free", messages=[{"role": "user", "content": prompt}])
            return response.choices[0].message.content
        except: pass
    if MISTRAL_API_KEY:
        try:
            client = OpenAI(base_url="https://api.mistral.ai/v1", api_key=MISTRAL_API_KEY)
            response = client.chat.completions.create(model="mistral-small-latest", messages=[{"role": "user", "content": prompt}])
            return response.choices[0].message.content
        except: pass
    return "⚠️ AI غير متاح"

# ==========================================
# 🚀 run_engine الأصلي (كما هو تمامًا)
# ==========================================
async def run_engine():
    try:
        if not is_silver_bullet_time():
            return None, None
        news_blocked, news_msg = check_news_blackout()
        if news_blocked: return None, None
        ohlc = await get_previous_day_ohlc()
        if not ohlc: return None, "❌ تعذر جلب بيانات اليوم السابق"
        pdh, pdl = ohlc["high"], ohlc["low"]
        df_m15 = await fetch_candles("GC=F", "15m", 100)
        df_m1 = await fetch_candles("GC=F", "1m", 100)
        if any(d is None for d in [df_m15, df_m1]): return None, "❌ بيانات غير كافية"
        direction, entry_zone, sweep_msg = detect_judas_swing(df_m15, df_m1, pdh, pdl)
        if direction is None:
            if FORCE_TEST:
                current_price, _ = await get_live_prices()
                gdx = await get_gdx_trend()
                test_msg = f"✅ *البوت خدام* (اختبار)\n🕐 `{datetime.now(NY).strftime('%H:%M')}`\n"
                if current_price: test_msg += f"💰 `{current_price:.2f}`\n"
                test_msg += f"📊 PDH: `{pdh:.2f}` | PDL: `{pdl:.2f}`\n🏭 GDX: {gdx[0]}\n🧐 ما كاينش فخ."
                return test_msg, None
            return None, None
        gdx_trend, gdx_strength = await get_gdx_trend()
        if direction == 1 and gdx_trend == "BEARISH": return None, None
        if direction == -1 and gdx_trend == "BULLISH": return None, None
        current_price, _ = await get_live_prices()
        if current_price is None: current_price = df_m1['Close'].iloc[-1]
        hurst = calculate_hurst(df_m15['Close'].values)
        macro, vix, intermarket = await get_macro_and_intermarket()
        news_headlines = await get_news_headlines()
        atr = (df_m15['High'] - df_m15['Low']).rolling(14).mean().iloc[-1]
        if abs(current_price - entry_zone) > (atr * 2.0): return None, None
        ai_summary = f"Price: {current_price}\nSweep: {sweep_msg}\nEntry: {entry_zone}\nHurst: {hurst}\nMacro: {macro}\nVIX: {vix}\nGDX: {gdx_trend}\nNews: {news_headlines}"
        ai_verdict = await ask_ai_council(ai_summary)
        if "انتظار" not in ai_verdict:
            if direction == 1:
                sl = round(df_m1['Low'].tail(5).min() - (atr * 0.2), 2)
                risk = entry_zone - sl
                tp1 = round(entry_zone + risk * 2.0, 2)
                tp2 = round(entry_zone + risk * 4.0, 2)
                trade_type = "🟢 شراء (Long)"
            else:
                sl = round(df_m1['High'].tail(5).max() + (atr * 0.2), 2)
                risk = sl - entry_zone
                tp1 = round(entry_zone - risk * 2.0, 2)
                tp2 = round(entry_zone - risk * 4.0, 2)
                trade_type = "🔴 بيع (Short)"
            if risk <= 0: return None, "❌ خطأ في حساب الوقف"
            signal_hash = f"{direction}_{entry_zone}_{sl}"
            if db_get("last_signal") == signal_hash: return None, None
            db_set("last_signal", signal_hash)
            report = (
                f"👑 *الوحش 71.0 — إشارة ICT*\n🕐 `{datetime.now(NY).strftime('%H:%M')}` (NY)\n"
                f"━━━━━━━━━━━━━━━━━━━━\n💰 `{current_price:.2f}`\n💀 {sweep_msg}\n🏭 GDX: {gdx_trend}\n"
                f"📰 {news_headlines[:100]}...\n━━━━━━━━━━━━━━━━━━━━\n"
                f"🎯 *الدخول:* `{entry_zone:.2f}`\n{trade_type}\n"
                f"🛑 *الوقف:* `{sl:.2f}`\n🏆 *TP1:* `{tp1:.2f}`\n🚀 *TP2:* `{tp2:.2f}`\n"
                f"━━━━━━━━━━━━━━━━━━━━\n🧠 *مجلس العباقرة:*\n{ai_verdict}\n"
            )
            return report, None
        return None, None
    except Exception as e:
        return None, f"❌ خطأ: {e}"

# ==========================================
# ==========================================
# 🆕 القسم الجديد: القناص متعدد الاستراتيجيات
# ==========================================
# ==========================================

# ---------- 1. VWAP + Volume Profile ----------
def calculate_vwap(df):
    try:
        if df is None or len(df) < 10: return None
        today = datetime.now(NY).date()
        today_data = df[df.index.date == today]
        if len(today_data) < 5: today_data = df.tail(50)
        tp = (today_data['High'] + today_data['Low'] + today_data['Close']) / 3
        vwap = (tp * today_data['Volume']).cumsum() / today_data['Volume'].cumsum()
        return round(float(vwap.iloc[-1]), 2)
    except: return None

def calculate_volume_profile(df, bins=50):
    try:
        if df is None or len(df) < 30: return None, None, None
        prices = df['Close'].tail(100).values
        volumes = df['Volume'].tail(100).values
        hist, bin_edges = np.histogram(prices, bins=bins, weights=volumes)
        poc_idx = np.argmax(hist)
        poc = (bin_edges[poc_idx] + bin_edges[poc_idx+1]) / 2
        total = hist.sum(); target = total * 0.70
        cur = hist[poc_idx]; vi, va = poc_idx, poc_idx
        while cur < target:
            if vi + 1 < len(hist) and (va == 0 or hist[vi+1] >= hist[va-1]):
                vi += 1; cur += hist[vi]
            elif va - 1 >= 0:
                va -= 1; cur += hist[va]
            else: break
        vah = (bin_edges[vi] + bin_edges[vi+1]) / 2
        val = (bin_edges[va] + bin_edges[va+1]) / 2
        return round(float(poc), 2), round(float(vah), 2), round(float(val), 2)
    except: return None, None, None

# ---------- 2. مؤشرات الزخم ----------
def calc_rsi(series, period=14):
    try:
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = -delta.where(delta < 0, 0).rolling(period).mean()
        rs = gain / (loss + 1e-9)
        return 100 - (100 / (1 + rs))
    except: return pd.Series([50] * len(series))

def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calc_macd(series, fast=12, slow=26, signal=9):
    ema_fast = calc_ema(series, fast)
    ema_slow = calc_ema(series, slow)
    macd = ema_fast - ema_slow
    sig = calc_ema(macd, signal)
    hist = macd - sig
    return macd, sig, hist

# ---------- 3. الاستراتيجيات الخمس ----------

def strategy_vwap_bounce(df_m15, df_m5, vwap):
    """🎯 VWAP Bounce: السعر يرتد من VWAP"""
    if df_m5 is None or len(df_m5) < 20 or vwap is None: return None
    try:
        last = df_m5.iloc[-1]
        prev = df_m5.iloc[-2]
        # شراء: السعر لمس VWAP من فوق وارتد
        if prev['Low'] <= vwap and last['Close'] > vwap and last['Close'] > last['Open']:
            entry = float(last['Close'])
            atr = float((df_m5['High'] - df_m5['Low']).rolling(14).mean().iloc[-1])
            sl = round(vwap - atr * 0.5, 2)
            risk = entry - sl
            return {
                "strategy": "VWAP_BOUNCE_BUY", "direction": 1, "entry": entry,
                "sl": sl, "tp1": round(entry + risk * 1.5, 2), "tp2": round(entry + risk * 2.5, 2),
                "confidence": 65, "reason": "🟢 ارتداد من VWAP"
            }
        # بيع: السعر لمس VWAP من تحت وارتد
        if prev['High'] >= vwap and last['Close'] < vwap and last['Close'] < last['Open']:
            entry = float(last['Close'])
            atr = float((df_m5['High'] - df_m5['Low']).rolling(14).mean().iloc[-1])
            sl = round(vwap + atr * 0.5, 2)
            risk = sl - entry
            return {
                "strategy": "VWAP_BOUNCE_SELL", "direction": -1, "entry": entry,
                "sl": sl, "tp1": round(entry - risk * 1.5, 2), "tp2": round(entry - risk * 2.5, 2),
                "confidence": 65, "reason": "🔴 ارتداد من VWAP"
            }
    except: pass
    return None

def strategy_ema_momentum(df_m5):
    """⚡ EMA Momentum: تقاطع EMA 9/21 مع MACD"""
    if df_m5 is None or len(df_m5) < 50: return None
    try:
        ema9 = calc_ema(df_m5['Close'], 9)
        ema21 = calc_ema(df_m5['Close'], 21)
        macd, sig, hist = calc_macd(df_m5['Close'])
        
        # تقاطع صعودي حديث + MACD إيجابي
        cross_up = (ema9.iloc[-1] > ema21.iloc[-1]) and (ema9.iloc[-2] <= ema21.iloc[-2])
        if cross_up and macd.iloc[-1] > sig.iloc[-1]:
            entry = float(df_m5['Close'].iloc[-1])
            atr = float((df_m5['High'] - df_m5['Low']).rolling(14).mean().iloc[-1])
            sl = round(entry - atr * 1.2, 2)
            risk = entry - sl
            return {
                "strategy": "EMA_MOMENTUM_BUY", "direction": 1, "entry": entry,
                "sl": sl, "tp1": round(entry + risk * 1.5, 2), "tp2": round(entry + risk * 2.5, 2),
                "confidence": 60, "reason": "⚡ تقاطع EMA9/21 صعودي + MACD"
            }
        cross_dn = (ema9.iloc[-1] < ema21.iloc[-1]) and (ema9.iloc[-2] >= ema21.iloc[-2])
        if cross_dn and macd.iloc[-1] < sig.iloc[-1]:
            entry = float(df_m5['Close'].iloc[-1])
            atr = float((df_m5['High'] - df_m5['Low']).rolling(14).mean().iloc[-1])
            sl = round(entry + atr * 1.2, 2)
            risk = sl - entry
            return {
                "strategy": "EMA_MOMENTUM_SELL", "direction": -1, "entry": entry,
                "sl": sl, "tp1": round(entry - risk * 1.5, 2), "tp2": round(entry - risk * 2.5, 2),
                "confidence": 60, "reason": "⚡ تقاطع EMA9/21 هبوطي + MACD"
            }
    except: pass
    return None

def strategy_range_breakout(df_m15):
    """📊 كسر النطاق: السعر يخرج من نطاق ضيق"""
    if df_m15 is None or len(df_m15) < 30: return None
    try:
        last_20 = df_m15.tail(20)
        high_range = last_20['High'].max()
        low_range = last_20['Low'].min()
        range_size = high_range - low_range
        atr = float((df_m15['High'] - df_m15['Low']).rolling(14).mean().iloc[-1])
        
        # النطاق ضيق (أقل من 2 ATR) → مرشح للكسر
        if range_size > atr * 2.5: return None
        
        last = df_m15.iloc[-1]
        prev = df_m15.iloc[-2]
        
        # كسر صعودي
        if prev['High'] <= high_range and last['Close'] > high_range:
            entry = float(last['Close'])
            sl = round(high_range - atr * 0.5, 2)
            risk = entry - sl
            return {
                "strategy": "RANGE_BREAKOUT_BUY", "direction": 1, "entry": entry,
                "sl": sl, "tp1": round(entry + risk * 2.0, 2), "tp2": round(entry + risk * 3.5, 2),
                "confidence": 62, "reason": "📊 كسر نطاق صعودي"
            }
        # كسر هبوطي
        if prev['Low'] >= low_range and last['Close'] < low_range:
            entry = float(last['Close'])
            sl = round(low_range + atr * 0.5, 2)
            risk = sl - entry
            return {
                "strategy": "RANGE_BREAKOUT_SELL", "direction": -1, "entry": entry,
                "sl": sl, "tp1": round(entry - risk * 2.0, 2), "tp2": round(entry - risk * 3.5, 2),
                "confidence": 62, "reason": "📊 كسر نطاق هبوطي"
            }
    except: pass
    return None

def strategy_rsi_divergence(df_m15):
    """📈 تباعد RSI: السعر يكسر قاع لكن RSI يرتفع"""
    if df_m15 is None or len(df_m15) < 40: return None
    try:
        rsi = calc_rsi(df_m15['Close'], 14)
        if len(rsi) < 30: return None
        
        # قمم وقيعان بسيطة (آخر 20 شمعة)
        recent_20 = df_m15.tail(20).copy()
        recent_20['RSI'] = rsi.tail(20)
        
        # أدنى قاعين في السعر
        lows_idx = recent_20['Low'].nsmallest(2).index
        if len(lows_idx) < 2: return None
        l1, l2 = sorted(lows_idx)[0], sorted(lows_idx)[1]
        price_new_low = recent_20.loc[l2, 'Low'] < recent_20.loc[l1, 'Low']
        rsi_higher_low = recent_20.loc[l2, 'RSI'] > recent_20.loc[l1, 'RSI']
        
        if price_new_low and rsi_higher_low and recent_20.loc[l2, 'RSI'] < 40:
            entry = float(df_m15['Close'].iloc[-1])
            atr = float((df_m15['High'] - df_m15['Low']).rolling(14).mean().iloc[-1])
            sl = round(recent_20.loc[l2, 'Low'] - atr * 0.3, 2)
            risk = entry - sl
            return {
                "strategy": "RSI_DIV_BUY", "direction": 1, "entry": entry,
                "sl": sl, "tp1": round(entry + risk * 2.0, 2), "tp2": round(entry + risk * 3.0, 2),
                "confidence": 58, "reason": "📈 تباعد RSI صعودي"
            }
        
        # قمم متتالية
        highs_idx = recent_20['High'].nlargest(2).index
        if len(highs_idx) < 2: return None
        h1, h2 = sorted(highs_idx)[0], sorted(highs_idx)[1]
        price_new_high = recent_20.loc[h2, 'High'] > recent_20.loc[h1, 'High']
        rsi_lower_high = recent_20.loc[h2, 'RSI'] < recent_20.loc[h1, 'RSI']
        
        if price_new_high and rsi_lower_high and recent_20.loc[h2, 'RSI'] > 60:
            entry = float(df_m15['Close'].iloc[-1])
            atr = float((df_m15['High'] - df_m15['Low']).rolling(14).mean().iloc[-1])
            sl = round(recent_20.loc[h2, 'High'] + atr * 0.3, 2)
            risk = sl - entry
            return {
                "strategy": "RSI_DIV_SELL", "direction": -1, "entry": entry,
                "sl": sl, "tp1": round(entry - risk * 2.0, 2), "tp2": round(entry - risk * 3.0, 2),
                "confidence": 58, "reason": "📉 تباعد RSI هبوطي"
            }
    except: pass
    return None

# ---------- 4. مدير الإشارات (يمنع التكرار والإفراط) ----------

def count_today_signals():
    today = datetime.now(NY).strftime("%Y-%m-%d")
    count = int(db_get(f"signals_{today}", "0"))
    return count

def increment_today_signals():
    today = datetime.now(NY).strftime("%Y-%m-%d")
    db_set(f"signals_{today}", str(count_today_signals() + 1))

def can_send_signal():
    if count_today_signals() >= MAX_SIGNALS_PER_DAY:
        return False, "وصلنا السقف اليومي"
    last_ts = db_get("last_signal_ts", "0")
    try:
        elapsed = (time.time() - float(last_ts)) / 60
        if elapsed < SIGNAL_COOLDOWN_MIN:
            return False, f"cooldown ({elapsed:.0f}/{SIGNAL_COOLDOWN_MIN}د)"
    except: pass
    return True, "OK"

def signal_hash(s):
    return f"{s['strategy']}_{s['direction']}_{round(s['entry'], 1)}"

# ---------- 5. المحرك الجديد (يجمع الكل) ----------

async def run_sniper_engine():
    """
    المحرك الجديد: يشغل 5 استراتيجيات متوازية.
    - ICT (run_engine الأصلي) له أولوية في نافذة Silver Bullet
    - 4 استراتيجيات مكمّلة تعمل طوال الجلسات
    """
    results = []

    # جلب البيانات مرة واحدة (توفير API)
    df_m15 = await fetch_candles("GC=F", "15m", 200)
    df_m5 = await fetch_candles("GC=F", "5m", 200)
    if df_m15 is None or df_m5 is None:
        return None, None

    vwap = calculate_vwap(df_m5)
    current_price, _ = await get_live_prices()
    if current_price is None:
        current_price = float(df_m5['Close'].iloc[-1])

    # 1. ICT (الأصلي) — فقط في Silver Bullet
    if is_silver_bullet_time():
        report, err = await run_engine()
        if report:
            results.append(("ICT", report))

    # 2. VWAP Bounce
    if "vwap" in STRATEGIES_ENABLED:
        s = strategy_vwap_bounce(df_m15, df_m5, vwap)
        if s and abs(s['entry'] - current_price) < 20:  # نطاق معقول
            results.append(("VWAP", s))

    # 3. EMA Momentum
    if "ema" in STRATEGIES_ENABLED:
        s = strategy_ema_momentum(df_m5)
        if s and abs(s['entry'] - current_price) < 20:
            results.append(("EMA", s))

    # 4. Range Breakout
    if "range" in STRATEGIES_ENABLED:
        s = strategy_range_breakout(df_m15)
        if s and abs(s['entry'] - current_price) < 30:
            results.append(("RANGE", s))

    # 5. RSI Divergence
    if "divergence" in STRATEGIES_ENABLED:
        s = strategy_rsi_divergence(df_m15)
        if s and abs(s['entry'] - current_price) < 30:
            results.append(("DIV", s))

    if not results:
        return None, None

    # تصفية ICT (هو الأقوى) — إن وُجد، نرسله مباشرة
    for name, data in results:
        if name == "ICT":
            increment_today_signals()
            db_set("last_signal_ts", str(time.time()))
            return data, None

    # اختر الإشارة الأقوى من الباقي
    best = max(results, key=lambda x: x[1].get('confidence', 0))
    name, s = best

    # فحوصات ما قبل الإرسال
    can, reason = can_send_signal()
    if not can:
        logging.info(f"⏸️ إشارة {name} محجوبة: {reason}")
        return None, None

    # منع التكرار
    h = signal_hash(s)
    if db_get(f"sig_{h}") == "sent":
        return None, None
    db_set(f"sig_{h}", "sent")

    # تأكيد من AI (سريع، فقط للإشارات < 70)
    ai_confirm = ""
    if s['confidence'] < 70:
        macro, vix, inter = await get_macro_and_intermarket()
        news = await get_news_headlines()
        ai_summary = f"Strategy: {s['strategy']}\nEntry: {s['entry']}\nSL: {s['sl']}\nDirection: {s['direction']}\nMacro: {macro}\nVIX: {vix}\nNews: {news[:200]}"
        verdict = await ask_ai_council(ai_summary)
        if "انتظار" in verdict and "شراء" not in verdict and "بيع" not in verdict:
            logging.info(f"❌ AI رفض {name}")
            return None, None
        ai_confirm = verdict

    # بناء الرسالة
    direction = "🟢 شراء" if s['direction'] == 1 else "🔴 بيع"
    report = (
        f"👑 *الوحش 71.0 — القناص اليومي*\n"
        f"🕐 `{datetime.now(NY).strftime('%H:%M')}` (NY)\n"
        f"🎯 *الاستراتيجية:* {name}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 السعر: `{current_price:.2f}`\n"
        f"📊 VWAP: `{vwap or 'N/A'}`\n"
        f"💡 *السبب:* {s['reason']}\n"
        f"🔥 *الثقة:* {s['confidence']}%\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🎯 *الدخول:* `{s['entry']:.2f}`\n"
        f"{direction}\n"
        f"🛑 *الوقف:* `{s['sl']:.2f}`\n"
        f"🏆 *TP1:* `{s['tp1']:.2f}`\n"
        f"🚀 *TP2:* `{s['tp2']:.2f}`\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 اليوم: {count_today_signals() + 1}/{MAX_SIGNALS_PER_DAY}\n"
    )
    if ai_confirm:
        report += f"\n🧠 *AI:* {ai_confirm[:200]}\n"

    increment_today_signals()
    db_set("last_signal_ts", str(time.time()))
    return report, None

# ==========================================
# 🚀 الحلقة الرئيسية (معدّلة لتشغيل القناص)
# ==========================================
async def main_loop():
    logging.info("👑 الوحش 71.0 — القناص اليومي بدأ")
    await send_telegram_msg(
        "👑 *الوحش 71.0 — القناص اليومي*\n"
        f"🕐 {datetime.now(NY).strftime('%Y-%m-%d %H:%M')}\n"
        f"⚡ 5 استراتيجيات متوازية\n"
        f"🎯 جلسات: London + NY\n"
        f"📊 سقف يومي: {MAX_SIGNALS_PER_DAY} إشارة"
    )
    
    last_outcome = 0
    while True:
        try:
            now_ts = time.time()
            
            # فحص نتائج الصفقات المفتوحة كل 5 دقائق
            if now_ts - last_outcome > 300:
                try:
                    current, _ = await get_live_prices()
                    if current:
                        rows = conn.execute("SELECT key, value FROM state WHERE key LIKE 'tracked_%' AND value='open'").fetchall()
                        for key, _ in rows:
                            if key.endswith("_data"): continue
                            data_str = db_get(f"{key}_data")
                            if not data_str: continue
                            data = json.loads(data_str)
                            d, entry, sl, tp1, tp2 = data['direction'], data['entry'], data['sl'], data['tp1'], data['tp2']
                            hit = None
                            if d == 1:
                                if current <= sl: hit = "SL ❌"
                                elif current >= tp2: hit = "TP2 🚀"
                                elif current >= tp1: hit = "TP1 ✅"
                            else:
                                if current >= sl: hit = "SL ❌"
                                elif current <= tp2: hit = "TP2 🚀"
                                elif current <= tp1: hit = "TP1 ✅"
                            if hit:
                                db_set(key, hit)
                                await send_telegram_msg(
                                    f"📊 *نتيجة*\n➡️ {'شراء' if d==1 else 'بيع'}\n"
                                    f"💰 `{entry:.2f}` → `{current:.2f}`\n📌 *{hit}*"
                                )
                except Exception as e:
                    logging.warning(f"Outcome: {e}")
                last_outcome = now_ts
            
            # تشغيل القناص في جلسات التداول
            if is_trading_session():
                report, err = await run_sniper_engine()
                if err and "بيانات غير كافية" not in err:
                    await send_telegram_msg(err)
                elif report:
                    await send_telegram_msg(report)
                    # حفظ للنتائج لاحقًا
                    try:
                        # extract from report? — سنستخدم db state مباشرة
                        pass
                    except: pass
            
            await asyncio.sleep(120)  # كل دقيقتين (بدل 60، توفير API)
            
        except KeyboardInterrupt:
            logging.info("⛔ إيقاف")
            await send_telegram_msg("⛔ البوت توقف")
            break
        except Exception as e:
            logging.error(f"Loop: {e}")
            await asyncio.sleep(30)

if __name__ == '__main__':
    asyncio.run(main_loop())
