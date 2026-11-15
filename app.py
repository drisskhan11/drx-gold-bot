import os, sqlite3, warnings, logging, asyncio, requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import yfinance as yf
import pandas as pd
import numpy as np
import telegram
from openai import OpenAI
from twelvedata import TDClient
from scipy.fft import fft

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==========================================
# ⚙️ الإعدادات
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
SILVER_BULLET_WINDOWS = [(3, 4), (10, 11)]

try:
    from google import genai
    ai_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
except Exception:
    ai_client = None

td = TDClient(apikey=TWELVEDATA_API_KEY) if TWELVEDATA_API_KEY else None

# ==========================================
# 🗄️ قاعدة البيانات
# ==========================================
conn = sqlite3.connect(DB_FILE, check_same_thread=False)
conn.executescript("PRAGMA journal_mode=WAL;")
conn.executescript("""
CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS signals(id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, direction INTEGER, entry REAL, sl REAL, tp1 REAL, tp2 REAL, status TEXT, mc_prob REAL, context_score INTEGER);
""")
conn.commit()

def db_get(key, default=None):
    r = conn.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
    return r[0] if r else default

def db_set(key, value):
    conn.execute("INSERT OR REPLACE INTO state VALUES(?,?)", (key, str(value)))
    conn.commit()

# ==========================================
# 📱 إرسال تيليجرام
# ==========================================
async def send_telegram_msg(text):
    try:
        bot = telegram.Bot(token=TOKEN)
        await bot.send_message(chat_id=CHAT_ID, text=text, parse_mode='Markdown')
    except Exception as e:
        logging.error(f"Telegram Error: {e}")

# ==========================================
# 🏛️ VWAP (السعر المتوسط المرجح بالحجم)
# ==========================================
def calculate_vwap(df):
    try:
        if df is None or len(df) < 10: return None
        today = datetime.now(NY).date()
        today_data = df[df.index.date == today]
        if len(today_data) < 5:
            today_data = df.tail(50)
        
        typical_price = (today_data['High'] + today_data['Low'] + today_data['Close']) / 3
        vwap = (typical_price * today_data['Volume']).cumsum() / today_data['Volume'].cumsum()
        return round(vwap.iloc[-1], 2)
    except:
        return None

# ==========================================
# 📊 Volume Profile (POC, VAH, VAL)
# ==========================================
def calculate_volume_profile(df, bins=50):
    try:
        if df is None or len(df) < 30: return None, None, None
        prices = df['Close'].tail(100).values
        volumes = df['Volume'].tail(100).values
        
        hist, bin_edges = np.histogram(prices, bins=bins, weights=volumes)
        poc_idx = np.argmax(hist)
        poc = (bin_edges[poc_idx] + bin_edges[poc_idx+1]) / 2
        
        total_vol = hist.sum()
        target = total_vol * 0.70
        current_vol = hist[poc_idx]
        vah_idx, val_idx = poc_idx, poc_idx
        
        while current_vol < target:
            if vah_idx + 1 < len(hist) and (val_idx == 0 or hist[vah_idx+1] >= hist[val_idx-1]):
                vah_idx += 1
                current_vol += hist[vah_idx]
            elif val_idx - 1 >= 0:
                val_idx -= 1
                current_vol += hist[val_idx]
            else:
                break
        
        vah = (bin_edges[vah_idx] + bin_edges[vah_idx+1]) / 2
        val = (bin_edges[val_idx] + bin_edges[val_idx+1]) / 2
        return round(float(poc), 2), round(float(vah), 2), round(float(val), 2)
    except:
        return None, None, None

# ==========================================
# 💧 Mitigation Blocks و Breaker Blocks
# ==========================================
def detect_mitigation_blocks(df, lookback=50):
    if df is None or len(df) < lookback: return [], []
    mit_blocks = []
    breaker_blocks = []
    
    try:
        highs = df['High'].values
        lows = df['Low'].values
        closes = df['Close'].values
        opens = df['Open'].values
        atr = (df['High'] - df['Low']).rolling(14).mean().iloc[-1]
        
        for i in range(-lookback, -3):
            body_size = abs(closes[i] - opens[i])
            if body_size > atr * 1.2:  # شمعة قوية
                if closes[i] > opens[i]:  # شمعة صاعدة
                    mit_blocks.append({
                        'type': 'bullish', 
                        'top': highs[i], 
                        'bottom': lows[i],
                        'index': i
                    })
                else:
                    mit_blocks.append({
                        'type': 'bearish',
                        'top': highs[i],
                        'bottom': lows[i],
                        'index': i
                    })
        
        # Breaker: كتلة انكسرت وتحولت
        current_price = closes[-1]
        for block in mit_blocks:
            if block['type'] == 'bullish' and current_price < block['bottom']:
                breaker_blocks.append({**block, 'new_type': 'bearish'})
            elif block['type'] == 'bearish' and current_price > block['top']:
                breaker_blocks.append({**block, 'new_type': 'bullish'})
        
        return mit_blocks[-3:], breaker_blocks[-3:]
    except:
        return [], []

def is_price_at_mitigation(current_price, mit_blocks, tolerance=5):
    for block in mit_blocks:
        if block['bottom'] <= current_price <= block['top']:
            return block['type']
        if abs(current_price - block['top']) < tolerance:
            return block['type']
        if abs(current_price - block['bottom']) < tolerance:
            return block['type']
    return None

# ==========================================
# 🌊 تدفق الأوامر (Delta Proxy)
# ==========================================
def calculate_delta_divergence(df):
    try:
        if df is None or len(df) < 20: return "NEUTRAL"
        
        recent = df.tail(10)
        
        # السعر طالع
        price_change = recent['Close'].iloc[-1] - recent['Close'].iloc[0]
        
        # الفوليوم (Delta Proxy): نقارن فوليوم الشموع الصاعدة مع الهابطة
        bullish_vol = recent[recent['Close'] > recent['Open']]['Volume'].sum()
        bearish_vol = recent[recent['Close'] < recent['Open']]['Volume'].sum()
        
        total_vol = bullish_vol + bearish_vol
        if total_vol == 0: return "NEUTRAL"
        
        delta = (bullish_vol - bearish_vol) / total_vol
        
        # تباعد: السعر طالع، ولكن الدلتا سلبية
        if price_change > 0 and delta < -0.2:
            return "BEARISH_DIVERGENCE"
        elif price_change < 0 and delta > 0.2:
            return "BULLISH_DIVERGENCE"
        
        return "NEUTRAL"
    except:
        return "NEUTRAL"

# ==========================================
# 🔬 محرك أينشتاين: الإنتروبيا
# ==========================================
def calculate_entropy(df, period=14):
    try:
        returns = df['Close'].pct_change().dropna().tail(100)
        if len(returns) < period: return 0.5
        
        hist, _ = np.histogram(returns, bins=period)
        probs = hist / hist.sum()
        probs = probs[probs > 0]
        
        entropy = -np.sum(probs * np.log2(probs))
        max_entropy = np.log2(period)
        return round(entropy / max_entropy if max_entropy > 0 else 0, 3)
    except:
        return 0.5

# ==========================================
# 🏹 محرك نيوتن: القوة
# ==========================================
def calculate_newtonian_force(df):
    try:
        if len(df) < 20: return 0
        close = df['Close']
        volume = df['Volume']
        velocity = close.diff()
        acceleration = velocity.diff()
        force = volume * acceleration
        current_force = force.tail(5).mean()
        force_normalized = current_force / (volume.tail(20).mean() * close.std() + 1e-9)
        return round(force_normalized, 3)
    except:
        return 0

# ==========================================
# 🎼 محرك موزار: الرنين
# ==========================================
def calculate_harmonic_resonance(df):
    try:
        if len(df) < 50: return 0
        prices = df['Close'].tail(100).values
        prices = prices - np.mean(prices)
        fft_vals = fft(prices)
        power = np.abs(fft_vals)
        dominant_power = np.max(power[1:])
        total_power = np.sum(power[1:])
        return round(dominant_power / total_power if total_power > 0 else 0, 3)
    except:
        return 0

# ==========================================
# 🎲 محاكاة مونت كارلو
# ==========================================
def monte_carlo_probability(current_price, sl, tp, atr, direction, simulations=500):
    try:
        sigma = atr / current_price
        mu = 0
        steps = 60
        dt = 1 / steps
        wins = 0
        
        for _ in range(simulations):
            price = current_price
            for step in range(steps):
                shock = np.random.normal(0, 1)
                price *= np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shock)
                
                if direction == 1 and price <= sl: break
                if direction == -1 and price >= sl: break
                if direction == 1 and price >= tp:
                    wins += 1
                    break
                if direction == -1 and price <= tp:
                    wins += 1
                    break
        return round(wins / simulations, 3)
    except:
        return 0.5

# ==========================================
# 💢 مؤشر الألم (Pain Index)
# ==========================================
async def calculate_pain_index(df_m15, current_price):
    try:
        if df_m15 is None or len(df_m15) < 20: return 50
        pain_score = 0
        round_levels = [round(current_price / 50) * 50 + (i * 50) for i in range(-2, 3)]
        for level in round_levels:
            if level != current_price and abs(current_price - level) < 5:
                pain_score += 15
        recent = df_m15.tail(30)
        local_highs = recent['High'].rolling(5).max()
        local_lows = recent['Low'].rolling(5).min()
        for h in local_highs.dropna().tail(5):
            if abs(current_price - h) < 5: pain_score += 10
        for l in local_lows.dropna().tail(5):
            if abs(current_price - l) < 5: pain_score += 10
        return min(100, pain_score)
    except:
        return 50

# ==========================================
# 🌐 جلب البيانات
# ==========================================
async def get_live_prices():
    try:
        d = requests.get("https://xaus.com/api/v1/spot?compact=1", timeout=8).json()
        if d.get("data_state", {}).get("status") == "fresh":
            return float(d["spot_usd_oz"]), float(d["silver_usd_oz"])
    except: pass
    try:
        gold = yf.Ticker("GC=F").history(period="1d", interval="1m")
        if not gold.empty: return float(gold["Close"].iloc[-1]), None
    except: pass
    return None, None

async def fetch_candles(symbol, interval, limit=200):
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
        return {
            "open": float(prev["open"]),
            "high": float(prev["high"]),
            "low": float(prev["low"]),
            "close": float(prev["close"])
        }
    except:
        return None

# ==========================================
# 🏭 GDX والأخبار
# ==========================================
async def get_gdx_trend():
    try:
        gdx = yf.Ticker("GDX").history(period="5d", interval="1h")
        if gdx.empty: return "NEUTRAL", 0
        ema = gdx['Close'].ewm(span=20).mean().iloc[-1]
        current = gdx['Close'].iloc[-1]
        strength = ((current - ema) / ema) * 100
        return ("BULLISH" if current > ema else "BEARISH"), round(strength, 2)
    except:
        return "NEUTRAL", 0

async def get_news_headlines():
    try:
        gold = yf.Ticker("GC=F")
        news = gold.news
        if not news: return "لا توجد أخبار حديثة."
        return "\n".join([n.get('title', '') for n in news[:5]])
    except:
        return "تعذر جلب الأخبار."

# ==========================================
# 🎯 وعي السياق
# ==========================================
async def get_context_score(direction):
    try:
        dxy = await fetch_candles("DX-Y.NYB", "1h", 50)
        vix = await fetch_candles("^VIX", "1h", 50)
        tnx = await fetch_candles("^TNX", "1h", 50)
        score = 30
        if vix is not None:
            vix_val = vix['Close'].iloc[-1]
            if vix_val > 20: score += 15
            elif vix_val < 15: score -= 5
        if dxy is not None and len(dxy) > 20:
            dxy_bull = dxy['Close'].iloc[-1] > dxy['Close'].ewm(span=20).mean().iloc[-1]
            if direction == 1 and not dxy_bull: score += 20
            elif direction == -1 and dxy_bull: score += 20
        if tnx is not None:
            tnx_bull = tnx['Close'].iloc[-1] > tnx['Close'].ewm(span=20).mean().iloc[-1]
            if direction == 1 and not tnx_bull: score += 15
            elif direction == -1 and tnx_bull: score += 15
        return max(0, min(100, score))
    except:
        return 30

async def get_volume_confirmation(df_m15):
    try:
        if len(df_m15) < 20: return False
        vol_avg = df_m15['Volume'].tail(20).mean()
        vol_current = df_m15['Volume'].iloc[-1]
        return vol_current > vol_avg * 1.2
    except:
        return False

# ==========================================
# ⏰ فلاتر الوقت والأخبار
# ==========================================
def is_silver_bullet_time():
    if FORCE_TEST: return True
    now = datetime.now(NY)
    if now.weekday() >= 5: return False
    t = now.hour + now.minute / 60
    return any(start <= t < end for start, end in SILVER_BULLET_WINDOWS)

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
    except:
        return False, None

# ==========================================
# 💀 كشف الفخ (Judas Swing) و CHoCH و FVG
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
# 🧠 مؤشر Hurst
# ==========================================
def calculate_hurst(prices, max_lag=20):
    try:
        s = np.log(np.asarray(prices, dtype=float))
        lags = np.arange(2, min(max_lag, len(s)//3))
        if len(lags) < 6: return 0.5
        tau = np.array([np.std(s[l:] - s[:-l]) for l in lags])
        m = tau > 0
        return float(np.polyfit(np.log(lags[m]), np.log(tau[m]), 1)[0]) if m.sum() > 5 else 0.5
    except:
        return 0.5

# ==========================================
# 🧠 مجلس العباقرة
# ==========================================
async def ask_ai_council(data_summary):
    prompt = f"""
    أنت مجلس خبراء تداول (Multi-Agent System).
    المعطيات:
    {data_summary}
    1. 🧠 [المحلل الفني]: حلل SMC و Hurst.
    2. 📰 [محلل الأخبار]: حلل العناوين.
    3. 🏭 [محلل المعادن]: حلل GDX.
    4. 🎲 [محلل مونت كارلو]: حلل الاحتمالية.
    5. 🧬 [محلل أينشتاين]: حلل الإنتروبيا.
    6. 🏹 [محلل نيوتن]: حلل القوة.
    7. 🎼 [محلل موزار]: حلل الرنين.
    8. 🌊 [محلل تدفق الأوامر]: حلل الدلتا.
    9. 😈 [محامي الشيطان]: هاجم الصفقة!
    10. ⚖️ [مدير المخاطر]: القرار النهائي.
    الرد:
    🧠 المحلل: [جملة]
    📰 الأخبار: [جملة]
    🏭 المعادن: [جملة]
    🎲 مونت كارلو: [جملة]
    🧬 أينشتاين: [جملة]
    🏹 نيوتن: [جملة]
    🎼 موزار: [جملة]
    🌊 تدفق الأوامر: [جملة]
    😈 محامي الشيطان: [جملة]
    ⚖️ مدير المخاطر: القرار: [شراء/بيع/انتظار] | السبب: [جملة]
    """
    if ai_client:
        try:
            response = await ai_client.aio.models.generate_content(model="gemini-3.5-flash-lite", contents=prompt)
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
    return None

# ==========================================
# 🚀 المحرك الرئيسي (الوحش 71.0)
# ==========================================
async def run_engine():
    try:
        if not is_silver_bullet_time():
            logging.info("💤 ماشي وقت الضربة الشيطانية.")
            return None, None

        news_blocked, news_msg = check_news_blackout()
        if news_blocked:
            logging.info(f"⏸️ حظر أخبار: {news_msg}")
            return None, None

        ohlc = await get_previous_day_ohlc()
        if not ohlc:
            return None, "❌ تعذر جلب بيانات اليوم السابق"

        pdh, pdl = ohlc["high"], ohlc["low"]

        df_m15 = await fetch_candles("GC=F", "15m", 200)
        df_m1 = await fetch_candles("GC=F", "1m", 100)
        if any(d is None for d in [df_m15, df_m1]):
            return None, "❌ بيانات غير كافية"

        current_price, _ = await get_live_prices()
        if current_price is None: current_price = df_m1['Close'].iloc[-1]

        # ✅ VWAP
        vwap = calculate_vwap(df_m15)
        
        # ✅ Volume Profile
        poc, vah, val = calculate_volume_profile(df_m15)
        
        # ✅ Mitigation / Breaker Blocks
        mit_blocks, breaker_blocks = detect_mitigation_blocks(df_m15)
        
        # ✅ Delta Divergence
        delta_signal = calculate_delta_divergence(df_m15)
        
        # ✅ Einstein, Newton, Mozart
        entropy = calculate_entropy(df_m15)
        force = calculate_newtonian_force(df_m15)
        resonance = calculate_harmonic_resonance(df_m15)

        # ✅ Judas Swing
        direction, entry_zone, sweep_msg = detect_judas_swing(df_m15, df_m1, pdh, pdl)
        
        # وضع الاختبار
        if direction is None:
            if FORCE_TEST:
                gdx_trend, gdx_strength = await get_gdx_trend()
                ctx_score = await get_context_score(0)
                pain_index = await calculate_pain_index(df_m15, current_price)
                
                test_msg = (
                    f"✅ *البوت الشامل خدام مزيان*\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"🕐 الوقت: `{datetime.now(NY).strftime('%H:%M')}`\n"
                    f"💰 السعر: `{current_price:.2f}`\n"
                    f"\n📊 *VWAP:* `{vwap if vwap else 'N/A'}`\n"
                    f"📊 *POC:* `{poc}` | VAH: `{vah}` | VAL: `{val}`\n"
                    f"🧬 *الإنتروبيا:* {entropy} (0=منظم)\n"
                    f"🏹 *القوة:* {force}\n"
                    f"🎼 *الرنين:* {resonance}\n"
                    f"🌊 *تدفق الأوامر:* {delta_signal}\n"
                    f"🏭 *GDX:* {gdx_trend} ({gdx_strength}%)\n"
                    f"🎯 *Context:* {ctx_score}/100\n"
                    f"💢 *Pain:* {pain_index}/100\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"🧐 *الوضع:* كنسناو فخ حقيقي."
                )
                return test_msg, None
            return None, None

        # ✅ فلتر GDX
        gdx_trend, gdx_strength = await get_gdx_trend()
        if direction == 1 and gdx_trend == "BEARISH":
            logging.info("GDX Divergence: Buy ignored.")
            return None, None
        elif direction == -1 and gdx_trend == "BULLISH":
            logging.info("GDX Divergence: Sell ignored.")
            return None, None
        if abs(gdx_strength) < 0.5:
            return None, None

        # ✅ Context Score
        ctx_score = await get_context_score(direction)
        if ctx_score < 40: return None, None

        # ✅ Volume
        vol_ok = await get_volume_confirmation(df_m15)
        if not vol_ok: return None, None

        # ✅ Einstein (Entropy)
        if entropy > 0.85: return None, None

        # ✅ Newton (Force)
        if direction == 1 and force < 0: return None, None
        if direction == -1 and force > 0: return None, None

        # ✅ Mozart (Resonance)
        if resonance < 0.3: return None, None

        # ✅ Delta Divergence
        if direction == 1 and delta_signal == "BEARISH_DIVERGENCE":
            logging.info("Delta Divergence against Buy. Ignored.")
            return None, None
        if direction == -1 and delta_signal == "BULLISH_DIVERGENCE":
            logging.info("Delta Divergence against Sell. Ignored.")
            return None, None

        # ✅ VWAP Filter
        if vwap:
            if direction == 1 and current_price < vwap:
                logging.info("Price below VWAP, no Buy.")
                return None, None
            if direction == -1 and current_price > vwap:
                logging.info("Price above VWAP, no Sell.")
                return None, None

        # ✅ Mitigation Block Filter
        mit_type = is_price_at_mitigation(current_price, mit_blocks)
        if mit_type and mit_type != ('bullish' if direction == 1 else 'bearish'):
            logging.info(f"Price at wrong Mitigation Block ({mit_type}). Ignored.")
            return None, None

        # ✅ Volume Profile Filter
        if vah and val and poc:
            if direction == 1 and current_price > vah:
                logging.info("Price above VAH, no Buy.")
                return None, None
            if direction == -1 and current_price < val:
                logging.info("Price below VAL, no Sell.")
                return None, None

        # حساب الوقف والهدف
        atr = (df_m15['High'] - df_m15['Low']).rolling(14).mean().iloc[-1]
        if abs(current_price - entry_zone) > (atr * 2.0):
            return None, None

        if direction == 1:
            sl = round(df_m1['Low'].tail(5).min() - (atr * 0.2), 2)
            risk = entry_zone - sl
            if risk < 5:
                risk = 5
                sl = round(entry_zone - 5, 2)
            tp1 = round(entry_zone + risk * 2.0, 2)
            tp2 = round(entry_zone + risk * 4.0, 2)
            trade_type = "🟢 شراء (Long)"
        else:
            sl = round(df_m1['High'].tail(5).max() + (atr * 0.2), 2)
            risk = sl - entry_zone
            if risk < 5:
                risk = 5
                sl = round(entry_zone + 5, 2)
            tp1 = round(entry_zone - risk * 2.0, 2)
            tp2 = round(entry_zone - risk * 4.0, 2)
            trade_type = "🔴 بيع (Short)"

        if risk <= 0: return None, "❌ خطأ في حساب الوقف"

        # ✅ Monte Carlo
        mc_prob = monte_carlo_probability(entry_zone, sl, tp1, atr, direction, simulations=500)
        if mc_prob < 0.60: return None, None

        # ✅ Pain Index
        pain_index = await calculate_pain_index(df_m15, current_price)
        if pain_index < 50: return None, None

        macro, vix, intermarket = "", "", ""
        try:
            dxy = await fetch_candles("DX-Y.NYB", "1h", 50)
            if dxy is not None and len(dxy) > 20:
                macro = "BEARISH_GOLD" if dxy['Close'].iloc[-1] > dxy['Close'].ewm(span=20).mean().iloc[-1] else "BULLISH_GOLD"
        except: pass
        
        news_headlines = await get_news_headlines()

        ai_summary = f"""
        Price: {current_price}
        Sweep: {sweep_msg}
        Entry: {entry_zone}
        Hurst: {calculate_hurst(df_m15['Close'].values)}
        VWAP: {vwap}
        POC: {poc}, VAH: {vah}, VAL: {val}
        Delta: {delta_signal}
        GDX: {gdx_trend} ({gdx_strength}%)
        Context: {ctx_score}/100
        MC Prob: {mc_prob*100:.1f}%
        Pain: {pain_index}/100
        Entropy: {entropy}
        Force: {force}
        Resonance: {resonance}
        News: {news_headlines}
        """
        ai_verdict = await ask_ai_council(ai_summary)
        if ai_verdict is None:
            logging.warning("AI Unavailable: Signal blocked.")
            return None, None

        if "انتظار" not in ai_verdict:
            signal_hash = f"{direction}_{entry_zone}_{sl}"
            if db_get("last_signal") == signal_hash:
                return None, None
            db_set("last_signal", signal_hash)

            report = (
                f"👑 *الوحش 71.0 (المنظومة الشاملة)*\n"
                f"🕐 الوقت: `{datetime.now(NY).strftime('%H:%M')}`\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"💰 السعر: `{current_price:.2f}`\n"
                f"💀 *الضربة الشيطانية:* {sweep_msg}\n"
                f"📊 *VWAP:* `{vwap}`\n"
                f"📊 *POC:* `{poc}` | VAH: `{vah}` | VAL: `{val}`\n"
                f"🌊 *Delta:* {delta_signal}\n"
                f"🧬 *Entropy:* {entropy}\n"
                f"🏹 *Force:* {force}\n"
                f"🎼 *Resonance:* {resonance}\n"
                f"🏭 *GDX:* {gdx_trend} ({gdx_strength}%)\n"
                f"🎯 *Context:* {ctx_score}/100\n"
                f"🎲 *MC:* {mc_prob*100:.1f}%\n"
                f"💢 *Pain:* {pain_index}/100\n"
                f"📰 *News:* {news_headlines[:80]}...\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"🎯 *الدخول:* `{entry_zone:.2f}`\n"
                f"{trade_type}\n"
                f"🛑 *الوقف:* `{sl:.2f}`\n"
                f"🏆 *الهدف 1:* `{tp1:.2f}`\n"
                f"🚀 *الهدف 2:* `{tp2:.2f}`\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"🧠 *مجلس العباقرة:*\n{ai_verdict}\n"
            )
            return report, None

        return None, None

    except Exception as e:
        return None, f"❌ خطأ: {e}"

async def main():
    signal, err = await run_engine()
    if err and "بيانات غير كافية" not in err:
        await send_telegram_msg(err)
    elif signal:
        await send_telegram_msg(signal)

if __name__ == '__main__':
    asyncio.run(main())
