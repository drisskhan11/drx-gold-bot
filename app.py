import os, sqlite3, warnings, logging, asyncio, requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import yfinance as yf
import pandas as pd
import numpy as np
import telegram
from openai import OpenAI

try:
    from biquote import Biquote
    bq = Biquote()
    bq_available = True
except ImportError:
    bq_available = False
    logging.warning("Biquote not installed. Falling back to yfinance only.")

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

# ✅ متغير الاختبار (إلا كان 1، كيتجاهل الوقت باش نتأكدو أن البوت خدام)
FORCE_TEST = os.environ.get("FORCE_TEST", "0").strip() == "1"

DB_FILE = "quantum_beast_61.db"
NY = ZoneInfo("America/New_York")
SILVER_BULLET_WINDOWS = [(3, 4), (10, 11)]

try:
    from google import genai
    ai_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
except Exception:
    ai_client = None

# ==========================================
# 🗄️ قاعدة البيانات
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
# 📱 إرسال تيليجرام
# ==========================================
async def send_telegram_msg(text):
    try:
        bot = telegram.Bot(token=TOKEN)
        await bot.send_message(chat_id=CHAT_ID, text=text, parse_mode='Markdown')
    except Exception as e:
        logging.error(f"Telegram Error: {e}")

# ==========================================
# 🌐 جلب البيانات
# ==========================================
async def get_live_prices():
    try:
        d = requests.get("https://xaus.com/api/v1/spot?compact=1", timeout=8).json()
        if d.get("data_state", {}).get("status") == "fresh":
            return float(d["spot_usd_oz"]), float(d["silver_usd_oz"])
    except: pass
    if bq_available:
        try:
            tick = bq.tick("XAUUSD")
            if tick: return float(tick['mid']), None
        except: pass
    try:
        gold = yf.Ticker("GC=F").history(period="1d", interval="1m")
        if not gold.empty: return float(gold["Close"].iloc[-1]), None
    except: pass
    return None, None

async def fetch_candles(symbol, interval, limit=100):
    if bq_available:
        try:
            bq_symbol = "XAUUSD" if symbol == "GC=F" else symbol
            ohlc = bq.ohlc(bq_symbol, interval=interval, limit=limit)
            if ohlc and 'bars' in ohlc:
                df = pd.DataFrame(ohlc['bars'])
                df['openTime'] = pd.to_datetime(df['openTime'])
                df = df.set_index('openTime').sort_index()
                df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
                df.index = df.index.tz_localize("UTC").tz_convert(NY) if df.index.tz is None else df.index.tz_convert(NY)
                return df
        except Exception as e:
            logging.warning(f"Biquote failed: {e}")

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

# ==========================================
# ⏰ فلتر الوقت (مع خاصية الاختبار)
# ==========================================
def is_silver_bullet_time():
    if FORCE_TEST:
        return True
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
    except: return False, None

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
# 🧠 مؤشرات إضافية
# ==========================================
def calculate_hurst(prices, max_lag=20):
    try:
        s = np.log(np.asarray(prices, dtype=float))
        lags = np.arange(2, min(max_lag, len(s)//3))
        if len(lags) < 6: return 0.5
        tau = np.array([np.std(s[l:] - s[:-l]) for l in lags]); m = tau > 0
        return float(np.polyfit(np.log(lags[m]), np.log(tau[m]), 1)[0]) if m.sum() > 5 else 0.5
    except: return 0.5

# ==========================================
# 🧠 مجلس العباقرة
# ==========================================
async def ask_ai_council(data_summary):
    prompt = f"""
    أنت مجلس خبراء تداول (Multi-Agent System).
    المعطيات:
    {data_summary}
    1. 🧠 [المحلل الفني]: قدم حجة.
    2. 😈 [محامي الشيطان]: هاجم الصفقة!
    3. ⚖️ [مدير المخاطر]: القرار النهائي (شراء/بيع/انتظار).
    الرد:
    🧠 المحلل: [جملة]
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
    return "⚠️ AI غير متاح"

# ==========================================
# 🚀 المحرك الرئيسي
# ==========================================
async def run_engine():
    try:
        if not is_silver_bullet_time():
            logging.info("💤 ماشي وقت الضربة الشيطانية. البوت ساكت.")
            return None, None

        news_blocked, news_msg = check_news_blackout()
        if news_blocked:
            logging.info(f"⏸️ حظر أخبار: {news_msg}")
            return None, None

        df_m15 = await fetch_candles("GC=F", "15m", 50)
        df_m1 = await fetch_candles("GC=F", "1m", 50)
        df_d1 = await fetch_candles("GC=F", "1d", 5)

        if any(d is None or len(d) < 2 for d in [df_m15, df_m1, df_d1]):
            return None, "❌ بيانات غير كافية"

        pdh = df_d1['High'].iloc[-2]
        pdl = df_d1['Low'].iloc[-2]
        direction, entry_zone, sweep_msg = detect_judas_swing(df_m15, df_m1, pdh, pdl)
        
        if direction is None:
            if FORCE_TEST:
                # فـ وضع الاختبار، نصيفطو رسالة باش نتأكدو
                current_price, _ = await get_live_prices()
                test_msg = (
                    f"✅ *البوت خدام مزيان* (وضع الاختبار)\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"🕐 الوقت: `{datetime.now(NY).strftime('%H:%M')}` (نيويورك)\n"
                    f"💰 السعر الحالي: `{current_price:.2f}`\n" if current_price else f"✅ *البوت خدام مزيان* (وضع الاختبار)\n"
                )
                test_msg += (
                    f"\n📊 *التحليل:*\n"
                    f"- H4 و H1: تم التحليل ✅\n"
                    f"- M15 و M1: تم التحليل ✅\n"
                    f"- PDH: `{pdh:.2f}` | PDL: `{pdl:.2f}`\n"
                    f"\n🧐 *الوضع:* ما كاينش فخ دابا. البوت كينتظر.\n"
                    f"💡 *ملاحظة:* فـ الوقت العادي، البوت كيخدم غير من 3-4 و 10-11 صباحاً (نيويورك)."
                )
                return test_msg, None
            return None, None

        current_price, _ = await get_live_prices()
        if current_price is None: current_price = df_m1['Close'].iloc[-1]
        hurst = calculate_hurst(df_m15['Close'].values)

        atr = (df_m15['High'] - df_m15['Low']).rolling(14).mean().iloc[-1]
        if abs(current_price - entry_zone) > (atr * 2.0):
            return None, None

        ai_summary = f"Price: {current_price}\nSweep: {sweep_msg}\nEntry Zone: {entry_zone}\nHurst: {hurst}"
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
            if db_get("last_signal") == signal_hash:
                return None, None
            db_set("last_signal", signal_hash)

            report = (
                f"👑 *الوحش 61.0 (الضربة الشيطانية)*\n"
                f"🕐 الوقت: `{datetime.now(NY).strftime('%H:%M')}` (نيويورك)\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"💰 السعر الحالي: `{current_price:.2f}`\n"
                f"💀 *الضربة الشيطانية:* {sweep_msg}\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"🎯 *الدخول (Limit):* `{entry_zone:.2f}`\n"
                f"{trade_type}\n"
                f"🛑 *الوقف:* `{sl:.2f}`\n"
                f"🏆 *الهدف 1 (1:2):* `{tp1:.2f}`\n"
                f"🚀 *الهدف 2 (1:4):* `{tp2:.2f}`\n"
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
