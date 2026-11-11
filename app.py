import os, sqlite3, warnings, logging, asyncio, requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import yfinance as yf
import pandas as pd
import numpy as np
import telegram
from openai import OpenAI
from twelvedata import TDClient

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

DB_FILE = "quantum_beast_66.db"
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
# 📈 العائد الحقيقي (Real Yield - TIPS)
# ==========================================
async def get_real_yield_bias():
    try:
        # Autario API (مجاني، بدون مفتاح) - بيانات FRED DFII10
        url = "https://autario.com/api/v1/public/datasets/1e5aab40-5930-4861-b286-17ce1cbb56f5/data?limit=2"
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            data = r.json()
            if data and len(data) >= 2:
                # نفترض أن البيانات مرتبة من الأحدث للأقدم
                current = data[0]
                previous = data[1]
                current_yield = float(current.get('value', 0))
                previous_yield = float(previous.get('value', 0))
                
                if current_yield > previous_yield:
                    bias = "RISING"  # صاعد (سلبي للذهب)
                elif current_yield < previous_yield:
                    bias = "FALLING"  # هابط (إيجابي للذهب)
                else:
                    bias = "STABLE"
                
                return bias, current_yield
    except Exception as e:
        logging.warning(f"Real Yield fetch failed: {e}")
    
    return "NEUTRAL", 0.0

# ==========================================
# 🐋 COT Report (متتبع الحيتان)
# ==========================================
async def get_cot_bias():
    last_fetch_str = db_get("cot_last_fetch")
    today = datetime.now(NY).date()
    if last_fetch_str:
        try:
            last_fetch = datetime.fromisoformat(last_fetch_str).date()
            if (today - last_fetch).days < 7:
                cached = db_get("cot_bias")
                if cached:
                    return cached, db_get("cot_net", "0")
        except: pass
    
    try:
        url = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
        params = {
            "$limit": 1,
            "$order": "report_date_as_yyyy_mm_dd DESC",
            "$where": "market_and_exchange_names like '%GOLD%'"
        }
        r = requests.get(url, params=params, timeout=15)
        if r.status_code == 200:
            data = r.json()
            if data:
                row = data[0]
                comm_long = float(row.get('comm_positions_long_all', 0))
                comm_short = float(row.get('comm_positions_short_all', 0))
                net = comm_long - comm_short
                
                if net > 50000:
                    bias = "BULLISH"
                elif net < -150000:
                    bias = "BEARISH"
                else:
                    bias = "NEUTRAL"
                
                db_set("cot_bias", bias)
                db_set("cot_net", str(int(net)))
                db_set("cot_last_fetch", today.isoformat())
                return bias, str(int(net))
    except Exception as e:
        logging.warning(f"COT fetch failed: {e}")
    
    return "NEUTRAL", "غير متاح"

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
        return {
            "open": float(prev["open"]),
            "high": float(prev["high"]),
            "low": float(prev["low"]),
            "close": float(prev["close"])
        }
    except Exception as e:
        logging.warning(f"Twelve Data failed: {e}")
        return None

# ==========================================
# 🏭 GDX والأخبار
# ==========================================
async def get_gdx_trend():
    try:
        gdx = yf.Ticker("GDX").history(period="5d", interval="1h")
        if gdx.empty: return "NEUTRAL"
        ema = gdx['Close'].ewm(span=20).mean().iloc[-1]
        current = gdx['Close'].iloc[-1]
        return "BULLISH" if current > ema else "BEARISH"
    except:
        return "NEUTRAL"

async def get_news_headlines():
    try:
        gold = yf.Ticker("GC=F")
        news = gold.news
        if not news: return "لا توجد أخبار حديثة."
        return "\n".join([n.get('title', '') for n in news[:5]])
    except:
        return "تعذر جلب الأخبار."

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
# 🧠 مؤشرات إضافية والماكرو
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

        vix_msg = "غير متاح"
        if vix is not None:
            vix_val = vix['Close'].iloc[-1]
            vix_msg = f"{vix_val:.1f}"

        intermarket = []
        if silver is not None:
            intermarket.append(f"🥈 فضة: {silver['Close'].iloc[-1]:.2f}")
        if spx is not None:
            intermarket.append(f"📉 S&P: {spx['Close'].iloc[-1]:.2f}")

        return macro, vix_msg, " | ".join(intermarket)
    except:
        return "NEUTRAL", "غير متاح", "بيانات غير متاحة"

# ==========================================
# 🧠 مجلس العباقرة
# ==========================================
async def ask_ai_council(data_summary):
    prompt = f"""
    أنت مجلس خبراء تداول (Multi-Agent System).
    المعطيات:
    {data_summary}
    1. 🧠 [المحلل الفني]: قدم حجة بناءً على SMC و Hurst.
    2. 📰 [محلل الأخبار]: حلل العناوين، هل تدعم الذهب أم لا؟
    3. 📈 [محلل العائد الحقيقي]: حلل اتجاه TIPS. العائد الصاعد كيخنق الذهب، والهابط كيدعمو.
    4. 🐋 [محلل الحيتان]: حلل COT. واش البنوك كيشريو ولا كيبيعو؟
    5. 😈 [محامي الشيطان]: هاجم الصفقة! ابحث عن أي سبب للفشل.
    6. ⚖️ [مدير المخاطر]: القرار النهائي (شراء/بيع/انتظار).
    الرد:
    🧠 المحلل: [جملة]
    📰 الأخبار: [جملة]
    📈 العائد الحقيقي: [جملة]
    🐋 الحيتان: [جملة]
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
# 🚀 المحرك الرئيسي (مع Real Yield)
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

        df_m15 = await fetch_candles("GC=F", "15m", 100)
        df_m1 = await fetch_candles("GC=F", "1m", 100)
        if any(d is None for d in [df_m15, df_m1]):
            return None, "❌ بيانات غير كافية"

        direction, entry_zone, sweep_msg = detect_judas_swing(df_m15, df_m1, pdh, pdl)
        
        # ✅ جلب بيانات COT و Real Yield
        cot_bias, cot_net = await get_cot_bias()
        ry_bias, ry_value = await get_real_yield_bias()
        
        if direction is None:
            if FORCE_TEST:
                current_price, _ = await get_live_prices()
                gdx = await get_gdx_trend()
                test_msg = (
                    f"✅ *البوت خدام مزيان* (وضع الاختبار)\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"🕐 الوقت: `{datetime.now(NY).strftime('%H:%M')}` (نيويورك)\n"
                )
                if current_price:
                    test_msg += f"💰 السعر الحالي: `{current_price:.2f}`\n"
                test_msg += (
                    f"\n📊 *التحليل:*\n"
                    f"- PDH: `{pdh:.2f}` | PDL: `{pdl:.2f}`\n"
                    f"- GDX (التعدين): {gdx}\n"
                    f"- COT (الحيتان): {cot_bias} (Net: {cot_net})\n"
                    f"- Real Yield (TIPS): {ry_bias} ({ry_value:.2f}%)\n"
                    f"\n🧐 *الوضع:* ما كاينش فخ دابا. البوت كينتظر."
                )
                return test_msg, None
            return None, None

        # ✅ فلتر GDX
        gdx_trend = await get_gdx_trend()
        if direction == 1 and gdx_trend == "BEARISH":
            logging.info("GDX Divergence: Buy signal ignored.")
            return None, None
        elif direction == -1 and gdx_trend == "BULLISH":
            logging.info("GDX Divergence: Sell signal ignored.")
            return None, None

        # ✅ فلتر COT
        if direction == 1 and cot_bias == "BEARISH":
            logging.info(f"COT Divergence: Buy signal ignored (COT: {cot_bias}).")
            return None, None
        elif direction == -1 and cot_bias == "BULLISH":
            logging.info(f"COT Divergence: Sell signal ignored (COT: {cot_bias}).")
            return None, None

        # ✅ فلتر Real Yield الصارم
        if direction == 1 and ry_bias == "RISING":
            logging.info(f"Real Yield Divergence: Buy signal ignored (RY: {ry_bias}, {ry_value:.2f}%).")
            return None, None
        elif direction == -1 and ry_bias == "FALLING":
            logging.info(f"Real Yield Divergence: Sell signal ignored (RY: {ry_bias}, {ry_value:.2f}%).")
            return None, None

        current_price, _ = await get_live_prices()
        if current_price is None: current_price = df_m1['Close'].iloc[-1]
        hurst = calculate_hurst(df_m15['Close'].values)
        macro, vix, intermarket = await get_macro_and_intermarket()
        news_headlines = await get_news_headlines()

        atr = (df_m15['High'] - df_m15['Low']).rolling(14).mean().iloc[-1]
        if abs(current_price - entry_zone) > (atr * 2.0):
            return None, None

        ai_summary = f"""
        Price: {current_price}
        Sweep: {sweep_msg}
        Entry: {entry_zone}
        Hurst: {hurst}
        Macro: {macro}
        VIX: {vix}
        GDX Trend: {gdx_trend}
        COT Bias: {cot_bias} (Net: {cot_net})
        Real Yield: {ry_bias} ({ry_value:.2f}%)
        Latest News: {news_headlines}
        """
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
                f"👑 *الوحش 66.0 (القناص + GDX + AI + COT + Real Yield)*\n"
                f"🕐 الوقت: `{datetime.now(NY).strftime('%H:%M')}` (نيويورك)\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"💰 السعر: `{current_price:.2f}`\n"
                f"💀 *الضربة الشيطانية:* {sweep_msg}\n"
                f"🏭 *GDX (التعدين):* {gdx_trend}\n"
                f"🐋 *COT (الحيتان):* {cot_bias} (Net: {cot_net})\n"
                f"📈 *Real Yield (TIPS):* {ry_bias} ({ry_value:.2f}%)\n"
                f"📰 *الأخبار:* {news_headlines[:100]}...\n"
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
