import os
import sqlite3
import warnings
import logging
import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo

import yfinance as yf
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
import telegram
from openai import OpenAI

# ✅ المكتبات الجديدة
from tickerlayer import TickerLayer
from flashalpha import FlashAlpha

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==========================================
# ⚙️ الإعدادات
# ==========================================
TOKEN = os.environ.get("BOT_TOKEN", "").strip()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

# ✅ مفاتيح جديدة (زيدهم فـ GitHub Secrets)
TICKERLAYER_API_KEY = os.environ.get("TICKERLAYER_API_KEY", "").strip()
FLASHALPHA_API_KEY = os.environ.get("FLASHALPHA_API_KEY", "").strip()

DB_FILE = "quantum_beast_31.db"
KILLZONE_LONDON = (2.0, 5.0)
KILLZONE_NY = (7.0, 11.0)

try:
    from google import genai
    ai_client = genai.Client(api_key=GEMINI_API_KEY)
    AI_MODEL = "gemini-3.5-flash-lite"
    ai_enabled = True
except Exception as e:
    logging.error(f"AI Error: {e}")
    ai_enabled = False

# ✅ تهيئة TickerLayer و FlashAlpha
tl_client = TickerLayer(TICKERLAYER_API_KEY) if TICKERLAYER_API_KEY else None
fa_client = FlashAlpha(FLASHALPHA_API_KEY) if FLASHALPHA_API_KEY else None

# ==========================================
# 📱 دالة إرسال رسالة لتيليجرام
# ==========================================
async def send_telegram_msg(text):
    try:
        bot = telegram.Bot(token=TOKEN)
        await bot.send_message(chat_id=CHAT_ID, text=text, parse_mode='Markdown')
    except Exception as e:
        logging.error(f"Telegram Send Error: {e}")

# ==========================================
# 🗄️ قاعدة البيانات
# ==========================================
def init_db():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.executescript("CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY, value TEXT);")
    conn.commit()
    return conn

conn = init_db()

def db_get(key, default=None):
    r = conn.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
    return r[0] if r else default

def db_set(key, value):
    conn.execute("INSERT OR REPLACE INTO state VALUES(?,?)", (key, str(value)))
    conn.commit()

def is_in_killzone():
    now = datetime.now(ZoneInfo("America/New_York"))
    if now.weekday() >= 5: return False
    t = now.hour + now.minute / 60
    return (KILLZONE_LONDON[0] <= t < KILLZONE_LONDON[1]) or (KILLZONE_NY[0] <= t < KILLZONE_NY[1])

# ==========================================
# ⚡ السعر الفوري (Real-time) عبر TickerLayer
# ==========================================
async def get_live_price():
    if not tl_client:
        return None
    try:
        # TickerLayer كيدعم XAUUSD مباشرة
        quote = tl_client.quote("commodities", "XAUUSD")
        return quote.get('last_price') or quote.get('bid')
    except Exception as e:
        logging.warning(f"TickerLayer failed: {e}")
        return None

# ==========================================
# 🔮 تحليل الجاما (Gamma) عبر FlashAlpha
# ==========================================
async def get_gamma_analysis():
    if not fa_client:
        return None, None, None, None
    try:
        # GLD هو صندوق الذهب المتداول، وبيانات الخيارات ديالو متوفرة
        gex = fa_client.gex("GLD")
        
        net_gex = gex.get('net_gex', 0)
        gamma_flip = gex.get('gamma_flip')
        flip_status = gex.get('gamma_flip_status')
        
        levels = fa_client.exposure_levels("GLD")
        call_wall = levels['levels'].get('call_wall')
        put_wall = levels['levels'].get('put_wall')
        
        return net_gex, gamma_flip, call_wall, put_wall
    except Exception as e:
        logging.warning(f"FlashAlpha failed: {e}")
        return None, None, None, None

# ==========================================
# 🌐 جلب البيانات (مع Fallback)
# ==========================================
async def fetch_data(symbol, period, interval):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if not df.empty: return df
    except Exception as e:
        logging.warning(f"Primary symbol {symbol} failed: {e}")

    try:
        fallback_symbol = "XAUUSD=X" if symbol == "GC=F" else symbol
        ticker = yf.Ticker(fallback_symbol)
        df = ticker.history(period=period, interval=interval)
        if not df.empty: return df
    except Exception as e:
        logging.warning(f"Fallback symbol {fallback_symbol} failed: {e}")

    if symbol == "GC=F":
        try:
            ticker = yf.Ticker("GLD")
            df = ticker.history(period=period, interval=interval)
            if not df.empty: return df
        except Exception as e:
            logging.warning(f"GLD failed: {e}")

    return None

# ==========================================
# 🧠 العلوم المخفية
# ==========================================
def calculate_hurst_exponent(price_series, max_lag=20):
    try:
        if len(price_series) < max_lag: return 0.5
        lags = range(2, max_lag)
        tau = [np.sqrt(np.std(np.subtract(price_series[lag:], price_series[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
        return round(poly[0] * 2.0, 2)
    except Exception:
        return 0.5

def calculate_adr(df_d1):
    try:
        if df_d1 is None or len(df_d1) < 14: return 0
        df_d1['Range'] = df_d1['High'] - df_d1['Low']
        return round(df_d1['Range'].rolling(14).mean().iloc[-1], 2)
    except Exception:
        return 0

def get_golden_pocket(df):
    try:
        if df is None or len(df) < 20: return None, None
        recent_high = df['High'].tail(20).max()
        recent_low = df['Low'].tail(20).min()
        diff = recent_high - recent_low
        fib_618 = recent_high - (diff * 0.618)
        fib_65 = recent_high - (diff * 0.65)
        return round(min(fib_618, fib_65), 2), round(max(fib_618, fib_65), 2)
    except Exception:
        return None, None

# ==========================================
# 🧠 SMC و المؤشرات
# ==========================================
def analyze_smc(df, label):
    if df is None or len(df) < 50: return 0, None, f"{label}: بيانات غير كافية"
    try:
        current_price = df['Close'].iloc[-1]
        atr = (df['High'] - df['Low']).rolling(14).mean().iloc[-1]
        ob_bull = df['Low'].shift(1).rolling(10).min().iloc[-1]
        ob_bear = df['High'].shift(1).rolling(10).max().iloc[-1]
        recent_high = df['High'].rolling(5).max().iloc[-1]
        recent_low = df['Low'].rolling(5).min().iloc[-1]
        sweep_bull = df['Low'].iloc[-1] < recent_low and df['Close'].iloc[-1] > recent_low
        sweep_bear = df['High'].iloc[-1] > recent_high and df['Close'].iloc[-1] < recent_high
        if sweep_bull: return 1, {'bottom': ob_bull}, f"{label}: 🟢 Sweep + Bullish OB"
        if sweep_bear: return -1, {'top': ob_bear}, f"{label}: 🔴 Sweep + Bearish OB"
        if abs(current_price - ob_bull) < atr * 2:
            return 1, {'bottom': ob_bull}, f"{label}: 🟢 Bullish OB"
        elif abs(current_price - ob_bear) < atr * 2:
            return -1, {'top': ob_bear}, f"{label}: 🔴 Bearish OB"
        return 0, None, f"{label}: محايد"
    except Exception:
        return 0, None, f"{label}: خطأ"

def pca_analysis(df):
    try:
        if df is None or len(df) < 50: return None
        features = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        pca = PCA(n_components=1)
        pca.fit(features)
        return round(pca.explained_variance_ratio_[0] * 100, 2)
    except Exception:
        return None

def market_maker_simulation(df):
    try:
        if df is None or len(df) < 20: return None, None
        high = df['High'].tail(20).max()
        low = df['Low'].tail(20).min()
        return round(high + (high - low) * 0.2, 2), round(low - (high - low) * 0.2, 2)
    except Exception:
        return None, None

def calculate_correlation(gold_df, dxy_df):
    try:
        if gold_df is None or dxy_df is None or len(gold_df) < 50: return 0.0
        return round(gold_df['Close'].tail(50).reset_index(drop=True).corr(dxy_df['Close'].tail(50).reset_index(drop=True)), 2)
    except Exception:
        return 0.0

def ai_predict_price(df):
    try:
        if df is None or len(df) < 100: return None
        data = df[['Open', 'High', 'Low', 'Close']].copy()
        data['Target'] = data['Close'].shift(-1)
        data = data.dropna()
        if len(data) < 50: return None
        X = data[['Open', 'High', 'Low', 'Close']].values
        y = data['Target'].values
        split = int(len(X) * 0.8)
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X[:split], y[:split])
        return round(model.predict(X[-1].reshape(1, -1))[0], 2)
    except Exception:
        return None

def calculate_heatmap(df):
    if df is None or len(df) < 30: return None, []
    try:
        hist, bin_edges = np.histogram(df['Close'].values, bins=50, weights=df['Volume'].values)
        poc_idx = np.argmax(hist)
        poc = (bin_edges[poc_idx] + bin_edges[poc_idx+1]) / 2
        threshold = np.percentile(hist, 90)
        hot_zones = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(hist)) if hist[i] >= threshold]
        return round(float(poc), 2), [round(float(z), 2) for z in hot_zones]
    except Exception:
        return None, []

async def get_macro_bias():
    try:
        dxy = await fetch_data("DX-Y.NYB", "5d", "1h")
        us10y = await fetch_data("^TNX", "5d", "1h")
        if dxy is None or us10y is None: return "NEUTRAL", "بيانات غير مكتملة"
        dxy_bull = dxy['Close'].iloc[-1] > dxy['Close'].ewm(span=20).mean().iloc[-1]
        us10y_bull = us10y['Close'].iloc[-1] > us10y['Close'].ewm(span=20).mean().iloc[-1]
        if dxy_bull and us10y_bull: return "BEARISH_GOLD", "💵 الدولار صاعد"
        elif not dxy_bull and not us10y_bull: return "BULLISH_GOLD", "✨ الدولار هابط"
        else: return "NEUTRAL", "⚖️ تضارب"
    except Exception:
        return "NEUTRAL", "تعذر"

def is_news_dangerous():
    try:
        r = requests.get("https://nfs.faireconomy.media/ff_calendar_thisweek.json", timeout=10)
        news_data = r.json()
        now_ny = datetime.now(ZoneInfo("America/New_York"))
        critical = ["Federal Funds Rate", "Non-Farm Employment", "CPI", "FOMC", "PPI"]
        for news in news_data:
            if news.get('country') not in ['USD', 'XAU'] or news.get('impact') != 'High': continue
            date_str = news.get('date', '')
            if not date_str: continue
            news_time = pd.to_datetime(date_str).tz_localize(None).replace(tzinfo=ZoneInfo("America/New_York"))
            title = news.get('title', '')
            if news_time.date() == now_ny.date() and any(k in title for k in critical):
                return True, f"حدث مفصلي: {title}"
        return False, "آمن"
    except Exception:
        return False, "تعذر جلب الأخبار"

# ==========================================
# 🧠 إدارة حجم المركز (Position Sizing)
# ==========================================
def calculate_position_size(account_balance, entry, sl, kelly_fraction=0.5):
    """
    حساب حجم المركز باستخدام Kelly Criterion المصغر + ATR.
    account_balance: رأس المال (مثلاً 1000$)
    entry: سعر الدخول
    sl: سعر وقف الخسارة
    kelly_fraction: نسبة Kelly (0.5 يعني نصف كيلي، محافظ)
    """
    risk_per_unit = abs(entry - sl)
    if risk_per_unit == 0: return 0.01
    
    # نفترضو نسبة نجاح 45% ومعدل ربح 2.5 (1:2.5)
    win_rate = 0.45
    reward_ratio = 2.5
    kelly = (win_rate * reward_ratio - (1 - win_rate)) / reward_ratio
    kelly = max(0.01, kelly * kelly_fraction) # نخدمو بنصف كيلي باش نكونو محافظين
    
    risk_amount = account_balance * kelly
    lot_size = risk_amount / (risk_per_unit * 100) # تقريباً لكل 100 وحدة
    return max(0.01, round(lot_size, 2))

# ==========================================
# 🧠 مجلس العباقرة (Multi-Provider AI)
# ==========================================
async def ask_ai_council(data_summary):
    prompt = f"""
    أنت مجلس خبراء تداول (Multi-Agent System) مكون من 3 أعضاء.
    المعطيات الحالية للسوق:
    {data_summary}

    المطلوب منك أن تلعب الأدوار الثلاثة بالدارجة المغربية:
    1. 🧠 [المحلل الفني]: قدم حجة بناءً على المعطيات (SMC, PCA, Hurst, Fib, Gamma).
    2. 😈 [محامي الشيطان]: هاجم الصفقة! ابحث عن أي سبب للفشل.
    3. ⚖️ [مدير المخاطر]: أعطي القرار النهائي (شراء / بيع / انتظار).
    الرد يجب أن يكون بهذا الشكل:
    🧠 المحلل: [جملة واحدة]
    😈 محامي الشيطان: [جملة واحدة]
    ⚖️ مدير المخاطر: القرار: [شراء/بيع/انتظار] | السبب: [جملة واحدة]
    """
    if ai_enabled:
        try:
            response = await ai_client.aio.models.generate_content(model=AI_MODEL, contents=prompt)
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
    return "⚠️ AI غير متاح دابا."

# ==========================================
# 🚀 المحرك الرئيسي (مع Gamma و Real-time)
# ==========================================
async def run_engine():
    try:
        macro_bias, macro_msg = await get_macro_bias()
        news_danger, news_msg = is_news_dangerous()
        
        df_d1 = await fetch_data("GC=F", "1y", "1d"); await asyncio.sleep(1)
        df_h4 = await fetch_data("GC=F", "3mo", "4h"); await asyncio.sleep(1)
        df_h1 = await fetch_data("GC=F", "1mo", "1h"); await asyncio.sleep(1)
        df_m15 = await fetch_data("GC=F", "5d", "15m")

        if any(d is None for d in [df_d1, df_h4, df_h1, df_m15]):
            return None, "❌ تعذر جلب بيانات."

        # ✅ السعر الفوري (Real-time)
        live_price = await get_live_price()
        current_price = live_price if live_price else df_m15['Close'].iloc[-1]

        h4_dir, h4_ob, h4_msg = analyze_smc(df_h4, "H4")
        h1_dir, h1_ob, h1_msg = analyze_smc(df_h1, "H1")
        m15_dir, m15_ob, m15_msg = analyze_smc(df_m15, "M15")
        
        hurst = calculate_hurst_exponent(df_h1['Close'].values)
        adr = calculate_adr(df_d1)
        poc, hot_zones = calculate_heatmap(df_h1)
        hidden_above, hidden_below = market_maker_simulation(df_h1)

        # ✅ تحليل الجاما
        net_gex, gamma_flip, call_wall, put_wall = await get_gamma_analysis()

        if news_danger:
            report = f"🚨 *تحذير أخبار:* {news_msg}\n⚠️ كنسناو الخبر يفوت باش ندخلو."
            return {"type": "NEWS", "msg": report}, None

        smc_status = f"H4: {h4_msg}\nH1: {h1_msg}\nM15: {m15_msg}"
        
        # ✅ بوابة إعادة التسعير (Repricing Gate)
        # ما ندخلوش إلا إلا كان الاتجاه متوافق مع المتوسطات
        repricing_gate = True
        if h4_dir == 1 and df_h1['Close'].iloc[-1] < df_h1['Close'].ewm(span=50).mean().iloc[-1]:
            repricing_gate = False
        if h4_dir == -1 and df_h1['Close'].iloc[-1] > df_h1['Close'].ewm(span=50).mean().iloc[-1]:
            repricing_gate = False

        signal_dir = 0
        if h4_dir == 1 and h1_dir == 1 and m15_dir == 1 and repricing_gate:
            signal_dir = 1
        elif h4_dir == -1 and h1_dir == -1 and m15_dir == -1 and repricing_gate:
            signal_dir = -1

        if signal_dir == 0:
            report = (
                f"👀 *تقرير استخبارات السوق*\n"
                f"🕐 الوقت: `{datetime.now(ZoneInfo('Africa/Casablanca')).strftime('%H:%M')}`\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"💰 السعر: `{current_price:.2f}`\n"
                f"💵 الدولار: {macro_msg}\n"
                f"📰 الأخبار: {news_msg}\n"
                f"⚖️ السوق (Hurst): {hurst}\n"
                f"📏 متوسط الحركة (ADR): {adr}\n"
                f"🔥 سيولة: فوق {hidden_above} | تحت {hidden_below}\n"
                f"🔮 الجاما: GEX={net_gex if net_gex else 'غير متاح'}\n"
                f"📊 SMC: {smc_status}\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"🧐 *الوضع:* لا يوجد توافق فالفريمات دابا."
            )
            return {"type": "REPORT", "msg": report}, None

        ai_summary = f"السعر: {current_price}\nSMC: {smc_status}\nالماكرو: {macro_msg}\nHurst: {hurst}\nسيولة: {hidden_above}/{hidden_below}\nGamma: GEX={net_gex}, Flip={gamma_flip}, CallWall={call_wall}, PutWall={put_wall}"
        ai_verdict = await ask_ai_council(ai_summary)
        
        if "انتظار" in ai_verdict:
            report = (
                f"⚖️ *مجلس العباقرة رفض الصفقة*\n"
                f"💰 السعر: `{current_price:.2f}`\n"
                f"📊 SMC: {smc_status}\n"
                f"🧠 *السبب:*\n{ai_verdict}"
            )
            return {"type": "REPORT", "msg": report}, None

        atr_m15 = (df_m15['High'] - df_m15['Low']).rolling(14).mean().iloc[-1]
        entry = round(current_price, 2)
        if signal_dir == 1:
            sl = round(m15_ob['bottom'] - (atr_m15 * 0.8), 2)
            tp1 = round(entry + (entry - sl) * 1.5, 2)
            trade_type = "🟢 شراء كمي"
        else:
            sl = round(m15_ob['top'] + (atr_m15 * 0.8), 2)
            tp1 = round(entry - (sl - entry) * 1.5, 2)
            trade_type = "🔴 بيع كمي"

        # ✅ إدارة حجم المركز
        account_balance = 1000.0 # يمكن تعديلها من إعدادات GitHub
        lot_size = calculate_position_size(account_balance, entry, sl)

        signal_report = (
            f"👑 *إشارة الوحش 31.0*\n"
            f"🕐 الوقت: `{datetime.now(ZoneInfo('Africa/Casablanca')).strftime('%H:%M')}`\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"{trade_type}\n"
            f"🎯 *الدخول:* `{entry}`\n"
            f"🛑 *الوقف:* `{sl}`\n"
            f"🏆 *الهدف:* `{tp1}`\n\n"
            f"📦 *حجم المركز:* `{lot_size}` لوت\n"
            f"📊 SMC: {smc_status}\n"
            f"🔮 الجاما: GEX={net_gex}, Flip={gamma_flip}\n"
            f"💵 الماكرو: {macro_msg}\n"
            f"🧠 *AI:* {ai_verdict}"
        )
        return {"type": "SIGNAL", "msg": signal_report}, None

    except Exception as e:
        return None, f"❌ خطأ: {e}"

async def main():
    payload, err = await run_engine()
    
    if err:
        await send_telegram_msg(f"⚠️ *خطأ:*\n{err}")
        return

    if payload:
        msg = payload["msg"]
        msg_type = payload["type"]

        if msg_type == "REPORT":
            last_report = db_get("last_report_time")
            current_time = datetime.now().strftime("%Y-%m-%d %H")
            if last_report == current_time:
                logging.info("Report already sent this hour.")
                return
            db_set("last_report_time", current_time)
            
        await send_telegram_msg(msg)

if __name__ == '__main__':
    asyncio.run(main())
