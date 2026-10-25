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

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==========================================
# ⚙️ الإعدادات
# ==========================================
TOKEN = os.environ.get("BOT_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

DB_FILE = "quantum_beast_28.db"
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
# 🌐 جلب البيانات
# ==========================================
async def fetch_data(symbol, period, interval):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        return df if not df.empty else None
    except Exception:
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
        return False, None
    except Exception:
        return False, None

# ==========================================
# 🧠 مجلس العباقرة (Multi-Provider AI)
# ==========================================
async def ask_ai_council(data_summary):
    prompt = f"""
    أنت مجلس خبراء تداول (Multi-Agent System) مكون من 3 أعضاء.
    المعطيات الحالية للسوق:
    {data_summary}

    المطلوب منك أن تلعب الأدوار الثلاثة بالدارجة المغربية:
    1. 🧠 [المحلل الفني]: قدم حجة قوية بناءً على المعطيات (SMC, PCA, Hurst, Fib, OI).
    2. 😈 [محامي الشيطان]: هاجم الصفقة! ابحث عن أي سبب للفشل (فخ سيولة، تضارب ماكرو، إشارة كاذبة).
    3. ⚖️ [مدير المخاطر]: استمع للرأيين، وأعطي القرار النهائي (شراء / بيع / انتظار).

    إذا كان محامي الشيطان عنده حجة قوية، يجب على مدير المخاطر أن يختار "انتظار" لحماية رأس المال.

    الرد يجب أن يكون بهذا الشكل بالضبط:
    🧠 المحلل: [جملة واحدة]
    😈 محامي الشيطان: [جملة واحدة]
    ⚖️ مدير المخاطر: القرار: [شراء/بيع/انتظار] | السبب: [جملة واحدة]
    """
    
    # 1. المحاولة الأولى: Gemini
    if ai_enabled:
        try:
            response = await ai_client.aio.models.generate_content(model=AI_MODEL, contents=prompt)
            return response.text
        except Exception as e:
            logging.warning(f"Gemini failed: {e}. Falling back to Groq.")

    # 2. المحاولة الثانية: Groq
    if GROQ_API_KEY:
        try:
            client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.warning(f"Groq failed: {e}. Falling back to OpenRouter.")

    # 3. المحاولة الثالثة: OpenRouter
    if OPENROUTER_API_KEY:
        try:
            client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
            response = client.chat.completions.create(
                model="meta-llama/llama-3.3-70b-instruct:free",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"All AI providers failed: {e}")

    return "⚠️ جميع مزودي الذكاء الاصطناعي مشغولين دابا، تم الاعتماد على التحليل الفني."

# ==========================================
# 🚀 المحرك الرئيسي
# ==========================================
async def run_engine(use_ai=True):
    danger, msg = is_news_dangerous()
    if danger: return None, f"⚠️ درع الأخبار: {msg}"
    try:
        macro_bias, macro_msg = await get_macro_bias()
        df_d1 = await fetch_data("GC=F", "1y", "1d"); await asyncio.sleep(1)
        df_h4 = await fetch_data("GC=F", "3mo", "4h"); await asyncio.sleep(1)
        df_h1 = await fetch_data("GC=F", "1mo", "1h"); await asyncio.sleep(1)
        df_m15 = await fetch_data("GC=F", "5d", "15m")
        dxy_data = await fetch_data("DX-Y.NYB", "1mo", "1h")

        if any(d is None for d in [df_d1, df_h4, df_h1, df_m15]):
            return None, "❌ تعذر جلب بيانات."

        h4_dir, h4_ob, h4_msg = analyze_smc(df_h4, "H4")
        h1_dir, h1_ob, h1_msg = analyze_smc(df_h1, "H1")
        m15_dir, m15_ob, m15_msg = analyze_smc(df_m15, "M15")
        
        corr = calculate_correlation(df_h1, dxy_data)
        predicted = ai_predict_price(df_h1)
        poc, hot_zones = calculate_heatmap(df_h1)
        pca_score = pca_analysis(df_h1)
        hidden_above, hidden_below = market_maker_simulation(df_h1)
        
        hurst = calculate_hurst_exponent(df_h1['Close'].values)
        adr = calculate_adr(df_d1)
        fib_low, fib_high = get_golden_pocket(df_h1)
        
        oi_msg = "غير متاح"
        try:
            if 'Open Interest' in df_h1.columns and df_h1['Open Interest'].iloc[-1] > 0:
                oi_msg = f"{int(df_h1['Open Interest'].iloc[-1])}"
        except Exception:
            pass

        signal_dir = 0; ob_zone = None
        if h4_dir == 1 and h1_dir == 1 and m15_dir == 1:
            signal_dir = 1; ob_zone = m15_ob
        elif h4_dir == -1 and h1_dir == -1 and m15_dir == -1:
            signal_dir = -1; ob_zone = m15_ob

        if signal_dir == 0:
            return None, None

        if signal_dir == 1 and macro_bias == "BEARISH_GOLD": return None, None
        if signal_dir == -1 and macro_bias == "BULLISH_GOLD": return None, None
        if abs(corr) < 0.3: return None, None

        if predicted:
            if signal_dir == 1 and predicted < df_h1['Close'].iloc[-1]: return None, None
            if signal_dir == -1 and predicted > df_h1['Close'].iloc[-1]: return None, None

        current_price = df_m15['Close'].iloc[-1]
        atr_m15 = (df_m15['High'] - df_m15['Low']).rolling(14).mean().iloc[-1]
        entry = round(current_price, 2)

        if signal_dir == 1:
            sl = round(ob_zone['bottom'] - (atr_m15 * 0.8), 2)
            tp1 = round(entry + (entry - sl) * 1.5, 2)
            tp2 = round(entry + (entry - sl) * 2.5, 2)
            trade_type = "🟢 شراء كمي"
        else:
            sl = round(ob_zone['top'] + (atr_m15 * 0.8), 2)
            tp1 = round(entry - (sl - entry) * 1.5, 2)
            tp2 = round(entry - (sl - entry) * 2.5, 2)
            trade_type = "🔴 بيع كمي"

        ai_verdict = "⚠️ AI غير مستعمل."
        if use_ai:
            ai_summary = f"""
            - السعر: {current_price} | ADR: {adr}
            - H4: {h4_msg} | H1: {h1_msg} | M15: {m15_msg}
            - الماكرو: {macro_msg} | الارتباط: {corr} | Open Interest: {oi_msg}
            - Hurst: {hurst} (فوق 0.5 ترند، تحت 0.5 تذبذب)
            - Fib Golden Pocket: {fib_low} - {fib_high}
            - توقع الشبكة: {predicted} | PCA: {pca_score}%
            - السيولة: فوق {hidden_above} | تحت {hidden_below}
            - نوع الإشارة الفنية: {trade_type}
            """
            ai_verdict = await ask_ai_council(ai_summary)
            
            if "انتظار" in ai_verdict:
                logging.info("Council Vetoed the trade.")
                return None, None

        payload = {
            "time": str(df_m15.index[-1]), "type": trade_type,
            "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2,
            "macro": macro_msg, "smc": f"H4: {h4_msg}\nH1: {h1_msg}\nM15: {m15_msg}",
            "corr": f"🔗 الارتباط: {corr}", "nn": f"🧬 توقع الشبكة: {predicted}",
            "heat": f"🔥 POC: {poc}", 
            "dark": f"🔮 PCA: {pca_score}% | ⚫ سيولة: فوق {hidden_above} | تحت {hidden_below}",
            "hidden_science": f"🔬 Hurst: {hurst} | 📏 ADR: {adr} | 🧬 OI: {oi_msg}\n🎯 Fib Pocket: {fib_low} - {fib_high}",
            "ai": ai_verdict
        }
        return payload, None
    except Exception as e:
        return None, f"❌ خطأ: {e}"

# ==========================================
# 📱 واجهة تليجرام
# ==========================================
def format_card(sig):
    now_ma = datetime.now(ZoneInfo("Africa/Casablanca")).strftime("%Y-%m-%d %H:%M:%S")
    return (
        f"👑 *الوحش 28.0 (Multi-AI)*\n"
        f"🕐 *وقت التحليل:* `{now_ma}`\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"{sig['type']}\n\n"
        f"🎯 *الدخول:* `{sig['entry']:.2f}`\n"
        f"🛑 *الوقف:* `{sig['sl']:.2f}`\n"
        f"🏆 *الهدف 1:* `{sig['tp1']:.2f}`\n"
        f"🚀 *الهدف 2:* `{sig['tp2']:.2f}`\n\n"
        f"📊 *SMC:*\n{sig['smc']}\n"
        f"{sig['corr']}\n"
        f"{sig['nn']}\n"
        f"{sig['heat']}\n"
        f"{sig['dark']}\n"
        f"{sig['hidden_science']}\n"
        f"💵 *الماكرو:* {sig['macro']}\n\n"
        f"🧠 *مجلس العباقرة:*\n{sig['ai']}\n"
    )

async def main():
    if not is_in_killzone():
        logging.info("خارج أوقات السيولة.")
        return

    sig, err = await run_engine(use_ai=True)
    bot = telegram.Bot(token=TOKEN)

    if err:
        await bot.send_message(chat_id=CHAT_ID, text=err)
    elif sig:
        card = format_card(sig)
        await bot.send_message(chat_id=CHAT_ID, text=card, parse_mode='Markdown')

if __name__ == '__main__':
    asyncio.run(main())
