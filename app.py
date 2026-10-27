import os
import sqlite3
import warnings
import logging
import asyncio
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import telegram
from openai import OpenAI
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==========================================
# ⚙️ الإعدادات
# ==========================================
TOKEN = os.environ.get("BOT_TOKEN", "").strip()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "").strip()
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

DB_FILE = "quantum_beast_40.db"
KILLZONE_LONDON = (2.0, 5.0)
KILLZONE_NY = (7.0, 11.0)
CONFLUENCE_THRESHOLD = 70 # الحد الأدنى للنقاط باش تصيفط الإشارة

try:
    from google import genai
    ai_client = genai.Client(api_key=GEMINI_API_KEY)
    AI_MODEL = "gemini-3.5-flash-lite"
    ai_enabled = True
except Exception:
    ai_enabled = False

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
# 🚀 XAUS API (السعر الفوري)
# ==========================================
async def get_live_prices():
    try:
        url = "https://xaus.com/api/v1/spot?compact=1"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            gold_price = data.get('spot_usd_oz')
            silver_price = data.get('silver_usd_oz')
            status = data.get('data_state', {}).get('status')
            if status == 'fresh':
                return round(gold_price, 2), round(silver_price, 2)
    except Exception as e:
        logging.warning(f"XAUS failed: {e}")
    try:
        gold = yf.Ticker("GC=F").history(period="1d", interval="1m")
        silver = yf.Ticker("SI=F").history(period="1d", interval="1m")
        if not gold.empty:
            return round(gold['Close'].iloc[-1], 2), round(silver['Close'].iloc[-1], 2) if not silver.empty else None
    except: pass
    return None, None

# ==========================================
# 🌐 جلب البيانات (yfinance)
# ==========================================
async def fetch_data(symbol, period, interval):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if not df.empty: return df
    except: pass
    try:
        ticker = yf.Ticker("XAUUSD=X" if symbol == "GC=F" else symbol)
        df = ticker.history(period=period, interval=interval)
        if not df.empty: return df
    except: pass
    return None

# ==========================================
# ⏰ حظر الأخبار الدقيق (Strict News Blackout)
# ==========================================
def strict_news_blackout():
    try:
        r = requests.get("https://nfs.faireconomy.media/ff_calendar_thisweek.json", timeout=10)
        news_data = r.json()
        now_ny = datetime.now(ZoneInfo("America/New_York"))
        critical = ["Federal Funds Rate", "Non-Farm Employment", "CPI", "FOMC", "PPI", "GDP"]
        
        for news in news_data:
            if news.get('country') not in ['USD', 'XAU'] or news.get('impact') != 'High': continue
            date_str = news.get('date', '')
            if not date_str: continue
            
            news_time = pd.to_datetime(date_str).tz_localize(None).replace(tzinfo=ZoneInfo("America/New_York"))
            title = news.get('title', '')
            
            # حظر من 30 دقيقة قبل لـ 45 دقيقة بعد الخبر
            diff_minutes = (news_time - now_ny).total_seconds() / 60
            
            if -45 <= diff_minutes <= 30 and any(k in title for k in critical):
                return True, f"⏸️ حظر أخبار: {title} (بعد {int(diff_minutes)} دقيقة)"
                
        return False, "آمن"
    except: return False, "تعذر جلب الأخبار"

# ==========================================
# 🧠 المؤشرات والعلوم المخفية
# ==========================================
def calculate_hurst_exponent(price_series, max_lag=20):
    try:
        if len(price_series) < max_lag: return 0.5
        lags = range(2, max_lag)
        tau = [np.sqrt(np.std(np.subtract(price_series[lag:], price_series[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
        return round(poly[0] * 2.0, 2)
    except: return 0.5

def calculate_adr(df_d1):
    try:
        if df_d1 is None or len(df_d1) < 14: return 0
        df_d1['Range'] = df_d1['High'] - df_d1['Low']
        return round(df_d1['Range'].rolling(14).mean().iloc[-1], 2)
    except: return 0

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
        if abs(current_price - ob_bull) < atr * 2: return 1, {'bottom': ob_bull}, f"{label}: 🟢 Bullish OB"
        elif abs(current_price - ob_bear) < atr * 2: return -1, {'top': ob_bear}, f"{label}: 🔴 Bearish OB"
        return 0, None, f"{label}: محايد"
    except: return 0, None, f"{label}: خطأ"

def pca_analysis(df):
    try:
        if df is None or len(df) < 50: return None
        features = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        pca = PCA(n_components=1)
        pca.fit(features)
        return round(pca.explained_variance_ratio_[0] * 100, 2)
    except: return None

def market_maker_simulation(df):
    try:
        if df is None or len(df) < 20: return None, None
        high = df['High'].tail(20).max()
        low = df['Low'].tail(20).min()
        return round(high + (high - low) * 0.2, 2), round(low - (high - low) * 0.2, 2)
    except: return None, None

def calculate_heatmap(df):
    if df is None or len(df) < 30: return None, []
    try:
        hist, bin_edges = np.histogram(df['Close'].values, bins=50, weights=df['Volume'].values)
        poc_idx = np.argmax(hist)
        poc = (bin_edges[poc_idx] + bin_edges[poc_idx+1]) / 2
        return round(float(poc), 2), []
    except: return None, []

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
    except: return "NEUTRAL", "تعذر"

def calculate_stoch_rsi(df, period=14):
    try:
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        stoch_rsi = (rsi - rsi.rolling(period).min()) / (rsi.rolling(period).max() - rsi.rolling(period).min())
        return stoch_rsi.iloc[-1] * 100
    except: return 50.0

# ==========================================
# 🧠 نظام النقاط (Confluence Scoring)
# ==========================================
def calculate_confluence_score(h4_dir, h1_dir, m15_dir, macro_bias, hurst, stoch_rsi, vol_divergence):
    score = 0
    details = []
    
    # 1. SMC Alignment (30 نقطة)
    if h4_dir == 1 and h1_dir == 1 and m15_dir == 1:
        score += 30
        details.append("SMC: +30 (توافق كامل صعود)")
    elif h4_dir == -1 and h1_dir == -1 and m15_dir == -1:
        score += 30
        details.append("SMC: +30 (توافق كامل هبوط)")
    elif h4_dir == 1 and h1_dir == 1:
        score += 15
        details.append("SMC: +15 (توافق جزئي)")
    else:
        details.append("SMC: +0 (تضارب)")
    
    # 2. Macro (20 نقطة)
    if (h4_dir == 1 and macro_bias == "BULLISH_GOLD") or (h4_dir == -1 and macro_bias == "BEARISH_GOLD"):
        score += 20
        details.append("Macro: +20 (متوافق)")
    else:
        details.append("Macro: +0 (غير متوافق)")
    
    # 3. Hurst (20 نقطة)
    if hurst > 0.55:
        score += 20
        details.append(f"Hurst: +20 (ترند قوي: {hurst})")
    elif hurst < 0.45:
        score += 10
        details.append(f"Hurst: +10 (تذبذب: {hurst})")
    else:
        details.append(f"Hurst: +0 (محايد: {hurst})")
    
    # 4. Momentum (15 نقطة)
    if (h4_dir == 1 and stoch_rsi < 40) or (h4_dir == -1 and stoch_rsi > 60):
        score += 15
        details.append(f"Momentum: +15 (مناسب: {stoch_rsi:.1f})")
    else:
        details.append(f"Momentum: +0 (غير مناسب: {stoch_rsi:.1f})")
    
    # 5. Volume (15 نقطة)
    if not vol_divergence:
        score += 15
        details.append("Volume: +15 (بدون تباعد)")
    else:
        details.append("Volume: +0 (تباعد)")
    
    return score, details

# ==========================================
# 🧠 إدارة حجم المركز (Kelly)
# ==========================================
def calculate_position_size(account_balance, entry, sl, kelly_fraction=0.5):
    risk_per_unit = abs(entry - sl)
    if risk_per_unit == 0: return 0.01
    win_rate = 0.45
    reward_ratio = 2.5
    kelly = (win_rate * reward_ratio - (1 - win_rate)) / reward_ratio
    kelly = max(0.01, kelly * kelly_fraction)
    risk_amount = account_balance * kelly
    lot_size = risk_amount / (risk_per_unit * 100)
    return max(0.01, round(lot_size, 2))

# ==========================================
# 🧠 مجلس العباقرة الموسع (4 مزودين AI)
# ==========================================
async def ask_ai_council(data_summary):
    prompt = f"""
    أنت مجلس خبراء تداول (Multi-Agent System) مكون من 5 أعضاء.
    المعطيات الحالية للسوق:
    {data_summary}

    المطلوب منك أن تلعب الأدوار الخمسة بالدارجة المغربية:
    1. 📊 [المحلل الفني]: حلل SMC و Hurst و Momentum.
    2. 📰 [المحلل الأساسي]: حلل الماكرو والأخبار.
    3. 🎭 [محلل المشاعر]: حلل Volume Divergence والفوليوم.
    4. 😈 [محامي الشيطان]: هاجم الصفقة! ابحث عن أي سبب للفشل.
    5. ⚖️ [مدير المخاطر]: أعطي القرار النهائي (شراء / بيع / انتظار) مع نسبة الثقة.

    الرد يجب أن يكون بهذا الشكل:
    📊 المحلل الفني: [جملة]
    📰 المحلل الأساسي: [جملة]
    🎭 محلل المشاعر: [جملة]
    😈 محامي الشيطان: [جملة]
    ⚖️ مدير المخاطر: القرار: [شراء/بيع/انتظار] | الثقة: [%] | السبب: [جملة]
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
    if MISTRAL_API_KEY:
        try:
            client = OpenAI(base_url="https://api.mistral.ai/v1", api_key=MISTRAL_API_KEY)
            response = client.chat.completions.create(model="mistral-small-latest", messages=[{"role": "user", "content": prompt}])
            return response.choices[0].message.content
        except: pass
    return "⚠️ جميع مزودي الذكاء الاصطناعي مشغولين دابا."

# ==========================================
# 🚀 المحرك الرئيسي
# ==========================================
async def run_engine():
    try:
        # 1. فحص الأخبار (Strict News Blackout)
        news_blocked, news_msg = strict_news_blackout()
        if news_blocked:
            return {"type": "NEWS_BLACKOUT", "msg": f"🚨 *حظر أخبار صارم:*\n{news_msg}\n⏸️ البوت مامنعش أي تداول دابا."}, None

        macro_bias, macro_msg = await get_macro_bias()
        
        df_d1 = await fetch_data("GC=F", "1y", "1d"); await asyncio.sleep(1)
        df_h4 = await fetch_data("GC=F", "3mo", "4h"); await asyncio.sleep(1)
        df_h1 = await fetch_data("GC=F", "1mo", "1h"); await asyncio.sleep(1)
        df_m15 = await fetch_data("GC=F", "5d", "15m")

        if any(d is None for d in [df_d1, df_h4, df_h1, df_m15]):
            return None, "❌ تعذر جلب بيانات."

        current_price, silver_price = await get_live_prices()
        if current_price is None:
            current_price = df_m15['Close'].iloc[-1]

        h4_dir, _, h4_msg = analyze_smc(df_h4, "H4")
        h1_dir, h1_ob, h1_msg = analyze_smc(df_h1, "H1")
        m15_dir, m15_ob, m15_msg = analyze_smc(df_m15, "M15")
        
        hurst = calculate_hurst_exponent(df_h1['Close'].values)
        adr = calculate_adr(df_d1)
        stoch_rsi_val = calculate_stoch_rsi(df_m15)
        vol_div, vol_div_msg = detect_volume_divergence(df_h1, "H1")
        
        # 2. حساب النقاط (Confluence Score)
        score, score_details = calculate_confluence_score(h4_dir, h1_dir, m15_dir, macro_bias, hurst, stoch_rsi_val, vol_div)
        
        score_status = "🟢 ممتاز" if score >= 80 else "🟡 جيد" if score >= 70 else "🔴 ضعيف"

        # 3. بناء التقرير
        report = (
            f"👑 *الوحش 40.0 (Confluence + AI Council)*\n"
            f"🕐 الوقت: `{datetime.now(ZoneInfo('Africa/Casablanca')).strftime('%H:%M')}`\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 *النقاط: {score}/100* ({score_status})\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 السعر: `{current_price:.2f}`\n"
            f"💵 الدولار: {macro_msg}\n"
            f"📰 الأخبار: {news_msg}\n"
            f"⚖️ Hurst: {hurst} | 📉 Stoch RSI: {stoch_rsi_val:.1f}\n"
            f"📊 Volume: {vol_div_msg}\n"
            f"📊 SMC: {h4_msg} | {h1_msg} | {m15_msg}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📋 *تفاصيل النقاط:*\n" + "\n".join(score_details) + "\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
        )

        # 4. إلا كانت النقاط عالية، نسولو الذكاء الاصطناعي
        if score >= CONFLUENCE_THRESHOLD:
            ai_summary = f"السعر: {current_price}\nSMC: H4={h4_dir}, H1={h1_dir}, M15={m15_dir}\nMacro: {macro_msg}\nHurst: {hurst}\nStoch RSI: {stoch_rsi_val:.1f}\nVolume Divergence: {vol_div_msg}\nConfluence Score: {score}/100"
            ai_verdict = await ask_ai_council(ai_summary)
            report += f"🧠 *مجلس العباقرة:*\n{ai_verdict}\n"
            
            if "انتظار" not in ai_verdict and "شراء" in ai_verdict or "بيع" in ai_verdict:
                # إشارة كاملة
                atr_m15 = (df_m15['High'] - df_m15['Low']).rolling(14).mean().iloc[-1]
                entry = round(current_price, 2)
                if "شراء" in ai_verdict:
                    sl = round(m15_ob['bottom'] - (atr_m15 * 0.8), 2)
                    tp1 = round(entry + (entry - sl) * 1.5, 2)
                    trade_type = "🟢 شراء كمي"
                else:
                    sl = round(m15_ob['top'] + (atr_m15 * 0.8), 2)
                    tp1 = round(entry - (sl - entry) * 1.5, 2)
                    trade_type = "🔴 بيع كمي"
                
                lot_size = calculate_position_size(1000.0, entry, sl)
                report += (
                    f"\n👑 *إشارة الوحش 40.0*\n"
                    f"{trade_type}\n"
                    f"🎯 *الدخول:* `{entry}`\n"
                    f"🛑 *الوقف:* `{sl}`\n"
                    f"🏆 *الهدف:* `{tp1}`\n"
                    f"📦 *حجم المركز:* `{lot_size}` لوت\n"
                )
            else:
                report += "\n⚖️ *مجلس العباقرة فضل الانتظار.*\n"
        else:
            report += f"\n🧐 *الوضع:* النقاط أقل من {CONFLUENCE_THRESHOLD}. البوت كينتظر توافق أحسن."

        return {"type": "REPORT", "msg": report}, None

    except Exception as e:
        return None, f"❌ خطأ: {e}"

# ==========================================
# 🚀 التشغيل الرئيسي
# ==========================================
async def main():
    print(f"🕐 Starting Bot at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    payload, err = await run_engine()
    
    if err:
        print(f"❌ Error occurred: {err}")
        await send_telegram_msg(f"⚠️ *خطأ:*\n{err}")
        return
        
    if payload:
        msg = payload["msg"]
        print("📤 Sending message to Telegram...")
        await send_telegram_msg(msg)
        print("✅ Message sent successfully!")
    else:
        print("⚠️ No payload and no error returned from engine.")

if __name__ == '__main__':
    asyncio.run(main())
