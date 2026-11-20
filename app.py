"""
=============================================================
🦅 النسر الكامل — النظام السيادي متعدد الطبقات
=============================================================
6 محركات مستقلة + تصويت + إدارة مخاطر تكيفية
=============================================================
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
        logging.FileHandler("eagle_full.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# ============================================================
# ⚙️ الإعدادات
# ============================================================
TOKEN = os.environ.get("BOT_TOKEN", "").strip()
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
TWELVEDATA_API_KEY = os.environ.get("TWELVEDATA_API_KEY", "").strip()

FORCE_TEST = os.environ.get("FORCE_TEST", "0").strip() == "1"

DB_FILE = "eagle_full.db"
NY = ZoneInfo("America/New_York")

# جلسات التداول (بتوقيت نيويورك)
SESSIONS = {
    "ASIA": (18, 2),    # 18:00 - 02:00
    "LONDON": (2, 8),   # 02:00 - 08:00
    "NY": (8, 17),      # 08:00 - 17:00
}

# نوافذ القتل (Kill Zones)
KILL_ZONES = [(2, 5), (8.5, 11.5)]  # London + NY

MAX_SIGNALS_PER_DAY = 12
SIGNAL_COOLDOWN_MIN = 25

# ============================================================
# 🗄️ قاعدة البيانات
# ============================================================
conn = sqlite3.connect(DB_FILE, check_same_thread=False)
conn.executescript("PRAGMA journal_mode=WAL;")
conn.executescript("""
CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS signals(id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, engine TEXT, direction INTEGER, entry REAL, sl REAL, tp1 REAL, tp2 REAL, status TEXT);
""")
conn.commit()

def db_get(key, default=None):
    r = conn.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
    return r[0] if r else default

def db_set(key, value):
    conn.execute("INSERT OR REPLACE INTO state VALUES(?,?)", (key, str(value)))
    conn.commit()

# ============================================================
# 📱 تيليجرام
# ============================================================
async def send_telegram(text):
    if not TOKEN or not CHAT_ID: return
    try:
        bot = telegram.Bot(token=TOKEN)
        await bot.send_message(chat_id=CHAT_ID, text=text, parse_mode='Markdown')
    except Exception as e:
        logging.error(f"Telegram: {e}")

# ============================================================
# 🌐 جلب البيانات
# ============================================================
_td = TDClient(apikey=TWELVEDATA_API_KEY) if TWELVEDATA_API_KEY else None

async def get_price():
    if _td:
        try:
            ts = _td.time_series(symbol="XAU/USD", interval="1min", outputsize=1).as_pandas()
            if ts is not None and not ts.empty: return float(ts['close'].iloc[0])
        except: pass
    try:
        g = yf.Ticker("GC=F").history(period="1d", interval="1m")
        if not g.empty: return float(g["Close"].iloc[-1])
    except: pass
    return None

async def get_candles(interval, limit=200):
    try:
        yf_interval = "1m" if interval == "1m" else interval
        yf_period = "1d" if interval == "1m" else "5d"
        df = yf.Ticker("GC=F").history(period=yf_period, interval=yf_interval, auto_adjust=False)
        if df is not None and not df.empty:
            df = df[["Open","High","Low","Close","Volume"]].dropna()
            df.index = (df.index.tz_localize("UTC") if df.index.tz is None else df.index).tz_convert(NY)
            return df
    except: pass
    return None

async def get_pdh_pdl(current_price):
    """PDH/PDL ذكي مع احتياطي 24h"""
    candidates = []
    try:
        g = yf.Ticker("GC=F").history(period="5d", interval="1d")
        if not g.empty and len(g) >= 2:
            p = g.iloc[-2]
            candidates.append({"high": float(p["High"]), "low": float(p["Low"]),
                               "close": float(p["Close"]), "source": "yf_daily"})
    except: pass
    if _td:
        try:
            ts = _td.time_series(symbol="XAU/USD", interval="1day", outputsize=10).as_pandas()
            if ts is not None and not ts.empty:
                ts.index = pd.to_datetime(ts.index)
                past = ts[ts.index.date < datetime.now(NY).date()]
                if not past.empty:
                    p = past.iloc[-1]
                    candidates.append({"high": float(p["high"]), "low": float(p["low"]),
                                       "close": float(p["close"]), "source": "td_daily"})
        except: pass
    if candidates and current_price:
        best, bd = None, 999
        for c in candidates:
            d = abs(current_price - c["close"]) / current_price * 100
            if d < bd: bd, best = d, c
        if bd < 3: return best
    try:
        df = await get_candles("15m", 500)
        if df is not None and len(df) >= 20:
            h24 = df.tail(96)
            return {"high": float(h24['High'].max()), "low": float(h24['Low'].min()),
                    "close": float(h24['Close'].iloc[-1]), "source": "24h"}
    except: pass
    return None

# ============================================================
# 🧬 المحرك 1: HMM واعي بالجلسة
# ============================================================
try:
    from hmmlearn import hmm
    HMM_OK = True
except:
    HMM_OK = False
    logging.warning("hmmlearn غير مثبت → HMM معطل")

def get_session():
    now = datetime.now(NY)
    h = now.hour + now.minute / 60
    if 18 <= h or h < 2: return "ASIA"
    if 2 <= h < 8: return "LONDON"
    return "NY"

def detect_regime_hmm(df_m15):
    """HMM مبسط: 3 حالات (0=نطاق، 1=اتجاه، 2=ذعر)"""
    if not HMM_OK or df_m15 is None or len(df_m15) < 100:
        return detect_regime_simple(df_m15)
    try:
        returns = df_m15['Close'].pct_change().dropna().values.reshape(-1, 1)
        if len(returns) < 50: return 1
        model = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=100, random_state=42)
        model.fit(returns)
        states = model.predict(returns)
        current = states[-1]
        # ترتيب: الحالة ذات أعلى تقلب = ذعر
        variances = model.covars_.flatten()
        sorted_idx = np.argsort(variances)
        mapping = {sorted_idx[0]: 0, sorted_idx[1]: 1, sorted_idx[2]: 2}
        return mapping.get(current, 1)
    except:
        return detect_regime_simple(df_m15)

def detect_regime_simple(df_m15):
    if df_m15 is None or len(df_m15) < 50: return 1
    close = df_m15['Close']
    ema20 = close.ewm(span=20).mean()
    ema50 = close.ewm(span=50).mean()
    atr = (df_m15['High'] - df_m15['Low']).rolling(14).mean().iloc[-1]
    slope = (ema20.iloc[-1] - ema20.iloc[-10]) / 10
    vol = df_m15['Close'].pct_change().tail(20).std()
    if vol > 0.008: return 2
    if abs(slope) > 0.05 * atr: return 1
    return 0

def session_rules(session, regime):
    """
    قواعد الجلسة:
    ASIA: نطاق قوي → لا نتداول الاتجاه
    LONDON: انتقال من نطاق لاتجاه → نسمح
    NY: اتجاه قوي → نسمح
    """
    if session == "ASIA" and regime == 0: return False, "آسيا/نطاق"
    if session == "ASIA" and regime == 2: return False, "آسيا/ذعر"
    if regime == 2: return False, "ذعر عالمي"
    return True, "OK"

# ============================================================
# 🏛️ المحرك 2: GARM-X (بوابة إعادة التسعير)
# ============================================================
def build_auction_map(df_m15, pdh, pdl):
    """خريطة السيولة"""
    levels = {"PDH": pdh, "PDL": pdl}
    try:
        today = datetime.now(NY).date()
        today_df = df_m15[df_m15.index.date == today]
        if len(today_df) > 10:
            levels["ASIA_H"] = float(today_df['High'].iloc[:50].max()) if len(today_df) > 50 else float(today_df['High'].max())
            levels["ASIA_L"] = float(today_df['Low'].iloc[:50].min()) if len(today_df) > 50 else float(today_df['Low'].min())
    except: pass
    try:
        df_d = yf.Ticker("GC=F").history(period="14d", interval="1d")
        if len(df_d) >= 7:
            levels["PWH"] = float(df_d['High'].iloc[-7:].max())
            levels["PWL"] = float(df_d['Low'].iloc[-7:].min())
    except: pass
    return levels

def calculate_vwap(df):
    try:
        if df is None or len(df) < 10: return None
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        vwap = (tp * df['Volume']).cumsum() / df['Volume'].cumsum()
        return float(vwap.iloc[-1])
    except: return None

def detect_liquidity_sweep(df_m5, levels, current_price):
    """كشف كنس السيولة"""
    if df_m5 is None or len(df_m5) < 20: return None
    try:
        recent = df_m5.tail(10)
        for name, level in levels.items():
            if level is None: continue
            # كنس تحت مستوى (شراء)
            if recent['Low'].min() < level and current_price > level:
                return {"type": "bullish", "level": level, "name": name}
            # كنس فوق مستوى (بيع)
            if recent['High'].max() > level and current_price < level:
                return {"type": "bearish", "level": level, "name": name}
    except: pass
    return None

def confirm_displacement(df_m1, sweep):
    """تأكيد الإزاحة"""
    if df_m1 is None or len(df_m1) < 5 or not sweep: return False
    try:
        last = df_m1.iloc[-1]
        body = abs(last['Close'] - last['Open'])
        atr = (df_m1['High'] - df_m1['Low']).rolling(14).mean().iloc[-1]
        if body < atr * 0.8: return False
        if sweep["type"] == "bullish" and last['Close'] > last['Open']:
            return True
        if sweep["type"] == "bearish" and last['Close'] < last['Open']:
            return True
    except: pass
    return False

def repricing_gate(vwap, current_price, regime, session):
    """بوابة إعادة التسعير GARM-X"""
    if vwap is None: return False, "لا VWAP"
    if regime == 0: return False, "نطاق"
    if regime == 2: return False, "ذعر"
    if session == "ASIA": return False, "آسيا لا تعيد التسعير"
    # السعر يجب أن يكون فوق/تحت VWAP باتجاه واضح
    if abs(current_price - vwap) / current_price < 0.001:
        return False, "قريب من VWAP"
    return True, "OK"

async def garm_x_engine(df_m15, df_m5, df_m1, current_price, pdh, pdl):
    """محرك GARM-X الكامل"""
    levels = build_auction_map(df_m15, pdh, pdl)
    vwap = calculate_vwap(df_m5)
    sweep = detect_liquidity_sweep(df_m5, levels, current_price)
    if not sweep: return None
    if not confirm_displacement(df_m1, sweep): return None
    session = get_session()
    regime = detect_regime_hmm(df_m15)
    gate_ok, gate_msg = repricing_gate(vwap, current_price, regime, session)
    if not gate_ok: return None
    
    atr = (df_m15['High'] - df_m15['Low']).rolling(14).mean().iloc[-1]
    if sweep["type"] == "bullish":
        entry = current_price
        sl = round(sweep["level"] - atr * 0.3, 2)
        risk = entry - sl
        if risk <= 0 or risk > atr * 3: return None
        return {
            "engine": "GARM-X", "direction": 1, "entry": entry, "sl": sl,
            "tp1": round(entry + risk * 1.5, 2), "tp2": round(entry + risk * 3.0, 2),
            "confidence": 72, "reason": f"🟢 كنس {sweep['name']} + إزاحة + بوابة"
        }
    else:
        entry = current_price
        sl = round(sweep["level"] + atr * 0.3, 2)
        risk = sl - entry
        if risk <= 0 or risk > atr * 3: return None
        return {
            "engine": "GARM-X", "direction": -1, "entry": entry, "sl": sl,
            "tp1": round(entry - risk * 1.5, 2), "tp2": round(entry - risk * 3.0, 2),
            "confidence": 72, "reason": f"🔴 كنس {sweep['name']} + إزاحة + بوابة"
        }

# ============================================================
# 🧠 المحرك 3: FinBERT Sentiment
# ============================================================
try:
    from transformers import pipeline
    _finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    FINBERT_OK = True
except:
    FINBERT_OK = False
    logging.warning("FinBERT غير متاح → استخدام LLM")

async def get_news():
    try:
        g = yf.Ticker("GC=F")
        news = g.news
        if not news: return []
        return [n.get('title', '') for n in news[:5] if n.get('title')]
    except: return []

async def sentiment_score(headlines):
    if not headlines: return 0.0
    if FINBERT_OK:
        try:
            scores = []
            for h in headlines:
                r = _finbert(h[:512])[0]
                label, conf = r['label'], r['score']
                if label == 'positive': scores.append(conf)
                elif label == 'negative': scores.append(-conf)
                else: scores.append(0)
            return float(np.mean(scores))
        except: pass
    # fallback LLM
    prompt = f"حلل مشاعر هذه العناوين تجاه الذهب. أعد رقمًا فقط بين -1 و +1:\n{chr(10).join(headlines)}"
    if GEMINI_API_KEY:
        try:
            from google import genai
            c = genai.Client(api_key=GEMINI_API_KEY)
            r = await c.aio.models.generate_content(model="gemini-2.5-flash", contents=prompt)
            return float(r.text.strip())
        except: pass
    return 0.0

# ============================================================
# 🎲 المحرك 4: PPO Ensemble (محاكاة)
# ============================================================
# ملاحظة: stable-baselines3 يتطلب تدريبًا. هنا نستخدم
# نموذجًا مبسطًا يحاكي منطق Ensemble عبر قواعد تعلم معزز.
try:
    from stable_baselines3 import PPO
    PPO_OK = True
    logging.info("stable-baselines3 متاح")
except:
    PPO_OK = False
    logging.info("SB3 غير متاح → PPO مبسط")

def ppo_signal(df_m15, df_m5):
    """
    إشارة PPO مبسطة: تستخدم مزيجًا من الاتجاه والزخم
    كبديل عملي عن النموذج المدرب.
    """
    if df_m15 is None or len(df_m15) < 50: return None
    try:
        ema9 = df_m5['Close'].ewm(span=9).mean()
        ema21 = df_m5['Close'].ewm(span=21).mean()
        rsi = calc_rsi(df_m5['Close'])
        macd, sig, _ = calc_macd(df_m5['Close'])
        
        buy_votes = 0; sell_votes = 0
        if ema9.iloc[-1] > ema21.iloc[-1]: buy_votes += 1
        else: sell_votes += 1
        if rsi.iloc[-1] < 40: buy_votes += 1
        elif rsi.iloc[-1] > 60: sell_votes += 1
        if macd.iloc[-1] > sig.iloc[-1]: buy_votes += 1
        else: sell_votes += 1
        
        if buy_votes >= 2 and sell_votes <= 1:
            entry = float(df_m5['Close'].iloc[-1])
            atr = (df_m15['High'] - df_m15['Low']).rolling(14).mean().iloc[-1]
            sl = round(entry - atr * 1.2, 2)
            risk = entry - sl
            return {"engine": "PPO", "direction": 1, "entry": entry, "sl": sl,
                    "tp1": round(entry + risk * 1.5, 2), "tp2": round(entry + risk * 2.5, 2),
                    "confidence": 58 + buy_votes * 5, "reason": "🤖 PPO Ensemble صعودي"}
        if sell_votes >= 2 and buy_votes <= 1:
            entry = float(df_m5['Close'].iloc[-1])
            atr = (df_m15['High'] - df_m15['Low']).rolling(14).mean().iloc[-1]
            sl = round(entry + atr * 1.2, 2)
            risk = sl - entry
            return {"engine": "PPO", "direction": -1, "entry": entry, "sl": sl,
                    "tp1": round(entry - risk * 1.5, 2), "tp2": round(entry - risk * 2.5, 2),
                    "confidence": 58 + sell_votes * 5, "reason": "🤖 PPO Ensemble هبوطي"}
    except: pass
    return None

# ============================================================
# 📊 مؤشرات مساعدة
# ============================================================
def calc_rsi(series, period=14):
    d = series.diff()
    g = d.where(d > 0, 0).rolling(period).mean()
    l = -d.where(d < 0, 0).rolling(period).mean()
    return 100 - (100 / (1 + g / (l + 1e-9)))

def calc_ema(series, p): return series.ewm(span=p, adjust=False).mean()
def calc_macd(s):
    f = calc_ema(s, 12); sl = calc_ema(s, 26); m = f - sl
    return m, calc_ema(m, 9), m - calc_ema(m, 9)

# ============================================================
# 🛡️ المحرك 6: إدارة المخاطر التكيفية
# ============================================================
def adaptive_risk(regime, session, confidence):
    """حجم المركز حسب النظام والجلسة والثقة"""
    base = 1.0
    if regime == 2: return 0  # ذعر = لا تداول
    if session == "ASIA": base *= 0.5
    if session == "LONDON": base *= 1.0
    if session == "NY": base *= 1.2
    if regime == 1: base *= 1.1
    base *= (confidence / 100)
    return min(base, 2.0)

# ============================================================
# 🚀 المحرك الرئيسي: التصويت
# ============================================================
def count_today():
    today = datetime.now(NY).strftime("%Y-%m-%d")
    return int(db_get(f"signals_{today}", "0"))

def can_send():
    if count_today() >= MAX_SIGNALS_PER_DAY: return False, "سقف يومي"
    lt = db_get("last_sig_ts", "0")
    try:
        if (time.time() - float(lt)) / 60 < SIGNAL_COOLDOWN_MIN:
            return False, "cooldown"
    except: pass
    return True, "OK"

async def run_eagle():
    # جلب البيانات
    current = await get_price()
    if current is None: return None, None
    
    df_m15 = await get_candles("15m", 200)
    df_m5 = await get_candles("5m", 200)
    df_m1 = await get_candles("1m", 100)
    if any(d is None for d in [df_m15, df_m5, df_m1]): return None, None
    
    pdh_pdl = await get_pdh_pdl(current)
    if not pdh_pdl: return None, None
    
    # فحص السلامة
    if abs(current - pdh_pdl["close"]) / current > 0.05:
        return None, None
    
    session = get_session()
    regime = detect_regime_hmm(df_m15)
    
    # قاعدة الجلسة
    session_ok, sess_msg = session_rules(session, regime)
    if not session_ok:
        if FORCE_TEST:
            return (f"🦅 *النسر — اختبار*\n"
                    f"🕐 `{datetime.now(NY).strftime('%H:%M')}` | {session}\n"
                    f"💰 `{current:.2f}` | PDH `{pdh_pdl['high']:.2f}` PDL `{pdh_pdl['low']:.2f}`\n"
                    f"🧬 النظام: {['نطاق','اتجاه','ذعر'][regime]}\n"
                    f"⏸️ {sess_msg}"), None
        return None, None
    
    # جمع إشارات المحركات
    votes = []
    
    # 1. GARM-X
    g = await garm_x_engine(df_m15, df_m5, df_m1, current, pdh_pdl["high"], pdh_pdl["low"])
    if g: votes.append(g)
    
    # 2. PPO
    p = ppo_signal(df_m15, df_m5)
    if p: votes.append(p)
    
    # 3. ICT (Judas Swing)
    if is_kill_zone():
        ict = detect_judas_swing_simple(df_m15, df_m1, pdh_pdl["high"], pdh_pdl["low"])
        if ict: votes.append(ict)
    
    # 4. VWAP Bounce
    vwap = calculate_vwap(df_m5)
    if vwap:
        vb = vwap_bounce(df_m5, vwap)
        if vb: votes.append(vb)
    
    if not votes:
        if FORCE_TEST:
            return (f"🦅 *النسر — اختبار*\n"
                    f"🕐 `{datetime.now(NY).strftime('%H:%M')}` | {session}\n"
                    f"💰 `{current:.2f}` | PDH `{pdh_pdl['high']:.2f}` PDL `{pdh_pdl['low']:.2f}`\n"
                    f"🧬 {['نطاق','اتجاه','ذعر'][regime]} | VWAP `{vwap:.2f}`\n"
                    f"🧐 لا إشارة من المحركات"), None
        return None, None
    
    # التصويت: نجمع الإشارات المتوافقة
    buy = [v for v in votes if v["direction"] == 1]
    sell = [v for v in votes if v["direction"] == -1]
    
    best_dir = 1 if len(buy) > len(sell) else (-1 if len(sell) > len(buy) else 0)
    if best_dir == 0: return None, None
    
    chosen = max(buy if best_dir == 1 else sell, key=lambda x: x["confidence"])
    
    # الفلاتر النهائية
    can, reason = can_send()
    if not can: return None, None
    
    # المشاعر
    headlines = await get_news()
    sent = await sentiment_score(headlines)
    if best_dir == 1 and sent < -0.35: 
        logging.info(f"❌ Sentiment {sent:.2f} يرفض الشراء")
        return None, None
    if best_dir == -1 and sent > 0.35: 
        logging.info(f"❌ Sentiment {sent:.2f} يرفض البيع")
        return None, None
    
    # المخاطرة
    risk_mult = adaptive_risk(regime, session, chosen["confidence"])
    if risk_mult == 0: return None, None
    
    # حفظ
    today = datetime.now(NY).strftime("%Y-%m-%d")
    db_set(f"signals_{today}", str(count_today() + 1))
    db_set("last_sig_ts", str(time.time()))
    
    # بناء الرسالة
    dir_txt = "🟢 شراء" if best_dir == 1 else "🔴 بيع"
    report = (
        f"🦅 *النسر الكامل — إشارة سيادية*\n"
        f"🕐 `{datetime.now(NY).strftime('%H:%M')}` | {session}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 السعر: `{current:.2f}`\n"
        f"📊 PDH `{pdh_pdl['high']:.2f}` | PDL `{pdh_pdl['low']:.2f}`\n"
        f"🧬 النظام: *{['نطاق','اتجاه','ذعر'][regime]}*\n"
        f"📊 VWAP: `{vwap:.2f if vwap else 'N/A'}`\n"
        f"📰 Sentiment: *{sent:+.2f}*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🎯 *المحرك:* {chosen['engine']}\n"
        f"💡 {chosen['reason']}\n"
        f"🔥 الثقة: *{chosen['confidence']}%*\n"
        f"⚖️ المخاطرة: *{risk_mult:.1f}x*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🎯 *الدخول:* `{chosen['entry']:.2f}`\n"
        f"{dir_txt}\n"
        f"🛑 *الوقف:* `{chosen['sl']:.2f}`\n"
        f"🏆 *TP1:* `{chosen['tp1']:.2f}`\n"
        f"🚀 *TP2:* `{chosen['tp2']:.2f}`\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 اليوم: {count_today()}/{MAX_SIGNALS_PER_DAY}\n"
        f"🗳️ تصويت: {len(buy)} شراء | {len(sell)} بيع\n"
    )
    return report, None

def is_kill_zone():
    if FORCE_TEST: return True
    h = datetime.now(NY).hour + datetime.now(NY).minute / 60
    return any(s <= h < e for s, e in KILL_ZONES)

def detect_judas_swing_simple(df_m15, df_m1, pdh, pdl):
    if df_m15 is None or len(df_m15) < 5: return None
    r = df_m15.tail(5)
    if r['High'].max() > pdh and df_m1['Close'].iloc[-1] < df_m1['Low'].iloc[-3]:
        return {"engine": "ICT", "direction": -1, "entry": float(df_m1['Close'].iloc[-1]),
                "sl": round(pdh + 1, 2), "tp1": round(pdh - 15, 2), "tp2": round(pdh - 30, 2),
                "confidence": 70, "reason": "💀 فخ فوق PDH"}
    if r['Low'].min() < pdl and df_m1['Close'].iloc[-1] > df_m1['High'].iloc[-3]:
        return {"engine": "ICT", "direction": 1, "entry": float(df_m1['Close'].iloc[-1]),
                "sl": round(pdl - 1, 2), "tp1": round(pdl + 15, 2), "tp2": round(pdl + 30, 2),
                "confidence": 70, "reason": "💀 فخ تحت PDL"}
    return None

def vwap_bounce(df_m5, vwap):
    if df_m5 is None or len(df_m5) < 20: return None
    try:
        last, prev = df_m5.iloc[-1], df_m5.iloc[-2]
        atr = (df_m5['High'] - df_m5['Low']).rolling(14).mean().iloc[-1]
        if prev['Low'] <= vwap and last['Close'] > vwap and last['Close'] > last['Open']:
            e = float(last['Close']); sl = round(vwap - atr * 0.5, 2)
            r = e - sl
            return {"engine": "VWAP", "direction": 1, "entry": e, "sl": sl,
                    "tp1": round(e + r * 1.5, 2), "tp2": round(e + r * 2.5, 2),
                    "confidence": 62, "reason": "🟢 ارتداد VWAP"}
        if prev['High'] >= vwap and last['Close'] < vwap and last['Close'] < last['Open']:
            e = float(last['Close']); sl = round(vwap + atr * 0.5, 2)
            r = sl - e
            return {"engine": "VWAP", "direction": -1, "entry": e, "sl": sl,
                    "tp1": round(e - r * 1.5, 2), "tp2": round(e - r * 2.5, 2),
                    "confidence": 62, "reason": "🔴 ارتداد VWAP"}
    except: pass
    return None

# ============================================================
# 🚀 الحلقة الرئيسية
# ============================================================
async def main_loop():
    logging.info("🦅 النسر الكامل بدأ")
    await send_telegram(
        "🦅 *النسر الكامل — النظام السيادي*\n"
        f"🕐 {datetime.now(NY).strftime('%Y-%m-%d %H:%M')}\n"
        f"🧬 6 محركات: HMM + GARM-X + FinBERT + PPO + ICT + مخاطر\n"
        f"📊 سقف: {MAX_SIGNALS_PER_DAY} إشارة/يوم"
    )
    
    last_check = 0
    while True:
        try:
            if time.time() - last_check > 120:
                report, err = await run_eagle()
                if err: await send_telegram(err)
                elif report: await send_telegram(report)
                last_check = time.time()
            await asyncio.sleep(30)
        except KeyboardInterrupt:
            logging.info("⛔ إيقاف")
            break
        except Exception as e:
            logging.error(f"Loop: {e}")
            await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(main_loop())
