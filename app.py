import os
import json
import sqlite3
import warnings
import logging
import asyncio
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from contextlib import asynccontextmanager

import yfinance as yf
import pandas as pd
import numpy as np
import requests
from fastapi import FastAPI, Request, Response
import uvicorn
from google import genai
from google.genai import types

from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, Application

from smart_money_concept.smc import SmartMoneyConcepts

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==========================================
# ⚙️ إعدادات النظام (تقرأ من Render Environment)
# ==========================================
TOKEN = os.environ.get("BOT_TOKEN", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
RENDER_URL = os.environ.get("RENDER_EXTERNAL_URL", "")

DB_FILE = "quantum_beast_16.db"
RISK_PER_TRADE = 1.0      # نسبة المخاطرة لكل صفقة
MAX_DAILY_LOSS = 2.0      # أقصى خسارة يومية
KILLZONE_LONDON = (2.0, 5.0)  # توقيت نيويورك
KILLZONE_NY = (7.0, 11.0)

# تهيئة الذكاء الاصطناعي (Gemini 3.8 Flash)
try:
    ai_client = genai.Client(api_key=GEMINI_API_KEY)
    AI_MODEL = "gemini-3.8-flash"
    ai_enabled = True
    logging.info("✅ AI Ready: Gemini 3.8 Flash")
except Exception as e:
    logging.error(f"⚠️ AI Initialization Failed: {e}")
    ai_enabled = False

# ==========================================
# 🗄️ قاعدة البيانات (SQLite)
# ==========================================
def init_db():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS trades(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        time TEXT, direction TEXT, entry REAL, sl REAL, tp1 REAL, tp2 REAL, result REAL
    );
    CREATE TABLE IF NOT EXISTS state(
        key TEXT PRIMARY KEY, value TEXT
    );
    """)
    conn.commit()
    return conn

conn = init_db()

def db_get(key, default=None):
    r = conn.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
    return r[0] if r else default

def db_set(key, value):
    conn.execute("INSERT OR REPLACE INTO state VALUES(?,?)", (key, str(value)))
    conn.commit()

# ==========================================
# 🕐 جلسات السيولة (Killzones)
# ==========================================
def is_in_killzone():
    now = datetime.now(ZoneInfo("America/New_York"))
    if now.weekday() >= 5: return False
    t = now.hour + now.minute / 60
    return (KILLZONE_LONDON[0] <= t < KILLZONE_LONDON[1]) or (KILLZONE_NY[0] <= t < KILLZONE_NY[1])

# ==========================================
# 🌐 جلب البيانات (yfinance) مع حماية
# ==========================================
async def fetch_data(symbol, period, interval):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if df.empty: return None
        return df
    except Exception:
        return None

# ==========================================
# 🧠 تحليل SMC (Order Blocks, FVG, Sweeps)
# ==========================================
def analyze_smc(df, label):
    if df is None or len(df) < 50: return 0, None, f"{label}: بيانات غير كافية"
    try:
        # استخدام مكتبة SMC الجاهزة
        smc = SmartMoneyConcepts(df)
        ob = smc.order_blocks()
        fvg = smc.fair_value_gaps()
        
        # منطق بسيط للاتجاه: إذا كان آخر OB شراء قريب من السعر
        current_price = df['Close'].iloc[-1]
        atr = (df['High'] - df['Low']).rolling(14).mean().iloc[-1]
        
        recent_bull_ob = ob[ob['type'] == 'bull'].tail(1)
        recent_bear_ob = ob[ob['type'] == 'bear'].tail(1)
        
        if not recent_bull_ob.empty and abs(current_price - recent_bull_ob['top'].iloc[-1]) < atr * 2:
            return 1, recent_bull_ob.iloc[-1], f"{label}: 🟢 Bullish OB"
        elif not recent_bear_ob.empty and abs(current_price - recent_bear_ob['bottom'].iloc[-1]) < atr * 2:
            return -1, recent_bear_ob.iloc[-1], f"{label}: 🔴 Bearish OB"
        return 0, None, f"{label}: محايد"
    except Exception as e:
        return 0, None, f"{label}: خطأ SMC ({e})"

# ==========================================
# 📊 تحليل الحجم (Volume Profile)
# ==========================================
def volume_profile(df):
    if df is None or len(df) < 30: return None, None, None
    try:
        prices = df['Close'].values
        volumes = df['Volume'].values
        hist, bin_edges = np.histogram(prices, bins=50, weights=volumes)
        poc_idx = np.argmax(hist)
        poc = (bin_edges[poc_idx] + bin_edges[poc_idx+1]) / 2
        total_vol = hist.sum()
        target = total_vol * 0.70
        current_vol = hist[poc_idx]
        vah_idx, val_idx = poc_idx, poc_idx
        while current_vol < target:
            if vah_idx + 1 < len(hist) and (val_idx == 0 or hist[vah_idx+1] >= hist[val_idx-1]):
                vah_idx += 1; current_vol += hist[vah_idx]
            elif val_idx - 1 >= 0:
                val_idx -= 1; current_vol += hist[val_idx]
            else: break
        vah = (bin_edges[vah_idx] + bin_edges[vah_idx+1]) / 2
        val = (bin_edges[val_idx] + bin_edges[val_idx+1]) / 2
        return poc, vah, val
    except Exception:
        return None, None, None

# ==========================================
# 🌐 الماكرو (DXY + US10Y)
# ==========================================
async def get_macro_bias():
    try:
        dxy = await fetch_data("DX-Y.NYB", "5d", "1h")
        us10y = await fetch_data("^TNX", "5d", "1h")
        if dxy is None or us10y is None: return "NEUTRAL", "بيانات غير مكتملة"
        dxy_bull = dxy['Close'].iloc[-1] > dxy['Close'].ewm(span=20).mean().iloc[-1]
        us10y_bull = us10y['Close'].iloc[-1] > us10y['Close'].ewm(span=20).mean().iloc[-1]
        if dxy_bull and us10y_bull: return "BEARISH_GOLD", "💵 الدولار والسندات صاعدان"
        elif not dxy_bull and not us10y_bull: return "BULLISH_GOLD", "✨ الدولار والسندات هابطان"
        else: return "NEUTRAL", "⚖️ تضارب ماكرو"
    except Exception:
        return "NEUTRAL", "تعذر جلب الماكرو"

# ==========================================
# 📰 درع الأخبار (ForexFactory)
# ==========================================
NEWS_CACHE = {"data": None, "fetched": None}
def is_news_dangerous():
    try:
        if NEWS_CACHE["data"] and NEWS_CACHE["fetched"] and (datetime.now() - NEWS_CACHE["fetched"]).seconds < 3600:
            news_data = NEWS_CACHE["data"]
        else:
            r = requests.get("https://nfs.faireconomy.media/ff_calendar_thisweek.json", timeout=10)
            news_data = r.json()
            NEWS_CACHE["data"] = news_data; NEWS_CACHE["fetched"] = datetime.now()
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
            diff = (news_time - now_ny).total_seconds() / 60
            if -30 <= diff <= 45: return True, f"اقتراب خبر: {title}"
        return False, None
    except Exception:
        return False, None

# ==========================================
# 🛡️ إدارة المخاطر
# ==========================================
def calc_lot(entry, sl):
    risk_usd = 1000 * (RISK_PER_TRADE / 100)  # يمكن تعديل الرصيد هنا
    risk_points = abs(entry - sl)
    if risk_points == 0: return 0.01
    lot = risk_usd / (risk_points * 100)
    return max(0.01, round(lot, 2))

def is_risk_limit_reached():
    today = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
    r = conn.execute("SELECT COALESCE(SUM(result),0) FROM trades WHERE time LIKE ? AND result < 0", (f"{today}%",)).fetchone()
    return abs(r[0] if r and r[0] else 0) >= MAX_DAILY_LOSS

# ==========================================
# 🧠 الذكاء الاصطناعي (Gemini)
# ==========================================
async def ask_ai(data_summary):
    if not ai_enabled: return "⚠️ AI غير متاح."
    try:
        prompt = f"""
        أنت خبير تداول كمي في الذهب (XAUUSD). عندك هذه المعطيات:
        {data_summary}
        بناءً على هذه المعطيات، أعطني قرارك النهائي في 3 أسطر بالدارجة المغربية:
        1. القرار: (شراء / بيع / انتظار)
        2. السبب الرئيسي
        3. نصيحة إدارة المخاطر
        """
        response = await ai_client.aio.models.generate_content(model=AI_MODEL, contents=prompt)
        return response.text
    except Exception as e:
        return f"⚠️ AI مشغول: {e}"

# ==========================================
# 🚀 المحرك الرئيسي (MTF + SMC + Volume + AI)
# ==========================================
async def run_engine():
    danger, msg = is_news_dangerous()
    if danger: return None, f"⚠️ درع الأخبار: {msg}"
    if not is_in_killzone(): return None, "💤 خارج أوقات السيولة (Killzones)."
    if is_risk_limit_reached(): return None, f"🛑 حد الخسارة اليومي ({MAX_DAILY_LOSS}%)."

    try:
        macro_bias, macro_msg = await get_macro_bias()

        df_d1 = await fetch_data("GC=F", "1y", "1d"); await asyncio.sleep(1)
        df_h4 = await fetch_data("GC=F", "3mo", "4h"); await asyncio.sleep(1)
        df_h1 = await fetch_data("GC=F", "1mo", "1h"); await asyncio.sleep(1)
        df_m15 = await fetch_data("GC=F", "5d", "15m")

        if any(d is None for d in [df_d1, df_h4, df_h1, df_m15]):
            return None, "❌ تعذر جلب بعض الفريمات."

        d1_dir, d1_ob, d1_msg = analyze_smc(df_d1, "D1")
        h4_dir, h4_ob, h4_msg = analyze_smc(df_h4, "H4")
        h1_dir, h1_ob, h1_msg = analyze_smc(df_h1, "H1")

        poc, vah, val = volume_profile(df_h1)
        vp_msg = f"POC: {poc:.2f} | VAH: {vah:.2f} | VAL: {val:.2f}" if poc else "VP: غير متاح"

        signal_dir = 0; ob_zone = None
        if d1_dir >= 0 and h4_dir == 1 and h1_dir == 1:
            signal_dir = 1; ob_zone = h1_ob
        elif d1_dir <= 0 and h4_dir == -1 and h1_dir == -1:
            signal_dir = -1; ob_zone = h1_ob

        if signal_dir == 0:
            return None, f"👑 الوحش 16.0: لا يوجد توافق MTF.\n📊 D1: {d1_msg}\n📊 H4: {h4_msg}\n📊 H1: {h1_msg}\n📈 {vp_msg}\n💵 {macro_msg}"

        if signal_dir == 1 and macro_bias == "BEARISH_GOLD":
            return None, "⚠️ تعارض: السيولة صاعدة لكن الماكرو هابط."
        if signal_dir == -1 and macro_bias == "BULLISH_GOLD":
            return None, "⚠️ تعارض: السيولة هابطة لكن الماكرو صاعد."

        current_price = df_m15['Close'].iloc[-1]
        atr_m15 = (df_m15['High'] - df_m15['Low']).rolling(14).mean().iloc[-1]
        entry = round(current_price, 2)

        if signal_dir == 1:
            sl = round(ob_zone['bottom'] - (atr_m15 * 0.8), 2)
            tp1 = round(entry + (entry - sl) * 1.5, 2)
            tp2 = round(entry + (entry - sl) * 3.0, 2)
            trade_type = "🟢 شراء كمي (Quantum Buy)"
        else:
            sl = round(ob_zone['top'] + (atr_m15 * 0.8), 2)
            tp1 = round(entry - (sl - entry) * 1.5, 2)
            tp2 = round(entry - (sl - entry) * 3.0, 2)
            trade_type = "🔴 بيع كمي (Quantum Sell)"

        lot = calc_lot(entry, sl)

        ai_summary = f"""
        - السعر: {current_price}
        - D1: {d1_msg} | H4: {h4_msg} | H1: {h1_msg}
        - Volume Profile: {vp_msg}
        - الماكرو: {macro_msg}
        - الإشارة: {trade_type}
        - الدخول: {entry} | الوقف: {sl} | الهدف: {tp1}
        """
        ai_verdict = await ask_ai(ai_summary)

        payload = {
            "time": str(df_m15.index[-1]),
            "type": trade_type,
            "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "lot": lot,
            "macro": macro_msg, "vp": vp_msg,
            "smc": f"D1: {d1_msg}\nH4: {h4_msg}\nH1: {h1_msg}",
            "ai": ai_verdict
        }
        return payload, None
    except Exception as e:
        return None, f"❌ خطأ في المحرك: {e}"

# ==========================================
# 📱 واجهة تليجرام
# ==========================================
def format_card(sig):
    return (
        f"👑 *الوحش 16.0 Quantum Edition*\n━━━━━━━━━━━━━━━━━━━━\n"
        f"{sig['type']}\n\n"
        f"🎯 *الدخول:* `{sig['entry']:.2f}`\n"
        f"🛑 *الوقف:* `{sig['sl']:.2f}`\n"
        f"🏆 *الهدف 1:* `{sig['tp1']:.2f}`\n"
        f"🚀 *الهدف 2:* `{sig['tp2']:.2f}`\n\n"
        f"📦 *اللوت:* `{sig['lot']}`\n"
        f"📊 *SMC:*\n{sig['smc']}\n"
        f"📈 *Volume Profile:* {sig['vp']}\n"
        f"💵 *الماكرو:* {sig['macro']}\n\n"
        f"🧠 *رأي الذكاء الاصطناعي:*\n{sig['ai']}\n"
        f"━━━━━━━━━━━━━━━━━━━━"
    )

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    db_set("chat_id", update.effective_chat.id)
    await update.message.reply_text("👋 مرحباً بك في الوحش 16.0!\n\nالأوامر:\n/analyze — تحليل شامل\n/status — حالة الحماية", parse_mode='Markdown')

async def analyze_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    db_set("chat_id", update.effective_chat.id)
    await update.message.reply_text("⚡ جاري التحليل الشامل (MTF + SMC + VP + AI)...", parse_mode='Markdown')
    sig, err = await run_engine()
    if err: await update.message.reply_text(err, parse_mode='Markdown')
    elif sig: await update.message.reply_text(format_card(sig), parse_mode='Markdown')

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    danger, msg = is_news_dangerous()
    macro_bias, macro_msg = await get_macro_bias()
    status = (
        f"⚙️ *حالة المنظومة:*\n━━━━━━━━━━━━━━━━━━━━\n"
        f"🕐 *Killzone:* {'✅' if is_in_killzone() else '❌'}\n"
        f"📰 *درع الأخبار:* {'⚠️ ' + msg if danger else '✅ آمن'}\n"
        f"💵 *الماكرو:* {macro_msg}\n"
        f"🛡️ *الخسارة اليومية:* {'✅' if not is_risk_limit_reached() else '🛑'}\n"
        f"━━━━━━━━━━━━━━━━━━━━"
    )
    await update.message.reply_text(status, parse_mode='Markdown')

async def auto_job(context: ContextTypes.DEFAULT_TYPE):
    try:
        chat_id = db_get("chat_id")
        if not chat_id: return
        sig, err = await run_engine()
        if sig and sig['time'] != db_get("last_signal_time"):
            db_set("last_signal_time", sig['time'])
            await context.bot.send_message(chat_id=int(chat_id), text=f"🚨 *إشارة حية!*\n\n" + format_card(sig), parse_mode='Markdown')
    except Exception as e:
        logging.error(f"Auto Job Error: {e}")

# ==========================================
# 🌐 FastAPI & Webhook & Self-Ping
# ==========================================
telegram_app = ApplicationBuilder().token(TOKEN).build()
telegram_app.add_handler(CommandHandler('start', start_cmd))
telegram_app.add_handler(CommandHandler('analyze', analyze_cmd))
telegram_app.add_handler(CommandHandler('status', status_cmd))
if telegram_app.job_queue:
    telegram_app.job_queue.run_repeating(auto_job, interval=300, first=10)

async def self_ping():
    while True:
        await asyncio.sleep(600)
        if RENDER_URL:
            try: requests.get(RENDER_URL, timeout=10)
            except: pass

@asynccontextmanager
async def lifespan(app: FastAPI):
    await telegram_app.initialize()
    if RENDER_URL:
        await telegram_app.bot.set_webhook(f"{RENDER_URL}/webhook")
    await telegram_app.start()
    asyncio.create_task(self_ping())
    logging.info("🚀 الوحش 16.0 يعمل عبر Webhook...")
    yield
    await telegram_app.stop()
    await telegram_app.shutdown()

app = FastAPI(lifespan=lifespan)

@app.post("/webhook")
async def webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, telegram_app.bot)
    await telegram_app.process_update(update)
    return Response(status_code=200)

@app.get("/")
def read_root(): return {"status": "alive", "bot": "Quantum Beast 16.0"}

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
