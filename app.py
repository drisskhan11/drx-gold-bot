import os
import re
import math
import sqlite3
import warnings
import logging
import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx
import yfinance as yf
import pandas as pd
import numpy as np
import telegram
from openai import OpenAI
from twelvedata import TDClient

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("QuantumSovereign")

# ==========================================
# ⚙️ الإعدادات الأساسية وإدارة المخاطر
# ==========================================
TOKEN = os.environ.get("BOT_TOKEN", "").strip()
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "").strip()
TWELVEDATA_API_KEY = os.environ.get("TWELVEDATA_API_KEY", "").strip()

FORCE_TEST = os.environ.get("FORCE_TEST", "0").strip() == "1"
MIN_RR_RATIO = float(os.environ.get("MIN_RR_RATIO", "2.0"))
MAX_ENTRY_ATR_MULT = float(os.environ.get("MAX_ENTRY_ATR_MULT", "1.6"))
MIN_CONFLUENCE_SCORE = int(os.environ.get("MIN_CONFLUENCE_SCORE", "60"))

ACCOUNT_BALANCE = float(os.environ.get("ACCOUNT_BALANCE", "10000.0"))
RISK_PER_TRADE_PCT = float(os.environ.get("RISK_PER_TRADE_PCT", "1.0"))
MAX_DAILY_LOSS_PCT = float(os.environ.get("MAX_DAILY_LOSS_PCT", "3.0"))
CONTRACT_SIZE = float(os.environ.get("CONTRACT_SIZE", "100.0"))

DB_FILE = os.environ.get("DB_FILE", "quantum_beast_66.db")
NY = ZoneInfo("America/New_York")

SILVER_BULLET_WINDOWS = [
    (3, 4),   # 03:00 - 04:00 NY
    (10, 11), # 10:00 - 11:00 NY
]

try:
    from google import genai
    ai_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
except Exception as e:
    logger.warning(f"Gemini init failed: {e}")
    ai_client = None

td = TDClient(apikey=TWELVEDATA_API_KEY) if TWELVEDATA_API_KEY else None


def is_valid_num(val) -> bool:
    try:
        f = float(val)
        return math.isfinite(f) and f > 0.0
    except Exception:
        return False


# ==========================================
# 🛠️ أداة إعادة المحاولة الذكية (Retry)
# ==========================================
async def retry_async(coro_factory, retries: int = 3, base_delay: float = 0.8, label: str = "op"):
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return await coro_factory()
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                sleep_for = base_delay * (2 ** (attempt - 1))
                logger.debug(f"{label} attempt {attempt}/{retries} failed ({exc}), retrying in {sleep_for:.1f}s")
                await asyncio.sleep(sleep_for)
    raise last_exc


# ==========================================
# 🗄️ قاعدة البيانات (دورة حياة الصفقة الكاملة)
# ==========================================
conn = sqlite3.connect(DB_FILE, check_same_thread=False)
conn.executescript("""
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS state (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    direction INTEGER NOT NULL,
    entry REAL NOT NULL,
    sl REAL NOT NULL,
    tp1 REAL NOT NULL,
    tp2 REAL NOT NULL,
    lot_size REAL DEFAULT 0.01,
    risk_usd REAL DEFAULT 0.0,
    score INTEGER DEFAULT 0,
    hurst REAL,
    macro TEXT,
    status TEXT DEFAULT 'PENDING',
    updated_at TEXT,
    ai_summary TEXT
);
""")
conn.commit()


def db_get(key: str, default=None):
    row = conn.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
    return row[0] if row else default


def db_set(key: str, value):
    conn.execute(
        "INSERT INTO state(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, str(value))
    )
    conn.commit()


def save_signal(direction: int, entry: float, sl: float, tp1: float, tp2: float,
                lot_size: float, risk_usd: float, score: int, hurst: float,
                macro: str, ai_summary: str) -> int:
    now_iso = datetime.now(NY).isoformat()
    cur = conn.execute(
        """
        INSERT INTO signals(
            ts, direction, entry, sl, tp1, tp2, lot_size, risk_usd,
            score, hurst, macro, status, updated_at, ai_summary
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', ?, ?)
        """,
        (now_iso, direction, entry, sl, tp1, tp2, lot_size, risk_usd,
         score, hurst, macro, now_iso, ai_summary)
    )
    conn.commit()
    return cur.lastrowid


def get_daily_realized_loss_usd(day_str: str) -> float:
    rows = conn.execute(
        """
        SELECT entry, sl, lot_size
        FROM signals
        WHERE status = 'SL_HIT' AND substr(updated_at, 1, 10) = ?
        """,
        (day_str,)
    ).fetchall()
    total_loss = 0.0
    for entry, sl, lot in rows:
        pts_risk = abs(float(entry) - float(sl))
        lot_val = float(lot) if is_valid_num(lot) else 0.01
        total_loss += pts_risk * lot_val * CONTRACT_SIZE
    return round(total_loss, 2)


def calculate_position_size(price_distance: float):
    risk_usd = round(ACCOUNT_BALANCE * (RISK_PER_TRADE_PCT / 100.0), 2)
    if not is_valid_num(price_distance) or not is_valid_num(CONTRACT_SIZE):
        return 0.01, risk_usd
    raw_lots = risk_usd / (price_distance * CONTRACT_SIZE)
    lots = max(0.01, min(50.0, round(raw_lots, 2)))
    return lots, risk_usd


def build_daily_summary(day_str: str) -> str:
    rows = conn.execute(
        """
        SELECT direction, entry, sl, tp1, tp2, lot_size, status
        FROM signals
        WHERE substr(ts, 1, 10) = ?
        """,
        (day_str,)
    ).fetchall()

    if not rows:
        return (
            f"📒 *تقرير نهاية الجلسة ({day_str})*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"لا توجد صفقات مسجلة اليوم. انضباط وحماية رأس المال أولاً."
        )

    total = len(rows)
    tp2_cnt = sum(1 for r in rows if r[6] == "TP2_HIT")
    tp1_cnt = sum(1 for r in rows if r[6] == "TP1_HIT")
    be_cnt = sum(1 for r in rows if r[6] == "BE_CLOSED")
    sl_cnt = sum(1 for r in rows if r[6] == "SL_HIT")
    pending_cnt = sum(1 for r in rows if r[6] == "PENDING")
    active_cnt = sum(1 for r in rows if r[6] == "ACTIVE")

    wins = tp1_cnt + tp2_cnt
    closed_trades = wins + sl_cnt + be_cnt
    win_rate = (wins / closed_trades * 100.0) if closed_trades > 0 else 0.0

    est_pnl = 0.0
    for direction, entry, sl, tp1, tp2, lot, status in rows:
        d = int(direction)
        e, s, t1, t2 = float(entry), float(sl), float(tp1), float(tp2)
        l = float(lot) if is_valid_num(lot) else 0.01
        if status == "SL_HIT":
            est_pnl -= abs(e - s) * l * CONTRACT_SIZE
        elif status == "TP1_HIT":
            # نصف الصفقة على TP1 والباقي على BE
            est_pnl += abs(t1 - e) * (l * 0.5) * CONTRACT_SIZE
        elif status == "TP2_HIT":
            est_pnl += (abs(t1 - e) * 0.5 + abs(t2 - e) * 0.5) * l * CONTRACT_SIZE

    pnl_sign = "+" if est_pnl >= 0 else ""
    return (
        f"📒 *تقرير نهاية الجلسة ({day_str})*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📌 إجمالي الإشارات: `{total}` (نشطة: `{active_cnt}` | معلقة: `{pending_cnt}`)\n"
        f"🚀 ضرب TP2: `{tp2_cnt}` | 🎯 ضرب TP1: `{tp1_cnt}`\n"
        f"🛡️ أغلق BE: `{be_cnt}` | 🛑 ضرب SL: `{sl_cnt}`\n"
        f"📈 نسبة النجاح (Win Rate): `{win_rate:.1f}%`\n"
        f"💵 الربح/الخسارة التقديري: `{pnl_sign}${est_pnl:.2f}`"
    )


# ==========================================
# 📱 الاتصال بتيليجرام (إرسال آمن وموثوق)
# ==========================================
def _strip_markdown(text: str) -> str:
    return re.sub(r"[*_`]", "", text)


async def send_telegram_msg(text: str):
    if not text:
        return
    if not TOKEN or not CHAT_ID:
        logger.warning("BOT_TOKEN or TELEGRAM_CHAT_ID missing. Output:\n" + text)
        return

    bot = telegram.Bot(token=TOKEN)
    chunks = [text[i:i + 3900] for i in range(0, len(text), 3900)]
    for chunk in chunks:
        try:
            await retry_async(
                lambda c=chunk: bot.send_message(chat_id=CHAT_ID, text=c, parse_mode="Markdown"),
                retries=3,
                base_delay=0.7,
                label="telegram_md"
            )
        except Exception as e:
            logger.warning(f"Markdown send failed ({e}), fallback to plain text...")
            try:
                await retry_async(
                    lambda c=chunk: bot.send_message(chat_id=CHAT_ID, text=_strip_markdown(c)),
                    retries=2,
                    base_delay=0.5,
                    label="telegram_plain"
                )
            except Exception as e2:
                logger.error(f"Telegram failed completely: {e2}")


# ==========================================
# 🌐 جلب البيانات (متعدد المصادر + Retry)
# ==========================================
async def get_live_prices():
    # 1. XAUS API
    try:
        async def _xaus():
            async with httpx.AsyncClient(timeout=7.0) as client:
                r = await client.get("https://xaus.com/api/v1/spot?compact=1")
                r.raise_for_status()
                return r.json()

        d = await retry_async(_xaus, retries=2, base_delay=0.5, label="xaus_spot")
        if d.get("data_state", {}).get("status") == "fresh":
            gold = float(d["spot_usd_oz"])
            silver_raw = d.get("silver_usd_oz")
            silver = float(silver_raw) if is_valid_num(silver_raw) else None
            if is_valid_num(gold):
                return gold, silver
    except Exception as e:
        logger.debug(f"XAUS spot fallback: {e}")

    # 2. yfinance GC=F
    try:
        def _yf_spot():
            df = yf.Ticker("GC=F").history(period="1d", interval="1m", auto_adjust=False)
            if df is None or df.empty:
                return None
            last_val = float(df["Close"].dropna().iloc[-1])
            return last_val if is_valid_num(last_val) else None

        gold_price = await retry_async(
            lambda: asyncio.to_thread(_yf_spot),
            retries=2,
            base_delay=0.6,
            label="yf_spot"
        )
        if gold_price:
            return gold_price, None
    except Exception as e:
        logger.debug(f"yfinance spot fallback: {e}")

    return None, None


async def fetch_candles(symbol: str, interval: str, limit: int = 120):
    def _download():
        yf_symbol = "GC=F" if symbol in ("XAUUSD", "XAU/USD", "GC=F") else symbol
        period_map = {
            "1m": "2d",
            "5m": "5d",
            "15m": "10d",
            "1h": "1mo",
            "4h": "3mo",
            "1d": "6mo",
        }
        period = period_map.get(interval, "10d")
        df = yf.Ticker(yf_symbol).history(period=period, interval=interval, auto_adjust=False)
        if df is None or df.empty:
            return None
        cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        df = df[cols].dropna()
        df = df[(df["High"] > 0) & (df["Low"] > 0) & (df["Close"] > 0)]
        if df.empty:
            return None
        df.index = (df.index.tz_localize("UTC") if df.index.tz is None else df.index).tz_convert(NY)
        return df.tail(limit)

    try:
        return await retry_async(
            lambda: asyncio.to_thread(_download),
            retries=2,
            base_delay=0.6,
            label=f"candles_{symbol}_{interval}"
        )
    except Exception as e:
        logger.warning(f"fetch_candles error ({symbol}, {interval}): {e}")
        return None


async def get_previous_day_ohlc():
    today = datetime.now(NY).date()

    if td:
        try:
            def _td_daily():
                ts = td.time_series(symbol="XAU/USD", interval="1day", outputsize=10).as_pandas()
                if ts is None or ts.empty:
                    return None
                ts = ts.copy()
                ts.index = pd.to_datetime(ts.index)
                past = ts[ts.index.date < today].sort_index()
                if past.empty:
                    return None
                prev = past.iloc[-1]
                h, l, o, c = float(prev["high"]), float(prev["low"]), float(prev["open"]), float(prev["close"])
                if not all(is_valid_num(x) for x in (h, l, o, c)) or h < l:
                    return None
                return {"open": o, "high": h, "low": l, "close": c, "source": "TwelveData"}

            res = await retry_async(
                lambda: asyncio.to_thread(_td_daily),
                retries=2,
                base_delay=0.6,
                label="td_daily"
            )
            if res:
                return res
        except Exception as e:
            logger.warning(f"Twelve Data failed: {e}")

    df_d = await fetch_candles("GC=F", "1d", limit=15)
    if df_d is not None and not df_d.empty:
        past = df_d[df_d.index.date < today]
        if not past.empty:
            prev = past.iloc[-1]
            h, l, o, c = float(prev["High"]), float(prev["Low"]), float(prev["Open"]), float(prev["Close"])
            if all(is_valid_num(x) for x in (h, l, o, c)) and h >= l:
                return {"open": o, "high": h, "low": l, "close": c, "source": "yfinance"}

    return None


# ==========================================
# 📊 إدارة ومتابعة دورة حياة الصفقات
# ==========================================
async def monitor_signals(current_price: float | None):
    if not is_valid_num(current_price):
        return []

    rows = conn.execute(
        """
        SELECT id, direction, entry, sl, tp1, tp2, status
        FROM signals
        WHERE status IN ('PENDING', 'ACTIVE', 'TP1_HIT')
        ORDER BY id ASC
        """
    ).fetchall()

    if not rows:
        return []

    alerts = []
    now_iso = datetime.now(NY).isoformat()

    for sig_id, direction, entry, sl, tp1, tp2, status in rows:
        d = int(direction)
        e, s, t1, t2 = float(entry), float(sl), float(tp1), float(tp2)
        new_status = None
        msg = None

        if d == 1:  # شراء
            if status == "PENDING":
                if current_price <= e:
                    new_status = "ACTIVE"
                    msg = f"⚡ *تفعيل صفقة الشراء #{sig_id}* عند `{current_price:.2f}` (Entry: `{e:.2f}` | SL: `{s:.2f}` | TP1: `{t1:.2f}`)."
                elif current_price <= s:
                    new_status = "CANCELLED_SL"
                    msg = f"🗑️ *إلغاء الأمر المعلق #{sig_id} (شراء)*: السعر كسر الوقف `{s:.2f}` قبل التفعيل."
            elif status == "ACTIVE":
                if current_price <= s:
                    new_status = "SL_HIT"
                    msg = f"🛑 *الصفقة #{sig_id} (شراء)* ضربت الوقف عند `{current_price:.2f}` (SL: `{s:.2f}`)."
                elif current_price >= t1:
                    new_status = "TP1_HIT"
                    msg = (
                        f"🎯 *الصفقة #{sig_id} (شراء)* ضربت *الهدف 1* عند `{current_price:.2f}` (TP1: `{t1:.2f}`)!\n"
                        f"🛡️ انقل الوقف إلى Break-Even (`{e:.2f}`) واستهدف TP2 (`{t2:.2f}`)."
                    )
            elif status == "TP1_HIT":
                if current_price >= t2:
                    new_status = "TP2_HIT"
                    msg = f"🚀 *الصفقة #{sig_id} (شراء)* حققت *الهدف 2 النهائي* عند `{current_price:.2f}` (TP2: `{t2:.2f}`)!"
                elif current_price <= e:
                    new_status = "BE_CLOSED"
                    msg = f"🛡️ *الصفقة #{sig_id} (شراء)* أُغلقت على *Break-Even* بعد تأمين الهدف 1."

        elif d == -1:  # بيع
            if status == "PENDING":
                if current_price >= e:
                    new_status = "ACTIVE"
                    msg = f"⚡ *تفعيل صفقة البيع #{sig_id}* عند `{current_price:.2f}` (Entry: `{e:.2f}` | SL: `{s:.2f}` | TP1: `{t1:.2f}`)."
                elif current_price >= s:
                    new_status = "CANCELLED_SL"
                    msg = f"🗑️ *إلغاء الأمر المعلق #{sig_id} (بيع)*: السعر تجاوز الوقف `{s:.2f}` قبل التفعيل."
            elif status == "ACTIVE":
                if current_price >= s:
                    new_status = "SL_HIT"
                    msg = f"🛑 *الصفقة #{sig_id} (بيع)* ضربت الوقف عند `{current_price:.2f}` (SL: `{s:.2f}`)."
                elif current_price <= t1:
                    new_status = "TP1_HIT"
                    msg = (
                        f"🎯 *الصفقة #{sig_id} (بيع)* ضربت *الهدف 1* عند `{current_price:.2f}` (TP1: `{t1:.2f}`)!\n"
                        f"🛡️ انقل الوقف إلى Break-Even (`{e:.2f}`) واستهدف TP2 (`{t2:.2f}`)."
                    )
            elif status == "TP1_HIT":
                if current_price <= t2:
                    new_status = "TP2_HIT"
                    msg = f"🚀 *الصفقة #{sig_id} (بيع)* حققت *الهدف 2 النهائي* عند `{current_price:.2f}` (TP2: `{t2:.2f}`)!"
                elif current_price >= e:
                    new_status = "BE_CLOSED"
                    msg = f"🛡️ *الصفقة #{sig_id} (بيع)* أُغلقت على *Break-Even* بعد تأمين الهدف 1."

        if new_status and msg:
            conn.execute(
                "UPDATE signals SET status=?, updated_at=? WHERE id=?",
                (new_status, now_iso, sig_id)
            )
            conn.commit()
            alerts.append(msg)

    return alerts


# ==========================================
# ⏰ فلاتر الوقت والأخبار
# ==========================================
def is_silver_bullet_time(now: datetime | None = None) -> bool:
    if FORCE_TEST:
        return True
    now = now or datetime.now(NY)
    if now.weekday() >= 5:
        return False
    t = now.hour + now.minute / 60.0
    return any(start <= t < end for start, end in SILVER_BULLET_WINDOWS)


def should_send_hourly_report(now: datetime) -> bool:
    if now.minute > 4:
        return False
    slot_key = now.strftime("%Y-%m-%d_%H")
    if db_get("last_hourly_slot") == slot_key:
        return False
    db_set("last_hourly_slot", slot_key)
    return True


def should_send_eod_report(now: datetime) -> bool:
    # إرسال ملخص نهاية الجلسة عند الساعة 17:00-17:05 بتوقيت نيويورك أيام العمل
    if now.weekday() >= 5:
        return False
    if now.hour == 17 and now.minute <= 5:
        day_key = now.strftime("%Y-%m-%d")
        if db_get("last_eod_report") != day_key:
            db_set("last_eod_report", day_key)
            return True
    return False


async def check_news_blackout():
    if FORCE_TEST:
        return False, None
    try:
        async def _news():
            async with httpx.AsyncClient(timeout=9.0) as client:
                r = await client.get("https://nfs.faireconomy.media/ff_calendar_thisweek.json")
                r.raise_for_status()
                return r.json()

        events = await retry_async(_news, retries=2, base_delay=0.5, label="ff_calendar")
        now = datetime.now(NY)
        critical = ["cpi", "fomc", "non-farm", "nfp", "ppi", "gdp", "powell", "core pce"]
        for n in events:
            if n.get("country") == "USD" and n.get("impact") == "High":
                title = (n.get("title") or "").strip()
                if any(k in title.lower() for k in critical):
                    try:
                        n_time = pd.to_datetime(n["date"], utc=True).tz_convert(NY)
                        diff_mins = (n_time - now).total_seconds() / 60.0
                        if -30.0 <= diff_mins <= 45.0:
                            return True, f"⏸️ حظر أخبار عالية التأثير: {title} ({diff_mins:+.0f}m)"
                    except Exception:
                        continue
        return False, None
    except Exception as e:
        logger.debug(f"News check error: {e}")
        return False, None


# ==========================================
# 💀 كشف الفخ (Judas Swing) + CHoCH + FVG + التقييم
# ==========================================
def detect_fvg(df: pd.DataFrame, direction: int, lookback: int = 8):
    if df is None or len(df) < 4:
        return None

    start_idx = max(2, len(df) - lookback)
    for i in range(len(df) - 1, start_idx - 1, -1):
        if direction == 1:
            low_curr = float(df["Low"].iloc[i])
            high_prev2 = float(df["High"].iloc[i - 2])
            if low_curr > high_prev2:
                return (high_prev2, low_curr, low_curr - high_prev2)
        elif direction == -1:
            high_curr = float(df["High"].iloc[i])
            low_prev2 = float(df["Low"].iloc[i - 2])
            if high_curr < low_prev2:
                return (high_curr, low_prev2, low_prev2 - high_curr)
    return None


def get_htf_bias(df_h1: pd.DataFrame) -> str:
    if df_h1 is None or len(df_h1) < 55:
        return "NEUTRAL"
    close = df_h1["Close"]
    ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
    ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
    last_close = close.iloc[-1]
    if last_close > ema20 > ema50:
        return "BULLISH"
    if last_close < ema20 < ema50:
        return "BEARISH"
    return "NEUTRAL"


def detect_judas_swing(df_m15: pd.DataFrame, df_m1: pd.DataFrame, pdh: float, pdl: float):
    if df_m15 is None or df_m1 is None or len(df_m15) < 6 or len(df_m1) < 12:
        return None, None, None, "بيانات غير كافية"

    recent_m15 = df_m15.tail(6)
    swept_high = float(recent_m15["High"].max()) > pdh
    swept_low = float(recent_m15["Low"].min()) < pdl

    last_m1 = df_m1.tail(12)
    last_close = float(last_m1["Close"].iloc[-1])

    if swept_high and not swept_low:
        swing_low = float(last_m1["Low"].iloc[-6:-1].min())
        if last_close < swing_low:
            fvg = detect_fvg(df_m1, -1, lookback=8)
            if fvg:
                entry = round((fvg[0] + fvg[1]) / 2.0, 2)
                return -1, entry, fvg, f"🔴 فخ فوق PDH ({pdh:.2f}) + CHoCH هبوطي + FVG"
            fallback_entry = round(float(last_m1["High"].iloc[-4:-1].max()), 2)
            return -1, fallback_entry, None, f"🔴 فخ فوق PDH ({pdh:.2f}) + CHoCH هبوطي"

    if swept_low and not swept_high:
        swing_high = float(last_m1["High"].iloc[-6:-1].max())
        if last_close > swing_high:
            fvg = detect_fvg(df_m1, 1, lookback=8)
            if fvg:
                entry = round((fvg[0] + fvg[1]) / 2.0, 2)
                return 1, entry, fvg, f"🟢 فخ تحت PDL ({pdl:.2f}) + CHoCH صعودي + FVG"
            fallback_entry = round(float(last_m1["Low"].iloc[-4:-1].min()), 2)
            return 1, fallback_entry, None, f"🟢 فخ تحت PDL ({pdl:.2f}) + CHoCH صعودي"

    return None, None, None, "في الانتظار (لا يوجد Sweep + CHoCH مؤكد)"


def compute_confluence_score(direction: int, fvg, htf_bias: str, macro: str, hurst: float) -> int:
    score = 45  # أساس تحقق Sweep + CHoCH
    if fvg is not None:
        score += 15
    if direction == 1 and htf_bias == "BULLISH":
        score += 15
    elif direction == -1 and htf_bias == "BEARISH":
        score += 15
    elif htf_bias != "NEUTRAL":
        score -= 10

    if direction == 1 and "BULLISH_GOLD" in macro:
        score += 15
    elif direction == -1 and "BEARISH_GOLD" in macro:
        score += 15
    elif "GOLD" in macro:
        score -= 10

    if hurst >= 0.55:
        score += 10
    elif hurst < 0.45:
        score -= 5

    return int(max(0, min(100, score)))


# ==========================================
# 🧠 مؤشرات إضافية والماكرو
# ==========================================
def calculate_hurst(prices, max_lag: int = 20) -> float:
    try:
        arr = np.asarray(prices, dtype=float)
        arr = arr[arr > 0]
        if len(arr) < 30:
            return 0.5
        s = np.log(arr)
        upper = min(max_lag, len(s) // 3)
        if upper < 8:
            return 0.5
        lags = np.arange(2, upper)
        tau = np.array([np.std(s[l:] - s[:-l]) for l in lags])
        mask = tau > 0
        if mask.sum() < 6:
            return 0.5
        slope = np.polyfit(np.log(lags[mask]), np.log(tau[mask]), 1)[0]
        return round(float(np.clip(slope, 0.0, 1.0)), 3)
    except Exception:
        return 0.5


def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    high = df["High"]
    low = df["Low"]
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    return float(atr) if is_valid_num(atr) else float((high - low).tail(period).mean())


async def get_macro_and_intermarket():
    dxy, vix, silver, spx = await asyncio.gather(
        fetch_candles("DX-Y.NYB", "1h", 50),
        fetch_candles("^VIX", "1h", 50),
        fetch_candles("SI=F", "1h", 50),
        fetch_candles("^GSPC", "1h", 50),
    )

    macro = "NEUTRAL"
    if dxy is not None and len(dxy) >= 20:
        dxy_last = float(dxy["Close"].iloc[-1])
        dxy_ema20 = float(dxy["Close"].ewm(span=20, adjust=False).mean().iloc[-1])
        if dxy_last > dxy_ema20 * 1.0005:
            macro = "BEARISH_GOLD (DXY صاعد)"
        elif dxy_last < dxy_ema20 * 0.9995:
            macro = "BULLISH_GOLD (DXY هابط)"

    vix_msg = "غير متاح"
    if vix is not None and not vix.empty:
        vix_val = float(vix["Close"].iloc[-1])
        regime = "هدوء" if vix_val < 16 else ("تذبذب متوسط" if vix_val < 24 else "خوف مرتفع")
        vix_msg = f"{vix_val:.1f} ({regime})"

    intermarket_parts = []
    if silver is not None and not silver.empty:
        intermarket_parts.append(f"🥈 فضة: {float(silver['Close'].iloc[-1]):.2f}")
    if spx is not None and not spx.empty:
        intermarket_parts.append(f"📉 S&P: {float(spx['Close'].iloc[-1]):.1f}")

    intermarket = " | ".join(intermarket_parts) if intermarket_parts else "بيانات غير متاحة"
    return macro, vix_msg, intermarket


# ==========================================
# 🧠 مجلس العباقرة (AI Council متعدد المزودين)
# ==========================================
async def ask_ai_council(data_summary: str) -> str:
    prompt = f"""أنت مجلس خبراء تداول ذهب (XAUUSD / ICT / Risk Management).
حلل المعطيات التالية بدقة وموضوعية:
{data_summary}

أجب بالضبط بهذا التنسيق المختصر:
🧠 المحلل: [حجة فنية واحدة واضحة]
😈 محامي الشيطان: [أقوى خطر أو سبب قد يفشل الصفقة]
⚖️ مدير المخاطر: القرار: [شراء/بيع/انتظار] | السبب: [سبب مختصر]"""

    if ai_client:
        try:
            response = await retry_async(
                lambda: ai_client.aio.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                ),
                retries=2,
                base_delay=0.6,
                label="ai_gemini"
            )
            if response and response.text:
                return response.text.strip()
        except Exception as e:
            logger.warning(f"Gemini AI failed: {e}")

    if GROQ_API_KEY:
        try:
            def _groq():
                client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)
                resp = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                )
                return resp.choices[0].message.content

            out = await retry_async(lambda: asyncio.to_thread(_groq), retries=2, base_delay=0.6, label="ai_groq")
            if out:
                return out.strip()
        except Exception as e:
            logger.warning(f"Groq AI failed: {e}")

    if OPENROUTER_API_KEY:
        try:
            def _or():
                client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
                resp = client.chat.completions.create(
                    model="meta-llama/llama-3.3-70b-instruct:free",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                )
                return resp.choices[0].message.content

            out = await retry_async(lambda: asyncio.to_thread(_or), retries=2, base_delay=0.6, label="ai_openrouter")
            if out:
                return out.strip()
        except Exception as e:
            logger.warning(f"OpenRouter AI failed: {e}")

    if MISTRAL_API_KEY:
        try:
            def _mistral():
                client = OpenAI(base_url="https://api.mistral.ai/v1", api_key=MISTRAL_API_KEY)
                resp = client.chat.completions.create(
                    model="mistral-small-latest",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                )
                return resp.choices[0].message.content

            out = await retry_async(lambda: asyncio.to_thread(_mistral), retries=2, base_delay=0.6, label="ai_mistral")
            if out:
                return out.strip()
        except Exception as e:
            logger.warning(f"Mistral AI failed: {e}")

    return (
        "🧠 المحلل: اعتماد إشارة الهيكل السعري والسيولة.\n"
        "😈 محامي الشيطان: تعذر الوصول لنماذج الذكاء الاصطناعي حالياً.\n"
        "⚖️ مدير المخاطر: القرار: انتظار | السبب: غياب مراجعة AI"
    )


# ==========================================
# 🚀 المحرك الرئيسي (Market Sovereign)
# ==========================================
async def run_engine():
    now = datetime.now(NY)
    today_str = now.strftime("%Y-%m-%d")

    current_price, silver_price = await get_live_prices()

    # 1. متابعة الصفقات المعلقة والنشطة
    trade_alerts = await monitor_signals(current_price)
    for alert_msg in trade_alerts:
        await send_telegram_msg(alert_msg)

    # 2. ملخص نهاية الجلسة إذا حان وقته
    if should_send_eod_report(now):
        await send_telegram_msg(build_daily_summary(today_str))

    # 3. فحص سقف الخسارة اليومية
    max_daily_loss_usd = round(ACCOUNT_BALANCE * (MAX_DAILY_LOSS_PCT / 100.0), 2)
    daily_loss_usd = get_daily_realized_loss_usd(today_str)
    daily_dd_breached = daily_loss_usd >= max_daily_loss_usd and not FORCE_TEST

    in_silver_bullet = is_silver_bullet_time(now)

    # 4. وضع الاستخبارات خارج نافذة Silver Bullet
    if not in_silver_bullet and not FORCE_TEST:
        if should_send_hourly_report(now):
            macro, vix, intermarket = await get_macro_and_intermarket()
            report = (
                f"👀 *مراقبة السوق (خارج نافذة Silver Bullet)*\n"
                f"🕐 الوقت: `{now.strftime('%H:%M')}` (نيويورك)\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
            )
            if is_valid_num(current_price):
                report += f"💰 الذهب الفوري: `{current_price:.2f}`\n"
            if is_valid_num(silver_price):
                report += f"🥈 الفضة الفورية: `{silver_price:.2f}`\n"
            report += (
                f"💵 الماكرو: {macro}\n"
                f"🌪️ VIX: {vix}\n"
                f"🌍 الأسواق: {intermarket}\n"
                f"🛡️ *خسائر اليوم:* `${daily_loss_usd:.2f}` / `${max_daily_loss_usd:.2f}`\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"🧐 *الوضع:* نترقب نوافذ السيولة (03:00-04:00 و 10:00-11:00 NY)."
            )
            return report, None
        return None, None

    if daily_dd_breached:
        logger.warning(f"Daily drawdown limit reached: ${daily_loss_usd:.2f} >= ${max_daily_loss_usd:.2f}")
        return None, None

    # 5. فحص الأخبار عالية التأثير
    news_blocked, news_msg = await check_news_blackout()
    if news_blocked:
        logger.info(news_msg)
        return None, None

    # 6. جلب قمم/قيعان اليوم السابق والشموع
    ohlc, df_m15, df_m1, df_h1 = await asyncio.gather(
        get_previous_day_ohlc(),
        fetch_candles("GC=F", "15m", limit=120),
        fetch_candles("GC=F", "1m", limit=120),
        fetch_candles("GC=F", "1h", limit=120),
    )

    if not ohlc:
        return None, "❌ تعذر جلب بيانات اليوم السابق (PDH/PDL)"

    if df_m15 is None or df_m1 is None or len(df_m15) < 20 or len(df_m1) < 15:
        return None, "❌ بيانات غير كافية للتحليل (M15/M1)"

    pdh, pdl = ohlc["high"], ohlc["low"]
    htf_bias = get_htf_bias(df_h1)
    direction, entry_zone, fvg, sweep_msg = detect_judas_swing(df_m15, df_m1, pdh, pdl)

    if direction is None:
        if FORCE_TEST:
            test_msg = (
                f"✅ *الوحش 66.0 جاهز وحاكم للمسار* (FORCE_TEST)\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"🕐 الوقت: `{now.strftime('%H:%M')}` (نيويورك)\n"
            )
            if is_valid_num(current_price):
                test_msg += f"💰 السعر الحالي: `{current_price:.2f}`\n"
            test_msg += (
                f"\n📊 *المعطيات الحالية:*\n"
                f"- PDH: `{pdh:.2f}` | PDL: `{pdl:.2f}` ({ohlc.get('source')})\n"
                f"- اتجاه H1: `{htf_bias}`\n"
                f"- حالة الكشف: {sweep_msg}\n"
                f"- عتبة الجودة المطلوبة: `{MIN_CONFLUENCE_SCORE}/100`\n"
                f"- الرصيد: `${ACCOUNT_BALANCE:.2f}` | المخاطرة: `{RISK_PER_TRADE_PCT:.1f}%`\n"
            )
            return test_msg, None
        return None, None

    # 7. التحقق من السعر، ATR، والجودة
    if not is_valid_num(current_price):
        current_price = float(df_m1["Close"].iloc[-1])

    atr_m15 = calculate_atr(df_m15, period=14)
    if not is_valid_num(atr_m15):
        return None, "❌ قيمة ATR غير صالحة"

    if abs(current_price - entry_zone) > (atr_m15 * MAX_ENTRY_ATR_MULT):
        logger.info(f"Skipped: price {current_price:.2f} too far from entry {entry_zone:.2f}")
        return None, None

    hurst = calculate_hurst(df_m15["Close"].values)
    macro, vix, intermarket = await get_macro_and_intermarket()
    score = compute_confluence_score(direction, fvg, htf_bias, macro, hurst)

    if score < MIN_CONFLUENCE_SCORE and not FORCE_TEST:
        logger.info(f"Skipped setup: confluence score {score} < {MIN_CONFLUENCE_SCORE}")
        return None, None

    if direction == 1:
        swing_low_m1 = float(df_m1["Low"].tail(8).min())
        sl = round(min(swing_low_m1, entry_zone - atr_m15 * 0.35) - (atr_m15 * 0.15), 2)
        risk = round(entry_zone - sl, 2)
        if not is_valid_num(risk):
            return None, "❌ خطأ في حساب وقف الشراء"
        tp1 = round(entry_zone + risk * MIN_RR_RATIO, 2)
        tp2 = round(entry_zone + risk * (MIN_RR_RATIO * 2.0), 2)
        trade_type = "🟢 شراء (Limit Buy)"
        expected_word = "شراء"
    else:
        swing_high_m1 = float(df_m1["High"].tail(8).max())
        sl = round(max(swing_high_m1, entry_zone + atr_m15 * 0.35) + (atr_m15 * 0.15), 2)
        risk = round(sl - entry_zone, 2)
        if not is_valid_num(risk):
            return None, "❌ خطأ في حساب وقف البيع"
        tp1 = round(entry_zone - risk * MIN_RR_RATIO, 2)
        tp2 = round(entry_zone - risk * (MIN_RR_RATIO * 2.0), 2)
        trade_type = "🔴 بيع (Limit Sell)"
        expected_word = "بيع"

    lot_size, risk_usd = calculate_position_size(risk)
    fvg_text = f"`{fvg[0]:.2f} - {fvg[1]:.2f}`" if fvg else "غير متوفر"

    ai_summary = (
        f"Current Price: {current_price:.2f}\n"
        f"Setup: {sweep_msg}\n"
        f"Confluence Score: {score}/100\n"
        f"Proposed Action: {expected_word}\n"
        f"Entry Zone: {entry_zone:.2f} | FVG: {fvg_text}\n"
        f"SL: {sl:.2f} | TP1: {tp1:.2f} | TP2: {tp2:.2f} (Risk Pts: {risk:.2f}, Lot: {lot_size})\n"
        f"HTF Bias (H1): {htf_bias} | Hurst (M15): {hurst}\n"
        f"Macro: {macro} | VIX: {vix} | Intermarket: {intermarket}"
    )

    ai_verdict = await ask_ai_council(ai_summary)

    if "انتظار" in ai_verdict:
        logger.info("AI Council voted for WAIT.")
        return None, None
    if direction == 1 and "بيع" in ai_verdict and "شراء" not in ai_verdict:
        logger.info("AI Council suggested SELL instead of BUY.")
        return None, None
    if direction == -1 and "شراء" in ai_verdict and "بيع" not in ai_verdict:
        logger.info("AI Council suggested BUY instead of SELL.")
        return None, None

    signal_hash = f"{today_str}_{direction}_{entry_zone}_{sl}"
    if db_get("last_signal") == signal_hash:
        return None, None
    db_set("last_signal", signal_hash)

    sig_id = save_signal(
        direction, entry_zone, sl, tp1, tp2,
        lot_size, risk_usd, score, hurst, macro, ai_verdict
    )

    report = (
        f"👑 *الوحش 66.0 (حاكم المارشي) — إشارة #{sig_id}*\n"
        f"🕐 الوقت: `{now.strftime('%H:%M')}` (نيويورك) | *الجودة: `{score}/100`*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 السعر الحالي: `{current_price:.2f}`\n"
        f"💀 *الإعداد:* {sweep_msg}\n"
        f"📦 *منطقة FVG:* {fvg_text}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"{trade_type} *(PENDING حتى لمس الدخول)*\n"
        f"🎯 *الدخول:* `{entry_zone:.2f}`\n"
        f"🛑 *الوقف (SL):* `{sl:.2f}` (`-{risk:.2f}`)\n"
        f"🏆 *الهدف 1 (1:{MIN_RR_RATIO:g}):* `{tp1:.2f}`\n"
        f"🚀 *الهدف 2 (1:{MIN_RR_RATIO * 2:g}):* `{tp2:.2f}`\n"
        f"📐 *اللوت:* `{lot_size:.2f}` (مخاطرة `${risk_usd:.2f}`)\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🧭 *اتجاه H1:* `{htf_bias}` | *Hurst:* `{hurst}`\n"
        f"📊 *الماكرو:* {macro}\n"
        f"🌪️ *VIX:* {vix}\n"
        f"🌍 *الأسواق:* {intermarket}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🧠 *مجلس العباقرة:*\n{ai_verdict}"
    )
    return report, None


async def main():
    try:
        signal, err = await run_engine()
        if err:
            logger.error(err)
            if "بيانات غير كافية" not in err and "ATR" not in err:
                await send_telegram_msg(err)
        elif signal:
            await send_telegram_msg(signal)
    except Exception as e:
        logger.exception(f"Unexpected fatal error: {e}")
        await send_telegram_msg(f"❌ خطأ غير متوقع في المحرك: `{e}`")
    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
