import yfinance as yf
import pandas as pd
import numpy as np

def run_backtest():
    print("🚀 بدء الاختبار التاريخي (استراتيجية SMC الحقيقية)...")
    try:
        ticker = yf.Ticker("GC=F")
        # جلب بيانات الساعة لسنتين
        df = ticker.history(period="2y", interval="1h")
        
        if df.empty:
            print("❌ تعذر جلب البيانات.")
            return

        wins = 0
        losses = 0
        timeouts = 0
        total_trades = 0
        total_profit = 0.0

        # حساب ATR
        df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
        
        # البحث عن ضربات السيولة (Liquidity Sweeps)
        for i in range(20, len(df) - 24): # كنقلبو على 24 ساعة بعد الدخول
            current_close = df['Close'].iloc[i]
            current_low = df['Low'].iloc[i]
            current_high = df['High'].iloc[i]
            current_open = df['Open'].iloc[i]
            atr = df['ATR'].iloc[i]
            
            if pd.isna(atr): continue
            
            # الشموع اللي قبل (آخر 10 شموع)
            past_lows = df['Low'].iloc[i-10:i]
            past_highs = df['High'].iloc[i-10:i]
            
            recent_low = past_lows.min()
            recent_high = past_highs.max()
            
            # 🟢 شراء: Liquidity Sweep للقاع + شمعة صاعدة
            if current_low < recent_low and current_close > recent_low and current_close > current_open:
                entry = current_close
                sl = current_low - (atr * 0.5)
                tp = entry + (atr * 2.0) # هدف ضعف الوقف
                
                for j in range(i+1, i+24): # كنعطيوه 24 ساعة باش يحقق
                    if df['High'].iloc[j] >= tp:
                        wins += 1
                        total_profit += (tp - entry)
                        break
                    elif df['Low'].iloc[j] <= sl:
                        losses += 1
                        total_profit += (sl - entry)
                        break
                else:
                    timeouts += 1
                    total_profit += (df['Close'].iloc[i+23] - entry)
                
                total_trades += 1
                continue
                
            # 🔴 بيع: Liquidity Sweep للقمة + شمعة هابطة
            elif current_high > recent_high and current_close < recent_high and current_close < current_open:
                entry = current_close
                sl = current_high + (atr * 0.5)
                tp = entry - (atr * 2.0)
                
                for j in range(i+1, i+24):
                    if df['Low'].iloc[j] <= tp:
                        wins += 1
                        total_profit += (entry - tp)
                        break
                    elif df['High'].iloc[j] >= sl:
                        losses += 1
                        total_profit += (entry - sl)
                        break
                else:
                    timeouts += 1
                    total_profit += (entry - df['Close'].iloc[i+23])
                
                total_trades += 1

        if total_trades > 0:
            win_rate = (wins / total_trades) * 100
            print(f"\n📊 *نتائج اختبار SMC الحقيقية:*")
            print(f"إجمالي الصفقات: {total_trades}")
            print(f"الصفقات الرابحة: {wins}")
            print(f"الصفقات الخاسرة: {losses}")
            print(f"الصفقات المعلقة (Timeout): {timeouts}")
            print(f"نسبة النجاح (Win Rate): {win_rate:.2f}%")
            print(f"إجمالي الربح/الخسارة: {total_profit:.2f} نقطة")
        else:
            print("⚠️ ما كاينش صفقات كافية.")

    except Exception as e:
        print(f"❌ خطأ: {e}")

if __name__ == '__main__':
    run_backtest()
