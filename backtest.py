import yfinance as yf
import pandas as pd
import numpy as np

def run_backtest():
    print("🚀 بدء الاختبار التاريخي (Quantum Liquidity Engine)...")
    try:
        ticker = yf.Ticker("GC=F")
        df = ticker.history(period="2y", interval="1h")
        
        if df.empty:
            print("❌ تعذر جلب البيانات.")
            return

        lookback = 20
        wins = 0
        losses = 0
        breakevens = 0
        total_trades = 0
        profit = 0.0

        df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()

        for i in range(lookback, len(df) - 50):
            recent_low = df['Low'].iloc[i-lookback:i].min()
            recent_high = df['High'].iloc[i-lookback:i].max()
            
            sweep_low = df['Low'].iloc[i] < recent_low
            sweep_high = df['High'].iloc[i] > recent_high
            
            # 🟢 صيد سيولة القاع + إزاحة صاعدة
            if sweep_low and df['Close'].iloc[i] > recent_low: 
                body = abs(df['Close'].iloc[i] - df['Open'].iloc[i])
                # شرط الإزاحة: الشمعة تكون قوية (أكبر من ATR بـ 1.2)
                if body > df['ATR'].iloc[i] * 1.2: 
                    entry = df['Close'].iloc[i]
                    sl = df['Low'].iloc[i] - (df['ATR'].iloc[i] * 0.2)
                    risk = entry - sl
                    tp = entry + (risk * 3.0) # هدف 1:3
                    be = entry + (risk * 1.0) # بريك إيفن 1:1
                    
                    hit_be = False
                    for j in range(i+1, i+50):
                        if df['High'].iloc[j] >= be and not hit_be:
                            hit_be = True
                            sl = entry # تحريك الوقف
                        if df['High'].iloc[j] >= tp:
                            wins += 1
                            profit += (tp - entry)
                            break
                        elif df['Low'].iloc[j] <= sl:
                            if hit_be:
                                breakevens += 1
                            else:
                                losses += 1
                                profit += (sl - entry)
                            break
                    total_trades += 1

            # 🔴 صيد سيولة القمة + إزاحة هابطة
            elif sweep_high and df['Close'].iloc[i] < recent_high:
                body = abs(df['Close'].iloc[i] - df['Open'].iloc[i])
                if body > df['ATR'].iloc[i] * 1.2:
                    entry = df['Close'].iloc[i]
                    sl = df['High'].iloc[i] + (df['ATR'].iloc[i] * 0.2)
                    risk = sl - entry
                    tp = entry - (risk * 3.0)
                    be = entry - (risk * 1.0)
                    
                    hit_be = False
                    for j in range(i+1, i+50):
                        if df['Low'].iloc[j] <= be and not hit_be:
                            hit_be = True
                            sl = entry
                        if df['Low'].iloc[j] <= tp:
                            wins += 1
                            profit += (entry - tp)
                            break
                        elif df['High'].iloc[j] >= sl:
                            if hit_be:
                                breakevens += 1
                            else:
                                losses += 1
                                profit += (entry - sl)
                            break
                    total_trades += 1

        if total_trades > 0:
            win_rate = (wins / total_trades) * 100
            print(f"\n📊 *نتائج محرك السيولة الكمي:*")
            print(f"إجمالي الصفقات: {total_trades}")
            print(f"الصفقات الرابحة (1:3): {wins}")
            print(f"الصفقات الخاسرة: {losses}")
            print(f"الصفقات المحمية (Breakeven): {breakevens}")
            print(f"نسبة النجاح (Win Rate): {win_rate:.2f}%")
            print(f"إجمالي الأرباح (بالنقاط): {profit:.2f} نقطة")
        else:
            print("⚠️ ما كاينش صفقات كافية.")

    except Exception as e:
        print(f"❌ خطأ: {e}")

if __name__ == '__main__':
    run_backtest()
