import yfinance as yf
import pandas as pd
import numpy as np

def run_backtest():
    print("🚀 بدء الاختبار التاريخي (2023-2024)...")
    try:
        # جلب بيانات السنة الماضية
        ticker = yf.Ticker("GC=F")
        df = ticker.history(period="1y", interval="1d")
        
        if df.empty:
            print("❌ تعذر جلب البيانات.")
            return

        wins = 0
        losses = 0
        total_trades = 0
        total_profit = 0.0

        # استراتيجية بسيطة للاختبار: SMC + ATR
        for i in range(50, len(df)-1):
            current_price = df['Close'].iloc[i]
            atr = (df['High'] - df['Low']).iloc[i-14:i].mean()
            
            # شروط شراء
            if df['Close'].iloc[i] > df['Close'].iloc[i-20:i].mean(): # ترند صاعد
                entry = current_price
                sl = entry - (atr * 1.5)
                tp = entry + (atr * 3.0)
                
                # محاكاة الصفقة فـ الشمعة الجاية
                next_high = df['High'].iloc[i+1]
                next_low = df['Low'].iloc[i+1]
                
                if next_high >= tp:
                    wins += 1
                    total_profit += (tp - entry)
                elif next_low <= sl:
                    losses += 1
                    total_profit += (sl - entry)
                else:
                    # ما ضربش لا هدف ولا وقف، كنحسبوها خسارة صغيرة
                    losses += 1
                    total_profit += (df['Close'].iloc[i+1] - entry)
                    
                total_trades += 1

        if total_trades > 0:
            win_rate = (wins / total_trades) * 100
            print(f"\n📊 *نتائج الاختبار:*")
            print(f"إجمالي الصفقات: {total_trades}")
            print(f"الصفقات الرابحة: {wins}")
            print(f"الصفقات الخاسرة: {losses}")
            print(f"نسبة النجاح (Win Rate): {win_rate:.2f}%")
            print(f"إجمالي الربح/الخسارة: {total_profit:.2f} نقطة")
        else:
            print("⚠️ ما كاينش صفقات كافية.")

    except Exception as e:
        print(f"❌ خطأ: {e}")

if __name__ == '__main__':
    run_backtest()
