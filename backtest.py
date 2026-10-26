import yfinance as yf
import pandas as pd
import numpy as np

def run_backtest():
    print("🚀 بدء الاختبار التاريخي (صياد البنوك PDH/PDL)...")
    try:
        ticker = yf.Ticker("GC=F")
        df = ticker.history(period="2y", interval="1h")
        
        if df.empty:
            print("❌ تعذر جلب البيانات.")
            return

        df.index = df.index.tz_convert('America/New_York')

        wins = 0
        losses = 0
        breakevens = 0
        total_trades = 0
        profit = 0.0

        # حساب قمة وقاع اليوم اللي فات
        df['Date'] = df.index.date
        daily_highs = df.groupby('Date')['High'].max()
        daily_lows = df.groupby('Date')['Low'].min()
        
        # تأخير القمم والقيعان باش نستعملو بيانات اليوم اللي فات
        prev_high = daily_highs.shift(1)
        prev_low = daily_lows.shift(1)

        df['Prev_High'] = df['Date'].map(prev_high)
        df['Prev_Low'] = df['Date'].map(prev_low)
        df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()

        for i in range(20, len(df) - 50):
            current_time = df.index[i]
            hour = current_time.hour
            
            # فلتر السيولة (لندن ونيويورك فقط)
            if not ((2 <= hour < 5) or (7 <= hour < 11)):
                continue

            current_close = df['Close'].iloc[i]
            current_high = df['High'].iloc[i]
            current_low = df['Low'].iloc[i]
            
            pdh = df['Prev_High'].iloc[i]
            pdl = df['Prev_Low'].iloc[i]
            atr = df['ATR'].iloc[i]
            
            if pd.isna(pdh) or pd.isna(pdl) or pd.isna(atr): continue

            # 🔴 صيد سيولة القمة (Short)
            if current_high > pdh and current_close < pdh:
                entry = current_close
                sl = current_high + (atr * 0.2)
                risk = sl - entry
                tp = entry - (risk * 3.0) # هدف 1:3
                be = entry - (risk * 1.0) # بريك إيفن 1:1
                
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

            # 🟢 صيد سيولة القاع (Long)
            elif current_low < pdl and current_close > pdl:
                entry = current_close
                sl = current_low - (atr * 0.2)
                risk = entry - sl
                tp = entry + (risk * 3.0)
                be = entry + (risk * 1.0)
                
                hit_be = False
                for j in range(i+1, i+50):
                    if df['High'].iloc[j] >= be and not hit_be:
                        hit_be = True
                        sl = entry
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

        if total_trades > 0:
            win_rate = (wins / total_trades) * 100
            print(f"\n📊 *نتائج صياد البنوك:*")
            print(f"إجمالي الصفقات: {total_trades}")
            print(f"الصفقات الرابحة (1:3): {wins}")
            print(f"الصفقات الخاسرة: {losses}")
            print(f"الصفقات المحمية (Breakeven): {breakevens}")
            print(f"نسبة النجاح: {win_rate:.2f}%")
            print(f"إجمالي الأرباح (بالنقاط): {profit:.2f} نقطة")
        else:
            print("⚠️ ما كاينش صفقات كافية.")

    except Exception as e:
        print(f"❌ خطأ: {e}")

if __name__ == '__main__':
    run_backtest()
