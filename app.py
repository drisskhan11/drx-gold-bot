import os
import requests
import asyncio

print("🚀 TEST: السكريبت بدا كيخدم!")

async def main():
    print("🕐 دخلنا لدالة main...")
    
    TOKEN = os.environ.get("BOT_TOKEN", "").strip()
    CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

    if not TOKEN or not CHAT_ID:
        print("❌ المفاتيح ناقصة فـ GitHub Secrets")
        return

    print("📤 كنحاول نصيفط رسالة لتيليجرام...")
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": "🧪 هادي رسالة تجريبية من GitHub Actions! إلا وصلاتك، فالاتصال خدام 100%."}
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        print(f"✅ جواب تيليجرام: {response.status_code}")
    except Exception as e:
        print(f"❌ فشل الإرسال: {e}")

    print("🏁 الاختبار سالا.")

if __name__ == '__main__':
    asyncio.run(main())
