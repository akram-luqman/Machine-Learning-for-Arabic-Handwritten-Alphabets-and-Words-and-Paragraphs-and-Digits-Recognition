# -*- coding: utf-8 -*-
"""
ملف التشغيل الرئيسي لمنظومة التعرف على الكتابة اليدوية العربية
==============================================================
شغّل هذا الملف عبر الطرفية (Terminal) لتشغيل الخادم وواجهة الويب:
    python run_app.py
"""

import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

try:
    from src.api.app import app, start_server
    print("\n" + "=" * 70)
    print("🚀 بدء تشغيل منظومة التعرف على الكتابة اليدوية العربية...")
    print("🌐 افتح المتصفح على الرابط: http://127.0.0.1:5001")
    print("⚡ النظام يعمل بتعلم الآلة الكلاسيكي فقط (SVM, RF, KNN, HOG, Hu Moments)")
    print("=" * 70 + "\n")
    start_server(port=5001, debug=False)
except ImportError as e:
    print(f"❌ خطأ في استيراد مكتبات المشروع: {e}")
    sys.exit(1)
