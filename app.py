# -*- coding: utf-8 -*-
"""
خادم الويب لمنظومة التعرف على الكتابة اليدوية العربية
=====================================================
نقطة الدخول الرئيسية للتشغيل:
    python app.py
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.api.app import app, start_server

if __name__ == '__main__':
    start_server(port=5001, debug=False)
