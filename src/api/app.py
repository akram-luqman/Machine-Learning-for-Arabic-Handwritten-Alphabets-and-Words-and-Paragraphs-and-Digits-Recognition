# -*- coding: utf-8 -*-
"""
خادم واجهة الويب والتطبيقات (Flask API Server)
==============================================
يوفر واجهة تفاعلية ونقاط نهاية RESTful:
- /: الصفحة الرئيسية مع لوحة الرسم ورفع الصور
- /api/recognize_canvas: التعرف على رسم لوحة Canvas
- /api/recognize_upload: التعرف على صورة مرفوعة
- /api/datasets_info: إحصائيات مجموعات البيانات المكتشفة
- /api/models_status: حالة النماذج المحمّلة
- /api/health: فحص سلامة الخادم
"""

import os
import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory

import numpy as np

# إضافة المسارات إلى sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.recognition.engine import ArabicRecognitionEngine
from src.utils.helpers import setup_logger

logger = setup_logger(__name__)

# تهيئة تطبيق Flask
template_dir = PROJECT_ROOT / "templates"
static_dir = PROJECT_ROOT / "static"

app = Flask(
    __name__,
    template_folder=str(template_dir),
    static_folder=str(static_dir)
)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

# تهيئة محرك التعرف الشامل
engine = ArabicRecognitionEngine()


@app.route('/')
def index():
    """الصفحة الرئيسية لمنصة التعرف على الخط العربي"""
    return render_template('index.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    """فحص سلامة وجاهزية المنصة"""
    return jsonify({
        "status": "healthy",
        "service": "Arabic Handwriting Recognition Intelligence Platform",
        "version": "2.0.0",
        "ml_paradigm": "Traditional Machine Learning Only (No Neural Networks)",
        "models": engine.models_status,
        "features": ["HOG", "Pixel Grid Density", "Projections", "Hu Moments", "Connected Components"],
        "supported_modes": ["digit", "character", "word", "paragraph", "auto"]
    })


@app.route('/api/models_status', methods=['GET'])
def models_status():
    """حالة النماذج المحمّلة والدقة المحققة"""
    return jsonify(engine.models_status)


@app.route('/api/recognize_canvas', methods=['POST'])
def recognize_canvas():
    """نقطة نهاية للتعرف على ما رسمه المستخدم في لوحة الرسم Canvas"""
    data = request.get_json(force=True, silent=True)
    if not data or "image" not in data:
        return jsonify({"success": False, "error": "لم يتم إرسال بيانات الصورة"}), 400

    mode = data.get("mode", "auto")

    img_np = engine.decode_base64_image(data["image"])
    if img_np is None:
        return jsonify({"success": False, "error": "فشل فك ترميز صورة الرسم"}), 400

    result = engine.recognize(img_np, mode=mode)
    return jsonify(result)


@app.route('/api/recognize_upload', methods=['POST'])
def recognize_upload():
    """نقطة نهاية للتعرف على ملف صورة تم رفعه"""
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "لم يتم اختيار أي ملف"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "اسم الملف فارغ"}), 400

    mode = request.form.get("mode", "auto")

    try:
        from PIL import Image
        import io
        image = Image.open(io.BytesIO(file.read())).convert("L")
        img_np = np.array(image)
        if img_np is None or img_np.size == 0:
            return jsonify({"success": False, "error": "تعذر قراءة ملف الصورة"}), 400

        result = engine.recognize(img_np, mode=mode)
        return jsonify(result)
    except Exception as e:
        logger.error(f"خطأ في معالجة الملف المرفوع: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/datasets_info', methods=['GET'])
def datasets_info():
    """إرجاع ملخص مجموعات البيانات المتاحة"""
    datasets_dir = PROJECT_ROOT / "datasets"
    info = {
        "datasets_found": datasets_dir.exists(),
        "categories": [
            {"name": "AHCD (قاعدة بيانات الحروف)", "type": "حروف منفصلة (28 حرف)", "samples": "16,800 عينة (13,440 تدريب + 3,360 اختبار)", "image_size": "32×32"},
            {"name": "AHDD (قاعدة بيانات الأرقام)", "type": "أرقام هندية (0-9)", "samples": "70,000 عينة (60,000 تدريب + 10,000 اختبار)", "image_size": "28×28"},
            {"name": "AHDBase & MAHDBase", "type": "أرقام بصيغة BMP متعددة الكتّاب", "samples": "آلاف النماذج", "image_size": "متنوعة"},
            {"name": "Arabic Alphabets, Words & Paragraphs", "type": "حروف (65 سياقاً) + كلمات + فقرات", "writers": "82 كاتباً", "image_size": "متنوعة"}
        ],
        "models": engine.models_status
    }
    return jsonify(info)


def start_server(port=5001, debug=False):
    """تشغيل الخادم المحلي"""
    logger.info(f"═══════════════════════════════════════════════════")
    logger.info(f"🖋️ بدء تشغيل منصة التعرف على الخط العربي")
    logger.info(f"📍 الرابط: http://127.0.0.1:{port}")
    logger.info(f"═══════════════════════════════════════════════════")
    status = engine.models_status
    logger.info(f"📊 حالة النماذج: {status}")
    app.run(host='127.0.0.1', port=port, debug=debug)


if __name__ == '__main__':
    start_server()
