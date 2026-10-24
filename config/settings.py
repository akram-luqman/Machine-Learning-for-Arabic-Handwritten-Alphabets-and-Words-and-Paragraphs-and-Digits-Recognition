# -*- coding: utf-8 -*-
"""
إعدادات المشروع المركزية
========================
يجمع هذا الملف جميع الإعدادات القابلة للضبط في مكان واحد،
بما في ذلك المسارات، والحدود الأمنية، ومعلمات المعالجة والتدريب.
"""

from pathlib import Path
import os

# ============================================================
# المسارات الأساسية - تحدد تلقائيًا بناءً على موقع هذا الملف
# ============================================================
_CONFIG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _CONFIG_DIR.parent
WORKSPACE_ROOT = PROJECT_ROOT

# مسار مجلد البيانات الخام
DATASETS_DIR = PROJECT_ROOT / "datasets"

# مسارات البيانات الداخلية
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
ARCHIVES_DIR = DATA_DIR / "archives"
EXTRACTED_DIR = DATA_DIR / "extracted"
MANIFESTS_DIR = DATA_DIR / "manifests"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
EXTERNAL_DIR = DATA_DIR / "external"

# مسارات النماذج والمخرجات
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = PROJECT_ROOT / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# ============================================================
# الحدود الأمنية لفك ضغط الأرشيفات
# ============================================================
ARCHIVE_SECURITY = {
    "max_archive_size_mb": 2048,        # الحد الأقصى لحجم الأرشيف بالميغابايت
    "max_extracted_size_mb": 8192,       # الحد الأقصى للحجم بعد الفك
    "max_files_per_archive": 500_000,   # الحد الأقصى لعدد الملفات
    "max_nesting_depth": 10,            # الحد الأقصى لمستويات التداخل
    "max_single_file_mb": 512,          # الحد الأقصى لحجم ملف واحد
    "blocked_extensions": {".exe", ".bat", ".cmd", ".ps1", ".sh", ".dll", ".so", ".py", ".js"},
    "allow_overwrite": False,           # منع الكتابة فوق الملفات
}

# ============================================================
# إعدادات رفع الصور عبر واجهة الويب
# ============================================================
UPLOAD_SETTINGS = {
    "max_file_size_mb": 16,
    "max_image_dimension": 4096,
    "min_image_dimension": 8,
    "allowed_extensions": {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"},
    "allowed_mime_types": {"image/png", "image/jpeg", "image/bmp", "image/tiff", "image/webp"},
    "upload_dir": PROJECT_ROOT / "data" / "uploads",
    "temp_dir": PROJECT_ROOT / "data" / "temp",
}

# ============================================================
# إعدادات معالجة الصور
# ============================================================
IMAGE_PROCESSING = {
    "target_size": (64, 64),              # الحجم المستهدف للحروف المنفصلة
    "digit_target_size": (28, 28),        # الحجم المستهدف للأرقام
    "word_target_height": 64,             # الارتفاع المستهدف للكلمات
    "max_word_width": 512,                # الحد الأقصى لعرض صورة الكلمة
    "binarization_method": "adaptive",    # adaptive أو otsu
    "adaptive_block_size": 11,            # حجم النافذة للعتبة التكيفية
    "adaptive_c": 7,                      # ثابت العتبة التكيفية
    "noise_kernel_size": 3,               # حجم نواة إزالة الضوضاء
    "preserve_dots": True,                # المحافظة على نقاط الحروف
    "padding_ratio": 0.1,                 # نسبة الحشو حول الحرف
}

# ============================================================
# إعدادات استخراج الميزات
# ============================================================
FEATURE_SETTINGS = {
    "hog": {
        "orientations": 9,
        "pixels_per_cell": (8, 8),
        "cells_per_block": (2, 2),
        "block_norm": "L2-Hys",
    },
    "pixel_density": {
        "grid_rows": 4,
        "grid_cols": 4,
    },
    "projections": {
        "normalize": True,
    },
    "hu_moments": {
        "use_log": True,
    },
    "connected_components": {
        "connectivity": 8,
    },
    "cache_features": True,
    "cache_dir": PROCESSED_DIR / "feature_cache",
    "feature_version": "v1.0",
}

# ============================================================
# إعدادات تدريب النماذج
# ============================================================
TRAINING_SETTINGS = {
    "test_split_ratio": 0.15,            # نسبة بيانات الاختبار
    "validation_split_ratio": 0.15,      # نسبة بيانات التحقق
    "random_seed": 42,
    "max_training_time_minutes": 120,    # الحد الأقصى لوقت التدريب
    "max_memory_gb": 4,                  # الحد الأقصى للذاكرة
    "n_jobs": -1,                        # عدد الأنوية (-1 = جميعها)
    "cv_folds": 5,                       # عدد طيات التحقق المتقاطع
    "scoring_metric": "f1_weighted",     # المقياس الأساسي
    "batch_size": 5000,                  # حجم الدفعة للمعالجة
}

# ============================================================
# قائمة الحروف العربية ومعلوماتها
# ============================================================
ARABIC_LETTERS = {
    "alif": "ا", "beh": "ب", "teh": "ت", "theh": "ث",
    "jeem": "ج", "hah": "ح", "khah": "خ", "dal": "د",
    "thal": "ذ", "raa": "ر", "zay": "ز", "seen": "س",
    "sheen": "ش", "sad": "ص", "dad": "ض", "tah": "ط",
    "zah": "ظ", "ain": "ع", "ghain": "غ", "feh": "ف",
    "qaf": "ق", "kaf": "ك", "lam": "ل", "meem": "م",
    "noon": "ن", "heh": "ه", "waw": "و", "yaa": "ي",
    "hamza": "ء", "alif_hamza": "أ", "alif_hamza_below": "إ",
    "alif_madda": "آ", "teh_marbuta": "ة", "alif_maqsura": "ى",
    "lam_alif": "لا",
}

# الأشكال السياقية للحروف العربية
CONTEXTUAL_FORMS = ["regular", "begin", "middle", "end"]

# الأرقام العربية الهندية
ARABIC_HINDI_DIGITS = {
    0: "٠", 1: "١", 2: "٢", 3: "٣", 4: "٤",
    5: "٥", 6: "٦", 7: "٧", 8: "٨", 9: "٩",
}

# الأرقام الغربية
WESTERN_DIGITS = {i: str(i) for i in range(10)}

# ============================================================
# المكتبات المحظورة - يمنع استيرادها
# ============================================================
BANNED_LIBRARIES = [
    "tensorflow", "keras", "torch", "pytorch",
    "transformers", "paddleocr", "easyocr",
    "caffe", "mxnet", "theano", "chainer",
]

BANNED_SKLEARN_CLASSES = [
    "MLPClassifier", "MLPRegressor",
]

# ============================================================
# إعدادات واجهة الويب
# ============================================================
WEB_SETTINGS = {
    "host": "127.0.0.1",
    "port": 5001,
    "debug": False,
    "max_content_length_mb": 16,
    "request_timeout_seconds": 60,
}

# ============================================================
# دوال مساعدة للإعدادات
# ============================================================
def ensure_dirs():
    """إنشاء جميع المجلدات المطلوبة إذا لم تكن موجودة"""
    dirs = [
        RAW_DIR, ARCHIVES_DIR, EXTRACTED_DIR, MANIFESTS_DIR,
        INTERIM_DIR, PROCESSED_DIR, SPLITS_DIR, EXTERNAL_DIR,
        MODELS_DIR, OUTPUTS_DIR, REPORTS_DIR, LOGS_DIR,
        MODELS_DIR / "character", MODELS_DIR / "word", MODELS_DIR / "text",
        OUTPUTS_DIR / "data_quality", OUTPUTS_DIR / "model_comparison",
        OUTPUTS_DIR / "predictions", OUTPUTS_DIR / "feature_importance",
        OUTPUTS_DIR / "diagnostics",
        REPORTS_DIR / "datasets", REPORTS_DIR / "evaluation", REPORTS_DIR / "errors",
        UPLOAD_SETTINGS["upload_dir"], UPLOAD_SETTINGS["temp_dir"],
        FEATURE_SETTINGS["cache_dir"],
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def get_datasets_search_paths():
    """إرجاع قائمة المسارات التي يمكن البحث فيها عن مجموعات البيانات"""
    paths = []
    if DATASETS_DIR.exists():
        paths.append(DATASETS_DIR)
    if RAW_DIR.exists():
        paths.append(RAW_DIR)
    if EXTERNAL_DIR.exists():
        paths.append(EXTERNAL_DIR)
    return paths
