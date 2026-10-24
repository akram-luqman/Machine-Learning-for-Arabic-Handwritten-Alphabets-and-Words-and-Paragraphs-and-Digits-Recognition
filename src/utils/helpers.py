# -*- coding: utf-8 -*-
"""
أدوات مساعدة عامة للمشروع
==========================
يحتوي على دوال التسجيل والتحقق وقياس الأداء
"""

import logging
import time
import hashlib
import sys
from pathlib import Path
from functools import wraps
from typing import Optional, Union


def setup_logger(name: str, log_file: Optional[Path] = None,
                 level: int = logging.INFO) -> logging.Logger:
    """
    إنشاء مسجل أحداث مع دعم الطباعة والملف

    المدخلات:
        name: اسم المسجل
        log_file: مسار ملف السجل (اختياري)
        level: مستوى التسجيل

    المخرجات:
        كائن Logger مهيأ
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # تجنب إضافة معالجات متكررة
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # معالج الطرفية
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # معالج الملف
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def measure_time(func):
    """مزخرف لقياس وتسجيل زمن تنفيذ الدالة"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger = logging.getLogger(func.__module__)
        logger.info(f"⏱ {func.__name__}: {elapsed:.2f} ثانية")
        return result
    return wrapper


def compute_file_hash(file_path: Union[str, Path], algorithm: str = "sha256") -> str:
    """
    حساب بصمة ملف للتحقق من سلامته

    المدخلات:
        file_path: مسار الملف
        algorithm: خوارزمية التجزئة (sha256 افتراضيًا)

    المخرجات:
        البصمة كنص سداسي عشري
    """
    h = hashlib.new(algorithm)
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"الملف غير موجود: {file_path}")

    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def compute_image_hash(image_array, algorithm: str = "md5") -> str:
    """حساب بصمة صورة من مصفوفة NumPy لاكتشاف التكرار"""
    h = hashlib.new(algorithm)
    h.update(image_array.tobytes())
    return h.hexdigest()


def safe_filename(filename: str) -> str:
    """
    تنظيف اسم الملف لمنع اجتياز المسارات والأحرف الخطرة

    المدخلات:
        filename: اسم الملف الأصلي

    المخرجات:
        اسم ملف آمن
    """
    # إزالة أي مسار
    name = Path(filename).name

    # استبدال الأحرف الخطرة
    dangerous_chars = ['..', '/', '\\', '\0', '\n', '\r', ':', '*', '?', '"', '<', '>', '|']
    for char in dangerous_chars:
        name = name.replace(char, '_')

    # التأكد من وجود اسم صالح
    if not name or name.startswith('.'):
        name = f"unnamed_{hashlib.md5(filename.encode()).hexdigest()[:8]}"

    return name


def validate_path_safety(path: Path, base_dir: Path) -> bool:
    """
    التحقق من أن المسار يقع داخل المجلد الأساسي (منع Zip Slip)

    المدخلات:
        path: المسار المراد التحقق منه
        base_dir: المجلد الأساسي

    المخرجات:
        True إذا كان المسار آمنًا
    """
    try:
        resolved = path.resolve()
        base_resolved = base_dir.resolve()
        return str(resolved).startswith(str(base_resolved))
    except (OSError, ValueError):
        return False


def format_size(size_bytes: int) -> str:
    """تحويل حجم الملف بالبايت إلى صيغة مقروءة"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if abs(size_bytes) < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def is_image_file(file_path: Union[str, Path]) -> bool:
    """التحقق مما إذا كان الملف صورة بناءً على الامتداد"""
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}
    return Path(file_path).suffix.lower() in image_extensions


def validate_image_content(file_path: Union[str, Path]) -> bool:
    """
    التحقق من محتوى ملف الصورة (لا يعتمد على الامتداد وحده)

    المخرجات:
        True إذا كان الملف صورة صالحة
    """
    file_path = Path(file_path)
    if not file_path.exists() or file_path.stat().st_size < 8:
        return False

    # قراءة البايتات السحرية
    try:
        with open(file_path, 'rb') as f:
            header = f.read(16)

        magic_numbers = {
            b'\x89PNG': 'png',
            b'\xff\xd8\xff': 'jpeg',
            b'BM': 'bmp',
            b'II': 'tiff_le',
            b'MM': 'tiff_be',
            b'RIFF': 'webp',
        }

        for magic, _ in magic_numbers.items():
            if header.startswith(magic):
                return True

        return False
    except (IOError, OSError):
        return False
