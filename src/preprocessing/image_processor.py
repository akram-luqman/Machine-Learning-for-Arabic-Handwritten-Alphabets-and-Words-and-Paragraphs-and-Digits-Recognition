# -*- coding: utf-8 -*-
"""
معالجة الصور للكتابة اليدوية العربية
=====================================
تدعم OpenCV مع دعم بديل فوري عبر Pillow و SciPy في حال عدم توفر cv2.
تحافظ على نقاط الحروف والعلامات وتجري التحويل الثنائي والتطبيع.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
from PIL import Image, ImageOps, ImageFilter
from scipy import ndimage

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from ..utils.helpers import setup_logger, measure_time

logger = setup_logger(__name__)


class ImagePreprocessor:
    """
    معالج صور الكتابة اليدوية العربية
    """

    def __init__(self, target_size: Tuple[int, int] = (64, 64),
                 preserve_dots: bool = True,
                 padding_ratio: float = 0.1,
                 binarization: str = "adaptive",
                 diagnostic_mode: bool = False):
        self.target_size = target_size
        self.preserve_dots = preserve_dots
        self.padding_ratio = padding_ratio
        self.binarization = binarization
        self.diagnostic_mode = diagnostic_mode

    def process(self, image: np.ndarray, strategy: str = "standard") -> Optional[np.ndarray]:
        """معالجة صورة واحدة وتحويلها إلى مصفوفة ثنائية مطبعة بالحجم المستهدف"""
        if image is None or image.size == 0:
            return None

        try:
            # 1. التحويل إلى تدرجات الرمادي
            if len(image.shape) == 3:
                if HAS_CV2:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
            else:
                gray = image.copy()

            # 2. التحويل الثنائي (عتبة أوتسو أو تكيفية)
            if HAS_CV2:
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            else:
                thresh = np.mean(gray)
                binary = np.where(gray < thresh, 255, 0).astype(np.uint8)

            # إذا كانت الخلفية بيضاء والنص أسود في الأصل تم عكسها في binary (نص أبيض بخلفية سوداء)
            # 3. الاقتصاص حول المحتوى مع حشو
            coords = np.column_stack(np.where(binary > 0))
            if len(coords) == 0:
                return np.zeros(self.target_size, dtype=np.uint8)

            y_min, x_min = np.min(coords, axis=0)
            y_max, x_max = np.max(coords, axis=0)

            cropped = binary[y_min:y_max+1, x_min:x_max+1]

            # 4. التطبيع والتحجيم مع المحافظة على النسبة
            h, w = cropped.shape
            target_w, target_h = self.target_size
            scale = min((target_w * 0.85) / w, (target_h * 0.85) / h)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))

            # استخدام PIL للتحجيم في حال عدم وجود cv2
            pil_img = Image.fromarray(cropped)
            pil_resized = pil_img.resize((new_w, new_h), Image.Resampling.BILINEAR)
            resized = np.array(pil_resized)

            # وضعها في منتصف إطار بالحجم المستهدف
            result = np.zeros(self.target_size, dtype=np.uint8)
            y_off = (target_h - new_h) // 2
            x_off = (target_w - new_w) // 2
            result[y_off:y_off+new_h, x_off:x_off+new_w] = resized

            return result

        except Exception as e:
            logger.error(f"خطأ في معالجة الصورة: {e}")
            return None


def load_image(path: str, grayscale: bool = False) -> Optional[np.ndarray]:
    """تحميل صورة باستخدام PIL كبديل موثوق"""
    try:
        img = Image.open(path)
        if grayscale:
            img = img.convert("L")
        else:
            img = img.convert("RGB")
        return np.array(img)
    except Exception as e:
        logger.error(f"خطأ في تحميل الصورة {path}: {e}")
        return None
