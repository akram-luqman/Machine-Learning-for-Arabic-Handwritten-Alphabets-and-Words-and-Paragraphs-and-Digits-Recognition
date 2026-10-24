# -*- coding: utf-8 -*-
"""
محرك استخراج الميزات للكتابة اليدوية العربية
============================================
يدعم استخراج الميزات التقليدية بالكامل:
- HOG (مدرج تكراري للتدرجات الموجهة عبر Sobel و Polar)
- كثافة شبكة البكسلات (Grid Pixel Density)
- المساقط الأفقية والعمودية (Horizontal & Vertical Projections)
- عزوم الصورة الإحصائية (Moments & Invariants)
- تحليل المكونات المتصلة ونقاط الحروف (Connected Components via SciPy/OpenCV)
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from scipy import ndimage
from PIL import Image

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from ..utils.helpers import setup_logger

logger = setup_logger(__name__)


class ArabicFeatureExtractor:
    """
    مستخرج الميزات المتكامل للكتابة العربية التقليدية
    """

    def __init__(self, target_size: Tuple[int, int] = (64, 64),
                 hog_orientations: int = 9):
        self.target_size = target_size
        self.hog_orientations = hog_orientations

    def extract_features(self, binary_image: np.ndarray) -> np.ndarray:
        """
        استخراج متجه ميزات مدمج شامل للصورة الثنائية
        """
        if binary_image is None or binary_image.size == 0:
            return np.zeros(self.feature_dimension, dtype=np.float32)

        # التأكد من الحجم المستهدف
        if binary_image.shape[:2] != self.target_size:
            pil_img = Image.fromarray(binary_image)
            pil_img = pil_img.resize(self.target_size, Image.Resampling.BILINEAR)
            img = np.array(pil_img)
        else:
            img = binary_image.copy()

        # 1. ميزات التدرجات الموجهة (HOG-like gradients)
        hog_feat = self._extract_gradients(img)

        # 2. ميزات كثافة الشبكة (4x4 = 16 خلية)
        density_feat = self._extract_grid_density(img, grid_size=(4, 4))

        # 3. ميزات المساقط (16 قيمة أفقية + 16 رأسية)
        proj_feat = self._extract_projections(img)

        # 4. عزوم الصورة الإحصائية
        moments_feat = self._extract_moments(img)

        # 5. ميزات المكونات المتصلة ونقاط الحروف
        comp_feat = self._extract_component_features(img)

        # دمج جميع الميزات في متجه موحد
        feature_vector = np.concatenate([
            hog_feat,
            density_feat,
            proj_feat,
            moments_feat,
            comp_feat
        ]).astype(np.float32)

        return feature_vector

    def _extract_gradients(self, img: np.ndarray) -> np.ndarray:
        """استخراج تدرجات Sobel والمدرج التكراري للزوايا"""
        # حساب التدرجات باستخدام مرشحات Sobel عبر SciPy
        gx = ndimage.sobel(img.astype(np.float32), axis=1)
        gy = ndimage.sobel(img.astype(np.float32), axis=0)

        mag = np.hypot(gx, gy)
        angles = np.rad2deg(np.arctan2(gy, gx)) % 180  # 0 to 180

        # تقسيم إلى خلايا 4x4 واستخراج هيستوجرام لكل خلية
        h, w = img.shape
        cell_h, cell_w = h // 4, w // 4
        hist_features = []

        for r in range(4):
            for c in range(4):
                cell_mag = mag[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
                cell_ang = angles[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
                hist, _ = np.histogram(
                    cell_ang, bins=self.hog_orientations, range=(0, 180), weights=cell_mag
                )
                # تطبيع L2
                norm = np.linalg.norm(hist) + 1e-6
                hist_features.extend(hist / norm)

        return np.array(hist_features, dtype=np.float32)

    def _extract_grid_density(self, img: np.ndarray, grid_size: Tuple[int, int] = (4, 4)) -> np.ndarray:
        """حساب نسبة الحبر الأسود في كل خلية من الشبكة"""
        rows, cols = grid_size
        h, w = img.shape
        cell_h = h // rows
        cell_w = w // cols
        densities = []

        for r in range(rows):
            for c in range(cols):
                cell = img[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
                density = np.count_nonzero(cell) / float(cell.size) if cell.size > 0 else 0.0
                densities.append(density)

        return np.array(densities, dtype=np.float32)

    def _extract_projections(self, img: np.ndarray) -> np.ndarray:
        """حساب المساقط الأفقية والعمودية"""
        h_proj = np.sum(img > 0, axis=1) / float(img.shape[1])
        v_proj = np.sum(img > 0, axis=0) / float(img.shape[0])

        # تقليص المساقط لأبعاد ثابتة (16 قيمة) عبر الاستيفاء الخطي
        x_orig_h = np.linspace(0, 1, len(h_proj))
        x_target = np.linspace(0, 1, 16)
        h_resampled = np.interp(x_target, x_orig_h, h_proj)

        x_orig_v = np.linspace(0, 1, len(v_proj))
        v_resampled = np.interp(x_target, x_orig_v, v_proj)

        return np.concatenate([h_resampled, v_resampled])

    def _extract_moments(self, img: np.ndarray) -> np.ndarray:
        """حساب عزوم هو أو عزوم مركزية إحصائية"""
        total = np.sum(img)
        if total == 0:
            return np.zeros(7, dtype=np.float32)

        # حساب مركز الثقل
        y_indices, x_indices = np.indices(img.shape)
        cx = np.sum(x_indices * img) / total
        cy = np.sum(y_indices * img) / total

        # العزوم المركزية
        mu20 = np.sum(((x_indices - cx) ** 2) * img) / total
        mu02 = np.sum(((y_indices - cy) ** 2) * img) / total
        mu11 = np.sum((x_indices - cx) * (y_indices - cy) * img) / total
        mu30 = np.sum(((x_indices - cx) ** 3) * img) / total
        mu03 = np.sum(((y_indices - cy) ** 3) * img) / total

        moments = np.array([cx / img.shape[1], cy / img.shape[0], mu20, mu02, mu11, mu30, mu03], dtype=np.float32)
        return moments

    def _extract_component_features(self, img: np.ndarray) -> np.ndarray:
        """ميزات المكونات المتصلة (عدد الأجسام والنقاط ونسب المحاور) عبر SciPy"""
        labeled, num_features = ndimage.label(img > 0)

        # فصل الجسم الرئيسي عن النقاط الإضافية
        if num_features > 0:
            sizes = ndimage.sum(img > 0, labeled, range(1, num_features + 1))
            main_area = np.max(sizes) if len(sizes) > 0 else 0
            dots_count = sum(1 for s in sizes if s < (main_area * 0.25) and s > 2)
        else:
            main_area = 0
            dots_count = 0

        aspect_ratio = float(img.shape[1]) / float(img.shape[0])
        return np.array([num_features, dots_count, aspect_ratio, main_area / (img.size + 1)], dtype=np.float32)

    @property
    def feature_dimension(self) -> int:
        """الأبعاد الإجمالية للمتجه"""
        dummy = np.zeros(self.target_size, dtype=np.uint8)
        return len(self.extract_features(dummy))
