# -*- coding: utf-8 -*-
"""
محرك التعرف على الكتابة اليدوية العربية (النسخة الشاملة)
=========================================================
يدعم التعرف على:
  - الأرقام العربية/الهندية (AHDD: 28×28، 784 بكسل، 10 فئات)
  - الحروف العربية المنفصلة (AHCD: 32×32، 1024 بكسل، 28 فئة)
  - الكلمات (عبر تقسيم الصورة إلى حروف فردية)
  - الفقرات (عبر تقسيم سطري ثم حرفي)
يعمل بسلاسة تامة باستخدام PIL و SciPy و NumPy.
"""

import base64
import io
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage
import joblib

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from ..utils.helpers import setup_logger

logger = setup_logger(__name__)

# ═══════════════════════════════════════════════════════════════════
# خريطة تحويل أرقام الفئات إلى حروف عربية حقيقية (AHCD)
# ترقيم الفئات من 1 إلى 28 حسب ترتيب الحروف في AHCD
# ═══════════════════════════════════════════════════════════════════
AHCD_LABEL_TO_ARABIC = {
    1: "ا", 2: "ب", 3: "ت", 4: "ث", 5: "ج", 6: "ح", 7: "خ",
    8: "د", 9: "ذ", 10: "ر", 11: "ز", 12: "س", 13: "ش", 14: "ص",
    15: "ض", 16: "ط", 17: "ظ", 18: "ع", 19: "غ", 20: "ف", 21: "ق",
    22: "ك", 23: "ل", 24: "م", 25: "ن", 26: "ه", 27: "و", 28: "ي"
}

# خريطة الأرقام الهندية/العربية (AHDD: فئات 0-9)
ARABIC_DIGIT_MAP = {
    0: "٠", 1: "١", 2: "٢", 3: "٣", 4: "٤",
    5: "٥", 6: "٦", 7: "٧", 8: "٨", 9: "٩"
}

# أسماء الكلمات المعزولة في مجموعة البيانات
WORD_NAMES = {
    "abjadiyah": "أبجدية", "azan": "أذان", "fasayakfeekahum": "فسيكفيكهم",
    "ghaleez": "غليظ", "ghazaal": "غزال", "mehras": "محراث",
    "mustadhafeen": "مستضعفين", "qashtah": "قشطة",
    "sakhar": "صخر", "shateerah": "شطيرة"
}


class ArabicRecognitionEngine:
    """
    محرك الاستدلال والتعرف الشامل — يدعم الأرقام والحروف والكلمات والفقرات
    """

    def __init__(self):
        self.digit_model = None
        self.char_model = None
        self.digit_scaler = None
        self.char_scaler = None
        self.digit_pca = None
        self.char_pca = None
        self.digit_classes = None
        self.char_classes = None
        self.digit_accuracy = 0.0
        self.char_accuracy = 0.0

        root = Path(__file__).resolve().parent.parent.parent

        # تحميل نموذج الأرقام
        digit_path = root / "models" / "digit" / "best_model.joblib"
        if digit_path.exists():
            try:
                payload = joblib.load(digit_path)
                self.digit_model = payload["classifier"]
                self.digit_scaler = payload.get("scaler")
                self.digit_pca = payload.get("pca")
                self.digit_classes = payload["classes"]
                self.digit_accuracy = payload.get("accuracy", 0)
                logger.info(f"✅ تم تحميل نموذج الأرقام (دقة: {self.digit_accuracy*100:.1f}%)")
            except Exception as e:
                logger.warning(f"⚠️ تعذر تحميل نموذج الأرقام: {e}")

        # تحميل نموذج الحروف
        char_path = root / "models" / "character" / "best_model.joblib"
        if char_path.exists():
            try:
                payload = joblib.load(char_path)
                self.char_model = payload["classifier"]
                self.char_scaler = payload.get("scaler")
                self.char_pca = payload.get("pca")
                self.char_classes = payload["classes"]
                self.char_accuracy = payload.get("accuracy", 0)
                logger.info(f"✅ تم تحميل نموذج الحروف (دقة: {self.char_accuracy*100:.1f}%)")
            except Exception as e:
                logger.warning(f"⚠️ تعذر تحميل نموذج الحروف: {e}")

    @property
    def models_status(self) -> Dict[str, Any]:
        """حالة النماذج المحملة"""
        return {
            "digit_model_loaded": self.digit_model is not None,
            "digit_accuracy": f"{self.digit_accuracy*100:.2f}%" if self.digit_model else "غير متوفر",
            "char_model_loaded": self.char_model is not None,
            "char_accuracy": f"{self.char_accuracy*100:.2f}%" if self.char_model else "غير متوفر",
        }

    def decode_base64_image(self, base64_str: str) -> Optional[np.ndarray]:
        """فك ترميز صورة Base64 من لوحة الرسم HTML5 Canvas إلى مصفوفة NumPy"""
        try:
            if "," in base64_str:
                base64_str = base64_str.split(",")[1]
            img_bytes = base64.b64decode(base64_str)
            image = Image.open(io.BytesIO(img_bytes)).convert("L")
            return np.array(image)
        except Exception as e:
            logger.error(f"خطأ في فك ترميز Base64: {e}")
            return None

    def _preprocess_for_model(self, img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """معالجة الصورة وتحويلها لمتجه بكسلات مطبّع مع محاذاة وتوسيط الرسم"""
        if len(img.shape) == 3:
            # تحويل إلى رمادي
            img = np.mean(img, axis=2).astype(np.uint8)

        # التحقق مما إذا كانت الخلفية بيضاء والرسم أسود (كما في المستندات الورقية المرفوعة) لعكسها
        if img.shape[0] > 10 and img.shape[1] > 10:
            corners = [img[0, 0], img[0, -1], img[-1, 0], img[-1, -1]]
            if np.mean(corners) > 127:
                img = 255 - img

        # اقتصاص حول محتوى الرسم إذا كانت الصورة أكبر من الحجم المستهدف (مثل Canvas 300x300)
        if img.shape[0] > target_size[0] or img.shape[1] > target_size[1]:
            non_zero = np.argwhere(img > 25)
            if non_zero.size > 0:
                y_min, x_min = non_zero.min(axis=0)
                y_max, x_max = non_zero.max(axis=0)
                cropped = img[y_min:y_max + 1, x_min:x_max + 1]
                h, w = cropped.shape
                side = max(h, w)
                pad = int(side * 0.18)
                new_side = side + 2 * pad
                square = np.zeros((new_side, new_side), dtype=np.uint8)
                y_offset = (new_side - h) // 2
                x_offset = (new_side - w) // 2
                square[y_offset:y_offset + h, x_offset:x_offset + w] = cropped
                img = square

        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize(target_size, Image.Resampling.BILINEAR)
        arr = np.array(pil_img).astype(np.float64) / 255.0
        return arr.flatten()

    def _transform_features(self, vector: np.ndarray, pca=None, scaler=None) -> np.ndarray:
        """تطبيق تقليل الأبعاد (PCA) والمعايرة (Scaler) بالترتيب المناسب حسب خصائص النموذج"""
        vec = vector.reshape(1, -1) if vector.ndim == 1 else vector

        if pca is not None and scaler is not None:
            pca_in = getattr(pca, "n_features_in_", None)
            scaler_in = getattr(scaler, "n_features_in_", None)

            if pca_in == vec.shape[1]:
                vec = pca.transform(vec)
                vec = scaler.transform(vec)
            elif scaler_in == vec.shape[1]:
                vec = scaler.transform(vec)
                vec = pca.transform(vec)
            else:
                vec = pca.transform(vec)
                vec = scaler.transform(vec)
        elif pca is not None:
            vec = pca.transform(vec)
        elif scaler is not None:
            vec = scaler.transform(vec)

        return vec

    def recognize(self, image: np.ndarray, mode: str = "auto") -> Dict[str, Any]:
        """
        التعرف الشامل على الصورة حسب الوضع المطلوب:
          - 'digit': التعرف على رقم عربي
          - 'character': التعرف على حرف عربي
          - 'auto': اكتشاف تلقائي (أرقام أو حروف)
          - 'word': محاولة التعرف حرف بحرف
          - 'paragraph': تحليل الفقرة
        """
        if image is None or image.size == 0:
            return {"success": False, "error": "الصورة غير صالحة أو فارغة"}

        if mode == "digit":
            return self._recognize_digit(image)
        elif mode == "character":
            return self._recognize_character(image)
        elif mode == "word":
            return self._recognize_word(image)
        elif mode == "paragraph":
            return self._recognize_paragraph(image)
        else:
            # الوضع التلقائي: جرّب كلا النموذجين واختر الأعلى ثقة
            return self._recognize_auto(image)

    def _recognize_digit(self, image: np.ndarray) -> Dict[str, Any]:
        """التعرف على رقم عربي/هندي باستخدام نموذج AHDD"""
        if self.digit_model is None:
            return {"success": False, "error": "نموذج الأرقام غير محمّل. شغّل التدريب أولاً."}

        vector = self._preprocess_for_model(image, (28, 28))
        vector = self._transform_features(vector, pca=self.digit_pca, scaler=self.digit_scaler)

        pred = self.digit_model.predict(vector)[0]
        probs = self._get_proba(self.digit_model, vector)
        classes = self.digit_classes

        top5 = np.argsort(probs[0])[::-1][:5]
        alternatives = []
        for idx in top5:
            cls = int(classes[idx])
            alternatives.append({
                "class": str(cls),
                "arabic_text": ARABIC_DIGIT_MAP.get(cls, str(cls)),
                "confidence": float(round(probs[0][idx] * 100, 2))
            })

        main_arabic = ARABIC_DIGIT_MAP.get(int(pred), str(pred))
        return {
            "success": True,
            "mode": "digit",
            "prediction": main_arabic,
            "prediction_raw": str(pred),
            "confidence": alternatives[0]["confidence"] if alternatives else 0,
            "alternatives": alternatives,
            "model_accuracy": f"{self.digit_accuracy*100:.2f}%"
        }

    def _recognize_character(self, image: np.ndarray) -> Dict[str, Any]:
        """التعرف على حرف عربي باستخدام نموذج AHCD"""
        if self.char_model is None:
            return {"success": False, "error": "نموذج الحروف غير محمّل. شغّل التدريب أولاً."}

        vector = self._preprocess_for_model(image, (32, 32))
        vector = self._transform_features(vector, pca=self.char_pca, scaler=self.char_scaler)

        pred = self.char_model.predict(vector)[0]
        probs = self._get_proba(self.char_model, vector)
        classes = self.char_classes

        top5 = np.argsort(probs[0])[::-1][:5]
        alternatives = []
        for idx in top5:
            cls = int(classes[idx])
            alternatives.append({
                "class": str(cls),
                "arabic_text": AHCD_LABEL_TO_ARABIC.get(cls, str(cls)),
                "confidence": float(round(probs[0][idx] * 100, 2))
            })

        main_arabic = AHCD_LABEL_TO_ARABIC.get(int(pred), str(pred))
        return {
            "success": True,
            "mode": "character",
            "prediction": main_arabic,
            "prediction_raw": str(pred),
            "confidence": alternatives[0]["confidence"] if alternatives else 0,
            "alternatives": alternatives,
            "model_accuracy": f"{self.char_accuracy*100:.2f}%"
        }

    def _recognize_auto(self, image: np.ndarray) -> Dict[str, Any]:
        """التعرف التلقائي: جرّب نموذج الأرقام والحروف واختر الأعلى ثقة"""
        results = []

        if self.digit_model is not None:
            digit_result = self._recognize_digit(image)
            if digit_result.get("success"):
                results.append(digit_result)

        if self.char_model is not None:
            char_result = self._recognize_character(image)
            if char_result.get("success"):
                results.append(char_result)

        if not results:
            return {"success": False, "error": "لا توجد نماذج محمّلة. شغّل التدريب في النوت بوك أولاً."}

        # اختيار النتيجة الأعلى ثقة
        best_chosen = max(results, key=lambda r: r.get("confidence", 0))
        best = dict(best_chosen)
        best["all_results"] = [dict(r) for r in results]
        return best

    def _recognize_word(self, image: np.ndarray) -> Dict[str, Any]:
        """
        التعرف على كلمة كاملة عبر تقسيمها إلى أجزاء
        يستخدم تقنية الإسقاط العمودي (Vertical Projection) لتحديد مواقع القطع
        """
        if self.char_model is None:
            return {"success": False, "error": "نموذج الحروف غير محمّل"}

        if len(image.shape) == 3:
            gray = np.mean(image, axis=2).astype(np.uint8)
        else:
            gray = image.copy()

        # عتبة تكيفية (Otsu)
        threshold = np.mean(gray)
        binary = (gray < threshold).astype(np.uint8) * 255

        # الإسقاط العمودي
        v_proj = np.sum(binary, axis=0)
        threshold_proj = np.max(v_proj) * 0.05

        # تحديد مناطق الأحرف
        segments = []
        in_segment = False
        start = 0
        for i, val in enumerate(v_proj):
            if val > threshold_proj and not in_segment:
                in_segment = True
                start = i
            elif val <= threshold_proj and in_segment:
                in_segment = False
                if i - start > 3:
                    segments.append((start, i))
        if in_segment and len(v_proj) - start > 3:
            segments.append((start, len(v_proj)))

        if not segments:
            # إذا فشل التقسيم، تعامل مع الصورة كحرف واحد
            return self._recognize_character(image)

        # ترتيب من اليمين لليسار (الكتابة العربية)
        segments.sort(key=lambda s: s[0], reverse=True)

        recognized_chars = []
        for start, end in segments:
            char_img = gray[:, start:end]
            result = self._recognize_character(char_img)
            if result.get("success"):
                recognized_chars.append({
                    "char": result["prediction"],
                    "confidence": result["confidence"]
                })

        word = "".join([c["char"] for c in recognized_chars])
        avg_conf = np.mean([c["confidence"] for c in recognized_chars]) if recognized_chars else 0

        return {
            "success": True,
            "mode": "word",
            "prediction": word,
            "confidence": float(round(avg_conf, 2)),
            "characters": recognized_chars,
            "segments_count": len(segments),
            "alternatives": []
        }

    def _recognize_paragraph(self, image: np.ndarray) -> Dict[str, Any]:
        """
        التعرف على فقرة عبر تقسيمها إلى أسطر ثم كلمات ثم حروف
        """
        if self.char_model is None:
            return {"success": False, "error": "نموذج الحروف غير محمّل"}

        if len(image.shape) == 3:
            gray = np.mean(image, axis=2).astype(np.uint8)
        else:
            gray = image.copy()

        threshold = np.mean(gray)
        binary = (gray < threshold).astype(np.uint8) * 255

        # الإسقاط الأفقي لتحديد الأسطر
        h_proj = np.sum(binary, axis=1)
        threshold_proj = np.max(h_proj) * 0.03

        lines = []
        in_line = False
        start = 0
        for i, val in enumerate(h_proj):
            if val > threshold_proj and not in_line:
                in_line = True
                start = i
            elif val <= threshold_proj and in_line:
                in_line = False
                if i - start > 5:
                    lines.append((start, i))
        if in_line and len(h_proj) - start > 5:
            lines.append((start, len(h_proj)))

        if not lines:
            return self._recognize_word(image)

        recognized_lines = []
        for line_start, line_end in lines:
            line_img = gray[line_start:line_end, :]
            line_result = self._recognize_word(line_img)
            if line_result.get("success"):
                recognized_lines.append(line_result["prediction"])

        full_text = "\n".join(recognized_lines)
        return {
            "success": True,
            "mode": "paragraph",
            "prediction": full_text,
            "confidence": 0,
            "lines_count": len(lines),
            "lines": recognized_lines,
            "alternatives": []
        }

    # ─── وضع التوافق مع الكود القديم ───
    def recognize_single_image(self, image: np.ndarray) -> Dict[str, Any]:
        """واجهة التوافق مع الكود السابق"""
        return self.recognize(image, mode="auto")

    def _get_proba(self, model, X):
        """الحصول على احتماليات التنبؤ بطريقة آمنة"""
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)
        elif hasattr(model, "decision_function"):
            df = model.decision_function(X)
            if df.ndim == 1:
                df = df.reshape(1, -1)
            exp_df = np.exp(df - np.max(df, axis=1, keepdims=True))
            return exp_df / np.sum(exp_df, axis=1, keepdims=True)
        else:
            preds = model.predict(X)
            classes = model.classes_
            probs = np.zeros((len(X), len(classes)))
            for idx, p in enumerate(preds):
                c_idx = np.where(classes == p)[0][0]
                probs[idx, c_idx] = 1.0
            return probs
