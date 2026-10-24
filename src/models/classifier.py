# -*- coding: utf-8 -*-
"""
نماذج التصنيف التقليدية للكتابة العربية
========================================
نماذج تعلم آلة كلاسيكية بحتة (بدون شبكات عصبية إطلاقاً):
- SVM (Support Vector Machine) مع نواة RBF و Linear
- Random Forest Classifier (الغابات العشوائية)
- Extra Trees Classifier
- K-Nearest Neighbors (KNN)
- Logistic Regression
"""

import logging
from typing import Dict, Any, Optional, Tuple
import joblib
from pathlib import Path

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

from ..utils.helpers import setup_logger

logger = setup_logger(__name__)


class TraditionalArabicClassifier:
    """
    مغلف لنماذج تعلم الآلة التقليدي للتعرف على الحروف والأرقام والكلمات
    """

    SUPPORTED_ALGORITHMS = {
        "svm_rbf": lambda: SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42),
        "svm_linear": lambda: SVC(kernel='linear', C=1.0, probability=True, random_state=42),
        "random_forest": lambda: RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1),
        "extra_trees": lambda: ExtraTreesClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1),
        "knn": lambda: KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1),
        "logistic_regression": lambda: LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
    }

    def __init__(self, algorithm: str = "svm_rbf"):
        if algorithm not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(f"خوارزمية غير مدعومة: {algorithm}. المتاح: {list(self.SUPPORTED_ALGORITHMS.keys())}")
        
        self.algorithm_name = algorithm
        self.classifier = self.SUPPORTED_ALGORITHMS[algorithm]()
        self.scaler = StandardScaler()
        self.classes_ = None
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """تدريب النموذج بعد ضبط المقاييس"""
        logger.info(f"بدء تدريب النموذج {self.algorithm_name} على {len(X)} عينة...")
        X_scaled = self.scaler.fit_transform(X)
        self.classifier.fit(X_scaled, y)
        self.classes_ = self.classifier.classes_
        self.is_fitted = True
        logger.info(f"اكتمل التدريب بنجاح! عدد الفئات: {len(self.classes_)}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """التنبؤ بالفئة"""
        if not self.is_fitted:
            raise RuntimeError("النموذج لم يتم تدريبه بعد!")
        X_scaled = self.scaler.transform(X) if self.scaler is not None else X
        return self.classifier.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """احتماليات التنبؤ لكل فئة"""
        if not self.is_fitted:
            raise RuntimeError("النموذج لم يتم تدريبه بعد!")
        X_scaled = self.scaler.transform(X) if self.scaler is not None else X
        if hasattr(self.classifier, "predict_proba"):
            return self.classifier.predict_proba(X_scaled)
        elif hasattr(self.classifier, "decision_function"):
            df = self.classifier.decision_function(X_scaled)
            exp_df = np.exp(df - np.max(df, axis=1, keepdims=True))
            return exp_df / np.sum(exp_df, axis=1, keepdims=True)
        else:
            preds = self.predict(X)
            probs = np.zeros((len(X), len(self.classes_)))
            for idx, p in enumerate(preds):
                c_idx = np.where(self.classes_ == p)[0][0]
                probs[idx, c_idx] = 1.0
            return probs

    def save(self, filepath: Path):
        """حفظ النموذج والمقياس"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "algorithm": self.algorithm_name,
            "classifier": self.classifier,
            "scaler": self.scaler,
            "classes": self.classes_,
            "is_fitted": self.is_fitted
        }
        joblib.dump(payload, filepath)
        logger.info(f"تم حفظ النموذج في: {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> 'TraditionalArabicClassifier':
        """تحميل نموذج محفوظ"""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"ملف النموذج غير موجود: {filepath}")
        payload = joblib.load(filepath)
        instance = cls(algorithm=payload["algorithm"])
        instance.classifier = payload["classifier"]
        instance.scaler = payload["scaler"]
        instance.classes_ = payload["classes"]
        instance.is_fitted = payload["is_fitted"]
        logger.info(f"تم تحميل النموذج بنجاح من: {filepath}")
        return instance
