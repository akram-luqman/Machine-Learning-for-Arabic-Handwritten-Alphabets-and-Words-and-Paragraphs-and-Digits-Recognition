"""train_character_svm_79.py
--------------------------------
تدريب نموذج SVM (RBF) لتصنيف الحروف العربية (مجموعة AHCD)
المعادلة التي حققت أعلى دقة عندنا في المحادثات السابقة هي **79.26 %**.
يتم حفظ النموذج في  ``models/character/best_model.joblib`` ليتم تحميله تلقائيًا من قبل ``engine.py``.

المتطلبات:
- pandas, numpy, scikit‑learn, joblib
- مجموعة البيانات موجودة في ``datasets/httpswww.kaggle.commloey1ahcd1``
"""

import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ------------------------------------------------------------------
# 1️⃣ تحميل البيانات
# ------------------------------------------------------------------
DATA_ROOT = Path('datasets/httpswww.kaggle.commloey1ahcd1')

# ملفات CSV التي تم استخراجها من مجموعة Kaggle (الأرقام في الاسم تشير إلى الأبعاد)
X_train = pd.read_csv(DATA_ROOT / 'csvTrainImages 13440x1024.csv', header=None).values.astype('float32') / 255.0
y_train = pd.read_csv(DATA_ROOT / 'csvTrainLabel 13440x1.csv', header=None)[0].values

X_test = pd.read_csv(DATA_ROOT / 'csvTestImages 3360x1024.csv', header=None).values.astype('float32') / 255.0
y_test = pd.read_csv(DATA_ROOT / 'csvTestLabel 3360x1.csv', header=None)[0].values

print(f"[📊] حجم التدريب: {X_train.shape}, حجم الاختبار: {X_test.shape}")

# ------------------------------------------------------------------
# 2️⃣ تقليل الأبعاد باستخدام PCA
# ------------------------------------------------------------------
# اختيار 250 مكوّن يُحوّّل التباين إلى ~96 %
pca = PCA(n_components=250, random_state=42)
X_train_pca = pca.fit_transform(X_train)
X_test_pca  = pca.transform(X_test)
print(f"[🔎] بعد PCA → {X_train_pca.shape[1]} مكوّن (تباين مُجمل: {pca.explained_variance_ratio_.sum()*100:.1f}%)")

# ------------------------------------------------------------------
# 3️⃣ معيارية البيانات (StandardScaler)
# ------------------------------------------------------------------
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train_pca)
X_test_std  = scaler.transform(X_test_pca)

# ------------------------------------------------------------------
# 4️⃣ إيجاد أفضل قيم للمعاملات C و gamma عبر GridSearchCV
# ------------------------------------------------------------------
param_grid = {
    "C": [10, 15, 20, 25, 30],
    "gamma": ["scale", 0.01, 0.05, 0.1]
}
svm = SVC(kernel='rbf', random_state=42)

grid = GridSearchCV(
    estimator=svm,
    param_grid=param_grid,
    scoring='accuracy',
    cv=3,
    n_jobs=-1,
    verbose=1
)

start = time.time()
grid.fit(X_train_std, y_train)
print(f"[⏱️] GridSearch انتهى في {time.time() - start:.1f}s")
print(f"[🏅] أفضل معاملات: C={grid.best_params_['C']}, gamma={grid.best_params_['gamma']}")
print(f"[🏆] أفضل دقة داخل الـ CV: {grid.best_score_*100:.2f}%")

best_clf = grid.best_estimator_

# ------------------------------------------------------------------
# 5️⃣ تقييم على مجموعة الاختبار
# ------------------------------------------------------------------
y_pred = best_clf.predict(X_test_std)
test_acc = accuracy_score(y_test, y_pred)
print(f"[🧪] دقة الاختبار على الحروف: {test_acc*100:.2f}%")

# ------------------------------------------------------------------
# 6️⃣ حفظ النموذج (مع جميع المكوّنات الضرورية للاستدلال)
# ------------------------------------------------------------------
model_dir = Path('models/character')
model_dir.mkdir(parents=True, exist_ok=True)
payload = {
    "algorithm": "svm_rbf_pca_grid",
    "classifier": best_clf,
    "scaler": scaler,
    "pca": pca,
    "classes": best_clf.classes_,
    "is_fitted": True,
    "accuracy": float(test_acc),
    "task": "character_recognition",
    "n_features": 1024,
    "image_size": (32, 32),
}
joblib.dump(payload, model_dir / 'best_model.joblib')
print(f"✅ النموذج حفظ في {model_dir / 'best_model.joblib'}")
