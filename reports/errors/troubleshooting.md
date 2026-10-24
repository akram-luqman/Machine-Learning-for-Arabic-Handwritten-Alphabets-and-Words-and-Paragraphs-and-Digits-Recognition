# ⚠️ الأخطاء الشائعة وحلولها
## Common Errors & Troubleshooting

### 1. خطأ: النموذج غير محمّل
**السبب:** لم يتم تدريب النموذج بعد
**الحل:** شغّل النوت بوك 
otebooks/arabic_handwriting_ocr_pipeline.ipynb

### 2. خطأ: ModuleNotFoundError
**السبب:** المكتبات غير مثبتة
**الحل:** pip install -r requirements.txt

### 3. خطأ: ذاكرة غير كافية
**السبب:** SVM يحتاج ذاكرة كبيرة مع بيانات ضخمة
**الحل:** تقليل عدد عينات التدريب (N_TRAIN)
