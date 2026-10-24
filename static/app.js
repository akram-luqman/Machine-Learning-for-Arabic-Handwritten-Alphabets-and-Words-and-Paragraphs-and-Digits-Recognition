/**
 * المنطق التفاعلي لمنصة التعرف على الكتابة اليدوية العربية (النسخة 2.0)
 * يدعم جميع أوضاع التعرف: أرقام، حروف، كلمات، فقرات، تلقائي
 */

// ═══════════════════════ المتغيرات العامة ═══════════════════════
const canvas = document.getElementById('paintCanvas');
const ctx = canvas.getContext('2d');
const brushSizeInput = document.getElementById('brushSize');
const brushSizeVal = document.getElementById('brushSizeVal');
const resultChar = document.getElementById('resultChar');
const confidenceText = document.getElementById('confidenceText');
const confidenceBar = document.getElementById('confidenceBar');
const alternativesList = document.getElementById('alternativesList');
const accumulatedTextArea = document.getElementById('accumulatedTextArea');
const loadingBadge = document.getElementById('loadingBadge');
const modeBadge = document.getElementById('modeBadge');
const dropzone = document.getElementById('dropzone');
const uploadPreviewWrapper = document.getElementById('uploadPreviewWrapper');
const uploadPreview = document.getElementById('uploadPreview');
const fileInput = document.getElementById('fileInput');

let isDrawing = false;
let currentPredictionChar = "";
let selectedFile = null;
let currentMode = "auto";

const MODE_LABELS = {
    "auto": "تلقائي",
    "digit": "أرقام",
    "character": "حروف",
    "word": "كلمات",
    "paragraph": "فقرات"
};

const CANVAS_HINTS = {
    "auto": "ارسم أي حرف، رقم، أو كلمة عربية:",
    "digit": "ارسم رقماً عربياً واحداً (٠-٩):",
    "character": "ارسم حرفاً عربياً واحداً (ا-ي):",
    "word": "ارسم كلمة عربية كاملة:",
    "paragraph": "ارسم جملة أو فقرة عربية:"
};

// ═══════════════════════ تهيئة الصفحة ═══════════════════════
function initCanvas() {
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = "#FFFFFF";
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.lineWidth = parseInt(brushSizeInput.value, 10);
}

function setupCanvasEvents() {
    function getCoords(e) {
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        let clientX = e.clientX, clientY = e.clientY;
        if (e.touches && e.touches.length > 0) {
            clientX = e.touches[0].clientX;
            clientY = e.touches[0].clientY;
        }
        return {
            x: (clientX - rect.left) * scaleX,
            y: (clientY - rect.top) * scaleY
        };
    }

    function startDraw(e) {
        e.preventDefault();
        isDrawing = true;
        const coords = getCoords(e);
        ctx.beginPath();
        ctx.moveTo(coords.x, coords.y);
    }

    function draw(e) {
        if (!isDrawing) return;
        e.preventDefault();
        const coords = getCoords(e);
        ctx.lineTo(coords.x, coords.y);
        ctx.stroke();
    }

    function stopDraw(e) {
        if (isDrawing) {
            e.preventDefault();
            isDrawing = false;
            ctx.closePath();
        }
    }

    canvas.addEventListener('mousedown', startDraw);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDraw);
    canvas.addEventListener('mouseleave', stopDraw);
    canvas.addEventListener('touchstart', startDraw, { passive: false });
    canvas.addEventListener('touchmove', draw, { passive: false });
    canvas.addEventListener('touchend', stopDraw, { passive: false });

    brushSizeInput.addEventListener('input', (e) => {
        const val = e.target.value;
        ctx.lineWidth = parseInt(val, 10);
        brushSizeVal.textContent = `${val}px`;
    });
}

// ═══════════════════════ أوضاع التعرف ═══════════════════════
function setMode(mode) {
    currentMode = mode;
    document.querySelectorAll('.mode-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelector(`.mode-btn[data-mode="${mode}"]`).classList.add('active');
    modeBadge.textContent = `الوضع: ${MODE_LABELS[mode]}`;
    document.getElementById('canvasHint').textContent = CANVAS_HINTS[mode];
}

// ═══════════════════════ التعرف ═══════════════════════
function clearCanvas() {
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    resetResultUI();
}

function resetResultUI() {
    resultChar.textContent = "—";
    resultChar.style.fontSize = "";
    confidenceText.textContent = "0%";
    confidenceBar.style.width = "0%";
    confidenceBar.className = "gauge-bar-fill";
    alternativesList.innerHTML = '<p class="text-muted">ارسم أو ارفع صورة للبدء بالتعرف...</p>';
    currentPredictionChar = "";
    const meta = document.getElementById('resultMeta');
    if (meta) meta.style.display = 'none';
}

function switchTab(tabName) {
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
    if (tabName === 'canvas') {
        document.getElementById('tab-canvas-btn').classList.add('active');
        document.getElementById('canvas-tab-content').classList.add('active');
    } else {
        document.getElementById('tab-upload-btn').classList.add('active');
        document.getElementById('upload-tab-content').classList.add('active');
    }
}

async function recognizeCanvas() {
    showLoading(true);
    try {
        const base64Data = canvas.toDataURL('image/png');
        const response = await fetch('/api/recognize_canvas', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: base64Data, mode: currentMode })
        });
        const data = await response.json();
        handleRecognitionResponse(data);
    } catch (err) {
        console.error("خطأ أثناء الاتصال بالخادم:", err);
        alert("حدث خطأ أثناء محاولة التعرف. تأكد من تشغيل الخادم عبر: python app.py");
    } finally {
        showLoading(false);
    }
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (!file) return;
    previewFile(file);
}

function previewFile(file) {
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        uploadPreview.src = e.target.result;
        dropzone.style.display = 'none';
        uploadPreviewWrapper.style.display = 'block';
    };
    reader.readAsDataURL(file);
}

function resetUpload() {
    selectedFile = null;
    fileInput.value = "";
    uploadPreview.src = "";
    dropzone.style.display = 'block';
    uploadPreviewWrapper.style.display = 'none';
    resetResultUI();
}

async function recognizeUpload() {
    if (!selectedFile) {
        alert("يرجى اختيار صورة أولاً.");
        return;
    }
    showLoading(true);
    try {
        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('mode', currentMode);

        const response = await fetch('/api/recognize_upload', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        handleRecognitionResponse(data);
    } catch (err) {
        console.error("خطأ في رفع الصورة:", err);
        alert("حدث خطأ أثناء معالجة الصورة المرفوعة.");
    } finally {
        showLoading(false);
    }
}

// ═══════════════════════ عرض النتائج ═══════════════════════
function handleRecognitionResponse(data) {
    if (!data.success) {
        alert(data.error || "تعذر التعرف على المحتوى");
        return;
    }

    currentPredictionChar = data.prediction;

    // تعديل حجم الخط بناءً على طول النتيجة
    if (data.prediction.length > 10) {
        resultChar.style.fontSize = "1.5rem";
    } else if (data.prediction.length > 3) {
        resultChar.style.fontSize = "2.5rem";
    } else {
        resultChar.style.fontSize = "";
    }

    resultChar.textContent = data.prediction;
    const conf = data.confidence || 0;
    confidenceText.textContent = `${conf}%`;
    confidenceBar.style.width = `${Math.min(100, conf)}%`;

    // لون شريط الثقة حسب القيمة
    if (conf >= 90) {
        confidenceBar.className = "gauge-bar-fill confidence-high";
    } else if (conf >= 70) {
        confidenceBar.className = "gauge-bar-fill confidence-medium";
    } else {
        confidenceBar.className = "gauge-bar-fill confidence-low";
    }

    // بيانات وصفية
    const meta = document.getElementById('resultMeta');
    const resultMode = document.getElementById('resultMode');
    const resultAccuracy = document.getElementById('resultAccuracy');
    if (meta && data.mode) {
        meta.style.display = 'flex';
        resultMode.textContent = `النوع: ${MODE_LABELS[data.mode] || data.mode}`;
        if (data.model_accuracy) {
            resultAccuracy.textContent = `دقة النموذج: ${data.model_accuracy}`;
        }
    }

    // البدائل
    if (data.alternatives && data.alternatives.length > 0) {
        alternativesList.innerHTML = "";
        data.alternatives.forEach(alt => {
            const pill = document.createElement('div');
            pill.className = 'alt-pill';
            pill.innerHTML = `
                <span class="alt-char">${alt.arabic_text}</span>
                <span class="alt-conf">${alt.confidence}%</span>
            `;
            pill.onclick = () => {
                currentPredictionChar = alt.arabic_text;
                resultChar.textContent = alt.arabic_text;
            };
            alternativesList.appendChild(pill);
        });
    } else {
        alternativesList.innerHTML = '<p class="text-muted">لا توجد بدائل</p>';
    }
}

// ═══════════════════════ تجميع النص ═══════════════════════
function appendCurrentChar() {
    if (!currentPredictionChar) return;
    accumulatedTextArea.value += currentPredictionChar;
}

function addSpace() {
    accumulatedTextArea.value += " ";
}

function addNewLine() {
    accumulatedTextArea.value += "\n";
}

function copyAccumulatedText() {
    const text = accumulatedTextArea.value;
    if (!text) return;
    navigator.clipboard.writeText(text).then(() => {
        alert("تم نسخ النص إلى الحافظة بنجاح! ✅");
    }).catch(err => {
        console.error("فشل النسخ:", err);
    });
}

function downloadAccumulatedText() {
    const text = accumulatedTextArea.value;
    if (!text) { alert("لا يوجد نص للتنزيل!"); return; }
    const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = "arabic_ocr_text.txt";
    a.click();
    URL.revokeObjectURL(url);
}

function clearAccumulatedText() {
    accumulatedTextArea.value = "";
}

function showLoading(show) {
    loadingBadge.style.display = show ? 'inline-block' : 'none';
}

// ═══════════════════════ السحب والإفلات ═══════════════════════
function setupDragAndDrop() {
    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.style.borderColor = 'var(--accent-primary)';
        }, false);
    });
    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.style.borderColor = 'rgba(0, 210, 255, 0.35)';
        }, false);
    });
    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        if (dt.files.length > 0) previewFile(dt.files[0]);
    });
}

// ═══════════════════════ فحص حالة النماذج ═══════════════════════
async function checkModelsStatus() {
    try {
        const response = await fetch('/api/models_status');
        const data = await response.json();
        const headerStatus = document.getElementById('headerStatus');
        const accuracyBadge = document.getElementById('accuracyBadge');

        if (data.digit_model_loaded || data.char_model_loaded) {
            headerStatus.innerHTML = '<span class="status-indicator status-ok"></span><span>النماذج جاهزة ✅</span>';

            let accText = [];
            if (data.digit_model_loaded) accText.push(`أرقام: ${data.digit_accuracy}`);
            if (data.char_model_loaded) accText.push(`حروف: ${data.char_accuracy}`);
            accuracyBadge.textContent = accText.join(' | ');
        } else {
            headerStatus.innerHTML = '<span class="status-indicator status-warn"></span><span>لا توجد نماذج مدربة ⚠️</span>';
            accuracyBadge.textContent = "شغّل النوت بوك أولاً";
        }
    } catch (e) {
        console.warn("تعذر فحص حالة النماذج");
    }
}

// ═══════════════════════ بدء التشغيل ═══════════════════════
document.addEventListener('DOMContentLoaded', () => {
    initCanvas();
    setupCanvasEvents();
    setupDragAndDrop();
    checkModelsStatus();
});
