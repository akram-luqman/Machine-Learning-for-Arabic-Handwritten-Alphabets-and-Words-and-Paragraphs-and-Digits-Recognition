# -*- coding: utf-8 -*-
"""
محرك اكتشاف مجموعات البيانات
==============================
يكتشف تلقائيًا جميع مجموعات البيانات في المجلدات المحددة،
بما في ذلك الأرشيفات والصور وملفات CSV والبيانات الوصفية.
يبني جردًا شاملًا ويصنف البيانات حسب نوع المهمة.
"""

import os
import json
import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from datetime import datetime

import numpy as np

from ..utils.helpers import (
    setup_logger, measure_time, format_size, is_image_file,
    validate_image_content, compute_file_hash, safe_filename
)

logger = setup_logger(__name__)


class DatasetDiscoveryEngine:
    """
    محرك الاكتشاف التلقائي لمجموعات البيانات

    يبحث في المجلدات المحددة عن:
    - ملفات الصور (PNG, JPG, BMP, TIFF, WEBP)
    - ملفات البيانات (CSV, TSV, JSON, JSONL, TXT, XML, Parquet, Excel)
    - الأرشيفات (ZIP, TAR, RAR, 7z)
    - المجلدات المتداخلة بأي عمق

    يبني جردًا شاملًا ويصنف البيانات حسب المهمة.
    """

    # الامتدادات المعروفة
    IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}
    DATA_EXTENSIONS = {'.csv', '.tsv', '.json', '.jsonl', '.txt', '.xml', '.parquet', '.xlsx', '.xls', '.mat'}
    ARCHIVE_EXTENSIONS = {'.zip', '.tar', '.tar.gz', '.tgz', '.rar', '.7z', '.gz'}

    # أنماط أسماء الملفات المعروفة
    LABEL_COLUMN_PATTERNS = [
        'label', 'class', 'target', 'character', 'char', 'letter',
        'word', 'text', 'digit', 'category', 'type', 'name',
    ]

    IMAGE_COLUMN_PATTERNS = [
        'image', 'img', 'file', 'path', 'filename', 'image_path',
        'filepath', 'image_name', 'img_path',
    ]

    def __init__(self, search_paths: List[Path],
                 manifests_dir: Optional[Path] = None,
                 reports_dir: Optional[Path] = None):
        """
        تهيئة محرك الاكتشاف

        المدخلات:
            search_paths: قائمة المسارات للبحث فيها
            manifests_dir: مجلد حفظ ملفات الجرد
            reports_dir: مجلد حفظ التقارير
        """
        self.search_paths = [Path(p) for p in search_paths]
        self.manifests_dir = manifests_dir
        self.reports_dir = reports_dir

        # نتائج الاكتشاف
        self.discovered_sources: List[Dict[str, Any]] = []
        self.image_files: List[Dict[str, Any]] = []
        self.data_files: List[Dict[str, Any]] = []
        self.archive_files: List[Dict[str, Any]] = []
        self.datasets: List[Dict[str, Any]] = []
        self.task_classification: Dict[str, List[Dict]] = defaultdict(list)
        self.errors: List[Dict[str, Any]] = []

    @measure_time
    def discover_all(self) -> Dict[str, Any]:
        """
        تنفيذ الاكتشاف الشامل لجميع مجموعات البيانات

        المخرجات:
            تقرير شامل بنتائج الاكتشاف
        """
        logger.info("=" * 60)
        logger.info("بدء اكتشاف مجموعات البيانات...")
        logger.info("=" * 60)

        for search_path in self.search_paths:
            if not search_path.exists():
                logger.warning(f"المسار غير موجود: {search_path}")
                continue
            logger.info(f"البحث في: {search_path}")
            self._scan_directory(search_path, source_name=search_path.name)

        # تحليل وتصنيف البيانات المكتشفة
        self._analyze_discovered_data()
        self._classify_by_task()

        report = self._build_discovery_report()

        # حفظ التقارير
        if self.manifests_dir:
            self._save_manifests()
        if self.reports_dir:
            self._save_reports(report)

        logger.info(f"اكتمل الاكتشاف: {len(self.image_files)} صورة، "
                     f"{len(self.data_files)} ملف بيانات، "
                     f"{len(self.archive_files)} أرشيف")

        return report

    def _scan_directory(self, directory: Path, source_name: str,
                       depth: int = 0, max_depth: int = 15):
        """
        مسح مجلد بشكل تعاودي لاكتشاف جميع الملفات

        المدخلات:
            directory: المجلد المراد مسحه
            source_name: اسم المصدر
            depth: عمق التداخل الحالي
            max_depth: الحد الأقصى للعمق
        """
        if depth > max_depth:
            self.errors.append({
                "type": "max_depth_exceeded",
                "path": str(directory),
                "message": f"تم تجاوز الحد الأقصى للتداخل ({max_depth})"
            })
            return

        try:
            entries = sorted(directory.iterdir())
        except PermissionError:
            self.errors.append({
                "type": "permission_error",
                "path": str(directory),
                "message": "لا توجد صلاحية للوصول"
            })
            return
        except OSError as e:
            self.errors.append({
                "type": "os_error",
                "path": str(directory),
                "message": str(e)
            })
            return

        for entry in entries:
            try:
                if entry.is_file():
                    self._categorize_file(entry, source_name)
                elif entry.is_dir():
                    # تسجيل المجلد كمصدر محتمل
                    self._scan_directory(entry, source_name, depth + 1, max_depth)
            except (PermissionError, OSError) as e:
                self.errors.append({
                    "type": "scan_error",
                    "path": str(entry),
                    "message": str(e)
                })

    def _categorize_file(self, file_path: Path, source_name: str):
        """تصنيف ملف بحسب نوعه"""
        suffix = file_path.suffix.lower()
        file_info = {
            "path": str(file_path),
            "name": file_path.name,
            "suffix": suffix,
            "size_bytes": file_path.stat().st_size,
            "source": source_name,
            "parent_dir": file_path.parent.name,
            "relative_path": str(file_path.relative_to(file_path.parents[min(5, len(file_path.parents) - 1)])),
        }

        # تصنيف الملف
        if suffix in self.IMAGE_EXTENSIONS:
            file_info["type"] = "image"
            self.image_files.append(file_info)
        elif suffix in self.DATA_EXTENSIONS:
            file_info["type"] = "data"
            self.data_files.append(file_info)
        elif suffix in self.ARCHIVE_EXTENSIONS or file_path.name.endswith('.tar.gz'):
            file_info["type"] = "archive"
            self.archive_files.append(file_info)
        # تجاهل الملفات غير ذات الصلة مثل Thumbs.db
        elif file_path.name.lower() not in {'thumbs.db', 'desktop.ini', '.ds_store'}:
            file_info["type"] = "other"

    def _analyze_discovered_data(self):
        """تحليل البيانات المكتشفة واكتشاف بنية مجموعات البيانات"""
        logger.info("تحليل البيانات المكتشفة...")

        # تجميع الصور حسب المجلد الأب
        images_by_parent = defaultdict(list)
        for img in self.image_files:
            parent = Path(img["path"]).parent
            images_by_parent[str(parent)].append(img)

        # تحليل ملفات CSV
        for data_file in self.data_files:
            if data_file["suffix"] in {'.csv', '.tsv'}:
                self._analyze_csv(data_file)

        # اكتشاف مجموعات البيانات المنظمة بالمجلدات
        self._discover_folder_organized_datasets(images_by_parent)

    def _analyze_csv(self, file_info: Dict):
        """تحليل ملف CSV واكتشاف أعمدته وبنيته"""
        file_path = Path(file_info["path"])
        try:
            # قراءة عينة من الملف
            import pandas as pd
            sep = '\t' if file_info["suffix"] == '.tsv' else ','

            # قراءة أول 5 صفوف لفحص البنية
            df_sample = pd.read_csv(file_path, sep=sep, nrows=5,
                                     encoding='utf-8', on_bad_lines='skip')

            # حساب عدد الصفوف الكلي بدون تحميل الملف كاملًا
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                total_rows = sum(1 for _ in f) - 1  # طرح صف العناوين

            file_info["columns"] = list(df_sample.columns)
            file_info["num_columns"] = len(df_sample.columns)
            file_info["total_rows"] = total_rows
            file_info["dtypes"] = {col: str(dt) for col, dt in df_sample.dtypes.items()}
            file_info["sample_values"] = {
                col: str(df_sample[col].iloc[0]) if len(df_sample) > 0 else None
                for col in df_sample.columns[:10]
            }

            # اكتشاف أعمدة الملصقات والصور
            file_info["potential_label_columns"] = [
                col for col in df_sample.columns
                if any(pattern in col.lower() for pattern in self.LABEL_COLUMN_PATTERNS)
            ]
            file_info["potential_image_columns"] = [
                col for col in df_sample.columns
                if any(pattern in col.lower() for pattern in self.IMAGE_COLUMN_PATTERNS)
            ]

            # اكتشاف إذا كانت بيانات بكسلات مسطحة
            if file_info["num_columns"] > 100:
                file_info["likely_pixel_data"] = True
                # تقدير أبعاد الصورة
                num_pixels = file_info["num_columns"]
                sqrt_val = int(np.sqrt(num_pixels))
                if sqrt_val * sqrt_val == num_pixels:
                    file_info["estimated_image_size"] = f"{sqrt_val}x{sqrt_val}"
                file_info["data_category"] = "pixel_matrix"
            else:
                file_info["likely_pixel_data"] = False
                file_info["data_category"] = "metadata"

            logger.info(f"  CSV: {file_path.name} - {total_rows} صف × {file_info['num_columns']} عمود")

        except Exception as e:
            file_info["analysis_error"] = str(e)
            self.errors.append({
                "type": "csv_analysis_error",
                "path": str(file_path),
                "message": str(e)
            })

    def _discover_folder_organized_datasets(self, images_by_parent: Dict):
        """اكتشاف مجموعات البيانات المنظمة بهيكلية المجلدات"""

        # البحث عن مجلدات تحتوي كل منها على صور (نمط فئة/صور)
        parent_of_parents = defaultdict(list)
        for parent_path, images in images_by_parent.items():
            grandparent = str(Path(parent_path).parent)
            parent_of_parents[grandparent].append({
                "folder": Path(parent_path).name,
                "path": parent_path,
                "num_images": len(images),
                "extensions": list(set(img["suffix"] for img in images)),
            })

        # تحديد المجلدات التي تبدو كمجموعات بيانات منظمة بالفئات
        for grandparent, folders in parent_of_parents.items():
            if len(folders) >= 2:  # على الأقل فئتان
                total_images = sum(f["num_images"] for f in folders)
                if total_images >= 10:  # عدد كافٍ من الصور
                    dataset_info = {
                        "type": "folder_organized",
                        "root_path": grandparent,
                        "name": Path(grandparent).name,
                        "num_classes": len(folders),
                        "total_images": total_images,
                        "classes": [f["folder"] for f in folders],
                        "class_distribution": {f["folder"]: f["num_images"] for f in folders},
                        "image_extensions": list(set(ext for f in folders for ext in f["extensions"])),
                    }
                    self.datasets.append(dataset_info)
                    logger.info(
                        f"  مجموعة منظمة: {dataset_info['name']} - "
                        f"{dataset_info['num_classes']} فئة، {total_images} صورة"
                    )

    def _classify_by_task(self):
        """تصنيف مجموعات البيانات حسب المهمة"""
        logger.info("تصنيف البيانات حسب المهمة...")

        for dataset in self.datasets:
            task = self._determine_task(dataset)
            dataset["task"] = task
            self.task_classification[task].append(dataset)

        # تصنيف ملفات CSV
        for data_file in self.data_files:
            if data_file.get("data_category") == "pixel_matrix":
                name_lower = data_file["name"].lower()
                if "digit" in name_lower or "ahdd" in name_lower:
                    task = "digits"
                elif "char" in name_lower or "ahcd" in name_lower:
                    task = "isolated_characters"
                else:
                    task = "unknown_pixel_data"
                data_file["task"] = task
                self.task_classification[task].append(data_file)

    def _determine_task(self, dataset: Dict) -> str:
        """تحديد نوع المهمة لمجموعة بيانات بناءً على بنيتها"""
        name_lower = dataset.get("name", "").lower()
        classes = [c.lower() for c in dataset.get("classes", [])]
        classes_str = " ".join(classes)

        # الأحرف المنفصلة مع الأشكال السياقية
        if any(form in classes_str for form in ["_begin", "_middle", "_end", "_regular"]):
            return "contextual_characters"

        # الأحرف المنفصلة
        if "alphabet" in name_lower or "character" in name_lower:
            return "isolated_characters"

        # الكلمات
        if "word" in name_lower:
            return "words"

        # الفقرات
        if "paragraph" in name_lower:
            return "paragraphs"

        # الأرقام
        if "digit" in name_lower or "ahd" in name_lower:
            return "digits"

        # تخمين من أسماء الفئات
        arabic_letter_names = {"alif", "beh", "teh", "theh", "jeem", "hah", "khah",
                               "dal", "raa", "seen", "sad", "tah", "ain", "feh",
                               "qaf", "kaf", "lam", "meem", "noon", "heh", "waw", "yaa"}
        matched_letters = sum(1 for c in classes if any(letter in c for letter in arabic_letter_names))
        if matched_letters >= 5:
            return "contextual_characters"

        # بيانات بالكتّاب والمسارات
        if any("user" in c or "writer" in c for c in classes):
            if "word" in name_lower:
                return "words"
            elif "paragraph" in name_lower:
                return "paragraphs"
            return "per_user_data"

        return "unclassified"

    def _build_discovery_report(self) -> Dict[str, Any]:
        """بناء تقرير الاكتشاف الشامل"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "search_paths": [str(p) for p in self.search_paths],
            "summary": {
                "total_image_files": len(self.image_files),
                "total_data_files": len(self.data_files),
                "total_archive_files": len(self.archive_files),
                "total_datasets_discovered": len(self.datasets),
                "total_errors": len(self.errors),
            },
            "image_statistics": self._compute_image_stats(),
            "data_files": [
                {k: v for k, v in f.items() if k != "sample_values"}
                for f in self.data_files
            ],
            "datasets": self.datasets,
            "task_classification": {
                task: len(items) for task, items in self.task_classification.items()
            },
            "errors": self.errors[:100],  # أول 100 خطأ فقط
        }
        return report

    def _compute_image_stats(self) -> Dict:
        """حساب إحصاءات الصور المكتشفة"""
        if not self.image_files:
            return {"count": 0}

        extensions = defaultdict(int)
        sources = defaultdict(int)
        total_size = 0

        for img in self.image_files:
            extensions[img["suffix"]] += 1
            sources[img["source"]] += 1
            total_size += img.get("size_bytes", 0)

        return {
            "count": len(self.image_files),
            "total_size": format_size(total_size),
            "by_extension": dict(extensions),
            "by_source": dict(sources),
        }

    def _save_manifests(self):
        """حفظ ملفات الجرد"""
        if not self.manifests_dir:
            return

        self.manifests_dir.mkdir(parents=True, exist_ok=True)

        # حفظ جرد الصور
        manifest_path = self.manifests_dir / "image_manifest.json"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "total_images": len(self.image_files),
                "images": self.image_files[:50000],  # حد أقصى
            }, f, ensure_ascii=False, indent=2)

        # حفظ جرد مجموعات البيانات
        datasets_path = self.manifests_dir / "datasets_manifest.json"
        with open(datasets_path, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "datasets": self.datasets,
                "task_classification": {
                    task: len(items) for task, items in self.task_classification.items()
                },
            }, f, ensure_ascii=False, indent=2)

        logger.info(f"تم حفظ ملفات الجرد في: {self.manifests_dir}")

    def _save_reports(self, report: Dict):
        """حفظ تقارير الاكتشاف"""
        if not self.reports_dir:
            return

        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # تقرير JSON
        json_path = self.reports_dir / "discovery_report.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)

        # تقرير Markdown
        md_path = self.reports_dir / "discovery_report.md"
        self._write_markdown_report(md_path, report)

        logger.info(f"تم حفظ تقارير الاكتشاف في: {self.reports_dir}")

    def _write_markdown_report(self, path: Path, report: Dict):
        """كتابة تقرير الاكتشاف بصيغة Markdown"""
        lines = [
            "# تقرير اكتشاف مجموعات البيانات",
            "",
            f"**التاريخ:** {report['timestamp']}",
            "",
            "## ملخص الاكتشاف",
            "",
            f"- عدد ملفات الصور: **{report['summary']['total_image_files']:,}**",
            f"- عدد ملفات البيانات: **{report['summary']['total_data_files']}**",
            f"- عدد الأرشيفات: **{report['summary']['total_archive_files']}**",
            f"- عدد مجموعات البيانات المكتشفة: **{report['summary']['total_datasets_discovered']}**",
            f"- عدد الأخطاء: **{report['summary']['total_errors']}**",
            "",
            "## مجموعات البيانات المكتشفة",
            "",
        ]

        for ds in report.get("datasets", []):
            lines.append(f"### {ds.get('name', 'غير مسمى')}")
            lines.append(f"- النوع: {ds.get('type', '—')}")
            lines.append(f"- المهمة: {ds.get('task', '—')}")
            lines.append(f"- عدد الفئات: {ds.get('num_classes', '—')}")
            lines.append(f"- عدد الصور: {ds.get('total_images', '—'):,}")
            lines.append("")

        lines.append("## التصنيف حسب المهمة")
        lines.append("")
        for task, count in report.get("task_classification", {}).items():
            lines.append(f"- **{task}**: {count} مجموعة")

        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))


def run_discovery(datasets_dir: Path, manifests_dir: Path,
                  reports_dir: Path) -> Dict[str, Any]:
    """
    نقطة الدخول الرئيسية لاكتشاف البيانات

    المدخلات:
        datasets_dir: مجلد مجموعات البيانات
        manifests_dir: مجلد حفظ الجرد
        reports_dir: مجلد حفظ التقارير

    المخرجات:
        تقرير الاكتشاف
    """
    engine = DatasetDiscoveryEngine(
        search_paths=[datasets_dir],
        manifests_dir=manifests_dir,
        reports_dir=reports_dir,
    )
    return engine.discover_all()
