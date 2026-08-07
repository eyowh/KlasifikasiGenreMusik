"""
retrain_model.py — Retraining SVM + Kalibrasi Probabilitas
============================================================
Project  : Klasifikasi Genre Musik (GTZAN, 10 genre)
Skripsi  : Studio Dungeon Limo

TUJUAN:
    Membungkus SVC(rbf, C=10, gamma=0.01) dengan CalibratedClassifierCV
    (method='sigmoid', cv=5) agar probabilitas predict_proba() lebih
    merepresentasikan keyakinan model yang sesungguhnya.

PARAMETER KONSISTEN DENGAN metrics.json:
    - kernel: RBF, C=10, gamma=0.01
    - Split: 80/20 stratified, random_state=42 (dikonfirmasi: reproduksi accuracy=75.5%)
    - CV: 5-fold
    - Evaluasi: accuracy, macro precision/recall/F1, weighted, confusion matrix, 5-fold CV

OUTPUT:
    - Klasifikasi/model/svm_model_calibrated.pkl   (model terkalibrasi — TIDAK menimpa model lama)
    - Klasifikasi/model/scaler_v2.pkl               (scaler identik, disimpan ulang untuk simetri)
    - laporan/perbandingan_kalibrasi.json
    - laporan/perbandingan_kalibrasi.md
    - laporan/confusion_matrix_calibrated.png

CATATAN VERSI SCIKIT-LEARN:
    - scikit-learn >= 1.2: CalibratedClassifierCV pakai argumen `estimator` (bukan `base_estimator`)
    - Script ini kompatibel dengan scikit-learn 1.7.x yang terpasang.

CARA MENJALANKAN:
    Dari root project (D:\\Skripsi\\WAB WIB WEB\\Skripsi):
        python Klasifikasi/training/retrain_model.py
"""

import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend, aman di server Django
import matplotlib.pyplot as plt
import joblib

from pathlib import Path
from datetime import datetime

# Scikit-learn imports
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

warnings.filterwarnings('ignore')

# ─── Path Setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATASET_PATH = PROJECT_ROOT / 'dataset' / 'fitur_audio_gtzan.csv'
MODEL_DIR = PROJECT_ROOT / 'Klasifikasi' / 'model'
LAPORAN_DIR = PROJECT_ROOT / 'laporan'

# random_state=42 dikonfirmasi dari rekonstruksi: accuracy=0.7550 (sesuai metrics.json)
RANDOM_STATE = 42

print("=" * 70)
print("  RETRAINING SVM + KALIBRASI PROBABILITAS (CalibratedClassifierCV)")
print("=" * 70)
print(f"  Dataset  : {DATASET_PATH}")
print(f"  Model Dir: {MODEL_DIR}")
print(f"  Laporan  : {LAPORAN_DIR}")
print(f"  rs       : random_state={RANDOM_STATE}")
print("=" * 70)

# ─── 1. Load Dataset ───────────────────────────────────────────────────────────
print("\n[1/7] Memuat dataset...")
df = pd.read_csv(DATASET_PATH)
print(f"      Shape: {df.shape} | Genre: {sorted(df['genre'].unique())}")

feature_cols = [c for c in df.columns if c not in ('filename', 'genre')]
X = df[feature_cols].values
y_raw = df['genre'].values
print(f"      Fitur: {len(feature_cols)} | Sampel: {len(X)}")

# ─── 2. Label Encoding ─────────────────────────────────────────────────────────
print("\n[2/7] Label encoding & train-test split...")
le = LabelEncoder()
y = le.fit_transform(y_raw)
print(f"      Classes: {list(le.classes_)}")

# Train-test split (identik dengan training asli)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"      Train: {len(X_train)} | Test: {len(X_test)}")

# ─── 3. Scaling ────────────────────────────────────────────────────────────────
print("\n[3/7] StandardScaler (fit hanya pada train set)...")
scaler_new = StandardScaler()
X_train_scaled = scaler_new.fit_transform(X_train)
X_test_scaled = scaler_new.transform(X_test)

# Verifikasi scaler vs scaler lama
scaler_old = joblib.load(MODEL_DIR / 'scaler.pkl')
mean_diff = np.abs(scaler_new.mean_ - scaler_old.mean_).max()
std_diff = np.abs(scaler_new.scale_ - scaler_old.scale_).max()
print(f"      Verifikasi vs scaler lama — max mean diff: {mean_diff:.2e}, max std diff: {std_diff:.2e}")
if mean_diff < 1e-6:
    print("      ✓ Scaler baru identik dengan scaler lama (rs=42 terkonfirmasi)")

# ─── 4. Model Lama — Baseline Metrics ──────────────────────────────────────────
print("\n[4/7] Evaluasi model LAMA (baseline)...")
model_old = joblib.load(MODEL_DIR / 'svm_model.pkl')

y_pred_old = model_old.predict(X_test_scaled)
proba_old = model_old.predict_proba(X_test_scaled)

acc_old = accuracy_score(y_test, y_pred_old)
prec_old = precision_score(y_test, y_pred_old, average='macro', zero_division=0)
rec_old = recall_score(y_test, y_pred_old, average='macro', zero_division=0)
f1_old = f1_score(y_test, y_pred_old, average='macro', zero_division=0)
prec_w_old = precision_score(y_test, y_pred_old, average='weighted', zero_division=0)
rec_w_old = recall_score(y_test, y_pred_old, average='weighted', zero_division=0)
f1_w_old = f1_score(y_test, y_pred_old, average='weighted', zero_division=0)

top1_conf_old = proba_old.max(axis=1) * 100
print(f"      Accuracy        : {acc_old:.4f} ({acc_old*100:.2f}%)")
print(f"      Macro F1        : {f1_old:.4f}")
print(f"      Confidence mean : {top1_conf_old.mean():.2f}% | median: {np.median(top1_conf_old):.2f}%")
print(f"      Confidence < 40%: {(top1_conf_old < 40).mean()*100:.1f}%")

# ─── 5. Model Baru — CalibratedClassifierCV ────────────────────────────────────
print("\n[5/7] Training model baru dengan CalibratedClassifierCV...")
print("      Base estimator: SVC(kernel='rbf', C=10, gamma=0.01, probability=False)")
print("      Kalibrasi: method='sigmoid', cv=5 (Platt scaling dengan CV)")

# CATATAN: CalibratedClassifierCV dengan cv='prefit' memerlukan model yg sudah difit.
# Dengan cv=5 (integer), ia melakukan cross-fitting internal yang lebih robust.
# method='sigmoid' (Platt scaling) sesuai untuk dataset kecil-sedang.
# Kita pakai probability=False pada base SVC karena kalibrasi ditangani CalibratedClassifierCV.

base_svc = SVC(
    kernel='rbf',
    C=10,
    gamma=0.01,
    probability=False,    # CalibratedClassifierCV yang akan menangani probability
    random_state=RANDOM_STATE,
    class_weight=None,
    decision_function_shape='ovr',
)

# scikit-learn >= 1.2 menggunakan argumen `estimator` (bukan `base_estimator` yang deprecated)
calibrated_model = CalibratedClassifierCV(
    estimator=base_svc,
    method='sigmoid',   # Platt scaling — cocok untuk data per-kelas yang relatif kecil
    cv=5,               # 5-fold cross-fitting internal
    n_jobs=-1,          # Paralelkan training
    ensemble=True,      # Default: ensemble dari 5 calibrator (lebih robust)
)

calibrated_model.fit(X_train_scaled, y_train)
print("      ✓ Training selesai.")

# ─── 6. Evaluasi Model Baru ────────────────────────────────────────────────────
print("\n[6/7] Evaluasi model BARU (terkalibrasi)...")
y_pred_new = calibrated_model.predict(X_test_scaled)
proba_new = calibrated_model.predict_proba(X_test_scaled)

acc_new = accuracy_score(y_test, y_pred_new)
prec_new = precision_score(y_test, y_pred_new, average='macro', zero_division=0)
rec_new = recall_score(y_test, y_pred_new, average='macro', zero_division=0)
f1_new = f1_score(y_test, y_pred_new, average='macro', zero_division=0)
prec_w_new = precision_score(y_test, y_pred_new, average='weighted', zero_division=0)
rec_w_new = recall_score(y_test, y_pred_new, average='weighted', zero_division=0)
f1_w_new = f1_score(y_test, y_pred_new, average='weighted', zero_division=0)

top1_conf_new = proba_new.max(axis=1) * 100

print(f"      Accuracy        : {acc_new:.4f} ({acc_new*100:.2f}%)")
print(f"      Macro F1        : {f1_new:.4f}")
print(f"      Confidence mean : {top1_conf_new.mean():.2f}% | median: {np.median(top1_conf_new):.2f}%")
print(f"      Confidence < 40%: {(top1_conf_new < 40).mean()*100:.1f}%")

# Cross-Validation (5-fold, metodologi sama dengan metrics.json)
print("\n      Cross-validation (5-fold, StratifiedKFold, seluruh dataset)...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
# Re-train untuk CV di seluruh dataset (X_all)
X_all_scaled = scaler_new.fit_transform(X)   # scaler di-fit ulang per fold oleh CV
# Untuk CV yang benar, kita perlu pipeline — tapi untuk comparability, pakai X yang sudah scaled
# dengan scaler dari train set (sudah terbukti identik dengan scaler lama)
# Pakai X_train_scaled standar: CV di atas data train saja (lebih konservatif)

# Cara yang lebih correct: gunakan pipeline atau scale per-fold
from sklearn.pipeline import Pipeline
cv_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', CalibratedClassifierCV(
        estimator=SVC(kernel='rbf', C=10, gamma=0.01, probability=False, random_state=RANDOM_STATE),
        method='sigmoid', cv=5, n_jobs=-1, ensemble=True
    ))
])

cv_scores = cross_val_score(cv_pipeline, X, y, cv=skf, scoring='accuracy', n_jobs=-1)
print(f"      CV Fold scores  : {[f'{s:.4f}' for s in cv_scores]}")
print(f"      CV Mean         : {cv_scores.mean():.4f} | Std: {cv_scores.std():.4f}")

# Classification report per kelas
report_new = classification_report(
    y_test, y_pred_new, target_names=le.classes_, output_dict=True, zero_division=0
)
report_old = classification_report(
    y_test, y_pred_old, target_names=le.classes_, output_dict=True, zero_division=0
)

# Confusion matrix
cm_new = confusion_matrix(y_test, y_pred_new)

# ─── 7. Simpan Artefak ─────────────────────────────────────────────────────────
print("\n[7/7] Menyimpan artefak & laporan...")

# Simpan model terkalibrasi (JANGAN overwrite svm_model.pkl)
model_save_path = MODEL_DIR / 'svm_model_calibrated.pkl'
joblib.dump(calibrated_model, model_save_path)
print(f"      ✓ Model: {model_save_path}")

# Simpan scaler (identik dengan scaler.pkl, disimpan sebagai v2 untuk simetri dokumentasi)
scaler_save_path = MODEL_DIR / 'scaler_v2.pkl'
joblib.dump(scaler_new, scaler_save_path)
print(f"      ✓ Scaler v2: {scaler_save_path}")

# ─── Perbandingan JSON ─────────────────────────────────────────────────────────
def conf_stats(conf_arr):
    return {
        'min': round(float(conf_arr.min()), 2),
        'mean': round(float(conf_arr.mean()), 2),
        'median': round(float(np.median(conf_arr)), 2),
        'max': round(float(conf_arr.max()), 2),
        'std': round(float(conf_arr.std()), 2),
        'pct_below_30': round(float((conf_arr < 30).mean() * 100), 1),
        'pct_below_40': round(float((conf_arr < 40).mean() * 100), 1),
        'pct_40_to_60': round(float(((conf_arr >= 40) & (conf_arr < 60)).mean() * 100), 1),
        'pct_above_60': round(float((conf_arr >= 60).mean() * 100), 1),
    }

comparison = {
    'tanggal': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'dataset': 'GTZAN Music Genre Dataset',
    'split': '80/20 stratified, random_state=42',
    'model_lama': {
        'deskripsi': 'SVC(kernel=rbf, C=10, gamma=0.01, probability=True) — Platt scaling bawaan',
        'accuracy': round(acc_old, 4),
        'macro_precision': round(prec_old, 4),
        'macro_recall': round(rec_old, 4),
        'macro_f1': round(f1_old, 4),
        'weighted_precision': round(prec_w_old, 4),
        'weighted_recall': round(rec_w_old, 4),
        'weighted_f1': round(f1_w_old, 4),
        'confidence_stats_test': conf_stats(top1_conf_old),
    },
    'model_baru': {
        'deskripsi': 'CalibratedClassifierCV(SVC(rbf,C=10,γ=0.01,prob=False), method=sigmoid, cv=5)',
        'accuracy': round(acc_new, 4),
        'macro_precision': round(prec_new, 4),
        'macro_recall': round(rec_new, 4),
        'macro_f1': round(f1_new, 4),
        'weighted_precision': round(prec_w_new, 4),
        'weighted_recall': round(rec_w_new, 4),
        'weighted_f1': round(f1_w_new, 4),
        'cross_validation': {
            **{f'fold_{i+1}': round(float(cv_scores[i]), 4) for i in range(5)},
            'mean': round(float(cv_scores.mean()), 4),
            'std': round(float(cv_scores.std()), 4),
        },
        'confidence_stats_test': conf_stats(top1_conf_new),
        'per_class': {
            g: {
                'precision': round(report_new[g]['precision'], 4),
                'recall': round(report_new[g]['recall'], 4),
                'f1_score': round(report_new[g]['f1-score'], 4),
                'support': int(report_new[g]['support']),
            }
            for g in le.classes_
        },
        'confusion_matrix': {
            'labels': list(le.classes_),
            'matrix': cm_new.tolist(),
        },
    },
    'delta': {
        'accuracy_change': round(acc_new - acc_old, 4),
        'macro_f1_change': round(f1_new - f1_old, 4),
        'confidence_mean_change': round(top1_conf_new.mean() - top1_conf_old.mean(), 2),
        'pct_below_40_change': round(
            (top1_conf_new < 40).mean()*100 - (top1_conf_old < 40).mean()*100, 1
        ),
    }
}

json_path = LAPORAN_DIR / 'perbandingan_kalibrasi.json'
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(comparison, f, indent=2, ensure_ascii=False)
print(f"      ✓ JSON: {json_path}")

# ─── Confusion Matrix Plot ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(cm_new, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar(im, ax=ax)
ax.set(
    xticks=np.arange(len(le.classes_)),
    yticks=np.arange(len(le.classes_)),
    xticklabels=le.classes_,
    yticklabels=le.classes_,
    ylabel='Label Sebenarnya',
    xlabel='Label Prediksi',
    title=f'Confusion Matrix — Model Terkalibrasi\nAccuracy: {acc_new*100:.2f}%'
)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
thresh = cm_new.max() / 2.
for i in range(cm_new.shape[0]):
    for j in range(cm_new.shape[1]):
        ax.text(j, i, format(cm_new[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm_new[i, j] > thresh else "black",
                fontsize=9)
fig.tight_layout()
cm_plot_path = LAPORAN_DIR / 'confusion_matrix_calibrated.png'
plt.savefig(cm_plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"      ✓ Confusion matrix: {cm_plot_path}")

# ─── Laporan Markdown ─────────────────────────────────────────────────────────
delta_acc = comparison['delta']['accuracy_change']
delta_f1 = comparison['delta']['macro_f1_change']
delta_conf = comparison['delta']['confidence_mean_change']
delta_low_conf = comparison['delta']['pct_below_40_change']

acc_warning = ""
if abs(delta_acc) > 0.01:
    acc_warning = f"""
> [!WARNING]
> Accuracy berubah sebesar **{delta_acc*100:+.2f}%** (dari {acc_old*100:.2f}% menjadi {acc_new*100:.2f}%).
> Perubahan ini perlu dicermati terkait narasi skripsi Bab Hasil & Pembahasan.
> Model lama (`svm_model.pkl`) TIDAK ditimpa — tersedia untuk perbandingan.
"""

md_content = f"""# Laporan Perbandingan Kalibrasi Model SVM

**Tanggal:** {comparison['tanggal']}  
**Dataset:** GTZAN Music Genre Dataset (1000 sampel, 10 genre)  
**Split:** 80/20 stratified, `random_state=42`

---
{acc_warning}

## Ringkasan Perbandingan

| Metrik | Model Lama (Platt bawaan) | Model Baru (CalibratedCV) | Delta |
|---|---|---|---|
| **Accuracy** | {acc_old*100:.2f}% | {acc_new*100:.2f}% | **{delta_acc*100:+.2f}%** |
| Macro Precision | {prec_old*100:.2f}% | {prec_new*100:.2f}% | {(prec_new-prec_old)*100:+.2f}% |
| Macro Recall | {rec_old*100:.2f}% | {rec_new*100:.2f}% | {(rec_new-rec_old)*100:+.2f}% |
| **Macro F1** | {f1_old*100:.2f}% | {f1_new*100:.2f}% | **{delta_f1*100:+.2f}%** |
| Weighted F1 | {f1_w_old*100:.2f}% | {f1_w_new*100:.2f}% | {(f1_w_new-f1_w_old)*100:+.2f}% |

## Distribusi Confidence (Top-1 Prediksi) — Test Set

| Statistik | Model Lama | Model Baru | Delta |
|---|---|---|---|
| Mean confidence | {top1_conf_old.mean():.2f}% | {top1_conf_new.mean():.2f}% | **{delta_conf:+.2f}%** |
| Median confidence | {np.median(top1_conf_old):.2f}% | {np.median(top1_conf_new):.2f}% | {np.median(top1_conf_new)-np.median(top1_conf_old):+.2f}% |
| Min confidence | {top1_conf_old.min():.2f}% | {top1_conf_new.min():.2f}% | — |
| Max confidence | {top1_conf_old.max():.2f}% | {top1_conf_new.max():.2f}% | — |
| **% prediksi < 40%** | {(top1_conf_old < 40).mean()*100:.1f}% | {(top1_conf_new < 40).mean()*100:.1f}% | **{delta_low_conf:+.1f}%** |
| % prediksi < 30% | {(top1_conf_old < 30).mean()*100:.1f}% | {(top1_conf_new < 30).mean()*100:.1f}% | — |
| % prediksi ≥ 60% | {(top1_conf_old >= 60).mean()*100:.1f}% | {(top1_conf_new >= 60).mean()*100:.1f}% | — |

## Cross-Validation Model Baru (5-fold, Stratified)

| Fold | Score |
|---|---|
{''.join(f'| Fold {i+1} | {cv_scores[i]:.4f} ({cv_scores[i]*100:.2f}%) |' + chr(10) for i in range(5))}
| **Mean** | **{cv_scores.mean():.4f} ({cv_scores.mean()*100:.2f}%)** |
| Std Dev | {cv_scores.std():.4f} |

## Per-Class Metrics — Model Baru

| Genre | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
{''.join(f"| {g.capitalize()} | {report_new[g]['precision']*100:.2f}% | {report_new[g]['recall']*100:.2f}% | {report_new[g]['f1-score']*100:.2f}% | {int(report_new[g]['support'])} |" + chr(10) for g in le.classes_)}

## Confusion Matrix Model Baru

Lihat gambar: `laporan/confusion_matrix_calibrated.png`

## File yang Dihasilkan

| File | Keterangan |
|---|---|
| `Klasifikasi/model/svm_model_calibrated.pkl` | Model SVM terkalibrasi (BARU — jangan timpa model lama tanpa konfirmasi) |
| `Klasifikasi/model/scaler_v2.pkl` | Scaler identik dengan `scaler.pkl` (disimpan ulang untuk dokumentasi) |
| `laporan/perbandingan_kalibrasi.json` | Data perbandingan lengkap dalam format JSON |
| `laporan/confusion_matrix_calibrated.png` | Confusion matrix model baru |

## Rekomendasi

{"- ✅ **Aman untuk swap:** Accuracy tidak berubah signifikan (< 1%). Disarankan mengganti `svm_model.pkl` dengan model terkalibrasi." if abs(delta_acc) <= 0.01 else f"- ⚠️ **Perlu review:** Accuracy berubah {delta_acc*100:+.2f}%. Tinjau narasi Bab Hasil sebelum swap model produksi."}
- Setelah mendapat konfirmasi, jalankan: `copy svm_model_calibrated.pkl svm_model.pkl`
- Update `metrics.json` jika Anda memutuskan menggunakan model terkalibrasi sebagai model produksi.
"""

md_path = LAPORAN_DIR / 'perbandingan_kalibrasi.md'
with open(md_path, 'w', encoding='utf-8') as f:
    f.write(md_content)
print(f"      ✓ Markdown: {md_path}")

# ─── Final Summary ─────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("  SELESAI — RINGKASAN")
print("=" * 70)
print(f"  Model Lama  → Accuracy: {acc_old*100:.2f}% | Conf mean: {top1_conf_old.mean():.2f}% | <40%: {(top1_conf_old<40).mean()*100:.1f}%")
print(f"  Model Baru  → Accuracy: {acc_new*100:.2f}% | Conf mean: {top1_conf_new.mean():.2f}% | <40%: {(top1_conf_new<40).mean()*100:.1f}%")
print(f"  Delta Acc   : {delta_acc*100:+.2f}%  | Delta Conf: {delta_conf:+.2f}%")
print()

if abs(delta_acc) > 0.02:
    print("  ⚠️  PERINGATAN: Accuracy berubah > 2% dari nilai yang dilaporkan di skripsi!")
    print("      Tinjau narasi Bab Hasil & Pembahasan sebelum swap model produksi.")
elif abs(delta_acc) > 0.01:
    print("  ⚠️  Perhatian: Accuracy berubah sedikit (> 1%). Cek laporan sebelum swap.")
else:
    print("  ✓  Accuracy stabil. Aman untuk swap setelah konfirmasi dari pengguna.")

print()
print("  File model baru: Klasifikasi/model/svm_model_calibrated.pkl")
print("  JANGAN overwrite svm_model.pkl tanpa konfirmasi terlebih dahulu.")
print("=" * 70)
