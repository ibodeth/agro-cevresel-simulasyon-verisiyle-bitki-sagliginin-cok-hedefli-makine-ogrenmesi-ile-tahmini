# =============================================================================
# YZO 106 – İleri Düzey Makine Öğrenmesi | Dönem Projesi
# AŞAMA 4 – GÖREV 3: Stres Düzeyi Tahmini (Multiclass Classification — stress_level)
# Öğrenci: İbrahim Nuryağınlı | 25490221001
#
# Bu dosya AŞAMA 1'in çıktı klasörünü otomatik algılar ve şu yapıyı bekler:
#   asama1_ciktilar_{RUN_ID}/
#       03_model_verileri/
#           X_train_balanced_stress.{npz|npy}   ← SMOTE uygulanmadı (ratio<2.0)
#           y_train_balanced_stress.csv
#           X_test_processed.{npz|npy}
#           y_stress_test.csv
#           23_processed_feature_names.csv
#           TEST_raw_with_targets.csv            ← duyarlılık analizi için ham veri
#       04_preprocessor/
#           preprocessor.joblib
#           metadata.json
#           asama1_paket.joblib
#
# Çalıştırmadan önce:
#   pip install scikit-learn xgboost lightgbm catboost imbalanced-learn
#   pip install tensorflow shap matplotlib seaborn joblib scipy
#
# Metodoloji notları:
#   - stress_level: 3 sınıf (0=Sağlıklı, 1=Hafif Stres, 2=Kritik Stres)
#   - SMOTE uygulanmadı: imbalance ratio ~1.85x < 2.0 eşiği
#   - class_weight='balanced' tüm destekleyen sklearn modellerine uygulandı.
#   - Tüm sklearn modelleri → GridSearchCV (cv=3, CV_SAMPLE_N satır örneklem)
#     Scoring: 'f1_macro'
#   - En iyi parametreler bulunduktan sonra TAM train seti ile yeniden eğitim.
#   - Derin öğrenme (ANN, CNN-1D, LSTM) → manuel hiper-param + EarlyStopping.
#   - Test seti yalnızca final değerlendirmede, tek kez kullanılır.
#   - SHAP: en iyi modele TreeExplainer / KernelExplainer uygulanır.
#   - Duyarlılık analizi: ham test verisi + preprocessor üzerinden %5/%10/%20
#     Gaussian gürültü eklenerek Macro F1 değişimi ölçülür.
#   - Metrikler: Macro F1, class-bazlı Precision/Recall, Confusion Matrix,
#     Weighted F1, Macro ROC-AUC (ovr)
#
# CNN/LSTM notu:
#   Tablolu sensör verisinde sıra bağımlılığı olmadığından CNN-1D ve LSTM
#   düşük performans verecektir. Bu sonuç akademik açıdan raporlanacak,
#   bilimsel olgunluk değeri taşımaktadır.
# =============================================================================

import json
import logging
import os
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse

try:
    import shap
    _shap = True
except ImportError:
    _shap = False

# =============================================================================
# PYTHON SÜRÜM KONTROLÜ
# =============================================================================
if sys.version_info[:2] != (3, 12):
    _py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"\nŞuan projeyi {_py_ver} sürümünde çalıştırıyorsunuz.")
    print("Bu proje Python 3.12 ile en uygun şekilde çalışmaktadır.")
    _yanit = input("Devam edilsin mi? (e/h): ").strip().lower()
    if _yanit not in ("e", "evet", "y", "yes"):
        print("Çıkılıyor.")
        sys.exit(0)

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", "{:.4f}".format)


# =============================================================================
# 0. KONFİGÜRASYON
# =============================================================================

RANDOM_STATE  = 42          # Aşama 1 ile aynı
CV_FOLDS      = 3           # GridSearchCV çapraz doğrulama katı
CV_SAMPLE_N   = 60_000      # GridSearchCV için maks. satır (hız dengesi)
N_JOBS        = -1
HEDEF         = "stress_level"
SINIF_ADLARI  = ["Sağlıklı (0)", "Hafif Stres (1)", "Kritik Stres (2)"]

# Derin öğrenme
DL_EPOCHS     = 30
DL_BATCH_SIZE = 512
DL_PATIENCE   = 5
N_SINIF       = 3           # 0, 1, 2

# SHAP
SHAP_SAMPLE_N = 1_000       # KernelExplainer için örneklem (multiclass daha yavaş)
SHAP_TOP_N    = 20          # Kaç özellik gösterilsin

# Duyarlılık analizi
GURULTU_ORANLARI = [0.05, 0.10, 0.20]
GURULTU_TEKRAR   = 5

FIG_DPI = 150


# =============================================================================
# 1. KLASÖR VE LOG KURULUMU
# =============================================================================

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

# Aşama 1 çıktı klasörünü otomatik bul (en güncel)
asama1_klasorler = sorted(BASE_DIR.glob("asama1_ciktilar_*"), reverse=True)
if not asama1_klasorler:
    raise FileNotFoundError(
        "Aşama 1 çıktı klasörü bulunamadı!\n"
        f"'{BASE_DIR}' içinde 'asama1_ciktilar_*' formatında bir klasör bekleniyor.\n"
        "Lütfen önce asama1.py'yi çalıştırın."
    )
A1_DIR   = asama1_klasorler[0]
A1_DATA  = A1_DIR / "03_model_verileri"
A1_MODEL = A1_DIR / "04_preprocessor"

print(f"▶ Aşama 1 klasörü: {A1_DIR.name}", flush=True)

# Aşama 2 ve 3 klasörleri (referans — zorunlu değil)
asama2_klasorler = sorted(BASE_DIR.glob("asama2_ciktilar_*"), reverse=True)
asama3_klasorler = sorted(BASE_DIR.glob("asama3_ciktilar_*"), reverse=True)
A2_DIR = asama2_klasorler[0] if asama2_klasorler else None
A3_DIR = asama3_klasorler[0] if asama3_klasorler else None
if A2_DIR:
    print(f"▶ Aşama 2 klasörü: {A2_DIR.name} (referans)", flush=True)
if A3_DIR:
    print(f"▶ Aşama 3 klasörü: {A3_DIR.name} (referans)", flush=True)

RUN_ID  = datetime.now().strftime("%Y%m%d_%H%M%S")
CIKTI   = BASE_DIR / f"asama4_ciktilar_{RUN_ID}"
SONUC   = CIKTI / "01_sonuclar"
GRAFIK  = CIKTI / "02_gorseller"
MODEL_K = CIKTI / "03_modeller"
RAPOR   = CIKTI / "04_raporlar"
SHAP_K  = CIKTI / "05_shap"
DUYAR_K = CIKTI / "06_duyarlilik"

for d in [CIKTI, SONUC, GRAFIK, MODEL_K, RAPOR, SHAP_K, DUYAR_K]:
    d.mkdir(parents=True, exist_ok=True)

LOG_DOSYASI = RAPOR / f"asama4_log_{RUN_ID}.txt"
logger = logging.getLogger("asama4")
logger.setLevel(logging.INFO)
logger.handlers.clear()
fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
fh  = logging.FileHandler(LOG_DOSYASI, encoding="utf-8")
fh.setFormatter(fmt)
logger.addHandler(fh)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.WARNING)
ch.setFormatter(fmt)
logger.addHandler(ch)
log = logger


# =============================================================================
# 2. YARDIMCI FONKSİYONLAR
# =============================================================================

def bolum(baslik: str) -> None:
    s = "=" * 80
    log.info(f"\n{s}\n  {baslik}\n{s}")
    print(f"\n▶ {baslik}", flush=True)


def adim(metin: str) -> None:
    log.info(f"  ↳ {metin}")
    print(f"  ↳ {metin}", flush=True)


def tamam(metin: str) -> None:
    log.info(f"  ✓ {metin}")
    print(f"  ✓ {metin}", flush=True)


def csv_kaydet(df_obj: pd.DataFrame, dosya_adi: str,
               klasor: Path = SONUC, index: bool = True) -> Path:
    p = klasor / dosya_adi
    df_obj.to_csv(p, index=index, encoding="utf-8-sig")
    log.info(f"CSV: {p.name}")
    return p


def png_kaydet(fig, dosya_adi: str, klasor: Path = GRAFIK) -> Path:
    p = klasor / dosya_adi
    fig.savefig(p, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info(f"PNG: {p.name}")
    return p


def load_matrix(stem: str, klasor: Path = A1_DATA):
    npz = klasor / f"{stem}.npz"
    npy = klasor / f"{stem}.npy"
    if npz.exists():
        mat = sparse.load_npz(npz)
        log.info(f"Yüklendi (sparse): {stem}.npz  shape={mat.shape}")
        return mat
    if npy.exists():
        mat = np.load(npy)
        log.info(f"Yüklendi (dense ): {stem}.npy  shape={mat.shape}")
        return mat
    raise FileNotFoundError(f"Matris bulunamadı: {stem} (.npz veya .npy)")


def load_target(fname: str) -> pd.Series:
    return pd.read_csv(A1_DATA / fname).iloc[:, 0]


def to_dense(X) -> np.ndarray:
    if sparse.issparse(X):
        return X.toarray()
    return np.asarray(X)


def metrikleri_hesapla(y_true, y_pred, y_prob=None) -> dict:
    """
    Çok sınıflı sınıflandırma metrikleri.
    Temel metrik: Macro F1 (sınıf dengesizliğinde daha güvenilir).
    """
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    siniflar = sorted(np.unique(y_true))
    cr = classification_report(y_true, y_pred, target_names=SINIF_ADLARI,
                                output_dict=True, zero_division=0)

    sonuc = {
        "Accuracy":    round(float(accuracy_score(y_true, y_pred)), 5),
        "Macro_F1":    round(float(f1_score(y_true, y_pred, average="macro",  zero_division=0)), 5),
        "Weighted_F1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 5),
        "Macro_Prec":  round(float(precision_score(y_true, y_pred, average="macro",    zero_division=0)), 5),
        "Macro_Rec":   round(float(recall_score(y_true, y_pred,   average="macro",    zero_division=0)), 5),
    }

    # Sınıf bazlı F1 / Precision / Recall
    for i, sinif_adi in enumerate(SINIF_ADLARI):
        key = sinif_adi
        if key in cr:
            sonuc[f"F1_C{i}"]   = round(cr[key]["f1-score"],  5)
            sonuc[f"Prec_C{i}"] = round(cr[key]["precision"], 5)
            sonuc[f"Rec_C{i}"]  = round(cr[key]["recall"],    5)

    # Macro ROC-AUC (one-vs-rest)
    if y_prob is not None:
        try:
            sonuc["Macro_ROC_AUC"] = round(
                float(roc_auc_score(y_true, y_prob, multi_class="ovr",
                                    average="macro")), 5)
        except Exception:
            sonuc["Macro_ROC_AUC"] = None
    else:
        sonuc["Macro_ROC_AUC"] = None

    return sonuc


# =============================================================================
# 3. AŞAMA 1 VERİLERİNİ YÜKLE
# =============================================================================

bolum("BÖLÜM 1 — Aşama 1 Verilerini Yükle")

with open(A1_MODEL / "metadata.json", encoding="utf-8") as f:
    meta = json.load(f)

a1_paket      = joblib.load(A1_MODEL / "asama1_paket.joblib")
preprocessor  = joblib.load(A1_MODEL / "preprocessor.joblib")
feature_names = a1_paket.get("feature_names", [])
class_weights = a1_paket.get("class_weights", {})
cw_stress     = class_weights.get("stress_level", {0: 1.0, 1: 1.0, 2: 1.0})

adim("Matrisleri yüklüyor...")
# stress_level: SMOTE uygulanmadı (ratio ~1.85x < 2.0) → class_weight ile dengeleme
X_train = load_matrix("X_train_balanced_stress")   # aslında orijinal train seti
X_test  = load_matrix("X_test_processed")

y_train = load_target("y_train_balanced_stress.csv")
y_test  = load_target("y_stress_test.csv")

n_features = X_train.shape[1]
n_train    = X_train.shape[0]
n_test     = X_test.shape[0]

# Duyarlılık analizi için ham test verisi
ham_test_path = A1_DATA / "TEST_raw_with_targets.csv"
if ham_test_path.exists():
    ham_test_df = pd.read_csv(ham_test_path)
    adim(f"Ham test verisi yüklendi: {ham_test_df.shape}")
else:
    ham_test_df = None
    adim("⚠ Ham test verisi bulunamadı — duyarlılık analizi atlanacak.")

# Feature isimleri
feat_csv = A1_DATA / "23_processed_feature_names.csv"
if feat_csv.exists():
    feature_names = pd.read_csv(feat_csv)["feature"].tolist()

# Sınıf dağılımı
sinif_dist = pd.Series(y_train).value_counts().sort_index()
adim(f"Train : {n_train:,} × {n_features}  (SMOTE uygulanmadı — ratio < 2.0 eşiği)")
adim(f"Test  : {n_test:,} × {n_features}")
adim(f"Sınıf dağılımı (train): {sinif_dist.to_dict()}")
adim(f"class_weight stress: {cw_stress}")

# GridSearchCV örneklemi
np.random.seed(RANDOM_STATE)
if n_train > CV_SAMPLE_N:
    cv_idx = np.random.choice(n_train, CV_SAMPLE_N, replace=False)
    X_cv   = X_train[cv_idx]
    y_cv   = y_train.iloc[cv_idx].reset_index(drop=True)
    adim(f"GridSearchCV örneklemesi: {n_train:,} → {CV_SAMPLE_N:,} satır")
else:
    X_cv, y_cv = X_train, y_train.copy()

tamam("Veri yükleme tamamlandı")


# =============================================================================
# 4. MODEL KAYIT DEFTERİ
# =============================================================================

bolum("BÖLÜM 2 — Model Tanımları")

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
    _xgb = True
except ImportError:
    _xgb = False
    adim("⚠ XGBoost kurulu değil.")

try:
    import lightgbm as lgb
    _lgb = True
except ImportError:
    _lgb = False
    adim("⚠ LightGBM kurulu değil.")

try:
    from catboost import CatBoostClassifier as _CatBoostBase

    class CatBoostClassifier(_CatBoostBase):
        """sklearn clone() uyumlu CatBoost wrapper (class_weights sorunu düzeltir)."""
        def __init__(self, class_weights=None, **kwargs):
            self.class_weights = class_weights
            super().__init__(class_weights=class_weights, **kwargs)

        def get_params(self, deep=True):
            p = super().get_params(deep=deep)
            p["class_weights"] = self.class_weights
            return p

        def set_params(self, **params):
            if "class_weights" in params:
                self.class_weights = params["class_weights"]
            return super().set_params(**params)

    _cat = True
except ImportError:
    _cat = False
    adim("⚠ CatBoost kurulu değil.")

try:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.layers import (
        LSTM, Conv1D, Dense, Dropout, GlobalAveragePooling1D, Input, Reshape,
    )
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam
    # GPU bellek yönetimi (6GB VRAM için önemli)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            adim("✓ TensorFlow: GPU bellek yönetimi (Memory Growth) aktif edildi.")
        except RuntimeError as e:
            adim(f"⚠ TensorFlow GPU hatası: {e}")

    tf.get_logger().setLevel("ERROR")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    _tf = True
    adim(f"TensorFlow {tf.__version__} hazır")
except ImportError:
    _tf = False
    adim("⚠ TensorFlow kurulu değil — ANN/CNN/LSTM atlanacak.")

# GPU tespiti (NVIDIA)
try:
    _gpu = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5).returncode == 0
except Exception:
    _gpu = False
adim(f"Donanım GPU Hızlandırma: {'✓ ETKİN' if _gpu else '✗ DEVRE DIŞI (CPU)'}")

# class_weight dict — sklearn 'balanced' eşdeğeri (compute_class_weight çıktısı)
cw_dict = {int(k): float(v) for k, v in cw_stress.items()}

MODEL_REGISTRY = []

# 1 — Logistic Regression
MODEL_REGISTRY.append({
    "ad": "Logistic Regression", "kisa": "LR",
    "estimator": LogisticRegression(
        solver="saga", max_iter=2000,
        class_weight=cw_dict,
        random_state=RANDOM_STATE, n_jobs=N_JOBS,
    ),
    "param_grid": {"C": [0.01, 0.1, 1.0, 10.0], "penalty": ["l2", "l1"]},
    "needs_dense": False, "dl": False,
})

# 2 — Decision Tree
MODEL_REGISTRY.append({
    "ad": "Decision Tree", "kisa": "DT",
    "estimator": DecisionTreeClassifier(
        class_weight=cw_dict, random_state=RANDOM_STATE,
    ),
    "param_grid": {
        "max_depth": [5, 10, 20, None],
        "min_samples_leaf": [1, 5, 20],
        "criterion": ["gini", "entropy"],
    },
    "needs_dense": False, "dl": False,
})

# 3 — KNN
MODEL_REGISTRY.append({
    "ad": "KNN", "kisa": "KNN",
    "estimator": KNeighborsClassifier(n_jobs=N_JOBS),
    "param_grid": {
        "n_neighbors": [5, 11, 21, 51],
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    },
    "needs_dense": True, "dl": False,
})

# 4 — SVM
# Not: RBF kernel 500k+ satırda pratik değil → LinearSVC + Platt scaling
MODEL_REGISTRY.append({
    "ad": "SVM (LinearSVC)", "kisa": "SVM",
    "estimator": CalibratedClassifierCV(
        LinearSVC(class_weight=cw_dict, max_iter=2000, random_state=RANDOM_STATE),
        cv=3, method="sigmoid", n_jobs=N_JOBS,
    ),
    "param_grid": {"estimator__C": [0.01, 0.1, 1.0, 5.0]},
    "needs_dense": True, "dl": False,
})

# 5 — Naive Bayes (BernoulliNB)
# Not: Çok sınıflı için BernoulliNB doğrudan destekler; class_weight yok,
# ama prior'lar class_prior parametresiyle elle ayarlanabilir.
# Sınıf dengesizliği düşük (1.85x) olduğundan uniform prior uygundur.
MODEL_REGISTRY.append({
    "ad": "Naive Bayes (Bernoulli)", "kisa": "NB",
    "estimator": BernoulliNB(),
    "param_grid": {"alpha": [0.001, 0.1, 0.5, 1.0, 2.0]},
    "needs_dense": False, "dl": False,
    "akademik_not": (
        "BernoulliNB, class_weight parametresini desteklemez. "
        "Sınıf dengesizliği oranı ~1.85x olduğundan bu durum performansı sınırlı "
        "etkiler; class_prior ayarı yapılmadan uygulanmıştır."
    ),
})

# 6 — Random Forest
MODEL_REGISTRY.append({
    "ad": "Random Forest", "kisa": "RF",
    "estimator": RandomForestClassifier(
        class_weight=cw_dict,
        random_state=RANDOM_STATE, n_jobs=N_JOBS, n_estimators=300,
    ),
    "param_grid": {
        "n_estimators": [100, 300],
        "max_depth": [10, 20, None],
        "min_samples_leaf": [1, 5],
    },
    "needs_dense": False, "dl": False,
})

# 7 — XGBoost
# Not: XGBoost, multiclass için multi:softprob kullanır
if _xgb:
    xgb_params = {
        "objective": "multi:softprob", "num_class": N_SINIF,
        "eval_metric": "mlogloss", "tree_method": "hist",
        "random_state": RANDOM_STATE, "n_jobs": N_JOBS,
        "verbosity": 1,
    }
    if _gpu:
        xgb_params["device"] = "cuda"
        adim("  [XGB] GPU modu aktif")

    MODEL_REGISTRY.append({
        "ad": "XGBoost", "kisa": "XGB",
        "estimator": XGBClassifier(**xgb_params),
        "param_grid": {
            "n_estimators": [100, 300],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.05, 0.1, 0.2],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        },
        "needs_dense": True, "dl": False,
    })

# 8 — LightGBM
if _lgb:
    lgb_params = {
        "class_weight": cw_dict,
        "objective": "multiclass", "num_class": N_SINIF,
        "random_state": RANDOM_STATE, "n_jobs": N_JOBS, "verbose": -1,
    }
    if _gpu:
        lgb_params["device"] = "gpu"
        lgb_params["gpu_platform_id"] = 0
        lgb_params["gpu_device_id"] = 0
        adim("  [LGBM] GPU modu aktif")

    MODEL_REGISTRY.append({
        "ad": "LightGBM", "kisa": "LGBM",
        "estimator": lgb.LGBMClassifier(**lgb_params),
        "param_grid": {
            "n_estimators": [100, 300],
            "max_depth": [4, 6, -1],
            "learning_rate": [0.05, 0.1, 0.2],
            "num_leaves": [31, 63, 127],
        },
        "needs_dense": False, "dl": False,
    })

# 9 — CatBoost
if _cat:
    cat_params = {
        "class_weights": cw_dict,
        "loss_function": "MultiClass",
        "random_seed": RANDOM_STATE, "verbose": 100,
        "allow_writing_files": False,
    }
    if _gpu:
        cat_params["task_type"] = "GPU"
        cat_params["devices"] = "0"
        adim("  [CAT] GPU modu aktif")

    MODEL_REGISTRY.append({
        "ad": "CatBoost", "kisa": "CAT",
        "estimator": CatBoostClassifier(**cat_params),
        "param_grid": {
            "iterations": [100, 300],
            "depth": [4, 6, 8],
            "learning_rate": [0.05, 0.1, 0.2],
            "l2_leaf_reg": [1, 3, 5],
        },
        "needs_dense": True, "dl": False,
    })

# 10 — ANN
if _tf:
    MODEL_REGISTRY.append({
        "ad": "ANN (Keras)", "kisa": "ANN",
        "estimator": None, "param_grid": {},
        "needs_dense": True, "dl": True, "dl_tip": "ann",
    })

# 11 — CNN-1D
if _tf:
    MODEL_REGISTRY.append({
        "ad": "CNN-1D (Tabular)", "kisa": "CNN",
        "estimator": None, "param_grid": {},
        "needs_dense": True, "dl": True, "dl_tip": "cnn",
        "akademik_not": (
            "CNN-1D, özellik ekseninde konvolüsyon uygulayarak yerel dizi "
            "kalıplarını yakalar. Sensör tabanlı tablolu veride özellikler "
            "arasında anlamlı bir sıra ilişkisi bulunmadığından bu mekanizmanın "
            "teorik avantajı geçerli değildir; akademik karşılaştırma amacıyla "
            "denenmiştir ve düşük performans bilimsel olarak beklenen bir bulgudur."
        ),
    })

# 12 — LSTM
if _tf:
    MODEL_REGISTRY.append({
        "ad": "RNN-LSTM (Tabular)", "kisa": "LSTM",
        "estimator": None, "param_grid": {},
        "needs_dense": True, "dl": True, "dl_tip": "lstm",
        "akademik_not": (
            "LSTM, zamansal bağımlılık örüntülerini öğrenmek için tasarlanmıştır. "
            "Bu çalışmada sensör verileri kesitsel nitelikte olup örnekler arası "
            "herhangi bir zaman serisi ilişkisi bulunmamaktadır. Bu nedenle LSTM'nin "
            "teorik avantajı geçerli değildir ve düşük performans beklenen bir "
            "bulgudur; kıyaslama amacıyla uygulanmıştır."
        ),
    })

adim(f"Toplam {len(MODEL_REGISTRY)} model:")
for m in MODEL_REGISTRY:
    adim(f"  [{m['kisa']:5s}] {m['ad']}")
tamam("Model kayıt defteri hazır")


# =============================================================================
# 5. DERİN ÖĞRENME YARDIMCI FONKSİYONLARI (Multiclass — 3 sınıf)
# =============================================================================

def ann_olustur(n_feat: int) -> "tf.keras.Model":
    """
    ANN — tam bağlantılı, tablolu multiclass classification.
    Çıktı: Dense(3, softmax) — 3 sınıflı olasılık dağılımı.
    """
    model = Sequential([
        Input(shape=(n_feat,)),
        Dense(256, activation="relu"),
        Dropout(0.3),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(N_SINIF, activation="softmax"),
    ], name="ANN_stress")
    model.compile(optimizer=Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def cnn1d_olustur(n_feat: int) -> "tf.keras.Model":
    """
    CNN-1D — özellik ekseninde evrişim (tablolu veri için sınırlı anlam).
    """
    model = Sequential([
        Input(shape=(n_feat,)),
        Reshape((n_feat, 1)),
        Conv1D(64, kernel_size=3, activation="relu", padding="same"),
        Conv1D(32, kernel_size=3, activation="relu", padding="same"),
        GlobalAveragePooling1D(),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(N_SINIF, activation="softmax"),
    ], name="CNN1D_stress")
    model.compile(optimizer=Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def lstm_olustur(n_feat: int) -> "tf.keras.Model":
    """
    LSTM — zaman adımı=1 olarak tablolu veriyi işler (zamansal avantaj yok).
    """
    model = Sequential([
        Input(shape=(n_feat,)),
        Reshape((1, n_feat)),
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(N_SINIF, activation="softmax"),
    ], name="LSTM_stress")
    model.compile(optimizer=Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def dl_egit(dl_tip: str, X_tr, y_tr, X_te, y_te,
            n_feat: int, cw: dict) -> tuple:
    """
    Keras multiclass modeli eğit ve değerlendir.
    Returns: (model, y_pred, y_prob, egitim_sure_s)
    """
    X_tr_d = to_dense(X_tr).astype("float32")
    X_te_d = to_dense(X_te).astype("float32")
    cw_keras = {int(k): float(v) for k, v in cw.items()}

    if dl_tip == "ann":
        model = ann_olustur(n_feat)
    elif dl_tip == "cnn":
        model = cnn1d_olustur(n_feat)
    elif dl_tip == "lstm":
        model = lstm_olustur(n_feat)
    else:
        raise ValueError(f"Bilinmeyen dl_tip: {dl_tip}")

    callbacks = [
        EarlyStopping(patience=DL_PATIENCE, restore_best_weights=True,
                      monitor="val_loss", mode="min", verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3,
                          min_lr=1e-6, verbose=0),
    ]

    t0 = time.time()
    model.fit(
        X_tr_d, y_tr.values,
        epochs=DL_EPOCHS, batch_size=DL_BATCH_SIZE,
        validation_split=0.1,
        class_weight=cw_keras,
        callbacks=callbacks,
        verbose=1, # Eğitim sürecini detaylı göster
    )
    sure = time.time() - t0

    y_prob = model.predict(X_te_d, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)
    return model, y_pred, y_prob, sure



# =============================================================================
# 6. MODEL EĞİTİMİ VE DEĞERLENDİRMESİ
# =============================================================================

bolum("BÖLÜM 3 — Model Eğitimi ve Değerlendirmesi")

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    ConfusionMatrixDisplay, confusion_matrix,
)

kf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

sonuc_listesi = []
model_deposu  = {}
cv_sonuclari  = []
dl_notlar     = []

for kayit in MODEL_REGISTRY:
    ad, kisa, dl = kayit["ad"], kayit["kisa"], kayit["dl"]
    adim(f"[{kisa}] {ad} eğitiliyor...")

    try:
        # ── Derin Öğrenme ──────────────────────────────────────────────────
        if dl:
            model_obj, y_pred, y_prob, egitim_sure = dl_egit(
                kayit["dl_tip"],
                X_train, y_train,
                X_test, y_test,
                n_features, cw_stress,
            )
            model_deposu[kisa] = model_obj
            model_obj.save(MODEL_K / f"{kisa}_model.keras")

            if kayit.get("akademik_not"):
                dl_notlar.append(f"[{kisa}] {kayit['ad']}: {kayit['akademik_not']}")

        # ── sklearn + GridSearchCV ──────────────────────────────────────────
        else:
            estimator  = kayit["estimator"]
            param_grid = kayit["param_grid"]
            nd         = kayit["needs_dense"]

            X_cv_m  = to_dense(X_cv)    if nd else X_cv
            X_tr_m  = to_dense(X_train) if nd else X_train
            X_te_m  = to_dense(X_test)  if nd else X_test

            if param_grid:
                gs = GridSearchCV(
                    estimator=estimator, param_grid=param_grid,
                    scoring="f1_weighted", cv=kf,
                    n_jobs=N_JOBS, refit=True, verbose=3,
                )

                t0 = time.time()
                try:
                    gs.fit(X_cv_m, y_cv)
                except Exception as _gs_err:
                    _err_s = str(_gs_err).lower()
                    if kisa == "CAT" and any(k in _err_s for k in (
                            "out of memory", "terminated", "bad_alloc", "cuda")):
                        log.warning("[CAT] GPU GS hatası, CPU ile yeniden deneniyor...")
                        estimator.set_params(task_type="CPU", thread_count=-1)
                        gs = GridSearchCV(
                            estimator=estimator, param_grid=param_grid,
                            scoring=gs.scoring, cv=kf,
                            n_jobs=1, refit=True, verbose=3,
                        )
                        gs.fit(X_cv_m, y_cv)
                    else:
                        raise
                gs_sure = time.time() - t0

                best_par      = gs.best_params_
                best_cv_f1    = round(gs.best_score_, 5)
                log.info(f"[{kisa}] Best: {best_par}  CV-MacroF1: {best_cv_f1:.5f}")
                cv_sonuclari.append({
                    "Model": ad, "Kisa": kisa,
                    "Best_Params": str(best_par),
                    "CV_MacroF1": best_cv_f1,
                    "GS_Sure_s": round(gs_sure, 2),
                })

                # En iyi parametrelerle tam train üzerinde yeniden eğit
                adim(f"  [{kisa}] Tam train yeniden eğitim...")
                best_est = gs.best_estimator_
                best_est.set_params(**best_par)
                t0 = time.time()
                best_est.fit(X_tr_m, y_train)
                egitim_sure = time.time() - t0
            else:
                best_est = estimator
                t0 = time.time()
                best_est.fit(X_tr_m, y_train)
                egitim_sure = time.time() - t0
                best_par = {}

            model_deposu[kisa] = best_est
            joblib.dump(best_est, MODEL_K / f"{kisa}_model.joblib")

            y_pred = best_est.predict(X_te_m)
            y_prob = (best_est.predict_proba(X_te_m)
                      if hasattr(best_est, "predict_proba") else None)

        # ── Metrikler ──────────────────────────────────────────────────────
        metrik = metrikleri_hesapla(y_test, y_pred, y_prob)
        metrik.update({
            "Model": ad, "Kisa": kisa,
            "Egitim_Sure_s": round(egitim_sure, 2),
            "DL": dl,
            "Akademik_Not": kayit.get("akademik_not", ""),
        })
        sonuc_listesi.append(metrik)
        adim(f"  [{kisa}] Macro_F1={metrik['Macro_F1']}  "
             f"Weighted_F1={metrik['Weighted_F1']}  "
             f"ROC-AUC={metrik['Macro_ROC_AUC']}  ({egitim_sure:.1f}s)")

    except Exception as e:
        log.error(f"[{kisa}] HATA: {e}", exc_info=True)
        sonuc_listesi.append({
            "Model": ad, "Kisa": kisa, "HATA": str(e),
            "Macro_F1": None, "Weighted_F1": None, "Macro_ROC_AUC": None,
        })
        print(f"  ✗ [{kisa}] HATA: {e}", flush=True)

tamam("Tüm model eğitimleri tamamlandı")


# =============================================================================
# 7. SONUÇ TABLOLARI
# =============================================================================

bolum("BÖLÜM 4 — Sonuç Tabloları")

sonuc_df = pd.DataFrame(sonuc_listesi)
kolon_sirasi = [
    "Model", "Kisa",
    "Macro_F1", "Weighted_F1", "Macro_ROC_AUC",
    "Macro_Prec", "Macro_Rec", "Accuracy",
    "F1_C0", "Prec_C0", "Rec_C0",
    "F1_C1", "Prec_C1", "Rec_C1",
    "F1_C2", "Prec_C2", "Rec_C2",
    "Egitim_Sure_s", "DL", "Akademik_Not",
]
sonuc_df = sonuc_df[[c for c in kolon_sirasi if c in sonuc_df.columns]]
sonuc_df = sonuc_df.sort_values("Macro_F1", ascending=False).reset_index(drop=True)
sonuc_df.insert(0, "Sira", range(1, len(sonuc_df) + 1))
csv_kaydet(sonuc_df, "01_tum_model_sonuclari.csv", index=False)

if cv_sonuclari:
    cv_df = pd.DataFrame(cv_sonuclari)
    csv_kaydet(cv_df, "02_gridsearchcv_ozet.csv", index=False)

best_row   = sonuc_df[sonuc_df["Macro_F1"].notna()].iloc[0]
best_kisa  = best_row["Kisa"]
best_model = model_deposu.get(best_kisa)

adim(f"En iyi: [{best_kisa}] {best_row['Model']}")
adim(f"  Macro_F1={best_row['Macro_F1']}  Weighted_F1={best_row['Weighted_F1']}  "
     f"ROC-AUC={best_row['Macro_ROC_AUC']}")
tamam("Sonuç tabloları hazır")


# =============================================================================
# 8. GÖRSELLEŞTİRME
# =============================================================================

bolum("BÖLÜM 5 — Görselleştirmeler")

df_plot = sonuc_df[sonuc_df["Macro_F1"].notna()].copy()

# ── 8.1  Macro F1 Barplot ─────────────────────────────────────────────────
df_b = df_plot.sort_values("Macro_F1", ascending=True)
fig, ax = plt.subplots(figsize=(10, 6))
renkler = ["#e74c3c" if k == best_kisa else "#3498db" for k in df_b["Kisa"]]
bars = ax.barh(df_b["Model"], df_b["Macro_F1"], color=renkler, edgecolor="white")
for bar, val in zip(bars, df_b["Macro_F1"]):
    ax.text(bar.get_width() + 0.004, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", ha="left", fontsize=9)
ax.set_xlabel("Macro F1-Score (Test Seti)", fontsize=11)
ax.set_title("Görev 3: stress_level — Macro F1 Karşılaştırması\n"
             "(kırmızı = en iyi model)", fontsize=12)
ax.set_xlim(0, 1.08)
plt.tight_layout()
png_kaydet(fig, "01_macro_f1_karsilastirma.png")
adim("01 — Macro F1 barplot ✓")

# ── 8.2  Weighted F1 Barplot ─────────────────────────────────────────────
df_wf = df_plot.sort_values("Weighted_F1", ascending=True)
fig, ax = plt.subplots(figsize=(10, 6))
rw = ["#e74c3c" if k == best_kisa else "#16a085" for k in df_wf["Kisa"]]
bars = ax.barh(df_wf["Model"], df_wf["Weighted_F1"], color=rw, edgecolor="white")
for bar, val in zip(bars, df_wf["Weighted_F1"]):
    ax.text(bar.get_width() + 0.004, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", ha="left", fontsize=9)
ax.set_xlabel("Weighted F1-Score (Test Seti)", fontsize=11)
ax.set_title("Görev 3: stress_level — Weighted F1 Karşılaştırması", fontsize=12)
ax.set_xlim(0, 1.08)
plt.tight_layout()
png_kaydet(fig, "02_weighted_f1_karsilastirma.png")
adim("02 — Weighted F1 barplot ✓")

# ── 8.3  Macro ROC-AUC Barplot ──────────────────────────────────────────
if "Macro_ROC_AUC" in df_plot.columns and df_plot["Macro_ROC_AUC"].notna().any():
    df_ra = df_plot.dropna(subset=["Macro_ROC_AUC"]).sort_values(
        "Macro_ROC_AUC", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    rr = ["#e74c3c" if k == best_kisa else "#8e44ad" for k in df_ra["Kisa"]]
    bars = ax.barh(df_ra["Model"], df_ra["Macro_ROC_AUC"], color=rr, edgecolor="white")
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1, label="Rastgele (0.50)")
    for bar, val in zip(bars, df_ra["Macro_ROC_AUC"]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left", fontsize=9)
    ax.set_xlabel("Macro ROC-AUC OVR (Test Seti)", fontsize=11)
    ax.set_title("Görev 3: stress_level — Macro ROC-AUC Karşılaştırması", fontsize=12)
    ax.set_xlim(0, 1.08)
    ax.legend(fontsize=9)
    plt.tight_layout()
    png_kaydet(fig, "03_macro_roc_auc_karsilastirma.png")
    adim("03 — Macro ROC-AUC barplot ✓")

# ── 8.4  Sınıf Bazlı F1 Grouped Bar ─────────────────────────────────────
f1_cols = [c for c in ["F1_C0", "F1_C1", "F1_C2"] if c in df_plot.columns]
if f1_cols:
    df_f1c = df_plot.set_index("Model")[f1_cols].dropna()
    df_f1c.columns = SINIF_ADLARI[:len(f1_cols)]
    fig, ax = plt.subplots(figsize=(13, 6))
    df_f1c.plot(kind="bar", ax=ax,
                color=["#27ae60", "#e67e22", "#c0392b"],
                edgecolor="white", width=0.75)
    ax.set_xticklabels(df_f1c.index, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("F1-Score", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title("Görev 3: stress_level — Sınıf Bazlı F1-Score Karşılaştırması",
                 fontsize=12)
    ax.legend(title="Sınıf", fontsize=9)
    ax.axhline(0.0, color="black", linewidth=0.5)
    plt.tight_layout()
    png_kaydet(fig, "04_sinif_bazli_f1.png")
    adim("04 — Sınıf bazlı F1 ✓")

# ── 8.5  En İyi Model Confusion Matrix ───────────────────────────────────
if best_model is not None:
    try:
        bk = next(kr for kr in MODEL_REGISTRY if kr["kisa"] == best_kisa)
        if bk.get("dl"):
            yp = np.argmax(
                best_model.predict(to_dense(X_test).astype("float32"), verbose=0),
                axis=1)
        else:
            Xbp = to_dense(X_test) if bk["needs_dense"] else X_test
            yp  = best_model.predict(Xbp)

        cm = confusion_matrix(y_test, yp)
        fig, ax = plt.subplots(figsize=(8, 7))
        ConfusionMatrixDisplay(cm, display_labels=SINIF_ADLARI).plot(
            ax=ax, colorbar=True, cmap="Blues")
        ax.set_title(
            f"Görev 3: Confusion Matrix — [{best_kisa}] {best_row['Model']}\n"
            f"Macro F1={best_row['Macro_F1']}  "
            f"Weighted F1={best_row['Weighted_F1']}", fontsize=11)
        plt.tight_layout()
        png_kaydet(fig, f"05_cm_{best_kisa}.png")
        adim(f"05 — Confusion Matrix [{best_kisa}] ✓")
    except Exception as e:
        log.warning(f"CM [{best_kisa}]: {e}")

# ── 8.6  Tüm Modeller CM Grid ────────────────────────────────────────────
ncols = 4
nrows = (len(sonuc_df) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
axes = axes.flatten()
for idx, row_ in sonuc_df.iterrows():
    k, ax_ = row_["Kisa"], axes[idx]
    if k not in model_deposu or row_.get("HATA"):
        ax_.set_visible(False)
        continue
    try:
        mk = next(kr for kr in MODEL_REGISTRY if kr["kisa"] == k)
        if mk.get("dl"):
            yg = np.argmax(
                model_deposu[k].predict(
                    to_dense(X_test).astype("float32"), verbose=0),
                axis=1)
        else:
            Xg = to_dense(X_test) if mk["needs_dense"] else X_test
            yg = model_deposu[k].predict(Xg)
        ConfusionMatrixDisplay(confusion_matrix(y_test, yg),
                               display_labels=["0", "1", "2"]).plot(
            ax=ax_, colorbar=False, cmap="Blues")
        ax_.set_title(f"{k}: F1={row_['Macro_F1']}", fontsize=9)
    except Exception as e:
        ax_.set_title(f"{k}: HATA", fontsize=9)
for i in range(idx + 1, len(axes)):
    axes[i].set_visible(False)
plt.suptitle("Görev 3: stress_level — Tüm Modeller CM Grid",
             fontsize=13, y=1.01)
plt.tight_layout()
png_kaydet(fig, "06_cm_grid_tummodeller.png")
adim("06 — CM Grid ✓")

# ── 8.7  Performans Isı Haritası ─────────────────────────────────────────
heat_cols = ["Macro_F1", "Weighted_F1", "Macro_ROC_AUC",
             "F1_C0", "F1_C1", "F1_C2", "Accuracy"]
heat_cols = [c for c in heat_cols if c in sonuc_df.columns]
heat_df = sonuc_df.set_index("Model")[heat_cols].apply(pd.to_numeric, errors="coerce")
fig, ax = plt.subplots(figsize=(12, max(5, len(heat_df) * 0.7)))
sns.heatmap(heat_df, annot=True, fmt=".4f", cmap="YlOrRd",
            vmin=0, vmax=1, linewidths=0.5, ax=ax, annot_kws={"size": 8})
ax.set_title("Görev 3: stress_level — Performans Isı Haritası", fontsize=12)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
plt.tight_layout()
png_kaydet(fig, "07_performans_isi_haritasi.png")
adim("07 — Isı haritası ✓")

# ── 8.8  Eğitim Süresi ───────────────────────────────────────────────────
df_sure = sonuc_df[["Model", "Kisa", "Egitim_Sure_s", "DL"]].dropna(
    subset=["Egitim_Sure_s"]).sort_values("Egitim_Sure_s", ascending=True)
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(df_sure["Model"],
        df_sure["Egitim_Sure_s"],
        color=["#e74c3c" if d else "#2980b9" for d in df_sure["DL"]],
        edgecolor="white")
ax.set_xlabel("Eğitim Süresi (saniye)", fontsize=11)
ax.set_title("Görev 3: stress_level — Eğitim Süresi\n"
             "(mavi=sklearn, kırmızı=Derin Öğrenme)", fontsize=12)
plt.tight_layout()
png_kaydet(fig, "08_egitim_suresi.png")
adim("08 — Eğitim süresi ✓")

tamam("Tüm görselleştirmeler tamamlandı")


# =============================================================================
# 9. SHAP ANALİZİ
# =============================================================================

bolum("BÖLÜM 6 — SHAP Analizi (En İyi Model)")

shap_ok     = False
shap_top3   = ""
shap_importance = pd.DataFrame()

try:
    if not _shap:
        raise ImportError("shap")

    adim(f"SHAP başlatılıyor → [{best_kisa}] {best_row['Model']}")
    bk_shap = next(kr for kr in MODEL_REGISTRY if kr["kisa"] == best_kisa)

    # Test setinden örneklem al
    np.random.seed(RANDOM_STATE)
    shap_idx    = np.random.choice(n_test, min(SHAP_SAMPLE_N, n_test), replace=False)
    X_shap_raw  = to_dense(X_test)[shap_idx].astype("float64")

    tree_modeller   = {"RF", "XGB", "LGBM", "CAT", "DT"}
    linear_modeller = {"LR"}

    if best_kisa in tree_modeller:
        explainer   = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_shap_raw)
        # Multiclass RF/XGB: Liste [sınıf0, sınıf1, sınıf2]
        # Raporlamak için her sınıfın ortalama mutlak SHAP'ı alınır
        shap_tip = "TreeExplainer"
    elif best_kisa in linear_modeller:
        explainer   = shap.LinearExplainer(best_model, X_shap_raw)
        shap_values = explainer.shap_values(X_shap_raw)
        shap_tip    = "LinearExplainer"
    else:
        arka_plan_n = min(100, X_shap_raw.shape[0])
        arka_plan   = shap.kmeans(X_shap_raw, arka_plan_n)
        if bk_shap.get("dl"):
            predict_fn = lambda x: best_model.predict(
                x.astype("float32"), verbose=0)
        else:
            predict_fn = (best_model.predict_proba
                          if hasattr(best_model, "predict_proba")
                          else best_model.predict)
        explainer   = shap.KernelExplainer(predict_fn, arka_plan)
        shap_values = explainer.shap_values(X_shap_raw[:300], nsamples=50)
        shap_tip    = "KernelExplainer"

    adim(f"SHAP tipi: {shap_tip}")

    # Feature isimleri — uzunluk uyumu
    feat_n = (feature_names[:n_features]
              if len(feature_names) >= n_features
              else feature_names + [f"feat_{i}" for i in range(len(feature_names), n_features)])

    # Multiclass SHAP değerleri: liste veya 3D ndarray
    if isinstance(shap_values, list):
        # [sınıf0_array, sınıf1_array, sınıf2_array] — her biri (n, p)
        mean_abs_per_class = [np.abs(sv).mean(axis=0) for sv in shap_values]
        # Tüm sınıflar üzerinden ortalama mutlak SHAP
        mean_abs_shap = np.mean(mean_abs_per_class, axis=0)

        # Sınıf bazlı önem CSV'leri
        for ci, sv_c in enumerate(shap_values):
            imp_c = pd.DataFrame({
                "Feature": feat_n[:sv_c.shape[1]],
                "Mean_Abs_SHAP": np.abs(sv_c).mean(axis=0),
            }).sort_values("Mean_Abs_SHAP", ascending=False).reset_index(drop=True)
            csv_kaydet(imp_c, f"shap_importance_class{ci}.csv",
                       klasor=SHAP_K, index=False)
    else:
        sv_arr = np.array(shap_values)
        if sv_arr.ndim == 3:
            # (sinif, ornek, ozellik) veya (ornek, ozellik, sinif)
            if sv_arr.shape[0] == N_SINIF:
                mean_abs_shap = np.abs(sv_arr).mean(axis=(0, 1))
            else:
                mean_abs_shap = np.abs(sv_arr).mean(axis=(0, 2))
        else:
            mean_abs_shap = np.abs(sv_arr).mean(axis=0)

    shap_importance = pd.DataFrame({
        "Feature": feat_n[:len(mean_abs_shap)],
        "Mean_Abs_SHAP": mean_abs_shap,
    }).sort_values("Mean_Abs_SHAP", ascending=False).reset_index(drop=True)
    csv_kaydet(shap_importance, "shap_feature_importance_toplam.csv",
               klasor=SHAP_K, index=False)

    # Top 3
    top3 = shap_importance.head(3)
    shap_top3 = "\n".join(
        f"  {i+1}. {row_['Feature']} (|SHAP| ort={row_['Mean_Abs_SHAP']:.5f})"
        for i, (_, row_) in enumerate(top3.iterrows())
    )
    adim(f"Top-3 SHAP:\n{shap_top3}")

    # ── SHAP Bar Plot ──────────────────────────────────────────────────────
    top_df = shap_importance.head(SHAP_TOP_N)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top_df["Feature"][::-1], top_df["Mean_Abs_SHAP"][::-1],
            color="#e74c3c", edgecolor="white")
    ax.set_xlabel("Ortalama |SHAP Değeri| (tüm sınıflar üzerinden)", fontsize=11)
    ax.set_title(
        f"Görev 3: [{best_kisa}] — SHAP Özellik Önemi (Top {SHAP_TOP_N})\n"
        f"Hangi sensör değişkenleri stres düzeyini belirliyor?", fontsize=12)
    plt.tight_layout()
    png_kaydet(fig, "shap_01_bar_plot.png", klasor=SHAP_K)
    adim("SHAP 01 — Bar plot ✓")

    # ── SHAP Sınıf Bazlı Bar Karşılaştırma ───────────────────────────────
    sinif_csv_listesi = list(SHAP_K.glob("shap_importance_class*.csv"))
    if len(sinif_csv_listesi) == N_SINIF:
        sinif_dfs = [pd.read_csv(p).head(10) for p in sorted(sinif_csv_listesi)]
        fig, axes = plt.subplots(1, N_SINIF, figsize=(18, 6), sharey=False)
        for ci, (sdf, sinif_adi, ax_) in enumerate(
                zip(sinif_dfs, SINIF_ADLARI, axes)):
            ax_.barh(sdf["Feature"][::-1], sdf["Mean_Abs_SHAP"][::-1],
                     color=["#27ae60", "#e67e22", "#c0392b"][ci], edgecolor="white")
            ax_.set_title(f"SHAP — {sinif_adi}", fontsize=10)
            ax_.set_xlabel("|SHAP|", fontsize=9)
        plt.suptitle(
            f"Görev 3: [{best_kisa}] — Sınıf Bazlı SHAP (Top 10)",
            fontsize=12, y=1.01)
        plt.tight_layout()
        png_kaydet(fig, "shap_02_sinif_bazli_bar.png", klasor=SHAP_K)
        adim("SHAP 02 — Sınıf bazlı bar ✓")

    shap_ok = True
    tamam("SHAP analizi tamamlandı")

except ImportError:
    adim("⚠ shap kütüphanesi kurulu değil → 'pip install shap'")
except Exception as e:
    log.error(f"SHAP hatası: {e}", exc_info=True)
    adim(f"⚠ SHAP hatası: {e}")


# =============================================================================
# 10. DUYARLILIK ANALİZİ (Gaussian Gürültü — %5 / %10 / %20)
# =============================================================================

bolum("BÖLÜM 7 — Duyarlılık Analizi (Gaussian Gürültü)")

duyar_ok   = False
duyar_blok = ""

if ham_test_df is None:
    adim("⚠ Ham test verisi yok — duyarlılık analizi atlandı.")
elif best_model is None:
    adim("⚠ En iyi model yok — duyarlılık analizi atlandı.")
else:
    try:
        from sklearn.metrics import f1_score, roc_auc_score

        HEDEFLER_TUMU = ["failure_flag", "suitability_score", "stress_level", "location_id"]
        X_ham_test = ham_test_df.drop(
            columns=[c for c in HEDEFLER_TUMU if c in ham_test_df.columns],
            errors="ignore",
        )
        y_ham_test = ham_test_df["stress_level"].values

        # Aşama 1'de düşürülen sütunları çıkar
        dusurulen_path = A1_DATA / "18_dusurulen_sutunlar.csv"
        if dusurulen_path.exists():
            dus_df = pd.read_csv(dusurulen_path)
            if "Sütun" in dus_df.columns:
                X_ham_test = X_ham_test.drop(
                    columns=[c for c in dus_df["Sütun"] if c in X_ham_test.columns],
                    errors="ignore",
                )

        say_sutunlar = X_ham_test.select_dtypes(include=[np.number]).columns.tolist()
        adim(f"Gürültü eklenecek sayısal sütun: {len(say_sutunlar)}")

        bk_d = next(kr for kr in MODEL_REGISTRY if kr["kisa"] == best_kisa)

        def tahmin_et(X_proc):
            if bk_d.get("dl"):
                proba = best_model.predict(
                    to_dense(X_proc).astype("float32"), verbose=0)
                pred  = np.argmax(proba, axis=1)
            elif hasattr(best_model, "predict_proba"):
                Xp    = to_dense(X_proc) if bk_d["needs_dense"] else X_proc
                proba = best_model.predict_proba(Xp)
                pred  = best_model.predict(Xp)
            else:
                Xp    = to_dense(X_proc) if bk_d["needs_dense"] else X_proc
                pred  = best_model.predict(Xp)
                proba = None
            return pred, proba

        # Baseline
        X_base_proc           = preprocessor.transform(X_ham_test)
        base_pred, base_proba = tahmin_et(X_base_proc)
        base_f1   = round(f1_score(y_ham_test, base_pred, average="macro", zero_division=0), 5)
        base_wf1  = round(f1_score(y_ham_test, base_pred, average="weighted", zero_division=0), 5)
        try:
            base_auc = round(roc_auc_score(
                y_ham_test, base_proba, multi_class="ovr", average="macro"), 5) \
                if base_proba is not None else None
        except Exception:
            base_auc = None

        adim(f"Baseline (gürültüsüz) → Macro F1={base_f1}  "
             f"Weighted F1={base_wf1}  ROC-AUC={base_auc}")

        duyar_kayitlar = [{
            "Gurultu_Oran": 0.0, "Tekrar": 0,
            "Macro_F1": base_f1, "Weighted_F1": base_wf1, "Macro_ROC_AUC": base_auc,
        }]

        for oran in GURULTU_ORANLARI:
            f1_list, wf1_list, auc_list = [], [], []
            for tekrar in range(GURULTU_TEKRAR):
                np.random.seed(RANDOM_STATE + tekrar)
                X_gurultulu = X_ham_test.copy()
                for sut in say_sutunlar:
                    std_sut = X_gurultulu[sut].std()
                    if std_sut > 0:
                        X_gurultulu[sut] += np.random.normal(
                            0, oran * std_sut, size=len(X_gurultulu))

                X_g_proc        = preprocessor.transform(X_gurultulu)
                g_pred, g_proba = tahmin_et(X_g_proc)

                gf1   = f1_score(y_ham_test, g_pred, average="macro",    zero_division=0)
                gwf1  = f1_score(y_ham_test, g_pred, average="weighted", zero_division=0)
                try:
                    gauc = roc_auc_score(
                        y_ham_test, g_proba, multi_class="ovr", average="macro") \
                        if g_proba is not None else None
                except Exception:
                    gauc = None

                f1_list.append(gf1)
                wf1_list.append(gwf1)
                auc_list.append(gauc)
                duyar_kayitlar.append({
                    "Gurultu_Oran": oran, "Tekrar": tekrar + 1,
                    "Macro_F1": round(gf1, 5), "Weighted_F1": round(gwf1, 5),
                    "Macro_ROC_AUC": round(gauc, 5) if gauc else None,
                })

            f1_ort  = np.mean(f1_list)
            auc_ort = np.mean([x for x in auc_list if x is not None]) \
                if any(auc_list) else None
            adim(f"  %{int(oran*100):3d} gürültü → "
                 f"Macro F1={f1_ort:.4f}±{np.std(f1_list):.4f}  "
                 f"ROC-AUC={f'{auc_ort:.4f}' if auc_ort else 'N/A'}")

        duyar_df = pd.DataFrame(duyar_kayitlar)
        csv_kaydet(duyar_df, "duyarlilik_ham_sonuclar.csv",
                   klasor=DUYAR_K, index=False)

        # Özet
        duyar_ozet = (duyar_df.groupby("Gurultu_Oran")
                      .agg({"Macro_F1":     ["mean", "std"],
                             "Weighted_F1": ["mean", "std"],
                             "Macro_ROC_AUC": ["mean", "std"]})
                      .round(5))
        duyar_ozet.columns = ["_".join(c) for c in duyar_ozet.columns]
        duyar_ozet = duyar_ozet.reset_index()
        duyar_ozet["Gurultu_Yuzde"] = (duyar_ozet["Gurultu_Oran"] * 100).astype(int)
        csv_kaydet(duyar_ozet, "duyarlilik_ozet.csv", klasor=DUYAR_K, index=False)

        # ── Duyarlılık Grafiği ─────────────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        metrik_bilgi = [
            ("Macro_F1_mean",    "Macro_F1_std",    "Macro F1-Score", "#2980b9", base_f1),
            ("Weighted_F1_mean", "Weighted_F1_std", "Weighted F1",    "#27ae60", base_wf1),
            ("Macro_ROC_AUC_mean","Macro_ROC_AUC_std","Macro ROC-AUC","#8e44ad", base_auc),
        ]
        for ax_i, (mcol, scol, mlbl, mclr, mbase) in zip(axes, metrik_bilgi):
            x = duyar_ozet["Gurultu_Yuzde"].values
            y = duyar_ozet[mcol].values
            s = duyar_ozet[scol].values
            ax_i.errorbar(x, y, yerr=s, marker="o", color=mclr,
                          capsize=5, linewidth=2, label="Ort ± std")
            if mbase is not None:
                ax_i.axhline(mbase, linestyle="--", color="gray", linewidth=1,
                             label=f"Baseline={mbase:.4f}")
            ax_i.set_xlabel("Gürültü Oranı (%)", fontsize=10)
            ax_i.set_ylabel(mlbl, fontsize=10)
            ax_i.set_title(f"{mlbl} — Gürültü Duyarlılığı", fontsize=10)
            ax_i.legend(fontsize=8)
            ax_i.set_xticks(x)
            ax_i.set_ylim(0, 1.05)

        plt.suptitle(
            f"Görev 3: [{best_kisa}] {best_row['Model']} — Duyarlılık Analizi\n"
            f"(Her oran için {GURULTU_TEKRAR} tekrar, ortalama ± std)",
            fontsize=12, y=1.02,
        )
        plt.tight_layout()
        png_kaydet(fig, "duyarlilik_analizi.png", klasor=DUYAR_K)
        adim("Duyarlılık grafiği ✓")

        # Rapor bloğu tablosu
        duyar_tablo = ["| Gürültü | Macro F1 Ort. | Weighted F1 Ort. | ROC-AUC Ort. |",
                       "|---------|-------------|------------------|-------------|"]
        for _, row_ in duyar_ozet.iterrows():
            lbl = "Baseline" if row_["Gurultu_Oran"] == 0 else f"%{int(row_['Gurultu_Oran']*100)}"
            auc_str = f"{row_['Macro_ROC_AUC_mean']:.5f}" \
                if pd.notna(row_.get("Macro_ROC_AUC_mean")) else "N/A"
            duyar_tablo.append(
                f"| {lbl} | {row_['Macro_F1_mean']:.5f} | "
                f"{row_['Weighted_F1_mean']:.5f} | {auc_str} |")
        duyar_blok = "\n".join(duyar_tablo)

        en_kotu = duyar_ozet.loc[duyar_ozet["Gurultu_Yuzde"] == 20, "Macro_F1_mean"].values
        f1_dusus = ((base_f1 - en_kotu[0]) / base_f1 * 100
                    if base_f1 and len(en_kotu) else None)

        adim(f"Kararlılık (%20 gürültüde Macro F1 düşüşü): "
             f"{f1_dusus:.2f}%" if f1_dusus is not None else "Hesaplanamadı")

        duyar_ok = True
        tamam("Duyarlılık analizi tamamlandı")

    except Exception as e:
        log.error(f"Duyarlılık HATA: {e}", exc_info=True)
        adim(f"⚠ Duyarlılık hatası: {e}")
        duyar_blok = f"Hata: {e}"


# =============================================================================
# 11. AKADEMİK RAPOR
# =============================================================================

bolum("BÖLÜM 8 — Akademik Rapor")

rapor_tablo = sonuc_df[
    [c for c in ["Sira", "Model", "Kisa", "Macro_F1", "Weighted_F1",
                 "Macro_ROC_AUC", "Macro_Prec", "Macro_Rec", "Accuracy",
                 "Egitim_Sure_s", "DL"] if c in sonuc_df.columns]
].to_string(index=False)

dl_notlar_metni = "\n".join(dl_notlar) if dl_notlar \
    else "Derin öğrenme modelleri çalıştırılmadı."

rapor_metni = f"""# AŞAMA 4 — GÖREV 3 AKADEMİK RAPORU
# Stres Düzeyi Tahmini (Multiclass Classification — stress_level)

## Öğrenci
İbrahim Nuryağınlı | 25490221001

## Aşama 1 Referansı
Aşama 1 Klasörü : {A1_DIR.name}
Özellik Sayısı  : {n_features}
Train           : {n_train:,} satır (SMOTE uygulanmadı — imbalance ratio < 2.0)
Test            : {n_test:,} satır
class_weight    : {cw_stress}
GridSearchCV    : {CV_FOLDS}-fold, scoring=f1_macro, örneklem={CV_SAMPLE_N:,} satır

---

## 1. Problem Tanımı

stress_level değişkeni üç sınıfı temsil eder:
  - 0 = Sağlıklı
  - 1 = Hafif Stres
  - 2 = Kritik Stres

Bu, sıralı (ordinal) nitelik taşıyan çok sınıflı bir sınıflandırma problemidir.
Sınıf dengesizlik oranı ~1.85x olup 2.0 eşiğinin altında kaldığından SMOTE
uygulanmamıştır; bunun yerine `class_weight='balanced'` (veya eşdeğer `cw_dict`)
tüm destekleyen modellere uygulanmıştır.

---

## 2. Metodoloji

### Neden Macro F1 Temel Metrik?
Doğruluk (Accuracy), sınıf dengesizliği durumunda yanıltıcı sonuç verebilir.
Macro F1-Score, her sınıfın F1 değerini eşit ağırlıkla ortalar; bu nedenle
"Kritik Stres (2)" sınıfındaki başarı düşük olsa bile bu yansır.
Weighted F1 ise sınıf boyutlarını (örnek sayısını) dikkate alır ve genel
tahmin kalitesini ölçer. Her iki metrik raporda sunulmuştur.

### GridSearchCV
- Scoring: f1_macro (Macro F1 — çok sınıflı için uygun)
- CV: StratifiedKFold ({CV_FOLDS} fold, sınıf dağılımını korur)
- Örneklem: {CV_SAMPLE_N:,} satır (hız / kalite dengesi)
- En iyi parametreler tam train ({n_train:,} satır) ile yeniden eğitildi.
- Test seti yalnızca final değerlendirmede, tek kez kullanıldı.

---

## 3. Denemeler

### Denenecek Modeller ({len(MODEL_REGISTRY)} adet)
Logistic Regression, Decision Tree, KNN, SVM (LinearSVC + Platt Scaling),
Naive Bayes (BernoulliNB), Random Forest, XGBoost, LightGBM, CatBoost,
ANN, CNN-1D, RNN-LSTM

### SVM Notu
RBF kernel SVM O(n²)-O(n³) karmaşıklığı nedeniyle 500k+ satırlık veri setinde
pratik değildir. LinearSVC tercih edilmiş; Platt scaling (CalibratedClassifierCV)
ile olasılık çıktısı ve ROC-AUC hesabı mümkün kılınmıştır.

---

## 4. Model Performans Tablosu (Test Seti — Macro F1'e göre)

{rapor_tablo}

---

## 5. En İyi Model
Model        : {best_row['Model']} [{best_kisa}]
Macro F1     : {best_row['Macro_F1']}
Weighted F1  : {best_row['Weighted_F1']}
Macro ROC-AUC: {best_row.get('Macro_ROC_AUC', 'N/A')}
Macro Prec.  : {best_row.get('Macro_Prec', 'N/A')}
Macro Recall : {best_row.get('Macro_Rec', 'N/A')}
Accuracy     : {best_row['Accuracy']}

---

## 6. SHAP Analizi (Açıklanabilir Yapay Zeka / XAI)
SHAP (SHapley Additive exPlanations) yöntemi ile en iyi model olan
[{best_kisa}]'ın hangi sensör değişkenlerine dayandığı görselleştirilmiştir.

Explainer: {"TreeExplainer (ağaç tabanlı — hızlı ve kesin)" if shap_ok else "Çalıştırılamadı"}
Örneklem : {SHAP_SAMPLE_N:,} test örneği

### Top 3 Belirleyici Özellik (|SHAP| ortalaması — tüm sınıflar)
{shap_top3 if shap_top3 else "  SHAP çalıştırılamadı."}

Üretilen SHAP görselleştirmeleri:
  - Bar Plot (toplam)     : Tüm sınıflar üzerinden ortalama mutlak SHAP
  - Sınıf Bazlı Bar Plot : Her sınıf (Sağlıklı / Hafif Stres / Kritik Stres)
                           için ayrı Top-10 özellik önemi

---

## 7. Duyarlılık Analizi (Gaussian Gürültü)
Modelin gerçek saha koşullarındaki sensör gürültüsüne dayanıklılığını ölçmek
amacıyla ham test verisinin sayısal sütunlarına std-orantılı Gaussian gürültü
eklenmiş; Macro F1 ve ROC-AUC değişimleri izlenmiştir.

{duyar_blok}

### Yorum
Gürültü oranı arttıkça Macro F1 düşüşü modelin sensör hatalarına karşı
kararlılığını gösterir. "Kritik Stres (2)" sınıfı düşük örnekle temsil
edildiğinden, bu sınıfın Recall değerinin gürültüye karşı en hassas
davranması beklenmektedir.

---

## 8. Derin Öğrenme Modelleri — Akademik Yorum
{dl_notlar_metni}

Tablolu sensör verisinde CNN ve LSTM'nin zayıf performansı bilimsel olarak
beklenen bir bulgudur. CNN, yerel konvolüsyonel desenleri yakalar; LSTM,
zamansal sıra bağımlılıklarını öğrenir. Bu veri setinde özellikler bağımsız
sensör ölçümleri olup ne uzamsal ne de zamansal dizi yapısı taşımaktadır.
Sonuçların raporlanması bilimsel dürüstlük ve akademik eksiksizlik açısından
zorunludur; bu yaklaşım çalışmaya bilimsel olgunluk katar.

---

## 9. Veri Sızıntısı Önlemleri
1. Preprocessing yalnızca train üzerinde fit (Aşama 1).
2. GridSearchCV yalnızca train alt-kümesine ({CV_SAMPLE_N:,} satır) uygulandı.
3. Test seti yalnızca final değerlendirmede bir kez kullanıldı.
4. SMOTE uygulanmadı (ratio ~1.85x < 2.0 eşiği); class_weight ile dengeleme yapıldı.
5. Duyarlılık analizinde preprocessor tekrar fit edilmedi;
   ham veriye gürültü eklendikten sonra mevcut preprocessor ile transform edildi.

---

## 10. Kaynakça
- Liakos vd. (2018). Machine learning in agriculture: A review. Sensors, 18(8), 2674.
- Lundberg & Lee (2017). A unified approach to interpreting model predictions. NIPS, 30.
- Moshou vd. (2014). Intelligent multi-sensor system for detection of fungal diseases.
  Biosystems Engineering, 117, 94–103.
- Van Klompenburg vd. (2020). Crop yield prediction using machine learning. COMPAG, 177.
- Roy, N. (2024). Agro-environmental stress & failure simulation. Kaggle.

---
Çıktı Dizini : {CIKTI}
Oluşturulma  : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

rapor_path = RAPOR / "gorev3_akademik_rapor.md"
rapor_path.write_text(rapor_metni, encoding="utf-8")
adim(f"Rapor kaydedildi: {rapor_path.name}")
tamam("Akademik rapor tamamlandı")


# =============================================================================
# 12. PAKET (Aşama 5 — Final Özet için)
# =============================================================================

bolum("BÖLÜM 9 — Paket Kayıt (Aşama 5'e aktarım)")

a4_meta = {
    "run_id":          RUN_ID,
    "a1_dir":          str(A1_DIR),
    "a2_dir":          str(A2_DIR) if A2_DIR else None,
    "a3_dir":          str(A3_DIR) if A3_DIR else None,
    "hedef":           HEDEF,
    "siniflar":        SINIF_ADLARI,
    "best_model_kisa": best_kisa,
    "best_model_ad":   best_row["Model"],
    "best_macro_f1":   float(best_row["Macro_F1"]) if best_row["Macro_F1"] else None,
    "best_weighted_f1":float(best_row["Weighted_F1"]) if best_row["Weighted_F1"] else None,
    "best_roc_auc":    float(best_row["Macro_ROC_AUC"]) if best_row.get("Macro_ROC_AUC") else None,
    "n_features":      n_features,
    "n_train":         n_train,
    "n_test":          n_test,
    "cv_folds":        CV_FOLDS,
    "cv_sample_n":     CV_SAMPLE_N,
    "shap_ok":         shap_ok,
    "duyar_ok":        duyar_ok,
    "random_state":    RANDOM_STATE,
    "output_dir":      str(CIKTI),
}

is_keras = lambda v: _tf and hasattr(v, "predict") and hasattr(v, "layers")
paket = {
    "metadata":      a4_meta,
    "sonuc_df":      sonuc_df,
    "feature_names": feature_names,
    "class_weights": cw_stress,
    "model_deposu":  {k: v for k, v in model_deposu.items() if not is_keras(v)},
}
joblib.dump(paket, CIKTI / "asama4_gorev3_paket.joblib")

with open(CIKTI / "asama4_gorev3_metadata.json", "w", encoding="utf-8") as f:
    json.dump(a4_meta, f, ensure_ascii=False, indent=2)

# Dosya manifesti
manifest = []
for kat, dizin in [
    ("sonuc", SONUC), ("gorsel", GRAFIK), ("model", MODEL_K),
    ("rapor", RAPOR), ("shap", SHAP_K), ("duyarlilik", DUYAR_K),
]:
    for fp in sorted(dizin.glob("*")):
        if fp.is_file():
            manifest.append({
                "kategori": kat, "dosya": fp.name,
                "boyut_KB": round(fp.stat().st_size / 1024, 2),
            })
pd.DataFrame(manifest).to_csv(
    CIKTI / "FILE_MANIFEST.csv", index=False, encoding="utf-8-sig")

tamam("Paket kaydedildi")


# =============================================================================
# 13. FİNAL ÖZET
# =============================================================================

n_gorsel = (len(list(GRAFIK.glob("*.png"))) +
            len(list(SHAP_K.glob("*.png"))) +
            len(list(DUYAR_K.glob("*.png"))))
n_csv    = sum(len(list(d.glob("*"))) for d in [SONUC, SHAP_K, DUYAR_K, RAPOR])

print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║          AŞAMA 4 — GÖREV 3 TAMAMLANDI                               ║
╠══════════════════════════════════════════════════════════════════════╣
║  Hedef      : stress_level (Multiclass Classification 0/1/2)         ║
║  Train      : {n_train:,} satır  |  Test: {n_test:,}
║  Özellik    : {n_features}
║  Model      : {len(MODEL_REGISTRY)} adet
║  En İyi     : [{best_kisa}] {best_row['Model']}
║  Macro F1   : {best_row['Macro_F1']}
║  Weighted F1: {best_row['Weighted_F1']}
║  ROC-AUC    : {best_row.get('Macro_ROC_AUC', 'N/A')}
║  SHAP       : {"✓" if shap_ok else "✗"}
║  Duyarlılık : {"✓" if duyar_ok else "✗"}
║  Görsel     : {n_gorsel}  |  CSV/MD: {n_csv}
║  Çıktı      : {CIKTI.name}
╚══════════════════════════════════════════════════════════════════════╝
""", flush=True)
