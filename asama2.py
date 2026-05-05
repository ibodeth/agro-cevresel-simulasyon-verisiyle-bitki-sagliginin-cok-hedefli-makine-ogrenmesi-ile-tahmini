# =============================================================================
# YZO 106 – İleri Düzey Makine Öğrenmesi | Dönem Projesi
# AŞAMA 2 – GÖREV 1: Başarısızlık Tahmini (Binary Classification — failure_flag)
# Öğrenci: İbrahim Nuryağınlı | 25490221001
#
# Bu dosya AŞAMA 1'in çıktı klasörünü otomatik algılar ve şu yapıyı bekler:
#   asama1_ciktilar_{RUN_ID}/
#       03_model_verileri/
#           X_train_balanced_failure.{npz|npy}
#           y_train_balanced_failure.csv
#           X_test_processed.{npz|npy}
#           y_failure_test.csv
#           23_processed_feature_names.csv
#           TRAIN_raw_with_targets.csv     ← duyarlılık analizi için ham veri
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
#   - Tüm sklearn modelleri → GridSearchCV (cv=3, CV_SAMPLE_N satır örneklem)
#   - GridSearchCV yalnızca train setine uygulanır; test seti hiç görülmez.
#   - En iyi parametreler bulunduktan sonra TAM train seti ile yeniden eğitim.
#   - Derin öğrenme (ANN, CNN-1D, LSTM) → manuel hiper-param + EarlyStopping.
#   - Test seti yalnızca final değerlendirmede, tek kez kullanılır.
#   - SHAP: en iyi modele TreeExplainer / KernelExplainer uygulanır.
#   - Duyarlılık analizi: ham test verisi + preprocessor üzerinden %5/%10/%20
#     Gaussian gürültü eklenerek metrik değişimi ölçülür.
#   - Metrikler: ROC-AUC, F1, PR-AUC, Precision, Recall, Confusion Matrix
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
HEDEF         = "failure_flag"

# Derin öğrenme
DL_EPOCHS     = 30
DL_BATCH_SIZE = 512
DL_PATIENCE   = 5

# SHAP
SHAP_SAMPLE_N     = 2_000   # KernelExplainer için örneklem (hız)
SHAP_TOP_N        = 20      # Kaç özellik gösterilsin

# Duyarlılık analizi
GURULTU_ORANLARI  = [0.05, 0.10, 0.20]   # %5, %10, %20 Gaussian gürültü
GURULTU_TEKRAR    = 5                     # Her oran için kaç tekrar

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

RUN_ID  = datetime.now().strftime("%Y%m%d_%H%M%S")
CIKTI   = BASE_DIR / f"asama2_ciktilar_{RUN_ID}"
SONUC   = CIKTI / "01_sonuclar"
GRAFIK  = CIKTI / "02_gorseller"
MODEL_K = CIKTI / "03_modeller"
RAPOR   = CIKTI / "04_raporlar"
SHAP_K  = CIKTI / "05_shap"
DUYAR_K = CIKTI / "06_duyarlilik"

for d in [CIKTI, SONUC, GRAFIK, MODEL_K, RAPOR, SHAP_K, DUYAR_K]:
    d.mkdir(parents=True, exist_ok=True)

LOG_DOSYASI = RAPOR / f"asama2_log_{RUN_ID}.txt"
logger = logging.getLogger("asama2")
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
    from sklearn.metrics import (
        accuracy_score, average_precision_score, confusion_matrix,
        f1_score, precision_score, recall_score, roc_auc_score,
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sonuc = {
        "Accuracy":  round(float(accuracy_score(y_true, y_pred)), 5),
        "F1":        round(float(f1_score(y_true, y_pred, zero_division=0)), 5),
        "Precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 5),
        "Recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 5),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }
    if y_prob is not None:
        sonuc["ROC_AUC"] = round(float(roc_auc_score(y_true, y_prob)), 5)
        sonuc["PR_AUC"]  = round(float(average_precision_score(y_true, y_prob)), 5)
    else:
        sonuc["ROC_AUC"] = None
        sonuc["PR_AUC"]  = None
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
cw_failure    = class_weights.get("failure_flag", {0: 1.0, 1: 1.0})

adim("Matrisleri yüklüyor...")
X_train = load_matrix("X_train_balanced_failure")   # SMOTE sonrası
X_test  = load_matrix("X_test_processed")

y_train = load_target("y_train_balanced_failure.csv")
y_test  = load_target("y_failure_test.csv")

n_features = X_train.shape[1]
n_train    = X_train.shape[0]
n_test     = X_test.shape[0]

# Duyarlılık analizi için ham test verisi (preprocessor ile yeniden işlenecek)
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

adim(f"Train : {n_train:,} × {n_features}  (SMOTE sonrası)")
adim(f"Test  : {n_test:,} × {n_features}")
adim(f"class_weight failure: {cw_failure}")

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

MODEL_REGISTRY = []

# 1 — Logistic Regression
MODEL_REGISTRY.append({
    "ad": "Logistic Regression", "kisa": "LR",
    "estimator": LogisticRegression(
        solver="saga", max_iter=2000,
        class_weight=cw_failure,
        random_state=RANDOM_STATE, n_jobs=1,
    ),
    "param_grid": {"C": [0.01, 0.1, 1.0, 10.0], "penalty": ["l2", "l1"]},
    "needs_dense": False, "dl": False,
})

# 2 — Decision Tree
MODEL_REGISTRY.append({
    "ad": "Decision Tree", "kisa": "DT",
    "estimator": DecisionTreeClassifier(
        class_weight=cw_failure, random_state=RANDOM_STATE,
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
        LinearSVC(class_weight=cw_failure, max_iter=2000,
                  random_state=RANDOM_STATE),
        cv=3, method="sigmoid", n_jobs=N_JOBS,
    ),
    "param_grid": {"estimator__C": [0.01, 0.1, 1.0, 5.0]},
    "needs_dense": True, "dl": False,
})

# 5 — Naive Bayes
# Not: OHE sonrası binary özellikler → BernoulliNB uygundur
MODEL_REGISTRY.append({
    "ad": "Naive Bayes (Bernoulli)", "kisa": "NB",
    "estimator": BernoulliNB(),
    "param_grid": {"alpha": [0.001, 0.1, 0.5, 1.0, 2.0]},
    "needs_dense": False, "dl": False,
})

# 6 — Random Forest
MODEL_REGISTRY.append({
    "ad": "Random Forest", "kisa": "RF",
    "estimator": RandomForestClassifier(
        class_weight=cw_failure,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS, n_estimators=300,
    ),
    "param_grid": {
        "n_estimators": [100, 300],
        "max_depth": [10, 20, None],
        "min_samples_leaf": [1, 5],
    },
    "needs_dense": False, "dl": False,
})

# 7 — XGBoost
if _xgb:
    spw = float((y_cv == 0).sum()) / max(float((y_cv == 1).sum()), 1)
    xgb_params = {
        "scale_pos_weight": spw, "eval_metric": "logloss",
        "tree_method": "hist", "random_state": RANDOM_STATE,
        "n_jobs": 1 if _gpu else N_JOBS,
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
        "class_weight": cw_failure,
        "random_state": RANDOM_STATE,
        "n_jobs": 1 if _gpu else N_JOBS, "verbose": -1,
        "min_child_samples": 20,
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
        "class_weights": [cw_failure[0], cw_failure[1]],
        "random_seed": RANDOM_STATE,
        "verbose": 100,
        "allow_writing_files": False,
        "thread_count": 1 if _gpu else -1,
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
            "CNN-1D tablolu veride özellik sırası bağımlılığı varsayar. "
            "Bu veri setinde sensör özellikleri bağımsız olduğundan düşük performans "
            "beklenmektedir; akademik karşılaştırma amacıyla eklenmiştir."
        ),
    })

# 12 — LSTM
if _tf:
    MODEL_REGISTRY.append({
        "ad": "RNN-LSTM (Tabular)", "kisa": "LSTM",
        "estimator": None, "param_grid": {},
        "needs_dense": True, "dl": True, "dl_tip": "lstm",
        "akademik_not": (
            "LSTM zamansal dizi bağımlılığı için tasarlanmıştır. "
            "Sensör verisinde herhangi bir zamansal sıra ilişkisi bulunmadığından "
            "teorik avantajı geçerli değildir; kıyaslama amacıyla uygulanmıştır."
        ),
    })

adim(f"Toplam {len(MODEL_REGISTRY)} model:")
for m in MODEL_REGISTRY:
    adim(f"  [{m['kisa']:5s}] {m['ad']}")
tamam("Model kayıt defteri hazır")


# =============================================================================
# 5. DERİN ÖĞRENME YARDIMCI FONKSİYONLARI
# =============================================================================

def ann_olustur(n_feat: int) -> "tf.keras.Model":
    """
    ANN — tam bağlantılı, tablolu binary classification.
    Mimari: Input → Dense(256,relu) → Dropout(0.3)
                  → Dense(128,relu) → Dropout(0.3)
                  → Dense(64,relu)  → Dropout(0.2)
                  → Dense(1,sigmoid)
    """
    model = Sequential([
        Input(shape=(n_feat,)),
        Dense(256, activation="relu"),
        Dropout(0.3),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid"),
    ], name="ANN_failure")
    model.compile(optimizer=Adam(1e-3), loss="binary_crossentropy",
                  metrics=["AUC"])
    return model


def cnn1d_olustur(n_feat: int) -> "tf.keras.Model":
    """
    CNN-1D — özellik ekseninde evrişim (tablolu veri için sınırlı anlam).
    Girdi: (batch, n_feat) → Reshape(n_feat,1) → Conv1D → GAP → Dense → Output
    """
    model = Sequential([
        Input(shape=(n_feat,)),
        Reshape((n_feat, 1)),
        Conv1D(64, kernel_size=3, activation="relu", padding="same"),
        Conv1D(32, kernel_size=3, activation="relu", padding="same"),
        GlobalAveragePooling1D(),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid"),
    ], name="CNN1D_failure")
    model.compile(optimizer=Adam(1e-3), loss="binary_crossentropy",
                  metrics=["AUC"])
    return model


def lstm_olustur(n_feat: int) -> "tf.keras.Model":
    """
    LSTM — zaman adımı=1 olarak tablolu veriyi işler (zamansal avantaj yok).
    Girdi: (batch, n_feat) → Reshape(1, n_feat) → LSTM(128) → LSTM(64) → Output
    """
    model = Sequential([
        Input(shape=(n_feat,)),
        Reshape((1, n_feat)),
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ], name="LSTM_failure")
    model.compile(optimizer=Adam(1e-3), loss="binary_crossentropy",
                  metrics=["AUC"])
    return model


def dl_egit(dl_tip: str, X_tr, y_tr, X_te, y_te,
            n_feat: int, cw: dict) -> tuple:
    """
    Keras modeli eğit ve değerlendir.
    Returns: (model, y_pred, y_prob, egitim_sure_s)
    """
    X_tr_d = to_dense(X_tr).astype("float32")
    X_te_d = to_dense(X_te).astype("float32")
    cw_dict = {int(k): float(v) for k, v in cw.items()}

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
                      monitor="val_auc", mode="max", verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3,
                          min_lr=1e-6, verbose=0),
    ]

    t0 = time.time()
    model.fit(
        X_tr_d, y_tr.values,
        epochs=DL_EPOCHS, batch_size=DL_BATCH_SIZE,
        validation_split=0.1,
        class_weight=cw_dict,
        callbacks=callbacks,
        verbose=1, # Eğitim sürecini detaylı göster
    )
    sure = time.time() - t0

    y_prob = model.predict(X_te_d, verbose=1).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    return model, y_pred, y_prob, sure


# =============================================================================
# 6. MODEL EĞİTİMİ VE DEĞERLENDİRMESİ
# =============================================================================

bolum("BÖLÜM 3 — Model Eğitimi ve Değerlendirmesi")

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    ConfusionMatrixDisplay, confusion_matrix,
    precision_recall_curve, roc_curve,
)

kf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

sonuc_listesi = []
model_deposu  = {}
cv_sonuclari  = []

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
                n_features, cw_failure,
            )
            model_deposu[kisa] = model_obj
            model_obj.save(MODEL_K / f"{kisa}_model.keras")

        # ── sklearn + GridSearchCV ──────────────────────────────────────────
        else:
            estimator  = kayit["estimator"]
            param_grid = kayit["param_grid"]
            nd         = kayit["needs_dense"]

            X_cv_m = to_dense(X_cv)   if nd else X_cv
            X_tr_m = to_dense(X_train) if nd else X_train
            X_te_m = to_dense(X_test)  if nd else X_test

            if param_grid:
                gs = GridSearchCV(
                    estimator=estimator, param_grid=param_grid,
                    scoring="roc_auc", cv=kf,
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

                best_par    = gs.best_params_
                best_cv_auc = round(gs.best_score_, 5)
                log.info(f"[{kisa}] Best: {best_par}  CV-AUC: {best_cv_auc:.5f}")
                cv_sonuclari.append({
                    "Model": ad, "Kisa": kisa,
                    "Best_Params": str(best_par),
                    "CV_AUC": best_cv_auc,
                    "GS_Sure_s": round(gs_sure, 2),
                })

                # En iyi parametrelerle tam train üzerinde yeniden eğit
                adim(f"  [{kisa}] Tam train yeniden eğitim...")
                best_est = gs.best_estimator_
                best_est.set_params(**best_par)
                t0 = time.time()
                try:
                    best_est.fit(X_tr_m, y_train)
                except Exception as _fit_err:
                    # LightGBM GPU tam-train hatası: CPU'ya düş
                    if kisa == "LGBM" and "best_split_info.left_count" in str(_fit_err):
                        log.warning("[LGBM] GPU tam-train hatası, CPU ile yeniden deneniyor...")
                        best_est.set_params(device="cpu", n_jobs=N_JOBS)
                        best_est.fit(X_tr_m, y_train)
                    else:
                        raise
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
            y_prob = (best_est.predict_proba(X_te_m)[:, 1]
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
        adim(f"  [{kisa}] ROC-AUC={metrik['ROC_AUC']}  "
             f"F1={metrik['F1']}  PR-AUC={metrik['PR_AUC']}  "
             f"({egitim_sure:.1f}s)")

    except Exception as e:
        log.error(f"[{kisa}] HATA: {e}", exc_info=True)
        sonuc_listesi.append({
            "Model": ad, "Kisa": kisa, "HATA": str(e),
            "ROC_AUC": None, "F1": None, "PR_AUC": None,
        })
        print(f"  ✗ [{kisa}] HATA: {e}", flush=True)

tamam("Tüm model eğitimleri tamamlandı")


# =============================================================================
# 7. SONUÇ TABLOLARI
# =============================================================================

bolum("BÖLÜM 4 — Sonuç Tabloları")

sonuc_df = pd.DataFrame(sonuc_listesi)
kolon_sirasi = [
    "Model", "Kisa", "ROC_AUC", "PR_AUC", "F1",
    "Precision", "Recall", "Accuracy",
    "TP", "TN", "FP", "FN",
    "Egitim_Sure_s", "DL", "Akademik_Not",
]
sonuc_df = sonuc_df[[c for c in kolon_sirasi if c in sonuc_df.columns]]
sonuc_df = sonuc_df.sort_values("ROC_AUC", ascending=False).reset_index(drop=True)
sonuc_df.insert(0, "Sira", range(1, len(sonuc_df) + 1))
csv_kaydet(sonuc_df, "01_tum_model_sonuclari.csv", index=False)

if cv_sonuclari:
    cv_df = pd.DataFrame(cv_sonuclari)
    csv_kaydet(cv_df, "02_gridsearchcv_ozet.csv", index=False)

best_row  = sonuc_df[sonuc_df["ROC_AUC"].notna()].iloc[0]
best_kisa = best_row["Kisa"]
best_model = model_deposu.get(best_kisa)

adim(f"En iyi: [{best_kisa}] {best_row['Model']}")
adim(f"  ROC-AUC={best_row['ROC_AUC']}  F1={best_row['F1']}  PR-AUC={best_row['PR_AUC']}")
tamam("Sonuç tabloları hazır")


# =============================================================================
# 8. GÖRSELLEŞTİRME
# =============================================================================

bolum("BÖLÜM 5 — Görselleştirmeler")

df_plot = sonuc_df[sonuc_df["ROC_AUC"].notna()].copy()

# ── 8.1  ROC-AUC Barplot ──────────────────────────────────────────────────
df_b = df_plot.sort_values("ROC_AUC", ascending=True)
fig, ax = plt.subplots(figsize=(10, 6))
renkler = ["#e74c3c" if k == best_kisa else "#3498db" for k in df_b["Kisa"]]
bars = ax.barh(df_b["Model"], df_b["ROC_AUC"], color=renkler, edgecolor="white")
ax.axvline(0.5, color="gray", linestyle="--", linewidth=1, label="Rastgele (AUC=0.50)")
for bar, val in zip(bars, df_b["ROC_AUC"]):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", ha="left", fontsize=9)
ax.set_xlabel("ROC-AUC (Test Seti)", fontsize=11)
ax.set_title("Görev 1: failure_flag — ROC-AUC Karşılaştırması\n"
             "(kırmızı = en iyi model)", fontsize=12)
ax.set_xlim(0, 1.05)
ax.legend(fontsize=9)
plt.tight_layout()
png_kaydet(fig, "01_roc_auc_karsilastirma.png")
adim("01 — ROC-AUC barplot ✓")

# ── 8.2  F1 / Precision / Recall Grouped Bar ──────────────────────────────
df_met = df_plot.set_index("Model")[["F1", "Precision", "Recall"]].dropna()
fig, ax = plt.subplots(figsize=(12, 6))
df_met.plot(kind="bar", ax=ax, color=["#2ecc71", "#e67e22", "#9b59b6"],
            edgecolor="white", width=0.7)
ax.set_xticklabels(df_met.index, rotation=40, ha="right", fontsize=9)
ax.set_ylabel("Skor", fontsize=11)
ax.set_title("Görev 1: failure_flag — F1 / Precision / Recall", fontsize=12)
ax.legend(fontsize=10)
ax.set_ylim(0, 1.05)
plt.tight_layout()
png_kaydet(fig, "02_f1_precision_recall.png")
adim("02 — F1/Precision/Recall ✓")

# ── 8.3  PR-AUC Barplot ───────────────────────────────────────────────────
df_pr = df_plot.sort_values("PR_AUC", ascending=True)
fig, ax = plt.subplots(figsize=(10, 6))
rpr = ["#e74c3c" if k == best_kisa else "#16a085" for k in df_pr["Kisa"]]
brs = ax.barh(df_pr["Model"], df_pr["PR_AUC"], color=rpr, edgecolor="white")
for bar, val in zip(brs, df_pr["PR_AUC"]):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", ha="left", fontsize=9)
ax.set_xlabel("PR-AUC (Test Seti)", fontsize=11)
ax.set_title("Görev 1: failure_flag — PR-AUC Karşılaştırması\n"
             "(dengesiz veri için temel metrik)", fontsize=12)
ax.set_xlim(0, 1.05)
plt.tight_layout()
png_kaydet(fig, "03_pr_auc_karsilastirma.png")
adim("03 — PR-AUC barplot ✓")

# ── 8.4  ROC Eğrileri ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Rastgele (AUC=0.50)")
for kr in MODEL_REGISTRY:
    k = kr["kisa"]
    if k not in model_deposu:
        continue
    try:
        if kr["dl"]:
            proba = model_deposu[k].predict(
                to_dense(X_test).astype("float32"), verbose=0).ravel()
        elif hasattr(model_deposu[k], "predict_proba"):
            Xp = to_dense(X_test) if kr["needs_dense"] else X_test
            proba = model_deposu[k].predict_proba(Xp)[:, 1]
        else:
            continue
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc_v = sonuc_df.loc[sonuc_df["Kisa"] == k, "ROC_AUC"].values
        auc_v = auc_v[0] if len(auc_v) else None
        lw = 2.5 if k == best_kisa else 1.2
        ax.plot(fpr, tpr, lw=lw,
                label=f"{k} ({auc_v:.4f})" if auc_v else k)
    except Exception as e:
        log.warning(f"ROC [{k}]: {e}")
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate", fontsize=11)
ax.set_title("Görev 1: failure_flag — ROC Eğrileri (Test Seti)", fontsize=12)
ax.legend(loc="lower right", fontsize=8)
plt.tight_layout()
png_kaydet(fig, "04_roc_egriler.png")
adim("04 — ROC eğrileri ✓")

# ── 8.5  PR Eğrileri ─────────────────────────────────────────────────────
baseline_pr = float(y_test.mean())
fig, ax = plt.subplots(figsize=(10, 8))
ax.axhline(baseline_pr, color="gray", linestyle="--", linewidth=1,
           label=f"Baseline Precision={baseline_pr:.4f}")
for kr in MODEL_REGISTRY:
    k = kr["kisa"]
    if k not in model_deposu:
        continue
    try:
        if kr["dl"]:
            proba = model_deposu[k].predict(
                to_dense(X_test).astype("float32"), verbose=0).ravel()
        elif hasattr(model_deposu[k], "predict_proba"):
            Xp = to_dense(X_test) if kr["needs_dense"] else X_test
            proba = model_deposu[k].predict_proba(Xp)[:, 1]
        else:
            continue
        prec, rec, _ = precision_recall_curve(y_test, proba)
        pr_v = sonuc_df.loc[sonuc_df["Kisa"] == k, "PR_AUC"].values
        pr_v = pr_v[0] if len(pr_v) else None
        lw = 2.5 if k == best_kisa else 1.2
        ax.plot(rec, prec, lw=lw,
                label=f"{k} ({pr_v:.4f})" if pr_v else k)
    except Exception as e:
        log.warning(f"PR [{k}]: {e}")
ax.set_xlabel("Recall", fontsize=11)
ax.set_ylabel("Precision", fontsize=11)
ax.set_title("Görev 1: failure_flag — Precision-Recall Eğrileri\n"
             "(dengesiz veri için kritik metrik)", fontsize=12)
ax.legend(loc="upper right", fontsize=8)
plt.tight_layout()
png_kaydet(fig, "05_pr_egriler.png")
adim("05 — PR eğrileri ✓")

# ── 8.6  En İyi Model Confusion Matrix ───────────────────────────────────
if best_model is not None:
    try:
        bk = next(kr for kr in MODEL_REGISTRY if kr["kisa"] == best_kisa)
        if bk.get("dl"):
            yp = (best_model.predict(
                to_dense(X_test).astype("float32"), verbose=0).ravel() >= 0.5
            ).astype(int)
        else:
            Xbp = to_dense(X_test) if bk["needs_dense"] else X_test
            yp  = best_model.predict(Xbp)
        cm = confusion_matrix(y_test, yp)
        fig, ax = plt.subplots(figsize=(7, 6))
        ConfusionMatrixDisplay(cm, display_labels=["Yaşıyor (0)", "Öldü (1)"]).plot(
            ax=ax, colorbar=True, cmap="Blues")
        ax.set_title(
            f"Görev 1: Confusion Matrix — [{best_kisa}] {best_row['Model']}\n"
            f"ROC-AUC={best_row['ROC_AUC']}  F1={best_row['F1']}", fontsize=11)
        plt.tight_layout()
        png_kaydet(fig, f"06_cm_{best_kisa}.png")
        adim(f"06 — Confusion Matrix [{best_kisa}] ✓")
    except Exception as e:
        log.warning(f"CM [{best_kisa}]: {e}")

# ── 8.7  Tüm Modeller CM Grid ────────────────────────────────────────────
ncols = 4
nrows = (len(sonuc_df) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
axes = axes.flatten()
for idx, row in sonuc_df.iterrows():
    k, ax = row["Kisa"], axes[idx]
    if k not in model_deposu or row.get("HATA"):
        ax.set_visible(False)
        continue
    try:
        mk = next(kr for kr in MODEL_REGISTRY if kr["kisa"] == k)
        if mk.get("dl"):
            yg = (model_deposu[k].predict(
                to_dense(X_test).astype("float32"), verbose=0).ravel() >= 0.5
            ).astype(int)
        else:
            Xg = to_dense(X_test) if mk["needs_dense"] else X_test
            yg = model_deposu[k].predict(Xg)
        ConfusionMatrixDisplay(confusion_matrix(y_test, yg),
                               display_labels=["0", "1"]).plot(
            ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{k}: F1={row['F1']}", fontsize=9)
    except Exception as e:
        ax.set_title(f"{k}: HATA", fontsize=9)
for i in range(idx + 1, len(axes)):
    axes[i].set_visible(False)
plt.suptitle("Görev 1: failure_flag — Tüm Modeller CM Grid",
             fontsize=13, y=1.01)
plt.tight_layout()
png_kaydet(fig, "07_cm_grid_tummodeller.png")
adim("07 — CM Grid ✓")

# ── 8.8  Performans Isı Haritası ─────────────────────────────────────────
heat_cols = ["ROC_AUC", "PR_AUC", "F1", "Precision", "Recall", "Accuracy"]
heat_df = sonuc_df.set_index("Model")[heat_cols].apply(pd.to_numeric, errors="coerce")
fig, ax = plt.subplots(figsize=(10, max(5, len(heat_df) * 0.7)))
sns.heatmap(heat_df, annot=True, fmt=".4f", cmap="YlOrRd",
            vmin=0, vmax=1, linewidths=0.5, ax=ax, annot_kws={"size": 8})
ax.set_title("Görev 1: failure_flag — Performans Isı Haritası", fontsize=12)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
plt.tight_layout()
png_kaydet(fig, "08_performans_isi_haritasi.png")
adim("08 — Isı haritası ✓")

# ── 8.9  Eğitim Süresi ───────────────────────────────────────────────────
df_sure = sonuc_df[["Model", "Kisa", "Egitim_Sure_s", "DL"]].dropna(
    subset=["Egitim_Sure_s"]).sort_values("Egitim_Sure_s", ascending=True)
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(df_sure["Model"],
        df_sure["Egitim_Sure_s"],
        color=["#e74c3c" if d else "#2980b9" for d in df_sure["DL"]],
        edgecolor="white")
ax.set_xlabel("Eğitim Süresi (saniye)", fontsize=11)
ax.set_title("Görev 1: failure_flag — Eğitim Süresi\n"
             "(mavi=sklearn, kırmızı=Derin Öğrenme)", fontsize=12)
plt.tight_layout()
png_kaydet(fig, "09_egitim_suresi.png")
adim("09 — Eğitim süresi ✓")

# ── 8.10  Eşik Duyarlılık Çizgisi ────────────────────────────────────────
try:
    bk_f = next(kr for kr in MODEL_REGISTRY if kr["kisa"] == best_kisa)
    if bk_f.get("dl"):
        y_bp = best_model.predict(
            to_dense(X_test).astype("float32"), verbose=0).ravel()
    elif hasattr(best_model, "predict_proba"):
        Xbp2 = to_dense(X_test) if bk_f["needs_dense"] else X_test
        y_bp = best_model.predict_proba(Xbp2)[:, 1]
    else:
        y_bp = None

    if y_bp is not None:
        esikler = np.arange(0.05, 0.96, 0.05)
        esik_kayitlar = []
        for esik in esikler:
            ye = (y_bp >= esik).astype(int)
            me = metrikleri_hesapla(y_test, ye)
            me["Esik"] = round(float(esik), 2)
            esik_kayitlar.append(me)
        esik_df = pd.DataFrame(esik_kayitlar)
        csv_kaydet(esik_df, "03_esik_analizi.csv", index=False)

        best_esik = float(esik_df.loc[esik_df["F1"].idxmax(), "Esik"])

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(esik_df["Esik"], esik_df["F1"],
                "o-", lw=2, color="#2ecc71", label="F1")
        ax.plot(esik_df["Esik"], esik_df["Precision"],
                "s-", lw=2, color="#e67e22", label="Precision")
        ax.plot(esik_df["Esik"], esik_df["Recall"],
                "^-", lw=2, color="#9b59b6", label="Recall")
        ax.axvline(best_esik, color="gray", linestyle="--",
                   label=f"Best F1 eşiği={best_esik:.2f}")
        ax.set_xlabel("Sınıflandırma Eşiği", fontsize=11)
        ax.set_ylabel("Metrik Değeri", fontsize=11)
        ax.set_title(f"Görev 1: [{best_kisa}] Eşik Duyarlılık Analizi", fontsize=12)
        ax.legend(fontsize=10)
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        png_kaydet(fig, "10_esik_duyarlilik.png")
        adim(f"10 — Eşik analizi ✓  (Best F1 eşiği: {best_esik:.2f})")
except Exception as e:
    log.warning(f"Eşik analizi: {e}")

tamam("Tüm görselleştirmeler tamamlandı")


# =============================================================================
# 9. SHAP ANALİZİ
# =============================================================================

bolum("BÖLÜM 6 — SHAP Analizi (En İyi Model)")

try:
    if not _shap:
        raise ImportError("shap")

    adim(f"SHAP başlatılıyor → [{best_kisa}] {best_row['Model']}")

    bk_shap = next(kr for kr in MODEL_REGISTRY if kr["kisa"] == best_kisa)

    # Test setinden örneklem al (KernelExplainer ağır olabilir)
    np.random.seed(RANDOM_STATE)
    shap_idx   = np.random.choice(n_test, min(SHAP_SAMPLE_N, n_test), replace=False)
    X_shap_raw = to_dense(X_test)[shap_idx].astype("float64")

    # SHAP Explainer seçimi
    # TreeExplainer: RF, XGB, LGBM, CAT, DT için hızlı ve kesin
    # LinearExplainer: LR için
    # KernelExplainer: SVM, NB, KNN, DL için genel (yavaş)
    tree_modeller  = {"RF", "XGB", "LGBM", "CAT", "DT"}
    linear_modeller = {"LR"}

    if best_kisa in tree_modeller:
        explainer    = shap.TreeExplainer(best_model)
        shap_values  = explainer.shap_values(X_shap_raw)
        # İkili sınıflandırmada bazı modeller (RF) [class0, class1] döner
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        shap_tip = "TreeExplainer"

    elif best_kisa in linear_modeller:
        explainer   = shap.LinearExplainer(best_model, X_shap_raw)
        shap_values = explainer.shap_values(X_shap_raw)
        shap_tip    = "LinearExplainer"

    else:
        # KernelExplainer — genel ama yavaş; 200 örnekle arka plan
        arka_plan_n = min(200, X_shap_raw.shape[0])
        arka_plan   = shap.kmeans(X_shap_raw, arka_plan_n)

        if bk_shap.get("dl"):
            predict_fn = lambda x: best_model.predict(
                x.astype("float32"), verbose=0).ravel()
        else:
            predict_fn = (best_model.predict_proba
                          if hasattr(best_model, "predict_proba")
                          else best_model.predict)

        explainer   = shap.KernelExplainer(predict_fn, arka_plan)
        shap_values = explainer.shap_values(X_shap_raw[:500], nsamples=100)
        shap_tip    = "KernelExplainer"

    # KernelExplainer sadece ilk 500 satırı hesaplar; diğer explainer'lar tam setde çalışır.
    # X_shap_raw'ı shap_values satır sayısıyla hizala.
    X_shap_raw = X_shap_raw[:shap_values.shape[0]]
    shap_idx   = shap_idx[:shap_values.shape[0]]

    adim(f"SHAP tipi: {shap_tip}  |  örneklem: {X_shap_raw.shape[0]}")

    # Feature isimleri — uzunluk uyumu
    feat_n = (feature_names[:n_features]
              if len(feature_names) >= n_features
              else feature_names + [f"feat_{i}" for i in range(len(feature_names), n_features)])

    shap_df = pd.DataFrame(
        shap_values,
        columns=feat_n[:shap_values.shape[1]],
    )
    csv_kaydet(shap_df, "shap_values.csv", klasor=SHAP_K, index=False)

    # Özellik önemi (|SHAP|'ın ortalaması)
    shap_importance = (pd.DataFrame({
        "Feature": feat_n[:shap_values.shape[1]],
        "Mean_Abs_SHAP": np.abs(shap_values).mean(axis=0),
    }).sort_values("Mean_Abs_SHAP", ascending=False).reset_index(drop=True))
    csv_kaydet(shap_importance, "shap_feature_importance.csv",
               klasor=SHAP_K, index=False)

    # ── SHAP 9.1  Bar Plot (Top N özellik) ──────────────────────────────
    top_df = shap_importance.head(SHAP_TOP_N)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top_df["Feature"][::-1], top_df["Mean_Abs_SHAP"][::-1],
            color="#e74c3c", edgecolor="white")
    ax.set_xlabel("Ortalama |SHAP Değeri|", fontsize=11)
    ax.set_title(
        f"Görev 1: [{best_kisa}] — SHAP Özellik Önemi (Top {SHAP_TOP_N})\n"
        f"Hangi sensör değişkenleri bitki ölümünü belirliyor?", fontsize=12)
    plt.tight_layout()
    png_kaydet(fig, "shap_01_bar_plot.png", klasor=SHAP_K)
    adim("SHAP 01 — Bar plot ✓")

    # ── SHAP 9.2  Summary Plot (Beeswarm) ─────────────────────────────────
    top_feat_idx = [feat_n.index(f) for f in top_df["Feature"].tolist()
                    if f in feat_n][:SHAP_TOP_N]
    X_top = X_shap_raw[:, top_feat_idx]
    sv_top = shap_values[:, top_feat_idx]
    feat_top = [feat_n[i] for i in top_feat_idx]

    fig, ax = plt.subplots(figsize=(11, 8))
    shap.summary_plot(sv_top, X_top, feature_names=feat_top,
                      plot_type="dot", show=False, max_display=SHAP_TOP_N,
                      plot_size=None)
    plt.title(
        f"Görev 1: [{best_kisa}] — SHAP Beeswarm (Top {SHAP_TOP_N})\n"
        "Kırmızı=yüksek değer, Mavi=düşük değer", fontsize=12)
    plt.tight_layout()
    png_kaydet(fig, "shap_02_beeswarm.png", klasor=SHAP_K)
    adim("SHAP 02 — Beeswarm ✓")

    # ── SHAP 9.3  Waterfall — 1 pozitif, 1 negatif örnek ─────────────────
    y_test_arr = np.array(y_test)
    pos_idx = np.where(y_test_arr[shap_idx] == 1)[0]
    neg_idx = np.where(y_test_arr[shap_idx] == 0)[0]

    for ornk_ad, ornk_idx in [
        ("pozitif_ornek_oldu", pos_idx[:1]),
        ("negatif_ornek_yasadi", neg_idx[:1]),
    ]:
        if len(ornk_idx) == 0:
            continue
        i = ornk_idx[0]
        exp = shap.Explanation(
            values=shap_values[i],
            base_values=float(explainer.expected_value
                              if not isinstance(explainer.expected_value, np.ndarray)
                              else explainer.expected_value[0]),
            data=X_shap_raw[i],
            feature_names=feat_n[:shap_values.shape[1]],
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(exp, max_display=15, show=False)
        plt.title(
            f"Görev 1: [{best_kisa}] — SHAP Waterfall ({ornk_ad})", fontsize=11)
        plt.tight_layout()
        png_kaydet(fig, f"shap_03_waterfall_{ornk_ad}.png", klasor=SHAP_K)
    adim("SHAP 03 — Waterfall ✓")

    # ── SHAP 9.4  Dependence Plot — En önemli 2 özellik ──────────────────
    top2_feats = top_df["Feature"].tolist()[:2]
    for feat_adi in top2_feats:
        if feat_adi not in feat_n:
            continue
        fidx = feat_n.index(feat_adi)
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.dependence_plot(
            fidx, shap_values, X_shap_raw,
            feature_names=feat_n[:shap_values.shape[1]],
            ax=ax, show=False,
        )
        ax.set_title(
            f"Görev 1: [{best_kisa}] — SHAP Dependence: {feat_adi}", fontsize=11)
        plt.tight_layout()
        fname = feat_adi.replace(" ", "_").replace("/", "_")
        png_kaydet(fig, f"shap_04_dependence_{fname}.png", klasor=SHAP_K)
    adim("SHAP 04 — Dependence plots ✓")

    # SHAP özeti CSV
    shap_ozet = pd.DataFrame([{
        "Explainer_Tipi": shap_tip,
        "Model": best_row["Model"],
        "Ornek_Sayisi": X_shap_raw.shape[0],
        "Top1_Feature": shap_importance.iloc[0]["Feature"],
        "Top1_SHAP":    round(shap_importance.iloc[0]["Mean_Abs_SHAP"], 5),
        "Top2_Feature": shap_importance.iloc[1]["Feature"] if len(shap_importance) > 1 else "",
        "Top2_SHAP":    round(shap_importance.iloc[1]["Mean_Abs_SHAP"], 5) if len(shap_importance) > 1 else None,
        "Top3_Feature": shap_importance.iloc[2]["Feature"] if len(shap_importance) > 2 else "",
        "Top3_SHAP":    round(shap_importance.iloc[2]["Mean_Abs_SHAP"], 5) if len(shap_importance) > 2 else None,
    }])
    csv_kaydet(shap_ozet, "shap_ozet.csv", klasor=SHAP_K, index=False)

    tamam("SHAP analizi tamamlandı")
    shap_ok = True

except ImportError:
    adim("⚠ shap kütüphanesi kurulu değil → 'pip install shap'")
    shap_ok = False
    shap_importance = pd.DataFrame()
except Exception as e:
    log.error(f"SHAP hatası: {e}", exc_info=True)
    adim(f"⚠ SHAP hatası: {e}")
    shap_ok = False
    shap_importance = pd.DataFrame()


# =============================================================================
# 10. DUYARLILIK ANALİZİ (Gaussian Gürültü — %5 / %10 / %20)
# =============================================================================

bolum("BÖLÜM 7 — Duyarlılık Analizi (Gaussian Gürültü)")

# Metodoloji:
#   Ham test verisi okunur → sayısal sütunlara sıfır ortalamalı, std-orantılı
#   Gaussian gürültü eklenir → preprocessor.transform ile dönüştürülür →
#   eğitilmiş en iyi model ile tahmin yapılır → metrik değişimi ölçülür.
#   Her oran için GURULTU_TEKRAR kez tekrar edilir; ortalama ve std raporlanır.

duyar_ok = False

if ham_test_df is None:
    adim("⚠ Ham test verisi yok — duyarlılık analizi atlandı.")
elif best_model is None:
    adim("⚠ En iyi model yok — duyarlılık analizi atlandı.")
else:
    try:
        HEDEFLER = ["failure_flag", "suitability_score", "stress_level"]
        from sklearn.metrics import roc_auc_score, f1_score, average_precision_score

        # Hedef ve bağımsız değişkenleri ayır
        X_ham_test = ham_test_df.drop(
            columns=[c for c in HEDEFLER if c in ham_test_df.columns],
            errors="ignore",
        )
        y_ham_test = ham_test_df[HEDEF].values

        # Aşama 1'de düşürülen sütunları çıkar
        dusurulen_path = A1_DATA / "18_dusurulen_sutunlar.csv"
        if dusurulen_path.exists():
            dus_df = pd.read_csv(dusurulen_path)
            if "Sütun" in dus_df.columns:
                X_ham_test = X_ham_test.drop(
                    columns=[c for c in dus_df["Sütun"] if c in X_ham_test.columns],
                    errors="ignore",
                )

        # Sayısal sütunlar (gürültü ekleme hedefi)
        say_sutunlar = X_ham_test.select_dtypes(include=[np.number]).columns.tolist()
        adim(f"Gürültü eklenecek sayısal sütun: {len(say_sutunlar)}")

        # Baseline (gürültüsüz) metrikleri
        X_base_proc = preprocessor.transform(X_ham_test)
        bk_d = next(kr for kr in MODEL_REGISTRY if kr["kisa"] == best_kisa)
        if bk_d.get("dl"):
            y_base_prob = best_model.predict(
                to_dense(X_base_proc).astype("float32"), verbose=0).ravel()
        elif hasattr(best_model, "predict_proba"):
            Xb = to_dense(X_base_proc) if bk_d["needs_dense"] else X_base_proc
            y_base_prob = best_model.predict_proba(Xb)[:, 1]
        else:
            Xb = to_dense(X_base_proc) if bk_d["needs_dense"] else X_base_proc
            y_base_prob = None

        y_base_pred = (y_base_prob >= 0.5).astype(int) if y_base_prob is not None else \
            best_model.predict(to_dense(X_base_proc) if bk_d["needs_dense"] else X_base_proc)

        base_roc = round(roc_auc_score(y_ham_test, y_base_prob), 5) if y_base_prob is not None else None
        base_f1  = round(f1_score(y_ham_test, y_base_pred, zero_division=0), 5)
        base_pr  = round(average_precision_score(y_ham_test, y_base_prob), 5) if y_base_prob is not None else None

        adim(f"Baseline (gürültüsüz) → ROC-AUC={base_roc}  F1={base_f1}  PR-AUC={base_pr}")

        duyar_kayitlar = [{
            "Gurultu_Oran": 0.0,
            "Tekrar": 0,
            "ROC_AUC": base_roc,
            "F1": base_f1,
            "PR_AUC": base_pr,
        }]

        for oran in GURULTU_ORANLARI:
            roc_list, f1_list, pr_list = [], [], []

            for tekrar in range(GURULTU_TEKRAR):
                np.random.seed(RANDOM_STATE + tekrar)
                X_gurultulu = X_ham_test.copy()

                # Her sayısal sütunun std'sine oranla gürültü ekle
                for sut in say_sutunlar:
                    std_sut = X_gurultulu[sut].std()
                    if std_sut > 0:
                        gurultu = np.random.normal(
                            loc=0,
                            scale=oran * std_sut,
                            size=len(X_gurultulu),
                        )
                        X_gurultulu[sut] = X_gurultulu[sut] + gurultu

                # Preprocessor ile dönüştür
                X_g_proc = preprocessor.transform(X_gurultulu)

                # Tahmin
                if bk_d.get("dl"):
                    y_g_prob = best_model.predict(
                        to_dense(X_g_proc).astype("float32"), verbose=0).ravel()
                elif hasattr(best_model, "predict_proba"):
                    Xgp = to_dense(X_g_proc) if bk_d["needs_dense"] else X_g_proc
                    y_g_prob = best_model.predict_proba(Xgp)[:, 1]
                else:
                    Xgp = to_dense(X_g_proc) if bk_d["needs_dense"] else X_g_proc
                    y_g_prob = None

                y_g_pred = ((y_g_prob >= 0.5).astype(int) if y_g_prob is not None
                            else best_model.predict(Xgp))

                roc_g = roc_auc_score(y_ham_test, y_g_prob) if y_g_prob is not None else None
                f1_g  = f1_score(y_ham_test, y_g_pred, zero_division=0)
                pr_g  = average_precision_score(y_ham_test, y_g_prob) if y_g_prob is not None else None

                roc_list.append(roc_g)
                f1_list.append(f1_g)
                pr_list.append(pr_g)

                duyar_kayitlar.append({
                    "Gurultu_Oran": oran,
                    "Tekrar": tekrar + 1,
                    "ROC_AUC": round(roc_g, 5) if roc_g else None,
                    "F1": round(f1_g, 5),
                    "PR_AUC": round(pr_g, 5) if pr_g else None,
                })

            adim(f"  %{int(oran*100):3d} gürültü → "
                 f"ROC-AUC={np.mean([x for x in roc_list if x]):.4f}±{np.std([x for x in roc_list if x]):.4f}  "
                 f"F1={np.mean(f1_list):.4f}±{np.std(f1_list):.4f}")

        duyar_df = pd.DataFrame(duyar_kayitlar)
        csv_kaydet(duyar_df, "duyarlilik_ham_sonuclar.csv",
                   klasor=DUYAR_K, index=False)

        # Özet (ortalama ± std)
        duyar_ozet = (duyar_df.groupby("Gurultu_Oran")
                      .agg({"ROC_AUC": ["mean", "std"],
                             "F1": ["mean", "std"],
                             "PR_AUC": ["mean", "std"]})
                      .round(5))
        duyar_ozet.columns = ["_".join(c) for c in duyar_ozet.columns]
        duyar_ozet = duyar_ozet.reset_index()
        duyar_ozet["Gurultu_Yuzde"] = (duyar_ozet["Gurultu_Oran"] * 100).astype(int)

        # ROC-AUC değişim oranı (baseline'a göre)
        if base_roc:
            duyar_ozet["ROC_AUC_Degisim_Pct"] = (
                (duyar_ozet["ROC_AUC_mean"] - base_roc) / base_roc * 100
            ).round(3)

        csv_kaydet(duyar_ozet, "duyarlilik_ozet.csv",
                   klasor=DUYAR_K, index=False)

        # ── Duyarlılık Grafiği ─────────────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        metrik_bilgi = [
            ("ROC_AUC_mean", "ROC_AUC_std", "ROC-AUC", "#2980b9", base_roc),
            ("F1_mean",      "F1_std",      "F1-Score","#27ae60", base_f1),
            ("PR_AUC_mean",  "PR_AUC_std",  "PR-AUC",  "#8e44ad", base_pr),
        ]
        for ax_i, (mcol, scol, mlbl, mclr, mbase) in zip(axes, metrik_bilgi):
            x = duyar_ozet["Gurultu_Yuzde"].values
            y = duyar_ozet[mcol].values
            s = duyar_ozet[scol].values
            ax_i.errorbar(x, y, yerr=s, marker="o", color=mclr,
                          capsize=5, linewidth=2, label="Ortalama ± std")
            if mbase is not None:
                ax_i.axhline(mbase, linestyle="--", color="gray",
                             linewidth=1, label=f"Baseline={mbase:.4f}")
            ax_i.set_xlabel("Gürültü Oranı (%)", fontsize=10)
            ax_i.set_ylabel(mlbl, fontsize=10)
            ax_i.set_title(f"{mlbl} — Gürültüye Duyarlılık", fontsize=11)
            ax_i.legend(fontsize=8)
            ax_i.set_xticks(x)
            ax_i.set_ylim(0, 1.05)

        plt.suptitle(
            f"Görev 1: [{best_kisa}] {best_row['Model']} — Gaussian Gürültü Duyarlılık Analizi\n"
            f"(Her oran için {GURULTU_TEKRAR} tekrar, ortalama ± standart sapma)",
            fontsize=12, y=1.02,
        )
        plt.tight_layout()
        png_kaydet(fig, "duyarlilik_analizi.png", klasor=DUYAR_K)
        adim("Duyarlılık grafiği ✓")

        # ── Kararlılık Raporu ──────────────────────────────────────────────
        en_kotu_roc = duyar_ozet.loc[duyar_ozet["Gurultu_Yuzde"] == 20, "ROC_AUC_mean"].values
        en_kotu_f1  = duyar_ozet.loc[duyar_ozet["Gurultu_Yuzde"] == 20, "F1_mean"].values

        roc_dusus = ((base_roc - en_kotu_roc[0]) / base_roc * 100
                     if base_roc and len(en_kotu_roc) else None)
        f1_dusus  = ((base_f1  - en_kotu_f1[0])  / base_f1  * 100
                     if base_f1 and len(en_kotu_f1) else None)

        adim(f"Kararlılık (%20 gürültüde düşüş) → "
             f"ROC-AUC: {roc_dusus:.2f}%  F1: {f1_dusus:.2f}%"
             if roc_dusus and f1_dusus else "Kararlılık hesaplanamadı")

        tamam("Duyarlılık analizi tamamlandı")
        duyar_ok = True

    except Exception as e:
        log.error(f"Duyarlılık analizi hatası: {e}", exc_info=True)
        adim(f"⚠ Duyarlılık analizi hatası: {e}")
        roc_dusus, f1_dusus = None, None


# =============================================================================
# 11. AKADEMİK RAPOR
# =============================================================================

bolum("BÖLÜM 8 — Akademik Rapor")

rapor_tablo = sonuc_df[
    ["Sira", "Model", "ROC_AUC", "PR_AUC", "F1", "Precision", "Recall", "Accuracy"]
].to_string(index=False)

dl_notlar = "\n".join(
    f"- [{k['kisa']}] {k.get('akademik_not', '')}"
    for k in MODEL_REGISTRY if k.get("akademik_not")
)

shap_top3 = ""
if shap_ok and not shap_importance.empty:
    for i in range(min(3, len(shap_importance))):
        r = shap_importance.iloc[i]
        shap_top3 += f"  {i+1}. {r['Feature']} (|SHAP|={r['Mean_Abs_SHAP']:.5f})\n"

duyar_blok = ""
if duyar_ok:
    duyar_blok = (
        f"  Baseline (gürültüsüz): ROC-AUC={base_roc}  F1={base_f1}\n"
        f"  %5  gürültü: ROC-AUC düşüşü ölçüldü ({GURULTU_TEKRAR} tekrar)\n"
        f"  %10 gürültü: ROC-AUC düşüşü ölçüldü ({GURULTU_TEKRAR} tekrar)\n"
        f"  %20 gürültü: ROC-AUC düşüşü ≈{roc_dusus:.2f}%  F1 düşüşü ≈{f1_dusus:.2f}%\n"
    )
else:
    duyar_blok = "  Duyarlılık analizi çalıştırılamadı.\n"

rapor_metni = f"""# AŞAMA 2 — GÖREV 1 AKADEMİK RAPORU
# Başarısızlık Tahmini (Binary Classification — failure_flag)

## Öğrenci
İbrahim Nuryağınlı | 25490221001

## Aşama 1 Referansı
Aşama 1 Klasörü : {A1_DIR.name}
Özellik Sayısı  : {n_features}
Train           : {n_train:,} satır (SMOTE sonrası)
Test            : {n_test:,} satır
class_weight    : {cw_failure}
GridSearchCV    : {CV_FOLDS}-fold, örneklem={CV_SAMPLE_N:,} satır

---

## 1. Problem Tanımı
failure_flag değişkeni (0=Yaşıyor, 1=Öldü) için ikili sınıflandırma problemi.
Sınıf dengesizliği ~5.17:1 oranındadır; SMOTE ile train seti dengelenmiştir.

## 2. Metodoloji

### Veri Hazırlığı
- Tüm preprocessing (PowerTransformer, StandardScaler, OneHotEncoder) Aşama 1'de
  yalnızca train seti üzerinde fit edilmiştir → veri sızıntısı yoktur.
- GridSearchCV için train setinden {CV_SAMPLE_N:,} satır örneklenmiş,
  en iyi parametreler tam train ({n_train:,} satır) ile yeniden eğitilmiştir.
- Test seti yalnızca final değerlendirmede bir kez kullanılmıştır.

### Neden Accuracy Kullanılmıyor?
Sınıf dengesizliği (~5.17:1) nedeniyle tüm örnekler "Yaşıyor (0)" tahmin
edilse bile ~%83 accuracy elde edilir. Bu sebeple F1-Score ve PR-AUC
temel metrikler olarak kullanılmıştır.

---

## 3. Denemeler

### Denenecek Modeller (12 adet)
Logistic Regression, Decision Tree, KNN, SVM (LinearSVC + Platt Scaling),
Naive Bayes (BernoulliNB), Random Forest, XGBoost, LightGBM, CatBoost,
ANN, CNN-1D, RNN-LSTM

### SVM Notu
RBF kernel SVM 500k+ satırlı veri setinde eğitim süresi açısından pratik
değildir (O(n²) - O(n³) karmaşıklık). Bu nedenle LinearSVC tercih edilmiş,
ROC-AUC hesabı için Platt scaling (CalibratedClassifierCV) eklenmiştir.

---

## 4. Model Performans Tablosu (Test Seti — ROC-AUC'ya göre)

{rapor_tablo}

---

## 5. En İyi Model
Model    : {best_row['Model']} [{best_kisa}]
ROC-AUC  : {best_row['ROC_AUC']}
PR-AUC   : {best_row['PR_AUC']}
F1-Score : {best_row['F1']}
Precision: {best_row['Precision']}
Recall   : {best_row['Recall']}
Accuracy : {best_row['Accuracy']}

---

## 6. SHAP Analizi (Açıklanabilir Yapay Zeka / XAI)
SHAP (SHapley Additive exPlanations) yöntemi ile en iyi model olan
[{best_kisa}]'ın hangi sensör değişkenlerine dayandığı görselleştirilmiştir.

Explainer: {"TreeExplainer (ağaç tabanlı modeller için hızlı ve kesin)" if shap_ok else "Çalıştırılamadı"}
Örneklem : {SHAP_SAMPLE_N:,} test örneği

### Top 3 Belirleyici Özellik (|SHAP| ortalaması)
{shap_top3 if shap_top3 else "  SHAP çalıştırılamadı."}

Üretilen SHAP görselleştirmeleri:
  - Bar Plot      : Her özelliğin ortalama mutlak SHAP değeri
  - Beeswarm Plot : Özellik değeri (kırmızı=yüksek) ile SHAP etkisi
  - Waterfall     : Tekil bir pozitif ve negatif örnek için karar süreci
  - Dependence    : En önemli 2 özelliğin SHAP etkisi vs özellik değeri

---

## 7. Duyarlılık Analizi (Gaussian Gürültü)
Modelin gerçek saha koşullarındaki sensör gürültüsüne ne kadar dayanıklı
olduğunu test etmek amacıyla ham test verisinin sayısal sütunlarına
std-orantılı sıfır ortalamalı Gaussian gürültü eklenmiştir.

{duyar_blok}

### Yorum
Gürültü oranı arttıkça ROC-AUC ve F1 değerlerindeki değişim modelin
robustluğunu ölçer. Küçük düşüşler modelin sensör hatalarına karşı
kararlı olduğunu; büyük düşüşler ise özellik kalitesinin kritik
önem taşıdığını göstermektedir.

---

## 8. Derin Öğrenme Modelleri — Akademik Yorum
{dl_notlar if dl_notlar else "Derin öğrenme modelleri çalıştırılmadı."}

Tablolu veri setlerinde CNN ve LSTM'nin zayıf performansı beklenen bir bulgudur.
Bu modeller zamansal veya uzamsal dizi bağımlılıkları için tasarlanmıştır.
Sonuçların raporlanması bilimsel dürüstlük ve kapsam açısından önem taşımaktadır.

---

## 9. Veri Sızıntısı Önlemleri
1. Preprocessing yalnızca train üzerinde fit (Aşama 1).
2. GridSearchCV yalnızca train alt-kümesine ({CV_SAMPLE_N:,} satır) uygulandı.
3. Test seti yalnızca final değerlendirmede bir kez kullanıldı.
4. SMOTE yalnızca train setine; test orijinal dağılımda bırakıldı.
5. Duyarlılık analizinde preprocessor tekrar fit edilmedi;
   ham veriye gürültü eklendikten sonra mevcut preprocessor ile transform edildi.

---

## 10. Kaynakça
- Liakos vd. (2018). Machine learning in agriculture: A review. Sensors, 18(8), 2674.
- Lundberg & Lee (2017). A unified approach to interpreting model predictions. NIPS, 30.
- Van Klompenburg vd. (2020). Crop yield prediction using machine learning. COMPAG, 177.
- Roy, N. (2024). Agro-environmental stress & failure simulation. Kaggle.

---
Çıktı Dizini : {CIKTI}
Oluşturulma  : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

rapor_path = RAPOR / "gorev1_akademik_rapor.md"
rapor_path.write_text(rapor_metni, encoding="utf-8")
adim(f"Rapor kaydedildi: {rapor_path.name}")
tamam("Akademik rapor tamamlandı")


# =============================================================================
# 12. PAKET (Aşama 3 için)
# =============================================================================

bolum("BÖLÜM 9 — Paket Kayıt (Aşama 3'e aktarım)")

a2_meta = {
    "run_id":          RUN_ID,
    "a1_dir":          str(A1_DIR),
    "best_model_kisa": best_kisa,
    "best_model_ad":   best_row["Model"],
    "best_roc_auc":    float(best_row["ROC_AUC"]) if best_row["ROC_AUC"] else None,
    "best_f1":         float(best_row["F1"])       if best_row["F1"]      else None,
    "best_pr_auc":     float(best_row["PR_AUC"])   if best_row["PR_AUC"]  else None,
    "n_features":      n_features,
    "n_train":         n_train,
    "n_test":          n_test,
    "cv_folds":        CV_FOLDS,
    "cv_sample_n":     CV_SAMPLE_N,
    "shap_ok":         shap_ok if "shap_ok" in dir() else False,
    "duyar_ok":        duyar_ok,
    "random_state":    RANDOM_STATE,
    "output_dir":      str(CIKTI),
}

# sklearn modellerini paketle (Keras modeller .keras formatında zaten kaydedildi)
is_keras = lambda v: _tf and hasattr(v, "predict") and hasattr(v, "layers")
paket = {
    "metadata":      a2_meta,
    "sonuc_df":      sonuc_df,
    "feature_names": feature_names,
    "class_weights": cw_failure,
    "model_deposu":  {k: v for k, v in model_deposu.items() if not is_keras(v)},
}
joblib.dump(paket, CIKTI / "asama2_gorev1_paket.joblib")

with open(CIKTI / "asama2_gorev1_metadata.json", "w", encoding="utf-8") as f:
    json.dump(a2_meta, f, ensure_ascii=False, indent=2)

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

n_gorsel = len(list(GRAFIK.glob("*.png"))) + \
           len(list(SHAP_K.glob("*.png"))) + \
           len(list(DUYAR_K.glob("*.png")))
n_csv    = sum(len(list(d.glob("*"))) for d in [SONUC, SHAP_K, DUYAR_K, RAPOR])

print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║          AŞAMA 2 — GÖREV 1 TAMAMLANDI                               ║
╠══════════════════════════════════════════════════════════════════════╣
║  Hedef      : failure_flag (Binary Classification)                   ║
║  Train      : {n_train:,} satır (SMOTE)  |  Test: {n_test:,}
║  Özellik    : {n_features}
║  Model      : {len(MODEL_REGISTRY)} adet
║  En İyi     : [{best_kisa}] {best_row['Model']}
║  ROC-AUC    : {best_row['ROC_AUC']}
║  PR-AUC     : {best_row['PR_AUC']}
║  F1-Score   : {best_row['F1']}
║  SHAP       : {"✓" if (shap_ok if "shap_ok" in dir() else False) else "✗"}
║  Duyarlılık : {"✓" if duyar_ok else "✗"}
║  Görsel     : {n_gorsel}  |  CSV/MD: {n_csv}
║  Çıktı      : {CIKTI.name}
╚══════════════════════════════════════════════════════════════════════╝
""", flush=True)
