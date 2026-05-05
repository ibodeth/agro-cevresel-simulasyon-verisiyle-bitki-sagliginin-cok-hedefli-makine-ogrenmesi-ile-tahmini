# =============================================================================
# YZO 106 – İleri Düzey Makine Öğrenmesi | Dönem Projesi
# AŞAMA 3 – GÖREV 2: Uygunluk Skoru Tahmini (Regression — suitability_score)
# Öğrenci: İbrahim Nuryağınlı | 25490221001
#
# Bu dosya AŞAMA 1'in çıktı klasörünü otomatik algılar ve şu yapıyı bekler:
#   asama1_ciktilar_{RUN_ID}/
#       03_model_verileri/
#           X_train_processed.{npz|npy}         ← SMOTE uygulanmamış (regresyon)
#           y_suitability_train.csv
#           X_test_processed.{npz|npy}
#           y_suitability_test.csv
#           23_processed_feature_names.csv
#           TEST_raw_with_targets.csv            ← duyarlılık analizi için ham veri
#       04_preprocessor/
#           preprocessor.joblib
#           metadata.json
#           asama1_paket.joblib
#
# Ayrıca asama2_ciktilar_* varsa referans için yüklenir (zorunlu değil).
#
# Çalıştırmadan önce:
#   pip install scikit-learn xgboost lightgbm catboost
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
#   - Metrikler: R², MSE, RMSE, MAE
#   - suitability_score regresyon hedefi olduğu için SMOTE uygulanmaz.
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
HEDEF         = "suitability_score"

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

# Aşama 2 klasörü (referans — zorunlu değil)
asama2_klasorler = sorted(BASE_DIR.glob("asama2_ciktilar_*"), reverse=True)
A2_DIR = asama2_klasorler[0] if asama2_klasorler else None
if A2_DIR:
    print(f"▶ Aşama 2 klasörü: {A2_DIR.name} (referans)", flush=True)

RUN_ID  = datetime.now().strftime("%Y%m%d_%H%M%S")
CIKTI   = BASE_DIR / f"asama3_ciktilar_{RUN_ID}"
SONUC   = CIKTI / "01_sonuclar"
GRAFIK  = CIKTI / "02_gorseller"
MODEL_K = CIKTI / "03_modeller"
RAPOR   = CIKTI / "04_raporlar"
SHAP_K  = CIKTI / "05_shap"
DUYAR_K = CIKTI / "06_duyarlilik"

for d in [CIKTI, SONUC, GRAFIK, MODEL_K, RAPOR, SHAP_K, DUYAR_K]:
    d.mkdir(parents=True, exist_ok=True)

LOG_DOSYASI = RAPOR / f"asama3_log_{RUN_ID}.txt"
logger = logging.getLogger("asama3")
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


def metrikleri_hesapla(y_true, y_pred) -> dict:
    """Regresyon metriklerini hesaplar: R², MSE, RMSE, MAE."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    r2   = float(r2_score(y_true, y_pred))
    mse  = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae  = float(mean_absolute_error(y_true, y_pred))
    return {
        "R2":   round(r2,   5),
        "MSE":  round(mse,  5),
        "RMSE": round(rmse, 5),
        "MAE":  round(mae,  5),
    }


# =============================================================================
# 3. AŞAMA 1 VERİLERİNİ YÜKLE
# =============================================================================

bolum("BÖLÜM 1 — Aşama 1 Verilerini Yükle")

with open(A1_MODEL / "metadata.json", encoding="utf-8") as f:
    meta = json.load(f)

a1_paket      = joblib.load(A1_MODEL / "asama1_paket.joblib")
preprocessor  = joblib.load(A1_MODEL / "preprocessor.joblib")
feature_names = a1_paket.get("feature_names", [])

adim("Matrisleri yüklüyor...")
# Regresyon: SMOTE uygulanmamış, orijinal train seti kullanılır
X_train = load_matrix("X_train_processed")
X_test  = load_matrix("X_test_processed")

y_train = load_target("y_suitability_train.csv")
y_test  = load_target("y_suitability_test.csv")

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

adim(f"Train : {n_train:,} × {n_features}  (SMOTE uygulanmadı — regresyon)")
adim(f"Test  : {n_test:,} × {n_features}")
adim(f"y_train: min={y_train.min():.2f}  max={y_train.max():.2f}  "
     f"mean={y_train.mean():.2f}  std={y_train.std():.2f}")
adim(f"y_test : min={y_test.min():.2f}  max={y_test.max():.2f}  "
     f"mean={y_test.mean():.2f}  std={y_test.std():.2f}")

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

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor

try:
    from xgboost import XGBRegressor
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
    from catboost import CatBoostRegressor
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

# 1 — Linear Regression
MODEL_REGISTRY.append({
    "ad": "Linear Regression", "kisa": "LIN",
    "estimator": LinearRegression(n_jobs=N_JOBS),
    "param_grid": {},    # Hiperparametresi yok; doğrudan eğitilir
    "needs_dense": True, "dl": False,
})

# 2 — Decision Tree Regressor
MODEL_REGISTRY.append({
    "ad": "Decision Tree", "kisa": "DT",
    "estimator": DecisionTreeRegressor(random_state=RANDOM_STATE),
    "param_grid": {
        "max_depth": [5, 10, 20, None],
        "min_samples_leaf": [1, 5, 20],
        "criterion": ["squared_error", "friedman_mse"],
    },
    "needs_dense": False, "dl": False,
})

# 3 — KNN Regressor
MODEL_REGISTRY.append({
    "ad": "KNN", "kisa": "KNN",
    "estimator": KNeighborsRegressor(n_jobs=N_JOBS),
    "param_grid": {
        "n_neighbors": [5, 11, 21, 51],
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    },
    "needs_dense": True, "dl": False,
})

# 4 — SVR
# Not: RBF kernel 500k+ satırda pratik değil → LinearSVR tercih edilir
MODEL_REGISTRY.append({
    "ad": "SVR (Linear)", "kisa": "SVR",
    "estimator": LinearSVR(max_iter=3000, random_state=RANDOM_STATE),
    "param_grid": {
        "C": [0.01, 0.1, 1.0, 5.0],
        "epsilon": [0.01, 0.1, 0.5],
    },
    "needs_dense": True, "dl": False,
    "akademik_not": (
        "RBF kernel SVR O(n²)-O(n³) karmaşıklığı nedeniyle 500k+ satırlık "
        "veri setlerinde uygulanabilir değildir. LinearSVR lineer karmaşıklıkla "
        "büyük ölçekli veri setlerine uygun bir alternatiftir."
    ),
})

# 5 — Random Forest Regressor
MODEL_REGISTRY.append({
    "ad": "Random Forest", "kisa": "RF",
    "estimator": RandomForestRegressor(
        random_state=RANDOM_STATE, n_jobs=N_JOBS, n_estimators=300,
    ),
    "param_grid": {
        "n_estimators": [100, 300],
        "max_depth": [10, 20, None],
        "min_samples_leaf": [1, 5],
    },
    "needs_dense": False, "dl": False,
})

# 6 — XGBoost Regressor
if _xgb:
    xgb_params = {
        "eval_metric": "rmse", "tree_method": "hist",
        "random_state": RANDOM_STATE, "n_jobs": N_JOBS,
        "verbosity": 1,
    }
    if _gpu:
        xgb_params["device"] = "cuda"
        adim("  [XGB] GPU modu aktif")

    MODEL_REGISTRY.append({
        "ad": "XGBoost", "kisa": "XGB",
        "estimator": XGBRegressor(**xgb_params),
        "param_grid": {
            "n_estimators": [100, 300],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.05, 0.1, 0.2],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        },
        "needs_dense": True, "dl": False,
    })

# 7 — LightGBM Regressor
if _lgb:
    lgb_params = {
        "random_state": RANDOM_STATE, "n_jobs": N_JOBS, "verbose": -1,
    }
    if _gpu:
        lgb_params["device"] = "gpu"
        lgb_params["gpu_platform_id"] = 0
        lgb_params["gpu_device_id"] = 0
        adim("  [LGBM] GPU modu aktif")

    MODEL_REGISTRY.append({
        "ad": "LightGBM", "kisa": "LGBM",
        "estimator": lgb.LGBMRegressor(**lgb_params),
        "param_grid": {
            "n_estimators": [100, 300],
            "max_depth": [4, 6, -1],
            "learning_rate": [0.05, 0.1, 0.2],
            "num_leaves": [31, 63, 127],
        },
        "needs_dense": False, "dl": False,
    })

# 8 — CatBoost Regressor
if _cat:
    cat_params = {
        "random_seed": RANDOM_STATE, "verbose": 100,
        "allow_writing_files": False,
    }
    if _gpu:
        cat_params["task_type"] = "GPU"
        cat_params["devices"] = "0"
        adim("  [CAT] GPU modu aktif")

    MODEL_REGISTRY.append({
        "ad": "CatBoost", "kisa": "CAT",
        "estimator": CatBoostRegressor(**cat_params),
        "param_grid": {
            "iterations": [100, 300],
            "depth": [4, 6, 8],
            "learning_rate": [0.05, 0.1, 0.2],
            "l2_leaf_reg": [1, 3, 5],
        },
        "needs_dense": True, "dl": False,
    })

# 9 — ANN
if _tf:
    MODEL_REGISTRY.append({
        "ad": "ANN (Keras)", "kisa": "ANN",
        "estimator": None, "param_grid": {},
        "needs_dense": True, "dl": True, "dl_tip": "ann",
    })

# 10 — CNN-1D
if _tf:
    MODEL_REGISTRY.append({
        "ad": "CNN-1D (Tabular)", "kisa": "CNN",
        "estimator": None, "param_grid": {},
        "needs_dense": True, "dl": True, "dl_tip": "cnn",
        "akademik_not": (
            "CNN-1D tablolu veride özellik sırası bağımlılığı varsayar. "
            "Bu veri setinde sensör özellikleri birbirinden bağımsız ölçüldüğünden "
            "teorik avantajı geçerli değildir; akademik karşılaştırma amacıyla dahil edilmiştir."
        ),
    })

# 11 — LSTM
if _tf:
    MODEL_REGISTRY.append({
        "ad": "RNN-LSTM (Tabular)", "kisa": "LSTM",
        "estimator": None, "param_grid": {},
        "needs_dense": True, "dl": True, "dl_tip": "lstm",
        "akademik_not": (
            "LSTM zamansal seri bağımlılığı için tasarlanmıştır. "
            "Sensör tabanlı kesitsel veride bu bağımlılık yapısı mevcut olmadığından "
            "teorik avantajı geçerli değildir; kıyaslama amacıyla uygulanmıştır."
        ),
    })

adim(f"Toplam {len(MODEL_REGISTRY)} model:")
for m in MODEL_REGISTRY:
    adim(f"  [{m['kisa']:5s}] {m['ad']}")
tamam("Model kayıt defteri hazır")


# =============================================================================
# 5. DERİN ÖĞRENME YARDIMCI FONKSİYONLARI (Regresyon)
# =============================================================================

def ann_olustur(n_feat: int) -> "tf.keras.Model":
    """
    ANN — tam bağlantılı, tablolu regresyon.
    Mimari: Input → Dense(256,relu) → Dropout(0.3)
                  → Dense(128,relu) → Dropout(0.3)
                  → Dense(64,relu)  → Dropout(0.2)
                  → Dense(1,linear)
    """
    model = Sequential([
        Input(shape=(n_feat,)),
        Dense(256, activation="relu"),
        Dropout(0.3),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="linear"),
    ], name="ANN_regression")
    model.compile(optimizer=Adam(1e-3), loss="mse", metrics=["mae"])
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
        Dense(1, activation="linear"),
    ], name="CNN1D_regression")
    model.compile(optimizer=Adam(1e-3), loss="mse", metrics=["mae"])
    return model


def lstm_olustur(n_feat: int) -> "tf.keras.Model":
    """
    LSTM — zaman adımı=1 olarak tablolu veriyi işler.
    """
    model = Sequential([
        Input(shape=(n_feat,)),
        Reshape((1, n_feat)),
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1, activation="linear"),
    ], name="LSTM_regression")
    model.compile(optimizer=Adam(1e-3), loss="mse", metrics=["mae"])
    return model


def dl_egit(dl_tip: str, X_tr, y_tr, X_te, y_te,
            n_feat: int) -> tuple:
    """
    Keras regresyon modeli eğit ve değerlendir.
    Returns: (model, y_pred, egitim_sure_s)
    """
    X_tr_d = to_dense(X_tr).astype("float32")
    X_te_d = to_dense(X_te).astype("float32")
    y_tr_v = y_tr.values.astype("float32")

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
        X_tr_d, y_tr_v,
        epochs=DL_EPOCHS, batch_size=DL_BATCH_SIZE,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1,
    )
    sure = time.time() - t0

    y_pred = model.predict(X_te_d, verbose=1).ravel()
    return model, y_pred, sure


# =============================================================================
# 6. MODEL EĞİTİMİ VE DEĞERLENDİRMESİ
# =============================================================================

bolum("BÖLÜM 3 — Model Eğitimi ve Değerlendirmesi")

from sklearn.model_selection import GridSearchCV, KFold

kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

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
            model_obj, y_pred, egitim_sure = dl_egit(
                kayit["dl_tip"],
                X_train, y_train,
                X_test, y_test,
                n_features,
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

            X_cv_m = to_dense(X_cv)    if nd else X_cv
            X_tr_m = to_dense(X_train) if nd else X_train
            X_te_m = to_dense(X_test)  if nd else X_test

            if param_grid:
                gs = GridSearchCV(
                    estimator=estimator, param_grid=param_grid,
                    scoring="r2", cv=kf,
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
                best_cv_r2  = round(gs.best_score_, 5)
                log.info(f"[{kisa}] Best: {best_par}  CV-R²: {best_cv_r2:.5f}")
                cv_sonuclari.append({
                    "Model": ad, "Kisa": kisa,
                    "Best_Params": str(best_par),
                    "CV_R2": best_cv_r2,
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

        # ── Metrikler ──────────────────────────────────────────────────────
        metrik = metrikleri_hesapla(y_test, y_pred)
        metrik.update({
            "Model": ad, "Kisa": kisa,
            "Egitim_Sure_s": round(egitim_sure, 2),
            "DL": dl,
            "Akademik_Not": kayit.get("akademik_not", ""),
        })
        sonuc_listesi.append(metrik)
        adim(f"  [{kisa}] R²={metrik['R2']:.5f}  RMSE={metrik['RMSE']:.4f}  "
             f"MAE={metrik['MAE']:.4f}  ({egitim_sure:.1f}s)")

        # y_pred kaydet (scatter için)
        pd.Series(y_pred, name="y_pred").to_csv(
            SONUC / f"ypred_{kisa}.csv", index=False)

    except Exception as e:
        log.error(f"[{kisa}] HATA: {e}", exc_info=True)
        sonuc_listesi.append({
            "Model": ad, "Kisa": kisa, "HATA": str(e),
            "R2": None, "MSE": None, "RMSE": None, "MAE": None,
        })
        print(f"  ✗ [{kisa}] HATA: {e}", flush=True)

tamam("Tüm model eğitimleri tamamlandı")


# =============================================================================
# 7. SONUÇ TABLOLARI
# =============================================================================

bolum("BÖLÜM 4 — Sonuç Tabloları")

sonuc_df = pd.DataFrame(sonuc_listesi)
kolon_sirasi = [
    "Model", "Kisa", "R2", "RMSE", "MSE", "MAE",
    "Egitim_Sure_s", "DL", "Akademik_Not",
]
sonuc_df = sonuc_df[[c for c in kolon_sirasi if c in sonuc_df.columns]]
sonuc_df = sonuc_df.sort_values("R2", ascending=False).reset_index(drop=True)
sonuc_df.insert(0, "Sira", range(1, len(sonuc_df) + 1))
csv_kaydet(sonuc_df, "01_tum_model_sonuclari.csv", index=False)

if cv_sonuclari:
    cv_df = pd.DataFrame(cv_sonuclari)
    csv_kaydet(cv_df, "02_gridsearchcv_ozet.csv", index=False)

best_row   = sonuc_df[sonuc_df["R2"].notna()].iloc[0]
best_kisa  = best_row["Kisa"]
best_model = model_deposu.get(best_kisa)

adim(f"En iyi: [{best_kisa}] {best_row['Model']}")
adim(f"  R²={best_row['R2']}  RMSE={best_row['RMSE']}  MAE={best_row['MAE']}")

# y_pred best model
best_ypred_path = SONUC / f"ypred_{best_kisa}.csv"
if best_ypred_path.exists():
    y_pred_best = pd.read_csv(best_ypred_path)["y_pred"].values
else:
    bk_reg = next((kr for kr in MODEL_REGISTRY if kr["kisa"] == best_kisa), None)
    if bk_reg and not bk_reg["dl"] and best_model is not None:
        Xtp = to_dense(X_test) if bk_reg["needs_dense"] else X_test
        y_pred_best = best_model.predict(Xtp)
    elif bk_reg and bk_reg["dl"] and best_model is not None:
        y_pred_best = best_model.predict(
            to_dense(X_test).astype("float32"), verbose=0).ravel()
    else:
        y_pred_best = np.zeros(n_test)

tamam("Sonuç tabloları hazır")


# =============================================================================
# 8. GÖRSELLEŞTİRME
# =============================================================================

bolum("BÖLÜM 5 — Görselleştirmeler")

df_plot = sonuc_df[sonuc_df["R2"].notna()].copy()

# ── 8.1  R² Barplot ───────────────────────────────────────────────────────
df_b = df_plot.sort_values("R2", ascending=True)
fig, ax = plt.subplots(figsize=(10, 6))
renkler = ["#e74c3c" if k == best_kisa else "#3498db" for k in df_b["Kisa"]]
bars = ax.barh(df_b["Model"], df_b["R2"], color=renkler, edgecolor="white")
ax.axvline(0, color="gray", linestyle="--", linewidth=1, label="R²=0 (Null model)")
ax.axvline(1, color="#27ae60", linestyle=":", linewidth=1, label="R²=1 (Mükemmel)")
for bar, val in zip(bars, df_b["R2"]):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", ha="left", fontsize=9)
ax.set_xlabel("R² Skoru (Test Seti)", fontsize=11)
ax.set_title("Görev 2: suitability_score — R² Karşılaştırması\n"
             "(kırmızı = en iyi model)", fontsize=12)
ax.set_xlim(min(-0.1, df_b["R2"].min() - 0.05), 1.08)
ax.legend(fontsize=9)
plt.tight_layout()
png_kaydet(fig, "01_r2_karsilastirma.png")
adim("01 — R² barplot ✓")

# ── 8.2  RMSE Barplot ─────────────────────────────────────────────────────
df_rmse = df_plot.sort_values("RMSE", ascending=False)
fig, ax = plt.subplots(figsize=(10, 6))
rr = ["#e74c3c" if k == best_kisa else "#e67e22" for k in df_rmse["Kisa"]]
brs = ax.barh(df_rmse["Model"], df_rmse["RMSE"], color=rr, edgecolor="white")
for bar, val in zip(brs, df_rmse["RMSE"]):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", ha="left", fontsize=9)
ax.set_xlabel("RMSE (Test Seti)", fontsize=11)
ax.set_title("Görev 2: suitability_score — RMSE Karşılaştırması\n"
             "(düşük = iyi, kırmızı = en iyi model)", fontsize=12)
plt.tight_layout()
png_kaydet(fig, "02_rmse_karsilastirma.png")
adim("02 — RMSE barplot ✓")

# ── 8.3  MSE / RMSE / MAE Grouped Bar ────────────────────────────────────
df_err = df_plot.set_index("Model")[["MSE", "RMSE", "MAE"]].dropna()
fig, ax = plt.subplots(figsize=(12, 6))
df_err.plot(kind="bar", ax=ax, color=["#e74c3c", "#e67e22", "#9b59b6"],
            edgecolor="white", width=0.7)
ax.set_xticklabels(df_err.index, rotation=40, ha="right", fontsize=9)
ax.set_ylabel("Hata Değeri", fontsize=11)
ax.set_title("Görev 2: suitability_score — MSE / RMSE / MAE", fontsize=12)
ax.legend(fontsize=10)
plt.tight_layout()
png_kaydet(fig, "03_hata_metrikleri.png")
adim("03 — MSE/RMSE/MAE ✓")

# ── 8.4  En İyi Model — Gerçek vs Tahmin Scatter ─────────────────────────
if len(y_pred_best) == len(y_test):
    fig, ax = plt.subplots(figsize=(8, 7))
    lim_min = min(float(y_test.min()), float(y_pred_best.min())) - 2
    lim_max = max(float(y_test.max()), float(y_pred_best.max())) + 2
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", lw=1.5,
            label="Mükemmel Tahmin")
    # örneklem (büyük test setleri için)
    n_scatter = min(5000, len(y_test))
    idx_s = np.random.choice(len(y_test), n_scatter, replace=False)
    ax.scatter(np.array(y_test)[idx_s], y_pred_best[idx_s],
               alpha=0.3, s=12, color="#3498db", edgecolors="none")
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_xlabel("Gerçek suitability_score", fontsize=11)
    ax.set_ylabel("Tahmin edilen suitability_score", fontsize=11)
    ax.set_title(
        f"Görev 2: Gerçek vs Tahmin — [{best_kisa}] {best_row['Model']}\n"
        f"R²={best_row['R2']}  RMSE={best_row['RMSE']}  MAE={best_row['MAE']}",
        fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    png_kaydet(fig, f"04_scatter_{best_kisa}.png")
    adim(f"04 — Scatter [{best_kisa}] ✓")

# ── 8.5  Residual Dağılımı ────────────────────────────────────────────────
if len(y_pred_best) == len(y_test):
    residuals = np.array(y_test) - y_pred_best
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residual histogram
    axes[0].hist(residuals, bins=60, color="#3498db", edgecolor="white", alpha=0.8)
    axes[0].axvline(0, color="#e74c3c", lw=1.5, linestyle="--")
    axes[0].axvline(residuals.mean(), color="#27ae60", lw=1.5,
                    linestyle="-", label=f"Ort.={residuals.mean():.3f}")
    axes[0].set_xlabel("Artık (Gerçek − Tahmin)", fontsize=11)
    axes[0].set_ylabel("Frekans", fontsize=11)
    axes[0].set_title(f"Residual Dağılımı — [{best_kisa}]", fontsize=11)
    axes[0].legend(fontsize=9)

    # Residual vs Tahmin
    n_res = min(5000, len(y_pred_best))
    idx_r = np.random.choice(len(y_pred_best), n_res, replace=False)
    axes[1].scatter(y_pred_best[idx_r], residuals[idx_r],
                    alpha=0.3, s=12, color="#9b59b6", edgecolors="none")
    axes[1].axhline(0, color="#e74c3c", lw=1.5, linestyle="--")
    axes[1].set_xlabel("Tahmin edilen suitability_score", fontsize=11)
    axes[1].set_ylabel("Artık (Gerçek − Tahmin)", fontsize=11)
    axes[1].set_title(f"Artık vs Tahmin — [{best_kisa}]", fontsize=11)

    plt.suptitle(
        f"Görev 2: Artık Analizi — [{best_kisa}] {best_row['Model']}", fontsize=12, y=1.01)
    plt.tight_layout()
    png_kaydet(fig, f"05_residual_{best_kisa}.png")
    adim(f"05 — Residual analizi [{best_kisa}] ✓")

# ── 8.6  Tüm Modeller Scatter Grid ───────────────────────────────────────
ncols = 3
valids = df_plot["Kisa"].tolist()
nrows  = (len(valids) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
axes = axes.flatten()
for idx, kisa_ in enumerate(valids):
    ax = axes[idx]
    ypred_f = SONUC / f"ypred_{kisa_}.csv"
    if not ypred_f.exists():
        ax.set_visible(False)
        continue
    yp_ = pd.read_csv(ypred_f)["y_pred"].values
    r2_ = df_plot.loc[df_plot["Kisa"] == kisa_, "R2"].values
    r2_ = r2_[0] if len(r2_) else None
    n_s = min(2000, len(y_test))
    idx_s = np.random.choice(len(y_test), n_s, replace=False)
    ax.scatter(np.array(y_test)[idx_s], yp_[idx_s],
               alpha=0.25, s=8, color="#3498db" if kisa_ != best_kisa else "#e74c3c",
               edgecolors="none")
    lm = max(float(y_test.max()), float(yp_.max()))
    ax.plot([0, lm], [0, lm], "k--", lw=1)
    title_str = f"[{kisa_}] R²={r2_:.4f}" if r2_ is not None else f"[{kisa_}]"
    ax.set_title(title_str, fontsize=10)
    ax.set_xlabel("Gerçek", fontsize=8)
    ax.set_ylabel("Tahmin", fontsize=8)
for j in range(len(valids), len(axes)):
    axes[j].set_visible(False)
plt.suptitle("Görev 2: Tüm Modeller — Gerçek vs Tahmin", fontsize=13, y=1.01)
plt.tight_layout()
png_kaydet(fig, "06_scatter_tum_modeller.png")
adim("06 — Tüm modeller scatter ✓")

# ── 8.7  R² - MAE İkili Karşılaştırma ────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(12, 6))
x_pos = np.arange(len(df_plot))
bar_w = 0.35
ax2 = ax1.twinx()
bars1 = ax1.bar(x_pos - bar_w / 2, df_plot["R2"], bar_w,
                color="#3498db", alpha=0.8, label="R²")
bars2 = ax2.bar(x_pos + bar_w / 2, df_plot["MAE"], bar_w,
                color="#e74c3c", alpha=0.8, label="MAE")
ax1.set_xticks(x_pos)
ax1.set_xticklabels(df_plot["Model"], rotation=40, ha="right", fontsize=9)
ax1.set_ylabel("R² (yüksek = iyi)", fontsize=11, color="#3498db")
ax2.set_ylabel("MAE (düşük = iyi)", fontsize=11, color="#e74c3c")
ax1.tick_params(axis="y", labelcolor="#3498db")
ax2.tick_params(axis="y", labelcolor="#e74c3c")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)
ax1.set_title("Görev 2: suitability_score — R² ve MAE Karşılaştırması", fontsize=12)
plt.tight_layout()
png_kaydet(fig, "07_r2_mae_karsilastirma.png")
adim("07 — R² & MAE ikili ✓")

tamam("Tüm görselleştirmeler tamamlandı")


# =============================================================================
# 9. SHAP ANALİZİ
# =============================================================================

bolum("BÖLÜM 6 — SHAP Analizi (Açıklanabilir Yapay Zeka / XAI)")

shap_ok    = False
shap_top3  = ""

try:
    if not _shap:
        raise ImportError("shap")
    try:
        shap.initjs()
    except Exception:
        pass  # IPython kurulu değilse sessizce geç

    bk_reg = next((kr for kr in MODEL_REGISTRY if kr["kisa"] == best_kisa), None)
    if bk_reg is None or best_model is None:
        raise RuntimeError("En iyi model bulunamadı.")

    X_te_d = to_dense(X_test)
    n_shap = min(SHAP_SAMPLE_N, n_test)
    idx_shap = np.random.choice(n_test, n_shap, replace=False)
    X_shap = X_te_d[idx_shap]

    feat_n = feature_names if len(feature_names) == n_features else [
        f"F{i}" for i in range(n_features)
    ]

    # TreeExplainer: ağaç tabanlı modeller için hızlı ve kesin
    tree_models = {"RF", "XGB", "LGBM", "CAT", "DT"}
    adim(f"SHAP Explainer tipi belirleniyor — [{best_kisa}]")

    if best_kisa in tree_models and not bk_reg["dl"]:
        explainer   = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_shap)
        adim("TreeExplainer kullanıldı (ağaç tabanlı — hızlı ve kesin)")
    elif not bk_reg["dl"]:
        # Linear/KNN/SVR için KernelExplainer
        X_bg = shap.sample(X_shap, min(200, len(X_shap)))
        explainer   = shap.KernelExplainer(best_model.predict, X_bg)
        shap_values = explainer.shap_values(X_shap[:500], nsamples=100)
        adim("KernelExplainer kullanıldı (model-agnostik)")
    else:
        # DL model
        X_bg = shap.sample(X_shap, min(100, len(X_shap)))
        explainer   = shap.KernelExplainer(
            lambda x: best_model.predict(x.astype("float32"), verbose=0).ravel(), X_bg)
        shap_values = explainer.shap_values(X_shap[:300], nsamples=50)
        adim("KernelExplainer kullanıldı (Keras DL)")

    shap_arr = np.array(shap_values)
    if shap_arr.ndim > 2:
        shap_arr = shap_arr[0]

    mean_abs_shap = np.abs(shap_arr).mean(axis=0)
    shap_df = pd.DataFrame({
        "feature": feat_n[:len(mean_abs_shap)],
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    csv_kaydet(shap_df, "shap_feature_importance.csv", klasor=SHAP_K, index=False)

    top3 = shap_df.head(3)
    shap_top3 = "\n".join(
        f"  {i+1}. {row['feature']} (|SHAP| ort={row['mean_abs_shap']:.4f})"
        for i, (_, row) in enumerate(top3.iterrows())
    )
    adim(f"Top-3 SHAP:\n{shap_top3}")

    # ── SHAP Bar Plot ──────────────────────────────────────────────────────
    top_n = SHAP_TOP_N
    shap_top = shap_df.head(top_n)
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))
    bars = ax.barh(shap_top["feature"][::-1], shap_top["mean_abs_shap"][::-1],
                   color="#e74c3c", edgecolor="white")
    for bar, val in zip(bars, shap_top["mean_abs_shap"][::-1]):
        ax.text(bar.get_width() + 0.0001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left", fontsize=8)
    ax.set_xlabel("Ortalama |SHAP Değeri|", fontsize=11)
    ax.set_title(
        f"Görev 2: SHAP Özellik Önemi — [{best_kisa}] {best_row['Model']}\n"
        f"(suitability_score tahminine katkı — Top {top_n})", fontsize=11)
    plt.tight_layout()
    png_kaydet(fig, f"shap_01_bar_{best_kisa}.png", klasor=SHAP_K)
    adim("SHAP bar plot ✓")

    # ── SHAP Beeswarm Plot ────────────────────────────────────────────────
    try:
        exp_obj = shap.Explanation(
            values=shap_arr,
            base_values=np.full(len(shap_arr), explainer.expected_value
                                if hasattr(explainer, "expected_value") else 0.0),
            data=X_shap[:len(shap_arr)],
            feature_names=feat_n[:shap_arr.shape[1]],
        )
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.plots.beeswarm(exp_obj, max_display=top_n, show=False)
        plt.title(f"SHAP Beeswarm — [{best_kisa}] suitability_score", fontsize=11)
        plt.tight_layout()
        png_kaydet(fig, f"shap_02_beeswarm_{best_kisa}.png", klasor=SHAP_K)
        adim("SHAP beeswarm ✓")
    except Exception as e:
        log.warning(f"SHAP beeswarm: {e}")

    # ── SHAP Dependence Plot — Top 2 özellik ─────────────────────────────
    for i_dep in range(min(2, len(shap_df))):
        dep_feat = shap_df.iloc[i_dep]["feature"]
        feat_idx = feat_n.index(dep_feat) if dep_feat in feat_n else i_dep
        if feat_idx < shap_arr.shape[1]:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(X_shap[:, feat_idx], shap_arr[:, feat_idx],
                       alpha=0.3, s=10, color="#3498db", edgecolors="none")
            ax.axhline(0, color="gray", lw=1, linestyle="--")
            ax.set_xlabel(dep_feat, fontsize=11)
            ax.set_ylabel(f"SHAP Değeri ({dep_feat})", fontsize=11)
            ax.set_title(
                f"SHAP Dependence — {dep_feat}\n"
                f"suitability_score üzerindeki etkisi", fontsize=11)
            plt.tight_layout()
            png_kaydet(fig, f"shap_03_dep_{i_dep+1}_{best_kisa}.png",
                       klasor=SHAP_K)
    adim("SHAP dependence plot ✓")

    shap_ok = True
    tamam("SHAP analizi tamamlandı")

except ImportError:
    adim("⚠ SHAP kurulu değil — pip install shap")
except Exception as e:
    log.error(f"SHAP HATA: {e}", exc_info=True)
    adim(f"⚠ SHAP başarısız: {e}")


# =============================================================================
# 10. DUYARLILIK (SENSİTİVİTE) ANALİZİ
# =============================================================================

bolum("BÖLÜM 7 — Duyarlılık Analizi (Gaussian Gürültü)")

duyar_ok     = False
duyar_blok   = ""

try:
    if ham_test_df is None:
        raise RuntimeError("Ham test verisi yüklenmemiş.")
    if best_model is None:
        raise RuntimeError("En iyi model yüklenmemiş.")

    bk_reg = next((kr for kr in MODEL_REGISTRY if kr["kisa"] == best_kisa), None)
    if bk_reg is None:
        raise RuntimeError("Model kaydı bulunamadı.")

    # Preprocessor'a giren sütunları belirle (hedef sütunları çıkar)
    hedefler = ["failure_flag", "suitability_score", "stress_level", "location_id"]
    X_ham_test = ham_test_df.drop(columns=[c for c in hedefler if c in ham_test_df.columns],
                                  errors="ignore")
    y_ham_test = ham_test_df["suitability_score"] if "suitability_score" in ham_test_df.columns \
        else y_test.reset_index(drop=True)

    # Sayısal sütunları belirle (gürültü sadece bunlara eklenir)
    num_cols = X_ham_test.select_dtypes(include=[np.number]).columns.tolist()
    adim(f"Gürültü uygulanacak sayısal sütun sayısı: {len(num_cols)}")
    adim(f"Gürültü oranları: {GURULTU_ORANLARI}  |  Her oran tekrarı: {GURULTU_TEKRAR}")

    duyar_satirlar = []
    # Temiz (gürültüsüz) baseline
    X_clean_proc = preprocessor.transform(X_ham_test)
    if bk_reg["dl"]:
        y_clean_pred = best_model.predict(
            to_dense(X_clean_proc).astype("float32"), verbose=0).ravel()
    elif bk_reg["needs_dense"]:
        y_clean_pred = best_model.predict(to_dense(X_clean_proc))
    else:
        y_clean_pred = best_model.predict(X_clean_proc)
    clean_met = metrikleri_hesapla(y_ham_test, y_clean_pred)
    duyar_satirlar.append({
        "Gurultu_Orani": 0.0,
        "Tekrar": 0,
        "R2": clean_met["R2"],
        "RMSE": clean_met["RMSE"],
        "MAE": clean_met["MAE"],
    })
    adim(f"Baseline (gürültüsüz) — R²={clean_met['R2']:.5f}  "
         f"RMSE={clean_met['RMSE']:.4f}  MAE={clean_met['MAE']:.4f}")

    np.random.seed(RANDOM_STATE)
    for oran in GURULTU_ORANLARI:
        for tekrar in range(1, GURULTU_TEKRAR + 1):
            X_gurultulu = X_ham_test.copy()
            for col in num_cols:
                std_val = X_gurultulu[col].std()
                if std_val > 0:
                    gurultu = np.random.normal(0, oran * std_val, size=len(X_gurultulu))
                    X_gurultulu[col] = X_gurultulu[col] + gurultu
            X_proc = preprocessor.transform(X_gurultulu)
            if bk_reg["dl"]:
                y_pred_g = best_model.predict(
                    to_dense(X_proc).astype("float32"), verbose=0).ravel()
            elif bk_reg["needs_dense"]:
                y_pred_g = best_model.predict(to_dense(X_proc))
            else:
                y_pred_g = best_model.predict(X_proc)
            m = metrikleri_hesapla(y_ham_test, y_pred_g)
            duyar_satirlar.append({
                "Gurultu_Orani": oran,
                "Tekrar": tekrar,
                "R2": m["R2"],
                "RMSE": m["RMSE"],
                "MAE": m["MAE"],
            })
        tekrar_sonuclar = [d for d in duyar_satirlar
                           if d["Gurultu_Orani"] == oran]
        r2_ort   = np.mean([d["R2"]   for d in tekrar_sonuclar])
        rmse_ort = np.mean([d["RMSE"] for d in tekrar_sonuclar])
        mae_ort  = np.mean([d["MAE"]  for d in tekrar_sonuclar])
        adim(f"%{int(oran*100):2d} gürültü — R²={r2_ort:.5f}  "
             f"RMSE={rmse_ort:.4f}  MAE={mae_ort:.4f}")

    duyar_df = pd.DataFrame(duyar_satirlar)
    csv_kaydet(duyar_df, "duyarlilik_ham.csv", klasor=DUYAR_K, index=False)

    duyar_ozet = duyar_df.groupby("Gurultu_Orani").agg(
        R2_ort=("R2", "mean"),
        R2_std=("R2", "std"),
        RMSE_ort=("RMSE", "mean"),
        RMSE_std=("RMSE", "std"),
        MAE_ort=("MAE", "mean"),
        MAE_std=("MAE", "std"),
    ).reset_index()
    csv_kaydet(duyar_ozet, "duyarlilik_ozet.csv", klasor=DUYAR_K, index=False)

    # ── Duyarlılık Grafiği — R² ───────────────────────────────────────────
    oranlabels = [f"%{int(r*100)}" for r in duyar_ozet["Gurultu_Orani"]]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for i, (met, label, renk) in enumerate([
        ("R2_ort",   "R²",   "#3498db"),
        ("RMSE_ort", "RMSE", "#e67e22"),
        ("MAE_ort",  "MAE",  "#9b59b6"),
    ]):
        std_col = met.replace("_ort", "_std")
        axes[i].plot(oranlabels, duyar_ozet[met], "o-", color=renk,
                     linewidth=2, markersize=7)
        if std_col in duyar_ozet.columns:
            axes[i].fill_between(
                oranlabels,
                duyar_ozet[met] - duyar_ozet[std_col],
                duyar_ozet[met] + duyar_ozet[std_col],
                alpha=0.15, color=renk)
        axes[i].set_xlabel("Gaussian Gürültü Oranı", fontsize=10)
        axes[i].set_ylabel(label, fontsize=10)
        axes[i].set_title(f"Duyarlılık — {label}", fontsize=10)
        axes[i].grid(True, alpha=0.3)
    plt.suptitle(
        f"Görev 2: Duyarlılık Analizi — [{best_kisa}] {best_row['Model']}\n"
        f"(suitability_score — sensör gürültüsüne karşı dayanıklılık)", fontsize=12)
    plt.tight_layout()
    png_kaydet(fig, f"duyarlilik_metriks_{best_kisa}.png", klasor=DUYAR_K)
    adim("Duyarlılık grafiği ✓")

    # Rapor bloğu
    duyar_blok_satirlar = ["| Gürültü | R² Ort. | RMSE Ort. | MAE Ort. |",
                           "|---------|---------|-----------|---------|"]
    for _, row_ in duyar_ozet.iterrows():
        lbl = "Baseline" if row_["Gurultu_Orani"] == 0 else f"%{int(row_['Gurultu_Orani']*100)}"
        duyar_blok_satirlar.append(
            f"| {lbl} | {row_['R2_ort']:.5f} | {row_['RMSE_ort']:.4f} | {row_['MAE_ort']:.4f} |"
        )
    duyar_blok = "\n".join(duyar_blok_satirlar)

    duyar_ok = True
    tamam("Duyarlılık analizi tamamlandı")

except Exception as e:
    log.error(f"Duyarlılık HATA: {e}", exc_info=True)
    adim(f"⚠ Duyarlılık analizi başarısız: {e}")
    duyar_blok = f"Hata: {e}"


# =============================================================================
# 11. AKADEMİK RAPOR
# =============================================================================

bolum("BÖLÜM 8 — Akademik Rapor")

try:
    rapor_tablo = sonuc_df[
        ["Sira", "Model", "Kisa", "R2", "RMSE", "MSE", "MAE", "Egitim_Sure_s", "DL"]
    ].to_markdown(index=False)
except Exception:
    rapor_tablo = sonuc_df[
        ["Sira", "Model", "Kisa", "R2", "RMSE", "MSE", "MAE", "Egitim_Sure_s", "DL"]
    ].to_string(index=False)

rapor_metni = f"""# AŞAMA 3 — GÖREV 2 AKADEMİK RAPORU
# Uygunluk Skoru Tahmini (Regresyon — suitability_score)

## Öğrenci
İbrahim Nuryağınlı | 25490221001

## Aşama 1 Referansı
Aşama 1 Klasörü : {A1_DIR.name}
Özellik Sayısı  : {n_features}
Train           : {n_train:,} satır (SMOTE uygulanmadı — regresyon)
Test            : {n_test:,} satır
GridSearchCV    : {CV_FOLDS}-fold, örneklem={CV_SAMPLE_N:,} satır

---

## 1. Problem Tanımı
suitability_score değişkeni (0–100 arası sürekli sayısal değer) için regresyon problemi.
Hedef değişken sürekli yapıda olduğundan SMOTE uygulanmaz; orijinal train seti kullanılır.
Değerlendirme metrikleri: R² (açıklama gücü), MSE, RMSE, MAE.

## 2. Metodoloji

### Veri Hazırlığı
- Tüm preprocessing (PowerTransformer, StandardScaler, OneHotEncoder) Aşama 1'de
  yalnızca train seti üzerinde fit edilmiştir → veri sızıntısı yoktur.
- GridSearchCV için train setinden {CV_SAMPLE_N:,} satır örneklenmiş,
  en iyi parametreler tam train ({n_train:,} satır) ile yeniden eğitilmiştir.
- Test seti yalnızca final değerlendirmede bir kez kullanılmıştır.

### Neden R² Temel Metrik?
R² (determinasyon katsayısı), modelin bağımsız değişkenler aracılığıyla hedef
değişkendeki varyansı ne ölçüde açıkladığını gösterir. R²=1 mükemmel uyumu,
R²=0 modelin ortalama tahmin ile eşdeğer olduğunu, R²<0 ise modelin anlamsız
olduğunu ifade eder. RMSE hata büyüklüğünü orijinal birim cinsinden ölçerken,
MAE aykırı değerlere daha dayanıklı bir alternatif sunar.

---

## 3. Denemeler

### Denenecek Modeller ({len(MODEL_REGISTRY)} adet)
Linear Regression, Decision Tree Regressor, KNN Regressor, SVR (Linear),
Random Forest Regressor, XGBoost Regressor, LightGBM Regressor, CatBoost Regressor,
ANN, CNN-1D, RNN-LSTM

### SVR Notu
RBF kernel SVR O(n²)-O(n³) karmaşıklığı nedeniyle 500k+ satırlık veri setinde
uygulanabilir değildir. LinearSVR, lineer karmaşıklığıyla büyük ölçekli veri
setleri için akademik açıdan geçerli bir alternatiftir.

---

## 4. Model Performans Tablosu (Test Seti — R²'ye göre)

{rapor_tablo}

---

## 5. En İyi Model
Model    : {best_row['Model']} [{best_kisa}]
R²       : {best_row['R2']}
RMSE     : {best_row['RMSE']}
MSE      : {best_row['MSE']}
MAE      : {best_row['MAE']}

---

## 6. SHAP Analizi (Açıklanabilir Yapay Zeka / XAI)
SHAP (SHapley Additive exPlanations) yöntemi ile en iyi model olan
[{best_kisa}]'ın hangi sensör değişkenlerini esas aldığı görselleştirilmiştir.

Explainer: {"TreeExplainer (ağaç tabanlı modeller için hızlı ve kesin)" if shap_ok else "Çalıştırılamadı / SHAP kurulu değil"}
Örneklem : {SHAP_SAMPLE_N:,} test örneği

### Top 3 Belirleyici Özellik (|SHAP| ortalaması)
{shap_top3 if shap_top3 else "  SHAP çalıştırılamadı."}

Üretilen SHAP görselleştirmeleri:
  - Bar Plot      : Her özelliğin ortalama mutlak SHAP değeri
  - Beeswarm Plot : Özellik değeri (kırmızı=yüksek) ile SHAP etkisi
  - Dependence    : En önemli 2 özelliğin SHAP etkisi vs özellik değeri

---

## 7. Duyarlılık Analizi (Gaussian Gürültü)
Modelin gerçek saha koşullarındaki sensör gürültüsüne ne kadar dayanıklı
olduğunu test etmek amacıyla ham test verisinin sayısal sütunlarına
std-orantılı sıfır ortalamalı Gaussian gürültü eklenmiştir.

{duyar_blok}

### Yorum
Gürültü oranı arttıkça R² değerindeki düşüş ve RMSE/MAE'deki artış modelin
sensör hatalarına karşı dayanıklılığını ölçer. Küçük değişimler modelin
saha koşullarına karşı kararlı olduğunu; büyük değişimler ise özellik
kalitesinin tahmin başarısı için kritik önem taşıdığını göstermektedir.

---

## 8. Derin Öğrenme Modelleri — Akademik Yorum
{chr(10).join(dl_notlar) if dl_notlar else "Derin öğrenme modelleri çalıştırılmadı."}

Tablolu veri setlerinde CNN ve LSTM'nin zayıf regresyon performansı beklenen
bir bulgudur. Bu modeller sırasıyla uzamsal ve zamansal dizi bağımlılıkları
için tasarlanmıştır. Kesitsel sensör verisinde bu yapısal ön koşullar
sağlanmadığından performans kısıtlı kalır. Sonuçların raporlanması bilimsel
dürüstlük ve akademik kapsam açısından zorunludur.

---

## 9. Veri Sızıntısı Önlemleri
1. Preprocessing yalnızca train üzerinde fit (Aşama 1).
2. GridSearchCV yalnızca train alt-kümesine ({CV_SAMPLE_N:,} satır) uygulandı.
3. Test seti yalnızca final değerlendirmede bir kez kullanıldı.
4. Regresyon için SMOTE uygulanmadı; orijinal train dağılımı korundu.
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

rapor_path = RAPOR / "gorev2_akademik_rapor.md"
rapor_path.write_text(rapor_metni, encoding="utf-8")
adim(f"Rapor kaydedildi: {rapor_path.name}")
tamam("Akademik rapor tamamlandı")


# =============================================================================
# 12. PAKET (Aşama 4 için)
# =============================================================================

bolum("BÖLÜM 9 — Paket Kayıt (Aşama 4'e aktarım)")

a3_meta = {
    "run_id":          RUN_ID,
    "a1_dir":          str(A1_DIR),
    "a2_dir":          str(A2_DIR) if A2_DIR else None,
    "hedef":           HEDEF,
    "best_model_kisa": best_kisa,
    "best_model_ad":   best_row["Model"],
    "best_r2":         float(best_row["R2"])   if best_row["R2"]   else None,
    "best_rmse":       float(best_row["RMSE"]) if best_row["RMSE"] else None,
    "best_mae":        float(best_row["MAE"])  if best_row["MAE"]  else None,
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
    "metadata":      a3_meta,
    "sonuc_df":      sonuc_df,
    "feature_names": feature_names,
    "model_deposu":  {k: v for k, v in model_deposu.items() if not is_keras(v)},
}
joblib.dump(paket, CIKTI / "asama3_gorev2_paket.joblib")

with open(CIKTI / "asama3_gorev2_metadata.json", "w", encoding="utf-8") as f:
    json.dump(a3_meta, f, ensure_ascii=False, indent=2)

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
║          AŞAMA 3 — GÖREV 2 TAMAMLANDI                               ║
╠══════════════════════════════════════════════════════════════════════╣
║  Hedef      : suitability_score (Regresyon)                          ║
║  Train      : {n_train:,} satır  |  Test: {n_test:,}
║  Özellik    : {n_features}
║  Model      : {len(MODEL_REGISTRY)} adet
║  En İyi     : [{best_kisa}] {best_row['Model']}
║  R²         : {best_row['R2']}
║  RMSE       : {best_row['RMSE']}
║  MAE        : {best_row['MAE']}
║  SHAP       : {"✓" if shap_ok else "✗"}
║  Duyarlılık : {"✓" if duyar_ok else "✗"}
║  Görsel     : {n_gorsel}  |  CSV/MD: {n_csv}
║  Çıktı      : {CIKTI.name}
╚══════════════════════════════════════════════════════════════════════╝
""", flush=True)