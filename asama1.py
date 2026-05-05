# =============================================================================
# YZO 106 – İleri Düzey Makine Öğrenmesi | Dönem Projesi
# AŞAMA 1 – Veri Keşfi, Ön İşleme, Train/Test Ayrımı, Modelleme Veri Seti Hazırlığı
# Öğrenci: İbrahim Nuryağınlı | 25490221001
#
# Veri seti özellikleri (dataset.csv):
#   - 543.210 satır, 25 sütun
#   - Eksik değer yok, tekrar eden satır yok
#   - location_id: tam ID sütunu (her satır benzersiz) → model girdisinden çıkarılır
#   - Kategorik: soil_type (9), moisture_regime (3), thermal_regime (3),
#                nutrient_balance (3), plant_category (3)
#   - Sayısal (bağımsız): 16 sütun; organic_matter_pct ve salinity_ec yüksek skew
#   - ph_stress_flag: 0/1 binary ama sayısal tipte → sayısal pipeline'a dahil edilir
#   - failure_flag   → Binary classification  | imbalance ~5.17x → SMOTE uygulanır
#   - stress_level   → Multiclass (0/1/2)     | imbalance ~1.85x → eşik altı, SMOTE yok
#   - suitability_score → Regression [0,1]    → SMOTE uygulanmaz
#
# Metodoloji notları:
#   - EDA (korelasyon, IQR) tam veri üzerinde yapılır; keşifsel amaçlıdır.
#   - VIF, train/test ayrımından SONRA yalnızca train seti üzerinde hesaplanır.
#   - Preprocessor yalnızca train seti üzerinde fit edilir; test seti bu nesneyle
#     transform edilir. Bu yaklaşım veri sızıntısını önler.
#   - SMOTE yalnızca train setine uygulanır; test seti orijinal dağılımda bırakılır.
#   - Yüksek skew'li sayısal sütunlar (|skew| > 1.5) Yeo-Johnson PowerTransformer
#     ile dönüştürülür. Diğer sayısal sütunlar StandardScaler ile ölçeklenir.
#
# Gereksinimler:
#   pip install pandas numpy scikit-learn matplotlib seaborn joblib scipy imbalanced-learn
# =============================================================================

import hashlib
import json
import logging
import os
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
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

try:
    from imblearn.over_sampling import SMOTE
    _imblearn = True
except ImportError:
    _imblearn = False

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

pd.set_option("display.max_columns", 100)
pd.set_option("display.float_format", "{:.4f}".format)


# =============================================================================
# 0. KONFİGÜRASYON
# =============================================================================

RANDOM_STATE   = 42
TEST_ORANI     = 0.20
SKEW_ESIK      = 1.5    # |skew| > bu değer → PowerTransformer (Yeo-Johnson)
IMBALANCE_ESIK = 2.0    # max/min sınıf oranı > bu değer → SMOTE değerlendirilir
SMOTE_FAILURE  = True   # failure_flag için SMOTE aktif mi?

# Grafik ve analiz limitleri
MAX_NUMERIC_PLOT  = 30
MAX_HEATMAP_COLS  = 40
MAX_VIF_COLS      = 80
BOXPLOT_SAMPLE_N  = 50_000
MAX_CAT_TOP_N     = 50

# Hedef sütunlar
HEDEF_FAILURE = "failure_flag"
HEDEF_REG     = "suitability_score"
HEDEF_STRESS  = "stress_level"
HEDEFLER      = [HEDEF_FAILURE, HEDEF_REG, HEDEF_STRESS]


# =============================================================================
# 1. KLASÖR ve LOG KURULUMU
# =============================================================================

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

VERI = BASE_DIR / "dataset.csv"

RUN_ID    = datetime.now().strftime("%Y%m%d_%H%M%S")
CIKTI     = BASE_DIR / f"asama1_ciktilar_{RUN_ID}"
ANALIZ    = CIKTI / "01_analiz"
GRAFIK    = CIKTI / "02_gorseller"
DATA      = CIKTI / "03_model_verileri"
MODEL_DIR = CIKTI / "04_preprocessor"
RAPOR     = CIKTI / "05_raporlar"

for d in [CIKTI, ANALIZ, GRAFIK, DATA, MODEL_DIR, RAPOR]:
    d.mkdir(parents=True, exist_ok=True)

LOG_DOSYASI = RAPOR / f"asama1_log_{RUN_ID}.txt"

logger = logging.getLogger("asama1")
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


def csv_kaydet(df_obj: pd.DataFrame, dosya_adi: str, klasor: Path = ANALIZ, index: bool = True) -> Path:
    p = klasor / dosya_adi
    df_obj.to_csv(p, index=index, encoding="utf-8-sig")
    log.info(f"CSV: {p.name}")
    return p


def seri_kaydet(seri: pd.Series, dosya_adi: str) -> Path:
    p = DATA / dosya_adi
    seri.to_csv(p, index=False, encoding="utf-8-sig", header=[seri.name or "target"])
    log.info(f"Seri: {p.name}")
    return p


def png_kaydet(fig, dosya_adi: str) -> Path:
    p = GRAFIK / dosya_adi
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"PNG: {p.name}")
    return p


def matris_kaydet(X, stem: str) -> Path:
    if sparse.issparse(X):
        p = DATA / f"{stem}.npz"
        sparse.save_npz(p, X.tocsr())
    else:
        p = DATA / f"{stem}.npy"
        np.save(p, X)
    log.info(f"Matris: {p.name}  shape={X.shape}")
    return p


def sinif_dagilim(y: pd.Series, ad: str) -> pd.DataFrame:
    vc  = y.value_counts(dropna=False).sort_index()
    pct = y.value_counts(normalize=True, dropna=False).sort_index() * 100
    return pd.DataFrame({
        "Hedef": ad,
        "Sınıf": vc.index.astype(str),
        "Sayı":  vc.values,
        "Oran%": pct.values.round(4)
    })


def imbalance_ratio(y: pd.Series) -> float:
    vc = y.value_counts(dropna=False)
    if vc.empty or vc.min() <= 0:
        return float("nan")
    return float(vc.max() / vc.min())


def vif_hesapla(df_num: pd.DataFrame) -> pd.DataFrame:
    """VIF — sadece train sayısal sütunları üzerinde çalışır."""
    sonuc = []
    df_t  = df_num.replace([np.inf, -np.inf], np.nan).fillna(df_num.median(numeric_only=True))
    sabit = [c for c in df_t.columns if df_t[c].nunique() <= 1]
    aktif = [c for c in df_t.columns if c not in sabit]

    for col in sabit:
        sonuc.append({"Değişken": col, "R2": np.nan, "VIF": np.nan, "Not": "Sabit sütun"})

    if len(aktif) < 2:
        return pd.DataFrame(sonuc).set_index("Değişken")

    arr = df_t[aktif].values
    for i, col in enumerate(aktif):
        y   = arr[:, i]
        X_d = np.delete(arr, i, axis=1)
        try:
            pred    = LinearRegression().fit(X_d, y).predict(X_d)
            ss_res  = np.sum((y - pred) ** 2)
            ss_tot  = np.sum((y - y.mean()) ** 2)
            if ss_tot < 1e-12:
                r2, vif, not_ = np.nan, np.nan, "Varyans yok"
            else:
                r2   = min(max(1 - ss_res / ss_tot, 0), 0.9999)
                vif  = 1 / (1 - r2)
                not_ = ""
            sonuc.append({"Değişken": col, "R2": round(r2, 4), "VIF": round(vif, 2), "Not": not_})
        except Exception as e:
            sonuc.append({"Değişken": col, "R2": np.nan, "VIF": np.nan, "Not": str(e)})

    return (pd.DataFrame(sonuc)
              .set_index("Değişken")
              .sort_values("VIF", ascending=False, na_position="last"))


def smote_uygula(X_tr, y_tr: pd.Series, hedef: str, ratio: float):
    """SMOTE — yalnızca train setine, yalnızca sınıflandırma hedefleri için."""
    if ratio < IMBALANCE_ESIK:
        log.info(f"{hedef}: ratio={ratio:.2f} < eşik {IMBALANCE_ESIK}. SMOTE uygulanmadı.")
        return X_tr, y_tr.reset_index(drop=True), False, f"Ratio {ratio:.2f} < eşik"

    if not _imblearn:
        return X_tr, y_tr.reset_index(drop=True), False, "imbalanced-learn eksik"

    try:
        k = min(5, int(y_tr.value_counts().min()) - 1)
        if k < 1:
            return X_tr, y_tr.reset_index(drop=True), False, "Minimum sınıf örnek sayısı yetersiz"

        # SMOTE sparse CSR gerektirir
        X_in = X_tr.tocsr() if sparse.issparse(X_tr) else X_tr
        t0   = time.time()
        X_s, y_s = SMOTE(random_state=RANDOM_STATE, k_neighbors=k).fit_resample(X_in, y_tr)
        sure = time.time() - t0
        log.info(f"{hedef}: SMOTE {X_tr.shape[0]:,} → {X_s.shape[0]:,} satır ({sure:.1f}s)")
        return X_s, pd.Series(y_s, name=hedef), True, f"SMOTE ({sure:.1f}s)"

    except Exception as e:
        log.warning(f"{hedef}: SMOTE hatası: {e}")
        return X_tr, y_tr.reset_index(drop=True), False, str(e)


def dataset_hash(dosya: Path) -> str:
    h = hashlib.md5()
    with open(dosya, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    # =============================================================================
    # 3. VERİ YÜKLEME
    # =============================================================================

    bolum("BÖLÜM 1 — Veri Yükleme")

    if not VERI.exists():
        raise FileNotFoundError(f"dataset.csv bulunamadı → {VERI}")

    t0 = time.time()
    df = pd.read_csv(VERI)
    yukle_sure = time.time() - t0
    veri_hash  = dataset_hash(VERI)

    eksik_hedef = [c for c in HEDEFLER if c not in df.columns]
    if eksik_hedef:
        raise ValueError(f"Hedef sütunlar eksik: {eksik_hedef}")

    log.info(f"Dosya : {VERI}")
    log.info(f"Hash  : {veri_hash}")
    log.info(f"Shape : {df.shape}")
    log.info(f"Süre  : {yukle_sure:.2f}s")

    adim(f"{df.shape[0]:,} satır × {df.shape[1]} sütun yüklendi ({yukle_sure:.2f}s)")
    tamam("Veri yükleme tamamlandı")


    # =============================================================================
    # 4. GENEL VERİ KALİTESİ (EDA – Tam Veri)
    # =============================================================================
    # Not: Bu bölümdeki analizler (korelasyon, IQR) keşifsel amaçlıdır ve
    # tüm veri üzerinde yapılır. Hiçbir model kararı bu analizlere dayandırılmaz.
    # =============================================================================

    bolum("BÖLÜM 2 — Veri Kalitesi ve EDA (Keşifsel, Tam Veri)")

    # Sütun bilgi tablosu
    sutun_info = pd.DataFrame({
        "Sütun":          df.columns,
        "Dtype":          df.dtypes.astype(str).values,
        "Null_Sayı":      df.isnull().sum().values,
        "Null_Oran%":     (df.isnull().mean() * 100).round(4).values,
        "Benzersiz":      df.nunique(dropna=True).values,
        "Benzersiz_Oran": (df.nunique(dropna=True) / len(df)).round(6).values,
    })
    csv_kaydet(sutun_info, "01_sutun_bilgileri.csv", index=False)

    # Genel kalite özeti
    duplikat  = int(df.duplicated().sum())
    kalite_df = pd.DataFrame([
        {"Kontrol": "Satır",               "Değer": df.shape[0]},
        {"Kontrol": "Sütun",               "Değer": df.shape[1]},
        {"Kontrol": "Tam tekrar satır",    "Değer": duplikat},
        {"Kontrol": "Toplam eksik hücre",  "Değer": int(df.isnull().sum().sum())},
        {"Kontrol": "Bellek (MB)",         "Değer": round(df.memory_usage(deep=True).sum() / 1e6, 2)},
        {"Kontrol": "Dataset MD5 hash",    "Değer": veri_hash},
    ])
    csv_kaydet(kalite_df, "02_veri_kalite_ozeti.csv", index=False)

    # Önizleme
    csv_kaydet(df.head(5),   "03a_ilk5_satir.csv",    index=False)
    csv_kaydet(df.sample(5, random_state=RANDOM_STATE), "03b_rastgele5_satir.csv", index=False)

    adim(f"Eksik hücre: {int(df.isnull().sum().sum()):,}  |  Tekrar satır: {duplikat:,}")
    tamam("Kalite kontrolleri tamamlandı")


    # =============================================================================
    # 5. HEDEF DEĞİŞKEN ANALİZİ
    # =============================================================================

    bolum("BÖLÜM 3 — Hedef Değişken Analizi")

    # failure_flag
    ff_ratio = imbalance_ratio(df[HEDEF_FAILURE])
    ff_dist  = sinif_dagilim(df[HEDEF_FAILURE], HEDEF_FAILURE)
    csv_kaydet(ff_dist, "04_failure_flag_dagilimi.csv", index=False)
    log.info(f"failure_flag imbalance: {ff_ratio:.2f}x")

    # stress_level
    sl_ratio = imbalance_ratio(df[HEDEF_STRESS])
    sl_dist  = sinif_dagilim(df[HEDEF_STRESS], HEDEF_STRESS)
    csv_kaydet(sl_dist, "05_stress_level_dagilimi.csv", index=False)
    log.info(f"stress_level imbalance: {sl_ratio:.2f}x")

    # suitability_score
    suit_desc = df[HEDEF_REG].describe().to_frame()
    suit_ek   = pd.DataFrame([
        {"Metrik": "Skewness",     "Değer": round(df[HEDEF_REG].skew(), 4)},
        {"Metrik": "Kurtosis",     "Değer": round(df[HEDEF_REG].kurt(), 4)},
        {"Metrik": "Sıfır_Sayısı", "Değer": int((df[HEDEF_REG] == 0).sum())},
    ])
    csv_kaydet(suit_desc, "06_suitability_score_describe.csv")
    csv_kaydet(suit_ek,   "07_suitability_score_ek.csv", index=False)

    # Hedef özeti
    hedef_ozet = pd.DataFrame([
        {
            "Hedef":          HEDEF_FAILURE,
            "Problem_Tipi":   "Binary Classification",
            "Imbalance_Oran": round(ff_ratio, 4),
            "SMOTE":          "Uygulanır (ratio > 2.0)",
            "Metrik_Önerileri": "ROC-AUC, F1, Precision, Recall"
        },
        {
            "Hedef":          HEDEF_REG,
            "Problem_Tipi":   "Regression",
            "Imbalance_Oran": "—",
            "SMOTE":          "Uygulanmaz",
            "Metrik_Önerileri": "RMSE, MAE, R2"
        },
        {
            "Hedef":          HEDEF_STRESS,
            "Problem_Tipi":   "Multiclass Classification (0/1/2)",
            "Imbalance_Oran": round(sl_ratio, 4),
            "SMOTE":          f"Uygulanmaz (ratio {sl_ratio:.2f} < eşik {IMBALANCE_ESIK})",
            "Metrik_Önerileri": "Macro-F1, Confusion Matrix"
        },
    ])
    csv_kaydet(hedef_ozet, "08_hedef_ozet.csv", index=False)

    adim(f"failure_flag: {ff_ratio:.2f}x imbalance → SMOTE uygulanacak")
    adim(f"stress_level: {sl_ratio:.2f}x imbalance → SMOTE uygulanmayacak (eşik altı)")
    tamam("Hedef analizi tamamlandı")


    # =============================================================================
    # 6. DEĞİŞKEN TİPLERİ VE KATEGORİK ANALİZ (EDA – Tam Veri)
    # =============================================================================

    bolum("BÖLÜM 4 — Değişken Tipleri ve Kategorik Analiz")

    # Ham bağımsız değişkenleri tespit et
    X_ham   = df.drop(columns=HEDEFLER)
    kat_ham = X_ham.select_dtypes(include=["object", "category"]).columns.tolist()
    say_ham = X_ham.select_dtypes(include=[np.number]).columns.tolist()

    degisken_tip = pd.DataFrame([
        {"Tip": "Sayısal bağımsız",   "Adet": len(say_ham), "Sütunlar": ", ".join(say_ham)},
        {"Tip": "Kategorik bağımsız", "Adet": len(kat_ham), "Sütunlar": ", ".join(kat_ham)},
        {"Tip": "Hedef",              "Adet": len(HEDEFLER), "Sütunlar": ", ".join(HEDEFLER)},
    ])
    csv_kaydet(degisken_tip, "09_degisken_tipleri.csv", index=False)

    # Kategorik sütun özetleri
    kat_ozet_liste = []
    for col in kat_ham:
        nuniq = df[col].nunique(dropna=True)
        oran  = nuniq / len(df)
        kat_ozet_liste.append({
            "Sütun":          col,
            "Benzersiz":      int(nuniq),
            "Benzersiz_Oran": round(float(oran), 6),
            "Null_Sayısı":    int(df[col].isnull().sum()),
            "Model_Dahil":    "HAYIR" if (nuniq > 50 and oran > 0.05) else "EVET"
        })
        vc  = df[col].value_counts(dropna=False).head(MAX_CAT_TOP_N)
        pct = df[col].value_counts(normalize=True, dropna=False).head(MAX_CAT_TOP_N) * 100
        t   = pd.DataFrame({"Kategori": vc.index.astype(str), "Sayı": vc.values, "Oran%": pct.values.round(4)})
        csv_kaydet(t, f"10_kategorik_{col}.csv", index=False)

    csv_kaydet(pd.DataFrame(kat_ozet_liste), "11_kategorik_ozet.csv", index=False)

    adim(f"Sayısal: {len(say_ham)}, Kategorik: {len(kat_ham)}")
    tamam("Değişken tipleri tamamlandı")


    # =============================================================================
    # 7. SAYISAL İSTATİSTİK ve SKEWNESS ANALİZİ (EDA – Tam Veri)
    # =============================================================================

    bolum("BÖLÜM 5 — Sayısal İstatistik, Skewness, IQR Aykırı Değer, Korelasyon")

    # Describe
    sayisal_desc = df[say_ham].describe().T.round(4)
    csv_kaydet(sayisal_desc, "12_sayisal_describe.csv")

    # Skewness — pipeline kararı burada verilir
    skewness_df = pd.DataFrame({
        "Sütun":    say_ham,
        "Skewness": [round(df[c].skew(), 4) for c in say_ham],
        "Kurtosis": [round(df[c].kurt(), 4) for c in say_ham],
        "Min":      [df[c].min() for c in say_ham],
        "Max":      [df[c].max() for c in say_ham],
        "Null":     [int(df[c].isnull().sum()) for c in say_ham],
        "Sıfır":    [int((df[c] == 0).sum()) for c in say_ham],
        "Negatif":  [int((df[c] < 0).sum()) for c in say_ham],
    }).round(4)
    skewness_df["Transform"] = skewness_df["Skewness"].abs().apply(
        lambda s: f"PowerTransformer (Yeo-Johnson) — |skew|={s:.2f} > {SKEW_ESIK}" if s > SKEW_ESIK else "StandardScaler"
    )
    csv_kaydet(skewness_df, "13_sayisal_skewness_ve_transform.csv", index=False)

    # IQR aykırı değer analizi (tüm veri, keşifsel)
    aykiri_liste = []
    for col in say_ham:
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR     = Q3 - Q1
        alt, ust = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        n = int(((df[col] < alt) | (df[col] > ust)).sum())
        aykiri_liste.append({
            "Sütun":       col,
            "Q1":          round(Q1, 4),
            "Q3":          round(Q3, 4),
            "IQR":         round(IQR, 4),
            "Alt_Sınır":   round(alt, 4),
            "Üst_Sınır":   round(ust, 4),
            "Aykırı_Sayı": n,
            "Aykırı%":     round(n / len(df) * 100, 4),
        })
    aykiri_df = pd.DataFrame(aykiri_liste).sort_values("Aykırı%", ascending=False)
    csv_kaydet(aykiri_df, "14_aykiri_deger_iqr.csv", index=False)

    # Korelasyon (keşifsel, tüm veri)
    corr_cols = say_ham + [h for h in HEDEFLER if pd.api.types.is_numeric_dtype(df[h])]
    kor        = pd.DataFrame()
    yuksek_df  = pd.DataFrame()

    if len(corr_cols) >= 2:
        kor = df[corr_cols].corr()
        csv_kaydet(kor, "15_korelasyon_matrisi_EDA.csv")

        # Yüksek korelasyonlu çiftler (bağımsız değişkenler arası)
        yuksek = []
        for i in range(len(say_ham)):
            for j in range(i + 1, len(say_ham)):
                r = kor.loc[say_ham[i], say_ham[j]]
                if abs(r) > 0.70:
                    yuksek.append({"Değişken_1": say_ham[i], "Değişken_2": say_ham[j], "r": round(r, 4)})
        yuksek_df = pd.DataFrame(yuksek)
        csv_kaydet(yuksek_df, "16_yuksek_korelasyonlu_ciftler.csv", index=False)

        # Hedeflerle korelasyon
        hedef_kor_list = []
        for h in HEDEFLER:
            if h in kor.columns:
                s = kor[h].drop(labels=[x for x in HEDEFLER if x in kor.index], errors="ignore")
                for deg, val in s.sort_values(key=lambda x: x.abs(), ascending=False).items():
                    hedef_kor_list.append({"Hedef": h, "Değişken": deg, "r": round(val, 4), "|r|": round(abs(val), 4)})
        csv_kaydet(pd.DataFrame(hedef_kor_list), "17_hedeflerle_korelasyon.csv", index=False)

    adim(f"Yüksek skew sütunlar (|skew|>{SKEW_ESIK}): {skewness_df[skewness_df['Skewness'].abs() > SKEW_ESIK]['Sütun'].tolist()}")
    adim(f"Yüksek korelasyonlu çift: {len(yuksek_df)}")
    tamam("Sayısal analiz tamamlandı")


    # =============================================================================
    # 8. MODEL GİRDİSİ TEMİZLİĞİ
    # =============================================================================
    # Kural: Benzersiz değer sayısı > 50 VE benzersiz oran > %5 olan kategorik sütunlar
    # ID benzeri kabul edilerek düşürülür. location_id bu kritere girer.
    # =============================================================================

    bolum("BÖLÜM 6 — Model Girdisi Temizliği")

    X = df.drop(columns=HEDEFLER).copy()

    dusurulen = []

    # Tamamen boş
    for col in X.columns:
        if X[col].isnull().all():
            dusurulen.append({"Sütun": col, "Sebep": "Tamamen boş", "Benzersiz": 0, "Benzersiz_Oran": 0.0})

    # Sabit
    for col in X.columns:
        if col in [d["Sütun"] for d in dusurulen]:
            continue
        if X[col].nunique(dropna=True) <= 1:
            dusurulen.append({"Sütun": col, "Sebep": "Sabit/tek değer",
                              "Benzersiz": int(X[col].nunique()), "Benzersiz_Oran": 0.0})

    # Yüksek kardinaliteli kategorik (ID benzeri)
    for col in X.select_dtypes(include=["object", "category"]).columns:
        if col in [d["Sütun"] for d in dusurulen]:
            continue
        nuniq = X[col].nunique(dropna=True)
        oran  = nuniq / len(X)
        if nuniq > 50 and oran > 0.05:
            dusurulen.append({"Sütun": col, "Sebep": f"ID benzeri: {nuniq} benzersiz değer, oran={oran:.4f}",
                              "Benzersiz": int(nuniq), "Benzersiz_Oran": round(oran, 6)})

    dusurulen_df = pd.DataFrame(dusurulen)
    csv_kaydet(dusurulen_df if not dusurulen_df.empty else
               pd.DataFrame(columns=["Sütun", "Sebep", "Benzersiz", "Benzersiz_Oran"]),
               "18_dusurulen_sutunlar.csv", index=False)

    if not dusurulen_df.empty:
        X_model = X.drop(columns=dusurulen_df["Sütun"].tolist()).copy()
        log.warning(f"Düşürülen sütunlar: {dusurulen_df['Sütun'].tolist()}")
    else:
        X_model = X.copy()

    say_g = X_model.select_dtypes(include=[np.number]).columns.tolist()
    kat_g = X_model.select_dtypes(include=["object", "category"]).columns.tolist()

    # Skewness'a göre sayısal alt gruplar
    say_skewed  = [c for c in say_g if abs(df[c].skew()) > SKEW_ESIK]
    say_normal  = [c for c in say_g if abs(df[c].skew()) <= SKEW_ESIK]

    girdi_ozet = pd.DataFrame([
        {"Metrik": "Ham bağımsız sütun",           "Değer": X.shape[1]},
        {"Metrik": "Düşürülen sütun",              "Değer": len(dusurulen_df)},
        {"Metrik": "Model sütun (toplam)",          "Değer": X_model.shape[1]},
        {"Metrik": "  Sayısal (PowerTransformer)",  "Değer": len(say_skewed)},
        {"Metrik": "  Sayısal (StandardScaler)",    "Değer": len(say_normal)},
        {"Metrik": "  Kategorik (OHE)",             "Değer": len(kat_g)},
        {"Metrik": "PowerTransformer sütunları",    "Değer": ", ".join(say_skewed)},
        {"Metrik": "StandardScaler sütunları",      "Değer": ", ".join(say_normal)},
        {"Metrik": "OHE kategorik sütunları",       "Değer": ", ".join(kat_g)},
    ])
    csv_kaydet(girdi_ozet, "19_model_girdi_ozeti.csv", index=False)

    adim(f"Düşürülen: {[d['Sütun'] for d in dusurulen]}")
    adim(f"PowerTransformer: {say_skewed}")
    adim(f"StandardScaler  : {say_normal}")
    adim(f"OHE             : {kat_g}")
    tamam("Model girdisi hazırlandı")


    # =============================================================================
    # 9. TRAIN / TEST AYRIMI
    # =============================================================================

    bolum("BÖLÜM 7 — Train / Test Ayrımı")

    y_failure = df[HEDEF_FAILURE]
    y_reg     = df[HEDEF_REG]
    y_stress  = df[HEDEF_STRESS]

    # Her kombinasyonda en az 2 örnek varsa combined stratify; yoksa fallback
    combined_strata  = y_failure.astype(str) + "__" + y_stress.astype(str)
    combined_min     = combined_strata.value_counts().min()

    if combined_min >= 2:
        stratify_col    = combined_strata
        stratify_yontem = f"{HEDEF_FAILURE} + {HEDEF_STRESS} combined stratify"
    else:
        stratify_col    = y_failure
        stratify_yontem = f"Fallback: yalnızca {HEDEF_FAILURE} stratify"
        log.warning(f"Combined stratify başarısız (min={combined_min}). Fallback uygulandı.")

    train_idx, test_idx = train_test_split(
        df.index,
        test_size=TEST_ORANI,
        random_state=RANDOM_STATE,
        stratify=stratify_col
    )

    X_train_raw = X_model.loc[train_idx].reset_index(drop=True)
    X_test_raw  = X_model.loc[test_idx].reset_index(drop=True)

    y_failure_train = y_failure.loc[train_idx].reset_index(drop=True)
    y_failure_test  = y_failure.loc[test_idx].reset_index(drop=True)
    y_reg_train     = y_reg.loc[train_idx].reset_index(drop=True)
    y_reg_test      = y_reg.loc[test_idx].reset_index(drop=True)
    y_stress_train  = y_stress.loc[train_idx].reset_index(drop=True)
    y_stress_test   = y_stress.loc[test_idx].reset_index(drop=True)

    overlap = len(set(train_idx).intersection(set(test_idx)))
    if overlap != 0:
        log.error(f"HATA: Train/test overlap = {overlap}")

    split_kontrol = pd.DataFrame([
        {"Kontrol": "Stratify yöntemi",  "Değer": stratify_yontem},
        {"Kontrol": "Train satır",       "Değer": len(train_idx)},
        {"Kontrol": "Test satır",        "Değer": len(test_idx)},
        {"Kontrol": "Test oranı",        "Değer": TEST_ORANI},
        {"Kontrol": "Train/Test overlap","Değer": overlap},
        {"Kontrol": "Random state",      "Değer": RANDOM_STATE},
    ])
    csv_kaydet(split_kontrol, "20_split_kontrol.csv", index=False)

    # Dağılım doğrulamaları
    split_dist = pd.concat([
        sinif_dagilim(y_failure,       "tam_veri_failure"),
        sinif_dagilim(y_failure_train, "train_failure"),
        sinif_dagilim(y_failure_test,  "test_failure"),
        sinif_dagilim(y_stress,        "tam_veri_stress"),
        sinif_dagilim(y_stress_train,  "train_stress"),
        sinif_dagilim(y_stress_test,   "test_stress"),
    ], ignore_index=True)
    csv_kaydet(split_dist, "21_split_dagilim_dogrulama.csv", index=False)

    # Ham train/test CSV (model temizliği sonrası, preprocessing öncesi)
    train_ham = X_train_raw.copy()
    train_ham[HEDEF_FAILURE] = y_failure_train.values
    train_ham[HEDEF_REG]     = y_reg_train.values
    train_ham[HEDEF_STRESS]  = y_stress_train.values
    train_ham.to_csv(DATA / "TRAIN_raw_with_targets.csv", index=False, encoding="utf-8-sig")

    test_ham = X_test_raw.copy()
    test_ham[HEDEF_FAILURE] = y_failure_test.values
    test_ham[HEDEF_REG]     = y_reg_test.values
    test_ham[HEDEF_STRESS]  = y_stress_test.values
    test_ham.to_csv(DATA / "TEST_raw_with_targets.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame({"train_idx": train_idx}).to_csv(DATA / "train_indexleri.csv", index=False)
    pd.DataFrame({"test_idx":  test_idx}).to_csv(DATA / "test_indexleri.csv",  index=False)

    adim(f"Train: {len(train_idx):,}  |  Test: {len(test_idx):,}  |  Overlap: {overlap}")
    adim(f"Stratify: {stratify_yontem}")
    tamam("Train/test ayrımı tamamlandı")


    # =============================================================================
    # 10. VIF ANALİZİ — SADECE TRAIN SETİ ÜZERİNDE
    # =============================================================================
    # Akademik not: VIF, train seti bilinmeden önce hesaplanamaz.
    # Bu nedenle split'ten SONRA ve yalnızca X_train_raw üzerinde hesaplanır.
    # =============================================================================

    bolum("BÖLÜM 8 — VIF (Train Seti)")

    vif_df   = pd.DataFrame(columns=["R2", "VIF", "Not"])
    vif_y10  = pd.DataFrame()
    vif_o5   = pd.DataFrame()

    if 2 <= len(say_ham) <= MAX_VIF_COLS:
        adim("VIF hesaplanıyor (train seti, sayısal bağımsız değişkenler)...")
        t0 = time.time()
        # Sadece sayısal sütunları train'den al
        vif_df   = vif_hesapla(X_train_raw[say_g])
        vif_sure = time.time() - t0
        csv_kaydet(vif_df, "22_vif_train_seti.csv")

        if "VIF" in vif_df.columns:
            vif_y10 = vif_df[vif_df["VIF"] >= 10]
            vif_o5  = vif_df[(vif_df["VIF"] >= 5) & (vif_df["VIF"] < 10)]

        adim(f"VIF süresi: {vif_sure:.1f}s  |  VIF≥10: {len(vif_y10)}  |  5≤VIF<10: {len(vif_o5)}")
    else:
        neden = "Yeterli sayısal değişken yok" if len(say_ham) < 2 else f"Sütun sayısı {len(say_ham)} > limit {MAX_VIF_COLS}"
        csv_kaydet(pd.DataFrame([{"Durum": "Atlandı", "Sebep": neden}]), "22_vif_atlandi.csv", index=False)
        adim(f"VIF atlandı: {neden}")

    tamam("VIF analizi tamamlandı")


    # =============================================================================
    # 11. PREPROCESSING PIPELINE
    # =============================================================================

    bolum("BÖLÜM 9 — Preprocessing Pipeline (Fit: sadece train)")

    # Pipeline tanımları
    skewed_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("power",   PowerTransformer(method="yeo-johnson")),
    ])

    normal_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    kat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])

    transformers = []
    if say_skewed:
        transformers.append(("skewed", skewed_pipe, say_skewed))
    if say_normal:
        transformers.append(("normal", normal_pipe, say_normal))
    if kat_g:
        transformers.append(("kat",    kat_pipe,    kat_g))

    if not transformers:
        raise ValueError("Kullanılabilir bağımsız değişken kalmadı.")

    preprocessor = ColumnTransformer(transformers=transformers, sparse_threshold=0.3)

    # FIT yalnızca train üzerinde
    t0 = time.time()
    preprocessor.fit(X_train_raw)
    fit_sure = time.time() - t0
    adim(f"Preprocessor fit: {fit_sure:.2f}s")

    t0 = time.time()
    X_train_proc = preprocessor.transform(X_train_raw)
    X_test_proc  = preprocessor.transform(X_test_raw)
    trn_sure     = time.time() - t0
    adim(f"Transform süresi: {trn_sure:.2f}s")

    # Feature isimleri
    feature_names = []
    if say_skewed:
        feature_names.extend(say_skewed)
    if say_normal:
        feature_names.extend(say_normal)
    if kat_g:
        try:
            ohe_cols = preprocessor.named_transformers_["kat"]["encoder"].get_feature_names_out(kat_g).tolist()
            feature_names.extend(ohe_cols)
        except Exception as e:
            log.warning(f"OHE feature isimleri alınamadı: {e}")

    feature_df = pd.DataFrame({"idx": range(len(feature_names)), "feature": feature_names})
    csv_kaydet(feature_df, "23_processed_feature_names.csv", index=False)

    # Matrix özetleri
    def mat_ozet(Xm, ad):
        rows = [{"Matris": ad, "Metrik": "Satır",  "Değer": Xm.shape[0]},
                {"Matris": ad, "Metrik": "Sütun",  "Değer": Xm.shape[1]},
                {"Matris": ad, "Metrik": "Format", "Değer": "sparse" if sparse.issparse(Xm) else "dense"}]
        if sparse.issparse(Xm):
            rows.append({"Matris": ad, "Metrik": "Yoğunluk",
                         "Değer": round(Xm.nnz / (Xm.shape[0] * Xm.shape[1]), 6)})
        return pd.DataFrame(rows)

    csv_kaydet(pd.concat([mat_ozet(X_train_proc, "X_train_proc"), mat_ozet(X_test_proc, "X_test_proc")]),
               "24_processed_matrix_ozet.csv", index=False)

    # İşlenmiş matrisleri kaydet
    X_train_proc_path = matris_kaydet(X_train_proc, "X_train_processed")
    X_test_proc_path  = matris_kaydet(X_test_proc,  "X_test_processed")

    # Hedef serilerini kaydet
    seri_kaydet(y_failure_train.rename("y_failure_train"),    "y_failure_train.csv")
    seri_kaydet(y_failure_test.rename("y_failure_test"),      "y_failure_test.csv")
    seri_kaydet(y_reg_train.rename("y_suitability_train"),    "y_suitability_train.csv")
    seri_kaydet(y_reg_test.rename("y_suitability_test"),      "y_suitability_test.csv")
    seri_kaydet(y_stress_train.rename("y_stress_train"),      "y_stress_train.csv")
    seri_kaydet(y_stress_test.rename("y_stress_test"),        "y_stress_test.csv")

    # Tüm hedefler tek tabloda
    pd.DataFrame({HEDEF_FAILURE: y_failure_train.values, HEDEF_REG: y_reg_train.values,
                  HEDEF_STRESS: y_stress_train.values}).to_csv(DATA / "Y_train_all_targets.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({HEDEF_FAILURE: y_failure_test.values,  HEDEF_REG: y_reg_test.values,
                  HEDEF_STRESS: y_stress_test.values}).to_csv(DATA / "Y_test_all_targets.csv", index=False, encoding="utf-8-sig")

    # Preprocessor kaydet
    preprocessor_path = MODEL_DIR / "preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path)

    adim(f"Train: {X_train_proc.shape}  |  Test: {X_test_proc.shape}")
    tamam("Preprocessing pipeline tamamlandı")


    # =============================================================================
    # 12. SINIF DENGESİZLİĞİ VE SMOTE
    # =============================================================================

    bolum("BÖLÜM 10 — Sınıf Dengesizliği ve SMOTE")

    # failure_flag: SMOTE (ratio ~5.17x)
    failure_train_ratio = imbalance_ratio(y_failure_train)
    adim(f"failure_flag train ratio: {failure_train_ratio:.2f}x → SMOTE uygulanıyor...")
    (X_bal_failure, y_bal_failure,
     failure_smote_ok, failure_smote_not) = smote_uygula(
        X_train_proc, y_failure_train, HEDEF_FAILURE, failure_train_ratio
    )

    # stress_level: SMOTE yok (ratio ~1.85x < 2.0)
    stress_train_ratio = imbalance_ratio(y_stress_train)
    adim(f"stress_level train ratio: {stress_train_ratio:.2f}x → eşik altı, SMOTE atlandı")
    X_bal_stress    = X_train_proc
    y_bal_stress    = y_stress_train.reset_index(drop=True)
    stress_smote_ok = False
    stress_smote_not = f"Ratio {stress_train_ratio:.2f} < eşik {IMBALANCE_ESIK}"

    # class_weight hesapla (model eğitiminde kullanmak için)
    cw_failure = compute_class_weight("balanced", classes=np.unique(y_failure_train), y=y_failure_train)
    cw_stress  = compute_class_weight("balanced", classes=np.unique(y_stress_train),  y=y_stress_train)

    cw_failure_dict = dict(zip(np.unique(y_failure_train).tolist(), cw_failure.round(4).tolist()))
    cw_stress_dict  = dict(zip(np.unique(y_stress_train).tolist(),  cw_stress.round(4).tolist()))

    class_weight_df = pd.DataFrame([
        {"Hedef": HEDEF_FAILURE, "Sınıf": k, "Ağırlık": v} for k, v in cw_failure_dict.items()
    ] + [
        {"Hedef": HEDEF_STRESS,  "Sınıf": k, "Ağırlık": v} for k, v in cw_stress_dict.items()
    ])
    csv_kaydet(class_weight_df, "25_class_weights.csv", index=False)
    log.info(f"class_weight failure: {cw_failure_dict}")
    log.info(f"class_weight stress : {cw_stress_dict}")

    # Dengeleme özeti
    dengeleme_ozet = pd.DataFrame([
        {
            "Hedef":          HEDEF_FAILURE,
            "Problem":        "Binary Classification",
            "Train_Ratio":    round(failure_train_ratio, 4),
            "SMOTE":          failure_smote_ok,
            "Not":            failure_smote_not,
            "Train_Önce":     X_train_proc.shape[0],
            "Train_Sonra":    X_bal_failure.shape[0],
            "class_weight":   str(cw_failure_dict),
        },
        {
            "Hedef":          HEDEF_REG,
            "Problem":        "Regression",
            "Train_Ratio":    "—",
            "SMOTE":          False,
            "Not":            "Regresyon hedefi; SMOTE uygulanmaz",
            "Train_Önce":     X_train_proc.shape[0],
            "Train_Sonra":    X_train_proc.shape[0],
            "class_weight":   "—",
        },
        {
            "Hedef":          HEDEF_STRESS,
            "Problem":        "Multiclass Classification",
            "Train_Ratio":    round(stress_train_ratio, 4),
            "SMOTE":          stress_smote_ok,
            "Not":            stress_smote_not,
            "Train_Önce":     X_train_proc.shape[0],
            "Train_Sonra":    X_bal_stress.shape[0],
            "class_weight":   str(cw_stress_dict),
        },
    ])
    csv_kaydet(dengeleme_ozet, "26_dengeleme_ozet.csv", index=False)

    # SMOTE çıktılarını kaydet
    X_bal_failure_path = matris_kaydet(X_bal_failure, "X_train_balanced_failure")
    seri_kaydet(pd.Series(y_bal_failure, name="y_balanced_failure"), "y_train_balanced_failure.csv")

    # stress için balanced = processed (SMOTE yok)
    X_bal_stress_path = matris_kaydet(X_bal_stress, "X_train_balanced_stress")
    seri_kaydet(pd.Series(y_bal_stress, name="y_balanced_stress"),  "y_train_balanced_stress.csv")

    # SMOTE öncesi/sonrası dağılımlar
    csv_kaydet(pd.concat([
        sinif_dagilim(y_failure_train,             "failure_oncesi"),
        sinif_dagilim(pd.Series(y_bal_failure),    "failure_sonrasi")
    ], ignore_index=True), "27_balanced_failure_dagilim.csv", index=False)

    csv_kaydet(pd.concat([
        sinif_dagilim(y_stress_train,              "stress_train"),
        sinif_dagilim(pd.Series(y_bal_stress),     "stress_balanced")
    ], ignore_index=True), "28_balanced_stress_dagilim.csv", index=False)

    adim(f"failure → {X_train_proc.shape[0]:,} ➜ {X_bal_failure.shape[0]:,} satır (SMOTE: {failure_smote_ok})")
    adim(f"stress  → {X_train_proc.shape[0]:,} satır (SMOTE: {stress_smote_ok})")
    adim(f"class_weight failure: {cw_failure_dict}")
    adim(f"class_weight stress : {cw_stress_dict}")
    tamam("Dengeleme tamamlandı")


    # =============================================================================
    # 13. GÖRSELLEŞTİRMELER
    # =============================================================================

    bolum("BÖLÜM 11 — Görselleştirmeler")

    # 11.1 Hedef dağılümleri
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Hedef Değişken Dağılımları", fontsize=14, fontweight="bold")

    df[HEDEF_FAILURE].value_counts().sort_index().plot(kind="bar", ax=axes[0], edgecolor="black", color=["steelblue","tomato"])
    axes[0].set_title(f"failure_flag (imbalance: {ff_ratio:.1f}x)")
    axes[0].tick_params(axis="x", rotation=0)
    for p in axes[0].patches:
        axes[0].annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width()/2, p.get_height()), ha="center", va="bottom", fontsize=9)

    df[HEDEF_REG].plot(kind="hist", bins=60, ax=axes[1], edgecolor="white", alpha=0.85, color="steelblue")
    axes[1].set_title("suitability_score (Regresyon)")
    axes[1].axvline(df[HEDEF_REG].mean(),   linestyle="--", color="red",    label=f"Ort={df[HEDEF_REG].mean():.3f}")
    axes[1].axvline(df[HEDEF_REG].median(), linestyle=":",  color="orange", label=f"Med={df[HEDEF_REG].median():.3f}")
    axes[1].legend(fontsize=9)

    df[HEDEF_STRESS].value_counts().sort_index().plot(kind="bar", ax=axes[2], edgecolor="black", color=["steelblue","orange","tomato"])
    axes[2].set_title(f"stress_level (imbalance: {sl_ratio:.2f}x)")
    axes[2].tick_params(axis="x", rotation=0)
    for p in axes[2].patches:
        axes[2].annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width()/2, p.get_height()), ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    png_kaydet(fig, "01_hedef_dagilimlari.png")
    tamam("01_hedef_dagilimlari.png")

    # 11.2 Sayısal histogramlar
    if say_ham:
        cizilecek = say_ham[:MAX_NUMERIC_PLOT]
        nc  = 4
        nr  = max(1, (len(cizilecek) + nc - 1) // nc)
        fig, axes = plt.subplots(nr, nc, figsize=(20, nr * 4))
        axes = np.array(axes).reshape(-1)
        for i, col in enumerate(cizilecek):
            axes[i].hist(df[col].dropna(), bins=50, edgecolor="white", alpha=0.85, color="steelblue")
            skv = df[col].skew()
            axes[i].set_title(f"{col}\nskew={skv:.2f}", fontsize=9, fontweight="bold")
            axes[i].axvline(df[col].mean(), linestyle="--", color="red",    linewidth=1.2)
            axes[i].axvline(df[col].median(), linestyle=":", color="orange", linewidth=1.2)
            # Kırmızı çerçeve: yüksek skew
            if abs(skv) > SKEW_ESIK:
                for spine in axes[i].spines.values():
                    spine.set_edgecolor("red")
                    spine.set_linewidth(2)
        for j in range(len(cizilecek), len(axes)):
            axes[j].set_visible(False)
        plt.suptitle(f"Sayısal Dağılımlar (kırmızı çerçeve: |skew|>{SKEW_ESIK} → PowerTransformer)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        png_kaydet(fig, "02_sayisal_dagilimlar.png")
        tamam("02_sayisal_dagilimlar.png")

    # 11.3 Korelasyon heatmap
    if not kor.empty and len(kor.columns) <= MAX_HEATMAP_COLS:
        fig, ax = plt.subplots(figsize=(16, 13))
        mask = np.triu(np.ones_like(kor, dtype=bool))
        sns.heatmap(kor, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                    square=True, ax=ax, linewidths=0.5, annot_kws={"size": 7}, cbar_kws={"shrink": 0.75})
        ax.set_title("Pearson Korelasyon Matrisi (EDA – Tam Veri)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        png_kaydet(fig, "03_korelasyon_heatmap.png")
        tamam("03_korelasyon_heatmap.png")

    # 11.4 VIF grafiği
    if not vif_df.empty and "VIF" in vif_df.columns and vif_df["VIF"].notna().any():
        vif_plot = vif_df.dropna(subset=["VIF"]).sort_values("VIF", ascending=True)
        fig, ax  = plt.subplots(figsize=(10, max(5, len(vif_plot) * 0.45)))
        colors   = ["tomato" if v >= 10 else "orange" if v >= 5 else "steelblue" for v in vif_plot["VIF"]]
        bars     = ax.barh(vif_plot.index.astype(str), vif_plot["VIF"], edgecolor="black", height=0.65, color=colors)
        ax.axvline(5,  linestyle="--", lw=1.5, color="orange", label="VIF=5 (orta risk)")
        ax.axvline(10, linestyle="--", lw=1.5, color="tomato", label="VIF=10 (yüksek risk)")
        for b, v in zip(bars, vif_plot["VIF"]):
            ax.text(b.get_width() + 0.2, b.get_y() + b.get_height()/2, f"{v:.1f}", va="center", fontsize=8)
        ax.set_xlabel("VIF Değeri")
        ax.set_title("VIF Analizi — Train Seti", fontweight="bold")
        ax.legend()
        plt.tight_layout()
        png_kaydet(fig, "04_vif_analizi.png")
        tamam("04_vif_analizi.png")

    # 11.5 Boxplot
    if say_ham:
        cizilecek = say_ham[:MAX_NUMERIC_PLOT]
        df_box    = df[cizilecek].sample(min(BOXPLOT_SAMPLE_N, len(df)), random_state=RANDOM_STATE)
        nc, nr    = 4, max(1, (len(cizilecek) + 3) // 4)
        fig, axes = plt.subplots(nr, nc, figsize=(20, nr * 4))
        axes      = np.array(axes).reshape(-1)
        for i, col in enumerate(cizilecek):
            axes[i].boxplot(df_box[col].dropna(), patch_artist=True,
                            medianprops=dict(linewidth=2, color="red"),
                            flierprops=dict(marker="o", markersize=2, alpha=0.3))
            axes[i].set_title(col, fontsize=9, fontweight="bold")
        for j in range(len(cizilecek), len(axes)):
            axes[j].set_visible(False)
        plt.suptitle("Boxplot — Aykırı Değer Analizi", fontsize=13, fontweight="bold")
        plt.tight_layout()
        png_kaydet(fig, "05_boxplot_aykiri.png")
        tamam("05_boxplot_aykiri.png")

    # 11.6 Hedef ilişkileri
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    df.boxplot(column=HEDEF_REG, by=HEDEF_FAILURE, ax=axes[0], grid=False)
    axes[0].set_title(f"{HEDEF_REG} × {HEDEF_FAILURE}")
    axes[0].set_xlabel(HEDEF_FAILURE)
    axes[0].set_ylabel(HEDEF_REG)
    df.boxplot(column=HEDEF_REG, by=HEDEF_STRESS, ax=axes[1], grid=False)
    axes[1].set_title(f"{HEDEF_REG} × {HEDEF_STRESS}")
    axes[1].set_xlabel(HEDEF_STRESS)
    axes[1].set_ylabel(HEDEF_REG)
    fig.suptitle("")
    plt.tight_layout()
    png_kaydet(fig, "06_hedef_iliskileri.png")
    tamam("06_hedef_iliskileri.png")

    # 11.7 SMOTE karşılaştırması (sadece failure_flag için)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    y_failure_train.value_counts().sort_index().plot(kind="bar", ax=axes[0], edgecolor="black", color=["steelblue","tomato"])
    axes[0].set_title(f"failure_flag — SMOTE Öncesi\n(imbalance: {failure_train_ratio:.1f}x)")
    axes[0].tick_params(axis="x", rotation=0)
    pd.Series(y_bal_failure).value_counts().sort_index().plot(kind="bar", ax=axes[1], edgecolor="black", color=["steelblue","tomato"])
    axes[1].set_title("failure_flag — SMOTE Sonrası\n(dengelenmiş)")
    axes[1].tick_params(axis="x", rotation=0)
    plt.suptitle("failure_flag Dengeleme Karşılaştırması", fontsize=13, fontweight="bold")
    plt.tight_layout()
    png_kaydet(fig, "07_smote_failure_karsilastirma.png")
    tamam("07_smote_failure_karsilastirma.png")

    # 11.8 Skewness karşılaştırma çubuğu
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ["tomato" if abs(s) > SKEW_ESIK else "steelblue" for s in skewness_df["Skewness"]]
    ax.bar(skewness_df["Sütun"], skewness_df["Skewness"], color=colors, edgecolor="black")
    ax.axhline( SKEW_ESIK, linestyle="--", color="red",   linewidth=1.5, label=f"+{SKEW_ESIK} eşik")
    ax.axhline(-SKEW_ESIK, linestyle="--", color="red",   linewidth=1.5, label=f"-{SKEW_ESIK} eşik")
    ax.axhline(0,          linestyle="-",  color="black", linewidth=0.8)
    ax.set_xticklabels(skewness_df["Sütun"], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Skewness")
    ax.set_title(f"Sayısal Değişken Skewness (kırmızı: |skew|>{SKEW_ESIK} → PowerTransformer)", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    png_kaydet(fig, "08_skewness_analizi.png")
    tamam("08_skewness_analizi.png")

    tamam("Tüm görseller tamamlandı")


    # =============================================================================
    # 14. METADATA VE RAPORLAR
    # =============================================================================

    bolum("BÖLÜM 12 — Metadata, Raporlar ve Kullanım Rehberi")

    n_gorsel = len(list(GRAFIK.glob("*.png")))
    n_csv    = len(list(ANALIZ.glob("*.csv")))

    # Görev → dosya haritası (Aşama 2 için makine okunabilir rehber)
    task_map = pd.DataFrame([
        {
            "task":         "failure_flag",
            "problem":      "binary_classification",
            "train_X":      "X_train_balanced_failure.npz/.npy",
            "train_y":      "y_train_balanced_failure.csv",
            "test_X":       "X_test_processed.npz/.npy",
            "test_y":       "y_failure_test.csv",
            "class_weight": str(cw_failure_dict),
            "not":          "SMOTE train'e uygulandı. Test orijinal dağılımda."
        },
        {
            "task":         "suitability_score",
            "problem":      "regression",
            "train_X":      "X_train_processed.npz/.npy",
            "train_y":      "y_suitability_train.csv",
            "test_X":       "X_test_processed.npz/.npy",
            "test_y":       "y_suitability_test.csv",
            "class_weight": "—",
            "not":          "SMOTE/oversampling uygulanmaz."
        },
        {
            "task":         "stress_level",
            "problem":      "multiclass_classification",
            "train_X":      "X_train_balanced_stress.npz/.npy",
            "train_y":      "y_train_balanced_stress.csv",
            "test_X":       "X_test_processed.npz/.npy",
            "test_y":       "y_stress_test.csv",
            "class_weight": str(cw_stress_dict),
            "not":          f"SMOTE uygulanmadı (ratio={stress_train_ratio:.2f} < {IMBALANCE_ESIK}). class_weight kullanılabilir."
        },
    ])
    task_map.to_csv(DATA / "TASK_TO_DATASET_MAP.csv", index=False, encoding="utf-8-sig")

    # Metadata JSON
    metadata = {
        "run_id":          RUN_ID,
        "dataset_hash":    veri_hash,
        "dataset_path":    str(VERI),
        "output_dir":      str(CIKTI),
        "random_state":    RANDOM_STATE,
        "test_orani":      TEST_ORANI,
        "skew_esik":       SKEW_ESIK,
        "imbalance_esik":  IMBALANCE_ESIK,
        "rows":            int(df.shape[0]),
        "cols":            int(df.shape[1]),
        "targets": {
            "failure_flag":    "binary_classification",
            "suitability_score": "regression",
            "stress_level":    "multiclass_classification"
        },
        "stratify_yontem": stratify_yontem,
        "train_rows":      int(len(train_idx)),
        "test_rows":       int(len(test_idx)),
        "overlap":         int(overlap),
        "model_columns":   int(X_model.shape[1]),
        "say_skewed":      say_skewed,
        "say_normal":      say_normal,
        "kat_g":           kat_g,
        "dusurulen":       [d["Sütun"] for d in dusurulen],
        "processed_features": int(X_train_proc.shape[1]),
        "class_weights": {
            "failure_flag": cw_failure_dict,
            "stress_level": cw_stress_dict,
        },
        "smote": {
            "failure_flag":    {"applied": bool(failure_smote_ok), "note": failure_smote_not,
                                "rows_before": int(X_train_proc.shape[0]), "rows_after": int(X_bal_failure.shape[0])},
            "suitability_score": {"applied": False, "note": "Regresyon hedefi"},
            "stress_level":    {"applied": bool(stress_smote_ok), "note": stress_smote_not,
                                "rows_before": int(X_train_proc.shape[0]), "rows_after": int(X_bal_stress.shape[0])},
        },
        "files": {
            "preprocessor":        str(preprocessor_path),
            "X_train_processed":   str(X_train_proc_path),
            "X_test_processed":    str(X_test_proc_path),
            "X_balanced_failure":  str(X_bal_failure_path),
            "X_balanced_stress":   str(X_bal_stress_path),
            "log":                 str(LOG_DOSYASI),
        }
    }
    metadata_path = MODEL_DIR / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # Joblib paketi
    ciktilar = {
        "metadata":     metadata,
        "preprocessor": preprocessor,
        "train_idx":    train_idx,
        "test_idx":     test_idx,
        "say_skewed":   say_skewed,
        "say_normal":   say_normal,
        "kat_g":        kat_g,
        "feature_names": feature_names,
        "dusurulen_df": dusurulen_df,
        "vif_df":       vif_df,
        "kor_df":       kor,
        "smote_ozet":   dengeleme_ozet,
        "class_weights": {"failure_flag": cw_failure_dict, "stress_level": cw_stress_dict},
    }
    pkl_path = MODEL_DIR / "asama1_paket.joblib"
    joblib.dump(ciktilar, pkl_path)

    # Aşama 2 örnek yükleme scripti
    loader_code = '''from pathlib import Path
import pandas as pd
import joblib
import numpy as np
from scipy import sparse

BASE = Path(__file__).resolve().parent
DATA      = BASE / "03_model_verileri"
MODEL_DIR = BASE / "04_preprocessor"

def load_matrix(stem):
    npz = DATA / f"{stem}.npz"
    npy = DATA / f"{stem}.npy"
    if npz.exists(): return sparse.load_npz(npz)
    if npy.exists(): return np.load(npy)
    raise FileNotFoundError(f"{stem} bulunamadı")

def load_target(fname):
    return pd.read_csv(DATA / fname).iloc[:, 0]

# failure_flag — binary classification
X_tr_failure = load_matrix("X_train_balanced_failure")
y_tr_failure = load_target("y_train_balanced_failure.csv")
X_test       = load_matrix("X_test_processed")
y_te_failure = load_target("y_failure_test.csv")

# suitability_score — regression
X_tr_reg  = load_matrix("X_train_processed")
y_tr_reg  = load_target("y_suitability_train.csv")
y_te_reg  = load_target("y_suitability_test.csv")

# stress_level — multiclass
X_tr_stress = load_matrix("X_train_balanced_stress")
y_tr_stress = load_target("y_train_balanced_stress.csv")
y_te_stress = load_target("y_stress_test.csv")

preprocessor = joblib.load(MODEL_DIR / "preprocessor.joblib")
meta         = joblib.load(MODEL_DIR / "asama1_paket.joblib")
class_weights = meta["class_weights"]

print("Yükleme başarılı.")
print(f"failure train  : {X_tr_failure.shape}  y: {y_tr_failure.shape}")
print(f"regression train: {X_tr_reg.shape}  y: {y_tr_reg.shape}")
print(f"stress train   : {X_tr_stress.shape}  y: {y_tr_stress.shape}")
print(f"test (ortak)   : {X_test.shape}")
print(f"class_weights  : {class_weights}")
'''
    (CIKTI / "load_phase2_data.py").write_text(loader_code, encoding="utf-8")

    # Akademik rapor
    rapor_metni = f"""# AŞAMA 1 AKADEMİK RAPORU

## Öğrenci
İbrahim Nuryağınlı | 25490221001

## Veri Seti
- Dosya    : dataset.csv
- MD5 Hash : {veri_hash}
- Satır    : {df.shape[0]:,}
- Sütun    : {df.shape[1]:,}
- Eksik    : {int(df.isnull().sum().sum()):,}
- Duplikat : {duplikat:,}

## Hedef Değişkenler
| Hedef | Problem | Imbalance | SMOTE |
|-------|---------|-----------|-------|
| failure_flag | Binary | {ff_ratio:.2f}x | ✓ Uygulandı |
| suitability_score | Regression | — | ✗ Uygulanmaz |
| stress_level | Multiclass | {sl_ratio:.2f}x | ✗ Eşik altı |

## Model Girdisi Temizliği
- Düşürülen sütunlar: {[d['Sütun'] for d in dusurulen]}
  - Sebep: ID benzeri (her satır benzersiz, modele katkısı yok)
- Model bağımsız sütun sayısı: {X_model.shape[1]}

## Preprocessing Pipeline
| Grup | Sütunlar | Dönüşüm |
|------|----------|---------|
| Yüksek skew | {', '.join(say_skewed)} | PowerTransformer (Yeo-Johnson) |
| Normal sayısal | {', '.join(say_normal)} | StandardScaler |
| Kategorik | {', '.join(kat_g)} | OneHotEncoder (sparse) |

- Preprocessor fit: YALNIZCA train seti üzerinde yapılmıştır.
- Test seti bu nesneyle transform edilmiştir.

## Train/Test Ayrımı
- Stratify : {stratify_yontem}
- Train    : {len(train_idx):,} satır (%{(1-TEST_ORANI)*100:.0f})
- Test     : {len(test_idx):,} satır (%{TEST_ORANI*100:.0f})
- Overlap  : {overlap}

## VIF Analizi (Train Seti)
- VIF≥10 (yüksek çoklu doğrusallık): {len(vif_y10)} değişken
- 5≤VIF<10 (orta): {len(vif_o5)} değişken
- Not: VIF yalnızca train seti üzerinde hesaplanmıştır.

## Sınıf Dengeleme
- failure_flag SMOTE: {failure_smote_ok} | {X_train_proc.shape[0]:,} → {X_bal_failure.shape[0]:,} satır
- stress_level SMOTE: {stress_smote_ok} | {stress_smote_not}
- class_weight (failure): {cw_failure_dict}
- class_weight (stress) : {cw_stress_dict}

## Çıktılar
- Analiz CSV : {n_csv}
- Görseller  : {n_gorsel}
- Preprocessor: 04_preprocessor/preprocessor.joblib
- Metadata    : 04_preprocessor/metadata.json
- Aşama 2 yükleyici: load_phase2_data.py

## Veri Sızıntısı Önlemleri
1. Train/test ayrımı preprocessing öncesi yapıldı.
2. Preprocessor yalnızca train üzerinde fit edildi.
3. VIF yalnızca train seti üzerinde hesaplandı.
4. SMOTE yalnızca train setine uygulandı; test orijinal bırakıldı.
5. EDA (korelasyon, IQR): tüm veri üzerinde, keşifsel amaçlı.
   Model kararları bu analizlere dayandırılmadı.
"""
    (RAPOR / "akademik_rapor.md").write_text(rapor_metni, encoding="utf-8")

    # Dosya manifesti
    manifest_rows = []
    for kat, dizin in [("analiz", ANALIZ), ("gorsel", GRAFIK), ("model_verisi", DATA),
                       ("preprocessor", MODEL_DIR), ("rapor", RAPOR)]:
        for fp in sorted(dizin.glob("*")):
            if fp.is_file():
                manifest_rows.append({"kategori": kat, "dosya": fp.name,
                                       "boyut_KB": round(fp.stat().st_size / 1024, 2), "path": str(fp)})
    pd.DataFrame(manifest_rows).to_csv(CIKTI / "FILE_MANIFEST.csv", index=False, encoding="utf-8-sig")

    tamam("Metadata ve raporlar tamamlandı")


    # =============================================================================
    # 15. FİNAL ÖZET
    # =============================================================================

    print(f"""
    ╔══════════════════════════════════════════════════════════════════╗
    ║              AŞAMA 1 TAMAMLANDI                                  ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  Veri    : {df.shape[0]:,} satır × {df.shape[1]} sütun
    ║  Hash    : {veri_hash}
    ║  Train   : {len(train_idx):,}  |  Test: {len(test_idx):,}  |  Overlap: {overlap}
    ║  Feature : {X_train_proc.shape[1]:,} (işlenmiş)
    ║  SMOTE   : failure ✓ ({X_bal_failure.shape[0]:,} satır) | stress ✗ (eşik altı)
    ║  Görseller: {n_gorsel}  |  CSV: {n_csv}
    ║  Çıktı   : {CIKTI}
    ╚══════════════════════════════════════════════════════════════════╝
    """, flush=True)

if __name__ == "__main__":
    main()