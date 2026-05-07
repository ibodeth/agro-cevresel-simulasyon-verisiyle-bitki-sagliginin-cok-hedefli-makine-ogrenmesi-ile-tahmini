# =============================================================================
# YZO 106 – İleri Düzey Makine Öğrenmesi | Dönem Projesi
# AŞAMA 5 – Konsolidasyon: Üç Görev Karşılaştırması, SHAP ve Duyarlılık Özeti
# Öğrenci: İbrahim Nuryağınlı | 25490221001
#
# Bu script asama2, asama3 ve asama4 çıktılarını yükler;
# rapor için hazır unified tablolar ve görseller üretir.
#
# Üretilen çıktılar:
#   01_tablolar/
#       unified_model_karsilastirma.csv      ← 3 görev × tüm modeller
#       en_iyi_modeller_ozet.csv             ← tek satırda en iyi modeller
#       shap_cross_task_onem.csv             ← özellik önemi (3 görev yan yana)
#       duyarlilik_cross_task.csv            ← duyarlılık oranları (3 görev)
#   02_gorseller/
#       01_en_iyi_modeller_radar.png         ← radar chart (3 görev en iyileri)
#       02_unified_performance_heatmap.png   ← tüm modeller × tüm metrikler
#       03_shap_cross_task_bar.png           ← top-10 özellik 3 görevde karşılaştırma
#       04_duyarlilik_cross_task.png         ← %5/%10/%20 gürültü etkisi (3 görev)
#       05_best_models_metrik_ozet.png       ← en iyi modellerin metrik özet barplot
#   03_raporlar/
#       asama5_akademik_ozet.md              ← rapor "Deneysel Sonuçlar" bölümü için
#
# Bağımlılıklar: pandas, numpy, matplotlib, seaborn, joblib
# (Keras/TF veya SHAP kurulu olmasına gerek yok.)
# =============================================================================

import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

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
pd.set_option("display.float_format", "{:.4f}".format)

FIG_DPI = 150


# =============================================================================
# 0. KLASÖR KURULUMU
# =============================================================================

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

RUN_ID  = datetime.now().strftime("%Y%m%d_%H%M%S")
CIKTI   = BASE_DIR / f"asama5_ciktilar_{RUN_ID}"
TABLO   = CIKTI / "01_tablolar"
GRAFIK  = CIKTI / "02_gorseller"
RAPOR   = CIKTI / "03_raporlar"

for d in [CIKTI, TABLO, GRAFIK, RAPOR]:
    d.mkdir(parents=True, exist_ok=True)

LOG_DOSYASI = RAPOR / f"asama5_log_{RUN_ID}.txt"
logger = logging.getLogger("asama5")
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
# 1. YARDIMCI FONKSİYONLAR
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
               klasor: Path = TABLO, index: bool = False) -> Path:
    p = klasor / dosya_adi
    df_obj.to_csv(p, index=index, encoding="utf-8-sig")
    log.info(f"CSV: {p.name}")
    return p


def csv_yukle(dosya_yolu: Path) -> pd.DataFrame:
    """UTF-8-BOM ile kaydedilmiş CSV dosyasını güvenli okur.

    Tüm asama2/3/4 çıktıları encoding='utf-8-sig' ile yazılmıştır.
    Windows'ta varsayılan sistem kodlaması (cp1252 vb.) BOM karakterini
    yanlış yorumlayabileceğinden encoding açıkça belirtilir.
    Herhangi bir okuma hatası durumunda programın çökmesi yerine
    boş DataFrame döner ve uyarı mesajı yazdırılır.
    """
    try:
        return pd.read_csv(dosya_yolu, encoding="utf-8-sig")
    except Exception as exc:
        log.warning(f"CSV okuma hatası ({dosya_yolu.name}): {exc}")
        adim(f"⚠ {dosya_yolu.name} okunamadı: {type(exc).__name__}: {exc}")
        return pd.DataFrame()


def png_kaydet(fig, dosya_adi: str, klasor: Path = GRAFIK) -> Path:
    p = klasor / dosya_adi
    fig.savefig(p, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info(f"PNG: {p.name}")
    return p


# =============================================================================
# 2. ASAMA ÇIKTILARINI YÜKLEYİCİ
# =============================================================================

bolum("BÖLÜM 1 — Aşama Çıktılarını Yükle")

def en_yeni_klasor(pattern: str) -> Path | None:
    klasorler = sorted(BASE_DIR.glob(pattern), reverse=True)
    return klasorler[0] if klasorler else None


A2_DIR = en_yeni_klasor("asama2_ciktilar_*")
A3_DIR = en_yeni_klasor("asama3_ciktilar_*")
A4_DIR = en_yeni_klasor("asama4_ciktilar_*")

for tag, d in [("Aşama 2", A2_DIR), ("Aşama 3", A3_DIR), ("Aşama 4", A4_DIR)]:
    if d:
        adim(f"{tag}: {d.name}")
    else:
        adim(f"⚠ {tag} klasörü bulunamadı!")

# ── Görev 1 (failure_flag — Binary Classification) ────────────────────────
gorev1_df, g1_shap, g1_duyar = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
g1_best_kisa, g1_best_meta = "?", {}

if A2_DIR:
    sonuc_path = A2_DIR / "01_sonuclar" / "01_tum_model_sonuclari.csv"
    if sonuc_path.exists():
        gorev1_df = csv_yukle(sonuc_path)
        adim(f"Görev 1 sonuç: {len(gorev1_df)} model")

    shap_path = A2_DIR / "05_shap" / "shap_feature_importance.csv"
    if shap_path.exists():
        g1_shap = csv_yukle(shap_path)

    duyar_path = A2_DIR / "06_duyarlilik" / "duyarlilik_ozet.csv"
    if duyar_path.exists():
        g1_duyar = csv_yukle(duyar_path)

    meta_path = A2_DIR / "asama2_gorev1_metadata.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            g1_best_meta = json.load(f)
        g1_best_kisa = g1_best_meta.get("best_model_kisa", "?")

# ── Görev 2 (suitability_score — Regression) ──────────────────────────────
gorev2_df, g2_shap, g2_duyar = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
g2_best_kisa, g2_best_meta = "?", {}

if A3_DIR:
    sonuc_path = A3_DIR / "01_sonuclar" / "01_tum_model_sonuclari.csv"
    if sonuc_path.exists():
        gorev2_df = csv_yukle(sonuc_path)
        adim(f"Görev 2 sonuç: {len(gorev2_df)} model")

    shap_path = A3_DIR / "05_shap" / "shap_feature_importance.csv"
    if shap_path.exists():
        g2_shap = csv_yukle(shap_path)

    duyar_path = A3_DIR / "06_duyarlilik" / "duyarlilik_ozet.csv"
    if duyar_path.exists():
        g2_duyar = csv_yukle(duyar_path)

    meta_path = A3_DIR / "asama3_gorev2_metadata.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            g2_best_meta = json.load(f)
        g2_best_kisa = g2_best_meta.get("best_model_kisa", "?")

# ── Görev 3 (stress_level — Multiclass Classification) ────────────────────
gorev3_df, g3_shap, g3_duyar = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
g3_best_kisa, g3_best_meta = "?", {}

if A4_DIR:
    sonuc_path = A4_DIR / "01_sonuclar" / "01_tum_model_sonuclari.csv"
    if sonuc_path.exists():
        gorev3_df = csv_yukle(sonuc_path)
        adim(f"Görev 3 sonuç: {len(gorev3_df)} model")

    shap_path = A4_DIR / "05_shap" / "shap_feature_importance_toplam.csv"
    if shap_path.exists():
        g3_shap = csv_yukle(shap_path)

    duyar_path = A4_DIR / "06_duyarlilik" / "duyarlilik_ozet.csv"
    if duyar_path.exists():
        g3_duyar = csv_yukle(duyar_path)

    meta_path = A4_DIR / "asama4_gorev3_metadata.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            g3_best_meta = json.load(f)
        g3_best_kisa = g3_best_meta.get("best_model_kisa", "?")

tamam("Yükleme tamamlandı")


# =============================================================================
# 3. UNIFIED MODEL KARŞILAŞTIRMA TABLOSU
# =============================================================================

bolum("BÖLÜM 2 — Unified Model Karşılaştırma Tablosu")

birlesik_satirlar = []

# Görev 1 — binary classification metrikleri
if not gorev1_df.empty:
    for _, row in gorev1_df.iterrows():
        birlesik_satirlar.append({
            "Görev":          "Görev 1: Başarısızlık (Binary)",
            "Hedef":          "failure_flag",
            "Model":          row.get("Model", ""),
            "Kisa":           row.get("Kisa", ""),
            "DL":             row.get("DL", False),
            "Birincil_Metrik":row.get("ROC_AUC", None),   # ROC-AUC
            "Ikincil_Metrik": row.get("F1", None),         # F1
            "Ucuncul_Metrik": row.get("PR_AUC", None),     # PR-AUC
            "Accuracy":       row.get("Accuracy", None),
            "Egitim_Sure_s":  row.get("Egitim_Sure_s", None),
            "M1_Adi":         "ROC-AUC",
            "M2_Adi":         "F1",
            "M3_Adi":         "PR-AUC",
        })

# Görev 2 — regression metrikleri
if not gorev2_df.empty:
    for _, row in gorev2_df.iterrows():
        birlesik_satirlar.append({
            "Görev":          "Görev 2: Uygunluk Skoru (Regresyon)",
            "Hedef":          "suitability_score",
            "Model":          row.get("Model", ""),
            "Kisa":           row.get("Kisa", ""),
            "DL":             row.get("DL", False),
            "Birincil_Metrik":row.get("R2", None),
            "Ikincil_Metrik": row.get("RMSE", None),
            "Ucuncul_Metrik": row.get("MAE", None),
            "Accuracy":       None,
            "Egitim_Sure_s":  row.get("Egitim_Sure_s", None),
            "M1_Adi":         "R²",
            "M2_Adi":         "RMSE",
            "M3_Adi":         "MAE",
        })

# Görev 3 — multiclass metrikleri
if not gorev3_df.empty:
    for _, row in gorev3_df.iterrows():
        birlesik_satirlar.append({
            "Görev":          "Görev 3: Stres Düzeyi (Multiclass)",
            "Hedef":          "stress_level",
            "Model":          row.get("Model", ""),
            "Kisa":           row.get("Kisa", ""),
            "DL":             row.get("DL", False),
            "Birincil_Metrik":row.get("Macro_F1", None),
            "Ikincil_Metrik": row.get("Weighted_F1", None),
            "Ucuncul_Metrik": row.get("Macro_ROC_AUC", None),
            "Accuracy":       row.get("Accuracy", None),
            "Egitim_Sure_s":  row.get("Egitim_Sure_s", None),
            "M1_Adi":         "Macro F1",
            "M2_Adi":         "Weighted F1",
            "M3_Adi":         "Macro ROC-AUC",
        })

unified_df = pd.DataFrame(birlesik_satirlar)
csv_kaydet(unified_df, "unified_model_karsilastirma.csv")
adim(f"Unified tablo: {len(unified_df)} satır")

# En iyi modeller özet tablosu
g1_best = g1_best_meta if g1_best_meta else {}
g2_best = g2_best_meta if g2_best_meta else {}
g3_best = g3_best_meta if g3_best_meta else {}

en_iyi = pd.DataFrame([
    {
        "Görev":         "Görev 1: Başarısızlık Tahmini",
        "Problem_Tipi":  "Binary Classification",
        "Hedef":         "failure_flag",
        "En_İyi_Model":  g1_best.get("best_model_ad",  "—"),
        "Model_Kisa":    g1_best.get("best_model_kisa","—"),
        "Birincil":      f"ROC-AUC = {g1_best.get('best_roc_auc', '—')}",
        "Ikincil":       f"F1 = {g1_best.get('best_f1', '—')}",
        "Ucuncul":       f"PR-AUC = {g1_best.get('best_pr_auc', '—')}",
        "SHAP_OK":       g1_best.get("shap_ok", False),
        "Duyarlilik_OK": g1_best.get("duyar_ok", False),
    },
    {
        "Görev":         "Görev 2: Uygunluk Skoru Tahmini",
        "Problem_Tipi":  "Regression",
        "Hedef":         "suitability_score",
        "En_İyi_Model":  g2_best.get("best_model_ad",  "—"),
        "Model_Kisa":    g2_best.get("best_model_kisa","—"),
        "Birincil":      f"R² = {g2_best.get('best_r2', '—')}",
        "Ikincil":       f"RMSE = {g2_best.get('best_rmse', '—')}",
        "Ucuncul":       f"MAE = {g2_best.get('best_mae', '—')}",
        "SHAP_OK":       g2_best.get("shap_ok", False),
        "Duyarlilik_OK": g2_best.get("duyar_ok", False),
    },
    {
        "Görev":         "Görev 3: Stres Düzeyi Tahmini",
        "Problem_Tipi":  "Multiclass Classification",
        "Hedef":         "stress_level",
        "En_İyi_Model":  g3_best.get("best_model_ad",  "—"),
        "Model_Kisa":    g3_best.get("best_model_kisa","—"),
        "Birincil":      f"Macro F1 = {g3_best.get('best_macro_f1', '—')}",
        "Ikincil":       f"Weighted F1 = {g3_best.get('best_weighted_f1', '—')}",
        "Ucuncul":       f"ROC-AUC = {g3_best.get('best_roc_auc', '—')}",
        "SHAP_OK":       g3_best.get("shap_ok", False),
        "Duyarlilik_OK": g3_best.get("duyar_ok", False),
    },
])
csv_kaydet(en_iyi, "en_iyi_modeller_ozet.csv")
adim("En iyi modeller özet tablosu kaydedildi")

tamam("Unified tablolar hazır")


# =============================================================================
# 4. SHAP CROSS-TASK ÖZELLİK ÖNEMİ
# =============================================================================

bolum("BÖLÜM 3 — SHAP Cross-Task Özellik Önemi")

TOP_N_SHAP = 15

def shap_top_n(df: pd.DataFrame, n: int = TOP_N_SHAP) -> pd.DataFrame:
    """Standart kolon isimlerine normalize eder ve top-n döner."""
    if df.empty:
        return pd.DataFrame(columns=["Feature", "Mean_Abs_SHAP"])
    # feature/Feature ve mean_abs_shap/Mean_Abs_SHAP kolonlarını bul
    col_feat = next((c for c in df.columns if c.lower() in ("feature", "features")), None)
    col_shap = next((c for c in df.columns if "shap" in c.lower()), None)
    if col_feat is None or col_shap is None:
        return pd.DataFrame(columns=["Feature", "Mean_Abs_SHAP"])
    out = df[[col_feat, col_shap]].copy()
    out.columns = ["Feature", "Mean_Abs_SHAP"]
    out = out.dropna().sort_values("Mean_Abs_SHAP", ascending=False).head(n)
    return out.reset_index(drop=True)

g1_top = shap_top_n(g1_shap)
g2_top = shap_top_n(g2_shap)
g3_top = shap_top_n(g3_shap)

# Cross-task önem tablosu: tüm özelliklerin 3 görevdeki sırasını karşılaştır
tum_ozellikler = pd.Index(
    g1_top["Feature"].tolist() +
    g2_top["Feature"].tolist() +
    g3_top["Feature"].tolist()
).unique().tolist()

cross_shap_rows = []
for feat in tum_ozellikler:
    row = {"Feature": feat}
    for tag, df_t in [("G1_SHAP", g1_top), ("G2_SHAP", g2_top), ("G3_SHAP", g3_top)]:
        match = df_t.loc[df_t["Feature"] == feat, "Mean_Abs_SHAP"]
        row[tag] = round(float(match.values[0]), 6) if len(match) > 0 else None
    cross_shap_rows.append(row)

cross_shap_df = pd.DataFrame(cross_shap_rows)
if not cross_shap_df.empty:
    _sort_cols = [c for c in ["G1_SHAP", "G2_SHAP", "G3_SHAP"] if c in cross_shap_df.columns]
    if _sort_cols:
        cross_shap_df = cross_shap_df.sort_values(
            _sort_cols, ascending=False, na_position="last"
        ).reset_index(drop=True)
csv_kaydet(cross_shap_df, "shap_cross_task_onem.csv")
adim(f"Cross-task SHAP tablo: {len(cross_shap_df)} özellik")

tamam("SHAP cross-task tablosu hazır")


# =============================================================================
# 5. DUYARLILIK CROSS-TASK ÖZET
# =============================================================================

bolum("BÖLÜM 4 — Duyarlılık Cross-Task Özeti")

def duyar_normalize(df: pd.DataFrame, metrik_col: str, oran_col: str = None) -> pd.DataFrame:
    """Duyarlılık özet tablosunu standart formata getirir."""
    if df.empty:
        return pd.DataFrame()
    if oran_col is None:
        oran_col = next((c for c in df.columns
                         if "oran" in c.lower() or "gurultu" in c.lower()), None)
    if oran_col is None:
        return pd.DataFrame()
    metrik = next((c for c in df.columns if metrik_col.lower() in c.lower()), None)
    if metrik is None:
        return pd.DataFrame()
    out = df[[oran_col, metrik]].copy()
    out.columns = ["Gurultu_Oran", "Metrik"]
    out["Gurultu_Pct"] = (out["Gurultu_Oran"] * 100).round(0).astype(int)
    return out.sort_values("Gurultu_Oran").reset_index(drop=True)

# Görev 1: birincil metrik = ROC-AUC ortalama
g1_d = duyar_normalize(g1_duyar, "ROC_AUC_mean")
# Görev 2: birincil metrik = R2_ort
g2_d = duyar_normalize(g2_duyar, "R2_ort", "Gurultu_Orani")
if g2_d.empty:
    g2_d = duyar_normalize(g2_duyar, "R2_mean")
# Görev 3: birincil metrik = Macro_F1_mean
g3_d = duyar_normalize(g3_duyar, "Macro_F1_mean")
if g3_d.empty:
    g3_d = duyar_normalize(g3_duyar, "Macro_F1")

cross_duyar = pd.DataFrame()
for gorev_tag, df_d, metrik_adi in [
    ("G1_ROC_AUC", g1_d, "ROC-AUC"),
    ("G2_R2",      g2_d, "R²"),
    ("G3_MacroF1", g3_d, "Macro F1"),
]:
    if df_d.empty:
        continue
    tmp = df_d[["Gurultu_Pct", "Metrik"]].copy()
    tmp.columns = ["Gurultu_Pct", gorev_tag]
    cross_duyar = tmp if cross_duyar.empty else \
        cross_duyar.merge(tmp, on="Gurultu_Pct", how="outer")

if not cross_duyar.empty:
    cross_duyar = cross_duyar.sort_values("Gurultu_Pct").reset_index(drop=True)
    csv_kaydet(cross_duyar, "duyarlilik_cross_task.csv")
    adim(f"Cross-task duyarlılık tablo: {len(cross_duyar)} satır")
else:
    adim("⚠ Duyarlılık verileri yüklenemedi — grafik atlanacak.")

tamam("Duyarlılık cross-task tablosu hazır")


# =============================================================================
# 6. GÖRSELLEŞTİRMELER
# =============================================================================

bolum("BÖLÜM 5 — Görselleştirmeler")

# Renk paleti (görev bazlı)
RENKLER = {
    "Görev 1 (failure_flag)":       "#e74c3c",
    "Görev 2 (suitability_score)":  "#3498db",
    "Görev 3 (stress_level)":       "#27ae60",
}
GOREV_RENKLERI = list(RENKLER.values())

# ── 6.1  En İyi Modeller Metrik Özet Barplot ────────────────────────────
adim("Görsel 1: En iyi modeller metrik özet barplot...")
gorev_etiketleri = [
    f"G1: {g1_best.get('best_model_kisa','?')} [{g1_best.get('best_model_ad','—')}]",
    f"G2: {g2_best.get('best_model_kisa','?')} [{g2_best.get('best_model_ad','—')}]",
    f"G3: {g3_best.get('best_model_kisa','?')} [{g3_best.get('best_model_ad','—')}]",
]
birincil_degerler = [
    g1_best.get("best_roc_auc"),
    g2_best.get("best_r2"),
    g3_best.get("best_macro_f1"),
]
ikincil_degerler = [
    g1_best.get("best_f1"),
    None,   # RMSE — farklı ölçek, bar'da gösterilmez
    g3_best.get("best_weighted_f1"),
]
birincil_etiketler = ["ROC-AUC", "R²", "Macro F1"]

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle("Üç Görevin En İyi Modelleri — Birincil Metrik Karşılaştırması",
             fontsize=13, fontweight="bold")

for i, (ax_, gorev_lbl, val, m_lbl, renk) in enumerate(zip(
        axes, gorev_etiketleri, birincil_degerler,
        birincil_etiketler, GOREV_RENKLERI)):
    if val is None:
        ax_.text(0.5, 0.5, "Veri Yok", ha="center", va="center", transform=ax_.transAxes)
        ax_.set_title(gorev_lbl, fontsize=9, fontweight="bold")
        continue
    # Sadece o görevin modellerini al
    gorev_adi_map = {0: "Görev 1", 1: "Görev 2", 2: "Görev 3"}
    if i == 0 and not gorev1_df.empty:
        df_g = gorev1_df.copy()
        metrik_col = "ROC_AUC"
    elif i == 1 and not gorev2_df.empty:
        df_g = gorev2_df.copy()
        metrik_col = "R2"
    elif i == 2 and not gorev3_df.empty:
        df_g = gorev3_df.copy()
        metrik_col = "Macro_F1"
    else:
        ax_.text(0.5, 0.5, "Veri Yok", ha="center", va="center", transform=ax_.transAxes)
        continue

    df_g = df_g[[c for c in ["Model", "Kisa", metrik_col, "DL"] if c in df_g.columns]]
    df_g = df_g.dropna(subset=[metrik_col]).sort_values(metrik_col, ascending=True)
    barlar_renk = [renk if k == [g1_best_kisa, g2_best_kisa, g3_best_kisa][i]
                   else "#bdc3c7" for k in df_g.get("Kisa", df_g["Model"])]
    bars = ax_.barh(df_g["Kisa"] if "Kisa" in df_g.columns else df_g["Model"],
                    df_g[metrik_col], color=barlar_renk, edgecolor="white")
    for bar, val_ in zip(bars, df_g[metrik_col]):
        ax_.text(bar.get_width() + 0.005,
                 bar.get_y() + bar.get_height() / 2,
                 f"{val_:.4f}", va="center", ha="left", fontsize=7.5)
    ax_.set_xlabel(m_lbl, fontsize=10)
    ax_.set_title(gorev_lbl, fontsize=9, fontweight="bold", wrap=True)
    ax_.set_xlim(0 if i == 1 else 0,
                 1.10 if i != 1 else df_g[metrik_col].max() * 1.12)

plt.tight_layout()
png_kaydet(fig, "05_best_models_metrik_ozet.png")
adim("Görsel 1 ✓")

# ── 6.2  Unified Performance Heatmap ────────────────────────────────────
adim("Görsel 2: Unified performans ısı haritası...")
heat_frames = []
if not gorev1_df.empty:
    tmp = gorev1_df[["Model"] + [c for c in ["ROC_AUC", "F1", "PR_AUC", "Accuracy"]
                                  if c in gorev1_df.columns]].copy()
    tmp.columns = ["Model"] + [f"G1_{c}" for c in tmp.columns[1:]]
    heat_frames.append(tmp.set_index("Model"))

if not gorev2_df.empty:
    tmp = gorev2_df[["Model"] + [c for c in ["R2"] if c in gorev2_df.columns]].copy()
    tmp.columns = ["Model"] + [f"G2_{c}" for c in tmp.columns[1:]]
    heat_frames.append(tmp.set_index("Model"))

if not gorev3_df.empty:
    tmp = gorev3_df[["Model"] + [c for c in ["Macro_F1", "Weighted_F1", "Accuracy"]
                                  if c in gorev3_df.columns]].copy()
    tmp.columns = ["Model"] + [f"G3_{c}" for c in tmp.columns[1:]]
    heat_frames.append(tmp.set_index("Model"))

if heat_frames:
    heat_merged = heat_frames[0]
    for hf in heat_frames[1:]:
        heat_merged = heat_merged.join(hf, how="outer")
    heat_num = heat_merged.apply(pd.to_numeric, errors="coerce")

    fig, ax = plt.subplots(figsize=(16, max(6, len(heat_num) * 0.6)))
    sns.heatmap(heat_num, annot=True, fmt=".3f", cmap="RdYlGn",
                vmin=0, vmax=1, linewidths=0.4, ax=ax,
                annot_kws={"size": 7}, cbar_kws={"shrink": 0.7})
    ax.set_title("Üç Görev × Tüm Modeller — Unified Performans Isı Haritası\n"
                 "(G1=Sınıflandırma, G2=Regresyon, G3=Çok Sınıflı)",
                 fontsize=12, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    plt.tight_layout()
    png_kaydet(fig, "02_unified_performance_heatmap.png")
    adim("Görsel 2 ✓")
else:
    adim("⚠ Görsel 2 atlandı — model sonucu yok.")

# ── 6.3  SHAP Cross-Task Bar ─────────────────────────────────────────────
adim("Görsel 3: SHAP cross-task özellik önemi...")
tum_shap_bos = all(df.empty for df in [g1_top, g2_top, g3_top])

if not tum_shap_bos:
    # Top-10 özellik her görev için yan yana
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    shap_info = [
        (g1_top, f"Görev 1: failure_flag\n[{g1_best_kisa}]", "#e74c3c"),
        (g2_top, f"Görev 2: suitability_score\n[{g2_best_kisa}]", "#3498db"),
        (g3_top, f"Görev 3: stress_level\n[{g3_best_kisa}]", "#27ae60"),
    ]
    for ax_, (df_s, basl, renk) in zip(axes, shap_info):
        if df_s.empty:
            ax_.text(0.5, 0.5, "SHAP verisi yok",
                     ha="center", va="center", transform=ax_.transAxes)
            ax_.set_title(basl, fontsize=10, fontweight="bold")
            continue
        top10 = df_s.head(10)
        bars = ax_.barh(top10["Feature"][::-1], top10["Mean_Abs_SHAP"][::-1],
                        color=renk, edgecolor="white", alpha=0.85)
        for bar, val_ in zip(bars, top10["Mean_Abs_SHAP"][::-1]):
            ax_.text(bar.get_width() * 1.01,
                     bar.get_y() + bar.get_height() / 2,
                     f"{val_:.4f}", va="center", ha="left", fontsize=7.5)
        ax_.set_xlabel("Ortalama |SHAP Değeri|", fontsize=9)
        ax_.set_title(basl, fontsize=10, fontweight="bold")

    plt.suptitle("SHAP Özellik Önemi — Üç Görev Karşılaştırması (Top 10)\n"
                 "Hangi sensör değişkenleri her hedefi belirliyor?",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    png_kaydet(fig, "03_shap_cross_task_bar.png")
    adim("Görsel 3 ✓")

    # ── SHAP ortak özellikler: Her görevde ilk 10'a giren özellikler ──────
    g1_set = set(g1_top["Feature"].tolist()) if not g1_top.empty else set()
    g2_set = set(g2_top["Feature"].tolist()) if not g2_top.empty else set()
    g3_set = set(g3_top["Feature"].tolist()) if not g3_top.empty else set()
    ortak_3 = g1_set & g2_set & g3_set
    ortak_2 = (g1_set & g2_set) | (g1_set & g3_set) | (g2_set & g3_set)
    adim(f"3 görevde de Top-10'da olan özellikler: {ortak_3}")
    adim(f"En az 2 görevde Top-10'da olan özellikler: {ortak_2 - ortak_3}")

    # Ortak özellik tablosunu kaydet
    ortak_df = pd.DataFrame({
        "Feature":           list(set(g1_top["Feature"]) | set(g2_top["Feature"]) | set(g3_top["Feature"])),
    })
    ortak_df["G1_Top10"] = ortak_df["Feature"].isin(g1_set)
    ortak_df["G2_Top10"] = ortak_df["Feature"].isin(g2_set)
    ortak_df["G3_Top10"] = ortak_df["Feature"].isin(g3_set)
    ortak_df["Kac_Gorevde"] = ortak_df[["G1_Top10","G2_Top10","G3_Top10"]].sum(axis=1)
    ortak_df = ortak_df.sort_values("Kac_Gorevde", ascending=False).reset_index(drop=True)
    csv_kaydet(ortak_df, "shap_ortak_ozellikler.csv")
    adim("SHAP ortak özellik tablosu ✓")
else:
    adim("⚠ Görsel 3 atlandı — SHAP verisi yok.")

# ── 6.4  Duyarlılık Cross-Task ───────────────────────────────────────────
adim("Görsel 4: Duyarlılık cross-task...")

if not cross_duyar.empty:
    fig, ax = plt.subplots(figsize=(10, 6))
    col_bilgi = [
        ("G1_ROC_AUC", "Görev 1: failure_flag (ROC-AUC)", "#e74c3c", "o-"),
        ("G2_R2",      "Görev 2: suitability_score (R²)", "#3498db", "s--"),
        ("G3_MacroF1", "Görev 3: stress_level (Macro F1)", "#27ae60", "^:"),
    ]
    for col, lbl, renk, stil in col_bilgi:
        if col not in cross_duyar.columns:
            continue
        df_plot = cross_duyar[["Gurultu_Pct", col]].dropna()
        ax.plot(df_plot["Gurultu_Pct"], df_plot[col],
                stil, color=renk, lw=2.5, markersize=8, label=lbl)

    ax.set_xlabel("Gaussian Gürültü Oranı (%)", fontsize=12)
    ax.set_ylabel("Birincil Metrik Değeri", fontsize=12)
    ax.set_title("Üç Görev — Duyarlılık Analizi Karşılaştırması\n"
                 "(Sensör gürültüsüne karşı model kararlılığı)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(cross_duyar["Gurultu_Pct"].dropna().unique())
    plt.tight_layout()
    png_kaydet(fig, "04_duyarlilik_cross_task.png")
    adim("Görsel 4 ✓")
else:
    adim("⚠ Görsel 4 atlandı — duyarlılık verisi yok.")

# ── 6.5  Radar Chart — 3 görevin en iyi modelleri ────────────────────────
adim("Görsel 5: Radar chart (en iyi modeller)...")

radar_degerler = [
    [
        g1_best.get("best_roc_auc") or 0,
        g1_best.get("best_f1") or 0,
        g1_best.get("best_pr_auc") or 0,
        1 - (g1_best.get("best_roc_auc", 1) * 0),  # placeholder symmetry
        g1_best.get("best_f1") or 0,
    ],
]

# Radar için uygun: farklı ölçekleri normalize et
# G1: ROC-AUC, F1, PR-AUC (tümü 0-1)
# G2: R² (0-1), 1-RMSE/max_rmse (normalize)
# G3: Macro F1, Weighted F1, ROC-AUC (0-1)
radar_kategoriler = ["ROC-AUC / R²", "F1 / R²", "PR-AUC / W-F1", "Accuracy", "3. Metrik"]

# Görev bazlı normalize radar
g1_vals = [
    g1_best.get("best_roc_auc") or 0,
    g1_best.get("best_f1") or 0,
    g1_best.get("best_pr_auc") or 0,
]
g2_vals = [
    max(0, g2_best.get("best_r2") or 0),
    max(0, g2_best.get("best_r2") or 0),       # R² ikinci eksende de
    0,                                           # PR-AUC yok
]
g3_vals = [
    g3_best.get("best_roc_auc") or 0,
    g3_best.get("best_macro_f1") or 0,
    g3_best.get("best_weighted_f1") or 0,
]

# Basit 3-eksen radar
kategoriler_3 = ["Birincil Metrik\n(ROC-AUC / R² / Macro F1)",
                  "İkincil Metrik\n(F1 / R² / Weighted F1)",
                  "Üçüncül Metrik\n(PR-AUC / — / ROC-AUC)"]
vals_liste = [g1_vals, g2_vals, g3_vals]

N = len(kategoriler_3)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
for (vals, gorev_adi, renk) in zip(
        vals_liste,
        [f"Görev 1 [{g1_best_kisa}]",
         f"Görev 2 [{g2_best_kisa}]",
         f"Görev 3 [{g3_best_kisa}]"],
        GOREV_RENKLERI):
    vals_c = vals + vals[:1]
    ax.plot(angles, vals_c, lw=2.5, color=renk, label=gorev_adi)
    ax.fill(angles, vals_c, alpha=0.10, color=renk)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(kategoriler_3, fontsize=10)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
ax.set_title("Üç Görevin En İyi Modelleri — Radar Karşılaştırması",
             fontsize=12, fontweight="bold", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
plt.tight_layout()
png_kaydet(fig, "01_en_iyi_modeller_radar.png")
adim("Görsel 5 ✓")

tamam("Tüm görselleştirmeler tamamlandı")


# =============================================================================
# 7. AKADEMİK ÖZET RAPORU
# =============================================================================

bolum("BÖLÜM 6 — Akademik Özet Raporu")

# Birincil metrik değerleri için formatlanmış metin
def fmt(val, fallback="—"):
    if val is None:
        return fallback
    return f"{float(val):.5f}"

# SHAP ortak özellikler metni
shap_ortak_metni = ""
if not tum_shap_bos:
    ortak_3_sirali = sorted(ortak_3) if ortak_3 else []
    shap_ortak_metni = (
        f"Her üç görevde de ilk 10 özellik arasında yer alanlar: "
        f"{', '.join(ortak_3_sirali) if ortak_3_sirali else 'Ortak özellik yok'}"
    )

# Duyarlılık özet metni
duyar_ozet_metni = ""
if not cross_duyar.empty:
    satirlar = ["| Gürültü | G1 ROC-AUC | G2 R² | G3 Macro F1 |",
                "|---------|-----------|-------|------------|"]
    for _, row_ in cross_duyar.iterrows():
        lbl = f"%{int(row_['Gurultu_Pct'])}"
        g1v = f"{row_.get('G1_ROC_AUC', float('nan')):.5f}" \
              if pd.notna(row_.get("G1_ROC_AUC")) else "—"
        g2v = f"{row_.get('G2_R2', float('nan')):.5f}" \
              if pd.notna(row_.get("G2_R2")) else "—"
        g3v = f"{row_.get('G3_MacroF1', float('nan')):.5f}" \
              if pd.notna(row_.get("G3_MacroF1")) else "—"
        satirlar.append(f"| {lbl} | {g1v} | {g2v} | {g3v} |")
    duyar_ozet_metni = "\n".join(satirlar)
else:
    duyar_ozet_metni = "Duyarlılık verileri yüklenemedi."

rapor_metni = f"""# AŞAMA 5 — KONSOLİDASYON RAPORU
# Üç Görev Karşılaştırması, SHAP ve Duyarlılık Özeti

## Öğrenci
İbrahim Nuryağınlı | 25490221001

## Kaynak Aşamalar
- Aşama 2: {A2_DIR.name if A2_DIR else "Bulunamadı"}
- Aşama 3: {A3_DIR.name if A3_DIR else "Bulunamadı"}
- Aşama 4: {A4_DIR.name if A4_DIR else "Bulunamadı"}

---

## 1. Proje Genel Bakış

Bu proje; toprak sensörü verilerinden bitki sağlığını üç farklı boyutta
tahmin eden entegre bir makine öğrenmesi sistemi geliştirmeyi amaçlamaktadır:

| # | Görev | Hedef Değişken | Problem Tipi | Temel Metrik |
|---|-------|----------------|--------------|--------------|
| 1 | Başarısızlık Tahmini | failure_flag | Binary Classification | ROC-AUC |
| 2 | Uygunluk Skoru Tahmini | suitability_score | Regression | R² |
| 3 | Stres Düzeyi Tahmini | stress_level | Multiclass Classification | Macro F1 |

---

## 2. En İyi Modeller Özeti

| Görev | En İyi Model | Birincil Metrik | İkincil Metrik | Üçüncül Metrik |
|-------|-------------|-----------------|----------------|----------------|
| Görev 1: failure_flag | {g1_best.get('best_model_ad','—')} [{g1_best_kisa}] | ROC-AUC = {fmt(g1_best.get('best_roc_auc'))} | F1 = {fmt(g1_best.get('best_f1'))} | PR-AUC = {fmt(g1_best.get('best_pr_auc'))} |
| Görev 2: suitability_score | {g2_best.get('best_model_ad','—')} [{g2_best_kisa}] | R² = {fmt(g2_best.get('best_r2'))} | RMSE = {fmt(g2_best.get('best_rmse'))} | MAE = {fmt(g2_best.get('best_mae'))} |
| Görev 3: stress_level | {g3_best.get('best_model_ad','—')} [{g3_best_kisa}] | Macro F1 = {fmt(g3_best.get('best_macro_f1'))} | Weighted F1 = {fmt(g3_best.get('best_weighted_f1'))} | ROC-AUC = {fmt(g3_best.get('best_roc_auc'))} |

---

## 3. SHAP Analizi Konsolidasyonu

Her görevin en iyi modeline TreeExplainer (ağaç tabanlı) veya
KernelExplainer (diğerleri) uygulanmıştır.

### 3.1 Görev Bazlı Top-3 SHAP Özellikleri

**Görev 1 (failure_flag):**
{chr(10).join(f"  {i+1}. {r['Feature']} (|SHAP|={r['Mean_Abs_SHAP']:.5f})" for i, (_, r) in enumerate(g1_top.head(3).iterrows())) if not g1_top.empty else "  SHAP verisi yok."}

**Görev 2 (suitability_score):**
{chr(10).join(f"  {i+1}. {r['Feature']} (|SHAP|={r['Mean_Abs_SHAP']:.5f})" for i, (_, r) in enumerate(g2_top.head(3).iterrows())) if not g2_top.empty else "  SHAP verisi yok."}

**Görev 3 (stress_level):**
{chr(10).join(f"  {i+1}. {r['Feature']} (|SHAP|={r['Mean_Abs_SHAP']:.5f})" for i, (_, r) in enumerate(g3_top.head(3).iterrows())) if not g3_top.empty else "  SHAP verisi yok."}

### 3.2 Ortak Önemli Özellikler
{shap_ortak_metni if shap_ortak_metni else "SHAP verisi yetersiz — karşılaştırma yapılamadı."}

---

## 4. Duyarlılık Analizi Konsolidasyonu

Tüm modellere %5, %10 ve %20 oranında Gaussian gürültü eklenerek
sensör kalitesindeki bozulmaların model performansına etkisi ölçülmüştür.

{duyar_ozet_metni}

### Yorum
Tablodan görüleceği üzere, gürültü artışı karşısında [modellerin karşılaştırmalı
kararlılığı] görülmektedir. Bu bulgu, gerçek saha koşullarında sensör
hatalarının kabul edilebilir bir bant genişliğinde tutulması gerektiğini
göstermektedir.

---

## 5. Üç Görev Karşılaştırması — Metodolojik Yorum

### Neden Farklı Metrikler?
Görev 1'de sınıf dengesizliği (~5.17:1) nedeniyle Accuracy yanıltıcıdır;
ROC-AUC ve PR-AUC tercih edilmiştir. Görev 2 sürekli değişken içerdiğinden
R² ve RMSE uygundur. Görev 3'te çok sınıflı yapı nedeniyle her sınıfın
başarısını eşit ağırlıkla yansıtan Macro F1 kullanılmıştır.

### Ensemble Yöntemlerin Baskınlığı
Random Forest, XGBoost, LightGBM veya CatBoost modellerinden biri
her üç görevde de üst sıralarda yer almaktadır. Bu bulgu, literatürdeki
ensemble yöntemlerinin tablolu veri setlerindeki üstünlüğüyle (Van Klompenburg
vd., 2020; Moshou vd., 2014) örtüşmektedir.

### CNN ve LSTM — Akademik Değerlendirme
CNN-1D ve RNN-LSTM modelleri, tablolu sensör verisinde ensemble modellerine
kıyasla düşük performans göstermiştir. Bu beklenen bir bulgudur: CNN uzamsal
yakınlık ilişkilerini, LSTM ise zamansal sıra bağımlılıklarını öğrenmek için
tasarlanmıştır; kesitsel sensör verisinde bu ön koşullar sağlanmamaktadır.
Sonuçların raporlanması bilimsel dürüstlük ve akademik eksiksizlik açısından
zorunludur.

---

## 6. Rapor Bölümlerine Doğrudan Katkı

Bu konsolidasyon dosyasının rapordaki karşılıkları:

| Rapor Bölümü | İlgili Dosya |
|---|---|
| Özet | asama5_akademik_ozet.md → özet paragrafı |
| Veri Seti ve Ön İşleme | asama1_ciktilar → akademik_rapor.md |
| Yöntem | asama2/3/4 rapor dosyaları |
| Deneysel Sonuçlar | unified_model_karsilastirma.csv + görseller |
| SHAP Analizi | 03_shap_cross_task_bar.png + shap_cross_task_onem.csv |
| Duyarlılık Analizi | 04_duyarlilik_cross_task.png + duyarlilik_cross_task.csv |
| Sonuç ve Değerlendirme | bu rapor § 5 |

---

## 7. Kaynakça
- Liakos vd. (2018). Machine learning in agriculture. Sensors, 18(8), 2674.
- Lundberg & Lee (2017). A unified approach to interpreting model predictions. NIPS, 30.
- Moshou vd. (2014). Intelligent multi-sensor system for detection of fungal diseases.
  Biosystems Engineering, 117, 94–103.
- Rashid vd. (2021). A comprehensive review of crop yield prediction using ML.
  IEEE Access, 9, 63406–63439.
- Roy, N. (2024). Agro-environmental stress & failure simulation. Kaggle.
- Van Klompenburg vd. (2020). Crop yield prediction using machine learning.
  Computers and Electronics in Agriculture, 177, 105709.

---
Oluşturulma: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Çıktı Dizini: {CIKTI}
"""

rapor_path = RAPOR / "asama5_akademik_ozet.md"
rapor_path.write_text(rapor_metni, encoding="utf-8")
adim(f"Rapor: {rapor_path.name}")
tamam("Akademik rapor tamamlandı")


# =============================================================================
# 8. DOSYA MANİFESTİ VE FİNAL ÖZET
# =============================================================================

bolum("BÖLÜM 7 — Dosya Manifesti ve Final Özet")

manifest = []
for kat, dizin in [("tablo", TABLO), ("gorsel", GRAFIK), ("rapor", RAPOR)]:
    for fp in sorted(dizin.glob("*")):
        if fp.is_file():
            manifest.append({
                "kategori": kat,
                "dosya": fp.name,
                "boyut_KB": round(fp.stat().st_size / 1024, 2),
            })
pd.DataFrame(manifest).to_csv(
    CIKTI / "FILE_MANIFEST.csv", index=False, encoding="utf-8-sig")

n_gorsel = len(list(GRAFIK.glob("*.png")))
n_tablo  = len(list(TABLO.glob("*.csv")))

print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║           AŞAMA 5 — KONSOLİDASYON TAMAMLANDI                        ║
╠══════════════════════════════════════════════════════════════════════╣
║  Kaynak: Aşama 2 + 3 + 4 çıktıları                                   ║
║  Tablolar: {n_tablo}  |  Görseller: {n_gorsel}                           
║                                                                       ║
║  EN İYİ MODELLER                                                      ║
║  G1 failure_flag     : [{g1_best_kisa}] {g1_best.get('best_model_ad','—')[:30]}
║    ROC-AUC={fmt(g1_best.get('best_roc_auc'))}  F1={fmt(g1_best.get('best_f1'))}
║  G2 suitability_score: [{g2_best_kisa}] {g2_best.get('best_model_ad','—')[:30]}
║    R²={fmt(g2_best.get('best_r2'))}  RMSE={fmt(g2_best.get('best_rmse'))}
║  G3 stress_level     : [{g3_best_kisa}] {g3_best.get('best_model_ad','—')[:30]}
║    Macro F1={fmt(g3_best.get('best_macro_f1'))}  W-F1={fmt(g3_best.get('best_weighted_f1'))}
║                                                                       ║
║  Çıktı: {CIKTI.name}
╚══════════════════════════════════════════════════════════════════════╝

▶ KOD AŞAMALARI TAMAMLANDI.
  Sıradaki adım: AŞAMA 6 — Rapor (PDF) ve Sunum (PPTX) hazırlığı.
  Bu aşamada .py dosyası yazılmaz; belgeler hazırlanır.
""", flush=True)
