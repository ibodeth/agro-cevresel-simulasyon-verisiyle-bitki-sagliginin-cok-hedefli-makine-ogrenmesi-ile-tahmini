# gro-Cevresel Simulasyon Verisiyle Bitki Sagliginin Cok Hedefli Makine Ogrenmesi ile Tahmini

Bu repo, cevresel simulasyon ve toprak sensor verileri uzerinden bitki sagligini **cok hedefli makine ogrenmesi** yaklasimi ile tahmin etmek icin hazirlanmistir. Calisma 3 temel goreve odaklanir:

1. `failure_flag` ile basarisizlik/bitki olum tahmini (ikili siniflandirma)
2. `suitability_score` ile uygunluk skoru tahmini (regresyon)
3. `stress_level` ile stres duzeyi tahmini (cok sinifli siniflandirma)

Proje akisinin tasarimi, ders kapsam gereklilikleri ve ogretim uyesi geri bildirimlerine uygun sekilde kurgulanmistir:
- F1, PR-AUC, ROC-AUC, confusion matrix gibi metriklerin kullanimi
- Sinif dengesizliginde SMOTE ve/veya `class_weight` yaklasimi
- SHAP ile aciklanabilirlik
- %5/%10/%20 gurultu altinda duyarlilik (sensitivite) analizi

## Repo Icerigi

- `asama1.py`: Veri kesfi, on isleme, train/test ayrimi, leakage-onleyici pipeline, dengeleme ve model girdilerinin hazirlanmasi
- `asama2.py`: Gorev 1 - basarisizlik tahmini (binary classification)
- `asama3.py`: Gorev 2 - uygunluk skoru tahmini (regresyon)
- `asama4.py`: Gorev 3 - stres duzeyi tahmini (multiclass classification)
- `asama5.py`: Asama 2-3-4 ciktilarinin konsolidasyonu, karsilastirma tablolari ve genel ozet raporu
- `dataset.csv`: Projede kullanilan ana veri seti
- `requirements.txt`: Python kutuphane bagimliliklari
- `Context/`: Proje briefi, geri donusler ve yol haritasi
- `Bizim_raporlar/`: Raporlama notlari ve eklenebilecek icerikler

## Veri Seti Referansi

**Kullanilan veri seti:** `dataset.csv` (bu repoda bulunur)

- Kayit sayisi: 543,210
- Sutun sayisi: 25
- Ana degiskenler:
  - Cevresel/toprak ozellikleri: `soil_type`, `soil_moisture_pct`, `soil_temp_c`, `air_temp_c`, `light_intensity_par`, `soil_ph`, `nitrogen_ppm`, `phosphorus_ppm`, `potassium_ppm` vb.
  - Hedefler: `suitability_score`, `stress_level`, `failure_flag`

Not: Veri seti simulasyon/sentetik karakterdedir; bu nedenle cok yuksek metrikler gercek saha kosullarinda bire bir beklenmemelidir.

## Kurulum

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Calistirma Sirasi

```bash
python asama1.py
python asama2.py
python asama3.py
python asama4.py
python asama5.py
```

Her asama kendi zaman damgali cikti klasorunu olusturur:
- `asama1_ciktilar_*`
- `asama2_ciktilar_*`
- `asama3_ciktilar_*`
- `asama4_ciktilar_*`
- `asama5_ciktilar_*`

## Yontemsel Ozet

- On isleme: `ColumnTransformer` + `Pipeline` ile sayisal/kategorik akislarin ayrik yonetimi
- Dengesizlik yonetimi: `failure_flag` icin SMOTE degerlendirmesi, cok sinifli gorevde `class_weight`
- Modelleme: Klasik ML + ensemble + derin ogrenme (ANN/CNN/LSTM)
- Secim: GridSearchCV, birincil metrik bazli model karsilastirmasi
- Aciklanabilirlik: SHAP ile ozellik onemi
- Dayaniklilik: Gaussian gurultu ile duyarlilik analizi

## Akademik Not

Bu calisma ders projesi kapsaminda hazirlanmistir. Nihai rapor ve sunumda kod ciktilari yerine grafik, tablo ve yorum odakli anlatim tercih edilmelidir.
