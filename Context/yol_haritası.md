YZO 106 – Makine Öğrenmesi Dönem Projesi – Tam Plan
Teslim tarihi: 14 Mayıs 2026
Teslim şekli: .zip → PDF rapor + PPTX/PDF sunum + .py/.ipynb kodlar → abdulsamet.aca@erbakan.edu.tr

AŞAMA 1 – Veri Keşfi ve Ön İşleme
Veri seti 543.000+ satır ve 25 sütundan oluşuyor. İlk iş olarak tüm sütunları modele dahil ediyoruz — hoca başlangıçta hiçbir değişkeni çıkarmamamızı istedi. Sayısal değişkenlere eksik değer doldurma ve StandardScaler, kategorik değişkenlere mod ile doldurma ve OneHotEncoder uygulanacak. Tüm bu adımlar sklearn Pipeline ve ColumnTransformer içinde tutulacak, böylece data leakage olmayacak.
Sonrasında keşifsel veri analizi yapılacak: dağılım grafikleri, korelasyon matrisi ve VIF (Variance Inflation Factor) analizi ile çoklu doğrusallık kontrol edilecek. VIF değeri 10'un üzerinde olan değişkenler için PCA ya da LDA uygulanacak.
failure_flag sütununda sınıf dengesizliği bekleniyor. Bunu çözmek için SMOTE ve class_weight='balanced' denenecek, hangi yöntemin daha iyi çalıştığı raporda karşılaştırmalı gösterilecek.

AŞAMA 2 – Görev 1: Başarısızlık Tahmini (İkili Sınıflandırma)
Hedef değişken: failure_flag (0 = yaşıyor, 1 = öldü)
Denenecek modeller:

Logistic Regression
Decision Tree
KNN
SVM
Naive Bayes
Random Forest
XGBoost
LightGBM
CatBoost
ANN
CNN
RNN / LSTM

Kullanılacak metrikler: F1-Score, Precision-Recall eğrisi, AUC-ROC, Confusion Matrix. Accuracy tek başına kullanılmayacak çünkü sınıf dengesizliği yanıltıcı sonuç verebilir.
Tüm modeller GridSearchCV ile hiperparametre optimizasyonuna tabi tutulacak. En iyi modele SHAP uygulanarak hangi değişkenlerin tahmin üzerinde belirleyici olduğu görselleştirilecek.

AŞAMA 3 – Görev 2: Uygunluk Skoru Tahmini (Regresyon)
Hedef değişken: suitability_score (0–100 arası sürekli sayı)
Denenecek modeller:

Linear Regression
Decision Tree Regressor
KNN Regressor
SVR
Random Forest Regressor
XGBoost Regressor
LightGBM Regressor
CatBoost Regressor
ANN
CNN
RNN / LSTM

Kullanılacak metrikler: R², MSE, MAE. Tüm modeller karşılaştırmalı tablo halinde raporda sunulacak. Gerekirse PCA ile boyut indirgeme uygulanacak ve sonuçlara etkisi incelenecek.

AŞAMA 4 – Görev 3: Stres Düzeyi Tahmini (Çok Sınıflı Sınıflandırma)
Hedef değişken: stress_level (0 = Sağlıklı, 1 = Hafif Stres, 2 = Kritik Stres)
Denenecek modeller:

Logistic Regression
Decision Tree
KNN
SVM
Naive Bayes
Random Forest
XGBoost
LightGBM
CatBoost
ANN
CNN
RNN / LSTM

Kullanılacak metrikler: Macro F1-Score, sınıf bazlı Precision ve Recall, Confusion Matrix. CNN ve RNN/LSTM tablo verisi için uygun olmasa da denenecek ve neden düşük performans verdiği raporda akademik olarak açıklanacak. Bu yaklaşım bilimsel olgunluk göstereceği için ek puan getirecek.

AŞAMA 5 – SHAP ve Duyarlılık Analizi
Her 3 görevin en iyi modeline SHAP uygulanacak. SHAP, hangi sensör değişkeninin tahmin üzerinde ne kadar etkili olduğunu görselleştirecek. Bu bölüm hocanın önerdiği bonus kısım olup tam puan için kritik.
Bunun üzerine duyarlılık (sensitivite) analizi eklenecek. Veriye sırasıyla %5, %10 ve %20 oranında Gaussian gürültü eklenerek modelin performansının nasıl değiştiği incelenecek. Sonuçlar grafik ve tablo halinde raporda gösterilecek.

AŞAMA 6 – Rapor, Sunum ve Teslim
Rapor şu sırayla yazılacak: Özet, Veri Seti Tanıtımı ve Ön İşleme, Yöntem, Deneysel Sonuçlar ve Analiz, Sonuç ve Değerlendirme, Kaynakça. Kaynaklar IEEE veya APA formatında olacak. Rapor dili akademik olacak, kod çıktısı yerine grafik ve tablo kullanılacak.
Sunum 10–12 dakika olacak. İçerik sırası: problem tanımı, veri analizi, model seçimi, performans metrikleri, canlı demo. Gruptaki her üye aktif olarak anlatım yapacak — sunum yapmayan ya da hazır bulunmayan kişi o aşamadan puan alamıyor.
Teslim edilecek dosyalar .zip içinde şunları içerecek: PDF formatında rapor, PPTX veya PDF formatında sunum, eksiksiz çalışabilir .py veya .ipynb kod dosyaları.