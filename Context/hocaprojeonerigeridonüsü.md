Merhaba Arkadaşlar,

Sunduğunuz "Toprak Sensör Verileriyle Bitki Sağlığının Çok Hedefli Makine Öğrenmesi Modelleriyle Tahmini" başlıklı proje önerisi tarafımca incelenmiş ve çalışma kabul edilebilir düzeyde bulunmuştur. Çalışmanızın geliştirilmesi ve dersin kapsamına uygunluğu açısından aşağıdaki hususları dikkate almanızı rica ederim:

-   Amaçlar kısmında belirttiğiniz Açıklanabilir Yapay Zeka (XAI/SHAP) yöntemleri ders müfredatımızda yer almamaktadır. Ancak bu konuyu kendi araştırmalarınızla projeye dahil etmeniz çalışmanıza değer katacaktır. Şayet bu yöntemleri başarıyla entegre ederseniz, çalışmanıza sensör verilerine farklı oranlarda yapay gürültü (noise) ekleyerek modelin kararlılığını test edeceğiniz bir "duyarlılık analizi" eklemeniz de yerinde olacaktır.
-   Proje önerisinde analiz dışı bırakılacağını belirttiğiniz değişkenleri (location_id, ph_stress_flag vb.) modele başlangıç aşamasında dahil etmenizi istiyorum. Aksi takdirde çalışma yalnızca 8-10 değişken üzerinden yürütülecek ve bu durum analizin derinliğini sınırlayacaktır. Tüm değişkenlerle çalışmaya başlayıp, istatistiksel anlamda ilişkisi bulunmayanları testler sonucunda elemeniz daha akademik bir yaklaşım olacaktır.
-   Bitki ölümünü temsil eden failure_flag değişkeninde, ölen bitki sayısının yaşayanlara oranla belirgin düzeyde az olması beklenmektedir. Bu tür dengesiz veri setlerinde "Doğruluk" (Accuracy) oranı tek başına yanıltıcı sonuçlar verebilir. Bu nedenle, model başarısını değerlendirirken F1-Skoru ve Precision-Recall eğrilerini temel metrikler olarak kullanmanız gerekmektedir.
-   Kullanacağınız veri setinin sentetik olması nedeniyle modelin gerçek saha koşullarını tam yansıtmama riski bulunmaktadır. Ayrıca bağımsız değişkenler arasında yer alan toprak nemi ve yağış miktarı gibi özelliklerin birbiriyle yüksek korelasyona sahip olması muhtemeldir. Model karmaşıklığını yönetmek adına eğitim öncesinde çoklu doğrusallık (multicollinearity) kontrolü yapmanızı ve gerekirse boyut indirgeme tekniklerini değerlendirmenizi öneririm.

Çalışmalarınızda başarılar dilerim.



İyi günler dilerim.

A.Aca