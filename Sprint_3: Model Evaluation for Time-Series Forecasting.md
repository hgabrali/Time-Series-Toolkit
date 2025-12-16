
# 📈 Model Evaluation for Time-Series Forecasting


<img width="1373" height="700" alt="image" src="https://github.com/user-attachments/assets/132a51e3-f996-4911-bb0a-c5ba8e49d395" />


Zaman serisi tahmini (Time-Series Forecasting), verinin sıralı doğası nedeniyle geleneksel makine öğrenimi görevlerinden ayrılır. Bir modeli değerlendirmek, zamana duyarlı teknikler ve stratejik bir yaklaşım gerektirir.

Bu döküman, zaman serisi modellerinin değerlendirilmesinde kullanılan temel prensipleri, validasyon stratejilerini ve başarı metriklerini teknik bir derinlikle ele alır.

---

## 1. Temel Zorluklar ve Farklılıklar

Geleneksel makine öğreniminde veriler genellikle bağımsız ve aynı dağılıma sahip (IID) olarak kabul edilirken, zaman serilerinde bu durum geçerli değildir.

### ⏳ Temporal Dependency (Zaman Bağımlılığı)
Zaman serilerinde gözlemler zaman içinde sıralıdır ve mevcut değer genellikle önceki değerlere bağlıdır (otokorelasyon). 
* **Sorun:** Veriyi rastgele (random shuffle) "Eğitim" ve "Test" setlerine ayırmak, modelin geçmişten geleceği değil, "gelecekten geçmişi" öğrenmesine neden olur.
* **Çözüm:** Veri setleri her zaman kronolojik sıraya göre kesilmelidir.

### 💧 Data Leakage (Veri Sızıntısı)
Geleceğe ait bilgilerin (test seti) eğitim sürecine dahil olması durumudur.
* **Sonuç:** Model eğitimde harika performans gösterir ancak canlıya (production) alındığında çuvallar.
* **Kural:** Gelecek, geçmişi tahmin etmek için kullanılamaz.

---

## 2. Validasyon Stratejileri (Cross-Validation Techniques)

Zaman serilerinde standart *k-fold cross-validation* kullanılmaz. Bunun yerine "Walk-Forward Validation" (İleriye Yürüyen Doğrulama) teknikleri uygulanır.

### A. Hold-Out Yöntemi (Basit Kronolojik Bölme)
Veri seti belirli bir zaman noktasından itibaren ikiye (veya üçe) bölünür.
* **Eğitim Seti:** $t_0$'dan $t_n$'e kadar.
* **Test Seti:** $t_{n+1}$'den $t_{n+m}$'e kadar.

### B. Time Series Cross-Validation (Walk-Forward)
Bu yöntem, modelin zaman içindeki stabilitesini ölçmek için en güvenilir yoldur. İki ana yaklaşımı vardır:

#### 1. Expanding Window (Genişleyen Pencere)
Eğitim seti her adımda büyürken, test seti zaman ekseninde ileriye doğru kayar.
* **Kullanım Alanı:** Veri geçmişinin tamamı önemliyse ve "Concept Drift" (Veri dağılımının zamanla değişmesi) az ise kullanılır.

```text
Adım 1: [Train: Yıl 1-3] -> [Test: Yıl 4]
Adım 2: [Train: Yıl 1-4] -> [Test: Yıl 5]
Adım 3: [Train: Yıl 1-5] -> [Test: Yıl 6]
```
####  2. Rolling Window (Kayan Pencere)
Eğitim setinin boyutu sabit tutulur. Yeni veri eklendikçe, en eski veri eğitim setinden çıkarılır.

* **Kullanım Alanı:** Veri yapısı zamanla değişiyorsa (rejim değişikliği) ve modelin sadece yakın geçmişe odaklanması isteniyorsa kullanılır.
  
```text
Adım 1: [Train: Yıl 2-3] -> [Test: Yıl 4]
Adım 2:       [Train: Yıl 3-4] -> [Test: Yıl 5]
Adım 3:             [Train: Yıl 4-5] -> [Test: Yıl 6]
```

### 🆚 Karşılaştırma: Rolling vs. Expanding Window

Zaman serisi modellerinde doğrulama (validation) yaparken veri setinin nasıl bölündüğü modelin başarısını doğrudan etkiler. Aşağıda iki ana yöntemin mekanizması ve karşılaştırmalı analizi yer almaktadır.

#### 🎨 Görsel Anlatım (Visual Explanation)

**1. Expanding Window (Genişleyen Pencere)**
Veri seti kümülatif olarak büyür. Başlangıç noktası sabittir.
```text
Adım 1: | Train (Yıl 1) | -> Test (Yıl 2)
Adım 2: | Train (Yıl 1 + 2)      | -> Test (Yıl 3)
Adım 3: | Train (Yıl 1 + 2 + 3)           | -> Test (Yıl 4)



# 📊 Model Evaluation Metrics & Residual Analysis for Time-Series

Zaman serisi tahminlemesinde (Time-Series Forecasting) model başarısını ölçmek, sadece "doğru tahmini" bulmak değil, hatanın karakterini ve iş problemine etkisini anlamaktır. Tek bir metrik asla resmin tamamını göstermez.

Bu döküman, tahmin modellerini değerlendirirken kullanılan **Performans Metriklerini**, **Hata Analizi Yöntemlerini** ve **Karar Destek Tablolarını** teknik bir derinlikle ele alır.

---

## 1. Temel Hata Metrikleri (Scale-Dependent)
Bu metrikler verinin ölçeğine bağlıdır. Yani, elma satışlarını (binlerce) ve araba satışlarını (onlarca) aynı metrik değeriyle kıyaslayamazsınız.

### 📉 MAE (Mean Absolute Error)
Hataların mutlak değerlerinin ortalamasıdır. 
* **Özellik:** Tüm hatalara eşit ağırlık verir.
* **Kullanım:** Modelin "ortalama kaç birim saptığını" en saf haliyle anlatır.
$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

### ⚡ RMSE (Root Mean Squared Error)
Hataların karesinin ortalamasının kareköküdür.
* **Özellik:** Hataların karesini aldığı için **büyük hataları (outliers)** küçük hatalara göre çok daha ağır cezalandırır.
* **Kullanım:** Büyük bir hata yapmanın maliyetinin çok yüksek olduğu durumlarda (örneğin enerji şebekesi yük tahmini) tercih edilir.
$$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

> **Uzman Notu (MAE vs RMSE):** Eğer RMSE değeri MAE değerinden çok büyükse, modeliniz bazı örneklerde çok büyük hatalar yapıyor (outlier üretiyor) demektir.

---

## 2. Yüzdesel ve Ölçekten Bağımsız Metrikler (Scale-Independent)
Farklı ölçekteki (yüksek hacimli vs. düşük hacimli) serileri karşılaştırmak için kullanılır.

### 📊 MAPE (Mean Absolute Percentage Error)
Hataları yüzdesel olarak ifade eder. İş birimlerinin (Business Stakeholders) en sevdiği metriktir.
* **Dezavantaj 1:** Gerçek değer ($y_i$) 0 olduğunda tanımsızdır (sonsuza gider).
* **Dezavantaj 2 (Asimetri):** Gereğinden az tahmin etmeyi (under-forecast), fazla tahmin etmeye (over-forecast) göre daha fazla cezalandırır.
$$MAPE = \frac{100\%}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|$$

### ⚖️ sMAPE (Symmetric MAPE)
MAPE'nin asimetrik yapısını ve 0'a bölünme sorununu (kısmen) düzeltmek için geliştirilmiştir. Değerler %0 ile %200 arasında değişir.
$$sMAPE = \frac{100\%}{n} \sum_{i=1}^{n} \frac{|y_i - \hat{y}_i|}{(|y_i| + |\hat{y}_i|)/2}$$

### 🌟 MASE (Mean Absolute Scaled Error) - Altın Standart
Modelin hatasını, "Naive Model"in (bir önceki değeri tahmin olarak kabul eden saf model: $\hat{y}_t = y_{t-1}$) hatasına oranlar.
* **MASE < 1:** Modeliniz, "dünü bugüne kopyalayan" saf modelden daha zeki.
* **MASE > 1:** Modeliniz başarısız, Naive yaklaşım daha iyi sonuç veriyor.
* **Avantajı:** Mevsimsellik içeren verilerde ve 0 değerlerinde oldukça stabildir.
$$MASE = \frac{MAE_{model}}{MAE_{naive}}$$

---

## 3. İleri Seviye ve İş Odaklı Metrikler

### ⚖️ WMAPE (Weighted MAPE)
Standart MAPE, hacmi 1 olan ürünle hacmi 1.000.000 olan ürünün hatasına eşit davranır. Perakende ve Tedarik Zinciri yönetiminde bu istenmez. WMAPE, hatayı hacme göre ağırlıklandırır.
$$WMAPE = \frac{\sum_{i=1}^{n} |y_i - \hat{y}_i|}{\sum_{i=1}^{n} |y_i|}$$

---

## 4. Residual (Artık) Analizi: Modeliniz Bilgiyi Tüketti mi?

İyi bir zaman serisi modelinde, hatalar (residuals) **White Noise (Beyaz Gürültü)** olmalıdır. Eğer hatalarda bir "desen" kaldıysa, model verideki sinyali tam yakalayamamış demektir.

### ✅ White Noise Kriterleri
1.  **Sıfır Ortalamalı:** $E[e_t] = 0$. Hatalar 0 etrafında rastgele dağılmalı. (Bias olmamalı).
2.  **İlişkisiz (Uncorrelated):** Hatalar geçmiş hatalarla korele olmamalıdır. (ACF grafiği temiz çıkmalı).
3.  **Sabit Varyans (Homoscedasticity):** Hataların boyutu zamanla artmamalı veya azalmamalıdır.
4.  **Normal Dağılım:** Hatalar Çan Eğrisi (Gaussian) şeklinde dağılmalıdır (Güven aralıklarının doğruluğu için şarttır).

### 🧪 İstatistiksel Test: Ljung-Box Testi
Sadece grafiğe bakmak yetmez. Otokorelasyonun olup olmadığını istatistiksel olarak test etmeliyiz.
* **H0 (Null Hypothesis):** Veri rastgele dağılmıştır (White Noise'dur).
* **p-value < 0.05:** H0 reddedilir. Hatalarda otokorelasyon var -> **Model Yetersiz.**
* **p-value > 0.05:** H0 reddedilemez. Hatalar rastgele -> **Model Başarılı.**

---

## 5. Karşılaştırmalı Karar Tablosu: Hangi Yöntem Ne Zaman?

| Senaryo | Önerilen Metrik | Neden? |
| :--- | :--- | :--- |
| **Büyük hatalar felaketse** (Örn: Enerji şebekesi çökmesi) | **RMSE** | Karesini aldığı için outlier'ları çok sert cezalandırır. |
| **Yorumlanabilirlik önemliyse** (Yönetim sunumları) | **MAE / MAPE** | "Ortalama X adet yanılıyoruz" demek kolaydır. |
| **Farklı hacimli binlerce ürün varsa** (Perakende) | **WMAPE** | Çok satan ürünlerin başarısı, az satanlardan daha önemlidir. |
| **Veride çok fazla 0 varsa** (Intermittent Demand) | **MASE** | MAPE sonsuza gider, MASE stabildir. |
| **Modelin ne kadar "zeki" olduğunu ölçmek için** | **MASE** | Basit bir kurala (Naive) göre ne kadar katma değer sağladığını gösterir. |

---

> **Pro Tip:** Asla tek bir metriğe güvenmeyin. Genellikle **RMSE** (model optimizasyonu için) ve **MAPE/WMAPE** (iş birimlerine raporlama için) birlikte kullanılır. Residual analizi ise modelin güvenilirliği ("Canlıya alınır mı?") sorusunun cevabıdır.
