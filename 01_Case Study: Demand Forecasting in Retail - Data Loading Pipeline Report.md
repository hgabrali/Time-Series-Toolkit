


# PART 1: 🚂 Loading the Train Data: Step-by-Step Process
*(Eğitim Verisini Yükleme: Adım Adım Süreç)*

Önceki aşamada çoğu dosya başarıyla yüklenmişti, ancak mağaza satışlarının zaman serisi verilerini (*time-series data*) içeren `train.csv` dosyası hariç. Bu dosya oldukça büyüktür ve standart yöntemlerle okunması zordur.

Bu bölümde, büyük veri dosyalarını yönetmek için **"Chunking"** (Parçalara Ayırma) ve **"Sampling"** (Örnekleme) stratejilerini uyguladık. İşte adım adım uygulanan işlemler:

---

### 1. Install the `gdown` Library
*(gdown Kütüphanesinin Kurulumu)*
* **İşlem (Action):** `gdown` kütüphanesini ortamımıza kurduk.
* **Neden? (Why?):** Büyük dosyamız Google Drive üzerinde barınmaktadır. Standart `requests` kütüphanesi yerine, büyük dosyaları indirme (*large file download*) konusunda daha yetenekli ve stabil olan `gdown` aracını tercih ettik.

### 2. Download the `train.csv` File
*(train.csv Dosyasının İndirilmesi)*
* **İşlem (Action):** Dosyayı indirdik ve çalışma ortamımıza `train.csv` adıyla kaydettik.
* **Neden? (Why?):** Veriyi işleyebilmek için yerel ortama (*local environment*) veya Colab diskine alınması gerekiyordu.

### 3. Select Stores from the "Pichincha" Region
*(Pichincha Bölgesinden Mağazaların Seçimi)*
* **İşlem (Action):** Analiz kapsamımızı daraltmak için mağaza listesini filtreledik ve sadece "Pichincha" bölgesindeki mağazaları seçtik.
* **Neden? (Why?):** Tüm veri yerine belirli bir bölgeye odaklanarak analizi daha yönetilebilir hale getirdik (*region filtering*).

### 4. Read the Data in Chunks and Filter by Store
*(Veriyi Parçalar Halinde Okuma ve Mağazaya Göre Filtreleme)*
* **İşlem (Action):** `train.csv` dosyası çok büyük olduğu için tek seferde okumak yerine, 1 milyon satırlık **parçalar halinde** (*chunks*) okuduk. Her parça okunduğunda, sadece yukarıda seçtiğimiz "Pichincha" mağazalarına ait satırları tuttuk.
* **Neden? (Why?):** Bellek yönetimi (*memory management*) için kritiktir. Tüm dosyayı RAM'e yüklemek yerine, parça parça işleyip gereksiz veriyi anında elemek (*on-the-fly filtering*) sistemin çökmesini engeller.

### 5. Combine the Chunks
*(Parçaların Birleştirilmesi)*
* **İşlem (Action):** Filtrelenmiş parçaları (*filtered chunks*) tek bir DataFrame çatısı altında birleştirdik (*concatenation*).
* **Neden? (Why?):** Analiz ve modelleme aşamasında bütüncül bir veri seti üzerinde çalışabilmek için parçaları tekrar bir araya getirmemiz gerekiyordu.

### 6. Sample 2 Million Rows
*(2 Milyon Satırın Örneklenmesi)*
* **İşlem (Action):** Filtrelenmiş veriden rastgele 2 milyon satır seçtik (*random sampling*).
* **Neden? (Why?):** Eğitim amaçlı (*educational sake*) hesaplamaları hızlandırmak için. Gerçek dünyada tüm veri kullanılabilir ancak öğrenme sürecinde işlem süresini kısaltmak (*computation speed*) için veri boyutu optimize edildi.

### 7. Clean Up
*(Temizlik)*
* **İşlem (Action):** Bellekte yer kaplayan geçici parça listelerini sildik.
* **Neden? (Why?):** Python'ın bellek yönetimini rahatlatmak (*garbage collection*) ve RAM'i verimli kullanmak için gereksiz değişkenler temizlendi.

### 8. Verification
*(Doğrulama)*
* **İşlem (Action):** Oluşturulan DataFrame'lerin ilk satılarını (`head`) yazdırarak kontrol ettik.
* **Neden? (Why?):** Verinin düzgün yüklenip yüklenmediğini ve formatın beklediğimiz gibi olup olmadığını teyit etmek (*sanity check*).

---

# PART 2: Veri Setini Anlamak (Understanding the Dataset)

İndirdiğimiz **Corporación Favorita Grocery Sales Forecasting** veri setine yakından bakalım.

### 📂 Girdi Verileri (Input Data)

Çalışacağımız birden fazla `csv` dosyası bulunmaktadır. Bunlar şunları içerir:

#### 1. `train.csv`
* Hedef değişken (*target*) olan `unit_sales` (birim satışlar) verisini `date` (tarih) bazında içerir. Ayrıca `store_nbr` (mağaza no), `item_nbr` (ürün no) ve satırları etiketlemek için benzersiz bir `id` sütunu bulunur.
* Hedef `unit_sales`, tamsayı (*integer*) (örn: bir paket cips) veya ondalıklı sayı (*float*) (örn: 1.5 kg peynir) olabilir.
* `unit_sales` değerinin negatif olması, o ürünün iade edildiğini (*returns*) gösterir.
* `onpromotion` sütunu, o `item_nbr`'ın belirtilen `date` ve `store_nbr` için promosyonda olup olmadığını belirtir.
* Bu dosyadaki `onpromotion` değerlerinin yaklaşık %16'sı `NaN` (eksik veri)'dır.

> ☝🏼 **NOT (NOTE):** Eğitim verileri (*training data*), bir mağaza/tarih kombinasyonu için sıfır `unit_sales` olan ürünlere ait satırları içermez. Ürünün o tarihte mağazada stokta olup olmadığına (*in stock*) dair bir bilgi yoktur ve ekiplerin bu durumu ele almanın en iyi yoluna karar vermesi gerekecektir. Ayrıca, eğitim verilerinde görülen ancak test verilerinde (*test data*) görülmeyen az sayıda ürün vardır.

#### 2. `stores.csv`
* Mağaza üst verilerini (*metadata*) içerir: `city` (şehir), `state` (eyalet), `type` (tür) ve `cluster` (küme).
* `cluster`, benzer mağazaların bir gruplandırmasıdır.

#### 3. `items.csv`
* Ürün üst verilerini (*metadata*) içerir: `family` (aile/kategori), `class` (sınıf) ve `perishable` (bozulabilir).

> ☝🏼 **NOT (NOTE):** `perishable` (bozulabilir) olarak işaretlenen ürünlerin skor ağırlığı (*score weight*) **1.25**'tir; diğerlerinin ağırlığı ise **1.0**'dır.

#### 4. `transactions.csv`
* Her bir `date` ve `store_nbr` kombinasyonu için satış işlemlerinin (*sales transactions*) sayısını içerir. Sadece eğitim verisi zaman aralığı (*training data timeframe*) için dahildir.

#### 5. `oil.csv`
* Günlük petrol fiyatı (*Daily oil price*). Hem eğitim (*train*) hem de test (*test*) verisi zaman aralığındaki değerleri içerir. Ekvador petrol bağımlı (*oil-dependent*) bir ülkedir ve ekonomik sağlığı petrol fiyatlarındaki şoklara karşı oldukça kırılgandır (*highly vulnerable*).

#### 6. `holidays_events.csv`
* Tatiller ve Etkinlikler ile bunlara ait üst veriler.
* `Additional` (Ek) tatiller, normal bir takvim tatiline eklenen günlerdir; örneğin Noel civarında tipik olarak gerçekleşen durumlar gibi (Noel Arifesini tatil yapmak).

> ☝🏼 **NOT (NOTE):** `transferred` (aktarılan) sütununa özellikle dikkat edin. `transferred` olan bir tatil resmi olarak o takvim gününe denk gelir, ancak hükümet tarafından başka bir tarihe taşınmıştır. Bir `transferred` gün, tatilden ziyade normal bir gün gibidir. Aslında kutlandığı günü bulmak için, `type` sütununun `Transfer` olduğu ilgili satıra bakın.
>
> *Örneğin:* `Independencia de Guayaquil` tatili 2012-10-09'dan 2012-10-12'ye aktarılmıştır (*transferred*), yani 2012-10-12'de kutlanmıştır.
>
> `type` değeri `Bridge` (Köprü) olan günler, bir tatile eklenen ekstra günlerdir (örn: tatili uzun bir hafta sonuna uzatmak için). Bunlar genellikle, `Bridge` gününü telafi etmek (*payback*) amacıyla normalde çalışma günü olmayan (örn: Cumartesi) bir günün çalışıldığı `Work Day` (Çalışma Günü) tipi ile telafi edilir.

---

### 📝 Ek Notlar (Additional Notes)

1.  **Maaşlar (Wages):** Kamu sektöründeki maaşlar iki haftada bir, ayın **15'inde** ve **son gününde** ödenir. Süpermarket satışları bundan etkilenebilir.
2.  **Deprem (Earthquake):** 16 Nisan 2016'da Ekvador'da 7.8 büyüklüğünde bir deprem meydana gelmiştir. İnsanlar su ve diğer temel ihtiyaç ürünlerini bağışlayarak yardım çalışmalarında (*relief efforts*) bir araya gelmiş, bu durum depremden sonraki birkaç hafta boyunca süpermarket satışlarını büyük ölçüde etkilemiştir.

---

# PART 3:🔍 EDA for Time-Series Data
*(Zaman Serisi Verileri için Keşifçi Veri Analizi)*

**EDA (Exploratory Data Analysis)** is a crucial step before applying machine learning models, especially in **time-series forecasting**. We will focus on:
*(EDA, özellikle zaman serisi tahmininde makine öğrenimi modellerini uygulamadan önce çok önemli bir adımdır. Şunlara odaklanacağız:)*

* 🏗️ **Understanding the Structure:** Understanding the structure of the dataset.
    *(Veri setinin yapısını anlamak.)*
* 🧩 **Handling Missing Data:** Handling missing data effectively.
    *(Eksik verileri etkili bir şekilde ele almak.)*
* 📈 **Visualizing Trends:** Visualizing sales trends.
    *(Satış trendlerini görselleştirmek.)*
* 🔗 **Investigating Relationships:** Investigating relationships among the various features.
    *(Çeşitli özellikler arasındaki ilişkileri araştırmak.)*

---

# PART 4:





## 🛤️ Workflow Steps


These are the steps we will follow:

### **Step 1: Checking for Missing Data**
*(Adım 1: Eksik Veri Kontrolü)*
Identify gaps in the data (e.g., missing sales records, null values in promotion columns).
*(Verideki boşlukları belirleme [örn. eksik satış kayıtları, promosyon sütunlarındaki boş değerler].)*

### **Step 2: Handling Outliers**
*(Adım 2: Aykırı Değerlerin Ele Alınması)*
Detect and manage extreme values (e.g., negative sales indicating returns, or massive spikes due to earthquakes) that could skew the model.
*(Modeli saptırabilecek aşırı değerleri [örn. iadeleri gösteren negatif satışlar veya depremlerden kaynaklanan büyük sıçramalar] tespit etme ve yönetme.)*

### **Step 3: Fill Missing Dates with Zero Sales**
*(Adım 3: Eksik Tarihleri Sıfır Satışla Doldurma)*
Time-series models require a continuous timeline. Missing rows usually imply no sales occurred, so we impute them with 0.
*(Zaman serisi modelleri sürekli bir zaman çizelgesi gerektirir. Eksik satırlar genellikle satış olmadığını ima eder, bu yüzden bunları 0 ile doldururuz.)*

### **Step 4: Feature Engineering: Turning a Date into Useful Signals**
*(Adım 4: Özellik Mühendisliği: Bir Tarihi Faydalı Sinyallere Dönüştürme)*
Extract components like "Day of Week", "Month", "Year", and "Is Weekend" from the raw date object to help the model learn cyclical patterns.
*(Modelin döngüsel kalıpları öğrenmesine yardımcı olmak için ham tarih nesnesinden "Haftanın Günü", "Ay", "Yıl" ve "Hafta Sonu mu" gibi bileşenleri çıkarma.)*

### **Step 5: Visualizing Time-Series Data**
*(Adım 5: Zaman Serisi Verilerini Görselleştirme)*
Plot sales over time to spot trends, seasonality, and potential structural breaks.
*(Trendleri, mevsimselliği ve potansiyel yapısal kırılmaları tespit etmek için zaman içindeki satışları grafiğe dökme.)*

### **Step 6: Examining the Impact of Holidays**
*(Adım 6: Tatillerin Etkisini İnceleme)*
Analyze how specific events (National holidays, transferred days, bridges) correlate with sales spikes or drops.
*(Belirli olayların [Ulusal tatiller, aktarılan günler, köprüler] satış artışları veya düşüşleriyle nasıl ilişkili olduğunu analiz etme.)*

### **Step 7: Analyzing Perishable Items**
*(Adım 7: Bozulabilir Ürünleri Analiz Etme)*
Investigate if perishable goods (weighted higher in scoring) show different sales patterns compared to non-perishables.
*(Bozulabilir malların [puanlamada ağırlığı daha yüksek olan], bozulmayanlara kıyasla farklı satış modelleri gösterip göstermediğini araştırma.)*

---
---

# 🕰️ Comprehensive Guide to Time-Series Modelling
*(Zaman Serisi Modelleme Kapsamlı Rehberi)*

Bu doküman, Zaman Serisi Tahmini (*Time-Series Forecasting*) projelerinde dikkate alınması gereken temel bileşenleri, analiz yöntemlerini ve modelleme stratejilerini karşılaştırmalı bir şekilde sunar.

---

## 1. 🏗️ Preprocessing & Structural Analysis
*(Ön İşleme ve Yapısal Analiz)*

Modellemeye geçmeden önce zaman serisinin karakteristiğini anlamak ve veriyi matematiksel olarak modele hazırlamak zorunludur.

| Bileşen (Component) | Açıklama (Description) | Teknikler & Testler (Techniques & Tests) |
| :--- | :--- | :--- |
| **Stationarity**<br>*(Durgunluk)* | Serinin istatistiksel özelliklerinin (ortalama, varyans) zamanla değişmemesi durumudur. Çoğu klasik model (ARIMA vb.) durgunluk gerektirir. | • **ADF Test (Augmented Dickey-Fuller):** Birim kök (*unit root*) varlığını test eder.<br>• **KPSS Test:** Serinin trend durağan olup olmadığını test eder.<br>• **Differencing (Fark Alma):** Durgunlaştırmak için $y_t - y_{t-1}$ işlemi. |
| **Seasonality & Trend**<br>*(Mevsimsellik ve Trend)* | Verideki uzun vadeli artış/azalış (Trend) ve belirli periyotlarla tekrar eden kalıplar (Mevsimsellik). | • **Decomposition (Ayrıştırma):** Additive (Toplamsal) veya Multiplicative (Çarpımsal) ayrıştırma.<br>• **STL Decomposition:** Mevsimsellik ve Trendi Loess kullanarak ayırma. |
| **Autocorrelation**<br>*(Otokorelasyon)* | Bir gözlemin geçmiş gözlemlerle olan ilişkisi. | • **ACF (Autocorrelation Function):** Doğrudan ve dolaylı geçmiş ilişkiler.<br>• **PACF (Partial Autocorrelation Function):** Ara gecikmelerin etkisini kaldırarak saf ilişki. |
| **Missing Values**<br>*(Eksik Değerler)* | Zaman serisinde boşluklar kabul edilemez. | • **Forward/Backward Fill:** Önceki/sonraki değerle doldurma.<br>• **Interpolation:** Lineer veya zamana bağlı enterpolasyon.<br>• **Imputation:** Ortalama veya 0 ile doldurma (satış yoksa). |

---

## 2. 🛠️ Feature Engineering Strategies
*(Özellik Mühendisliği Stratejileri)*

Zaman serisi verisini Makine Öğrenmesi (*Machine Learning*) modellerine (örn. XGBoost, Random Forest) sokabilmek için "zamanı" özelliklere dönüştürmek gerekir.

| Özellik Türü (Feature Type) | Yöntem (Method) | Neden Kullanılır? (Why Use It?) |
| :--- | :--- | :--- |
| **Lag Features**<br>*(Gecikmeli Özellikler)* | $t-1, t-7, t-30$ gibi geçmiş değerleri yeni sütun olarak eklemek. | Modelin otokorelasyonu (*autocorrelation*) öğrenmesini sağlar. "Bugünün satışı dünkü satışa benzer" mantığı. |
| **Rolling Window Statistics**<br>*(Hareketli Pencere İstatistikleri)* | Son 7 günün ortalaması, standart sapması, min/max değerleri. | Gürültüyü azaltır (*smoothing*) ve trend/momentum bilgisini yakalar. |
| **Date-Time Components**<br>*(Tarih-Zaman Bileşenleri)* | Ay, Yıl, Haftanın Günü, Yılın Günü, Hafta Sonu mu? | Modelin döngüsel (*cyclical*) ve takvimsel etkileri öğrenmesini sağlar. |
| **Cyclical Encoding**<br>*(Döngüsel Kodlama)* | Ay ve gün bilgisini Sinüs/Kosinüs fonksiyonlarına dönüştürmek. | Aralık (12) ile Ocak (1) ayının birbirine yakın olduğunu modele matematiksel olarak anlatır. |
| **Exogenous Variables**<br>*(Dışsal Değişkenler)* | Tatiller, Petrol Fiyatları, Hava Durumu, Promosyonlar. | Tahmin gücünü artıran dış faktörleri dahil eder. |

---

## 3. 🤖 Modelling Approaches: Comparative Table
*(Modelleme Yaklaşımları: Karşılaştırmalı Tablo)*

Hangi modelin seçileceği veri boyutuna, karmaşıklığına ve iş ihtiyacına bağlıdır.

| Yaklaşım (Approach) | Modeller (Models) | Avantajlar (Pros) | Dezavantajlar (Cons) | En İyi Kullanım (Best Use Case) |
| :--- | :--- | :--- | :--- | :--- |
| **Statistical (Classical)**<br>*(İstatistiksel/Klasik)* | • ARIMA / SARIMA<br>• ETS (Exponential Smoothing)<br>• Holt-Winters | • Az veriyle iyi çalışır.<br>• Yorumlanabilirliği (*interpretability*) yüksektir.<br>• İstatistiksel özellikleri (trend, mevsimsellik) doğrudan modeller. | • Çoklu dışsal değişkenleri (*multivariate*) yönetmek zordur.<br>• Doğrusal olmayan (*non-linear*) ilişkileri yakalayamaz.<br>• Büyük veride yavaştır. | Tek değişkenli, kısa vadeli, net trendi olan veriler. |
| **Machine Learning (Tree-Based)**<br>*(Makine Öğrenmesi / Ağaç Tabanlı)* | • XGBoost<br>• LightGBM<br>• Random Forest<br>• CatBoost | • Doğrusal olmayan karmaşık ilişkileri yakalar.<br>• Dışsal değişkenleri (promosyon, tatil) mükemmel yönetir.<br>• Büyük veride ölçeklenebilir (*scalable*). | • Trendi "extrapolate" edemez (görmediği yüksek değerleri tahmin edemez).<br>• Çok fazla Özellik Mühendisliği (*Feature Engineering*) gerektirir. | Karmaşık perakende satışları, çoklu değişkenler, büyük veri setleri. (Bizim projemiz için ideal). |
| **Deep Learning**<br>*(Derin Öğrenme)* | • LSTM / GRU (RNNs)<br>• CNN (1D)<br>• Transformers (TFT, Temporal Fusion) | • Sıralı bağımlılıkları (*sequential dependencies*) ve uzun vadeli hafızayı yönetir.<br>• Ham veriden özellik çıkarabilir.<br>• Çok karmaşık örüntüleri çözer. | • Çok büyük veri ve işlem gücü (*GPU*) gerektirir.<br>• "Black Box" (Kara Kutu) doğası vardır, yorumlaması zordur.<br>• Eğitim süresi uzundur. | Devasa veri setleri, web trafiği, finansal yüksek frekanslı işlemler. |
| **Modern Hybrid / Automated**<br>*(Modern Hibrit / Otomatik)* | • Prophet (Meta)<br>• NeuralProphet<br>• Auto-ARIMA | • Kullanımı kolaydır (*Out-of-the-box*).<br>• Tatilleri ve değişim noktalarını (*changepoints*) otomatik yönetir. | • Her zaman en yüksek doğruluğu vermeyebilir.<br>• Özelleştirme (*customization*) imkanları bazen sınırlıdır. | Hızlı prototipleme, iş zekası raporlaması, orta ölçekli veriler. |

---

## 4. 📉 Validation & Evaluation Metrics
*(Doğrulama ve Değerlendirme Metrikleri)*

Zaman serilerinde rastgele bölme (*random split*) yapılamaz; "Geleceği kullanarak geçmişi tahmin etmek" (*Data Leakage*) hatasına düşmemek gerekir.

### A. Validation Strategy (Doğrulama Stratejisi)

* **Time Series Split:** Veriyi zamana göre sıralı tutarak eğitim seti sürekli büyürken test seti ileri kayar.
* **Sliding Window (Walk-Forward):** Sabit boyutlu bir pencere zaman içinde kaydırılır.
* **Strict Cut-off:** Örn. 2016 sonuna kadar Train, 2017 başı Validation, 2017 sonu Test.

### B. Key Metrics (Temel Metrikler)

| Metrik (Metric) | Formül Mantığı (Logic) | Artılar/Eksiler (Pros/Cons) |
| :--- | :--- | :--- |
| **MAE**<br>*(Mean Absolute Error)* | Hataların mutlak değerlerinin ortalaması. | • Yorumlaması kolaydır (Satış adedi cinsinden hata).<br>• Aykırı değerlere (*outliers*) karşı daha dirençlidir. |
| **RMSE**<br>*(Root Mean Squared Error)* | Hataların karesinin ortalamasının karekökü. | • Büyük hataları daha çok cezalandırır (*penalizes large errors*).<br>• Aykırı değerlere karşı hassastır. |
| **MAPE**<br>*(Mean Absolute Percentage Error)* | Hatanın gerçek değere oranının yüzdesi. | • Ölçekten bağımsızdır (*scale-independent*), % olarak ifade edilir.<br>• Gerçek değer 0 ise tanımsız olur (Sonsuz hata). |
| **WMAPE**<br>*(Weighted MAPE)* | Ağırlıklı ortalama yüzde hatası. | • Hacme göre ağırlıklandırır.<br>• Düşük satışlı ürünlerdeki yüksek yüzdesel hataların genel skoru bozmasını engeller. |
| **RMSLE**<br>*(Root Mean Squared Logarithmic Error)* | Logaritmik ölçekte RMSE. | • Tahmin edilen değerin gerçek değere "oranı" ile ilgilenir.<br>• Düşük tahmin etmeyi (*under-prediction*) yüksek tahmin etmeye göre daha az cezalandırır (veya tam tersi duruma göre ayarlanabilir). |

---

## 🚀 Summary Checklist for a Successful Project
*(Başarılı Bir Proje İçin Özet Kontrol Listesi)*

1.  [ ] **EDA:** Veriyi görselleştir, mevsimselliği ve trendi anla.
2.  [ ] **Preprocessing:** Eksik verileri ve anomaliyi yönet.
3.  [ ] **Feature Engineering:** Lag, Rolling, Date özelliklerini üret.
4.  [ ] **Baseline Model:** Basit bir model (örn. Naive Forecast veya ortalama) kurarak referans noktası belirle.
5.  [ ] **Model Selection:** Veriye uygun algoritmayı (örn. XGBoost) seç.
6.  [ ] **Validation:** Zamana duyarlı (*time-aware*) bir doğrulama seti kullan.
7.  [ ] **Evaluation:** İş hedefine uygun metriği (örn. Stok yönetimi için RMSE) seç ve yorumla.
