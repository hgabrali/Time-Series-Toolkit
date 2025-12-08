# 📈 Time Series (Zaman Serisi) Analizi

Time Series (Zaman Serisi) analizi veri biliminin en "nazlı" ama en güçlü alanlarından biridir. Standart veri setlerinden (örneğin bir ev fiyatı tahminindeki tablolardan) çok farklı matematiksel ve istatistiksel kuralları vardır.

---

## 🧠 Bölüm 1: Time Series Modelling

Verilen metin, zaman serilerinin temel felsefesini ve bir proje döngüsünü (pipeline) anlatıyor. Bir uzman olarak satır aralarını şöyle okumalıyız:

### 1. Zaman Serisini "Benzersiz" Yapan Nedir? (The Unique Nature)

Standart Makine Öğrenmesi (Machine Learning) algoritmaları genellikle verilerin **I.I.D.** (Independent and Identically Distributed) olduğunu varsayar. Yani satırların birbirinden bağımsız olduğunu düşünür.

> **Time Series Farkı:** Burada veriler **birbirine bağımlıdır**. Bugünün hisse senedi fiyatı, dünün fiyatıyla doğrudan ilişkilidir (Auto-correlation).

* **Temporal Dependencies (Zaman Bağımlılıkları):** Geçmiş, geleceği şekillendirir. Modellerimiz bu "hafızayı" korumak zorundadır.

#### Temel Bileşenler:

* **Trend:** Verinin uzun vadede yukarı veya aşağı yönlü hareketi.
* **Seasonality (Mevsimsellik):** Belirli periyotlarda (günlük, haftalık, yıllık) tekrarlayan desenler. (Örn: Dondurma satışlarının yazın artması).
* **Stationarity (Durağanlık):** Çoğu klasik model (ARIMA gibi), verinin ortalamasının ve varyansının zamanla değişmemesini ister. Veri durağan değilse, onu durağan hale getirmek (Differencing) gerekir.

<img width="892" height="420" alt="image" src="https://github.com/user-attachments/assets/0ba18a14-a41c-4f8d-998f-ebd30da650df" />

---

### 2. Endüstriyel Kullanım Alanları ve Yöntemler (Derinlemesine Bakış)

Metinde geçen sektörlerde, zaman serisi şu kritik soruları çözer:

* **Finance (Finans):** Sadece fiyat tahmini değil, "Volatilite (Oynaklık)" tahmini yapılır.
    * *Yöntem:* `GARCH` modelleri, `LSTM` ağları.
* **Energy (Enerji):** Şebeke dengesi için hayati önem taşır. Üretilen elektriğin anında tüketilmesi gerekir.
    * *Yöntem:* `SARIMA` (Mevsimsellik güçlüdür), `Prophet`.
* **Healthcare (Sağlık):** EKG sinyalleri aslında milisaniyelik zaman serileridir. Anormallik tespiti (Anomaly Detection) burada hayat kurtarır.
* **Cybersecurity:** Ağ trafiğindeki ani "spike"lar (sıçramalar) DDoS saldırısı olabilir.
    * *Yöntem:* `Isolation Forest`, `Autoencoders`.


## 3. Karşılaştırmalı Yöntemler Tablosu (Model Comparison Matrix)

Metinde Sprint 2'de geçen modellerin teknik karşılaştırmasını senin için hazırladım:

| Model Türü | Örnekler | Ne Zaman Kullanılır? | Avantajı | Dezavantajı |
| :--- | :--- | :--- | :--- | :--- |
| **Klasik İstatistiksel** | **ARIMA, SARIMA** | Veri seti küçükse, mevsimsellik netse ve açıklanabilirlik (neden bu sonucu verdi?) önemliyse. | Hızlıdır, az veri ile çalışır, matematiksel temeli sağlamdır. | Karmaşık, doğrusal olmayan (non-linear) ilişkileri yakalayamaz. |
| **Makine Öğrenmesi (ML)** | **XGBoost, LightGBM, Random Forest** | Elimizde sadece zaman değil, dış faktörler (hava durumu, tatil günleri vb.) de varsa. | Çok güçlüdür, karmaşık ilişkileri çözer, şu an endüstri standardıdır. | Gelecekteki trendi (extrapolation) yakalamakta zorlanır (verinin aralığı dışına çıkamaz). |
| **Derin Öğrenme (DL)** | **RNN, LSTM, Transformers** | Veri seti devasaysa (Big Data) ve çok uzun vadeli karmaşık desenler varsa. | Uzun vadeli bağımlılıkları (Long-term dependencies) harika yakalar. | Eğitmesi çok uzun sürer, çok fazla veriye ihtiyaç duyar (Data hungry). |


# 📊 Zaman Serisi Verilerine Giriş (Introduction to Time-Series Data)


<img width="1760" height="629" alt="image" src="https://github.com/user-attachments/assets/5d20b93f-4e5a-4e60-afa2-a0b918313229" />


**Zaman serisi verileri**, birbirini takip eden zaman noktalarında toplanmış bir gözlem dizisidir. Günlük hisse senedi fiyatları, sıcaklık ölçümleri, satış rakamları ve web sitesi trafiği zaman serisi verilerine örnektir.

## 1. Tanım (Definition)

Zaman serisi, basitçe, olayların gerçekleşme sırasına göre zaman içinde kaydedilen veri noktaları dizisidir. Temel fikir şudur: Bu değerlere zaman içinde değiştikçe bakarak, gözlemlediğimiz sistemin desenlerini, trendlerini ve genel davranışını anlamaya başlayabiliriz.

Bu ölçümler, neyin takip edildiğine bağlı olarak düzenli aralıklarla (örneğin her saat, gün veya ay) veya düzensiz olarak toplanabilir.

> **💡 Uzman Notu (Technical Insight):**
> Veri biliminde zaman serilerini diğer verilerden ayıran en büyük özellik **"Bağımsızlık" (Independence)** varsayımının ihlal edilmesidir. Standart veri setlerinde satırlar birbirinden bağımsızdır; ancak zaman serilerinde bugünün değeri, dünün değerine matematiksel olarak bağımlıdır (**Autocorrelation**).

### Kullanım Alanları (Where it’s used)
Zaman serisi verilerini, borsa ve satışlardan arkeoloji ve tıbba kadar, farklı fenomenlerin zaman içinde nasıl evrildiğini yakalayan çeşitli alanlarda bulabilirsiniz.

---

## 2. Zaman Serisi Görev Türleri (Types of Time-Series Tasks)

Zaman serisi verileriyle gerçekleştirdiğimiz en yaygın görevler şunlardır:


### 1️⃣ Tahminleme (Forecasting)
**Hedef:** Geçmişi kullanarak önünü görmek. Geçmiş verilere dayanarak gelecek değerleri tahmin etmektir.



* **Senaryo:** Turuncu çizgi, üç yıllık aylık elektrik talebini gösterir; yazın yüksek ve kışın düşük olduğu hafif inişli çıkışlı ritme (mevsimsellik) dikkat edin. Sağdaki kesikli çizgi, bu mevsimsel deseni basitçe tekrarlayan, 12 aylık "saf" (naive) bir tahmindir.
* **Belirsizlik:** Etrafındaki soluk bant, ileriye baktıkça genişler ve ±15 MWh'lik bir belirsizlik bölgesini işaretler.
* **İş Dünyası Değeri:** Uygulamada bir elektrik şirketi modeli daha da iyileştirecektir, ancak bu basit resim bile iki temel iş sorusunu yanıtlar:
    1.  Gelecek yıl her ay talebin ne olmasını bekliyoruz?
    2.  Üretim planlarımızda ne kadar hareket alanı bırakmalıyız?

Bu ileriye dönük görüşle planlamacılar, yoğun dönem (peak) gelmeden önce bakımı planlayabilir, yakıt sözleşmelerini müzakere edebilir ve yedek kapasiteyi ayarlayabilir.

<img width="1103" height="538" alt="image" src="https://github.com/user-attachments/assets/38fd40c8-de7d-48dc-8b95-3b970391afc7" />

### 2️⃣ Anormallik Tespiti (Anomaly Detection)
**Hedef:** Seri olağandışı bir şey yaptığında bayrak kaldırmak (uyarı vermek).

* **Hızlı Örnek:** Bir çevrimiçi mağaza yoğun bir akşamda aniden sıfır ödeme (check-out) kaydederse, sistem ekibi uyarır; bu sakin bir gece değil, bir ödeme hatası (bug) olabilir.
* **Grafik Analizi:** Grafik tipik bir yoğun akşam desenini gösterir; ta ki 90. dakikaya kadar. Burada ödemeler yaklaşık 15 dakika boyunca sıfıra düşer. Bu düz segment, bir uyarı sisteminin ekibin incelemesi için işaretleyeceği türden bir anomalidir.

> **🛠 Kullanılan Yöntemler:** Isolation Forest, Autoencoders, Z-Score Analysis.

<img width="1071" height="639" alt="image" src="https://github.com/user-attachments/assets/a1bc920d-6cfe-4d7d-9f8c-183a33cb357b" />


### 3️⃣ Sınıflandırma / Bölütleme (Classification / Segmentation)
**Hedef:** Zaman aralıklarını belirli kategorilere ayırmak.

* **Hızlı Örnek:** Bir perakendeci, her satış haftasını **"sezon zirvesi" (peak-season)**, **"promosyon odaklı"** veya **"normal"** olarak etiketler. Böylece pazarlama ekibi kampanyaları doğru dönemlerle eşleştirebilir.
    * **Sezon Zirvesi:** Yıl sonuna yakın o uzun çubuklar sırasında perakendeci, ekstra stok getirerek, mağazaları daha uzun süre açık tutarak ve yoğunluğu daha ağır reklamlarla destekleyerek artışa hazırlanır.
    * **Promosyon Odaklı:** Orta yükseklikteki çubuklar kampanyaları işaret eder; burada odak noktası kısa ömürlü indirimlere, hedeflenmiş e-postalara ve trafiği birkaç günlüğüne artırmak için tasarlanmış sosyal medya gönderilerine kayar.
    * **Normal:** Diğer tüm haftalar normal aralıkta yer alır; burada rutin operasyonlar ve temel tahminler, rafları dolu ve personeli sabit tutmak için yeterlidir.

<img width="1122" height="436" alt="image" src="https://github.com/user-attachments/assets/ccfdfa61-714c-4158-8baf-2e0673fb8e9c" />


---

## 🚀 3. Karşılaştırmalı Yöntemler Tablosu (Expert Comparison Matrix)

Bir veri bilimci olarak, hangi göreve hangi silahla (algoritma) saldıracağımızı bilmek gerekir. İşte teknik bir özet:

| Görev Türü (Task) | Amaç | Klasik / İstatistiksel Yöntemler | Modern / ML & DL Yöntemleri | Kullanım Örneği |
| :--- | :--- | :--- | :--- | :--- |
| **Forecasting** (Tahminleme) | Gelecekteki $t+1$ değerini bulmak. | **ARIMA, SARIMA, Holt-Winters** (Durağan verilerde güçlüdür). | **Prophet, XGBoost, LSTM (Long Short-Term Memory)** (Karmaşık ve büyük verilerde liderdir). | Stok yönetimi, Enerji tüketim tahmini, Bütçe planlama. |
| **Anomaly Detection** (Anormallik Tespiti) | Normal desenden sapmaları bulmak. | **Z-Score, Moving Average** (Basit eşik değerleri). | **Isolation Forest, Autoencoders, One-Class SVM** (Bilinmeyen anormallikleri yakalar). | Kredi kartı dolandırıcılığı, Siber saldırı tespiti, Cihaz arızası. |
| **Classification** (Sınıflandırma) | Zaman serisini etiketlemek (Pattern Recognition). | **Nearest Neighbor (1-NN) with DTW** | **Time Series Forest, CNN (Convolutional Neural Nets)** | EKG sinyalinden aritmi tespiti, Müşteri segmentasyonu (Churn analizi). |

---

### 📌 Özet (Key Takeaways)
* **Zaman Serisi** sadece sayıların listesi değildir; **sıra (order)** ve **zaman (time)** bilgisi kritik öneme sahiptir.
* **Trend ve Mevsimsellik**, analizlerin belkemiğidir.
* İş dünyasında sadece "geleceği bilmek" (Forecast) değil, "ters gideni bulmak" (Anomaly) ve "dönemi anlamak" (Classification) da hayati öneme sahiptir.
  
---


# 🔑 Key Characteristics of Time-Series Data (Zaman Serisi Verilerinin Temel Karakteristikleri)

Time-series data is not just a list of numbers; it is a sequence where history matters. Unlike standard tabular data, time series has unique behaviors that require specific modeling strategies.

Let's explore three features that make time-series data special:
1.  **Temporal Order** (Zaman Sırası) – Why the sequence itself carries meaning.
2.  **Autocorrelation** (Otokorelasyon) – How today often echoes yesterday.
3.  **Stationarity** (Durağanlık) – How baselines drift as the world changes.

We will need to keep these three ideas in mind as they will guide every practical step of a time-series project.

* "Data Leakage" (Veri Sızıntısı) ve "Stationarity" (Durağanlık) kavramları, bir modelin üretim ortamında (production) çakılmaması için hayati önem taşır.

## 1. Temporal Order (Zaman Sırası)

<img width="721" height="434" alt="image" src="https://github.com/user-attachments/assets/dff6ebae-aac0-43fe-8803-84e27bd2f16e" />

Unlike tabular data where rows can be shuffled (e.g., predicting house prices based on size), time-series data has a rigid **temporal order** where each observation depends on previous time points.

### 🚨 The "Data Leakage" Trap (Temporal-Order Example)
**Scenario:** You’re asked to predict tomorrow’s online-shop revenue so you can set ad spend today. You have two years of daily data (date, revenue, marketing-budget, weather, etc.).

**The Mistake:** A teammate—used to tabular problems—randomly shuffles the rows, keeps 80% for training and 20% for testing, and hands the split to you.

**What goes wrong?**
> If you shuffle, your model might train on data from "next week" to predict "today." This is called **Look-Ahead Bias**. The model learns the future, achieving falsely high accuracy in testing, but fails in the real world because it can't see the future in production.

### ✅ Take-away & Best Practices
* **Keep the order:** Never shuffle time-series data before splitting.
* **Split Correctly:** Use a "train-past / test-future" split.
* **Validation:** Instead of random K-Fold, use **Rolling-Window** or **Expanding-Window (TimeSeriesSplit)** validation.

**Why this matters:**
Temporal order tells us to use models that respect sequence (e.g., Moving Averages, ARIMA, RNNs/LSTMs) instead of ordinary regression. It keeps the future out of the past.

---


## 2. Autocorrelation (Otokorelasyon)

<img width="741" height="510" alt="image" src="https://github.com/user-attachments/assets/6e0e00e9-dab3-46e1-a0b3-5c32c47c22d1" />

Observations are often correlated with past data points, which makes time-series data different from **i.i.d.** (independent and identically distributed) data seen in traditional regression tasks.

* **Concept:** "Today echoes yesterday."
* **Example:** If you sell a lot of ice cream today (high value), it is likely you will sell a lot tomorrow (high value) if the weather remains similar. This "stickiness" is autocorrelation.

### 🛠 Technical Insight: How to Measure?
In Data Science, we don't just guess autocorrelation; we measure it using:
1.  **ACF (Autocorrelation Function):** Shows correlation of the series with itself at different lags (today vs. yesterday, today vs. last week).
2.  **PACF (Partial Autocorrelation Function):** Shows the direct correlation after removing the effects of intermediate lags.

**Why this matters:**
Counting autocorrelated data without adjusting for it is like polling the same person every hour. Your "sample size" looks big, but the information hasn't grown. Models like **ARIMA** specifically use this feature (The 'AR' part stands for AutoRegressive).

---

## 3. Stationarity (Durağanlık)

<img width="701" height="598" alt="image" src="https://github.com/user-attachments/assets/0990a033-35a3-4f04-b83a-13fe6cd04007" />

<img width="706" height="365" alt="image" src="https://github.com/user-attachments/assets/09ebf7fe-f776-4722-bb10-ffbe9c4d8285" />

A time series is said to be **stationary** if its statistical properties (like **mean** and **variance**) don't change over time.

### What does Stationary Data look like?
* **Constant Average:** The data jitters around a straight line; it doesn't trend up or down.
* **Stable Spread (Variance):** The size of the fluctuations is constant; no "funnel" shape where waves get bigger over time.
* **No Seasonality:** There are no repeating periodic waves.

> **💡 Real-World Example:** Facility managers want stationary temperature readings (approx 20°C). If the mean starts drifting up, it indicates a cooling unit failure.

### 🛠 Technical Insight: Testing & Fixing
Non-stationary data is hard to model because the "rules" keep changing.
* **Test:** We use the **Augmented Dickey-Fuller (ADF)** test.
    * *p-value < 0.05:* Data is Stationary (Good ✅).
    * *p-value > 0.05:* Data is Non-Stationary (Needs work ❌).
* **Fix:** We usually apply **Differencing** ($y_t - y_{t-1}$) or **Log Transformation** to stabilize the mean and variance.

**Why this matters:**
Non-stationary data have a moving baseline. Before modeling with algorithms like ARIMA, we must transform the data to make it stationary.

---

## 📊 Summary: Comparison Matrix (Kavramsal Karşılaştırma)

A cheat sheet for Data Scientists to manage these characteristics.

| Characteristic | What is it? | Why is it a problem? | How to handle/fix it? | Related Models/Tests |
| :--- | :--- | :--- | :--- | :--- |
| **Temporal Order** | Data follows a strict time sequence ($t_1, t_2, t_3...$). | Shuffling destroys the relationship and causes **Data Leakage**. | **No Shuffling!** Use TimeSeriesSplit (Expanding Window) or Rolling Window. | RNN, LSTM, GRU, ARIMA |
| **Autocorrelation** | Current value depends on past values ($y_t \approx y_{t-1}$). | Violates standard regression "independence" assumption. | Use Lag Features (creating columns for $t-1, t-7$) to feed this info to the model. | **ACF / PACF Plots**, Durbin-Watson Test |
| **Stationarity** | Mean and variance do not change over time. | Most statistical models assume the "rules" of data stay constant. Trends break this. | **Differencing** ($y_t - y_{t-1}$), Detrending, or Log Transformation. | **Augmented Dickey-Fuller (ADF) Test**, KPSS Test |


# 🧠 Data Science Uzman Analizi ve Teknik Eklemeler

> **Uzman Görüşü:** Metnin verdiği temel çok sağlam, ancak bir uzman olarak şunları eklemeliyiz:

---

## 1. Temporal Order (Zaman Sırası) & Validation

* ❌ **Eksik:** Metin sadece "karıştırmayın" (don't shuffle) diyor.
* ✅ **Teknik Ekleme:** Doğrulama (Validation) için standart *K-Fold Cross Validation* kullanılamaz. Bunun yerine **TimeSeriesSplit** (Expanding Window) veya **Rolling Window** yöntemleri kullanılmalıdır.

---

## 2. Autocorrelation (Otokorelasyon)

* ❌ **Eksik:** "Bugün dünü tekrar eder" denmiş ama nasıl ölçülür?
* ✅ **Teknik Ekleme:** Bunu ölçmek için **ACF (Autocorrelation Function)** ve **PACF (Partial Autocorrelation Function)** grafikleri (korelogramlar) kullanılır. **ARIMA** modelindeki `p` ve `q` parametreleri bu grafiklere bakarak seçilir.

---

## 3. Stationarity (Durağanlık)

* ❌ **Eksik:** Gözle kontrol (visual inspection) yeterli değildir.
* ✅ **Teknik Ekleme:** İstatistiksel test şarttır. En meşhuru **Augmented Dickey-Fuller (ADF)** testidir.
    * Eğer **p-value < 0.05** ise durağandır deriz.
    * Değilse, **Differencing** ($y_t - y_{t-1}$) işlemi uygulanır.
---


# 🧩 Components of a Time Series (Zaman Serisi Bileşenleri)

 <img width="722" height="281" alt="image" src="https://github.com/user-attachments/assets/efe54c65-be23-4bb8-98ae-85d9b17dd2a5" />

Time-series data are built from three primary ingredients—**trend**, **seasonality**, and **noise**—layered on top of one another like tracks in a music mix.

In a composite plot, you’ll see all three playing at once:
1.  A line that generally climbs upward (**Trend**).
2.  Rises and falls in a smooth yearly rhythm (**Seasonality**).
3.  Jiggles unpredictably from point to point (**Noise**).

By zooming in on each component separately, we can explain where the shape of the full series comes from, choose the right modelling tools for each layer, and make cleaner forecasts than if we treated the whole tangle as a single line.

---
## 1. Trend (Eğilim)

<img width="699" height="252" alt="image" src="https://github.com/user-attachments/assets/d7af0bdd-5522-467c-9a1f-1a039e13cfc9" />

**Definition:** This represents the long-term direction or tendency of the data. It captures the overall upward or downward movement over time. Trends can be linear (constant increase or decrease) or nonlinear (curved or oscillating).

* **Visual:** In a “Trend Component” chart, notice the steady climb—no dips, no cycles.

> **💡 Uzman Notu (Technical Insight):**
> Trendi izole etmek için genellikle **Hareketli Ortalamalar (Moving Averages)** veya **LOESS (Locally Estimated Scatterplot Smoothing)** yöntemleri kullanılır. Trendi veriden çıkardığımızda (Detrending), geriye daha durağan (stationary) bir yapı kalır ki bu da modelleme için idealdir.

---

## 2. Seasonality (Mevsimsellik)

 <img width="668" height="245" alt="image" src="https://github.com/user-attachments/assets/3804b04f-f2f7-4923-b81a-5d5a4fdc921a" />

**Definition:** Refers to patterns that repeat at **fixed intervals** within a time series. These patterns can be daily, weekly, monthly, or yearly. External factors such as weather conditions, holidays, or economic cycles often have an impact on seasonality.

* **Visual:** A “Seasonality Component” shows a crisp repeating wave (e.g., a 12-month sine wave)—exactly the kind of yearly rhythm utilities see when summers get hot and winters cold.

> **⚠️ Kritik Ayrım (Expert Warning):**
> Metinde "ekonomik döngüler" mevsimsellik içinde geçse de, ileri seviye analizde **Cycle (Döngü)** ve **Seasonality (Mevsimsellik)** farklıdır.
> * **Seasonality:** Frekansı sabittir (Örn: Her Pazartesi).
> * **Cycle:** Frekansı değişkendir (Örn: Ekonomik krizler 5 yılda bir de olabilir, 10 yılda bir de). Döngüler genellikle Trend bileşeni içinde analiz edilir.

---

## 3. Noise / Residuals (Gürültü / Artıklar)
 
 <img width="730" height="272" alt="image" src="https://github.com/user-attachments/assets/08c4a1f2-60d9-4ba9-8aaf-dba14d91efe1" />

 **Definition:** Represents the unpredictable and random variations in the data and includes factors that cannot be explained by trend or seasonality. Measurement errors, random events, or unidentified factors can contribute to the presence of noise in the data.

* **Visual:** The “Noise Component” plot looks like pure scatter around zero; no clear trend or cycle.

> **💡 Uzman Notu (Technical Insight):**
> İdeal bir modelde Gürültü (Residuals) **"White Noise" (Beyaz Gürültü)** olmalıdır. Yani:
> 1.  Ortalaması sıfır olmalı.
> 2.  Varyansı sabit olmalı.
> 3.  Otokorelasyonu olmamalı (Rastgele olmalı).
> Eğer Gürültü kısmında hala bir desen (pattern) görüyorsanız, modeliniz verideki bilgiyi tam sömürememiş demektir (**Underfitting**).

---

## 4. Putting it Together (Birleştirme)

The first chart (“Full Series”) overlays all three ingredients:
* **Trend** lifts the whole series over time.
* **Seasonality** adds the rolling hills.
* **Noise** rattles each point up or down at random.

**Business Application:**
When we model a real business series—say, monthly revenue—we pull it apart the same way:
1.  Estimate the trend (growth).
2.  Capture repeating cycles (holidays, weekends).
3.  Treat what’s left as noise or anomalies.

Do that well, and forecasts become clearer, anomalies stand out sooner, and decisions (inventory, staffing, budget) get a firmer footing.

---

## 📊 Technical Comparison: Decomposition Models

Veri bilimciler olarak seriyi ayrıştırırken matematiksel yapısına göre şu iki modelden birini seçeriz:

| Özellik | Additive Decomposition (Toplamsal) | Multiplicative Decomposition (Çarpımsal) |
| :--- | :--- | :--- |
| **Matematiksel Formül** | $$y(t) = Trend + Seasonality + Noise$$|$$y(t) = Trend \times Seasonality \times Noise$$ |
| **Görsel İpucu** | Mevsimsel dalgalanmaların boyutu (genliği) zamanla **sabit** kalır. | Trend arttıkça (veya azaldıkça) mevsimsel dalgalanmalar da **büyür/küçülür**. |
| **Kullanım Alanı** | Sıcaklık değişimleri (Yazın hep +10 derece artar). | Satış verileri, Hisse senetleri (Satışlar 2 katına çıkarsa, yılbaşı yoğunluğu da 2 katına çıkar). |
| **Python Kodu** | `seasonal_decompose(model='additive')` | `seasonal_decompose(model='multiplicative')` |

# 🧩 Zaman Serisi Ayrıştırma (Time Series Decomposition)

Zaman Serisi Ayrıştırma (Time Series Decomposition), bir veri bilimcinin elindeki en güçlü analitik araçlardan biridir. 

> "Neden satışlar düştü?" sorusuna cevap verirken **"Genel bir düşüş mü var (Trend), yoksa sadece yaz bittiği için mi düştü (Mevsimsellik)?"** ayrımını yapmamızı sağlar.

---

## 🧠 Data Science Uzman Analizi ve Teknik Eklemeler

Bir Data Science Uzmanı olarak, metindeki kavramları derinleştirelim ve eksik teknik parçaları (Additive vs. Multiplicative modeller ve Decomposition algoritmaları) tamamlayalım.

Metin "Trend, Mevsimsellik ve Gürültü"yü anlatıyor ama bunların nasıl bir araya geldiğini (**Matematiksel Model**) ve nasıl ayrıştırıldığını (**Algoritma**) eksik bırakmış.

### 1. Toplamsal ve Çarpımsal Modeller (Additive vs. Multiplicative)

Zaman serisi bileşenleri iki ana şekilde birleşir:

#### ➕ Additive (Toplamsal) Model
$$Y(t) = Trend + Seasonality + Noise$$

* **Ne zaman kullanılır?** Mevsimsel dalgalanmaların boyutu zamanla değişmiyorsa (örneğin, her Aralık ayında satışlar hep 1000 birim artıyorsa).

#### ✖️ Multiplicative (Çarpımsal) Model
$$Y(t) = Trend \times Seasonality \times Noise$$

* **Ne zaman kullanılır?** Trend arttıkça mevsimsel dalgalanmalar da büyüyorsa (örneğin, şirket büyüdükçe Aralık ayı satış farkı 1000'den 10.000'e çıkıyorsa). Bu çok daha yaygındır.

### 2. Döngü (Cycle) vs. Mevsimsellik (Seasonality) Ayrımı

Metin "ekonomik döngüleri" mevsimsellik altında saymış. Bu teknik olarak yanlıştır.

* **Mevsimsellik (Seasonality):** Sabit frekanslıdır (Her 12 ayda bir, her 7 günde bir).
* **Döngü (Cycle):** Sabit olmayan dalgalanmalardır (Ekonomik krizler, Boğa/Ayı piyasaları). Genellikle Trend içinde saklanır veya ayrı bir "Cyclic" bileşen olarak ele alınır (Trend-Cycle).

### 3. Ayrıştırma Yöntemleri (Decomposition Algorithms)

Veriyi bu parçalara ayırmak için şu yöntemleri kullanırız:

* **Classical Decomposition:** Basit hareketli ortalamalar kullanır.
* **STL (Seasonal-Trend decomposition using LOESS):** En modern ve sağlam yöntemdir. Gürültüye karşı dayanıklıdır.
* **SEATS / X-11:** Özellikle resmi devlet istatistiklerinde kullanılır.

---

# 🍰 Neden "Süslü" Modellerden Önce "Temeller" ile Uğraşalım?


> **Analoji:** Hangi malzemelerin tuzlu, tatlı veya ekşi olduğunu bilmeden doğrudan pasta pişirmeye başlayan bir şef hayal edin; bazı kekler güzel olabilir, ancak çoğu sönecektir ve kimse nedenini bilmeyecektir.

**Zaman serisi modellemesi (Time-series modelling) de aynıdır.** Aşağıdaki tablo, temel kavramları atlamanın maliyetini göstermektedir:

| Eğer bu kavramı atlarsanız... (Kavram) | Muhtemel baş ağrısı (Sonuç) | Gerçek dünya maliyet örneği (Business Case) |
| :--- | :--- | :--- |
| **Temporal Order (Zaman Sırası)**<br>*(Satırları zaman sırasına göre tutun)* ⏳ | Tarihleri karıştırırsanız, model öğrenirken geleceğe "göz atabilir" (**Data Leakage**). Testlerde zeki görünür ama gerçek hayatta çuvallar. | 🛒 **Bakkal Örneği:** Bir bakkal karışık günlük satışlarla model eğitir. Model, Noel öncesi haftayı "tahmin ederken" Noel rakamlarını görür ve %99 doğruluk raporlar. Canlıya alındığında (Deployed), yoğun günlerde raflar boş kalır, durgun günlerde ise dolar taşar. |
| **Autocorrelation (Otokorelasyon)**<br>*(Bugün genellikle düne benzer)* 🔗 | Her noktayı yepyeni (bağımsız) gibi ele alırsanız, modeliniz elinde gerçekte olduğundan daha fazla bağımsız kanıt olduğunu sanır. Sonuç: Hata çubukları çok küçük görünür, bu yüzden kendinize aşırı güvenirsiniz (**Over-confidence**). | 🏦 **Banka Örneği:** Bir banka piyasa riskini dakika dakika ölçer ama her dakikanın bağlantısız olduğunu varsayar. Risk tahmininin dar olduğuna inanarak elinde çok az nakit tutar. Sonra büyük bir dalgalanma rezervleri siler süpürür. |
| **Stationarity (Durağanlık)**<br>*(Seviye ve yayılım sabit kalır)* ⚖️ | Yukarı veya aşağı sürüklenen verilerde düz bir taban çizgisi bekleyen bir model kullanmak tahminleri patlatır; ileriye baktıkça hatalar büyür. | ⚡ **Elektrik Şirketi Örneği:** Bir elektrik şirketi, istikrarlı bir şekilde artan talebe basit bir model uydurur. Yılın en sıcak gününde tahmin çok düşük kalır, bu yüzden zamanında ekstra güç satın alamazlar ve elektrik kesintileri (blackouts) yaşanır. |
| **Trend & Seasonality (Trend ve Mevsimsellik)**<br>*(Uzun yükseliş/düşüş & tekrarlayan döngüler)* 📈 | Bu desenleri tek bir yığın (blob) halinde toplarsanız, model istikrarlı bir büyüme trendini tatil zirveleriyle karıştırır ve hangisinin hangisi olduğunu ayırt edemez. | 🛍️ **Perakendeci Örneği:** Bir perakendeci, Aralık ayı devasa olduğu için satışların tüm yıl patladığını sanır. Şubat ayı için ekstra personel işe alırlar, ancak onları boş boş otururken izlerler. |
| **Noise (Gürültü)**<br>*(Açıklayamadığınız rastgele kıpırtılar)* 🔊 | Her küçük tümseği modellemeye çalışmak sistemi aşırı karmaşık ve kırılgan yapar (**Overfitting**); geçmişte harikadır, yeni verilerde berbattır. | 🚨 **Bakım Ekibi Örneği:** Bir bakım ekibi, sensörleri her kıpırtıyı yakalayacak şekilde ayarlar. Uyarı sistemi artık günde düzinelerce yanlış alarm verir, bu yüzden gerçek arızalar göz ardı edilir. |


## 🎯 Neden Önce Temelleri Anlamalıyız? (Strategic Value)

Bu parçaları (Trend, Mevsimsellik, Gürültü) anlamak size şu stratejik avantajları sağlar:

* 🛠️ **Doğru Aracı Seçmek (Choose the right tool)**
    Durağan olmayan (non-stationary) bir seri, genellikle düz bir model yerine **fark alma (differencing)** işlemini veya açıkça trend/mevsimsellik terimlerini içeren bir modeli gerektirir.

* 🧠 **Akıllıca Dönüştürmek (Transform smartly)**
    Verinizi akıllıca dönüştürün; tüm bunlar, artık nasıl çalıştıracağınızı bildiğiniz teşhis yöntemleri (diagnostics) tarafından yönlendirilecektir.

* 🔍 **Sonuçları Yorumlamak (Interpret results)**
    Bir tahmin saptığında, suçlunun mevsim dışı bir anormallik mi, trendde bir kırılma mı, yoksa sadece rastgele gürültü mü olduğunu ayırt edebilirsiniz.

* 🗣️ **Riski İletmek (Communicate risk)**
    Paydaşlar (Stakeholders), algoritmanızın marka adıyla daha az, çizginin **neden** hareket ettiği ve sizin **ne kadar emin olduğunuzla** daha çok ilgilenirler. Bileşenler size bu hikayeyi verir.

---

### 🚀 Özet: Kara Kutudan Şeffaf Modele

> **Kısacası:** Bu temellerin arkasındaki "neden", sadece sayı tüküren bir model ile **güven kazanan, maliyetli hataları önleyen ve daha iyi kararlar alınmasını sağlayan** bir model arasındaki farktır.




# 📝 Time Series Analysis: Quiz & Interview Questions (Technical Breakdown)

Bu dosya, Zaman Serisi (Time Series) analizinin temel kavramlarını, mülakatlarda veya sınavlarda çıkabilecek sorular üzerinden teknik olarak açıklar.

---

## ❓ 1. What is a key feature of time-series data that distinguishes it from other types of data?

> **✅ Cevap: Temporal Dependence / Chronological Order (Zaman Bağımlılığı / Kronolojik Sıra)**

### 💡 Teknik Açıklama (Technical Explanation)
Standart "Tabular Data" (Tablo verileri) veya kesitsel verilerde (Cross-sectional data), satırların sırası önemsizdir. Veriler genellikle **I.I.D.** (Independent and Identically Distributed) varsayımıyla ele alınır.

Ancak Zaman Serilerinde en ayırt edici özellik **Zaman Sırasıdır (Temporal Order)**.
* **Bağımlılık (Dependency):** $t$ anındaki bir gözlem ($y_t$), genellikle $t-1$ anındaki gözleme ($y_{t-1}$) matematiksel olarak bağlıdır. Buna **Otokorelasyon (Autocorrelation)** denir.
* **Sıra (Sequence):** Veriyi karıştıramazsınız (Shuffling is forbidden). Eğer karıştırırsanız, verinin içindeki "zaman bilgisini" ve "trend" yapısını yok edersiniz.

---

## ❓ 2. Which of the following is NOT a characteristic of time-series data?

> **✅ Cevap: Independent Observations (Bağımsız Gözlemler)**

### 💡 Teknik Açıklama (Technical Explanation)
Zaman serisi analizinin doğasına aykırı olan tek şey **Bağımsızlık (Independence)** kavramıdır.

| Karakteristik | Zaman Serisinde Var mı? | Açıklama |
| :--- | :---: | :--- |
| **Trend** | ✅ Evet | Verinin uzun vadeli yönelimi (Artış/Azalış). |
| **Seasonality** | ✅ Evet | Belirli periyotlarda tekrarlayan desenler. |
| **Noise / Irregularity** | ✅ Evet | Açıklanamayan rastgele dalgalanmalar (Stochastic component). |
| **Independence** | ❌ **HAYIR** | Zaman serisi verileri **birbirine bağımlıdır (Dependent)**. Bir gün önceki satış, bugünkü satışı etkiler. |

---

## ❓ 3. What does seasonality in time-series data represent?

> **✅ Cevap: Repeating patterns at fixed intervals (Sabit aralıklarla tekrarlayan desenler)**

### 💡 Teknik Açıklama (Technical Explanation)
Mevsimsellik (**Seasonality**), verinin bilinen ve sabit bir frekansta (frequency) kendini tekrar etmesidir.

* **Anahtar Kelime:** "Fixed Interval" (Sabit Aralık).
* **Örnek:** Dondurma satışlarının her yıl Haziran'da artıp, Ocak'ta düşmesi.
* **Teknik Ayrım:** Mevsimsellik, **Döngüsellikten (Cyclicity)** farklıdır.
    * *Seasonality:* Takvime bağlıdır, süresi bellidir (Örn: 12 ay).
    * *Cyclicity:* Ekonomik krizler gibi süresi belli olmayan, düzensiz dalgalanmalardır.

---

## ❓ 4. Which time-series task involves predicting future values based on past data?

> **✅ Cevap: Forecasting (Tahminleme)**

### 💡 Teknik Açıklama (Technical Explanation)
Bu süreç literatürde **Forecasting** olarak geçer. Matematiksel olarak, geçmiş verilerin ($y_{t-1}, y_{t-2}...$) ve bazen dış faktörlerin ($X_t$) bir fonksiyonu olarak gelecekteki $y_{t+h}$ değerini bulmaktır.

Diğer görevlerle karıştırılmamalıdır:
* **Forecasting:** Geleceği tahmin etmek ($t+1$ nedir?).
* **Anomaly Detection:** Geçmişteki veya şimdiki verideki gariplikleri bulmak (Bu değer normal mi?).
* **Classification:** Seriyi bir kategoriye atamak (Bu EKG sinyali "Hasta" mı "Sağlıklı" mı?).

---

## ❓ 5. Which of these is an example of a trend in time-series data?

> **✅ Cevap: A long-term increase or decrease in the data (Verideki uzun vadeli artış veya azalış)**

### 💡 Teknik Açıklama (Technical Explanation)
Trend, verinin **uzun vadeli (long-term)** hareketidir. Kısa vadeli dalgalanmalardan (Noise) veya mevsimsel hareketlerden (Seasonality) arındırıldığında geriye kalan ana yöndür.

* **Örnek:** Küresel sıcaklıkların son 50 yıldaki ortalama artışı.
* **Matematiksel Temsil:** Genellikle $T_t$ ile gösterilir.
    * Lineer Trend: $y = mx + c$
    * Eksponansiyel Trend: $y = e^{ax}$

> **Ayrım:**
> * Bir aylık satışın patlaması (Spike) -> **Noise** veya **Anomaly** olabilir.
> * Her Aralık ayında artış -> **Seasonality**.
> * Son 5 yıldır satışların sürekli artması -> **Trend**.



  
