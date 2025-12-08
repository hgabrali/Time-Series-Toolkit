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
